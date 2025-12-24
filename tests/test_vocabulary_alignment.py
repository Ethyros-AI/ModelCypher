"""
Tests for vocabulary alignment.

Validates cross-vocabulary alignment for model merging.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from modelcypher.core.domain.merging.vocabulary_alignment import (
    AlignmentMethod,
    TokenMapping,
    VocabularyAlignmentConfig,
    VocabularyAlignmentResult,
    VocabularyAligner,
    format_alignment_report,
)


class TestTokenMapping:
    """Test TokenMapping dataclass."""

    def test_mapped_token(self):
        """Exact match should be mapped."""
        mapping = TokenMapping(
            source_token_id=100,
            target_token_id=200,
            method=AlignmentMethod.EXACT,
            confidence=1.0,
        )
        assert mapping.is_mapped
        assert mapping.source_token_id == 100
        assert mapping.target_token_id == 200

    def test_unmapped_token(self):
        """Unmapped tokens have None target."""
        mapping = TokenMapping(
            source_token_id=100,
            target_token_id=None,
            method=AlignmentMethod.UNMAPPED,
            confidence=0.0,
        )
        assert not mapping.is_mapped

    def test_decomposed_token(self):
        """Decomposed tokens have decomposition tuple."""
        mapping = TokenMapping(
            source_token_id=100,
            target_token_id=50,
            method=AlignmentMethod.DECOMPOSED,
            confidence=0.85,
            decomposition=(50, 51, 52),
        )
        assert mapping.is_mapped
        assert mapping.decomposition == (50, 51, 52)


class TestVocabularyAlignmentResult:
    """Test VocabularyAlignmentResult properties."""

    def test_overlap_ratio(self):
        """Overlap ratio should be overlap / source size."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1200,
            overlap_count=800,
            decomposed_count=100,
            semantic_count=50,
            unmapped_count=50,
        )
        assert result.overlap_ratio == 0.8

    def test_coverage(self):
        """Coverage should be 1 - unmapped / source size."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1200,
            overlap_count=800,
            decomposed_count=100,
            semantic_count=50,
            unmapped_count=50,
        )
        assert result.coverage == 0.95

    def test_recommended_method_high_overlap(self):
        """High overlap should recommend FVT only."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1000,
            overlap_count=950,
            decomposed_count=30,
            semantic_count=10,
            unmapped_count=10,
        )
        assert result.recommended_method == "fvt"

    def test_recommended_method_medium_overlap(self):
        """Medium overlap should recommend FVT + Procrustes."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1200,
            overlap_count=600,
            decomposed_count=200,
            semantic_count=100,
            unmapped_count=100,
        )
        assert result.recommended_method == "fvt+procrustes"

    def test_recommended_method_low_overlap(self):
        """Low overlap should recommend Procrustes + Affine."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=2000,
            overlap_count=300,
            decomposed_count=200,
            semantic_count=200,
            unmapped_count=300,
        )
        assert result.recommended_method == "procrustes+affine"

    def test_merge_feasibility_high(self):
        """High coverage should be high feasibility."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1000,
            overlap_count=960,
            decomposed_count=30,
            semantic_count=5,
            unmapped_count=5,
        )
        assert result.merge_feasibility == "high"

    def test_merge_feasibility_infeasible(self):
        """Very low coverage should be infeasible."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=2000,
            overlap_count=100,
            decomposed_count=100,
            semantic_count=100,
            unmapped_count=700,
        )
        assert result.merge_feasibility == "infeasible"

    def test_to_dict(self):
        """to_dict should include all key fields."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1200,
            overlap_count=800,
            decomposed_count=100,
            semantic_count=50,
            unmapped_count=50,
        )
        d = result.to_dict()
        assert d["sourceVocabSize"] == 1000
        assert d["targetVocabSize"] == 1200
        assert d["overlapRatio"] == 0.8
        assert d["coverage"] == 0.95
        assert "recommendedMethod" in d
        assert "mergeFeasibility" in d


class TestVocabularyAligner:
    """Test VocabularyAligner methods."""

    def create_mock_tokenizer(self, vocab: dict[str, int]):
        """Create a mock tokenizer with given vocabulary."""
        mock = MagicMock()
        mock.get_vocab.return_value = vocab
        return mock

    def test_exact_match_full_overlap(self):
        """Identical vocabularies should have 100% exact match."""
        vocab = {"hello": 0, "world": 1, "test": 2}
        source = self.create_mock_tokenizer(vocab)
        target = self.create_mock_tokenizer(vocab)

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        assert result.overlap_count == 3
        assert result.overlap_ratio == 1.0
        assert result.unmapped_count == 0

    def test_exact_match_partial_overlap(self):
        """Partial overlap should be detected correctly."""
        source = self.create_mock_tokenizer({"hello": 0, "world": 1, "foo": 2})
        target = self.create_mock_tokenizer({"hello": 0, "world": 1, "bar": 2})

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        assert result.overlap_count == 2
        assert result.unmapped_count == 1

    def test_exact_match_no_overlap(self):
        """Disjoint vocabularies should have 0% overlap."""
        source = self.create_mock_tokenizer({"a": 0, "b": 1, "c": 2})
        target = self.create_mock_tokenizer({"x": 0, "y": 1, "z": 2})

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        assert result.overlap_count == 0
        assert result.overlap_ratio == 0.0

    def test_decomposition_match(self):
        """Tokens should decompose to target subtokens."""
        source_vocab = {"hello": 0, "world": 1, "helloworld": 2}
        target_vocab = {"hello": 0, "world": 1}

        source = self.create_mock_tokenizer(source_vocab)

        # Create target tokenizer that decomposes "helloworld" to ["hello", "world"]
        target = self.create_mock_tokenizer(target_vocab)

        # Mock encode to return decomposition
        encoded_result = MagicMock()
        encoded_result.ids = [0, 1]  # "helloworld" → ["hello", "world"]
        target.encode.return_value = encoded_result

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        # "hello" and "world" exact match, "helloworld" should decompose
        assert result.overlap_count == 2
        assert result.decomposed_count == 1
        assert result.unmapped_count == 0

    def test_semantic_match(self):
        """Similar embeddings should match semantically."""
        source_vocab = {"cat": 0, "dog": 1, "xyz": 2}
        target_vocab = {"feline": 0, "canine": 1, "abc": 2}

        source = self.create_mock_tokenizer(source_vocab)
        target = self.create_mock_tokenizer(target_vocab)

        # Create embeddings where cat≈feline, dog≈canine
        source_embeds = np.array([
            [1.0, 0.0, 0.0],  # cat
            [0.0, 1.0, 0.0],  # dog
            [0.0, 0.0, 1.0],  # xyz
        ])
        target_embeds = np.array([
            [0.99, 0.01, 0.0],  # feline (similar to cat)
            [0.01, 0.99, 0.0],  # canine (similar to dog)
            [0.5, 0.5, 0.0],    # abc (not similar to xyz)
        ])

        # Mock encode to return single token (no decomposition)
        encoded_result = MagicMock()
        encoded_result.ids = [99]  # Unknown single token
        target.encode.return_value = encoded_result

        config = VocabularyAlignmentConfig(semantic_threshold=0.9)
        aligner = VocabularyAligner(config)
        result = aligner.align(source, target, source_embeds, target_embeds)

        # cat→feline and dog→canine should match semantically
        # xyz→abc should not match (low similarity)
        assert result.semantic_count == 2
        assert result.unmapped_count == 1

    def test_compare_vocabularies(self):
        """Quick comparison should return statistics."""
        source = self.create_mock_tokenizer({"a": 0, "b": 1, "c": 2, "d": 3})
        target = self.create_mock_tokenizer({"a": 0, "b": 1, "x": 2, "y": 3})

        aligner = VocabularyAligner()
        stats = aligner.compare_vocabularies(source, target)

        assert stats["sourceVocabSize"] == 4
        assert stats["targetVocabSize"] == 4
        assert stats["overlapCount"] == 2
        assert stats["overlapRatio"] == 0.5
        assert stats["sourceOnlyCount"] == 2
        assert stats["targetOnlyCount"] == 2
        assert stats["compatible"] is False  # 50% < 90%


class TestFormatAlignmentReport:
    """Test report formatting."""

    def test_format_report(self):
        """Report should include all key information."""
        result = VocabularyAlignmentResult(
            source_vocab_size=32000,
            target_vocab_size=50000,
            overlap_count=28000,
            decomposed_count=2000,
            semantic_count=1000,
            unmapped_count=1000,
        )

        report = format_alignment_report(result)

        assert "32,000" in report  # Source vocab
        assert "50,000" in report  # Target vocab
        assert "87.5%" in report  # Overlap ratio
        assert "96.9%" in report  # Coverage
        assert "MEDIUM" in report or "HIGH" in report  # Feasibility
        assert "fvt" in report.lower()  # Recommended method


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_source_vocab(self):
        """Empty source vocabulary should handle gracefully."""
        source = MagicMock()
        source.get_vocab.return_value = {}
        target = MagicMock()
        target.get_vocab.return_value = {"a": 0, "b": 1}

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        assert result.source_vocab_size == 0
        assert result.overlap_ratio == 0.0
        assert result.coverage == 0.0

    def test_empty_target_vocab(self):
        """Empty target vocabulary should have no matches."""
        source = MagicMock()
        source.get_vocab.return_value = {"a": 0, "b": 1}
        target = MagicMock()
        target.get_vocab.return_value = {}

        aligner = VocabularyAligner()
        result = aligner.align(source, target)

        assert result.target_vocab_size == 0
        assert result.overlap_count == 0
        assert result.unmapped_count == 2

    def test_large_vocab_batching(self):
        """Large vocabularies should be processed in batches."""
        vocab_size = 5000
        source_vocab = {f"token_{i}": i for i in range(vocab_size)}
        target_vocab = {f"token_{i}": i for i in range(vocab_size)}

        source = MagicMock()
        source.get_vocab.return_value = source_vocab
        target = MagicMock()
        target.get_vocab.return_value = target_vocab

        config = VocabularyAlignmentConfig(batch_size=1000)
        aligner = VocabularyAligner(config)
        result = aligner.align(source, target)

        assert result.overlap_count == vocab_size
        assert result.coverage == 1.0


class TestPropertyBased:
    """Property-based tests."""

    def test_coverage_bounded(self):
        """Coverage should always be in [0, 1]."""
        for unmapped in range(0, 101, 10):
            result = VocabularyAlignmentResult(
                source_vocab_size=100,
                target_vocab_size=100,
                overlap_count=100 - unmapped,
                decomposed_count=0,
                semantic_count=0,
                unmapped_count=unmapped,
            )
            assert 0.0 <= result.coverage <= 1.0

    def test_overlap_ratio_bounded(self):
        """Overlap ratio should always be in [0, 1]."""
        for overlap in range(0, 101, 10):
            result = VocabularyAlignmentResult(
                source_vocab_size=100,
                target_vocab_size=100,
                overlap_count=overlap,
                decomposed_count=0,
                semantic_count=0,
                unmapped_count=100 - overlap,
            )
            assert 0.0 <= result.overlap_ratio <= 1.0

    def test_counts_sum_to_source_size(self):
        """All mapping counts should sum to source vocab size."""
        result = VocabularyAlignmentResult(
            source_vocab_size=1000,
            target_vocab_size=1200,
            overlap_count=600,
            decomposed_count=200,
            semantic_count=100,
            unmapped_count=100,
        )
        total = (
            result.overlap_count
            + result.decomposed_count
            + result.semantic_count
            + result.unmapped_count
        )
        assert total == result.source_vocab_size
