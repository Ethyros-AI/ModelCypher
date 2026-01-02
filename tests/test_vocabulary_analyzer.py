# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for VocabularyAnalyzer."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.vocabulary.vocabulary_analyzer import (
    TokenizerType,
    VocabularyAnalyzer,
    VocabularyAlignment,
    VocabularyStats,
)


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


@pytest.fixture
def analyzer(backend):
    """Create vocabulary analyzer."""
    return VocabularyAnalyzer(backend=backend)


class TestVocabularyStats:
    """Tests for VocabularyStats dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = VocabularyStats(
            vocab_size=32000,
            hidden_dim=768,
            embedding_mean_norm=1.5,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=5,
            has_tie_weights=True,
            mean_token_length=4.2,
            bpe_merge_count=1000,
        )

        d = stats.to_dict()

        assert d["vocab_size"] == 32000
        assert d["hidden_dim"] == 768
        assert d["embedding_mean_norm"] == 1.5
        assert d["tokenizer_type"] == "bpe"
        assert d["has_tie_weights"] is True


class TestVocabularyAlignment:
    """Tests for VocabularyAlignment dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        alignment = VocabularyAlignment(
            alignment_score=0.75,
            vocab_overlap_ratio=0.6,
            dimension_ratio=1.0,
            requires_projection=False,
            requires_vocab_mapping=True,
            shared_token_count=19200,
            source_only_tokens=6400,
            target_only_tokens=6400,
        )

        d = alignment.to_dict()

        assert d["alignment_score"] == 0.75
        assert d["requires_projection"] is False
        assert d["requires_vocab_mapping"] is True
        assert d["shared_token_count"] == 19200


class TestVocabularyAnalyzer:
    """Tests for VocabularyAnalyzer."""

    def test_analyze_embeddings_basic(self, analyzer, backend):
        """Test basic embedding analysis."""
        # Create random embedding matrix
        backend.random_seed(42)
        embeddings = backend.random_normal((1000, 256))

        stats = analyzer.analyze_embeddings(embeddings)

        assert stats.vocab_size == 1000
        assert stats.hidden_dim == 256
        assert stats.embedding_mean_norm > 0
        assert stats.embedding_std > 0
        assert stats.tokenizer_type == TokenizerType.UNKNOWN
        assert stats.special_token_count == 0

    def test_analyze_embeddings_with_tokenizer_config(self, analyzer, backend):
        """Test embedding analysis with tokenizer config."""
        backend.random_seed(42)
        embeddings = backend.random_normal((500, 128))

        config = {
            "tokenizer_class": "GPT2Tokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }

        stats = analyzer.analyze_embeddings(embeddings, tokenizer_config=config)

        assert stats.vocab_size == 500
        assert stats.hidden_dim == 128
        assert stats.tokenizer_type == TokenizerType.BPE
        assert stats.special_token_count == 3

    def test_analyze_embeddings_rejects_non_2d(self, analyzer, backend):
        """Test that non-2D embeddings raise error."""
        embeddings = backend.zeros((10, 20, 30))

        with pytest.raises(ValueError, match="Expected 2D"):
            analyzer.analyze_embeddings(embeddings)

    def test_analyze_alignment_same_dimensions(self, analyzer):
        """Test alignment analysis with same dimensions."""
        source = VocabularyStats(
            vocab_size=32000,
            hidden_dim=768,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=3,
            has_tie_weights=False,
        )
        target = VocabularyStats(
            vocab_size=32000,
            hidden_dim=768,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=3,
            has_tie_weights=False,
        )

        alignment = analyzer.analyze_alignment(source, target)

        assert alignment.requires_projection is False
        assert abs(alignment.dimension_ratio - 1.0) < _div_eps()

    def test_analyze_alignment_different_dimensions(self, analyzer):
        """Test alignment when dimensions differ."""
        source = VocabularyStats(
            vocab_size=32000,
            hidden_dim=768,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=3,
            has_tie_weights=False,
        )
        target = VocabularyStats(
            vocab_size=50000,
            hidden_dim=1024,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.SENTENCEPIECE,
            special_token_count=4,
            has_tie_weights=True,
        )

        alignment = analyzer.analyze_alignment(source, target)

        assert alignment.requires_projection is True
        expected = 768 / 1024
        assert abs(alignment.dimension_ratio - expected) < _div_eps()
        assert alignment.requires_vocab_mapping is True

    def test_analyze_alignment_with_vocab_dicts(self, analyzer):
        """Test alignment with actual vocabulary dictionaries."""
        source = VocabularyStats(
            vocab_size=100,
            hidden_dim=256,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=3,
            has_tie_weights=False,
        )
        target = VocabularyStats(
            vocab_size=100,
            hidden_dim=256,
            embedding_mean_norm=1.0,
            embedding_std=0.02,
            tokenizer_type=TokenizerType.BPE,
            special_token_count=3,
            has_tie_weights=False,
        )

        # 80 shared tokens, 20 unique each
        source_vocab = {f"token_{i}": i for i in range(100)}
        target_vocab = {f"token_{i}": i for i in range(20, 120)}

        alignment = analyzer.analyze_alignment(
            source, target, source_vocab=source_vocab, target_vocab=target_vocab
        )

        assert alignment.shared_token_count == 80  # tokens 20-99
        assert alignment.source_only_tokens == 20  # tokens 0-19
        assert alignment.target_only_tokens == 20  # tokens 100-119
        expected = 80 / 120
        assert alignment.vocab_overlap_ratio is not None
        assert abs(alignment.vocab_overlap_ratio - expected) < _div_eps()

    def test_compute_token_overlap(self, analyzer):
        """Test token overlap computation."""
        source = {"a": 0, "b": 1, "c": 2, "d": 3}
        target = {"b": 0, "c": 1, "e": 2, "f": 3}

        shared, source_only, target_only = analyzer.compute_token_overlap(source, target)

        assert shared == {"b", "c"}
        assert source_only == {"a", "d"}
        assert target_only == {"e", "f"}


class TestTokenizerTypeDetection:
    """Tests for tokenizer type detection."""

    def test_detect_bpe_from_class(self, analyzer, backend):
        """Test BPE detection from tokenizer class."""
        embeddings = backend.random_normal((100, 64))
        config = {"tokenizer_class": "GPT2Tokenizer"}

        stats = analyzer.analyze_embeddings(embeddings, config)
        assert stats.tokenizer_type == TokenizerType.BPE

    def test_detect_sentencepiece_from_model_type(self, analyzer, backend):
        """Test SentencePiece detection from model type."""
        embeddings = backend.random_normal((100, 64))
        config = {"model_type": "llama"}

        stats = analyzer.analyze_embeddings(embeddings, config)
        assert stats.tokenizer_type == TokenizerType.SENTENCEPIECE

    def test_detect_wordpiece_from_model_type(self, analyzer, backend):
        """Test WordPiece detection from model type."""
        embeddings = backend.random_normal((100, 64))
        config = {"model_type": "bert"}

        stats = analyzer.analyze_embeddings(embeddings, config)
        assert stats.tokenizer_type == TokenizerType.WORDPIECE

    def test_detect_from_tokenizer_json_model(self, analyzer, backend):
        """Test detection from tokenizer.json model field."""
        embeddings = backend.random_normal((100, 64))
        config = {"model": {"type": "BPE"}}

        stats = analyzer.analyze_embeddings(embeddings, config)
        assert stats.tokenizer_type == TokenizerType.BPE


class TestSpecialTokenCounting:
    """Tests for special token counting."""

    def test_count_standard_special_tokens(self, analyzer, backend):
        """Test counting standard special tokens."""
        embeddings = backend.random_normal((100, 64))
        config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }

        stats = analyzer.analyze_embeddings(embeddings, config)
        assert stats.special_token_count == 4

    def test_count_added_tokens(self, analyzer, backend):
        """Test counting added special tokens."""
        embeddings = backend.random_normal((100, 64))
        config = {
            "bos_token": "<s>",
            "added_tokens": [
                {"content": "<special1>", "special": True},
                {"content": "<special2>", "special": True},
                {"content": "regular", "special": False},
            ],
        }

        stats = analyzer.analyze_embeddings(embeddings, config)
        # 1 bos + 2 special added tokens
        assert stats.special_token_count == 3

    def test_no_config_returns_zero(self, analyzer, backend):
        """Test that missing config returns zero special tokens."""
        embeddings = backend.random_normal((100, 64))

        stats = analyzer.analyze_embeddings(embeddings)
        assert stats.special_token_count == 0
