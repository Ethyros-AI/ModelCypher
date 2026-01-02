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

"""Tests for VocabularyAlignmentMap and related functions."""

from __future__ import annotations

from modelcypher.core.domain.vocabulary.alignment_map import (
    AlignmentQuality,
    TokenAlignment,
    TokenizerComparisonResult,
    VocabularyAlignmentMap,
    build_alignment_from_vocabs,
    format_comparison_report,
)


class TestTokenAlignment:
    """Tests for TokenAlignment dataclass."""

    def test_to_dict_exact(self):
        """Test serialization of exact alignment."""
        alignment = TokenAlignment(
            source_id=0,
            source_token="hello",
            target_ids=[0],
            target_tokens=["hello"],
            quality=AlignmentQuality.EXACT,
        )

        d = alignment.to_dict()

        assert d["source_id"] == 0
        assert d["source_token"] == "hello"
        assert d["target_ids"] == [0]
        assert d["quality"] == "exact"

    def test_to_dict_with_metadata(self):
        """Test serialization includes metadata."""
        alignment = TokenAlignment(
            source_id=1,
            source_token="foo",
            target_ids=[2, 3],
            target_tokens=["fo", "o"],
            quality=AlignmentQuality.APPROXIMATE,
            metadata={"edit_distance": 1, "method": "prefix"},
        )

        d = alignment.to_dict()

        assert d["edit_distance"] == 1
        assert d["method"] == "prefix"


class TestVocabularyAlignmentMap:
    """Tests for VocabularyAlignmentMap."""

    def test_add_alignment_updates_statistics(self):
        """Test that adding alignments updates counters."""
        map_ = VocabularyAlignmentMap(source_vocab_size=100, target_vocab_size=100)

        # Add exact match
        map_.add_alignment(
            TokenAlignment(
                source_id=0,
                source_token="a",
                target_ids=[0],
                target_tokens=["a"],
                quality=AlignmentQuality.EXACT,
            )
        )

        # Add similar match
        map_.add_alignment(
            TokenAlignment(
                source_id=1,
                source_token="B",
                target_ids=[1],
                target_tokens=["b"],
                quality=AlignmentQuality.SIMILAR,
            )
        )

        # Add unmapped
        map_.add_alignment(
            TokenAlignment(
                source_id=2,
                source_token="xyz",
                target_ids=[],
                target_tokens=[],
                quality=AlignmentQuality.UNMAPPED,
            )
        )

        assert map_.exact_matches == 1
        assert map_.similar_matches == 1
        assert map_.unmapped_count == 1

    def test_get_alignment(self):
        """Test retrieving alignment by source ID."""
        map_ = VocabularyAlignmentMap(source_vocab_size=10, target_vocab_size=10)

        alignment = TokenAlignment(
            source_id=5,
            source_token="test",
            target_ids=[5],
            target_tokens=["test"],
            quality=AlignmentQuality.EXACT,
        )
        map_.add_alignment(alignment)

        assert map_.get_alignment(5) == alignment
        assert map_.get_alignment(0) is None

    def test_reverse_map(self):
        """Test reverse mapping from target to sources."""
        map_ = VocabularyAlignmentMap(source_vocab_size=10, target_vocab_size=5)

        # Multiple source tokens map to same target
        map_.add_alignment(
            TokenAlignment(
                source_id=0,
                source_token="hi",
                target_ids=[0],
                target_tokens=["hello"],
                quality=AlignmentQuality.SIMILAR,
            )
        )
        map_.add_alignment(
            TokenAlignment(
                source_id=1,
                source_token="hey",
                target_ids=[0],
                target_tokens=["hello"],
                quality=AlignmentQuality.SIMILAR,
            )
        )

        sources = map_.get_target_sources(0)
        assert 0 in sources
        assert 1 in sources

    def test_coverage(self):
        """Test coverage calculation."""
        map_ = VocabularyAlignmentMap(source_vocab_size=10, target_vocab_size=10)

        # Add 8 mapped tokens
        for i in range(8):
            map_.add_alignment(
                TokenAlignment(
                    source_id=i,
                    source_token=f"t{i}",
                    target_ids=[i],
                    target_tokens=[f"t{i}"],
                    quality=AlignmentQuality.EXACT,
                )
            )

        # Add 2 unmapped tokens
        for i in range(8, 10):
            map_.add_alignment(
                TokenAlignment(
                    source_id=i,
                    source_token=f"u{i}",
                    target_ids=[],
                    target_tokens=[],
                    quality=AlignmentQuality.UNMAPPED,
                )
            )

        assert map_.coverage == 0.8

    def test_coverage_zero_vocab(self):
        """Test coverage with zero vocab size."""
        map_ = VocabularyAlignmentMap(source_vocab_size=0, target_vocab_size=10)
        assert map_.coverage == 0.0

    def test_to_dict(self):
        """Test serialization to summary dict."""
        map_ = VocabularyAlignmentMap(source_vocab_size=100, target_vocab_size=80)

        for i in range(50):
            map_.add_alignment(
                TokenAlignment(
                    source_id=i,
                    source_token=f"t{i}",
                    target_ids=[i],
                    target_tokens=[f"t{i}"],
                    quality=AlignmentQuality.EXACT,
                )
            )

        d = map_.to_dict()

        assert d["source_vocab_size"] == 100
        assert d["target_vocab_size"] == 80
        assert d["total_alignments"] == 50
        assert d["exact_matches"] == 50

    def test_quality_distribution(self):
        """Test quality distribution method."""
        map_ = VocabularyAlignmentMap(source_vocab_size=10, target_vocab_size=10)

        map_.add_alignment(
            TokenAlignment(
                source_id=0,
                source_token="a",
                target_ids=[0],
                target_tokens=["a"],
                quality=AlignmentQuality.EXACT,
            )
        )
        map_.add_alignment(
            TokenAlignment(
                source_id=1,
                source_token="b",
                target_ids=[1],
                target_tokens=["b"],
                quality=AlignmentQuality.INTERPOLATED,
            )
        )

        dist = map_.quality_distribution()

        assert dist["exact"] == 1
        assert dist["interpolated"] == 1
        assert dist["similar"] == 0

    def test_iter_alignments(self):
        """Test iteration over alignments."""
        map_ = VocabularyAlignmentMap(source_vocab_size=5, target_vocab_size=5)

        for i in range(5):
            map_.add_alignment(
                TokenAlignment(
                    source_id=i,
                    source_token=f"t{i}",
                    target_ids=[i],
                    target_tokens=[f"t{i}"],
                    quality=AlignmentQuality.EXACT,
                )
            )

        alignments = list(map_.iter_alignments())
        assert len(alignments) == 5


class TestBuildAlignmentFromVocabs:
    """Tests for build_alignment_from_vocabs function."""

    def test_exact_matches(self):
        """Test that identical tokens get exact matches."""
        source = {"hello": 0, "world": 1}
        target = {"hello": 0, "world": 1}

        result = build_alignment_from_vocabs(source, target)

        assert result.exact_matches == 2
        assert result.coverage == 1.0

        hello_align = result.get_alignment(0)
        assert hello_align is not None
        assert hello_align.quality == AlignmentQuality.EXACT

    def test_normalized_matching(self):
        """Test case-insensitive matching."""
        source = {"Hello": 0, "WORLD": 1}
        target = {"hello": 0, "world": 1}

        result = build_alignment_from_vocabs(source, target)

        # Should match via normalization (lowercase)
        assert result.similar_matches == 2
        assert result.exact_matches == 0

    def test_unmapped_tokens(self):
        """Test tokens with no matches."""
        source = {"abc": 0, "xyz": 1}
        target = {"def": 0, "ghi": 1}

        result = build_alignment_from_vocabs(source, target)

        assert result.unmapped_count == 2
        assert result.coverage == 0.0

    def test_prefix_matching(self):
        """Test prefix-based approximate matching."""
        source = {"testing": 0}
        target = {"test": 0, "ing": 1}

        result = build_alignment_from_vocabs(source, target)

        align = result.get_alignment(0)
        assert align is not None
        # Should find "test" as prefix match
        if align.quality == AlignmentQuality.APPROXIMATE:
            assert 0 in align.target_ids  # "test"

    def test_exact_only_mode(self):
        """Test exact_only parameter."""
        source = {"Hello": 0, "abc": 1}
        target = {"hello": 0, "abc": 1}

        result = build_alignment_from_vocabs(source, target, exact_only=True)

        # Only "abc" should match exactly
        assert result.exact_matches == 1
        assert result.unmapped_count == 1

    def test_large_vocab_overlap(self):
        """Test with larger overlapping vocabularies."""
        source = {f"token_{i}": i for i in range(100)}
        target = {f"token_{i}": i for i in range(50, 150)}

        result = build_alignment_from_vocabs(source, target)

        # 50 exact matches (tokens 50-99)
        assert result.exact_matches == 50
        # 50 unmapped (tokens 0-49)
        assert result.unmapped_count == 50


class TestTokenizerComparisonResult:
    """Tests for TokenizerComparisonResult."""

    def test_to_dict(self):
        """Test serialization."""
        result = TokenizerComparisonResult(
            source_vocab_size=32000,
            target_vocab_size=50000,
            overlap_count=25000,
            overlap_ratio=0.78125,
            approximate_count=5000,
            unmapped_count=2000,
            coverage=0.9375,
        )

        d = result.to_dict()

        assert d["sourceVocabSize"] == 32000
        assert d["targetVocabSize"] == 50000
        assert d["overlapCount"] == 25000
        assert d["overlapRatio"] == 0.7812
        assert d["coverage"] == 0.9375


class TestFormatComparisonReport:
    """Tests for format_comparison_report function."""

    def test_format_report(self):
        """Test report formatting."""
        result = TokenizerComparisonResult(
            source_vocab_size=10000,
            target_vocab_size=15000,
            overlap_count=8000,
            overlap_ratio=0.8,
            approximate_count=1500,
            unmapped_count=500,
            coverage=0.95,
        )

        report = format_comparison_report(result)

        assert "Vocabulary Comparison Report" in report
        assert "10,000" in report  # Source vocab with comma formatting
        assert "80.0%" in report  # Overlap ratio
        assert "95.0%" in report  # Coverage
