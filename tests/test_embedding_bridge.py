"""
Tests for embedding bridge.

Validates cross-vocabulary embedding transformation.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from modelcypher.core.domain.merging.vocabulary_alignment import (
    AlignmentMethod,
    TokenMapping,
    VocabularyAlignmentResult,
)
from modelcypher.core.domain.merging.embedding_bridge import (
    BridgeMethod,
    EmbeddingBridgeConfig,
    EmbeddingBridgeResult,
    EmbeddingBridgeBuilder,
    format_bridge_report,
)


class TestBridgeMethod:
    """Test BridgeMethod enum."""

    def test_methods(self):
        """All methods should be accessible."""
        assert BridgeMethod.FVT.value == "fvt"
        assert BridgeMethod.PROCRUSTES.value == "procrustes"
        assert BridgeMethod.AFFINE.value == "affine"
        assert BridgeMethod.HYBRID.value == "hybrid"


class TestEmbeddingBridgeConfig:
    """Test EmbeddingBridgeConfig defaults."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = EmbeddingBridgeConfig()
        assert config.auto_select is True
        assert config.quality_threshold == 0.8
        assert config.fvt_threshold == 0.9
        assert config.min_anchor_pairs == 10

    def test_fallback_chain(self):
        """Fallback chain should be a tuple."""
        config = EmbeddingBridgeConfig()
        assert "fvt" in config.fallback_chain
        assert "procrustes" in config.fallback_chain


class TestEmbeddingBridgeResult:
    """Test EmbeddingBridgeResult properties."""

    def test_to_dict(self):
        """to_dict should include all key fields."""
        result = EmbeddingBridgeResult(
            method_used=BridgeMethod.FVT,
            bridged_embeddings=np.zeros((100, 64)),
            alignment_quality=0.85,
            anchor_pairs_used=50,
            methods_tried=["fvt", "procrustes"],
            per_method_quality={"fvt": 0.85, "procrustes": 0.82},
            warnings=["Test warning"],
        )
        d = result.to_dict()
        assert d["methodUsed"] == "fvt"
        assert d["alignmentQuality"] == 0.85
        assert d["anchorPairsUsed"] == 50
        assert d["embeddingShape"] == [100, 64]
        assert "warnings" in d


class TestEmbeddingBridgeBuilder:
    """Test EmbeddingBridgeBuilder methods."""

    def create_alignment(
        self, overlap_count: int, source_size: int, target_size: int
    ) -> VocabularyAlignmentResult:
        """Create a test alignment result."""
        mappings = {}
        for i in range(overlap_count):
            mappings[i] = TokenMapping(
                source_token_id=i,
                target_token_id=i,
                method=AlignmentMethod.EXACT,
                confidence=1.0,
            )
        for i in range(overlap_count, source_size):
            mappings[i] = TokenMapping(
                source_token_id=i,
                target_token_id=None,
                method=AlignmentMethod.UNMAPPED,
                confidence=0.0,
            )
        return VocabularyAlignmentResult(
            source_vocab_size=source_size,
            target_vocab_size=target_size,
            overlap_count=overlap_count,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=source_size - overlap_count,
            mappings=mappings,
        )

    def test_fvt_bridge_high_overlap(self):
        """FVT should be selected for high overlap."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        # High overlap - 95%
        alignment = self.create_alignment(95, 100, 100)

        builder = EmbeddingBridgeBuilder()
        result = builder.build(source_emb, target_emb, alignment)

        assert result.method_used == BridgeMethod.FVT
        assert result.bridged_embeddings.shape == (100, 64)

    def test_procrustes_bridge_medium_overlap(self):
        """Procrustes should be tried for medium overlap."""
        np.random.seed(42)
        hidden_dim = 64

        # Create related embeddings (rotation + noise)
        source_emb = np.random.randn(100, hidden_dim).astype(np.float32)

        # Generate rotation matrix
        Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
        target_emb = (source_emb @ Q + 0.1 * np.random.randn(100, hidden_dim)).astype(np.float32)

        # Medium overlap - 60%
        alignment = self.create_alignment(60, 100, 100)

        # Create anchor pairs from overlapping tokens
        anchor_pairs = [(i, i) for i in range(60)]

        builder = EmbeddingBridgeBuilder(EmbeddingBridgeConfig(min_anchor_pairs=5))
        result = builder.build(source_emb, target_emb, alignment, anchor_pairs)

        assert result.bridged_embeddings.shape == (100, hidden_dim)
        assert "fvt" in result.methods_tried

    def test_affine_bridge_low_overlap(self):
        """Affine should be tried for low overlap."""
        np.random.seed(42)
        hidden_dim = 64

        source_emb = np.random.randn(100, hidden_dim).astype(np.float32)

        # Apply affine transformation
        W = np.random.randn(hidden_dim, hidden_dim) * 0.5 + np.eye(hidden_dim)
        b = np.random.randn(hidden_dim) * 0.1
        target_emb = (source_emb @ W + b).astype(np.float32)

        # Low overlap - 30%
        alignment = self.create_alignment(30, 100, 100)
        anchor_pairs = [(i, i) for i in range(30)]

        config = EmbeddingBridgeConfig(
            min_anchor_pairs=5,
            quality_threshold=0.5,  # Lower threshold for test
        )
        builder = EmbeddingBridgeBuilder(config)
        result = builder.build(source_emb, target_emb, alignment, anchor_pairs)

        assert result.bridged_embeddings.shape == (100, hidden_dim)

    def test_method_selection_high_overlap(self):
        """Auto-select should choose FVT for high overlap."""
        alignment = self.create_alignment(95, 100, 100)

        builder = EmbeddingBridgeBuilder()
        methods = builder._select_method_order(alignment)

        assert methods == [BridgeMethod.FVT]

    def test_method_selection_medium_overlap(self):
        """Auto-select should try multiple methods for medium overlap."""
        alignment = self.create_alignment(60, 100, 100)

        builder = EmbeddingBridgeBuilder()
        methods = builder._select_method_order(alignment)

        assert len(methods) > 1
        assert BridgeMethod.FVT in methods
        assert BridgeMethod.PROCRUSTES in methods

    def test_method_selection_low_overlap(self):
        """Auto-select should prioritize Procrustes for low overlap."""
        alignment = self.create_alignment(30, 100, 100)

        builder = EmbeddingBridgeBuilder()
        methods = builder._select_method_order(alignment)

        assert methods[0] == BridgeMethod.PROCRUSTES

    def test_anchor_pair_extraction(self):
        """Should extract anchor pairs from EXACT mappings."""
        alignment = self.create_alignment(50, 100, 100)

        builder = EmbeddingBridgeBuilder()
        pairs = builder._extract_anchor_pairs(alignment)

        assert len(pairs) == 50
        assert all(s == t for s, t in pairs)

    def test_quality_measurement(self):
        """Quality should be high when embeddings are similar."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)

        # Identical embeddings should have perfect quality
        anchor_pairs = [(i, i) for i in range(50)]

        builder = EmbeddingBridgeBuilder()
        quality = builder._measure_quality(embeddings, embeddings, anchor_pairs)

        assert quality > 0.95  # Very high for identical embeddings

    def test_quality_measurement_different(self):
        """Quality should be lower when embeddings differ."""
        np.random.seed(42)
        bridged = np.random.randn(100, 64).astype(np.float32)
        target = np.random.randn(100, 64).astype(np.float32)  # Different embeddings

        anchor_pairs = [(i, i) for i in range(50)]

        builder = EmbeddingBridgeBuilder()
        quality = builder._measure_quality(bridged, target, anchor_pairs)

        # Random embeddings should have low quality
        assert quality < 0.8


class TestFVTBridge:
    """Test FVT bridging specifically."""

    def test_fvt_copies_exact_matches(self):
        """FVT should copy source embeddings for exact matches."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        # All exact matches
        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(100)
        }
        alignment = VocabularyAlignmentResult(
            source_vocab_size=100,
            target_vocab_size=100,
            overlap_count=100,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=0,
            mappings=mappings,
        )

        builder = EmbeddingBridgeBuilder()
        bridged = builder._fvt_bridge(source_emb, target_emb, alignment)

        # All positions should have source embeddings
        np.testing.assert_array_almost_equal(bridged, source_emb)

    def test_fvt_preserves_unmapped(self):
        """FVT should preserve target embeddings for unmapped tokens."""
        np.random.seed(42)
        source_emb = np.random.randn(50, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        # First 50 exact, rest unmapped
        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(50)
        }
        alignment = VocabularyAlignmentResult(
            source_vocab_size=50,
            target_vocab_size=100,
            overlap_count=50,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=0,
            mappings=mappings,
        )

        builder = EmbeddingBridgeBuilder()
        bridged = builder._fvt_bridge(source_emb, target_emb, alignment)

        # First 50 should be source
        np.testing.assert_array_almost_equal(bridged[:50], source_emb)
        # Last 50 should be target (unchanged)
        np.testing.assert_array_almost_equal(bridged[50:], target_emb[50:])


class TestProcrustesBridge:
    """Test Procrustes bridging specifically."""

    def test_procrustes_finds_rotation(self):
        """Procrustes should find optimal rotation."""
        np.random.seed(42)
        hidden_dim = 64
        n_points = 100

        # Create source embeddings
        source_emb = np.random.randn(n_points, hidden_dim).astype(np.float32)

        # Create target as rotated source
        Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
        target_emb = (source_emb @ Q).astype(np.float32)

        anchor_pairs = [(i, i) for i in range(n_points)]

        builder = EmbeddingBridgeBuilder()
        bridged = builder._procrustes_bridge(source_emb, target_emb, anchor_pairs)

        # Bridged should be close to target
        error = np.mean(np.abs(bridged - target_emb))
        assert error < 0.1  # Low reconstruction error

    def test_procrustes_empty_anchors(self):
        """Procrustes should return target copy when no anchors."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        builder = EmbeddingBridgeBuilder()
        bridged = builder._procrustes_bridge(source_emb, target_emb, [])

        np.testing.assert_array_equal(bridged, target_emb)


class TestAffineBridge:
    """Test Affine bridging specifically."""

    def test_affine_finds_transformation(self):
        """Affine should find optimal linear transformation."""
        np.random.seed(42)
        hidden_dim = 64
        n_points = 100

        # Create source embeddings
        source_emb = np.random.randn(n_points, hidden_dim).astype(np.float32)

        # Create target with affine transformation
        W = 0.1 * np.random.randn(hidden_dim, hidden_dim) + np.eye(hidden_dim)
        b = np.random.randn(hidden_dim) * 0.05
        target_emb = (source_emb @ W + b).astype(np.float32)

        anchor_pairs = [(i, i) for i in range(n_points)]

        builder = EmbeddingBridgeBuilder()
        bridged = builder._affine_bridge(source_emb, target_emb, anchor_pairs)

        # Bridged should be close to target
        error = np.mean(np.abs(bridged - target_emb))
        assert error < 0.5  # Reasonable reconstruction error

    def test_affine_empty_anchors(self):
        """Affine should return target copy when no anchors."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        builder = EmbeddingBridgeBuilder()
        bridged = builder._affine_bridge(source_emb, target_emb, [])

        np.testing.assert_array_equal(bridged, target_emb)


class TestHybridBridge:
    """Test Hybrid bridging specifically."""

    def test_hybrid_uses_exact_for_overlapping(self):
        """Hybrid should copy exact matches directly."""
        np.random.seed(42)
        hidden_dim = 64
        n_points = 100

        source_emb = np.random.randn(n_points, hidden_dim).astype(np.float32)
        target_emb = np.random.randn(n_points, hidden_dim).astype(np.float32)

        # First 50 are exact matches
        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(50)
        }
        for i in range(50, n_points):
            mappings[i] = TokenMapping(i, None, AlignmentMethod.UNMAPPED, 0.0)

        alignment = VocabularyAlignmentResult(
            source_vocab_size=n_points,
            target_vocab_size=n_points,
            overlap_count=50,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=50,
            mappings=mappings,
        )

        anchor_pairs = [(i, i) for i in range(50)]

        builder = EmbeddingBridgeBuilder()
        bridged = builder._hybrid_bridge(source_emb, target_emb, alignment, anchor_pairs)

        # Exact matches should use source directly
        np.testing.assert_array_almost_equal(bridged[:50], source_emb[:50])


class TestFormatBridgeReport:
    """Test report formatting."""

    def test_format_report_high_quality(self):
        """Report should indicate HIGH QUALITY for good alignments."""
        result = EmbeddingBridgeResult(
            method_used=BridgeMethod.FVT,
            bridged_embeddings=np.zeros((100, 64)),
            alignment_quality=0.95,
            anchor_pairs_used=50,
        )

        report = format_bridge_report(result)

        assert "HIGH QUALITY" in report
        assert "FVT" in report

    def test_format_report_low_quality(self):
        """Report should indicate LOW QUALITY for poor alignments."""
        result = EmbeddingBridgeResult(
            method_used=BridgeMethod.FVT,
            bridged_embeddings=np.zeros((100, 64)),
            alignment_quality=0.3,
            anchor_pairs_used=10,
        )

        report = format_bridge_report(result)

        assert "LOW QUALITY" in report

    def test_format_report_with_warnings(self):
        """Report should include warnings."""
        result = EmbeddingBridgeResult(
            method_used=BridgeMethod.FVT,
            bridged_embeddings=np.zeros((100, 64)),
            alignment_quality=0.7,
            anchor_pairs_used=20,
            warnings=["Skipping Procrustes: insufficient anchors"],
        )

        report = format_bridge_report(result)

        assert "Warnings" in report
        assert "insufficient anchors" in report


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_source_embeddings(self):
        """Should handle empty source embeddings."""
        source_emb = np.zeros((0, 64), dtype=np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        alignment = VocabularyAlignmentResult(
            source_vocab_size=0,
            target_vocab_size=100,
            overlap_count=0,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=0,
            mappings={},
        )

        builder = EmbeddingBridgeBuilder()
        result = builder.build(source_emb, target_emb, alignment)

        assert result.bridged_embeddings.shape == (100, 64)

    def test_different_vocab_sizes(self):
        """Should handle different vocabulary sizes."""
        np.random.seed(42)
        source_emb = np.random.randn(50, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(50)
        }
        alignment = VocabularyAlignmentResult(
            source_vocab_size=50,
            target_vocab_size=100,
            overlap_count=50,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=0,
            mappings=mappings,
        )

        builder = EmbeddingBridgeBuilder()
        result = builder.build(source_emb, target_emb, alignment)

        assert result.bridged_embeddings.shape == (100, 64)

    def test_insufficient_anchors_warning(self):
        """Should warn when insufficient anchor pairs."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        # Only 5 overlapping tokens
        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(5)
        }
        for i in range(5, 100):
            mappings[i] = TokenMapping(i, None, AlignmentMethod.UNMAPPED, 0.0)

        alignment = VocabularyAlignmentResult(
            source_vocab_size=100,
            target_vocab_size=100,
            overlap_count=5,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=95,
            mappings=mappings,
        )

        config = EmbeddingBridgeConfig(min_anchor_pairs=10)
        builder = EmbeddingBridgeBuilder(config)
        result = builder.build(source_emb, target_emb, alignment)

        # Should have warning about insufficient anchors
        assert any("anchor" in w.lower() for w in result.warnings)


class TestPropertyBased:
    """Property-based tests."""

    def test_bridged_shape_matches_target(self):
        """Bridged embeddings should always match target vocab size."""
        np.random.seed(42)
        for target_size in [50, 100, 200]:
            source_emb = np.random.randn(100, 64).astype(np.float32)
            target_emb = np.random.randn(target_size, 64).astype(np.float32)

            overlap = min(50, target_size)
            mappings = {
                i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
                for i in range(overlap)
            }
            for i in range(overlap, 100):
                mappings[i] = TokenMapping(i, None, AlignmentMethod.UNMAPPED, 0.0)

            alignment = VocabularyAlignmentResult(
                source_vocab_size=100,
                target_vocab_size=target_size,
                overlap_count=overlap,
                decomposed_count=0,
                semantic_count=0,
                unmapped_count=100 - overlap,
                mappings=mappings,
            )

            builder = EmbeddingBridgeBuilder()
            result = builder.build(source_emb, target_emb, alignment)

            assert result.bridged_embeddings.shape[0] == target_size
            assert result.bridged_embeddings.shape[1] == 64

    def test_quality_bounded(self):
        """Quality should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            bridged = np.random.randn(100, 64).astype(np.float32)
            target = np.random.randn(100, 64).astype(np.float32)
            anchor_pairs = [(i, i) for i in range(50)]

            builder = EmbeddingBridgeBuilder()
            quality = builder._measure_quality(bridged, target, anchor_pairs)

            assert 0.0 <= quality <= 1.0

    def test_fvt_is_deterministic(self):
        """FVT bridge should be deterministic."""
        np.random.seed(42)
        source_emb = np.random.randn(100, 64).astype(np.float32)
        target_emb = np.random.randn(100, 64).astype(np.float32)

        mappings = {
            i: TokenMapping(i, i, AlignmentMethod.EXACT, 1.0)
            for i in range(50)
        }
        for i in range(50, 100):
            mappings[i] = TokenMapping(i, None, AlignmentMethod.UNMAPPED, 0.0)

        alignment = VocabularyAlignmentResult(
            source_vocab_size=100,
            target_vocab_size=100,
            overlap_count=50,
            decomposed_count=0,
            semantic_count=0,
            unmapped_count=50,
            mappings=mappings,
        )

        builder = EmbeddingBridgeBuilder()
        result1 = builder._fvt_bridge(source_emb, target_emb, alignment)
        result2 = builder._fvt_bridge(source_emb, target_emb, alignment)

        np.testing.assert_array_equal(result1, result2)
