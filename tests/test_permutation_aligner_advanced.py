"""Tests for advanced PermutationAligner methods.

Tests fuse(), rebasin_mlp_only(), rebasin_mlp_with_activations(),
and helper methods added for TIES-Merging and MLP-focused re-basin.
"""

import mlx.core as mx
import numpy as np
import pytest

from modelcypher.core.domain.geometry.permutation_aligner import (
    PermutationAligner,
    AlignmentResult,
    Config,
    FusionConfig,
    AnchorActivationContext,
    PermutationAlignerError,
)


class TestPermutationAlignerFuse:
    """Tests for PermutationAligner.fuse() - TIES-Merging fusion."""

    @pytest.fixture
    def basic_alignment(self) -> AlignmentResult:
        """Create a basic alignment result for testing."""
        N = 4
        return AlignmentResult(
            permutation=mx.eye(N, dtype=mx.float32),
            signs=mx.eye(N, dtype=mx.float32),
            match_quality=0.95,
            match_confidences=[0.9, 0.8, 0.95, 0.85],
            sign_flip_count=0,
            is_sparse_permutation=False,
            assignment_indices=None,
        )

    def test_fuse_identical_weights(self, basic_alignment):
        """Fusing identical weights should return the same weights."""
        weights = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        fused = PermutationAligner.fuse(weights, weights, basic_alignment)

        mx.eval(fused)
        assert fused.shape == weights.shape
        assert mx.allclose(fused, weights).item()

    def test_fuse_uses_confidence_weighting(self):
        """High confidence rows should blend, low confidence rows should preserve source."""
        N = 2
        # High confidence for row 0, low for row 1
        alignment = AlignmentResult(
            permutation=mx.eye(N, dtype=mx.float32),
            signs=mx.eye(N, dtype=mx.float32),
            match_quality=0.7,
            match_confidences=[1.0, 0.0],  # Row 0: full blend, Row 1: source only
            sign_flip_count=0,
            is_sparse_permutation=False,
            assignment_indices=None,
        )

        source = mx.array([[1.0, 1.0], [1.0, 1.0]])
        target = mx.array([[3.0, 3.0], [3.0, 3.0]])

        config = FusionConfig(source_alpha=0.5)
        fused = PermutationAligner.fuse(source, target, alignment, config)

        mx.eval(fused)
        # Row 0 (confidence=1.0): full average = (1+3)/2 = 2.0
        # Row 1 (confidence=0.0): source only = 1.0
        assert mx.isclose(fused[0, 0], mx.array(2.0)).item()
        assert mx.isclose(fused[1, 0], mx.array(1.0)).item()

    def test_fuse_respects_source_alpha(self):
        """Source alpha should control blending ratio."""
        N = 2
        alignment = AlignmentResult(
            permutation=mx.eye(N, dtype=mx.float32),
            signs=mx.eye(N, dtype=mx.float32),
            match_quality=1.0,
            match_confidences=[1.0, 1.0],
            sign_flip_count=0,
            is_sparse_permutation=False,
            assignment_indices=None,
        )

        source = mx.array([[0.0], [0.0]])
        target = mx.array([[10.0], [10.0]])

        # 80% source, 20% target
        config = FusionConfig(source_alpha=0.8)
        fused = PermutationAligner.fuse(source, target, alignment, config)

        mx.eval(fused)
        # Expected: 0.8 * 0 + 0.2 * 10 = 2.0
        assert mx.allclose(fused, mx.array([[2.0], [2.0]]), atol=1e-5).item()

    def test_fuse_config_default(self):
        """Default FusionConfig should work."""
        config = FusionConfig.default()
        assert config.source_alpha == 0.5
        assert config.interference_threshold == 0.5
        assert config.normalize is False


class TestPermutationAlignerRebasinMLP:
    """Tests for PermutationAligner.rebasin_mlp_only()."""

    @pytest.fixture
    def mlp_weights(self):
        """Create mock MLP weights for testing."""
        hidden = 4
        intermediate = 8
        return {
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": mx.random.normal((intermediate, hidden)),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((hidden, intermediate)),
            "model.layers.0.self_attn.q_proj.weight": mx.random.normal((hidden, hidden)),
            "model.layers.0.self_attn.k_proj.weight": mx.random.normal((hidden, hidden)),
        }

    @pytest.fixture
    def anchors(self):
        """Create anchor embeddings."""
        return mx.random.normal((5, 4))  # 5 anchors, dim 4

    def test_rebasin_mlp_returns_aligned_weights(self, mlp_weights, anchors):
        """rebasin_mlp_only should return aligned MLP weights."""
        target_weights = {k: mx.random.normal(v.shape) for k, v in mlp_weights.items()}

        aligned, avg_quality, blocks_aligned = PermutationAligner.rebasin_mlp_only(
            source_weights=mlp_weights,
            target_weights=target_weights,
            anchors=anchors,
        )

        # Should have aligned the MLP blocks
        assert blocks_aligned >= 1
        assert 0.0 <= avg_quality <= 1.0
        # MLP weights should be in the result
        assert "model.layers.0.mlp.up_proj.weight" in aligned
        assert "model.layers.0.mlp.gate_proj.weight" in aligned
        assert "model.layers.0.mlp.down_proj.weight" in aligned
        # Attention weights should also be present (copied unchanged)
        assert "model.layers.0.self_attn.q_proj.weight" in aligned

    def test_rebasin_mlp_preserves_shapes(self, mlp_weights, anchors):
        """Aligned weights should have the same shapes as source."""
        target_weights = {k: mx.random.normal(v.shape) for k, v in mlp_weights.items()}

        aligned, _, _ = PermutationAligner.rebasin_mlp_only(
            source_weights=mlp_weights,
            target_weights=target_weights,
            anchors=anchors,
        )

        for key, value in mlp_weights.items():
            assert aligned[key].shape == value.shape

    def test_rebasin_mlp_handles_incomplete_blocks(self, anchors):
        """Should gracefully skip incomplete MLP blocks."""
        # Missing gate_proj
        incomplete = {
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((8, 4)),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((4, 8)),
        }
        target = {
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((8, 4)),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((4, 8)),
        }

        aligned, avg_quality, blocks_aligned = PermutationAligner.rebasin_mlp_only(
            source_weights=incomplete,
            target_weights=target,
            anchors=anchors,
        )

        # Should have 0 blocks aligned (incomplete)
        assert blocks_aligned == 0


class TestPermutationAlignerRebasinWithActivations:
    """Tests for PermutationAligner.rebasin_mlp_with_activations()."""

    @pytest.fixture
    def mlp_weights(self):
        """Create mock MLP weights for testing."""
        hidden = 4
        intermediate = 8
        return {
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": mx.random.normal((intermediate, hidden)),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((hidden, intermediate)),
        }

    @pytest.fixture
    def anchors(self):
        """Create anchor embeddings."""
        return mx.random.normal((5, 4))

    @pytest.fixture
    def anchor_context(self):
        """Create anchor activation context."""
        return AnchorActivationContext(
            anchor_ids=["a1", "a2", "a3"],
            source_by_layer={
                0: [[0.1, 0.2, 0.3, 0.4] for _ in range(3)],
            },
            target_by_layer={
                0: [[0.2, 0.3, 0.4, 0.5] for _ in range(3)],
            },
        )

    def test_rebasin_with_activations_uses_context(
        self, mlp_weights, anchors, anchor_context
    ):
        """rebasin_mlp_with_activations should use per-layer anchor activations."""
        target_weights = {k: mx.random.normal(v.shape) for k, v in mlp_weights.items()}

        aligned, avg_quality, blocks = PermutationAligner.rebasin_mlp_with_activations(
            source_weights=mlp_weights,
            target_weights=target_weights,
            anchors=anchors,
            anchor_activations=anchor_context,
        )

        assert blocks >= 0
        # Should complete without error
        assert aligned is not None

    def test_anchor_context_activations_method(self, anchor_context):
        """AnchorActivationContext.activations() should return layer data."""
        result = anchor_context.activations(0)
        assert result is not None
        source, target = result
        assert len(source) == 3
        assert len(target) == 3

        # Non-existent layer returns None
        assert anchor_context.activations(99) is None


class TestPermutationAlignerHelpers:
    """Tests for PermutationAligner helper methods."""

    def test_is_mlp_weight(self):
        """is_mlp_weight should identify MLP weight keys."""
        assert PermutationAligner.is_mlp_weight("layers.0.mlp.up_proj.weight") is True
        assert PermutationAligner.is_mlp_weight("layers.0.mlp.gate_proj.weight") is True
        assert PermutationAligner.is_mlp_weight("layers.0.mlp.down_proj.weight") is True
        assert PermutationAligner.is_mlp_weight("layers.0.ffn.w1.weight") is True
        assert PermutationAligner.is_mlp_weight("layers.0.ffn.w2.weight") is True
        assert PermutationAligner.is_mlp_weight("layers.0.ffn.w3.weight") is True

        # Not MLP
        assert PermutationAligner.is_mlp_weight("layers.0.attn.q_proj.weight") is False
        assert PermutationAligner.is_mlp_weight("embed_tokens.weight") is False

    def test_is_attention_weight(self):
        """is_attention_weight should identify attention weight keys."""
        assert PermutationAligner.is_attention_weight("layers.0.attn.q_proj.weight") is True
        assert PermutationAligner.is_attention_weight("layers.0.attn.k_proj.weight") is True
        assert PermutationAligner.is_attention_weight("layers.0.attn.v_proj.weight") is True
        assert PermutationAligner.is_attention_weight("layers.0.attn.o_proj.weight") is True
        assert PermutationAligner.is_attention_weight("layers.0.self_attn.wq.weight") is True
        assert PermutationAligner.is_attention_weight("layers.0.self_attn.wk.weight") is True

        # Not attention
        assert PermutationAligner.is_attention_weight("layers.0.mlp.up_proj.weight") is False
        assert PermutationAligner.is_attention_weight("lm_head.weight") is False

    def test_extract_layer_index(self):
        """_extract_layer_index should parse layer numbers from keys."""
        assert PermutationAligner._extract_layer_index("model.layers.5.mlp") == 5
        assert PermutationAligner._extract_layer_index("transformer.h.12.attn") == 12
        assert PermutationAligner._extract_layer_index("model.blocks.0.ffn") == 0
        assert PermutationAligner._extract_layer_index("model.block.42.proj") == 42

        # No layer index
        assert PermutationAligner._extract_layer_index("embed_tokens") is None


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_default_values(self):
        """Default FusionConfig should have expected values."""
        config = FusionConfig()
        assert config.interference_threshold == 0.5
        assert config.source_alpha == 0.5
        assert config.normalize is False

    def test_custom_values(self):
        """FusionConfig should accept custom values."""
        config = FusionConfig(
            interference_threshold=0.7,
            source_alpha=0.3,
            normalize=True,
        )
        assert config.interference_threshold == 0.7
        assert config.source_alpha == 0.3
        assert config.normalize is True


class TestPermutationAlignerError:
    """Tests for PermutationAlignerError exception."""

    def test_error_can_be_raised(self):
        """PermutationAlignerError should be raiseable."""
        with pytest.raises(PermutationAlignerError, match="test error"):
            raise PermutationAlignerError("test error")


class TestSparseMlpPermutation:
    """Tests for sparse permutation application."""

    def test_apply_sparse_mlp_permutation(self):
        """_apply_sparse_mlp_permutation should reorder without full matrix."""
        intermediate = 8
        hidden = 4

        source_up = mx.random.normal((intermediate, hidden))
        source_gate = mx.random.normal((intermediate, hidden))
        source_down = mx.random.normal((hidden, intermediate))

        # Simple swap: indices [1,0,2,3,4,5,6,7]
        indices = [1, 0, 2, 3, 4, 5, 6, 7]
        signs = mx.eye(intermediate, dtype=mx.float32)

        signed_up, signed_gate, signed_down = PermutationAligner._apply_sparse_mlp_permutation(
            source_up, source_gate, source_down, indices, signs
        )

        # Shapes should be preserved
        assert signed_up.shape == source_up.shape
        assert signed_gate.shape == source_gate.shape
        assert signed_down.shape == source_down.shape

        # First two rows should be swapped in up/gate
        mx.eval(signed_up, source_up)
        assert mx.allclose(signed_up[0], source_up[1]).item()
        assert mx.allclose(signed_up[1], source_up[0]).item()
