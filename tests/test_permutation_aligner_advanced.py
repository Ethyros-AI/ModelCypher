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

"""Tests for advanced PermutationAligner methods (requires MLX)."""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain.geometry.permutation_aligner import (
    AnchorActivationContext,
    PermutationAligner,
    PermutationAlignerError,
)


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
    def source_anchors(self):
        """Create source model anchor embeddings."""
        return mx.random.normal((5, 4))

    @pytest.fixture
    def target_anchors(self):
        """Create target model anchor embeddings."""
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
        self, mlp_weights, source_anchors, target_anchors, anchor_context
    ):
        """rebasin_mlp_with_activations should use per-layer anchor activations."""
        target_weights = {k: mx.random.normal(v.shape) for k, v in mlp_weights.items()}

        aligned, avg_quality, blocks = PermutationAligner.rebasin_mlp_with_activations(
            source_weights=mlp_weights,
            target_weights=target_weights,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            anchor_activations=anchor_context,
        )

        assert blocks >= 0
        # Should complete without error
        assert aligned is not None

    def test_rebasin_with_separate_anchors(
        self, mlp_weights, source_anchors, target_anchors
    ):
        """rebasin_mlp_with_activations should work with separate anchors (no per-layer context)."""
        target_weights = {k: mx.random.normal(v.shape) for k, v in mlp_weights.items()}

        aligned, avg_quality, blocks = PermutationAligner.rebasin_mlp_with_activations(
            source_weights=mlp_weights,
            target_weights=target_weights,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
        )

        assert blocks >= 0
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
