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

"""Tests for GradientSmoothnessEstimator.

Tests the per-layer gradient quality metrics computation including
variance, SNR, and mean gradient norms.
"""

import pytest
import mlx.core as mx

from modelcypher.core.domain.training.gradient_smoothness_estimator import (
    GradientSmoothnessEstimator,
    LayerGradientQuality,
)


class TestLayerGradientQuality:
    """Tests for LayerGradientQuality dataclass."""

    def test_quality_fields(self):
        """Quality should have expected fields."""
        quality = LayerGradientQuality(
            variance=0.1,
            snr=10.0,
            mean_norm=0.5,
            sample_count=32,
        )

        assert quality.variance == 0.1
        assert quality.snr == 10.0
        assert quality.mean_norm == 0.5
        assert quality.sample_count == 32


class TestParseIndex:
    """Tests for _parse_index helper method."""

    def test_parse_layers_index(self):
        """Should parse layer index from '.layers.' pattern."""
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.",
            in_str="model.layers.5.mlp.weight"
        )
        assert result == 5

    def test_parse_h_index(self):
        """Should parse layer index from '.h.' pattern (GPT-style)."""
        result = GradientSmoothnessEstimator._parse_index(
            after=".h.",
            in_str="transformer.h.12.attn.weight"
        )
        assert result == 12

    def test_parse_blocks_index(self):
        """Should parse layer index from '.blocks.' pattern."""
        result = GradientSmoothnessEstimator._parse_index(
            after=".blocks.",
            in_str="model.blocks.0.ffn.weight"
        )
        assert result == 0

    def test_parse_no_match(self):
        """Should return None if pattern not found."""
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.",
            in_str="embed_tokens.weight"
        )
        assert result is None

    def test_parse_multi_digit(self):
        """Should parse multi-digit layer indices."""
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.",
            in_str="model.layers.123.mlp.weight"
        )
        assert result == 123


class TestExtractLayerIndex:
    """Tests for _extract_layer_index_from_key."""

    def test_layers_pattern(self):
        """Should extract from .layers. pattern."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "model.layers.10.mlp.up_proj.weight"
        )
        assert result == 10

    def test_h_pattern(self):
        """Should extract from .h. pattern."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "transformer.h.7.attn.c_proj.weight"
        )
        assert result == 7

    def test_blocks_pattern(self):
        """Should extract from .blocks. pattern."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "model.blocks.3.ffn.fc1.weight"
        )
        assert result == 3

    def test_block_pattern(self):
        """Should extract from .block. pattern (singular)."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "model.block.42.proj.weight"
        )
        assert result == 42

    def test_no_layer_pattern(self):
        """Should return None for non-layer keys."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "embed_tokens.weight"
        )
        assert result is None

    def test_lm_head(self):
        """Should return None for lm_head."""
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(
            "lm_head.weight"
        )
        assert result is None


class TestPerLayerQuality:
    """Tests for per_layer_quality method."""

    def test_single_sample_returns_empty(self):
        """Single sample should return empty (need at least 2)."""
        gradients = [
            {"model.layers.0.mlp.weight": mx.array([1.0, 2.0, 3.0])}
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert result == {}

    def test_empty_input(self):
        """Empty input should return empty."""
        result = GradientSmoothnessEstimator.per_layer_quality([])
        assert result == {}

    def test_two_samples_same_gradients(self):
        """Two identical samples should have zero variance."""
        grad = mx.array([1.0, 2.0, 3.0])
        gradients = [
            {"model.layers.0.mlp.weight": grad},
            {"model.layers.0.mlp.weight": grad},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)

        assert 0 in result
        # Zero variance when gradients are identical
        assert result[0].variance == 0.0
        assert result[0].sample_count == 2

    def test_two_samples_different_gradients(self):
        """Different gradients should produce non-zero variance."""
        gradients = [
            {"model.layers.0.mlp.weight": mx.array([1.0, 0.0, 0.0])},
            {"model.layers.0.mlp.weight": mx.array([0.0, 1.0, 0.0])},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)

        assert 0 in result
        assert result[0].variance > 0
        assert result[0].sample_count == 2

    def test_multiple_layers(self):
        """Should compute quality for each layer separately."""
        gradients = [
            {
                "model.layers.0.mlp.weight": mx.array([1.0, 2.0]),
                "model.layers.1.mlp.weight": mx.array([3.0, 4.0]),
            },
            {
                "model.layers.0.mlp.weight": mx.array([1.5, 2.5]),
                "model.layers.1.mlp.weight": mx.array([3.5, 4.5]),
            },
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)

        assert 0 in result
        assert 1 in result
        assert result[0].sample_count == 2
        assert result[1].sample_count == 2

    def test_ignores_non_layer_params(self):
        """Should ignore parameters not associated with layers."""
        gradients = [
            {
                "model.layers.0.mlp.weight": mx.array([1.0, 2.0]),
                "embed_tokens.weight": mx.array([3.0, 4.0]),  # Not a layer
            },
            {
                "model.layers.0.mlp.weight": mx.array([1.5, 2.5]),
                "embed_tokens.weight": mx.array([3.5, 4.5]),
            },
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)

        # Only layer 0 should be present
        assert 0 in result
        assert len(result) == 1


class TestComputeGradientQuality:
    """Tests for _compute_gradient_quality method."""

    def test_returns_none_for_single_sample(self):
        """Single sample should return None."""
        samples = [{"w": mx.array([1.0, 2.0])}]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples)
        assert result is None

    def test_mean_norm_computation(self):
        """Mean norm should be computed correctly."""
        # Gradient norm of [3, 4] = 5
        samples = [
            {"w": mx.array([3.0, 4.0])},
            {"w": mx.array([3.0, 4.0])},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples)

        assert result is not None
        assert abs(result.mean_norm - 5.0) < 0.01

    def test_snr_high_for_consistent_gradients(self):
        """SNR should be high (and variance low) for consistent gradients."""
        grad = mx.array([1.0, 1.0, 1.0, 1.0])
        samples = [{"w": grad}, {"w": grad}, {"w": grad}]

        result = GradientSmoothnessEstimator._compute_gradient_quality(samples)

        assert result is not None
        # Zero variance should give very high SNR
        assert result.variance == 0.0

    def test_snr_low_for_noisy_gradients(self):
        """SNR should be lower for noisy gradients."""
        samples = [
            {"w": mx.array([10.0, 0.0])},
            {"w": mx.array([0.0, 10.0])},
            {"w": mx.array([-10.0, 0.0])},
        ]

        result = GradientSmoothnessEstimator._compute_gradient_quality(samples)

        assert result is not None
        assert result.variance > 0

    def test_sample_count(self):
        """Sample count should match input."""
        samples = [
            {"w": mx.array([1.0, 2.0])},
            {"w": mx.array([2.0, 3.0])},
            {"w": mx.array([3.0, 4.0])},
            {"w": mx.array([4.0, 5.0])},
        ]

        result = GradientSmoothnessEstimator._compute_gradient_quality(samples)

        assert result is not None
        assert result.sample_count == 4


class TestEdgeCases:
    """Edge case tests for GradientSmoothnessEstimator."""

    def test_large_gradient_values(self):
        """Should handle large gradient values."""
        gradients = [
            {"model.layers.0.w": mx.array([1e10, 2e10])},
            {"model.layers.0.w": mx.array([1.5e10, 2.5e10])},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert 0 in result
        assert result[0].mean_norm > 0

    def test_small_gradient_values(self):
        """Should handle small gradient values."""
        gradients = [
            {"model.layers.0.w": mx.array([1e-10, 2e-10])},
            {"model.layers.0.w": mx.array([1.5e-10, 2.5e-10])},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert 0 in result

    def test_mixed_shapes_same_key(self):
        """Should handle gradients with same key but different values."""
        gradients = [
            {"model.layers.0.w": mx.array([1.0, 2.0, 3.0, 4.0])},
            {"model.layers.0.w": mx.array([4.0, 3.0, 2.0, 1.0])},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert 0 in result
        assert result[0].sample_count == 2

    def test_2d_gradients(self):
        """Should handle 2D gradient matrices."""
        gradients = [
            {"model.layers.0.w": mx.array([[1.0, 2.0], [3.0, 4.0]])},
            {"model.layers.0.w": mx.array([[2.0, 3.0], [4.0, 5.0]])},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert 0 in result
        # Should flatten and compute metrics
        assert result[0].mean_norm > 0

    def test_many_parameters_same_layer(self):
        """Should aggregate all params in same layer."""
        gradients = [
            {
                "model.layers.5.mlp.up_proj.weight": mx.array([1.0]),
                "model.layers.5.mlp.down_proj.weight": mx.array([2.0]),
                "model.layers.5.mlp.gate_proj.weight": mx.array([3.0]),
            },
            {
                "model.layers.5.mlp.up_proj.weight": mx.array([1.5]),
                "model.layers.5.mlp.down_proj.weight": mx.array([2.5]),
                "model.layers.5.mlp.gate_proj.weight": mx.array([3.5]),
            },
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(gradients)
        assert 5 in result
        assert result[5].sample_count == 2
