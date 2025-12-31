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

"""Tests for gradient smoothness estimator (per-layer gradient quality)."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.gradient_smoothness_estimator import (
    GradientSmoothnessEstimator,
    LayerGradientQuality,
)


@pytest.fixture
def backend():
    """Get the default backend for tests."""
    return get_default_backend()


class TestLayerGradientQuality:
    """Tests for LayerGradientQuality dataclass."""

    def test_required_fields(self):
        quality = LayerGradientQuality(
            variance=0.1,
            snr=10.0,
            mean_norm=0.5,
            sample_count=8,
        )
        assert quality.variance == 0.1
        assert quality.snr == 10.0
        assert quality.mean_norm == 0.5
        assert quality.sample_count == 8

    def test_zero_values(self):
        quality = LayerGradientQuality(
            variance=0.0,
            snr=0.0,
            mean_norm=0.0,
            sample_count=0,
        )
        assert quality.variance == 0.0
        assert quality.snr == 0.0

    def test_large_values(self):
        quality = LayerGradientQuality(
            variance=1e6,
            snr=1e12,
            mean_norm=1e3,
            sample_count=10000,
        )
        assert quality.variance == 1e6
        assert quality.sample_count == 10000


class TestExtractLayerIndex:
    """Tests for _extract_layer_index_from_key() method."""

    def test_layers_pattern(self):
        key = "model.layers.5.self_attn.q_proj.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 5

    def test_h_pattern(self):
        # GPT-2 style: transformer.h.12.attn.c_attn.weight
        key = "transformer.h.12.attn.c_attn.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 12

    def test_blocks_pattern(self):
        key = "model.blocks.0.mlp.fc1.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 0

    def test_block_pattern(self):
        key = "encoder.block.7.layer.0.SelfAttention.q.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 7

    def test_no_match_returns_none(self):
        key = "embeddings.word_embeddings.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result is None

    def test_double_digit_layer(self):
        key = "model.layers.42.self_attn.k_proj.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 42

    def test_triple_digit_layer(self):
        key = "model.layers.123.mlp.gate_proj.weight"
        result = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
        assert result == 123


class TestParseIndex:
    """Tests for _parse_index() helper method."""

    def test_extracts_digits_after_pattern(self):
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.", in_str="model.layers.10.weight"
        )
        assert result == 10

    def test_returns_none_when_pattern_not_found(self):
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.", in_str="model.blocks.5.weight"
        )
        assert result is None

    def test_extracts_first_number_only(self):
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.", in_str="model.layers.3.sublayer.7.weight"
        )
        assert result == 3

    def test_handles_zero_index(self):
        result = GradientSmoothnessEstimator._parse_index(
            after=".h.", in_str="transformer.h.0.attn.weight"
        )
        assert result == 0

    def test_returns_none_for_no_digits(self):
        result = GradientSmoothnessEstimator._parse_index(
            after=".layers.", in_str="model.layers.abc.weight"
        )
        assert result is None


class TestPerLayerQuality:
    """Tests for per_layer_quality() method."""

    def test_empty_list_returns_empty_dict(self, backend):
        result = GradientSmoothnessEstimator.per_layer_quality([], backend)
        assert result == {}

    def test_single_sample_returns_empty_dict(self, backend):
        # Need at least 2 samples for variance
        sample = {"model.layers.0.weight": backend.ones((10, 10))}
        result = GradientSmoothnessEstimator.per_layer_quality([sample], backend)
        assert result == {}

    def test_groups_by_layer(self, backend):
        # Create samples with gradients for layers 0 and 1
        samples = [
            {
                "model.layers.0.weight": backend.ones((5, 5)),
                "model.layers.1.weight": backend.ones((5, 5)) * 2,
            },
            {
                "model.layers.0.weight": backend.ones((5, 5)) * 1.1,
                "model.layers.1.weight": backend.ones((5, 5)) * 2.1,
            },
        ]
        result = GradientSmoothnessEstimator.per_layer_quality(samples, backend)

        assert 0 in result
        assert 1 in result
        assert len(result) == 2

    def test_returns_quality_metrics(self, backend):
        samples = [
            {"model.layers.0.weight": backend.ones((5, 5))},
            {"model.layers.0.weight": backend.ones((5, 5)) * 2},
        ]
        result = GradientSmoothnessEstimator.per_layer_quality(samples, backend)

        assert 0 in result
        quality = result[0]
        assert isinstance(quality, LayerGradientQuality)
        assert quality.sample_count == 2
        assert quality.variance > 0
        assert quality.mean_norm > 0
        assert quality.snr >= 0

    def test_ignores_non_layer_params(self, backend):
        samples = [
            {
                "model.embed_tokens.weight": backend.ones((100, 64)),
                "model.layers.0.weight": backend.ones((5, 5)),
            },
            {
                "model.embed_tokens.weight": backend.ones((100, 64)) * 1.1,
                "model.layers.0.weight": backend.ones((5, 5)) * 1.1,
            },
        ]
        result = GradientSmoothnessEstimator.per_layer_quality(samples, backend)

        # Only layer 0 should be present
        assert 0 in result
        assert len(result) == 1


class TestComputeGradientQuality:
    """Tests for _compute_gradient_quality() method."""

    def test_single_sample_returns_none(self, backend):
        sample = {"weight": backend.ones((10, 10))}
        result = GradientSmoothnessEstimator._compute_gradient_quality([sample], backend)
        assert result is None

    def test_empty_list_returns_none(self, backend):
        result = GradientSmoothnessEstimator._compute_gradient_quality([], backend)
        assert result is None

    def test_identical_gradients_zero_variance(self, backend):
        # Identical gradients should have zero variance
        samples = [
            {"weight": backend.ones((5, 5))},
            {"weight": backend.ones((5, 5))},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        assert result.variance == pytest.approx(0.0, abs=1e-6)

    def test_different_gradients_nonzero_variance(self, backend):
        samples = [
            {"weight": backend.ones((5, 5))},
            {"weight": backend.ones((5, 5)) * 2},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        assert result.variance > 0

    def test_mean_norm_positive(self, backend):
        samples = [
            {"weight": backend.ones((5, 5))},
            {"weight": backend.ones((5, 5)) * 2},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        assert result.mean_norm > 0

    def test_snr_positive(self, backend):
        samples = [
            {"weight": backend.ones((5, 5))},
            {"weight": backend.ones((5, 5)) * 1.1},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        assert result.snr >= 0

    def test_sample_count_correct(self, backend):
        samples = [
            {"weight": backend.ones((3, 3))},
            {"weight": backend.ones((3, 3))},
            {"weight": backend.ones((3, 3))},
            {"weight": backend.ones((3, 3))},
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        assert result.sample_count == 4

    def test_multiple_params_aggregated(self, backend):
        samples = [
            {
                "weight1": backend.ones((5, 5)),
                "weight2": backend.ones((3, 3)),
            },
            {
                "weight1": backend.ones((5, 5)) * 2,
                "weight2": backend.ones((3, 3)) * 2,
            },
        ]
        result = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)

        assert result is not None
        # Mean norm should account for all parameters
        assert result.mean_norm > 0


class TestMathematicalProperties:
    """Tests for mathematical correctness of gradient quality metrics."""

    def test_higher_variance_lower_snr(self, backend):
        # Higher variance in gradients should lead to lower SNR
        low_variance_samples = [
            {"weight": backend.ones((10, 10))},
            {"weight": backend.ones((10, 10)) * 1.01},
        ]
        high_variance_samples = [
            {"weight": backend.ones((10, 10))},
            {"weight": backend.ones((10, 10)) * 10},
        ]

        low_var_result = GradientSmoothnessEstimator._compute_gradient_quality(
            low_variance_samples, backend
        )
        high_var_result = GradientSmoothnessEstimator._compute_gradient_quality(
            high_variance_samples, backend
        )

        assert low_var_result is not None
        assert high_var_result is not None
        assert low_var_result.variance < high_var_result.variance
        assert low_var_result.snr > high_var_result.snr

    def test_mean_norm_scales_with_gradient_magnitude(self, backend):
        small_grad_samples = [
            {"weight": backend.ones((5, 5)) * 0.1},
            {"weight": backend.ones((5, 5)) * 0.11},
        ]
        large_grad_samples = [
            {"weight": backend.ones((5, 5)) * 10},
            {"weight": backend.ones((5, 5)) * 11},
        ]

        small_result = GradientSmoothnessEstimator._compute_gradient_quality(
            small_grad_samples, backend
        )
        large_result = GradientSmoothnessEstimator._compute_gradient_quality(
            large_grad_samples, backend
        )

        assert small_result is not None
        assert large_result is not None
        assert small_result.mean_norm < large_result.mean_norm
