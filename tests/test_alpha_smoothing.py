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

"""Tests for alpha smoothing (Gaussian smoothing for model merging)."""

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alpha_smoothing import (
    AlphaSmoothingConfig,
    compute_gaussian_weights,
    gaussian_smooth_alpha_profile,
    interpolate_missing_layers,
    smooth_alpha_vectors,
)


class TestAlphaSmoothingConfig:
    """Tests for AlphaSmoothingConfig dataclass."""

    def test_default_values(self):
        config = AlphaSmoothingConfig()
        assert config.smoothing_window == 2
        assert config.sigma == 1.0
        assert config.alpha_min == 0.1
        assert config.alpha_max == 0.9

    def test_frozen(self):
        config = AlphaSmoothingConfig()
        with pytest.raises(AttributeError):
            config.smoothing_window = 5

    def test_with_parameters_defaults(self):
        config = AlphaSmoothingConfig.with_parameters()
        assert config.smoothing_window == 2
        assert config.sigma == 1.0
        assert config.alpha_min == 0.1
        assert config.alpha_max == 0.9

    def test_with_parameters_custom(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=3,
            sigma=2.0,
            alpha_min=0.2,
            alpha_max=0.8,
        )
        assert config.smoothing_window == 3
        assert config.sigma == 2.0
        assert config.alpha_min == 0.2
        assert config.alpha_max == 0.8

    def test_with_parameters_window_validation(self):
        with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
            AlphaSmoothingConfig.with_parameters(smoothing_window=0)

    def test_with_parameters_sigma_validation(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            AlphaSmoothingConfig.with_parameters(sigma=0)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            AlphaSmoothingConfig.with_parameters(sigma=-1.0)

    def test_with_parameters_alpha_min_validation(self):
        with pytest.raises(ValueError, match="alpha_min must be in"):
            AlphaSmoothingConfig.with_parameters(alpha_min=-0.1)
        with pytest.raises(ValueError, match="alpha_min must be in"):
            AlphaSmoothingConfig.with_parameters(alpha_min=1.5)

    def test_with_parameters_alpha_max_validation(self):
        with pytest.raises(ValueError, match="alpha_max must be in"):
            AlphaSmoothingConfig.with_parameters(alpha_max=-0.1)
        with pytest.raises(ValueError, match="alpha_max must be in"):
            AlphaSmoothingConfig.with_parameters(alpha_max=1.5)

    def test_with_parameters_alpha_range_validation(self):
        with pytest.raises(ValueError, match="alpha_min.*must be < alpha_max"):
            AlphaSmoothingConfig.with_parameters(alpha_min=0.5, alpha_max=0.5)
        with pytest.raises(ValueError, match="alpha_min.*must be < alpha_max"):
            AlphaSmoothingConfig.with_parameters(alpha_min=0.7, alpha_max=0.3)


class TestComputeGaussianWeights:
    """Tests for compute_gaussian_weights function."""

    def test_window_1_sigma_1(self):
        weights = compute_gaussian_weights(window=1, sigma=1.0)
        # Window 1 means offsets [-1, 0, +1]
        assert len(weights) == 3
        # Center should be 1.0 (exp(0) = 1)
        assert weights[1] == 1.0
        # Symmetric around center
        assert weights[0] == weights[2]
        # exp(-1/2) ≈ 0.6065
        assert abs(weights[0] - math.exp(-0.5)) < 0.0001

    def test_window_2_sigma_1(self):
        weights = compute_gaussian_weights(window=2, sigma=1.0)
        # Window 2 means offsets [-2, -1, 0, +1, +2]
        assert len(weights) == 5
        assert weights[2] == 1.0  # Center
        assert weights[1] == weights[3]  # Symmetric
        assert weights[0] == weights[4]  # Symmetric

    def test_symmetry(self):
        weights = compute_gaussian_weights(window=3, sigma=2.0)
        assert len(weights) == 7
        # Check symmetry
        for i in range(len(weights) // 2 + 1):
            assert abs(weights[i] - weights[len(weights) - 1 - i]) < 1e-10

    def test_center_always_one(self):
        for window in [1, 2, 3, 5]:
            for sigma in [0.5, 1.0, 2.0, 5.0]:
                weights = compute_gaussian_weights(window, sigma)
                center_idx = window  # offset 0 is at position window
                assert weights[center_idx] == 1.0

    def test_larger_sigma_flatter(self):
        # Larger sigma should give more uniform weights
        weights_narrow = compute_gaussian_weights(window=2, sigma=0.5)
        weights_wide = compute_gaussian_weights(window=2, sigma=2.0)

        # For wider sigma, edge weights should be closer to center
        narrow_ratio = weights_narrow[0] / weights_narrow[2]  # edge / center
        wide_ratio = weights_wide[0] / weights_wide[2]

        assert wide_ratio > narrow_ratio

    def test_all_positive(self):
        weights = compute_gaussian_weights(window=3, sigma=1.0)
        assert all(w > 0 for w in weights)


class TestGaussianSmoothAlphaProfile:
    """Tests for gaussian_smooth_alpha_profile function."""

    def test_empty_input(self):
        config = AlphaSmoothingConfig()
        result = gaussian_smooth_alpha_profile({}, config)
        assert result == {}

    def test_single_layer(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=2, alpha_min=0.0, alpha_max=1.0
        )
        raw = {0: 0.5}
        result = gaussian_smooth_alpha_profile(raw, config)
        # Single layer should stay at 0.5 (no neighbors to smooth with)
        assert 0 in result
        assert result[0] == 0.5

    def test_uniform_alphas_unchanged(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=2, alpha_min=0.0, alpha_max=1.0
        )
        raw = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
        result = gaussian_smooth_alpha_profile(raw, config)
        # Uniform alphas should remain uniform
        for layer in raw:
            assert abs(result[layer] - 0.5) < 0.0001

    def test_smoothing_effect(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, sigma=1.0, alpha_min=0.0, alpha_max=1.0
        )
        # Sharp transition from 0.2 to 0.8
        raw = {0: 0.2, 1: 0.2, 2: 0.8, 3: 0.8, 4: 0.8}
        result = gaussian_smooth_alpha_profile(raw, config)

        # The transition should be smoother
        # Layer 1 should be pulled up by layer 2
        assert result[1] > 0.2
        # Layer 2 should be pulled down by layer 1
        assert result[2] < 0.8

    def test_clamping_alpha_min(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.3, alpha_max=0.9
        )
        raw = {0: 0.1, 1: 0.1, 2: 0.1}
        result = gaussian_smooth_alpha_profile(raw, config)
        # All results should be clamped to alpha_min
        for layer in result:
            assert result[layer] >= 0.3

    def test_clamping_alpha_max(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.1, alpha_max=0.7
        )
        raw = {0: 0.9, 1: 0.9, 2: 0.9}
        result = gaussian_smooth_alpha_profile(raw, config)
        # All results should be clamped to alpha_max
        for layer in result:
            assert result[layer] <= 0.7

    def test_non_contiguous_layers(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.0, alpha_max=1.0
        )
        # Layers 0, 5, 10 (gaps in between)
        raw = {0: 0.3, 5: 0.5, 10: 0.7}
        result = gaussian_smooth_alpha_profile(raw, config)

        # Each layer should only be affected by itself (no neighbors within window)
        assert result[0] == 0.3
        assert result[5] == 0.5
        assert result[10] == 0.7

    def test_larger_window(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=3, sigma=2.0, alpha_min=0.0, alpha_max=1.0
        )
        raw = {i: 0.5 for i in range(10)}
        raw[5] = 0.9  # Spike at layer 5

        result = gaussian_smooth_alpha_profile(raw, config)

        # Spike should be smoothed out
        assert result[5] < 0.9
        # Neighbors should be slightly elevated
        assert result[4] > 0.5
        assert result[6] > 0.5


class TestSmoothAlphaVectors:
    """Tests for smooth_alpha_vectors function."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_empty_input(self, backend):
        config = AlphaSmoothingConfig()
        result = smooth_alpha_vectors({}, config, backend)
        assert result == {}

    def test_single_layer(self, backend):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.0, alpha_max=1.0
        )
        vec = backend.array([0.5, 0.5, 0.5, 0.5])
        raw = {0: vec}
        result = smooth_alpha_vectors(raw, config, backend)

        assert 0 in result
        result_np = backend.to_numpy(result[0])
        assert all(abs(v - 0.5) < 0.0001 for v in result_np)

    def test_uniform_vectors_unchanged(self, backend):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.0, alpha_max=1.0
        )
        raw = {i: backend.array([0.5, 0.5, 0.5, 0.5]) for i in range(5)}
        result = smooth_alpha_vectors(raw, config, backend)

        for layer in raw:
            result_np = backend.to_numpy(result[layer])
            assert all(abs(v - 0.5) < 0.0001 for v in result_np)

    def test_smoothing_effect(self, backend):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, sigma=1.0, alpha_min=0.0, alpha_max=1.0
        )
        # Create vectors with sharp transition
        raw = {
            0: backend.array([0.2, 0.2, 0.2, 0.2]),
            1: backend.array([0.2, 0.2, 0.2, 0.2]),
            2: backend.array([0.8, 0.8, 0.8, 0.8]),
            3: backend.array([0.8, 0.8, 0.8, 0.8]),
        }
        result = smooth_alpha_vectors(raw, config, backend)

        # Layer 1 should be pulled up
        result_1 = backend.to_numpy(result[1])
        assert result_1[0] > 0.2

        # Layer 2 should be pulled down
        result_2 = backend.to_numpy(result[2])
        assert result_2[0] < 0.8

    def test_clamping(self, backend):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=1, alpha_min=0.3, alpha_max=0.7
        )
        raw = {
            0: backend.array([0.1, 0.9, 0.5, 0.5]),
            1: backend.array([0.1, 0.9, 0.5, 0.5]),
        }
        result = smooth_alpha_vectors(raw, config, backend)

        for layer in result:
            result_np = backend.to_numpy(result[layer])
            assert all(0.3 <= v <= 0.7 for v in result_np)

    def test_preserves_dimension(self, backend):
        config = AlphaSmoothingConfig()
        hidden_dim = 64
        backend.random_seed(42)
        raw = {i: backend.random_normal((hidden_dim,)) for i in range(5)}
        result = smooth_alpha_vectors(raw, config, backend)

        for layer in result:
            assert result[layer].shape == (hidden_dim,)


class TestInterpolateMissingLayers:
    """Tests for interpolate_missing_layers function."""

    def test_empty_alphas(self):
        result = interpolate_missing_layers({}, [0, 1, 2, 3])
        assert result == {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}

    def test_all_layers_present(self):
        alphas = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}
        result = interpolate_missing_layers(alphas, [0, 1, 2, 3])
        assert result == alphas

    def test_interpolate_middle(self):
        alphas = {0: 0.2, 4: 0.6}
        result = interpolate_missing_layers(alphas, [0, 1, 2, 3, 4])

        assert result[0] == 0.2
        assert result[4] == 0.6
        # Linear interpolation: 0.2 + (0.6 - 0.2) * t
        assert abs(result[1] - 0.3) < 0.0001  # t = 0.25
        assert abs(result[2] - 0.4) < 0.0001  # t = 0.5
        assert abs(result[3] - 0.5) < 0.0001  # t = 0.75

    def test_extrapolate_before(self):
        alphas = {2: 0.5, 4: 0.7}
        result = interpolate_missing_layers(alphas, [0, 1, 2, 3, 4])

        # Layers before first known should copy first known
        assert result[0] == 0.5
        assert result[1] == 0.5

    def test_extrapolate_after(self):
        alphas = {0: 0.3, 2: 0.5}
        result = interpolate_missing_layers(alphas, [0, 1, 2, 3, 4])

        # Layers after last known should copy last known
        assert result[3] == 0.5
        assert result[4] == 0.5

    def test_single_known_layer(self):
        alphas = {5: 0.6}
        result = interpolate_missing_layers(alphas, [0, 2, 5, 8, 10])

        # All missing layers should use the single known value
        # Layers before: copy first known
        assert result[0] == 0.6
        assert result[2] == 0.6
        assert result[5] == 0.6
        # Layers after: copy last known
        assert result[8] == 0.6
        assert result[10] == 0.6

    def test_non_contiguous_known_layers(self):
        alphas = {0: 0.2, 10: 0.4, 20: 0.8}
        result = interpolate_missing_layers(alphas, [0, 5, 10, 15, 20])

        assert result[0] == 0.2
        # Between 0 and 10: t = 5/10 = 0.5
        assert abs(result[5] - 0.3) < 0.0001
        assert result[10] == 0.4
        # Between 10 and 20: t = 5/10 = 0.5
        assert abs(result[15] - 0.6) < 0.0001
        assert result[20] == 0.8


class TestMathematicalProperties:
    """Tests for mathematical properties of alpha smoothing."""

    def test_gaussian_weights_sum_normalization(self):
        # When applied, weights should be normalized
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=2, sigma=1.0, alpha_min=0.0, alpha_max=1.0
        )
        raw = {i: 0.5 for i in range(10)}
        result = gaussian_smooth_alpha_profile(raw, config)

        # All values should still be 0.5 (normalized properly)
        for layer in result:
            assert abs(result[layer] - 0.5) < 0.0001

    def test_smoothing_reduces_variance(self):
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=2, sigma=1.5, alpha_min=0.0, alpha_max=1.0
        )
        # High variance input
        raw = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.3}
        result = gaussian_smooth_alpha_profile(raw, config)

        # Compute variance before and after
        raw_vals = list(raw.values())
        raw_mean = sum(raw_vals) / len(raw_vals)
        raw_var = sum((v - raw_mean) ** 2 for v in raw_vals) / len(raw_vals)

        result_vals = list(result.values())
        result_mean = sum(result_vals) / len(result_vals)
        result_var = sum((v - result_mean) ** 2 for v in result_vals) / len(result_vals)

        # Smoothing should reduce variance
        assert result_var < raw_var

    def test_interpolation_monotonic(self):
        # Linear interpolation between monotonic points should be monotonic
        alphas = {0: 0.2, 5: 0.5, 10: 0.8}
        result = interpolate_missing_layers(alphas, list(range(11)))

        # Check monotonically increasing
        values = [result[i] for i in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]
