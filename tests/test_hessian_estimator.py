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

"""Tests for HessianEstimator.

Tests the Hessian estimation and gradient quality metrics used for
monitoring training dynamics and loss landscape geometry.
"""

import numpy as np
import pytest

from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
)
from modelcypher.core.domain.training.hessian_estimator import (
    Config,
    condition_proxy,
    config_for_level,
    effective_step_ratio,
    gradient_quality,
    hutchinson_trace_estimate,
    per_layer_analysis,
    top_eigenvalue,
    trajectory,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = Config()
        assert config.hutchinson_vectors == 5
        assert config.power_iterations == 20
        assert config.finite_difference_epsilon == pytest.approx(1e-4)
        assert config.power_iteration_tolerance == pytest.approx(1e-6)

    def test_moderate_config(self):
        """Moderate config should have reduced iterations."""
        config = Config.moderate()
        assert config.hutchinson_vectors == 3
        assert config.power_iterations == 10
        assert config.finite_difference_epsilon == pytest.approx(1e-3)

    def test_full_config(self):
        """Full config should have increased precision."""
        config = Config.full()
        assert config.hutchinson_vectors == 10
        assert config.power_iterations == 30
        assert config.finite_difference_epsilon == pytest.approx(1e-5)

    def test_config_for_level_minimal(self):
        """Minimal level should disable Hessian computation."""
        config = config_for_level(GeometricInstrumentationLevel.minimal)
        assert config.hutchinson_vectors == 0
        assert config.power_iterations == 0

    def test_config_for_level_moderate(self):
        """Moderate level should use moderate config."""
        config = config_for_level(GeometricInstrumentationLevel.moderate)
        assert config.hutchinson_vectors == 3

    def test_config_for_level_full(self):
        """Full level should use full config."""
        config = config_for_level(GeometricInstrumentationLevel.full)
        assert config.hutchinson_vectors == 10


class TestGradientQuality:
    """Tests for gradient_quality function."""

    def test_empty_input_returns_none(self):
        """Empty gradient list should return None."""
        result = gradient_quality([])
        assert result is None

    def test_single_sample_returns_none(self):
        """Single sample should return None (need variance)."""
        sample = {"layer1": np.array([1.0, 2.0, 3.0])}
        result = gradient_quality([sample])
        assert result is None

    def test_identical_gradients_zero_variance(self):
        """Identical gradients should have zero variance."""
        grad = {"layer1": np.array([1.0, 2.0, 3.0])}
        result = gradient_quality([grad, grad, grad])

        assert result is not None
        assert result.variance == pytest.approx(0.0, abs=1e-10)
        # SNR should be infinite (or very large) with zero variance
        assert result.snr == float("inf")

    def test_orthogonal_gradients_high_variance(self):
        """Orthogonal gradients should have high variance."""
        grad1 = {"layer1": np.array([1.0, 0.0, 0.0])}
        grad2 = {"layer1": np.array([0.0, 1.0, 0.0])}
        grad3 = {"layer1": np.array([0.0, 0.0, 1.0])}
        result = gradient_quality([grad1, grad2, grad3])

        assert result is not None
        assert result.variance > 0
        # Mean grad = [1/3, 1/3, 1/3], norm = sqrt(1/3)
        expected_mean_norm = np.sqrt(3 * (1 / 3) ** 2)
        assert result.mean_norm == pytest.approx(expected_mean_norm, rel=0.01)

    def test_known_variance_computation(self):
        """Test variance computation with known values."""
        # Two samples: [1, 0] and [0, 1]
        # Mean = [0.5, 0.5]
        # Sample 1: centered = [0.5, -0.5], squared_diff = 0.5
        # Sample 2: centered = [-0.5, 0.5], squared_diff = 0.5
        # Variance = mean(0.5, 0.5) = 0.5
        grad1 = {"layer1": np.array([1.0, 0.0])}
        grad2 = {"layer1": np.array([0.0, 1.0])}
        result = gradient_quality([grad1, grad2])

        assert result is not None
        assert result.variance == pytest.approx(0.5, rel=0.01)

    def test_snr_computation(self):
        """SNR should be mean_norm^2 / variance."""
        grad1 = {"layer1": np.array([2.0, 0.0])}
        grad2 = {"layer1": np.array([0.0, 2.0])}
        result = gradient_quality([grad1, grad2])

        assert result is not None
        # Mean = [1, 1], mean_norm = sqrt(2)
        # variance = 2 (each sample has squared_diff = 2)
        # SNR = 2 / 2 = 1
        assert result.snr == pytest.approx(1.0, rel=0.01)


class TestPerLayerAnalysis:
    """Tests for per_layer_analysis function."""

    def test_empty_gradients(self):
        """Empty gradients should return empty stats."""
        result = per_layer_analysis({})
        assert result.norms == {}
        assert result.fractions == {}
        assert result.active_layers == []

    def test_single_layer_norm(self):
        """Single layer should have norm = 1.0 fraction."""
        grads = {"layer1": np.array([3.0, 4.0])}  # norm = 5
        result = per_layer_analysis(grads)

        assert result.norms["layer1"] == pytest.approx(5.0)
        assert result.fractions["layer1"] == pytest.approx(1.0)
        assert "layer1" in result.active_layers

    def test_multiple_layers_fractions_sum_to_one(self):
        """Layer fractions should approximately sum to 1 (by L1 of norms)."""
        grads = {
            "layer1": np.array([3.0, 4.0]),  # norm = 5
            "layer2": np.array([12.0, 0.0]),  # norm = 12
        }
        result = per_layer_analysis(grads)

        # Total norm = sqrt(5^2 + 12^2) = 13
        assert result.norms["layer1"] == pytest.approx(5.0)
        assert result.norms["layer2"] == pytest.approx(12.0)
        assert result.fractions["layer1"] == pytest.approx(5.0 / 13.0)
        assert result.fractions["layer2"] == pytest.approx(12.0 / 13.0)

    def test_active_layers_threshold(self):
        """Only layers above threshold should be active."""
        grads = {
            "layer1": np.array([0.01]),  # small
            "layer2": np.array([10.0]),  # dominant
        }
        result = per_layer_analysis(grads, active_threshold=0.05)

        assert "layer2" in result.active_layers
        assert "layer1" not in result.active_layers


class TestTrajectory:
    """Tests for trajectory function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""
        assert trajectory({}, {"a": np.array([1.0])}) is None
        assert trajectory({"a": np.array([1.0])}, {}) is None

    def test_identical_params_zero_divergence(self):
        """Identical params should have zero divergence."""
        params = {"layer1": np.array([1.0, 2.0, 3.0])}
        result = trajectory(params, params)

        assert result is not None
        assert result.divergence == pytest.approx(0.0, abs=1e-10)
        assert result.cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_opposite_params_negative_cosine(self):
        """Opposite params should have cosine = -1."""
        current = {"layer1": np.array([1.0, 0.0])}
        initial = {"layer1": np.array([-1.0, 0.0])}
        result = trajectory(current, initial)

        assert result is not None
        assert result.cosine_similarity == pytest.approx(-1.0, abs=1e-6)
        assert result.divergence == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_params_zero_cosine(self):
        """Orthogonal params should have cosine = 0."""
        current = {"layer1": np.array([1.0, 0.0])}
        initial = {"layer1": np.array([0.0, 1.0])}
        result = trajectory(current, initial)

        assert result is not None
        assert result.cosine_similarity == pytest.approx(0.0, abs=1e-6)

    def test_divergence_computation(self):
        """Divergence should be L2 distance."""
        current = {"layer1": np.array([3.0, 4.0])}
        initial = {"layer1": np.array([0.0, 0.0])}
        result = trajectory(current, initial)

        assert result is not None
        assert result.divergence == pytest.approx(5.0, abs=1e-6)


class TestEffectiveStepRatio:
    """Tests for effective_step_ratio function."""

    def test_empty_inputs_return_none(self):
        """Empty inputs should return None."""
        assert effective_step_ratio({}, {"a": np.array([1.0])}, 0.1) is None
        assert effective_step_ratio({"a": np.array([1.0])}, {}, 0.1) is None

    def test_zero_learning_rate_returns_none(self):
        """Zero learning rate should return None."""
        step = {"layer1": np.array([1.0])}
        grad = {"layer1": np.array([1.0])}
        assert effective_step_ratio(step, grad, 0.0) is None

    def test_perfect_step_ratio_one(self):
        """When actual = lr * grad, ratio should be 1.0."""
        grad = {"layer1": np.array([1.0, 2.0])}
        lr = 0.1
        step = {"layer1": lr * grad["layer1"]}
        result = effective_step_ratio(step, grad, lr)

        assert result == pytest.approx(1.0, abs=1e-6)

    def test_doubled_step_ratio_two(self):
        """When actual = 2 * lr * grad, ratio should be 2.0."""
        grad = {"layer1": np.array([1.0, 2.0])}
        lr = 0.1
        step = {"layer1": 2.0 * lr * grad["layer1"]}
        result = effective_step_ratio(step, grad, lr)

        assert result == pytest.approx(2.0, abs=1e-6)


class TestHutchinsonTraceEstimate:
    """Tests for hutchinson_trace_estimate function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""

        def dummy_fn(params):
            return np.array(0.0), {}

        result = hutchinson_trace_estimate(dummy_fn, {}, Config())
        assert result is None

    def test_zero_vectors_returns_none(self):
        """Zero hutchinson vectors should return None."""

        def dummy_fn(params):
            return np.array(0.0), params

        params = {"layer1": np.array([1.0])}
        config = Config(hutchinson_vectors=0)
        result = hutchinson_trace_estimate(dummy_fn, params, config)
        assert result is None

    def test_quadratic_function_known_trace(self):
        """For f(x) = 0.5 * x^T A x, trace(H) = trace(A)."""
        # Simple 2D quadratic: f(x) = 0.5 * (a*x1^2 + b*x2^2)
        # Hessian = diag(a, b), trace = a + b
        a, b = 2.0, 3.0

        def quadratic_loss_and_grad(params):
            x = params["layer1"]
            loss = 0.5 * (a * x[0] ** 2 + b * x[1] ** 2)
            grad = np.array([a * x[0], b * x[1]], dtype=np.float32)
            return np.array(loss), {"layer1": grad}

        params = {"layer1": np.array([1.0, 1.0], dtype=np.float32)}
        config = Config(hutchinson_vectors=50, finite_difference_epsilon=1e-5)
        result = hutchinson_trace_estimate(quadratic_loss_and_grad, params, config)

        assert result is not None
        # Trace should be a + b = 5.0
        assert result == pytest.approx(5.0, rel=0.1)


class TestTopEigenvalue:
    """Tests for top_eigenvalue function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""

        def dummy_fn(params):
            return np.array(0.0), {}

        result = top_eigenvalue(dummy_fn, {}, Config())
        assert result is None

    def test_zero_iterations_returns_none(self):
        """Zero power iterations should return None."""

        def dummy_fn(params):
            return np.array(0.0), params

        params = {"layer1": np.array([1.0])}
        config = Config(power_iterations=0)
        result = top_eigenvalue(dummy_fn, params, config)
        assert result is None

    def test_quadratic_function_known_eigenvalue(self):
        """For f(x) = 0.5 * x^T A x, top eigenvalue = max(eigenvalues(A))."""
        # f(x) = 0.5 * (2*x1^2 + 5*x2^2), Hessian = diag(2, 5), top = 5
        a, b = 2.0, 5.0

        def quadratic_loss_and_grad(params):
            x = params["layer1"]
            loss = 0.5 * (a * x[0] ** 2 + b * x[1] ** 2)
            grad = np.array([a * x[0], b * x[1]], dtype=np.float32)
            return np.array(loss), {"layer1": grad}

        params = {"layer1": np.array([1.0, 1.0], dtype=np.float32)}
        config = Config(power_iterations=50, finite_difference_epsilon=1e-5)
        result = top_eigenvalue(quadratic_loss_and_grad, params, config)

        assert result is not None
        # Top eigenvalue should be max(2, 5) = 5
        assert result == pytest.approx(5.0, rel=0.1)


class TestConditionProxy:
    """Tests for condition_proxy function."""

    def test_zero_parameter_count_returns_none(self):
        """Zero parameter count should return None."""
        result = condition_proxy(top_eigenvalue=10.0, trace_estimate=5.0, parameter_count=0)
        assert result is None

    def test_zero_trace_returns_none(self):
        """Zero trace should return None."""
        result = condition_proxy(top_eigenvalue=10.0, trace_estimate=0.0, parameter_count=100)
        assert result is None

    def test_negative_avg_eigenvalue_returns_none(self):
        """Negative average eigenvalue should return None."""
        result = condition_proxy(top_eigenvalue=10.0, trace_estimate=-5.0, parameter_count=1)
        assert result is None

    def test_known_condition_number(self):
        """Test with known condition number."""
        # top_eigenvalue = 10, trace = 20, param_count = 4
        # avg_eigenvalue = 20/4 = 5
        # condition_proxy = 10/5 = 2
        result = condition_proxy(top_eigenvalue=10.0, trace_estimate=20.0, parameter_count=4)
        assert result == pytest.approx(2.0)

    def test_identity_hessian_condition_one(self):
        """Identity Hessian should have condition number ~1."""
        # For identity: all eigenvalues = 1, trace = n, top = 1
        # avg = n/n = 1, condition = 1/1 = 1
        n = 10
        result = condition_proxy(top_eigenvalue=1.0, trace_estimate=float(n), parameter_count=n)
        assert result == pytest.approx(1.0)


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_flatten_parameters_ordering(self):
        """Flattening should use sorted key order."""
        from modelcypher.core.domain.training.hessian_estimator import _flatten_parameters

        params = {
            "z_layer": np.array([1.0, 2.0]),
            "a_layer": np.array([3.0, 4.0]),
        }
        result = _flatten_parameters(params)

        # Sorted order: a_layer, z_layer
        expected = np.array([3.0, 4.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_flatten_empty_params(self):
        """Empty params should return empty array."""
        from modelcypher.core.domain.training.hessian_estimator import _flatten_parameters

        result = _flatten_parameters({})
        assert result.shape == (0,)

    def test_rademacher_direction_values(self):
        """Rademacher direction should be +1 or -1."""
        from modelcypher.core.domain.training.hessian_estimator import (
            _generate_rademacher_direction,
        )

        params = {"layer1": np.zeros((10, 10))}
        direction = _generate_rademacher_direction(params, seed=42)

        values = direction["layer1"].flatten()
        assert all(v in [-1.0, 1.0] for v in values)

    def test_rademacher_deterministic(self):
        """Same seed should give same direction."""
        from modelcypher.core.domain.training.hessian_estimator import (
            _generate_rademacher_direction,
        )

        params = {"layer1": np.zeros((5, 5))}
        dir1 = _generate_rademacher_direction(params, seed=123)
        dir2 = _generate_rademacher_direction(params, seed=123)

        np.testing.assert_array_equal(dir1["layer1"], dir2["layer1"])

    def test_normalize_direction_unit_norm(self):
        """Normalized direction should have unit norm."""
        from modelcypher.core.domain.training.hessian_estimator import _normalize_direction

        direction = {
            "layer1": np.array([3.0, 4.0]),
            "layer2": np.array([0.0, 0.0, 12.0]),
        }
        result = _normalize_direction(direction)

        # Total norm = sqrt(9 + 16 + 144) = sqrt(169) = 13
        total_norm_sq = sum(np.sum(v**2) for v in result.values())
        assert np.sqrt(total_norm_sq) == pytest.approx(1.0, abs=1e-6)

    def test_normalize_zero_direction(self):
        """Zero direction should return unchanged."""
        from modelcypher.core.domain.training.hessian_estimator import _normalize_direction

        direction = {"layer1": np.array([0.0, 0.0])}
        result = _normalize_direction(direction)

        np.testing.assert_array_equal(result["layer1"], direction["layer1"])
