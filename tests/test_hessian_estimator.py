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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
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


def _arrays_allclose(backend, a, b, atol: float | None = None):
    """Compare two backend arrays for approximate equality using backend ops."""
    backend.eval(a)
    backend.eval(b)
    diff = backend.abs(a - b)
    max_arr = backend.max(diff)
    backend.eval(max_arr)
    max_diff = float(backend.to_scalar(max_arr))
    if atol is None:
        atol = division_epsilon(backend, diff)
    return max_diff <= atol


def _assert_close(value: float, expected: float, backend, ref_array) -> None:
    tol = division_epsilon(backend, ref_array)
    scale = max(1.0, abs(expected))
    assert abs(value - expected) <= tol * scale


def _sum_scalar(backend, array) -> float:
    sum_arr = backend.sum(array)
    backend.eval(sum_arr)
    return float(backend.to_scalar(sum_arr))


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = Config()
        assert config.hutchinson_vectors == 5
        assert config.power_iterations == 20
        assert config.finite_difference_epsilon is None
        assert config.power_iteration_tolerance is None

    def test_moderate_config(self):
        """Moderate config should have reduced iterations."""
        config = Config.moderate()
        assert config.hutchinson_vectors == 3
        assert config.power_iterations == 10
        assert config.finite_difference_epsilon is None

    def test_full_config(self):
        """Full config should have increased precision."""
        config = Config.full()
        assert config.hutchinson_vectors == 10
        assert config.power_iterations == 30
        assert config.finite_difference_epsilon is None

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
        backend = get_default_backend()
        sample = {"layer1": backend.array([1.0, 2.0, 3.0])}
        result = gradient_quality([sample])
        assert result is None

    def test_identical_gradients_zero_variance(self):
        """Identical gradients should have zero variance."""
        backend = get_default_backend()
        grad = {"layer1": backend.array([1.0, 2.0, 3.0])}
        result = gradient_quality([grad, grad, grad])

        assert result is not None
        _assert_close(result.variance, 0.0, backend, grad["layer1"])
        # SNR should be infinite (or very large) with zero variance
        assert result.snr == float("inf")

    def test_orthogonal_gradients_high_variance(self):
        """Orthogonal gradients should have high variance."""
        backend = get_default_backend()
        grad1 = {"layer1": backend.array([1.0, 0.0, 0.0])}
        grad2 = {"layer1": backend.array([0.0, 1.0, 0.0])}
        grad3 = {"layer1": backend.array([0.0, 0.0, 1.0])}
        result = gradient_quality([grad1, grad2, grad3])

        assert result is not None
        assert result.variance > 0
        # Mean grad = [1/3, 1/3, 1/3], norm = sqrt(1/3)
        expected_mean_norm = sqrt_scalar(1.0 / 3.0, backend)
        _assert_close(result.mean_norm, expected_mean_norm, backend, grad1["layer1"])

    def test_known_variance_computation(self):
        """Test variance computation with known values."""
        backend = get_default_backend()
        # Two samples: [1, 0] and [0, 1]
        # Mean = [0.5, 0.5]
        # Sample 1: centered = [0.5, -0.5], squared_diff = 0.5
        # Sample 2: centered = [-0.5, 0.5], squared_diff = 0.5
        # Variance = mean(0.5, 0.5) = 0.5
        grad1 = {"layer1": backend.array([1.0, 0.0])}
        grad2 = {"layer1": backend.array([0.0, 1.0])}
        result = gradient_quality([grad1, grad2])

        assert result is not None
        _assert_close(result.variance, 0.5, backend, grad1["layer1"])

    def test_snr_computation(self):
        """SNR should be mean_norm^2 / variance."""
        backend = get_default_backend()
        grad1 = {"layer1": backend.array([2.0, 0.0])}
        grad2 = {"layer1": backend.array([0.0, 2.0])}
        result = gradient_quality([grad1, grad2])

        assert result is not None
        # Mean = [1, 1], mean_norm = sqrt(2)
        # variance = 2 (each sample has squared_diff = 2)
        # SNR = 2 / 2 = 1
        _assert_close(result.snr, 1.0, backend, grad1["layer1"])


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
        backend = get_default_backend()
        grads = {"layer1": backend.array([3.0, 4.0])}  # norm = 5
        result = per_layer_analysis(grads)

        _assert_close(result.norms["layer1"], 5.0, backend, grads["layer1"])
        _assert_close(result.fractions["layer1"], 1.0, backend, grads["layer1"])
        assert "layer1" in result.active_layers

    def test_multiple_layers_fractions_sum_to_one(self):
        """Layer fractions should approximately sum to 1 (by L1 of norms)."""
        backend = get_default_backend()
        grads = {
            "layer1": backend.array([3.0, 4.0]),  # norm = 5
            "layer2": backend.array([12.0, 0.0]),  # norm = 12
        }
        result = per_layer_analysis(grads)

        # Total norm = sqrt(5^2 + 12^2) = 13
        _assert_close(result.norms["layer1"], 5.0, backend, grads["layer1"])
        _assert_close(result.norms["layer2"], 12.0, backend, grads["layer2"])
        _assert_close(result.fractions["layer1"], 5.0 / 13.0, backend, grads["layer1"])
        _assert_close(result.fractions["layer2"], 12.0 / 13.0, backend, grads["layer2"])

    def test_active_layers_threshold(self):
        """Only layers above threshold should be active."""
        backend = get_default_backend()
        grads = {
            "layer1": backend.array([0.01]),  # small
            "layer2": backend.array([10.0]),  # dominant
        }
        result = per_layer_analysis(grads)

        assert "layer2" in result.active_layers
        assert "layer1" not in result.active_layers


class TestTrajectory:
    """Tests for trajectory function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""
        backend = get_default_backend()
        assert trajectory({}, {"a": backend.array([1.0])}) is None
        assert trajectory({"a": backend.array([1.0])}, {}) is None

    def test_identical_params_zero_divergence(self):
        """Identical params should have zero divergence."""
        backend = get_default_backend()
        params = {"layer1": backend.array([1.0, 2.0, 3.0])}
        result = trajectory(params, params)

        assert result is not None
        _assert_close(result.divergence, 0.0, backend, params["layer1"])
        _assert_close(result.cosine_similarity, 1.0, backend, params["layer1"])

    def test_opposite_params_negative_cosine(self):
        """Opposite params should have cosine = -1."""
        backend = get_default_backend()
        current = {"layer1": backend.array([1.0, 0.0])}
        initial = {"layer1": backend.array([-1.0, 0.0])}
        result = trajectory(current, initial)

        assert result is not None
        _assert_close(result.cosine_similarity, -1.0, backend, current["layer1"])
        _assert_close(result.divergence, 2.0, backend, current["layer1"])

    def test_orthogonal_params_zero_cosine(self):
        """Orthogonal params should have cosine = 0."""
        backend = get_default_backend()
        current = {"layer1": backend.array([1.0, 0.0])}
        initial = {"layer1": backend.array([0.0, 1.0])}
        result = trajectory(current, initial)

        assert result is not None
        _assert_close(result.cosine_similarity, 0.0, backend, current["layer1"])

    def test_divergence_computation(self):
        """Divergence should be L2 distance."""
        backend = get_default_backend()
        current = {"layer1": backend.array([3.0, 4.0])}
        initial = {"layer1": backend.array([0.0, 0.0])}
        result = trajectory(current, initial)

        assert result is not None
        _assert_close(result.divergence, 5.0, backend, current["layer1"])


class TestEffectiveStepRatio:
    """Tests for effective_step_ratio function."""

    def test_empty_inputs_return_none(self):
        """Empty inputs should return None."""
        backend = get_default_backend()
        assert effective_step_ratio({}, {"a": backend.array([1.0])}, 0.1) is None
        assert effective_step_ratio({"a": backend.array([1.0])}, {}, 0.1) is None

    def test_zero_learning_rate_returns_none(self):
        """Zero learning rate should return None."""
        backend = get_default_backend()
        step = {"layer1": backend.array([1.0])}
        grad = {"layer1": backend.array([1.0])}
        assert effective_step_ratio(step, grad, 0.0) is None

    def test_perfect_step_ratio_one(self):
        """When actual = lr * grad, ratio should be 1.0."""
        backend = get_default_backend()
        grad = {"layer1": backend.array([1.0, 2.0])}
        lr = 0.1
        step = {"layer1": lr * grad["layer1"]}
        result = effective_step_ratio(step, grad, lr)

        _assert_close(result, 1.0, backend, grad["layer1"])

    def test_doubled_step_ratio_two(self):
        """When actual = 2 * lr * grad, ratio should be 2.0."""
        backend = get_default_backend()
        grad = {"layer1": backend.array([1.0, 2.0])}
        lr = 0.1
        step = {"layer1": 2.0 * lr * grad["layer1"]}
        result = effective_step_ratio(step, grad, lr)

        _assert_close(result, 2.0, backend, grad["layer1"])


class TestHutchinsonTraceEstimate:
    """Tests for hutchinson_trace_estimate function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""
        backend = get_default_backend()

        def dummy_fn(params):
            return backend.array(0.0), {}

        result = hutchinson_trace_estimate(dummy_fn, {}, Config())
        assert result is None

    def test_zero_vectors_returns_none(self):
        """Zero hutchinson vectors should return None."""
        backend = get_default_backend()

        def dummy_fn(params):
            return backend.array(0.0), params

        params = {"layer1": backend.array([1.0])}
        config = Config(hutchinson_vectors=0)
        result = hutchinson_trace_estimate(dummy_fn, params, config)
        assert result is None

    def test_quadratic_function_known_trace(self):
        """For f(x) = 0.5 * x^T A x, trace(H) = trace(A)."""
        backend = get_default_backend()
        # Simple 2D quadratic: f(x) = 0.5 * (a*x1^2 + b*x2^2)
        # Hessian = diag(a, b), trace = a + b
        a, b = 2.0, 3.0

        def quadratic_loss_and_grad(params):
            x = params["layer1"]
            loss = 0.5 * (a * x[0] ** 2 + b * x[1] ** 2)
            grad = backend.array([a * x[0], b * x[1]], dtype="float32")
            return backend.array(loss), {"layer1": grad}

        params = {"layer1": backend.array([1.0, 1.0], dtype="float32")}
        config = Config(hutchinson_vectors=1)
        result = hutchinson_trace_estimate(quadratic_loss_and_grad, params, config)

        assert result is not None
        # Trace should be a + b = 5.0
        _assert_close(result, 5.0, backend, params["layer1"])


class TestTopEigenvalue:
    """Tests for top_eigenvalue function."""

    def test_empty_params_returns_none(self):
        """Empty params should return None."""
        backend = get_default_backend()

        def dummy_fn(params):
            return backend.array(0.0), {}

        result = top_eigenvalue(dummy_fn, {}, Config())
        assert result is None

    def test_zero_iterations_returns_none(self):
        """Zero power iterations should return None."""
        backend = get_default_backend()

        def dummy_fn(params):
            return backend.array(0.0), params

        params = {"layer1": backend.array([1.0])}
        config = Config(power_iterations=0)
        result = top_eigenvalue(dummy_fn, params, config)
        assert result is None

    def test_quadratic_function_known_eigenvalue(self):
        """For f(x) = 0.5 * x^T A x, top eigenvalue = max(eigenvalues(A))."""
        backend = get_default_backend()
        # f(x) = 0.5 * a * x^2, Hessian = [a], top = a
        a = 5.0

        def quadratic_loss_and_grad(params):
            x = params["layer1"]
            loss = 0.5 * (a * x[0] ** 2)
            grad = backend.array([a * x[0]], dtype="float32")
            return backend.array(loss), {"layer1": grad}

        params = {"layer1": backend.array([1.0], dtype="float32")}
        config = Config(power_iterations=1)
        result = top_eigenvalue(quadratic_loss_and_grad, params, config)

        assert result is not None
        # Top eigenvalue should be a = 5
        _assert_close(result, 5.0, backend, params["layer1"])


class TestHessianVectorProduct:
    """Tests for exact/fallback HVP computation path."""

    def test_prefers_exact_autodiff_when_available(self, monkeypatch):
        """Exact VJP path should be used before finite-difference fallback."""
        backend = get_default_backend()
        import modelcypher.core.domain.training.hessian_estimator as hessian_mod

        def fail_finite_difference(*args, **kwargs):
            del args, kwargs
            pytest.fail("Finite-difference fallback should not run when exact HVP is available")

        monkeypatch.setattr(
            hessian_mod,
            "_finite_difference_hessian_vector_product",
            fail_finite_difference,
        )

        a = 7.0

        def quadratic_loss_and_grad(params):
            x = params["layer1"]
            loss = 0.5 * a * x[0] * x[0]
            grad = {"layer1": backend.array([a * x[0]], dtype="float32")}
            return backend.array(loss), grad

        params = {"layer1": backend.array([2.0], dtype="float32")}
        direction = {"layer1": backend.array([3.0], dtype="float32")}

        hvp = hessian_mod._hessian_vector_product(
            loss_and_grad_function=quadratic_loss_and_grad,
            current_params=params,
            direction=direction,
            config=Config.moderate(),
            backend=backend,
        )

        assert hvp is not None
        expected = backend.array([a * 3.0], dtype="float32")
        assert _arrays_allclose(backend, hvp["layer1"], expected)

    def test_falls_back_to_finite_difference_when_transforms_unavailable(self, monkeypatch):
        """Transform unavailability should trigger finite-difference fallback."""
        backend = get_default_backend()
        import modelcypher.core.domain.training.hessian_estimator as hessian_mod

        sentinel = {"layer1": backend.array([13.0], dtype="float32")}
        fallback_calls = {"count": 0}

        def finite_difference_stub(*args, **kwargs):
            del args, kwargs
            fallback_calls["count"] += 1
            return sentinel

        def unavailable_vjp(*args, **kwargs):
            del args, kwargs
            raise NotImplementedError("vjp unavailable")

        def unavailable_jvp(*args, **kwargs):
            del args, kwargs
            raise NotImplementedError("jvp unavailable")

        monkeypatch.setattr(backend, "vjp", unavailable_vjp)
        monkeypatch.setattr(backend, "jvp", unavailable_jvp)
        monkeypatch.setattr(
            hessian_mod,
            "_finite_difference_hessian_vector_product",
            finite_difference_stub,
        )

        def dummy_loss_and_grad(params):
            x = params["layer1"]
            return backend.array(0.0), {"layer1": x}

        params = {"layer1": backend.array([1.0], dtype="float32")}
        direction = {"layer1": backend.array([1.0], dtype="float32")}
        result = hessian_mod._hessian_vector_product(
            loss_and_grad_function=dummy_loss_and_grad,
            current_params=params,
            direction=direction,
            config=Config.moderate(),
            backend=backend,
        )

        assert fallback_calls["count"] == 1
        assert result is sentinel


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
        backend = get_default_backend()
        _assert_close(result, 2.0, backend, backend.array([2.0]))

    def test_identity_hessian_condition_one(self):
        """Identity Hessian should have condition number ~1."""
        # For identity: all eigenvalues = 1, trace = n, top = 1
        # avg = n/n = 1, condition = 1/1 = 1
        n = 10
        result = condition_proxy(top_eigenvalue=1.0, trace_estimate=float(n), parameter_count=n)
        backend = get_default_backend()
        _assert_close(result, 1.0, backend, backend.array([1.0]))


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_flatten_parameters_ordering(self):
        """Flattening should use sorted key order."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import _flatten_parameters

        params = {
            "z_layer": backend.array([1.0, 2.0]),
            "a_layer": backend.array([3.0, 4.0]),
        }
        result = _flatten_parameters(params, backend)

        # Sorted order: a_layer, z_layer
        expected = backend.array([3.0, 4.0, 1.0, 2.0], dtype="float32")
        assert _arrays_allclose(backend, result, expected)

    def test_flatten_empty_params(self):
        """Empty params should return empty array."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import _flatten_parameters

        result = _flatten_parameters({}, backend)
        assert result.shape == (0,)

    def test_rademacher_direction_values(self):
        """Rademacher direction should be +1 or -1."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import (
            _generate_rademacher_direction,
        )

        params = {"layer1": backend.zeros((10, 10))}
        direction = _generate_rademacher_direction(params, backend, seed=42)

        flat = backend.tolist(backend.reshape(direction["layer1"], (-1,)))
        assert all(v in (-1.0, 1.0) for v in flat)

    def test_rademacher_deterministic(self):
        """Same seed should give same direction."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import (
            _generate_rademacher_direction,
        )

        params = {"layer1": backend.zeros((5, 5))}
        dir1 = _generate_rademacher_direction(params, backend, seed=123)
        dir2 = _generate_rademacher_direction(params, backend, seed=123)

        assert _arrays_allclose(backend, dir1["layer1"], dir2["layer1"])

    def test_normalize_direction_unit_norm(self):
        """Normalized direction should have unit norm."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import _normalize_direction

        direction = {
            "layer1": backend.array([3.0, 4.0]),
            "layer2": backend.array([0.0, 0.0, 12.0]),
        }
        result = _normalize_direction(direction, backend)

        # Total norm = sqrt(9 + 16 + 144) = sqrt(169) = 13
        total_norm_sq = 0.0
        for value in result.values():
            total_norm_sq += _sum_scalar(backend, value**2)
        total_norm = sqrt_scalar(total_norm_sq, backend)
        _assert_close(total_norm, 1.0, backend, backend.array([1.0]))

    def test_normalize_zero_direction(self):
        """Zero direction should return unchanged."""
        backend = get_default_backend()
        from modelcypher.core.domain.training.hessian_estimator import _normalize_direction

        direction = {"layer1": backend.array([0.0, 0.0])}
        result = _normalize_direction(direction, backend)

        assert _arrays_allclose(backend, result["layer1"], direction["layer1"])
