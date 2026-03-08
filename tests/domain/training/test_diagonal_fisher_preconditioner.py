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

"""Tests for diagonal Fisher preconditioner.

Validates the mathematical properties of the EMA-based diagonal Fisher
estimator used for curvature-aware Cayley-Stiefel training:

1. Convergence: constant gradient → v converges to g²
2. Bias correction: early steps are properly compensated
3. Preconditioning: constant gradient produces unit-scale components
4. Descent preservation: preconditioned gradient has positive inner product with raw
5. Numerical stability: near-zero and large gradients handled safely
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.diagonal_fisher_preconditioner import (
    DiagonalFisherState,
    _DEFAULT_BETA2,
    _SQRT_EPS_F32,
    init_fisher_state,
    precondition_gradient,
    update_fisher_state,
)


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


class TestInitFisherState:
    def test_initializes_zeros(self, backend):
        """v is initialized to zeros matching parameter shapes."""
        b = backend
        params = {
            "a": b.ones((3, 4)),
            "b": b.ones((5,)),
        }
        state = init_fisher_state(params, b)

        assert set(state.v.keys()) == {"a", "b"}
        assert state.step_count == 0
        assert state.beta2 == _DEFAULT_BETA2

        # All zeros
        v_a_sum = float(b.to_scalar(b.sum(b.abs(state.v["a"]))))
        v_b_sum = float(b.to_scalar(b.sum(b.abs(state.v["b"]))))
        assert v_a_sum == 0.0
        assert v_b_sum == 0.0

    def test_shapes_match_params(self, backend):
        """v arrays have same shapes as parameter arrays."""
        b = backend
        params = {"w": b.ones((8, 16))}
        state = init_fisher_state(params, b)

        assert tuple(b.shape(state.v["w"])) == (8, 16)

    def test_custom_beta2(self, backend):
        """Custom beta2 is stored correctly."""
        b = backend
        params = {"w": b.ones((2, 2))}
        state = init_fisher_state(params, b, beta2=0.99)
        assert state.beta2 == 0.99


class TestUpdateFisherState:
    def test_constant_gradient_converges_to_g_squared(self, backend):
        """After many steps with constant g, v → g² (up to bias)."""
        b = backend
        g_val = 2.0
        params = {"w": b.ones((4,)) * g_val}
        # Use β₂=0.99 for faster convergence in test (half-life ~69 steps).
        # At 500 steps: v = g² × (1 - 0.99^500) ≈ g² × 0.9934
        state = init_fisher_state(params, b, beta2=0.99)

        # Apply constant gradient for many steps
        grad = {"w": b.ones((4,)) * g_val}
        for _ in range(500):
            state = update_fisher_state(state, grad, b)

        # v should converge to g² = 4.0
        v_mean = float(b.to_scalar(b.mean(state.v["w"])))
        assert v_mean == pytest.approx(g_val ** 2, rel=0.01)

    def test_step_count_increments(self, backend):
        """step_count increases by 1 per update."""
        b = backend
        params = {"w": b.ones((2,))}
        state = init_fisher_state(params, b)

        grad = {"w": b.ones((2,))}
        for i in range(5):
            state = update_fisher_state(state, grad, b)
            assert state.step_count == i + 1

    def test_ema_formula(self, backend):
        """Single step: v = (1-β₂) × g²."""
        b = backend
        params = {"w": b.ones((2,))}
        state = init_fisher_state(params, b, beta2=0.9)

        g = b.ones((2,)) * 3.0
        state = update_fisher_state(state, {"w": g}, b)

        # v = 0.1 × 9.0 = 0.9
        v_val = float(b.to_scalar(state.v["w"][0]))
        assert v_val == pytest.approx(0.9, rel=1e-5)

    def test_two_steps(self, backend):
        """Two steps: v₂ = β₂ × v₁ + (1-β₂) × g₂²."""
        b = backend
        params = {"w": b.ones((1,))}
        state = init_fisher_state(params, b, beta2=0.9)

        # Step 1: g=2 → v₁ = 0.1 × 4 = 0.4
        state = update_fisher_state(state, {"w": b.ones((1,)) * 2.0}, b)
        v1 = float(b.to_scalar(state.v["w"][0]))
        assert v1 == pytest.approx(0.4, rel=1e-5)

        # Step 2: g=3 → v₂ = 0.9 × 0.4 + 0.1 × 9 = 0.36 + 0.9 = 1.26
        state = update_fisher_state(state, {"w": b.ones((1,)) * 3.0}, b)
        v2 = float(b.to_scalar(state.v["w"][0]))
        assert v2 == pytest.approx(1.26, rel=1e-5)


class TestPreconditionGradient:
    def test_constant_gradient_unit_scale(self, backend):
        """After convergence with constant g, preconditioned d ≈ sign(g)."""
        b = backend
        g_val = 5.0
        params = {"w": b.ones((4,)) * g_val}
        # Use β₂=0.99 for faster convergence in test
        state = init_fisher_state(params, b, beta2=0.99)

        grad = {"w": b.ones((4,)) * g_val}
        # Run until v converges (500 steps at β₂=0.99 → >99% convergence)
        for _ in range(500):
            state = update_fisher_state(state, grad, b)

        precond = precondition_gradient(grad, state, b)

        # d = g / (√v̂ + ε) where v̂ ≈ g² after convergence
        # d ≈ g / (|g| + ε) ≈ sign(g)
        d_val = float(b.to_scalar(precond["w"][0]))
        assert d_val == pytest.approx(1.0, abs=0.05)

    def test_bias_correction_step_1(self, backend):
        """At step 1, bias correction amplifies v̂ by 1/(1-β₂)."""
        b = backend
        beta2 = 0.9
        params = {"w": b.ones((1,))}
        state = init_fisher_state(params, b, beta2=beta2)

        g = b.ones((1,)) * 2.0
        state = update_fisher_state(state, {"w": g}, b)
        # v = (1-0.9) × 4 = 0.4
        # v̂ = 0.4 / (1 - 0.9^1) = 0.4 / 0.1 = 4.0
        # d = 2.0 / (√4.0 + ε) = 2.0 / (2.0 + ε) ≈ 1.0

        precond = precondition_gradient({"w": g}, state, b)
        d_val = float(b.to_scalar(precond["w"][0]))
        assert d_val == pytest.approx(1.0, abs=0.01)

    def test_bias_correction_step_1000(self, backend):
        """At step 1000, bias correction ≈ 1 (negligible)."""
        b = backend
        params = {"w": b.ones((1,))}
        state = init_fisher_state(params, b, beta2=_DEFAULT_BETA2)

        g = b.ones((1,)) * 2.0
        # Simulate 1000 constant-gradient steps
        for _ in range(1000):
            state = update_fisher_state(state, {"w": g}, b)

        # Bias correction: 1/(1-0.999^1000) ≈ 1.58 (small correction)
        # v ≈ 4.0, v̂ ≈ 4.0 × 1.58 ≈ 6.3 — but v has converged so v ≈ 4.0
        # Wait, v converges to 4.0, v̂ = 4.0 / (1 - 0.999^1000)
        # At step 1000: 0.999^1000 ≈ 0.368, so 1/(1-0.368) = 1.58
        # v̂ = 4.0 × 1.58 = 6.32 — this is overcorrection at step 1000
        # Actually v has been accumulating for 1000 steps so it's close to 4.0
        # v̂ = v / (1-β₂^t) — for large t this is ≈ v since β₂^t → 0
        # 0.999^1000 = exp(1000 × ln(0.999)) ≈ exp(-1.0005) ≈ 0.3676
        # So correction = 1/(1-0.368) = 1.582 — still noticeable at 1000
        # At step 5000: 0.999^5000 ≈ exp(-5) ≈ 0.0067, correction ≈ 1.007

        precond = precondition_gradient({"w": g}, state, b)
        d_val = float(b.to_scalar(precond["w"][0]))
        # At convergence: d ≈ g / |g| = 1.0, but bias correction inflates √v̂
        # slightly, so d slightly less than 1.0
        assert 0.5 < d_val < 1.5

    def test_descent_direction_preserved(self, backend):
        """Preconditioned gradient has positive inner product with raw gradient."""
        b = backend
        params = {"w": b.random_normal((10,))}
        b.eval(params["w"])
        state = init_fisher_state(params, b)

        # Random gradients for several steps
        for _ in range(50):
            g = b.random_normal((10,))
            b.eval(g)
            state = update_fisher_state(state, {"w": g}, b)

        # Final gradient
        g_final = b.random_normal((10,))
        b.eval(g_final)
        state = update_fisher_state(state, {"w": g_final}, b)

        precond = precondition_gradient({"w": g_final}, state, b)

        # Inner product should be positive (same direction)
        inner = float(b.to_scalar(b.sum(g_final * precond["w"])))
        assert inner > 0.0, f"Descent direction lost: inner product = {inner}"

    def test_numerical_stability_near_zero_gradient(self, backend):
        """Near-zero gradients don't cause NaN or Inf."""
        b = backend
        params = {"w": b.ones((4,))}
        state = init_fisher_state(params, b)

        # Non-zero gradient to build up v
        for _ in range(10):
            state = update_fisher_state(state, {"w": b.ones((4,))}, b)

        # Now apply near-zero gradient
        tiny_g = b.ones((4,)) * 1e-20
        state = update_fisher_state(state, {"w": tiny_g}, b)
        precond = precondition_gradient({"w": tiny_g}, state, b)

        d_vals = [float(b.to_scalar(precond["w"][i])) for i in range(4)]
        for d in d_vals:
            assert math.isfinite(d), f"Non-finite value: {d}"

    def test_numerical_stability_large_gradient(self, backend):
        """Large gradients don't cause overflow."""
        b = backend
        params = {"w": b.ones((4,))}
        state = init_fisher_state(params, b)

        # Large gradient
        big_g = b.ones((4,)) * 1e6
        for _ in range(10):
            state = update_fisher_state(state, {"w": big_g}, b)

        precond = precondition_gradient({"w": big_g}, state, b)

        d_vals = [float(b.to_scalar(precond["w"][i])) for i in range(4)]
        for d in d_vals:
            assert math.isfinite(d), f"Non-finite value: {d}"
            # Should be approximately 1.0 (g / |g| after convergence)
            assert abs(d) < 100.0, f"Unexpectedly large: {d}"

    def test_missing_key_passthrough(self, backend):
        """Gradient key not in Fisher state passes through unchanged."""
        b = backend
        params = {"w": b.ones((2,))}
        state = init_fisher_state(params, b)
        state = update_fisher_state(state, {"w": b.ones((2,))}, b)

        # "new_param" not in state.v
        grad = {"w": b.ones((2,)) * 3.0, "new_param": b.ones((2,)) * 5.0}
        precond = precondition_gradient(grad, state, b)

        # "new_param" should be unchanged (passthrough)
        new_val = float(b.to_scalar(precond["new_param"][0]))
        assert new_val == pytest.approx(5.0)


class TestMultiParameter:
    def test_multiple_parameters_independent(self, backend):
        """Each parameter's v evolves independently."""
        b = backend
        params = {
            "a": b.ones((2,)),
            "b": b.ones((2,)),
        }
        state = init_fisher_state(params, b, beta2=0.9)

        # Different gradients for each param
        grad = {
            "a": b.ones((2,)) * 2.0,  # g²=4
            "b": b.ones((2,)) * 10.0,  # g²=100
        }
        state = update_fisher_state(state, grad, b)

        # v_a = 0.1 × 4 = 0.4
        # v_b = 0.1 × 100 = 10.0
        v_a = float(b.to_scalar(state.v["a"][0]))
        v_b = float(b.to_scalar(state.v["b"][0]))
        assert v_a == pytest.approx(0.4, rel=1e-5)
        assert v_b == pytest.approx(10.0, rel=1e-5)
