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

"""Tests for numerical_stability.py.

Tests cover:
- Epsilon/threshold functions (machine_epsilon, division_epsilon, etc.)
- SVD via eigendecomposition (geodesic_svd)
- GPU-accelerated least squares (gpu_lstsq)
- Invariant alignment
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings as hypothesis_settings, strategies as st

from modelcypher.core.domain.geometry.numerical_stability import (
    condition_threshold,
    division_epsilon,
    geodesic_svd,
    gpu_lstsq,
    invariant_alignment,
    is_finite,
    is_inf,
    is_nan,
    log_scalar,
    machine_epsilon,
    regularization_epsilon,
    safe_log_epsilon,
    sqrt_scalar,
    svd_rank_threshold,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps(backend: "Backend", *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _div_eps(backend: "Backend", *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# Epsilon and Threshold Functions
# =============================================================================


class TestMachineEpsilon:
    """Tests for machine_epsilon function."""

    def test_float32_epsilon(self, any_backend: "Backend") -> None:
        """Float32 epsilon should reflect dtype precision."""
        b = any_backend
        arr = b.zeros((2, 2))
        eps = machine_epsilon(b, arr)
        one = b.array([1.0])
        half_eps_arr = b.array([eps / 2.0])
        eps_arr = b.array([eps])
        b.eval(one, half_eps_arr, eps_arr)
        assert float(b.to_scalar(one + half_eps_arr)) == 1.0
        assert float(b.to_scalar(one + eps_arr)) != 1.0

    # NOTE: float64 test removed - MLX doesn't support float64

    def test_epsilon_positive(self, any_backend: "Backend") -> None:
        """Machine epsilon should always be positive."""
        b = any_backend
        arr = b.zeros((3, 3))
        eps = machine_epsilon(b, arr)
        assert eps > 0


class TestDivisionEpsilon:
    """Tests for division_epsilon function."""

    def test_division_epsilon_larger_than_machine(self, any_backend: "Backend") -> None:
        """Division epsilon should be larger than machine epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        assert div_eps > mach_eps

    def test_division_epsilon_is_sqrt_machine(self, any_backend: "Backend") -> None:
        """Division epsilon should be sqrt of machine epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        expected = sqrt_scalar(mach_eps, b)
        eps = _eps(b, div_eps, expected)
        assert abs(div_eps - expected) <= eps


class TestRegularizationEpsilon:
    """Tests for regularization_epsilon function."""

    def test_regularization_epsilon_between_div_and_machine(self, any_backend: "Backend") -> None:
        """Regularization epsilon should be between machine and division epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        reg_eps = regularization_epsilon(b, arr)
        assert mach_eps < reg_eps
        assert reg_eps == div_eps

    def test_regularization_epsilon_positive(self, any_backend: "Backend") -> None:
        """Regularization epsilon should be positive."""
        b = any_backend
        arr = b.zeros((5, 5))
        reg_eps = regularization_epsilon(b, arr)
        assert reg_eps > 0


class TestConditionThreshold:
    """Tests for condition_threshold function."""

    def test_condition_threshold_is_inverse_epsilon(self, any_backend: "Backend") -> None:
        """Condition threshold should be 1/epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        mach_eps = machine_epsilon(b, arr)
        cond_thresh = condition_threshold(b, arr)
        expected = 1.0 / mach_eps
        eps = _eps(b, cond_thresh, expected)
        assert abs(cond_thresh - expected) <= eps

    def test_condition_threshold_very_large(self, any_backend: "Backend") -> None:
        """Condition threshold should match dtype-derived inverse epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        cond_thresh = condition_threshold(b, arr)
        expected = 1.0 / machine_epsilon(b, arr)
        eps = _eps(b, cond_thresh, expected)
        assert abs(cond_thresh - expected) <= eps


class TestSvdRankThreshold:
    """Tests for svd_rank_threshold function."""

    def test_svd_rank_threshold_scales_with_dimension(self, any_backend: "Backend") -> None:
        """SVD rank threshold should scale linearly with max_dim."""
        b = any_backend
        arr = b.zeros((2, 2))
        thresh_10 = svd_rank_threshold(b, arr, max_dim=10)
        thresh_100 = svd_rank_threshold(b, arr, max_dim=100)
        eps = _eps(b, thresh_10, thresh_100)
        assert abs(thresh_100 / thresh_10 - 10.0) <= eps

    def test_svd_rank_threshold_formula(self, any_backend: "Backend") -> None:
        """SVD rank threshold should equal max_dim * machine_epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        mach_eps = machine_epsilon(b, arr)
        thresh = svd_rank_threshold(b, arr, max_dim=50)
        expected = 50 * mach_eps
        eps = _eps(b, thresh, expected)
        assert abs(thresh - expected) <= eps


class TestTinyValue:
    """Tests for tiny_value function."""

    def test_tiny_value_very_small(self, any_backend: "Backend") -> None:
        """Tiny value should be extremely small but positive."""
        b = any_backend
        arr = b.zeros((2, 2))
        tiny = tiny_value(b, arr)
        assert tiny > 0
        expected = b.finfo(arr.dtype).tiny
        assert tiny == expected

    def test_tiny_value_smaller_than_epsilon(self, any_backend: "Backend") -> None:
        """Tiny value should be smaller than machine epsilon."""
        b = any_backend
        arr = b.zeros((2, 2))
        tiny = tiny_value(b, arr)
        mach_eps = machine_epsilon(b, arr)
        assert tiny < mach_eps


class TestSafeLogEpsilon:
    """Tests for safe_log_epsilon function."""

    def test_safe_log_epsilon_equals_tiny(self, any_backend: "Backend") -> None:
        """Safe log epsilon should equal tiny value."""
        b = any_backend
        arr = b.zeros((2, 2))
        log_eps = safe_log_epsilon(b, arr)
        tiny = tiny_value(b, arr)
        eps = _eps(b, log_eps, tiny)
        assert abs(log_eps - tiny) <= eps

    def test_safe_log_epsilon_prevents_log_zero(self, any_backend: "Backend") -> None:
        """Adding safe log epsilon should prevent log(0)."""
        b = any_backend
        arr = b.zeros((2, 2))
        log_eps = safe_log_epsilon(b, arr)
        result = log_scalar(log_eps, b)
        assert is_finite(result, b)


# =============================================================================
# Matrix Decomposition Tests
# =============================================================================


class TestGeodesicSvd:
    """Tests for geodesic_svd function."""

    def test_svd_basic_shapes(self, any_backend: "Backend") -> None:
        """SVD should return correct shapes."""
        b = any_backend
        A = b.random_normal((10, 5))
        b.eval(A)

        U, S, Vt = geodesic_svd(b, A)
        b.eval(U, S, Vt)

        assert U.shape == (10, 5)
        assert S.shape == (5,)
        assert Vt.shape == (5, 5)

    def test_svd_reconstruction(self, any_backend: "Backend") -> None:
        """SVD should allow matrix reconstruction."""
        b = any_backend
        b.random_seed(42)
        A = b.random_normal((8, 6))
        b.eval(A)

        U, S, Vt = geodesic_svd(b, A)
        b.eval(U, S, Vt)

        # Reconstruct: A ≈ U @ diag(S) @ Vt
        S_diag = b.diag(S)
        reconstructed = b.matmul(b.matmul(U, S_diag), Vt)
        b.eval(reconstructed)

        diff = b.abs(A - reconstructed)
        max_diff_arr = b.max(diff)
        b.eval(max_diff_arr)
        max_diff = float(b.to_scalar(max_diff_arr))

        eps = _div_eps(b)
        assert max_diff < eps

    def test_svd_singular_values_nonnegative(self, any_backend: "Backend") -> None:
        """Singular values should be non-negative."""
        b = any_backend
        b.random_seed(42)
        A = b.random_normal((10, 8))
        b.eval(A)

        _, S, _ = geodesic_svd(b, A)
        b.eval(S)

        min_s_arr = b.min(S)
        b.eval(min_s_arr)
        min_s = float(b.to_scalar(min_s_arr))

        assert min_s >= 0

    def test_svd_singular_values_descending(self, any_backend: "Backend") -> None:
        """Singular values should be in descending order."""
        b = any_backend
        b.random_seed(42)
        A = b.random_normal((10, 8))
        b.eval(A)

        _, S, _ = geodesic_svd(b, A)
        b.eval(S)

        S_list = b.tolist(S)
        for i in range(len(S_list) - 1):
            assert S_list[i] >= S_list[i + 1]


# =============================================================================
# GPU Least Squares Tests
# =============================================================================


class TestGpuLstsq:
    """Tests for gpu_lstsq function."""

    def test_lstsq_overdetermined(self, any_backend: "Backend") -> None:
        """GPU lstsq should solve overdetermined system."""
        b = any_backend
        b.random_seed(42)

        # Create consistent system: A @ x = b where x is known
        A = b.random_normal((20, 10))
        x_true = b.random_normal((10, 5))
        B = b.matmul(A, x_true)
        b.eval(A, x_true, B)

        X = gpu_lstsq(b, A, B)
        b.eval(X)

        # Should recover x_true
        diff = b.abs(X - x_true)
        max_diff_arr = b.max(diff)
        b.eval(max_diff_arr)
        max_diff = float(b.to_scalar(max_diff_arr))

        eps = _div_eps(b)
        assert max_diff < eps

    def test_lstsq_residual(self, any_backend: "Backend") -> None:
        """GPU lstsq should minimize residual."""
        b = any_backend
        b.random_seed(42)

        A = b.random_normal((15, 8))
        B = b.random_normal((15, 3))
        b.eval(A, B)

        X = gpu_lstsq(b, A, B)
        b.eval(X)

        # Compute residual
        residual = b.matmul(A, X) - B
        residual_norm = b.norm(residual)
        b.eval(residual_norm)

        # Residual should be finite
        assert is_finite(float(b.to_scalar(residual_norm)), b)


# =============================================================================
# Invariant Alignment Tests
# =============================================================================


class TestInvariantAlignment:
    """Tests for invariant_alignment function."""

    def test_alignment_basic(self, any_backend: "Backend") -> None:
        """Invariant alignment should produce valid transform."""
        b = any_backend
        b.random_seed(42)

        source = b.random_normal((50, 32))
        target = b.random_normal((50, 24))
        b.eval(source, target)

        F = invariant_alignment(b, source, target)
        b.eval(F)

        # F should be [d_source, d_target]
        assert F.shape == (32, 24)

        # F should not contain NaN
        nan_count = b.sum(b.astype(b.isnan(F), "int32"))
        b.eval(nan_count)
        assert int(b.to_scalar(nan_count)) == 0

    def test_alignment_gram_preservation(self, any_backend: "Backend") -> None:
        """Aligned source should preserve Gram structure (normal equations residual)."""
        b = any_backend
        b.random_seed(42)

        source = b.random_normal((30, 20))
        target = b.random_normal((30, 15))
        b.eval(source, target)

        # invariant_alignment centers the data before solving, so we must too
        source_mean = b.mean(source, axis=0, keepdims=True)
        target_mean = b.mean(target, axis=0, keepdims=True)
        source_c = source - source_mean
        target_c = target - target_mean
        b.eval(source_c, target_c)

        F = invariant_alignment(b, source, target)
        aligned = b.matmul(source_c, F)
        b.eval(F, aligned)

        # Normal equations: source_c.T @ (target_c - source_c @ F) should be ~0
        residual = target_c - aligned
        ortho = b.matmul(b.transpose(source_c), residual)
        b.eval(ortho)

        max_abs = b.max(b.abs(ortho))
        b.eval(max_abs)
        scale = b.max(b.abs(b.matmul(b.transpose(source_c), target_c)))
        b.eval(scale)

        # Compute condition number of centered source for error bound
        _, S, _ = geodesic_svd(b, source_c)
        b.eval(S)
        s_max = float(b.to_scalar(b.max(S)))
        s_min = float(b.to_scalar(b.min(S)))
        eps = _eps(b)
        denom = s_min if s_min > eps else eps
        cond = s_max / denom

        # Error in normal equations scales with cond(source) * sqrt(eps) * scale
        eps = _div_eps(b)
        tol = cond * eps * max(1.0, float(b.to_scalar(scale)))
        assert float(b.to_scalar(max_abs)) <= tol


# =============================================================================
# Hypothesis Property Tests
# =============================================================================


class TestNumericalStabilityHypothesis:
    """Hypothesis property-based tests for numerical stability functions."""

    @given(
        rows=st.integers(min_value=2, max_value=30),
        cols=st.integers(min_value=2, max_value=30),
    )
    @hypothesis_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_epsilon_functions_always_positive(
        self, any_backend: "Backend", rows: int, cols: int
    ) -> None:
        """All epsilon functions should return positive values for any array shape."""
        b = any_backend
        arr = b.zeros((rows, cols))

        mach_eps = machine_epsilon(b, arr)
        div_eps = division_epsilon(b, arr)
        reg_eps = regularization_epsilon(b, arr)
        cond_thresh = condition_threshold(b, arr)
        tiny = tiny_value(b, arr)
        log_eps = safe_log_epsilon(b, arr)

        assert mach_eps > 0, "machine_epsilon must be positive"
        assert div_eps > 0, "division_epsilon must be positive"
        assert reg_eps > 0, "regularization_epsilon must be positive"
        assert cond_thresh > 0, "condition_threshold must be positive"
        assert tiny > 0, "tiny_value must be positive"
        assert log_eps > 0, "safe_log_epsilon must be positive"

    @given(
        max_dim=st.integers(min_value=1, max_value=1000),
    )
    @hypothesis_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_svd_rank_threshold_scales_linearly(
        self, any_backend: "Backend", max_dim: int
    ) -> None:
        """SVD rank threshold should scale linearly with max_dim."""
        b = any_backend
        arr = b.zeros((2, 2))

        thresh = svd_rank_threshold(b, arr, max_dim=max_dim)
        mach_eps = machine_epsilon(b, arr)

        expected = float(max_dim) * mach_eps
        eps = _eps(b, thresh, expected)
        assert abs(thresh - expected) <= eps, f"Expected {expected}, got {thresh}"

    @given(
        rows=st.integers(min_value=3, max_value=20),
        cols=st.integers(min_value=3, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_geodesic_svd_no_nan_inf(
        self, any_backend: "Backend", rows: int, cols: int, seed: int
    ) -> None:
        """SVD should never produce NaN or Inf for random matrices."""
        b = any_backend
        b.random_seed(seed)
        A = b.random_normal((rows, cols))
        b.eval(A)

        U, S, Vt = geodesic_svd(b, A)
        b.eval(U, S, Vt)

        nan_s = b.sum(b.astype(b.isnan(S), "int32"))
        inf_s = b.sum(b.astype(b.isinf(S), "int32"))
        b.eval(nan_s, inf_s)
        assert int(b.to_scalar(nan_s)) == 0, "S contains NaN"
        assert int(b.to_scalar(inf_s)) == 0, "S contains Inf"

        if b.shape(U)[0] > 0 and b.shape(U)[1] > 0:
            nan_u = b.sum(b.astype(b.isnan(U), "int32"))
            b.eval(nan_u)
            assert int(b.to_scalar(nan_u)) == 0, "U contains NaN"

    @given(
        rows=st.integers(min_value=5, max_value=20),
        cols=st.integers(min_value=3, max_value=15),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @hypothesis_settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_svd_singular_values_nonnegative_hypothesis(
        self, any_backend: "Backend", rows: int, cols: int, seed: int
    ) -> None:
        """SVD singular values should always be non-negative."""
        b = any_backend
        b.random_seed(seed)
        A = b.random_normal((rows, cols))
        b.eval(A)

        _, S, _ = geodesic_svd(b, A)
        b.eval(S)

        S_np = b.tolist(S)
        eps = _eps(b, float(min(S_np, default=0.0)))
        assert all(s >= -eps for s in S_np), "Singular values must be non-negative"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCaseEpsilons:
    """Tests for edge case numerical behavior."""

    def test_very_small_array_values(self, any_backend: "Backend") -> None:
        """Arrays with very small values should not cause underflow issues."""
        b = any_backend
        small = b.array([[1e-30, 1e-35], [1e-35, 1e-30]])
        b.eval(small)

        eps = machine_epsilon(b, small)
        assert eps > 0
        assert not is_nan(eps, b)

    def test_very_large_array_values(self, any_backend: "Backend") -> None:
        """Arrays with very large values should not cause overflow issues."""
        b = any_backend
        large = b.array([[1e30, 1e35], [1e35, 1e30]])
        b.eval(large)

        eps = machine_epsilon(b, large)
        assert eps > 0
        assert not is_nan(eps, b)

    def test_mixed_scale_array(self, any_backend: "Backend") -> None:
        """Arrays with mixed scales should be handled properly."""
        b = any_backend
        b.random_seed(42)

        source = b.random_normal((20, 10))
        scales = b.array([10.0 ** (i - 5) for i in range(10)])
        scales = b.reshape(scales, (1, -1))
        source = source * scales
        b.eval(source, scales)

        U, S, Vt = geodesic_svd(b, source)
        b.eval(U, S, Vt)

        nan_s = b.sum(b.astype(b.isnan(S), "int32"))
        inf_s = b.sum(b.astype(b.isinf(S), "int32"))
        b.eval(nan_s, inf_s)
        assert int(b.to_scalar(nan_s)) == 0, "SVD produced NaN"
        assert int(b.to_scalar(inf_s)) == 0, "SVD produced Inf"

    def test_near_singular_matrix(self, any_backend: "Backend") -> None:
        """Near-singular matrices should be handled gracefully."""
        b = any_backend

        v = b.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        rank1 = b.matmul(v, b.transpose(v))
        eps = machine_epsilon(b, rank1)
        perturbation = b.eye(5) * eps
        near_singular = rank1 + perturbation
        b.eval(near_singular)

        U, S, Vt = geodesic_svd(b, near_singular)
        b.eval(U, S, Vt)

        S_np = [float(v) for v in b.tolist(S)]
        assert S_np[0] > 0, "Largest singular value should be positive"

    def test_diagonal_matrix_svd(self, any_backend: "Backend") -> None:
        """Diagonal matrices should have accurate SVD decomposition."""
        b = any_backend

        diag_vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        D = b.diag(b.array(diag_vals))
        b.eval(D)

        U, S, Vt = geodesic_svd(b, D)
        b.eval(U, S, Vt)

        S_np = sorted([float(v) for v in b.tolist(S)], reverse=True)
        expected = sorted(diag_vals, reverse=True)
        for s, e in zip(S_np, expected):
            eps = _eps(b, s, e) * 10
            assert abs(s - e) <= eps, f"Expected {e}, got {s}"
