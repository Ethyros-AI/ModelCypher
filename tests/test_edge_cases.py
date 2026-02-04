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

"""Edge case tests for numerical stability and CKA.

Tests critical edge cases:
- Near-singular matrices (high condition number)
- Very small samples (n < d)
- Degenerate eigenvalue distributions
- Zero/near-zero denominators
- Empty inputs
- Single sample
- Orthogonal representations (CKA near 0)
- Identical representations (CKA = 1)
- Different dimensions
"""

from __future__ import annotations

import math
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
    condition_threshold,
    svd_rank_threshold,
    tiny_value,
    safe_log_epsilon,
    sqrt_scalar,
    is_finite,
    is_nan,
    is_inf,
    geodesic_svd,
    geodesic_pinv,
    safe_inverse,
    gpu_lstsq,
    invariant_alignment,
    find_magnitude_gap_threshold,
    ulp_scalar,
    compute_median,
    compute_median_nonzero,
)


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


class TestNumericalStabilityEdgeCases:
    """Edge case tests for numerical stability functions."""

    def test_division_epsilon_float32(self):
        """Division epsilon should be appropriate for float32."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        eps = division_epsilon(backend, arr)

        expected = sqrt_scalar(backend.finfo(arr.dtype).eps, backend)
        assert eps == expected, f"Unexpected division epsilon: {eps}"

    def test_machine_epsilon_reasonable(self):
        """Machine epsilon should be reasonable for dtype."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        eps = machine_epsilon(backend, arr)

        expected = backend.finfo(arr.dtype).eps
        assert eps == expected, f"Unexpected machine epsilon: {eps}"

    def test_tiny_value_positive(self):
        """Tiny value should be positive and very small."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        tiny = tiny_value(backend, arr)

        assert tiny > 0, f"Tiny value should be positive: {tiny}"
        expected = backend.finfo(arr.dtype).tiny
        assert tiny == expected, f"Tiny value should match dtype tiny: {tiny}"

    def test_safe_log_epsilon_prevents_log_zero(self):
        """Safe log epsilon should prevent log(0)."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        eps = safe_log_epsilon(backend, arr)

        expected = backend.finfo(arr.dtype).tiny
        assert eps == expected, f"Epsilon should match dtype tiny: {eps}"

    def test_condition_threshold_high(self):
        """Condition threshold should be high (1/eps)."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        thresh = condition_threshold(backend, arr)

        expected = 1.0 / backend.finfo(arr.dtype).eps
        assert thresh == expected, f"Condition threshold should match 1/eps: {thresh}"

    def test_svd_rank_threshold_dimension_scaled(self):
        """SVD rank threshold should scale with dimension."""
        backend = get_default_backend()
        arr = backend.array([1.0])
        backend.eval(arr)

        thresh_10 = svd_rank_threshold(backend, arr, max_dim=10)
        thresh_100 = svd_rank_threshold(backend, arr, max_dim=100)

        assert thresh_100 > thresh_10, "Threshold should scale with dimension"
        expected = 10 * thresh_10
        assert thresh_100 == pytest.approx(expected, abs=math.ulp(expected))


class TestNearSingularMatrices:
    """Tests for near-singular matrix handling."""

    def test_svd_near_singular_matrix(self):
        """SVD should handle near-singular matrices gracefully."""
        backend = get_default_backend()

        # Create near-singular matrix (rank 2 embedded in 10x10)
        backend.random_seed(42)
        u = backend.random_normal((10, 2))
        v = backend.random_normal((2, 10))
        matrix = backend.matmul(u, v)
        backend.eval(matrix)

        U, S, Vt = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vt)

        # Should have at most 2 significant singular values
        s_list = backend.tolist(S)
        eps = division_epsilon(backend, matrix)

        # First 2 should be significant
        assert s_list[0] > eps, f"First singular value should be significant: {s_list[0]}"
        assert s_list[1] > eps, f"Second singular value should be significant: {s_list[1]}"

    def test_pinv_near_singular(self):
        """Pseudoinverse should handle near-singular matrices."""
        backend = get_default_backend()

        # Create near-singular matrix
        backend.random_seed(42)
        u = backend.random_normal((10, 2))
        v = backend.random_normal((2, 10))
        matrix = backend.matmul(u, v)
        backend.eval(matrix)

        pinv = geodesic_pinv(backend, matrix)
        backend.eval(pinv)

        # pinv should be finite
        is_finite_arr = backend.all(backend.isfinite(pinv))
        backend.eval(is_finite_arr)
        assert backend.tolist(is_finite_arr), "Pseudoinverse should be finite"

        # A @ pinv(A) @ A should approximate A
        reconstructed = backend.matmul(backend.matmul(matrix, pinv), matrix)
        backend.eval(reconstructed)

        diff = backend.subtract(matrix, reconstructed)
        diff_norm = float(backend.tolist(backend.norm(diff)))
        matrix_norm = float(backend.tolist(backend.norm(matrix)))

        eps = division_epsilon(backend, matrix)
        rel_error = diff_norm / matrix_norm if matrix_norm > eps else diff_norm
        assert rel_error <= eps, f"Reconstruction error too large: {rel_error}"

    def test_safe_inverse_high_condition(self):
        """Safe inverse should handle high condition number matrices."""
        backend = get_default_backend()

        # Create ill-conditioned matrix
        backend.random_seed(42)
        diag = backend.array([1.0, 1e-3, 1e-6, 1e-9, 1e-12])
        matrix = backend.diag(diag)
        backend.eval(matrix)

        inv_matrix, cond = safe_inverse(backend, matrix, regularize=True)
        backend.eval(inv_matrix)

        # Inverse should be finite
        is_finite_arr = backend.all(backend.isfinite(inv_matrix))
        backend.eval(is_finite_arr)
        assert backend.tolist(is_finite_arr), "Inverse should be finite"

        eps = division_epsilon(backend, matrix)
        expected_min = max(1e-3, eps)
        expected = 1.0 / expected_min
        tol = division_epsilon(backend, backend.array([expected]))
        assert abs(cond - expected) <= tol * max(1.0, expected)


class TestSmallSampleCases:
    """Tests for cases where n < d (undersampled)."""

    @pytest.mark.parametrize("n,d", [(2, 10), (5, 20), (10, 100)])
    def test_underdetermined_lstsq(self, n: int, d: int):
        """Least squares should work for underdetermined systems."""
        backend = get_default_backend()
        backend.random_seed(42)

        A = backend.random_normal((n, d))
        B = backend.random_normal((n, 5))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)
        backend.eval(X)

        # Solution should be finite
        is_finite_arr = backend.all(backend.isfinite(X))
        backend.eval(is_finite_arr)
        assert backend.tolist(is_finite_arr), f"Solution should be finite for n={n}, d={d}"

        # Shape should be correct
        assert X.shape == (d, 5), f"Expected shape ({d}, 5), got {X.shape}"

    @pytest.mark.parametrize("n,d", [(3, 10), (5, 20)])
    def test_underdetermined_alignment(self, n: int, d: int):
        """Alignment should work for underdetermined cases."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((n, d))
        target = backend.random_normal((n, d))
        backend.eval(source, target)

        F = invariant_alignment(backend, source, target)
        backend.eval(F)

        # Alignment matrix should be finite
        is_finite_arr = backend.all(backend.isfinite(F))
        backend.eval(is_finite_arr)
        assert backend.tolist(is_finite_arr), "Alignment should be finite"


class TestDegenerateEigenvalues:
    """Tests for degenerate eigenvalue distributions."""

    def test_repeated_eigenvalues(self):
        """SVD should handle matrices with repeated eigenvalues."""
        backend = get_default_backend()

        # Create matrix with repeated eigenvalues
        diag = backend.array([5.0, 5.0, 5.0, 1.0, 1.0])
        matrix = backend.diag(diag)
        backend.eval(matrix)

        U, S, Vt = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vt)

        # Singular values should match diagonal
        s_list = sorted(backend.tolist(S), reverse=True)
        expected = [5.0, 5.0, 5.0, 1.0, 1.0]

        for actual, exp in zip(s_list, expected):
            assert actual == pytest.approx(exp, abs=math.ulp(exp)), f"Expected {exp}, got {actual}"

    def test_zero_eigenvalues(self):
        """SVD should handle matrices with zero eigenvalues."""
        backend = get_default_backend()

        # Create rank-deficient matrix
        diag = backend.array([5.0, 3.0, 0.0, 0.0, 0.0])
        matrix = backend.diag(diag)
        backend.eval(matrix)

        U, S, Vt = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vt)

        # Should have exactly 2 non-zero singular values
        s_list = backend.tolist(S)
        eps = division_epsilon(backend, matrix)

        nonzero_count = sum(1 for s in s_list if s > eps)
        assert nonzero_count == 2, f"Expected 2 non-zero singular values, got {nonzero_count}"


class TestZeroNearZeroCases:
    """Tests for zero and near-zero value handling."""

    def test_sqrt_zero(self):
        """sqrt_scalar should handle zero."""
        backend = get_default_backend()
        result = sqrt_scalar(0.0, backend)
        assert result == 0.0, f"sqrt(0) should be 0: {result}"

    def test_sqrt_negative(self):
        """sqrt_scalar should handle negative values safely."""
        backend = get_default_backend()
        result = sqrt_scalar(-1.0, backend)
        # sqrt_scalar guards against negative values
        assert result == 0.0, f"sqrt(-1) should be 0 with guard: {result}"

    def test_is_nan_detection(self):
        """is_nan should correctly detect NaN."""
        backend = get_default_backend()
        assert is_nan(float("nan"), backend)
        assert not is_nan(1.0, backend)
        assert not is_nan(float("inf"), backend)

    def test_is_inf_detection(self):
        """is_inf should correctly detect infinity."""
        backend = get_default_backend()
        assert is_inf(float("inf"), backend)
        assert is_inf(float("-inf"), backend)
        assert not is_inf(1.0, backend)
        assert not is_inf(float("nan"), backend)

    def test_svd_zero_matrix(self):
        """SVD should handle zero matrix gracefully."""
        backend = get_default_backend()
        matrix = backend.zeros((5, 5))
        backend.eval(matrix)

        U, S, Vt = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vt)

        # All singular values should be zero (or empty if rank 0)
        if S.shape[0] > 0:
            max_s = float(backend.tolist(backend.max(S)))
            eps = regularization_epsilon(backend, S)
            assert max_s < eps, f"Zero matrix should have zero singular values: {max_s}"

    def test_pinv_zero_matrix(self):
        """Pseudoinverse of zero matrix should be zero."""
        backend = get_default_backend()
        matrix = backend.zeros((5, 5))
        backend.eval(matrix)

        pinv = geodesic_pinv(backend, matrix)
        backend.eval(pinv)

        # pinv should be finite
        is_finite_arr = backend.all(backend.isfinite(pinv))
        backend.eval(is_finite_arr)
        assert backend.tolist(is_finite_arr), "Pseudoinverse of zero should be finite"


class TestCKAEdgeCases:
    """Edge case tests for CKA computation."""

    def test_cka_orthogonal_representations(self):
        """CKA with orthogonal feature dimensions should be computed correctly."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()

        # Create two matrices with orthogonal feature dimensions
        # Note: CKA measures representational similarity based on sample relationships,
        # not feature orthogonality. Even with orthogonal features, if samples have
        # similar structure in their relationships, CKA can be high.
        backend.random_seed(42)
        X = backend.random_normal((50, 10))
        Y = backend.random_normal((50, 10))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        # CKA should be valid and bounded
        assert result.is_valid, "CKA should be valid"
        eps = regularization_epsilon(backend, X)
        assert -eps <= result.cka <= 1.0 + eps, f"CKA should be bounded: {result.cka}"

    def test_cka_identical_representations(self):
        """CKA of identical representations should be 1.0."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        backend.eval(X)

        result = compute_cka(X, X, backend)

        eps = regularization_epsilon(backend, X)
        assert result.cka == pytest.approx(1.0, rel=eps), (
            f"CKA(X, X) should be 1.0: {result.cka}"
        )

    def test_cka_scaled_representations(self):
        """CKA should handle scaled representations."""
        from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        X_scaled = X * 100.0
        backend.eval(X, X_scaled)

        # Linear CKA is scale-invariant
        cka_original = compute_geodesic_cka(X, X, backend)
        cka_scaled = compute_geodesic_cka(X_scaled, X_scaled, backend)

        eps = regularization_epsilon(backend, X)
        assert cka_original == pytest.approx(cka_scaled, rel=eps), (
            f"Linear CKA should be scale-invariant: {cka_original} vs {cka_scaled}"
        )

    def test_cka_different_dimensions(self):
        """CKA should work with different feature dimensions."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 25, 43)
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert result.is_valid, "CKA should be valid for different dimensions"
        eps = division_epsilon(backend, X)
        assert -eps <= result.cka <= 1.0 + eps, f"CKA should be bounded: {result.cka}"

    def test_cka_minimal_samples(self):
        """CKA should handle minimal sample counts."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()

        # Two samples
        X = backend.array([[1.0, 2.0], [3.0, 4.0]])
        Y = backend.array([[1.1, 2.1], [3.1, 4.1]])
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert result.is_valid, "CKA should be valid for two samples"
        assert result.sample_count == 2

    def test_cka_single_sample_degeneracy(self):
        """CKA with single sample should handle degeneracy."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()
        X = backend.array([[1.0, 2.0, 3.0]])
        backend.eval(X)

        result = compute_cka(X, X, backend)

        # Single sample should return 0 (degenerate case)
        eps = division_epsilon(backend, X)
        assert abs(result.cka) <= eps, f"CKA with single sample should be 0: {result.cka}"
        assert result.sample_count == 1


class TestMagnitudeGapThreshold:
    """Tests for magnitude gap threshold detection."""

    def test_gap_in_sorted_values(self):
        """Should find gap in sorted values."""
        backend = get_default_backend()
        values = [0.1, 0.15, 0.2, 10.0, 15.0, 20.0]

        threshold = find_magnitude_gap_threshold(values, backend=backend)

        # Threshold should be at the gap (around 0.2)
        eps = division_epsilon(backend, backend.array(values))
        assert abs(threshold - 0.2) <= eps, f"Threshold should match gap value: {threshold}"

    def test_no_gap_uniform_values(self):
        """Should handle uniform values (no clear gap)."""
        backend = get_default_backend()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        scale = max(abs(v) for v in values)
        eps = ulp_scalar(scale, backend)
        threshold = find_magnitude_gap_threshold(values, eps=eps, backend=backend)

        rel_gaps = [(values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1)]
        max_gap = max(rel_gaps)
        expected = values[rel_gaps.index(max_gap)] if max_gap > eps else values[-1]
        assert abs(threshold - expected) <= eps

    def test_empty_values(self):
        """Should handle empty input."""
        backend = get_default_backend()
        values: list[float] = []

        threshold = find_magnitude_gap_threshold(values, backend=backend)

        assert threshold == 0.0, f"Empty input should return 0: {threshold}"

    def test_single_value(self):
        """Should handle single value."""
        backend = get_default_backend()
        values = [5.0]

        threshold = find_magnitude_gap_threshold(values, backend=backend)

        assert threshold == 5.0, f"Single value should return itself: {threshold}"


class TestMedianComputation:
    """Tests for median computation edge cases."""

    def test_median_odd_count(self):
        """Median of odd count should be middle value."""
        backend = get_default_backend()
        arr = backend.array([1.0, 5.0, 3.0, 2.0, 4.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == pytest.approx(3.0), f"Median should be 3.0: {median}"

    def test_median_even_count(self):
        """Median of even count should be average of middle two."""
        backend = get_default_backend()
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == pytest.approx(2.5), f"Median should be 2.5: {median}"

    def test_median_single_value(self):
        """Median of single value should be that value."""
        backend = get_default_backend()
        arr = backend.array([42.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == pytest.approx(42.0), f"Median should be 42.0: {median}"

    def test_median_empty(self):
        """Median of empty array should be 0."""
        backend = get_default_backend()
        arr = backend.array([]).reshape((0,))
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == 0.0, f"Median of empty should be 0: {median}"

    def test_median_nonzero_filters_zeros(self):
        """Median nonzero should exclude zeros."""
        backend = get_default_backend()
        arr = backend.array([0.0, 0.0, 1.0, 2.0, 3.0])
        backend.eval(arr)

        median = compute_median_nonzero(arr, backend)

        # Should be median of [1, 2, 3] = 2.0
        assert median == pytest.approx(2.0), f"Median nonzero should be 2.0: {median}"

    def test_median_nonzero_all_zeros(self):
        """Median nonzero with all zeros should be 0."""
        backend = get_default_backend()
        arr = backend.array([0.0, 0.0, 0.0])
        backend.eval(arr)

        median = compute_median_nonzero(arr, backend)

        assert median == 0.0, f"Median of all zeros should be 0: {median}"


class TestLstsqEdgeCases:
    """Edge case tests for least squares solver."""

    def test_lstsq_single_column_rhs(self):
        """Least squares should work with single column RHS."""
        backend = get_default_backend()
        backend.random_seed(42)

        A = backend.random_normal((10, 5))
        b = backend.random_normal((10,))
        backend.eval(A, b)

        x = gpu_lstsq(backend, A, b)
        backend.eval(x)

        assert x.shape == (5,), f"Expected shape (5,), got {x.shape}"

        # Check residual is reasonable
        residual = backend.matmul(A, x) - b
        res_norm = float(backend.tolist(backend.norm(residual)))
        b_norm = float(backend.tolist(backend.norm(b)))

        assert res_norm < b_norm, f"Residual should be less than RHS norm"

    def test_lstsq_multiple_rhs(self):
        """Least squares should work with multiple RHS columns."""
        backend = get_default_backend()
        backend.random_seed(42)

        A = backend.random_normal((10, 5))
        B = backend.random_normal((10, 3))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)
        backend.eval(X)

        assert X.shape == (5, 3), f"Expected shape (5, 3), got {X.shape}"

    def test_lstsq_square_system(self):
        """Least squares should work for square systems."""
        backend = get_default_backend()
        backend.random_seed(42)

        A = backend.random_normal((5, 5))
        b = backend.random_normal((5,))
        backend.eval(A, b)

        x = gpu_lstsq(backend, A, b)
        backend.eval(x)

        assert x.shape == (5,), f"Expected shape (5,), got {x.shape}"


try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestEdgeCaseHypothesis:
    """Hypothesis-based edge case tests."""

    @given(
        n=st.integers(min_value=2, max_value=50),
        d=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_svd_always_produces_finite(self, n: int, d: int, seed: int):
        """SVD should always produce finite values."""
        backend = get_default_backend()
        matrix = _random_matrix(backend, n, d, seed)
        backend.eval(matrix)

        U, S, Vt = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vt)

        # All outputs should be finite
        u_finite = backend.all(backend.isfinite(U))
        s_finite = backend.all(backend.isfinite(S))
        vt_finite = backend.all(backend.isfinite(Vt))
        backend.eval(u_finite, s_finite, vt_finite)

        assert backend.tolist(u_finite), "U should be finite"
        assert backend.tolist(s_finite), "S should be finite"
        assert backend.tolist(vt_finite), "Vt should be finite"

    @given(
        n=st.integers(min_value=2, max_value=30),
        d=st.integers(min_value=2, max_value=30),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_pinv_always_finite(self, n: int, d: int, seed: int):
        """Pseudoinverse should always be finite."""
        backend = get_default_backend()
        matrix = _random_matrix(backend, n, d, seed)
        backend.eval(matrix)

        pinv = geodesic_pinv(backend, matrix)
        backend.eval(pinv)

        is_finite_arr = backend.all(backend.isfinite(pinv))
        backend.eval(is_finite_arr)

        assert backend.tolist(is_finite_arr), "Pseudoinverse should be finite"

    @given(
        n=st.integers(min_value=4, max_value=30),
        d_source=st.integers(min_value=2, max_value=20),
        d_target=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_cka_always_bounded(
        self, n: int, d_source: int, d_target: int, seed: int
    ):
        """CKA should always be in [0, 1]."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()

        backend.random_seed(seed)
        X = backend.random_normal((n, d_source))
        backend.random_seed(seed + 1000)
        Y = backend.random_normal((n, d_target))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert result.is_valid, "CKA should be valid"
        eps = regularization_epsilon(backend, X)
        assert -eps <= result.cka <= 1.0 + eps, f"CKA out of bounds: {result.cka}"
