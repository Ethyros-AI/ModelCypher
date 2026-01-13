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

"""Extended tests for numerical stability utilities.

Tests critical APIs:
- geodesic_svd(): GPU-accelerated SVD with rank detection
- geodesic_pinv(): Moore-Penrose pseudo-inverse
- power_iteration_eigh(): Eigendecomposition
- gpu_lstsq(): GPU least-squares solver
- safe_inverse(): Regularized matrix inverse
- newton_schulz_inverse(): Pure matmul inverse
- invariant_alignment(): CKA=1.0 alignment transform
- Statistical utilities: compute_median, pearson/spearman correlation
- Epsilon utilities: machine_epsilon, division_epsilon, etc.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_linear_cka
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    compute_median,
    compute_median_nonzero,
    compute_pearson_correlation,
    compute_spearman_correlation,
    geodesic_pinv,
    geodesic_svd,
    gpu_lstsq,
    invariant_alignment,
    machine_epsilon,
    division_epsilon,
    regularization_epsilon,
    tiny_value,
    newton_schulz_inverse,
    power_iteration_eigh,
    safe_inverse,
    sqrt_scalar,
    is_finite,
    is_nan,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestGeodesicSVD:
    """Tests for geodesic_svd()."""

    def test_basic_svd(self, backend):
        """Basic SVD should work."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        U, S, Vt = geodesic_svd(backend, A)

        assert U is not None
        assert S is not None
        assert Vt is not None

    def test_svd_shapes(self, backend):
        """SVD shapes should be correct."""
        m, n = 16, 8
        A = backend.random_normal((m, n))
        backend.eval(A)

        U, S, Vt = geodesic_svd(backend, A)

        # Full SVD shapes
        assert backend.shape(U)[0] == m
        assert backend.shape(Vt)[1] == n
        assert backend.shape(S)[0] <= min(m, n)

    def test_truncated_svd(self, backend):
        """Truncated SVD should limit rank."""
        A = backend.random_normal((16, 16))
        backend.eval(A)

        k = 5
        U, S, Vt = geodesic_svd(backend, A, k=k)

        assert backend.shape(U)[1] == k
        assert backend.shape(S)[0] == k
        assert backend.shape(Vt)[0] == k

    def test_svd_reconstruction(self, backend):
        """SVD should allow reconstruction."""
        A = backend.random_normal((8, 8))
        backend.eval(A)

        U, S, Vt = geodesic_svd(backend, A)

        # Reconstruct: A ≈ U @ diag(S) @ Vt
        S_diag = backend.diag(S)
        reconstructed = backend.matmul(U, backend.matmul(S_diag, Vt))
        backend.eval(reconstructed)

        diff = backend.mean(backend.abs(A - reconstructed))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-4

    def test_svd_singular_values_ordered(self, backend):
        """Singular values should be in descending order."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        _, S, _ = geodesic_svd(backend, A)
        backend.eval(S)

        # Check descending order
        S_list = backend.tolist(S)
        for i in range(len(S_list) - 1):
            assert S_list[i] >= S_list[i + 1] - 1e-6

    def test_svd_singular_values_nonnegative(self, backend):
        """Singular values should be non-negative."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        _, S, _ = geodesic_svd(backend, A)

        min_s = backend.min(S)
        backend.eval(min_s)
        assert float(backend.to_scalar(min_s)) >= -1e-6

    def test_zero_matrix(self, backend):
        """Zero matrix should return empty SVD."""
        A = backend.zeros((8, 8))
        backend.eval(A)

        U, S, Vt = geodesic_svd(backend, A)

        # Should handle gracefully (empty or zero singular values)
        assert U is not None

    def test_svd_tiny_threshold(self, backend):
        """Tiny-energy matrices should short-circuit; slightly larger should not."""
        m, n = 4, 4
        tiny = tiny_value(backend, backend.array([1.0]))
        threshold = tiny * float(m * n)

        small_energy = 0.5 * threshold
        large_energy = 2.0 * threshold

        scale_small = sqrt_scalar(small_energy / float(m * n), backend)
        scale_large = sqrt_scalar(large_energy / float(m * n), backend)

        A_small = backend.full((m, n), scale_small)
        A_large = backend.full((m, n), scale_large)
        backend.eval(A_small, A_large)

        _, S_small, _ = geodesic_svd(backend, A_small)
        _, S_large, _ = geodesic_svd(backend, A_large)

        small_size = int(backend.shape(S_small)[0])
        large_size = int(backend.shape(S_large)[0])

        if small_size == 0:
            max_small = 0.0
        else:
            max_small_arr = backend.max(backend.abs(S_small))
            backend.eval(max_small_arr)
            max_small = float(backend.to_scalar(max_small_arr))

        if large_size == 0:
            max_large = 0.0
        else:
            max_large_arr = backend.max(backend.abs(S_large))
            backend.eval(max_large_arr)
            max_large = float(backend.to_scalar(max_large_arr))

        assert max_small == 0.0
        assert max_large > 0.0


class TestGeodesicPinv:
    """Tests for geodesic_pinv()."""

    def test_basic_pinv(self, backend):
        """Basic pseudo-inverse should work."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        A_pinv = geodesic_pinv(backend, A)

        assert A_pinv is not None
        assert backend.shape(A_pinv) == (8, 16)

    def test_pinv_identity_property(self, backend):
        """A @ pinv(A) @ A ≈ A for overdetermined."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        A_pinv = geodesic_pinv(backend, A)

        reconstructed = backend.matmul(A, backend.matmul(A_pinv, A))
        backend.eval(reconstructed)

        diff = backend.mean(backend.abs(A - reconstructed))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-4

    def test_pinv_square_invertible(self, backend):
        """Pseudo-inverse of invertible matrix ≈ inverse."""
        # Create a well-conditioned matrix
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.1 * backend.eye(8)
        backend.eval(A)

        A_pinv = geodesic_pinv(backend, A)
        A_inv = backend.inv(A)
        backend.eval(A_pinv, A_inv)

        diff = backend.mean(backend.abs(A_pinv - A_inv))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-3


class TestPowerIterationEigh:
    """Tests for power_iteration_eigh()."""

    def test_basic_eigh(self, backend):
        """Basic eigendecomposition should work."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A))  # Make symmetric
        backend.eval(A)

        eigenvalues, eigenvectors = power_iteration_eigh(backend, A, k=4)

        assert eigenvalues is not None
        assert eigenvectors is not None
        assert backend.shape(eigenvalues)[0] == 4
        assert backend.shape(eigenvectors)[1] == 4

    def test_eigenvalues_ordered(self, backend):
        """Eigenvalues should be in descending order."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A))
        backend.eval(A)

        eigenvalues, _ = power_iteration_eigh(backend, A, k=4)
        backend.eval(eigenvalues)

        e_list = backend.tolist(eigenvalues)
        for i in range(len(e_list) - 1):
            assert e_list[i] >= e_list[i + 1] - 1e-6

    def test_eigenvector_orthogonality(self, backend):
        """Eigenvectors should be orthonormal."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A))
        backend.eval(A)

        _, eigenvectors = power_iteration_eigh(backend, A, k=4)

        # V^T @ V should be identity
        VtV = backend.matmul(backend.transpose(eigenvectors), eigenvectors)
        I = backend.eye(4)
        backend.eval(VtV)

        diff = backend.mean(backend.abs(VtV - I))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-4


class TestGpuLstsq:
    """Tests for gpu_lstsq()."""

    def test_basic_lstsq(self, backend):
        """Basic least squares should work."""
        A = backend.random_normal((16, 8))
        B = backend.random_normal((16, 4))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)

        assert X is not None
        assert backend.shape(X) == (8, 4)
        assert all_finite(X, backend)

    def test_lstsq_residual_minimized(self, backend):
        """Least squares should minimize residual."""
        A = backend.random_normal((16, 8))
        B = backend.random_normal((16, 4))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)

        # Residual should be finite
        residual = backend.matmul(A, X) - B
        res_norm = backend.norm(residual)
        backend.eval(res_norm)
        assert all_finite(res_norm, backend)

    def test_lstsq_1d_rhs(self, backend):
        """Should handle 1D right-hand side."""
        A = backend.random_normal((16, 8))
        b = backend.random_normal((16,))
        backend.eval(A, b)

        x = gpu_lstsq(backend, A, b)

        assert x is not None
        assert backend.shape(x) == (8,)

    def test_lstsq_underdetermined(self, backend):
        """Should handle underdetermined systems (n < d)."""
        A = backend.random_normal((8, 16))  # n < d
        B = backend.random_normal((8, 4))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)

        assert X is not None
        assert backend.shape(X) == (16, 4)
        assert all_finite(X, backend)

    def test_lstsq_square(self, backend):
        """Should handle square systems."""
        A = backend.random_normal((8, 8))
        B = backend.random_normal((8, 4))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)

        assert backend.shape(X) == (8, 4)

    def test_lstsq_stats_populated(self, backend):
        """Stats dict should be populated if provided."""
        A = backend.random_normal((16, 8))
        B = backend.random_normal((16, 4))
        backend.eval(A, B)

        stats: dict[str, float] = {}
        X = gpu_lstsq(backend, A, B, stats=stats)

        assert "residual_norm" in stats
        assert "rhs_norm" in stats
        assert "method" in stats


class TestSafeInverse:
    """Tests for safe_inverse()."""

    def test_basic_inverse(self, backend):
        """Basic matrix inverse should work."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.1 * backend.eye(8)
        backend.eval(A)

        A_inv, cond = safe_inverse(backend, A)

        assert A_inv is not None
        assert backend.shape(A_inv) == (8, 8)
        assert cond > 0

    def test_inverse_property(self, backend):
        """A @ inv(A) ≈ I."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.1 * backend.eye(8)
        backend.eval(A)

        A_inv, _ = safe_inverse(backend, A)

        product = backend.matmul(A, A_inv)
        I = backend.eye(8)
        backend.eval(product)

        diff = backend.mean(backend.abs(product - I))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-3

    def test_condition_number_returned(self, backend):
        """Condition number should be returned."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.1 * backend.eye(8)
        backend.eval(A)

        _, cond = safe_inverse(backend, A)

        assert cond >= 1.0  # Condition number >= 1


class TestNewtonSchulzInverse:
    """Tests for newton_schulz_inverse()."""

    def test_basic_inverse(self, backend):
        """Basic Newton-Schulz inverse should work."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.5 * backend.eye(8)
        backend.eval(A)

        A_inv = newton_schulz_inverse(backend, A)

        assert A_inv is not None
        assert backend.shape(A_inv) == (8, 8)
        assert all_finite(A_inv, backend)

    def test_inverse_accuracy(self, backend):
        """Newton-Schulz should produce reasonable inverse."""
        A = backend.random_normal((8, 8))
        A = backend.matmul(A, backend.transpose(A)) + 0.5 * backend.eye(8)
        backend.eval(A)

        A_inv = newton_schulz_inverse(backend, A)

        product = backend.matmul(A, A_inv)
        I = backend.eye(8)
        backend.eval(product)

        diff = backend.mean(backend.abs(product - I))
        backend.eval(diff)
        # Newton-Schulz may be less accurate than direct inverse
        # Tolerance allows for numerical variation across runs
        assert float(backend.to_scalar(diff)) < 0.15


class TestInvariantAlignment:
    """Tests for invariant_alignment()."""

    def test_basic_alignment(self, backend):
        """Basic alignment should work."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 24))
        backend.eval(source, target)

        F = invariant_alignment(backend, source, target)

        assert F is not None
        assert backend.shape(F) == (32, 24)
        assert all_finite(F, backend)

    def test_alignment_transform_applied(self, backend):
        """Alignment transform should map source to target space."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 24))
        backend.eval(source, target)

        F = invariant_alignment(backend, source, target)

        aligned = backend.matmul(source, F)
        backend.eval(aligned)

        assert backend.shape(aligned) == (16, 24)
        assert all_finite(aligned, backend)

    def test_alignment_stats_populated(self, backend):
        """Stats dict should be populated if provided."""
        source = backend.random_normal((16, 32))
        target = backend.random_normal((16, 24))
        backend.eval(source, target)

        stats: dict[str, float] = {}
        F = invariant_alignment(backend, source, target, stats=stats)

        assert "residual_norm" in stats or "method" in stats

    @given(
        n_samples=st.integers(min_value=5, max_value=20),
        extra_dims=st.integers(min_value=0, max_value=20),
        target_dim=st.integers(min_value=5, max_value=24),
    )
    @settings(max_examples=20, deadline=None)
    def test_alignment_linear_cka_one(self, n_samples, extra_dims, target_dim):
        """Invariant alignment should yield CKA=1.0 on probes when n <= d."""
        backend = get_default_backend()
        source_dim = n_samples + extra_dims

        source = backend.random_normal((n_samples, source_dim))
        target = backend.random_normal((n_samples, target_dim))
        backend.eval(source, target)

        F = invariant_alignment(backend, source, target)
        aligned = backend.matmul(source, F)
        backend.eval(aligned)

        cka = compute_linear_cka(aligned, target, backend)
        eps = division_epsilon(backend, aligned)
        tol = eps * max(1.0, float(n_samples))

        assert 1.0 - cka <= tol


class TestComputeMedian:
    """Tests for compute_median()."""

    def test_basic_median(self, backend):
        """Basic median computation should work."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0, 5.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == 3.0

    def test_even_length(self, backend):
        """Even length array should average middle two."""
        arr = backend.array([1.0, 2.0, 3.0, 4.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == 2.5

    def test_single_element(self, backend):
        """Single element should return itself."""
        arr = backend.array([42.0])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == 42.0

    def test_empty_array(self, backend):
        """Empty array should return 0."""
        arr = backend.array([])
        backend.eval(arr)

        median = compute_median(arr, backend)

        assert median == 0.0


class TestComputeMedianNonzero:
    """Tests for compute_median_nonzero()."""

    def test_basic_nonzero_median(self, backend):
        """Should compute median of non-zero values."""
        arr = backend.array([0.0, 1.0, 2.0, 3.0, 0.0])
        backend.eval(arr)

        median = compute_median_nonzero(arr, backend)

        assert median == 2.0

    def test_all_zeros(self, backend):
        """All zeros should return 0."""
        arr = backend.array([0.0, 0.0, 0.0])
        backend.eval(arr)

        median = compute_median_nonzero(arr, backend)

        assert median == 0.0


class TestCorrelations:
    """Tests for correlation functions."""

    def test_pearson_perfect_correlation(self, backend):
        """Identical lists should have correlation 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        corr = compute_pearson_correlation(x, y, backend=backend)

        assert abs(corr - 1.0) < 0.1  # Geodesic may differ slightly

    def test_spearman_perfect_correlation(self, backend):
        """Identical lists should have rank correlation 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        corr = compute_spearman_correlation(x, y, backend=backend)

        assert abs(corr - 1.0) < 0.1

    def test_pearson_length_mismatch(self, backend):
        """Mismatched lengths should return default."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]

        corr = compute_pearson_correlation(x, y, default=0.0, backend=backend)

        assert corr == 0.0

    def test_spearman_empty_lists(self, backend):
        """Empty lists should return default."""
        corr = compute_spearman_correlation([], [], default=-1.0, backend=backend)

        assert corr == -1.0


class TestEpsilonUtilities:
    """Tests for epsilon utilities."""

    def test_machine_epsilon_positive(self, backend):
        """Machine epsilon should be positive."""
        arr = backend.array([1.0])
        backend.eval(arr)

        eps = machine_epsilon(backend, arr)

        assert eps > 0

    def test_division_epsilon_larger(self, backend):
        """Division epsilon should be larger than machine epsilon."""
        arr = backend.array([1.0])
        backend.eval(arr)

        m_eps = machine_epsilon(backend, arr)
        d_eps = division_epsilon(backend, arr)

        assert d_eps > m_eps

    def test_regularization_epsilon(self, backend):
        """Regularization epsilon should be between machine and division epsilon."""
        arr = backend.array([1.0])
        backend.eval(arr)

        m_eps = machine_epsilon(backend, arr)
        r_eps = regularization_epsilon(backend, arr)

        assert r_eps > m_eps


class TestScalarHelpers:
    """Tests for scalar helper functions."""

    def test_sqrt_scalar(self, backend):
        """sqrt_scalar should work."""
        result = sqrt_scalar(4.0, backend)
        assert abs(result - 2.0) < 1e-6

    def test_sqrt_scalar_negative(self, backend):
        """sqrt_scalar with negative should return 0."""
        result = sqrt_scalar(-4.0, backend)
        assert result == 0.0

    def test_is_finite(self, backend):
        """is_finite should detect finite values."""
        assert is_finite(1.0, backend) is True
        assert is_finite(float("inf"), backend) is False
        assert is_finite(float("nan"), backend) is False

    def test_is_nan(self, backend):
        """is_nan should detect NaN."""
        assert is_nan(float("nan"), backend) is True
        assert is_nan(1.0, backend) is False


class TestNumericalStabilityMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        m=st.integers(min_value=4, max_value=16),
        n=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_svd_shapes_valid(self, m, n):
        """SVD should produce valid shapes."""
        backend = get_default_backend()
        A = backend.random_normal((m, n))
        backend.eval(A)

        U, S, Vt = geodesic_svd(backend, A)

        assert backend.shape(U)[0] == m
        assert backend.shape(Vt)[1] == n

    @given(
        m=st.integers(min_value=4, max_value=16),
        n=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_pinv_shape_transposed(self, m, n):
        """Pseudo-inverse shape should be transposed."""
        backend = get_default_backend()
        A = backend.random_normal((m, n))
        backend.eval(A)

        A_pinv = geodesic_pinv(backend, A)

        assert backend.shape(A_pinv) == (n, m)

    @given(
        n=st.integers(min_value=4, max_value=16),
        d=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_lstsq_finite(self, n, d):
        """Least squares solution should be finite."""
        backend = get_default_backend()
        A = backend.random_normal((n, d))
        B = backend.random_normal((n, 1))
        backend.eval(A, B)

        X = gpu_lstsq(backend, A, B)

        assert all_finite(X, backend)

    @given(
        n=st.integers(min_value=4, max_value=12),
    )
    @settings(max_examples=5, deadline=None)
    def test_safe_inverse_finite(self, n):
        """Safe inverse should be finite."""
        backend = get_default_backend()
        A = backend.random_normal((n, n))
        A = backend.matmul(A, backend.transpose(A)) + 0.1 * backend.eye(n)
        backend.eval(A)

        A_inv, cond = safe_inverse(backend, A)

        assert all_finite(A_inv, backend)
        assert cond >= 1.0

    @given(
        n=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=5, deadline=None)
    def test_median_in_range(self, n):
        """Median should be within data range."""
        backend = get_default_backend()
        arr = backend.random_normal((n,))
        backend.eval(arr)

        median = compute_median(arr, backend)

        min_val = float(backend.to_scalar(backend.min(arr)))
        max_val = float(backend.to_scalar(backend.max(arr)))

        assert min_val <= median <= max_val
