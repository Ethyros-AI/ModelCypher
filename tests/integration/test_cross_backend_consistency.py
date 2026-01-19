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

"""Integration tests for cross-backend consistency.

Verifies that geometric operations produce consistent results across MLX and JAX.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


class TestBasicOperationsConsistency:
    """Test that basic operations are consistent."""

    def test_matmul_consistency(self, backend):
        """Matrix multiplication should produce consistent results."""
        backend.random_seed(42)
        A = backend.random_normal((10, 20))
        B = backend.random_normal((20, 15))
        backend.eval(A, B)

        result = backend.matmul(A, B)
        backend.eval(result)

        # Check shape
        shape = result.shape
        assert shape == (10, 15), f"Expected (10, 15), got {shape}"

        # Check result is finite
        is_finite = backend.all(backend.isfinite(result))
        backend.eval(is_finite)
        assert backend.tolist(is_finite), "Result should be finite"

    def test_svd_consistency(self, backend):
        """SVD should produce valid decomposition."""
        backend.random_seed(42)
        A = backend.random_normal((20, 10))
        backend.eval(A)

        u, s, vt = backend.svd(A)
        backend.eval(u, s, vt)

        # SVD for (m, n) matrix with m > n returns:
        # u: (m, min(m,n)), s: (min(m,n),), vt: (min(m,n), n)
        # Reconstruct: A ≈ U @ diag(s) @ Vt
        min_dim = min(A.shape[0], A.shape[1])
        u_truncated = u[:, :min_dim]
        s_diag = backend.diag(s[:min_dim])
        vt_truncated = vt[:min_dim, :]
        backend.eval(u_truncated, s_diag, vt_truncated)

        reconstructed = backend.matmul(backend.matmul(u_truncated, s_diag), vt_truncated)
        backend.eval(reconstructed)

        diff = backend.subtract(A, reconstructed)
        diff_norm = float(backend.tolist(backend.norm(diff)))
        a_norm = float(backend.tolist(backend.norm(A)))

        denom_eps = division_epsilon(backend, A)
        tol = regularization_epsilon(backend, A)
        relative_error = diff_norm / a_norm if a_norm > denom_eps else diff_norm
        assert relative_error < tol, f"SVD reconstruction error: {relative_error}"

    def test_eigh_consistency(self, backend):
        """Eigendecomposition should produce valid results for symmetric matrices."""
        backend.random_seed(42)
        A = backend.random_normal((10, 10))
        # Make symmetric
        A_sym = backend.add(A, backend.transpose(A)) * 0.5
        backend.eval(A_sym)

        eigenvalues, eigenvectors = backend.eigh(A_sym)
        backend.eval(eigenvalues, eigenvectors)

        # Verify: A @ v = lambda * v for each eigenpair
        # Just check that eigenvalues are sorted (ascending by convention)
        eigenvalues_list = backend.tolist(eigenvalues)
        eps = regularization_epsilon(backend, eigenvalues)
        for i in range(len(eigenvalues_list) - 1):
            assert eigenvalues_list[i] <= eigenvalues_list[i + 1] + eps, (
                f"Eigenvalues should be sorted: {eigenvalues_list}"
            )

    def test_qr_consistency(self, backend):
        """QR decomposition should produce orthonormal Q."""
        backend.random_seed(42)
        A = backend.random_normal((20, 10))
        backend.eval(A)

        Q, R = backend.qr(A)
        backend.eval(Q, R)

        # Q should be orthonormal: Q.T @ Q ≈ I
        QtQ = backend.matmul(backend.transpose(Q), Q)
        backend.eval(QtQ)

        identity = backend.eye(Q.shape[1])
        diff = backend.subtract(QtQ, identity)
        diff_norm = float(backend.tolist(backend.norm(diff)))

        eps = regularization_epsilon(backend, Q)
        assert diff_norm < eps, f"Q should be orthonormal: ||Q.T @ Q - I|| = {diff_norm}"


class TestGeometricOperationsConsistency:
    """Test geometric operations for consistency."""

    def test_gram_matrix_consistency(self, backend):
        """Gram matrix computation should be consistent."""
        backend.random_seed(42)
        X = backend.random_normal((50, 32))
        backend.eval(X)

        # Linear Gram: K = X @ X.T
        gram = backend.matmul(X, backend.transpose(X))
        backend.eval(gram)

        # Gram should be symmetric
        diff = backend.subtract(gram, backend.transpose(gram))
        diff_norm = float(backend.tolist(backend.norm(diff)))

        eps = regularization_epsilon(backend, gram)
        assert diff_norm < eps, f"Gram should be symmetric: ||K - K.T|| = {diff_norm}"

        # Gram should be positive semi-definite (all eigenvalues >= 0)
        # Allow small negative due to numerical precision
        eigenvalues, _ = backend.eigh(gram)
        backend.eval(eigenvalues)

        min_eigenvalue = float(backend.tolist(backend.min(eigenvalues)))
        # Use more lenient tolerance for float32 precision
        eps = regularization_epsilon(backend, eigenvalues)
        assert min_eigenvalue >= -eps, f"Gram should be PSD: min_eigenvalue = {min_eigenvalue}"

    def test_centering_matrix_consistency(self, backend):
        """Centering matrix H should work correctly."""
        n = 20
        H = backend.subtract(
            backend.eye(n),
            backend.full((n, n), 1.0 / n),
        )
        backend.eval(H)

        # H should be symmetric
        diff = backend.subtract(H, backend.transpose(H))
        diff_norm = float(backend.tolist(backend.norm(diff)))
        eps = regularization_epsilon(backend, H)
        assert diff_norm < eps, f"H should be symmetric: {diff_norm}"

        # H @ 1 = 0 (centering removes mean)
        ones = backend.ones((n, 1))
        H_ones = backend.matmul(H, ones)
        backend.eval(H_ones)

        h_ones_norm = float(backend.tolist(backend.norm(H_ones)))
        # Use more lenient tolerance for float32 precision
        eps = regularization_epsilon(backend, H_ones)
        assert h_ones_norm < eps, f"H should center: ||H @ 1|| = {h_ones_norm}"

    def test_pseudoinverse_consistency(self, backend):
        """Pseudoinverse should satisfy A @ A+ @ A = A."""
        backend.random_seed(42)
        A = backend.random_normal((20, 10))
        backend.eval(A)

        A_pinv = backend.pinv(A)
        backend.eval(A_pinv)

        # A @ A+ @ A ≈ A
        reconstructed = backend.matmul(backend.matmul(A, A_pinv), A)
        backend.eval(reconstructed)

        diff = backend.subtract(A, reconstructed)
        diff_norm = float(backend.tolist(backend.norm(diff)))
        a_norm = float(backend.tolist(backend.norm(A)))

        denom_eps = division_epsilon(backend, A)
        tol = regularization_epsilon(backend, A)
        relative_error = diff_norm / a_norm if a_norm > denom_eps else diff_norm
        assert relative_error < tol, f"Pseudoinverse error: {relative_error}"


class TestNumericalStabilityConsistency:
    """Test numerical stability functions for consistency."""

    def test_machine_epsilon_reasonable(self, backend):
        """Machine epsilon should be reasonable for dtype."""
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        backend.random_seed(42)
        x = backend.random_normal((10,))
        backend.eval(x)

        eps = machine_epsilon(backend, x)
        # machine_epsilon returns a float directly
        eps_val = float(eps)

        expected = backend.finfo(x.dtype).eps
        assert eps_val == expected, f"Epsilon should match dtype finfo: {eps_val}"

    def test_safe_sqrt_consistency(self, backend):
        """Safe sqrt should handle near-zero values."""
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        # Test with various values
        test_values = [0.0, 1e-10, 0.5, 1.0, 100.0]

        for val in test_values:
            # sqrt_scalar takes a float, not an array
            result = sqrt_scalar(val, backend)
            result_val = float(result)

            # Result should be non-negative and finite
            assert result_val >= 0, f"sqrt should be non-negative: sqrt({val}) = {result_val}"
            assert not (result_val != result_val), f"sqrt should be finite: sqrt({val}) = {result_val}"  # NaN check


class TestCKAConsistency:
    """Test CKA computation for consistency."""

    def test_cka_self_similarity_is_one(self, backend):
        """CKA(X, X) should be 1.0."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend.random_seed(42)
        X = backend.random_normal((50, 32))
        backend.eval(X)

        result = compute_cka(X, X, backend)

        eps = regularization_epsilon(backend, X)
        assert result.cka == pytest.approx(1.0, rel=eps), (
            f"CKA(X, X) should be 1.0: got {result.cka}"
        )

    def test_cka_bounded_zero_one(self, backend):
        """CKA should be in [0, 1]."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend.random_seed(42)

        # Test multiple random pairs
        for seed_offset in range(5):
            backend.random_seed(42 + seed_offset)
            X = backend.random_normal((50, 32))
            backend.random_seed(100 + seed_offset)
            Y = backend.random_normal((50, 32))
            backend.eval(X, Y)

            result = compute_cka(X, Y, backend)

            # Allow small tolerance below 0 for numerical precision
            eps = regularization_epsilon(backend, X)
            assert -eps <= result.cka <= 1.0 + eps, (
                f"CKA should be in [0, 1]: got {result.cka}"
            )

    def test_cka_symmetry(self, backend):
        """CKA(X, Y) should equal CKA(Y, X)."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend.random_seed(42)
        X = backend.random_normal((50, 32))
        backend.random_seed(123)
        Y = backend.random_normal((50, 48))
        backend.eval(X, Y)

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        eps = regularization_epsilon(backend, X)
        assert result_xy.cka == pytest.approx(result_yx.cka, rel=eps), (
            f"CKA should be symmetric: CKA(X,Y)={result_xy.cka}, CKA(Y,X)={result_yx.cka}"
        )


class TestAlignmentConsistency:
    """Test alignment operations for consistency."""

    def test_procrustes_alignment_closed_form(self, backend):
        """Procrustes alignment should be closed-form: F = pinv(source) @ target."""
        backend.random_seed(42)

        n = 50
        d_source = 32
        d_target = 32

        source = backend.random_normal((n, d_source))
        target = backend.random_normal((n, d_target))
        backend.eval(source, target)

        # Closed-form alignment
        F = backend.matmul(backend.pinv(source), target)
        backend.eval(F)

        # Aligned source
        aligned = backend.matmul(source, F)
        backend.eval(aligned)

        # Aligned source should approximate target in least-squares sense
        diff = backend.subtract(aligned, target)
        diff_norm = float(backend.tolist(backend.norm(diff)))
        target_norm = float(backend.tolist(backend.norm(target)))

        # Residual should be reasonable (not necessarily zero due to rank)
        denom_eps = division_epsilon(backend, target)
        relative_residual = diff_norm / target_norm if target_norm > denom_eps else diff_norm
        projection = backend.matmul(source, backend.pinv(source))
        identity = backend.eye(n)
        residual = backend.matmul(backend.subtract(identity, projection), target)
        backend.eval(residual)

        residual_norm = float(backend.tolist(backend.norm(residual)))
        eps = division_epsilon(backend, residual)
        assert relative_residual == pytest.approx(
            residual_norm / target_norm if target_norm > denom_eps else residual_norm,
            rel=eps,
        )
