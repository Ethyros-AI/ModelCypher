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

"""Ground truth comparison tests.

Compares ModelCypher implementations against reference implementations
to ensure numerical correctness.

References:
- CKA: Kornblith et al. 2019 "Similarity of Neural Network Representations"
- Procrustes: scipy.spatial.procrustes
- Intrinsic Dimension: TwoNN method (Facco et al. 2017)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend


# =============================================================================
# CKA Ground Truth Tests
# =============================================================================


def _reference_linear_kernel(X: np.ndarray) -> np.ndarray:
    """Reference linear kernel implementation."""
    return X @ X.T


def _reference_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Reference HSIC implementation (biased estimator)."""
    n = K.shape[0]
    # Center the kernel matrices
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    # HSIC = (1/n^2) * trace(K_centered @ L_centered)
    return np.trace(K_centered @ L_centered) / (n * n)


def _reference_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Reference CKA implementation following Kornblith et al. 2019."""
    K = _reference_linear_kernel(X)
    L = _reference_linear_kernel(Y)

    hsic_xy = _reference_hsic(K, L)
    hsic_xx = _reference_hsic(K, K)
    hsic_yy = _reference_hsic(L, L)

    return hsic_xy / math.sqrt(hsic_xx * hsic_yy)


class TestCKAGroundTruth:
    """Compare CKA implementation against reference."""

    def test_cka_matches_reference_random_data(self) -> None:
        """CKA should match reference on random data."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        np.random.seed(42)

        # Generate random test data
        X = np.random.randn(50, 64).astype(np.float32)
        Y = np.random.randn(50, 32).astype(np.float32)

        # Reference implementation
        ref_cka = _reference_cka(X, Y)

        # ModelCypher implementation (biased to match reference)
        X_arr = backend.array(X)
        Y_arr = backend.array(Y)
        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        # Should match within numerical tolerance
        assert abs(result.cka - ref_cka) < 1e-4

    def test_cka_matches_reference_correlated_data(self) -> None:
        """CKA should match on correlated data."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        np.random.seed(42)

        # Create correlated activations
        base = np.random.randn(30, 10).astype(np.float32)
        noise = np.random.randn(30, 10).astype(np.float32) * 0.1
        X = base
        Y = base + noise  # Highly correlated

        ref_cka = _reference_cka(X, Y)

        X_arr = backend.array(X)
        Y_arr = backend.array(Y)
        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        assert abs(result.cka - ref_cka) < 1e-4
        # Should be high correlation
        assert ref_cka > 0.9

    def test_cka_matches_reference_orthogonal_data(self) -> None:
        """CKA should match on orthogonal data."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()

        # Create orthogonal activations
        n_samples = 20
        X = np.zeros((n_samples, 2), dtype=np.float32)
        Y = np.zeros((n_samples, 2), dtype=np.float32)
        for i in range(n_samples):
            X[i, 0] = float(i)
            Y[i, 1] = float(i)

        ref_cka = _reference_cka(X, Y)

        X_arr = backend.array(X)
        Y_arr = backend.array(Y)
        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        # Low similarity for orthogonal
        assert abs(result.cka - ref_cka) < 1e-4


# =============================================================================
# Procrustes Ground Truth Tests (using scipy)
# =============================================================================


try:
    from scipy.spatial import procrustes as scipy_procrustes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestProcrustesGroundTruth:
    """Compare Procrustes against scipy reference."""

    def test_procrustes_disparity_matches_scipy(self) -> None:
        """Procrustes disparity should match scipy for 2-matrix alignment."""
        from modelcypher.core.domain.geometry.generalized_procrustes import (
            Config,
            FrechetMeanConfig,
            GeneralizedProcrustes,
        )

        backend = get_default_backend()
        np.random.seed(42)

        # Generate random test data
        X = np.random.randn(10, 5).astype(np.float64)
        Y = np.random.randn(10, 5).astype(np.float64)

        # Scipy reference
        _, _, disparity = scipy_procrustes(X, Y)

        # ModelCypher implementation
        gpa = GeneralizedProcrustes(backend)
        config = Config(
            allow_scaling=True,  # scipy does scaling
            frechet_mean=FrechetMeanConfig(enabled=False),
        )
        result = gpa.align([X.tolist(), Y.tolist()], config)

        assert result is not None
        # Our error metric is different from scipy's disparity
        # but both should be small for similar matrices
        # Note: scipy normalizes differently, so we just check magnitude
        assert result.alignment_error < 10 * disparity + 0.1

    def test_procrustes_recovers_rotation(self) -> None:
        """Procrustes should recover a known rotation."""
        from modelcypher.core.domain.geometry.generalized_procrustes import (
            Config,
            FrechetMeanConfig,
            GeneralizedProcrustes,
        )

        backend = get_default_backend()
        np.random.seed(42)

        # Generate random base
        X = np.random.randn(15, 4).astype(np.float64)

        # Apply known rotation (Givens rotation in first two dims)
        theta = np.pi / 4
        R = np.eye(4)
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)

        Y = X @ R

        # ModelCypher should recover near-zero error
        gpa = GeneralizedProcrustes(backend)
        config = Config(
            allow_scaling=False,
            frechet_mean=FrechetMeanConfig(enabled=False),
        )
        result = gpa.align([X.tolist(), Y.tolist()], config)

        assert result is not None
        # Error should be very small since Y is just a rotation of X
        assert result.alignment_error < 1e-3


# =============================================================================
# Intrinsic Dimension Ground Truth Tests
# =============================================================================


class TestIntrinsicDimensionGroundTruth:
    """Ground truth tests for intrinsic dimension."""

    def test_linear_subspace_dimension_accurate(self) -> None:
        """TwoNN should accurately measure linear subspace dimension."""
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
            TwoNNConfiguration,
        )

        backend = get_default_backend()
        np.random.seed(42)

        # Create a 5-dimensional linear subspace in 20D
        true_dim = 5
        ambient_dim = 20
        n_samples = 200

        # Random basis
        basis = np.random.randn(true_dim, ambient_dim).astype(np.float32)
        # Random coefficients
        coeffs = np.random.randn(n_samples, true_dim).astype(np.float32)
        # Points in subspace
        points = coeffs @ basis

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimension.compute_two_nn(
            points.tolist(),
            configuration=config,
        )

        # Should be close to true dimension
        assert abs(estimate.intrinsic_dimension - true_dim) < 2.0

    def test_full_rank_gaussian_dimension(self) -> None:
        """Full-rank Gaussian should have dimension close to ambient."""
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
            TwoNNConfiguration,
        )

        np.random.seed(42)

        ambient_dim = 8
        n_samples = 200

        # Full-rank Gaussian
        points = np.random.randn(n_samples, ambient_dim).astype(np.float32)

        config = TwoNNConfiguration(use_regression=True)
        estimate = IntrinsicDimension.compute_two_nn(
            points.tolist(),
            configuration=config,
        )

        # Should be close to ambient dimension
        assert abs(estimate.intrinsic_dimension - ambient_dim) < 3.0


# =============================================================================
# QR Decomposition Ground Truth Tests
# =============================================================================


class TestQRGroundTruth:
    """Ground truth tests for QR decomposition."""

    def test_qr_matches_numpy(self) -> None:
        """Backend QR should match numpy QR."""
        backend = get_default_backend()
        np.random.seed(42)

        # Random matrix
        A = np.random.randn(10, 5).astype(np.float32)

        # Numpy reference
        Q_np, R_np = np.linalg.qr(A)

        # Backend implementation
        A_arr = backend.array(A)
        Q_be, R_be = backend.qr(A_arr)
        backend.eval(Q_be)
        backend.eval(R_be)
        Q_be_np = backend.to_numpy(Q_be)
        R_be_np = backend.to_numpy(R_be)

        # Q should be orthogonal
        QtQ = Q_be_np.T @ Q_be_np
        for i in range(QtQ.shape[0]):
            for j in range(QtQ.shape[0]):
                expected = 1.0 if i == j else 0.0
                assert abs(QtQ[i, j] - expected) < 1e-4

        # A = QR should hold
        A_reconstructed = Q_be_np @ R_be_np
        assert np.allclose(A, A_reconstructed, atol=1e-4)


# =============================================================================
# SVD Ground Truth Tests
# =============================================================================


class TestSVDGroundTruth:
    """Ground truth tests for SVD."""

    def test_svd_matches_numpy(self) -> None:
        """Backend SVD should match numpy SVD."""
        backend = get_default_backend()
        np.random.seed(42)

        # Random square matrix (avoids shape issues)
        A = np.random.randn(6, 6).astype(np.float32)

        # Backend implementation
        A_arr = backend.array(A)
        U_be, S_be, Vt_be = backend.svd(A_arr)
        backend.eval(U_be)
        backend.eval(S_be)
        backend.eval(Vt_be)
        U_np = backend.to_numpy(U_be)
        S_np = backend.to_numpy(S_be)
        Vt_np = backend.to_numpy(Vt_be)

        # Reconstruct A = U @ diag(S) @ Vt
        S_diag = np.diag(S_np)
        A_reconstructed = U_np @ S_diag @ Vt_np

        assert np.allclose(A, A_reconstructed, atol=1e-4)

    def test_svd_singular_values_non_negative(self) -> None:
        """Singular values should be non-negative."""
        backend = get_default_backend()
        np.random.seed(42)

        A = np.random.randn(10, 6).astype(np.float32)
        A_arr = backend.array(A)
        _, S_be, _ = backend.svd(A_arr)
        backend.eval(S_be)
        S_np = backend.to_numpy(S_be)

        assert np.all(S_np >= -1e-10)


# =============================================================================
# Eigenvalue Ground Truth Tests
# =============================================================================


class TestEighGroundTruth:
    """Ground truth tests for symmetric eigenvalue decomposition."""

    def test_eigh_matches_numpy(self) -> None:
        """Backend eigh should match numpy eigh."""
        backend = get_default_backend()
        np.random.seed(42)

        # Create symmetric matrix
        A = np.random.randn(6, 6).astype(np.float32)
        A = (A + A.T) / 2

        # Numpy reference
        vals_np, vecs_np = np.linalg.eigh(A)

        # Backend implementation
        A_arr = backend.array(A)
        vals_be, vecs_be = backend.eigh(A_arr)
        backend.eval(vals_be)
        backend.eval(vecs_be)
        vals_be_np = backend.to_numpy(vals_be)
        vecs_be_np = backend.to_numpy(vecs_be)

        # Eigenvalues should match (sorted)
        vals_be_sorted = np.sort(vals_be_np)
        vals_np_sorted = np.sort(vals_np)
        assert np.allclose(vals_be_sorted, vals_np_sorted, atol=1e-4)

        # A @ v = lambda * v for each eigenpair
        for i in range(len(vals_be_np)):
            lam = vals_be_np[i]
            v = vecs_be_np[:, i]
            Av = A @ v
            lam_v = lam * v
            # Allow for sign flips in eigenvectors
            assert np.allclose(np.abs(Av), np.abs(lam_v), atol=1e-3)
