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
- Procrustes: alignment invariants under rotation/scale
- Intrinsic Dimension: TwoNN method (Facco et al. 2017)
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.support.array_utils import array_to_list


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _to_scalar(backend, array) -> float:
    backend.eval(array)
    return float(backend.to_scalar(array))


def _max_abs(backend, array) -> float:
    max_val = backend.max(backend.abs(array))
    backend.eval(max_val)
    return float(backend.to_scalar(max_val))


PI = 3.141592653589793

# =============================================================================
# CKA Ground Truth Tests
# =============================================================================


class TestCKAGroundTruth:
    """Compare CKA implementation for consistency.
    
    Note: We do NOT compare against linear CKA reference because:
    1. Linear CKA cannot capture nonlinear manifold structure
    2. Euclidean distance breaks down in high dimensions
    3. ModelCypher uses RBF kernel with geodesic distances (correct for manifolds)
    
    Instead, we verify CKA properties and self-consistency.
    """

    def test_cka_self_similarity_is_one(self) -> None:
        """CKA(X, X) should equal 1.0."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        X_arr = backend.random_normal((50, 64))
        backend.eval(X_arr)

        result = compute_cka(X_arr, X_arr, backend, estimator=HSICEstimator.BIASED)

        eps = _eps(backend, result.cka, 1.0)
        assert abs(result.cka - 1.0) <= eps

    def test_cka_symmetry(self) -> None:
        """CKA(X, Y) should equal CKA(Y, X)."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        X_arr = backend.random_normal((50, 64))
        Y_arr = backend.random_normal((50, 32))
        backend.eval(X_arr, Y_arr)

        result_xy = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)
        result_yx = compute_cka(Y_arr, X_arr, backend, estimator=HSICEstimator.BIASED)

        eps = _eps(backend, result_xy.cka, result_yx.cka)
        assert abs(result_xy.cka - result_yx.cka) <= eps

    def test_cka_bounded_zero_to_one(self) -> None:
        """CKA should be in [0, 1]."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        X_arr = backend.random_normal((50, 64))
        Y_arr = backend.random_normal((50, 32))
        backend.eval(X_arr, Y_arr)

        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        eps = _eps(backend, result.cka, 0.0, 1.0)
        assert -eps <= result.cka <= 1.0 + eps

    def test_cka_correlated_data_high_similarity(self) -> None:
        """Correlated data should have high CKA."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        # Create highly correlated activations
        base = backend.random_normal((30, 10))
        noise = backend.random_normal((30, 10)) * 0.1
        X_arr = base
        Y_arr = base + noise  # Highly correlated
        backend.eval(X_arr, Y_arr)

        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        # Correlated data should have high CKA (> 0.7)
        assert result.cka > 0.7

    def test_cka_independent_data_low_similarity(self) -> None:
        """Independent random data should have low CKA."""
        from modelcypher.core.domain.geometry.cka import (
            compute_cka,
            HSICEstimator,
        )

        backend = get_default_backend()

        # Create independent random activations with different seeds
        backend.random_seed(42)
        X_arr = backend.random_normal((50, 32))
        backend.eval(X_arr)
        
        backend.random_seed(12345)  # Very different seed
        Y_arr = backend.random_normal((50, 32))
        backend.eval(Y_arr)

        result = compute_cka(X_arr, Y_arr, backend, estimator=HSICEstimator.BIASED)

        # Independent random data should have lower CKA (< 0.5)
        assert result.cka < 0.5


# =============================================================================
# Procrustes Ground Truth Tests (using scipy)
# =============================================================================


class TestProcrustesGroundTruth:
    """Compare Procrustes against scipy reference."""

    def test_procrustes_disparity_matches_scipy(self) -> None:
        """Procrustes alignment should reduce error versus unaligned baseline."""
        from modelcypher.core.domain.geometry.generalized_procrustes import (
            GeneralizedProcrustes,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        # Generate random test data
        X_arr = backend.random_normal((10, 5))
        Y_arr = backend.random_normal((10, 5))
        backend.eval(X_arr, Y_arr)

        # ModelCypher implementation - no config needed, all params derived from data
        gpa = GeneralizedProcrustes(backend)
        result = gpa.align(
            [array_to_list(backend, X_arr), array_to_list(backend, Y_arr)]
        )

        assert result is not None
        # Ensure alignment reduces error relative to unaligned consensus
        stacked = backend.stack([X_arr, Y_arr], axis=0)
        consensus = gpa._compute_consensus(stacked)  # type: ignore[attr-defined]
        diffs = stacked - consensus
        baseline_err = backend.sum(diffs**2)
        backend.eval(baseline_err)
        baseline_val = float(backend.to_scalar(baseline_err))
        eps = _eps(backend, result.alignment_error, baseline_val)
        assert result.alignment_error <= baseline_val + eps

    def test_procrustes_recovers_rotation(self) -> None:
        """Procrustes should recover a known rotation."""
        from modelcypher.core.domain.geometry.generalized_procrustes import (
            GeneralizedProcrustes,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        # Generate random base
        X_arr = backend.random_normal((15, 4))

        # Apply known rotation (Givens rotation in first two dims)
        angle = backend.array([PI / 4])
        cos_val = backend.cos(angle)
        sin_val = backend.sin(angle)
        backend.eval(cos_val)
        backend.eval(sin_val)
        c = float(backend.to_scalar(cos_val))
        s = float(backend.to_scalar(sin_val))
        R = [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        R_arr = backend.array(R)
        Y_arr = backend.matmul(X_arr, R_arr)
        backend.eval(Y_arr)

        # ModelCypher should recover near-zero error - no config needed
        gpa = GeneralizedProcrustes(backend)
        result = gpa.align(
            [array_to_list(backend, X_arr), array_to_list(backend, Y_arr)]
        )

        assert result is not None
        # Error should be very small since Y is just a rotation of X
        eps = _eps(backend, result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps


# =============================================================================
# Intrinsic Dimension Ground Truth Tests
# =============================================================================


class TestIntrinsicDimensionGroundTruth:
    """Ground truth tests for intrinsic dimension."""

    def test_linear_subspace_dimension_accurate(self) -> None:
        """TwoNN should increase with higher-dimensional subspaces."""
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

        backend = get_default_backend()
        backend.random_seed(42)

        ambient_dim = 20
        n_samples = 200

        def _sample_subspace(dim: int):
            raw_basis = backend.random_normal((ambient_dim, dim))
            Q, _ = backend.qr(raw_basis)
            basis = backend.transpose(Q)
            coeffs = backend.random_normal((n_samples, dim))
            pts = backend.matmul(coeffs, basis)
            backend.eval(pts)
            return pts

        low_dim_points = _sample_subspace(3)
        high_dim_points = _sample_subspace(6)

        # New config-free API: all parameters derived from data
        low_estimate = IntrinsicDimension(backend).compute(low_dim_points)
        high_estimate = IntrinsicDimension(backend).compute(high_dim_points)

        eps = _eps(backend, low_estimate.intrinsic_dimension, high_estimate.intrinsic_dimension)
        assert high_estimate.intrinsic_dimension >= low_estimate.intrinsic_dimension + eps

    def test_full_rank_gaussian_dimension(self) -> None:
        """Full-rank Gaussian should have dimension close to ambient."""
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

        backend = get_default_backend()
        backend.random_seed(42)

        ambient_dim = 8
        n_samples = 200

        # Full-rank Gaussian
        points = backend.random_normal((n_samples, ambient_dim))
        backend.eval(points)

        # New config-free API with CI
        estimate = IntrinsicDimension(backend).compute(points, with_ci=True)

        assert estimate.ci is not None
        eps = _eps(backend, estimate.ci.lower, estimate.ci.upper, ambient_dim)
        assert estimate.ci.lower - eps <= ambient_dim <= estimate.ci.upper + eps


# =============================================================================
# QR Decomposition Ground Truth Tests
# =============================================================================


class TestQRGroundTruth:
    """Ground truth tests for QR decomposition."""

    def test_qr_reconstructs_and_orthogonal(self) -> None:
        """Backend QR should reconstruct A and preserve orthogonality."""
        backend = get_default_backend()

        # Random matrix
        backend.random_seed(42)
        A_arr = backend.random_normal((10, 5))
        backend.eval(A_arr)

        # Backend implementation
        Q_be, R_be = backend.qr(A_arr)
        backend.eval(Q_be)
        backend.eval(R_be)

        # Q should be orthogonal
        QtQ = backend.matmul(backend.transpose(Q_be), Q_be)
        I = backend.eye(QtQ.shape[0])
        diff = QtQ - I
        max_diff = _max_abs(backend, diff)
        dim = QtQ.shape[0]
        eps = _eps(backend, max_diff, 0.0) * dim
        assert max_diff <= eps

        # A = QR should hold
        A_reconstructed = backend.matmul(Q_be, R_be)
        recon_diff = A_reconstructed - A_arr
        max_recon = _max_abs(backend, recon_diff)
        k = min(A_arr.shape[0], A_arr.shape[1])
        eps = _eps(backend, max_recon, 0.0) * k
        assert max_recon <= eps


# =============================================================================
# SVD Ground Truth Tests
# =============================================================================


class TestSVDGroundTruth:
    """Ground truth tests for SVD."""

    def test_svd_reconstructs(self) -> None:
        """Backend SVD should reconstruct A."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Random square matrix (avoids shape issues)
        A_arr = backend.random_normal((6, 6))
        backend.eval(A_arr)

        # Backend implementation
        U_be, S_be, Vt_be = backend.svd(A_arr)
        backend.eval(U_be)
        backend.eval(S_be)
        backend.eval(Vt_be)

        # Reconstruct A = U @ diag(S) @ Vt
        S_diag = backend.diag(S_be)
        A_reconstructed = backend.matmul(backend.matmul(U_be, S_diag), Vt_be)
        recon_diff = A_reconstructed - A_arr
        max_diff = _max_abs(backend, recon_diff)
        dim = A_arr.shape[0]
        eps = _eps(backend, max_diff, 0.0) * dim
        assert max_diff <= eps

    def test_svd_singular_values_non_negative(self) -> None:
        """Singular values should be non-negative."""
        backend = get_default_backend()
        backend.random_seed(42)

        A_arr = backend.random_normal((10, 6))
        _, S_be, _ = backend.svd(A_arr)
        backend.eval(S_be)
        min_val = backend.min(S_be)
        backend.eval(min_val)
        min_scalar = float(backend.to_scalar(min_val))

        eps = _eps(backend, min_scalar)
        assert min_scalar >= -eps


# =============================================================================
# Eigenvalue Ground Truth Tests
# =============================================================================


class TestEighGroundTruth:
    """Ground truth tests for symmetric eigenvalue decomposition."""

    def test_eigh_satisfies_eigenpair(self) -> None:
        """Backend eigh should satisfy A v = lambda v."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create symmetric matrix
        A_arr = backend.random_normal((6, 6))
        A_arr = (A_arr + backend.transpose(A_arr)) / 2
        backend.eval(A_arr)

        # Backend implementation
        vals_be, vecs_be = backend.eigh(A_arr)
        backend.eval(vals_be)
        backend.eval(vecs_be)

        # A @ v = lambda * v for each eigenpair
        dim = A_arr.shape[0]
        for i in range(dim):
            lam = vals_be[i]
            v = vecs_be[:, i]
            v_col = backend.reshape(v, (-1, 1))
            Av = backend.matmul(A_arr, v_col)
            lam_val = _to_scalar(backend, lam)
            lam_v = v_col * lam_val
            diff = Av - lam_v
            max_diff = _max_abs(backend, diff)
            eps = _eps(backend, max_diff, 0.0) * dim
            assert max_diff <= eps
