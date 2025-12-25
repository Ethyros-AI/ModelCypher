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

"""Tests for Backend-aware matrix utilities."""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.geometry.backend_matrix_utils import (
    BackendMatrixUtils,
)
from modelcypher.ports.backend import Backend
from tests.conftest import HAS_MLX


@pytest.fixture
def mlx_backend() -> Backend:
    """Provide MLXBackend for GPU-accelerated testing."""
    if not HAS_MLX:
        pytest.skip("MLX not available")
    from modelcypher.backends.mlx_backend import MLXBackend
    return MLXBackend()


@pytest.fixture
def utils(mlx_backend: Backend) -> BackendMatrixUtils:
    """Create BackendMatrixUtils with MLXBackend."""
    return BackendMatrixUtils(mlx_backend)


class TestGramMatrix:
    """Tests for Gram matrix computation."""

    def test_linear_gram_matrix_identity(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Identity matrix should give identity Gram matrix."""
        X = mlx_backend.eye(4)
        gram = utils.compute_gram_matrix(X, kernel="linear")
        gram_np = mlx_backend.to_numpy(gram)

        np.testing.assert_allclose(gram_np, np.eye(4), rtol=1e-5)

    def test_linear_gram_matrix_orthonormal(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Orthonormal rows should give identity-like Gram matrix."""
        # Create orthonormal matrix via QR
        X_random = mlx_backend.random_normal((4, 8))
        Q, _ = np.linalg.qr(mlx_backend.to_numpy(X_random).T)
        X = mlx_backend.array(Q.T)  # 4 orthonormal rows

        gram = utils.compute_gram_matrix(X, kernel="linear")
        gram_np = mlx_backend.to_numpy(gram)

        # Should be close to identity
        np.testing.assert_allclose(gram_np, np.eye(4), atol=1e-5)

    def test_gram_matrix_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Gram matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        gram_np = mlx_backend.to_numpy(gram)

        np.testing.assert_allclose(gram_np, gram_np.T, rtol=1e-5)

    def test_gram_matrix_positive_semidefinite(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Gram matrix should be positive semi-definite."""
        X = mlx_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        gram_np = mlx_backend.to_numpy(gram)

        eigenvalues = np.linalg.eigvalsh(gram_np)
        assert np.all(eigenvalues >= -1e-10)


class TestCenterMatrix:
    """Tests for matrix centering."""

    def test_centered_matrix_zero_mean(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Centered matrix should have zero row and column means."""
        K = mlx_backend.random_normal((10, 10))
        # Make symmetric
        K_np = mlx_backend.to_numpy(K)
        K_sym = (K_np + K_np.T) / 2
        K = mlx_backend.array(K_sym)

        centered = utils.center_matrix(K)
        centered_np = mlx_backend.to_numpy(centered)

        # Row means should be ~0
        row_means = np.mean(centered_np, axis=1)
        np.testing.assert_allclose(row_means, 0, atol=1e-10)

        # Column means should be ~0
        col_means = np.mean(centered_np, axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-10)

    def test_centering_idempotent(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Centering twice should give same result as once."""
        K = mlx_backend.random_normal((8, 8))
        K_np = mlx_backend.to_numpy(K)
        K_sym = (K_np + K_np.T) / 2
        K = mlx_backend.array(K_sym)

        centered_once = utils.center_matrix(K)
        centered_twice = utils.center_matrix(centered_once)

        np.testing.assert_allclose(
            mlx_backend.to_numpy(centered_once),
            mlx_backend.to_numpy(centered_twice),
            rtol=1e-5,
        )


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_self_distance_zero(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Distance from point to itself should be zero."""
        X = mlx_backend.random_normal((5, 3))
        sq_dists = utils.pairwise_squared_distances(X)
        sq_dists_np = mlx_backend.to_numpy(sq_dists)

        # Diagonal should be zeros
        np.testing.assert_allclose(np.diag(sq_dists_np), 0, atol=1e-10)

    def test_distance_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Distance matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        sq_dists_np = mlx_backend.to_numpy(sq_dists)

        np.testing.assert_allclose(sq_dists_np, sq_dists_np.T, rtol=1e-5)

    def test_distance_non_negative(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Squared distances should be non-negative."""
        X = mlx_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        sq_dists_np = mlx_backend.to_numpy(sq_dists)

        assert np.all(sq_dists_np >= 0)

    def test_distance_correct_value(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Verify distance calculation against known values."""
        X = mlx_backend.array([[0.0, 0.0], [3.0, 4.0]])  # Distance should be 5
        sq_dists = utils.pairwise_squared_distances(X)
        dists = utils.pairwise_distances(X)

        sq_dists_np = mlx_backend.to_numpy(sq_dists)
        dists_np = mlx_backend.to_numpy(dists)

        # d(0,1) = sqrt(9 + 16) = 5
        np.testing.assert_allclose(sq_dists_np[0, 1], 25.0, rtol=1e-5)
        np.testing.assert_allclose(dists_np[0, 1], 5.0, rtol=1e-5)


class TestProcrustesRotation:
    """Tests for Procrustes rotation."""

    def test_identity_alignment(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Aligning identical matrices should give identity rotation."""
        X = mlx_backend.random_normal((10, 4))
        result = utils.procrustes_rotation(X, X)

        R_np = mlx_backend.to_numpy(result.rotation)

        # Should be identity (or very close)
        np.testing.assert_allclose(R_np, np.eye(4), atol=1e-5)
        assert result.residual < 1e-10

    def test_rotation_is_orthogonal(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Procrustes rotation should be orthogonal (R^T R = I)."""
        source = mlx_backend.random_normal((20, 5))
        target = mlx_backend.random_normal((20, 5))

        result = utils.procrustes_rotation(source, target)
        R_np = mlx_backend.to_numpy(result.rotation)

        # R^T @ R should be identity
        should_be_identity = R_np.T @ R_np
        np.testing.assert_allclose(should_be_identity, np.eye(5), atol=1e-5)

    def test_determinant_positive(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Rotation should have determinant +1 (proper rotation, not reflection)."""
        source = mlx_backend.random_normal((15, 4))
        target = mlx_backend.random_normal((15, 4))

        result = utils.procrustes_rotation(source, target)
        R_np = mlx_backend.to_numpy(result.rotation)

        det = np.linalg.det(R_np)
        np.testing.assert_allclose(det, 1.0, atol=1e-5)

    def test_known_rotation(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test with a known 90-degree rotation."""
        # 2D 90-degree rotation matrix
        theta = np.pi / 2
        R_known = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Create source and apply known rotation
        source_np = np.random.randn(10, 2)
        target_np = source_np @ R_known

        source = mlx_backend.array(source_np)
        target = mlx_backend.array(target_np)

        result = utils.procrustes_rotation(source, target)
        R_np = mlx_backend.to_numpy(result.rotation)

        np.testing.assert_allclose(R_np, R_known, atol=1e-5)
        assert result.residual < 1e-10


class TestProcrustesAlign:
    """Tests for full Procrustes alignment."""

    def test_align_reduces_residual(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Alignment should reduce the Frobenius distance."""
        source = mlx_backend.random_normal((15, 4))
        target = mlx_backend.random_normal((15, 4))

        source_np = mlx_backend.to_numpy(source)
        target_np = mlx_backend.to_numpy(target)

        # Distance before alignment
        before = np.sum((target_np - source_np) ** 2)

        # Align
        aligned, result = utils.procrustes_align(source, target, center=True)
        aligned_np = mlx_backend.to_numpy(aligned)

        # Distance after alignment
        after = np.sum((target_np - aligned_np) ** 2)

        assert after <= before

    def test_align_with_scaling(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test alignment with scaling enabled."""
        source_np = np.random.randn(10, 3)
        target_np = source_np * 2.5  # Scaled version

        source = mlx_backend.array(source_np)
        target = mlx_backend.array(target_np)

        _, result = utils.procrustes_align(source, target, allow_scaling=True)

        # Scale should be close to 2.5
        np.testing.assert_allclose(result.scale, 2.5, rtol=0.1)


class TestCosineSimilarityMatrix:
    """Tests for cosine similarity matrix."""

    def test_diagonal_ones(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Diagonal should be 1 (self-similarity)."""
        X = mlx_backend.random_normal((8, 4))
        sim = utils.cosine_similarity_matrix(X)
        sim_np = mlx_backend.to_numpy(sim)

        np.testing.assert_allclose(np.diag(sim_np), 1.0, atol=1e-5)

    def test_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Cosine similarity matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 5))
        sim = utils.cosine_similarity_matrix(X)
        sim_np = mlx_backend.to_numpy(sim)

        np.testing.assert_allclose(sim_np, sim_np.T, rtol=1e-5)

    def test_range_bounded(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Cosine similarity should be in [-1, 1]."""
        X = mlx_backend.random_normal((15, 6))
        sim = utils.cosine_similarity_matrix(X)
        sim_np = mlx_backend.to_numpy(sim)

        assert np.all(sim_np >= -1.0 - 1e-5)
        assert np.all(sim_np <= 1.0 + 1e-5)


class TestEffectiveRank:
    """Tests for effective rank computation."""

    def test_full_rank(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Matrix with equal eigenvalues should have full rank."""
        # Equal eigenvalues = uniform variance
        eigenvalues = mlx_backend.array([1.0, 1.0, 1.0, 1.0])

        rank = utils.effective_rank(eigenvalues, variance_threshold=0.95)
        assert rank == 4  # All components needed for 95%

    def test_single_dominant(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """One dominant eigenvalue should give rank 1."""
        eigenvalues = mlx_backend.array([100.0, 0.1, 0.1, 0.1])

        rank = utils.effective_rank(eigenvalues, variance_threshold=0.95)
        assert rank == 1

    def test_entropy_effective_rank(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test entropy-based effective rank."""
        # Equal eigenvalues: entropy-based rank should equal dimension
        eigenvalues = mlx_backend.array([1.0, 1.0, 1.0, 1.0])

        erank = utils.entropy_effective_rank(eigenvalues)
        np.testing.assert_allclose(erank, 4.0, rtol=1e-5)

        # Single eigenvalue: entropy rank should be 1
        eigenvalues_single = mlx_backend.array([1.0, 0.0, 0.0, 0.0])
        erank_single = utils.entropy_effective_rank(eigenvalues_single)
        np.testing.assert_allclose(erank_single, 1.0, rtol=1e-5)


class TestEigendecomposition:
    """Tests for eigendecomposition."""

    def test_symmetric_matrix(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test eigendecomposition of symmetric matrix."""
        # Create symmetric matrix
        A = mlx_backend.random_normal((5, 5))
        A_np = mlx_backend.to_numpy(A)
        A_sym = (A_np + A_np.T) / 2
        A = mlx_backend.array(A_sym)

        eigenvalues, eigenvectors = utils.eigendecomposition(A)

        eig_np = mlx_backend.to_numpy(eigenvalues)
        vec_np = mlx_backend.to_numpy(eigenvectors)

        # Verify: A @ V = V @ diag(eigenvalues)
        AV = A_sym @ vec_np
        VD = vec_np @ np.diag(eig_np)
        np.testing.assert_allclose(AV, VD, atol=1e-5)


class TestTrace:
    """Tests for trace computation."""

    def test_identity_trace(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Trace of identity should equal dimension."""
        I = mlx_backend.eye(5)
        trace = utils.trace(I)
        np.testing.assert_allclose(trace, 5.0, rtol=1e-5)

    def test_trace_matches_numpy(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Trace should match numpy computation."""
        A = mlx_backend.random_normal((6, 6))
        A_np = mlx_backend.to_numpy(A)

        trace = utils.trace(A)
        expected = np.trace(A_np)

        np.testing.assert_allclose(trace, expected, rtol=1e-5)


# =============================================================================
# MLX Backend Tests (run on Apple Silicon)
# =============================================================================


@pytest.fixture
def mlx_utils(mlx_backend) -> BackendMatrixUtils:
    """Create BackendMatrixUtils with MLXBackend."""
    return BackendMatrixUtils(mlx_backend)


@pytest.mark.mlx
class TestMLXBackendMatrixUtils:
    """Tests that run on MLX backend to verify hardware acceleration."""

    def test_gram_matrix_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify Gram matrix works on MLX."""
        X = mlx_backend.random_normal((10, 5))
        gram = mlx_utils.compute_gram_matrix(X, kernel="linear")

        # Should be symmetric
        gram_np = mlx_backend.to_numpy(gram)
        np.testing.assert_allclose(gram_np, gram_np.T, rtol=1e-4)

    def test_procrustes_rotation_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify Procrustes rotation works on MLX."""
        source = mlx_backend.random_normal((20, 4))
        target = mlx_backend.random_normal((20, 4))

        result = mlx_utils.procrustes_rotation(source, target)

        # Rotation should be orthogonal
        R_np = mlx_backend.to_numpy(result.rotation)
        should_be_identity = R_np.T @ R_np
        np.testing.assert_allclose(should_be_identity, np.eye(4), atol=1e-4)

    def test_pairwise_distances_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify pairwise distances work on MLX."""
        X = mlx_backend.random_normal((15, 6))
        dists = mlx_utils.pairwise_distances(X)
        dists_np = mlx_backend.to_numpy(dists)

        # Should be symmetric and non-negative
        np.testing.assert_allclose(dists_np, dists_np.T, rtol=1e-4)
        assert np.all(dists_np >= 0)

    def test_cosine_similarity_matrix_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify cosine similarity works on MLX."""
        X = mlx_backend.random_normal((12, 8))
        sim = mlx_utils.cosine_similarity_matrix(X)
        sim_np = mlx_backend.to_numpy(sim)

        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(sim_np), 1.0, atol=1e-4)
