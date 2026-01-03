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

import pytest

from modelcypher.core.domain.geometry.backend_matrix_utils import (
    BackendMatrixUtils,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.ports.backend import Backend
from tests.conftest import HAS_MLX


def _max_abs(backend: Backend, array) -> float:
    diff = backend.max(backend.abs(array))
    backend.eval(diff)
    return float(backend.to_scalar(diff))


def _max_abs_diff(backend: Backend, left, right) -> float:
    return _max_abs(backend, left - right)


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
        tol = regularization_epsilon(mlx_backend, gram)

        eye = mlx_backend.eye(4)
        diff = _max_abs_diff(mlx_backend, gram, eye)
        assert diff <= tol

    def test_linear_gram_matrix_orthonormal(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Orthonormal rows should give identity-like Gram matrix."""
        X_random = mlx_backend.random_normal((8, 4))
        Q, _ = mlx_backend.qr(X_random)
        X = mlx_backend.transpose(Q)  # 4 orthonormal rows

        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(mlx_backend, gram)

        # Should be close to identity
        eye = mlx_backend.eye(4)
        diff = _max_abs_diff(mlx_backend, gram, eye)
        assert diff <= tol

    def test_gram_matrix_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Gram matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(mlx_backend, gram)

        diff = _max_abs_diff(mlx_backend, gram, mlx_backend.transpose(gram))
        assert diff <= tol

    def test_gram_matrix_positive_semidefinite(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Gram matrix should be positive semi-definite."""
        X = mlx_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(mlx_backend, gram)

        eigenvalues, _ = mlx_backend.eigh(gram)
        min_val = mlx_backend.min(eigenvalues)
        mlx_backend.eval(min_val)
        assert float(mlx_backend.to_scalar(min_val)) >= -tol


class TestCenterMatrix:
    """Tests for matrix centering."""

    def test_centered_matrix_zero_mean(
        self, utils: BackendMatrixUtils, mlx_backend: Backend
    ):
        """Centered matrix should have zero row and column means."""
        K = mlx_backend.random_normal((10, 10))
        # Make symmetric
        K_sym = (K + mlx_backend.transpose(K)) * 0.5

        centered = utils.center_matrix(K)
        tol = regularization_epsilon(mlx_backend, centered)

        row_means = mlx_backend.mean(centered, axis=1)
        row_max = _max_abs(mlx_backend, row_means)
        assert row_max <= tol

        # Column means should be ~0
        col_means = mlx_backend.mean(centered, axis=0)
        col_max = _max_abs(mlx_backend, col_means)
        assert col_max <= tol

    def test_centering_idempotent(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Centering twice should give same result as once."""
        K = mlx_backend.random_normal((8, 8))
        K = (K + mlx_backend.transpose(K)) * 0.5

        centered_once = utils.center_matrix(K)
        centered_twice = utils.center_matrix(centered_once)
        tol = regularization_epsilon(mlx_backend, centered_once)

        diff = _max_abs_diff(mlx_backend, centered_once, centered_twice)
        assert diff <= tol


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_self_distance_zero(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Distance from point to itself should be zero."""
        X = mlx_backend.random_normal((5, 3))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = machine_epsilon(mlx_backend, sq_dists)

        # Diagonal should be zeros
        diag = mlx_backend.diag(sq_dists)
        diff = _max_abs(mlx_backend, diag)
        assert diff <= tol

    def test_distance_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Distance matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = regularization_epsilon(mlx_backend, sq_dists)

        diff = _max_abs_diff(mlx_backend, sq_dists, mlx_backend.transpose(sq_dists))
        assert diff <= tol

    def test_distance_non_negative(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Squared distances should be non-negative."""
        X = mlx_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = machine_epsilon(mlx_backend, sq_dists)

        min_val = mlx_backend.min(sq_dists)
        mlx_backend.eval(min_val)
        assert float(mlx_backend.to_scalar(min_val)) >= -tol

    def test_distance_correct_value(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Verify distance calculation against known values."""
        X = mlx_backend.array([[0.0, 0.0], [3.0, 4.0]])  # Distance should be 5
        sq_dists = utils.pairwise_squared_distances(X)
        dists = utils.pairwise_distances(X)

        tol = regularization_epsilon(mlx_backend, sq_dists)

        # d(0,1) = sqrt(9 + 16) = 5
        sq_val = float(mlx_backend.to_scalar(sq_dists[0, 1]))
        dist_val = float(mlx_backend.to_scalar(dists[0, 1]))
        assert abs(sq_val - 25.0) <= tol
        assert abs(dist_val - 5.0) <= tol


class TestProcrustesRotation:
    """Tests for Procrustes rotation."""

    def test_identity_alignment(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Aligning identical matrices should give identity rotation."""
        X = mlx_backend.random_normal((10, 4))
        result = utils.procrustes_rotation(X, X)

        tol = regularization_epsilon(mlx_backend, result.rotation)
        residual_tol = division_epsilon(mlx_backend, result.rotation)

        # Should be identity (or very close)
        eye = mlx_backend.eye(4)
        diff = _max_abs_diff(mlx_backend, result.rotation, eye)
        assert diff <= tol
        assert result.residual <= residual_tol

    def test_rotation_is_orthogonal(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Procrustes rotation should be orthogonal (R^T R = I)."""
        source = mlx_backend.random_normal((20, 5))
        target = mlx_backend.random_normal((20, 5))

        result = utils.procrustes_rotation(source, target)

        # R^T @ R should be identity (within float32 precision)
        should_be_identity = mlx_backend.matmul(
            mlx_backend.transpose(result.rotation), result.rotation
        )
        eye = mlx_backend.eye(5)
        tol = division_epsilon(mlx_backend, result.rotation) * result.rotation.shape[0]
        diff = _max_abs_diff(mlx_backend, should_be_identity, eye)
        assert diff <= tol

    def test_determinant_positive(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Rotation should have determinant +1 (proper rotation, not reflection)."""
        source = mlx_backend.random_normal((15, 4))
        target = mlx_backend.random_normal((15, 4))

        result = utils.procrustes_rotation(source, target)

        det = mlx_backend.det(result.rotation)
        mlx_backend.eval(det)
        tol = division_epsilon(mlx_backend, result.rotation)
        assert abs(float(mlx_backend.to_scalar(det)) - 1.0) <= tol

    def test_known_rotation(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test with a known 90-degree rotation."""
        R_known = mlx_backend.array([[0.0, -1.0], [1.0, 0.0]])

        # Create source and apply known rotation
        source = mlx_backend.random_normal((10, 2))
        target = mlx_backend.matmul(source, R_known)

        result = utils.procrustes_rotation(source, target)
        tol = regularization_epsilon(mlx_backend, result.rotation)
        residual_tol = division_epsilon(mlx_backend, result.rotation)

        diff = _max_abs_diff(mlx_backend, result.rotation, R_known)
        assert diff <= tol
        assert result.residual <= residual_tol


class TestProcrustesAlign:
    """Tests for full Procrustes alignment."""

    def test_align_reduces_residual(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Alignment should reduce the Frobenius distance."""
        source = mlx_backend.random_normal((15, 4))
        target = mlx_backend.random_normal((15, 4))

        # Distance before alignment
        diff_before = target - source
        before = mlx_backend.sum(diff_before * diff_before)
        mlx_backend.eval(before)
        before_val = float(mlx_backend.to_scalar(before))

        # Align
        aligned, result = utils.procrustes_align(source, target, center=True)

        # Distance after alignment
        diff_after = target - aligned
        after = mlx_backend.sum(diff_after * diff_after)
        mlx_backend.eval(after)
        after_val = float(mlx_backend.to_scalar(after))

        eps = machine_epsilon(mlx_backend, target) * target.shape[0]
        assert after_val <= before_val + eps

    def test_align_with_scaling(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test alignment with scaling enabled."""
        source = mlx_backend.random_normal((10, 3))
        scale_factor = 1.0 + division_epsilon(mlx_backend, source)
        target = source * scale_factor

        _, result = utils.procrustes_align(source, target, allow_scaling=True)

        tol = division_epsilon(mlx_backend, source)
        assert abs(result.scale - scale_factor) <= tol


class TestCosineSimilarityMatrix:
    """Tests for cosine similarity matrix."""

    def test_diagonal_ones(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Diagonal should be 1 (self-similarity)."""
        X = mlx_backend.random_normal((8, 4))
        sim = utils.cosine_similarity_matrix(X)
        tol = regularization_epsilon(mlx_backend, sim)

        diag = mlx_backend.diag(sim)
        diff = _max_abs(mlx_backend, diag - 1.0)
        assert diff <= tol

    def test_symmetric(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Cosine similarity matrix should be symmetric."""
        X = mlx_backend.random_normal((10, 5))
        sim = utils.cosine_similarity_matrix(X)
        tol = regularization_epsilon(mlx_backend, sim)

        diff = _max_abs_diff(mlx_backend, sim, mlx_backend.transpose(sim))
        assert diff <= tol

    def test_range_bounded(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Cosine similarity should be in [-1, 1]."""
        X = mlx_backend.random_normal((15, 6))
        sim = utils.cosine_similarity_matrix(X)
        tol = division_epsilon(mlx_backend, sim)

        min_val = mlx_backend.min(sim)
        max_val = mlx_backend.max(sim)
        mlx_backend.eval(min_val, max_val)
        assert float(mlx_backend.to_scalar(min_val)) >= -1.0 - tol
        assert float(mlx_backend.to_scalar(max_val)) <= 1.0 + tol


class TestEffectiveRank:
    """Tests for effective rank computation."""

    def test_full_rank(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Matrix with equal eigenvalues should have full rank."""
        # Equal eigenvalues = uniform variance
        eigenvalues = mlx_backend.array([1.0, 1.0, 1.0, 1.0])

        rank = utils.effective_rank(eigenvalues)
        assert rank == 4

    def test_single_dominant(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """One dominant eigenvalue should give rank 1."""
        eigenvalues = mlx_backend.array([100.0, 0.1, 0.1, 0.1])

        rank = utils.effective_rank(eigenvalues)
        assert rank == 1

    def test_entropy_effective_rank(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test entropy-based effective rank."""
        # Equal eigenvalues: entropy-based rank should equal dimension
        eigenvalues = mlx_backend.array([1.0, 1.0, 1.0, 1.0])

        erank = utils.entropy_effective_rank(eigenvalues)
        tol = division_epsilon(mlx_backend, eigenvalues)
        assert abs(erank - 4.0) <= tol

        # Single eigenvalue: entropy rank should be 1
        eigenvalues_single = mlx_backend.array([1.0, 0.0, 0.0, 0.0])
        erank_single = utils.entropy_effective_rank(eigenvalues_single)
        assert abs(erank_single - 1.0) <= tol


class TestEigendecomposition:
    """Tests for eigendecomposition."""

    def test_symmetric_matrix(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Test eigendecomposition of symmetric matrix."""
        # Create symmetric matrix
        A = mlx_backend.random_normal((5, 5))
        A = (A + mlx_backend.transpose(A)) * 0.5

        eigenvalues, eigenvectors = utils.eigendecomposition(A)

        tol = regularization_epsilon(mlx_backend, A)

        # Verify: A @ V = V @ diag(eigenvalues)
        AV = mlx_backend.matmul(A, eigenvectors)
        VD = mlx_backend.matmul(eigenvectors, mlx_backend.diag(eigenvalues))
        diff = _max_abs_diff(mlx_backend, AV, VD)
        assert diff <= tol


class TestTrace:
    """Tests for trace computation."""

    def test_identity_trace(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Trace of identity should equal dimension."""
        I = mlx_backend.eye(5)
        trace = utils.trace(I)
        tol = division_epsilon(mlx_backend, I)
        assert abs(trace - 5.0) <= tol

    def test_trace_matches_diagonal_sum(self, utils: BackendMatrixUtils, mlx_backend: Backend):
        """Trace should match diagonal sum."""
        A = mlx_backend.random_normal((6, 6))
        trace = utils.trace(A)
        diag = mlx_backend.diag(A)
        expected_arr = mlx_backend.sum(diag)
        mlx_backend.eval(expected_arr)
        expected = float(mlx_backend.to_scalar(expected_arr))

        tol = division_epsilon(mlx_backend, A)
        assert abs(trace - expected) <= tol


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
        tol = regularization_epsilon(mlx_backend, gram)
        diff = _max_abs_diff(mlx_backend, gram, mlx_backend.transpose(gram))
        assert diff <= tol

    def test_procrustes_rotation_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify Procrustes rotation works on MLX."""
        source = mlx_backend.random_normal((20, 4))
        target = mlx_backend.random_normal((20, 4))

        result = mlx_utils.procrustes_rotation(source, target)

        # Rotation should be orthogonal
        should_be_identity = mlx_backend.matmul(
            mlx_backend.transpose(result.rotation), result.rotation
        )
        eye = mlx_backend.eye(4)
        tol = division_epsilon(mlx_backend, result.rotation) * result.rotation.shape[0]
        diff = _max_abs_diff(mlx_backend, should_be_identity, eye)
        assert diff <= tol

    def test_pairwise_distances_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify pairwise distances work on MLX."""
        X = mlx_backend.random_normal((15, 6))
        dists = mlx_utils.pairwise_distances(X)

        # Should be symmetric and non-negative
        tol = regularization_epsilon(mlx_backend, dists)
        diff = _max_abs_diff(mlx_backend, dists, mlx_backend.transpose(dists))
        assert diff <= tol
        min_val = mlx_backend.min(dists)
        mlx_backend.eval(min_val)
        assert float(mlx_backend.to_scalar(min_val)) >= 0.0

    def test_cosine_similarity_matrix_mlx(self, mlx_utils: BackendMatrixUtils, mlx_backend):
        """Verify cosine similarity works on MLX."""
        X = mlx_backend.random_normal((12, 8))
        sim = mlx_utils.cosine_similarity_matrix(X)

        # Diagonal should be 1
        tol = regularization_epsilon(mlx_backend, sim)
        diag = mlx_backend.diag(sim)
        diff = _max_abs(mlx_backend, diag - 1.0)
        assert diff <= tol
