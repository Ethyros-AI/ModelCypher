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
from modelcypher.core.domain.geometry.cka import _center_gram_matrix
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.ports.backend import Backend


def _max_abs(backend: Backend, array) -> float:
    diff = backend.max(backend.abs(array))
    backend.eval(diff)
    return float(backend.to_scalar(diff))


def _max_abs_diff(backend: Backend, left, right) -> float:
    return _max_abs(backend, left - right)


@pytest.fixture
def utils(any_backend: Backend) -> BackendMatrixUtils:
    """Create BackendMatrixUtils with the provided backend."""
    return BackendMatrixUtils(any_backend)


class TestGramMatrix:
    """Tests for Gram matrix computation."""

    def test_linear_gram_matrix_identity(
        self, utils: BackendMatrixUtils, any_backend: Backend
    ):
        """Identity matrix should give identity Gram matrix."""
        X = any_backend.eye(4)
        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(any_backend, gram)

        eye = any_backend.eye(4)
        diff = _max_abs_diff(any_backend, gram, eye)
        assert diff <= tol

    def test_linear_gram_matrix_orthonormal(
        self, utils: BackendMatrixUtils, any_backend: Backend
    ):
        """Orthonormal rows should give identity-like Gram matrix."""
        X_random = any_backend.random_normal((8, 4))
        Q, _ = any_backend.qr(X_random)
        X = any_backend.transpose(Q)  # 4 orthonormal rows

        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(any_backend, gram)

        # Should be close to identity
        eye = any_backend.eye(4)
        diff = _max_abs_diff(any_backend, gram, eye)
        assert diff <= tol

    def test_gram_matrix_symmetric(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Gram matrix should be symmetric."""
        X = any_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(any_backend, gram)

        diff = _max_abs_diff(any_backend, gram, any_backend.transpose(gram))
        assert diff <= tol

    def test_gram_matrix_positive_semidefinite(
        self, utils: BackendMatrixUtils, any_backend: Backend
    ):
        """Gram matrix should be positive semi-definite."""
        X = any_backend.random_normal((10, 5))
        gram = utils.compute_gram_matrix(X, kernel="linear")
        tol = regularization_epsilon(any_backend, gram)

        eigenvalues, _ = any_backend.eigh(gram)
        min_val = any_backend.min(eigenvalues)
        any_backend.eval(min_val)
        assert float(any_backend.to_scalar(min_val)) >= -tol


class TestCenterMatrix:
    """Tests for matrix centering (canonical implementation in cka.py)."""

    def test_centered_matrix_zero_mean(
        self, utils: BackendMatrixUtils, any_backend: Backend
    ):
        """Centered matrix should have zero row and column means."""
        K = any_backend.random_normal((10, 10))
        # Make symmetric
        (K + any_backend.transpose(K)) * 0.5

        centered = _center_gram_matrix(K, any_backend)
        tol = regularization_epsilon(any_backend, centered)

        row_means = any_backend.mean(centered, axis=1)
        row_max = _max_abs(any_backend, row_means)
        assert row_max <= tol

        # Column means should be ~0
        col_means = any_backend.mean(centered, axis=0)
        col_max = _max_abs(any_backend, col_means)
        assert col_max <= tol

    def test_centering_idempotent(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Centering twice should give same result as once."""
        K = any_backend.random_normal((8, 8))
        K = (K + any_backend.transpose(K)) * 0.5

        centered_once = _center_gram_matrix(K, any_backend)
        centered_twice = _center_gram_matrix(centered_once, any_backend)
        tol = regularization_epsilon(any_backend, centered_once)

        diff = _max_abs_diff(any_backend, centered_once, centered_twice)
        assert diff <= tol


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_self_distance_zero(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Distance from point to itself should be zero."""
        X = any_backend.random_normal((5, 3))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = machine_epsilon(any_backend, sq_dists)

        # Diagonal should be zeros
        diag = any_backend.diag(sq_dists)
        diff = _max_abs(any_backend, diag)
        assert diff <= tol

    def test_distance_symmetric(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Distance matrix should be symmetric."""
        X = any_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = regularization_epsilon(any_backend, sq_dists)

        diff = _max_abs_diff(any_backend, sq_dists, any_backend.transpose(sq_dists))
        assert diff <= tol

    def test_distance_non_negative(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Squared distances should be non-negative."""
        X = any_backend.random_normal((10, 4))
        sq_dists = utils.pairwise_squared_distances(X)
        tol = machine_epsilon(any_backend, sq_dists)

        min_val = any_backend.min(sq_dists)
        any_backend.eval(min_val)
        assert float(any_backend.to_scalar(min_val)) >= -tol

    def test_distance_correct_value(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Verify distance calculation against known values."""
        X = any_backend.array([[0.0, 0.0], [3.0, 4.0]])  # Distance should be 5
        sq_dists = utils.pairwise_squared_distances(X)
        dists = utils.pairwise_distances(X)

        tol = regularization_epsilon(any_backend, sq_dists)

        # d(0,1) = sqrt(9 + 16) = 5
        sq_val = float(any_backend.to_scalar(sq_dists[0, 1]))
        dist_val = float(any_backend.to_scalar(dists[0, 1]))
        assert abs(sq_val - 25.0) <= tol
        assert abs(dist_val - 5.0) <= tol


class TestProcrustesRotation:
    """Tests for Procrustes rotation."""

    def test_identity_alignment(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Aligning identical matrices should give identity rotation."""
        X = any_backend.random_normal((10, 4))
        result = utils.procrustes_rotation(X, X)

        tol = regularization_epsilon(any_backend, result.rotation)
        residual_tol = division_epsilon(any_backend, result.rotation)

        # Should be identity (or very close)
        eye = any_backend.eye(4)
        diff = _max_abs_diff(any_backend, result.rotation, eye)
        assert diff <= tol
        assert result.residual <= residual_tol

    def test_rotation_is_orthogonal(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Procrustes rotation should be orthogonal (R^T R = I)."""
        source = any_backend.random_normal((20, 5))
        target = any_backend.random_normal((20, 5))

        result = utils.procrustes_rotation(source, target)

        # R^T @ R should be identity (within float32 precision)
        should_be_identity = any_backend.matmul(
            any_backend.transpose(result.rotation), result.rotation
        )
        eye = any_backend.eye(5)
        tol = division_epsilon(any_backend, result.rotation) * result.rotation.shape[0]
        diff = _max_abs_diff(any_backend, should_be_identity, eye)
        assert diff <= tol

    def test_determinant_positive(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Rotation should have determinant +1 (proper rotation, not reflection)."""
        source = any_backend.random_normal((15, 4))
        target = any_backend.random_normal((15, 4))

        result = utils.procrustes_rotation(source, target)

        det = any_backend.det(result.rotation)
        any_backend.eval(det)
        tol = division_epsilon(any_backend, result.rotation)
        assert abs(float(any_backend.to_scalar(det)) - 1.0) <= tol

    def test_known_rotation(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Test with a known 90-degree rotation."""
        R_known = any_backend.array([[0.0, -1.0], [1.0, 0.0]])

        # Create source and apply known rotation
        source = any_backend.random_normal((10, 2))
        target = any_backend.matmul(source, R_known)

        result = utils.procrustes_rotation(source, target)
        tol = regularization_epsilon(any_backend, result.rotation)
        residual_tol = division_epsilon(any_backend, result.rotation)

        diff = _max_abs_diff(any_backend, result.rotation, R_known)
        assert diff <= tol
        assert result.residual <= residual_tol


class TestProcrustesAlign:
    """Tests for full Procrustes alignment."""

    def test_align_reduces_residual(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Alignment should reduce the Frobenius distance."""
        source = any_backend.random_normal((15, 4))
        target = any_backend.random_normal((15, 4))

        # Distance before alignment
        diff_before = target - source
        before = any_backend.sum(diff_before * diff_before)
        any_backend.eval(before)
        before_val = float(any_backend.to_scalar(before))

        # Align
        aligned, result = utils.procrustes_align(source, target, center=True)

        # Distance after alignment
        diff_after = target - aligned
        after = any_backend.sum(diff_after * diff_after)
        any_backend.eval(after)
        after_val = float(any_backend.to_scalar(after))

        eps = machine_epsilon(any_backend, target) * target.shape[0]
        assert after_val <= before_val + eps

    def test_align_with_scaling(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Test alignment with scaling enabled."""
        source = any_backend.random_normal((10, 3))
        scale_factor = 1.0 + division_epsilon(any_backend, source)
        target = source * scale_factor

        _, result = utils.procrustes_align(source, target, allow_scaling=True)

        tol = division_epsilon(any_backend, source)
        assert abs(result.scale - scale_factor) <= tol


class TestCosineSimilarityMatrix:
    """Tests for cosine similarity matrix."""

    def test_diagonal_ones(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Diagonal should be 1 (self-similarity)."""
        X = any_backend.random_normal((8, 4))
        sim = utils.cosine_similarity_matrix(X)
        tol = regularization_epsilon(any_backend, sim)

        diag = any_backend.diag(sim)
        diff = _max_abs(any_backend, diag - 1.0)
        assert diff <= tol

    def test_symmetric(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Cosine similarity matrix should be symmetric."""
        X = any_backend.random_normal((10, 5))
        sim = utils.cosine_similarity_matrix(X)
        tol = regularization_epsilon(any_backend, sim)

        diff = _max_abs_diff(any_backend, sim, any_backend.transpose(sim))
        assert diff <= tol

    def test_range_bounded(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Cosine similarity should be in [-1, 1]."""
        X = any_backend.random_normal((15, 6))
        sim = utils.cosine_similarity_matrix(X)
        tol = division_epsilon(any_backend, sim)

        min_val = any_backend.min(sim)
        max_val = any_backend.max(sim)
        any_backend.eval(min_val, max_val)
        assert float(any_backend.to_scalar(min_val)) >= -1.0 - tol
        assert float(any_backend.to_scalar(max_val)) <= 1.0 + tol


class TestEffectiveRank:
    """Tests for effective rank computation."""

    def test_full_rank(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Matrix with equal eigenvalues should have full rank."""
        # Equal eigenvalues = uniform variance
        eigenvalues = any_backend.array([1.0, 1.0, 1.0, 1.0])

        rank = utils.effective_rank(eigenvalues)
        assert rank == 4

    def test_single_dominant(self, utils: BackendMatrixUtils, any_backend: Backend):
        """One dominant eigenvalue should give rank 1."""
        eigenvalues = any_backend.array([100.0, 0.1, 0.1, 0.1])

        rank = utils.effective_rank(eigenvalues)
        assert rank == 1


class TestEigendecomposition:
    """Tests for eigendecomposition."""

    def test_symmetric_matrix(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Test eigendecomposition of symmetric matrix."""
        # Create symmetric matrix
        A = any_backend.random_normal((5, 5))
        A = (A + any_backend.transpose(A)) * 0.5

        eigenvalues, eigenvectors = utils.eigendecomposition(A)

        tol = regularization_epsilon(any_backend, A)

        # Verify: A @ V = V @ diag(eigenvalues)
        AV = any_backend.matmul(A, eigenvectors)
        VD = any_backend.matmul(eigenvectors, any_backend.diag(eigenvalues))
        diff = _max_abs_diff(any_backend, AV, VD)
        assert diff <= tol


class TestTrace:
    """Tests for trace computation."""

    def test_identity_trace(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Trace of identity should equal dimension."""
        I = any_backend.eye(5)
        trace = utils.trace(I)
        tol = division_epsilon(any_backend, I)
        assert abs(trace - 5.0) <= tol

    def test_trace_matches_diagonal_sum(self, utils: BackendMatrixUtils, any_backend: Backend):
        """Trace should match diagonal sum."""
        A = any_backend.random_normal((6, 6))
        trace = utils.trace(A)
        diag = any_backend.diag(A)
        expected_arr = any_backend.sum(diag)
        any_backend.eval(expected_arr)
        expected = float(any_backend.to_scalar(expected_arr))

        tol = division_epsilon(any_backend, A)
        assert abs(trace - expected) <= tol


# =============================================================================
# Backend-Specific Tests (run on available backend)
# =============================================================================


@pytest.fixture
def backend_utils(any_backend) -> BackendMatrixUtils:
    """Create BackendMatrixUtils with any available backend."""
    return BackendMatrixUtils(any_backend)


class TestBackendMatrixUtils:
    """Tests that run on available backend to verify hardware acceleration."""

    def test_gram_matrix(self, backend_utils: BackendMatrixUtils, any_backend):
        """Verify Gram matrix works on backend."""
        X = any_backend.random_normal((10, 5))
        gram = backend_utils.compute_gram_matrix(X, kernel="linear")

        # Should be symmetric
        tol = regularization_epsilon(any_backend, gram)
        diff = _max_abs_diff(any_backend, gram, any_backend.transpose(gram))
        assert diff <= tol

    def test_procrustes_rotation(self, backend_utils: BackendMatrixUtils, any_backend):
        """Verify Procrustes rotation works on backend."""
        source = any_backend.random_normal((20, 4))
        target = any_backend.random_normal((20, 4))

        result = backend_utils.procrustes_rotation(source, target)

        # Rotation should be orthogonal
        should_be_identity = any_backend.matmul(
            any_backend.transpose(result.rotation), result.rotation
        )
        eye = any_backend.eye(4)
        tol = division_epsilon(any_backend, result.rotation) * result.rotation.shape[0]
        diff = _max_abs_diff(any_backend, should_be_identity, eye)
        assert diff <= tol

    def test_pairwise_distances(self, backend_utils: BackendMatrixUtils, any_backend):
        """Verify pairwise distances work on backend."""
        X = any_backend.random_normal((15, 6))
        dists = backend_utils.pairwise_distances(X)

        # Should be symmetric and non-negative
        tol = regularization_epsilon(any_backend, dists)
        diff = _max_abs_diff(any_backend, dists, any_backend.transpose(dists))
        assert diff <= tol
        min_val = any_backend.min(dists)
        any_backend.eval(min_val)
        assert float(any_backend.to_scalar(min_val)) >= 0.0

    def test_cosine_similarity_matrix(self, backend_utils: BackendMatrixUtils, any_backend):
        """Verify cosine similarity works on backend."""
        X = any_backend.random_normal((12, 8))
        sim = backend_utils.cosine_similarity_matrix(X)

        # Diagonal should be 1
        tol = regularization_epsilon(any_backend, sim)
        diag = any_backend.diag(sim)
        diff = _max_abs(any_backend, diag - 1.0)
        assert diff <= tol
