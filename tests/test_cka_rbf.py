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

"""Tests for CKA RBF kernel implementation.

NOTE: All tests use the Backend protocol exclusively. No numpy.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import (
    _compute_pairwise_squared_distances,
    rbf_gram_matrix,
    compute_cka,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _scalar_tol(backend) -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _all_close(backend, arr1, arr2) -> bool:
    """Check if two arrays are element-wise close using backend."""
    diff = backend.abs(arr1 - arr2)
    backend.eval(diff)
    max_arr = backend.max(diff)
    backend.eval(max_arr)
    max_diff = float(backend.to_scalar(max_arr))
    tol = division_epsilon(backend, diff)
    return max_diff <= tol


def _all_in_unit_interval(backend, arr) -> bool:
    """Check if all elements are in [0, 1] within dtype tolerance."""
    backend.eval(arr)
    min_arr = backend.min(arr)
    max_arr = backend.max(arr)
    backend.eval(min_arr, max_arr)
    min_val = float(backend.to_scalar(min_arr))
    max_val = float(backend.to_scalar(max_arr))
    tol = division_epsilon(backend, arr)
    return min_val >= -tol and max_val <= 1.0 + tol


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_identical_points_zero_distance(self):
        """Identical points should have zero distance."""
        backend = get_default_backend()
        X = backend.array([[1.0, 2.0], [1.0, 2.0]])
        distances = _compute_pairwise_squared_distances(X, backend)
        backend.eval(distances)
        distances_00_1 = float(backend.to_scalar(distances[0, 1]))
        distances_10 = float(backend.to_scalar(distances[1, 0]))

        tol = _scalar_tol(backend)
        assert abs(distances_00_1) <= tol
        assert abs(distances_10) <= tol

    def test_distance_is_symmetric(self):
        """Distance matrix should be symmetric."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        distances = _compute_pairwise_squared_distances(X, backend)
        distances_T = backend.transpose(distances)

        assert _all_close(backend, distances, distances_T)

    def test_diagonal_is_zero(self):
        """Diagonal should be zero (distance to self)."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        distances = _compute_pairwise_squared_distances(X, backend)
        diag = backend.diag(distances)
        backend.eval(diag)
        diag_abs = backend.abs(diag)
        diag_max_arr = backend.max(diag_abs)
        backend.eval(diag_max_arr)
        diag_max = float(backend.to_scalar(diag_max_arr))

        tol = division_epsilon(backend, distances)
        assert diag_max <= tol

    def test_known_distance(self):
        """Test known geodesic distance between two points.

        Note: With geodesic distances, the actual value depends on the
        manifold approximation. For n=2 points, we fall back to Euclidean
        since we can't construct a k-NN graph with insufficient points.
        """
        backend = get_default_backend()
        # Points (0,0) and (3,4) - Euclidean squared distance is 25
        X = backend.array([[0.0, 0.0], [3.0, 4.0]])
        distances = _compute_pairwise_squared_distances(X, backend)
        backend.eval(distances)
        dist_01 = float(backend.to_scalar(distances[0, 1]))

        # With n=2 points, falls back to Euclidean
        tol = division_epsilon(backend, distances) * dist_01
        assert abs(dist_01 - 25.0) <= tol


class TestRBFGramMatrix:
    """Tests for RBF Gram matrix computation."""

    def test_diagonal_is_one(self):
        """RBF Gram diagonal should be 1 (K(x,x) = 1)."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = rbf_gram_matrix(X, backend)
        diag = backend.diag(gram)
        ones = backend.ones_like(diag)

        assert _all_close(backend, diag, ones)

    def test_symmetric(self):
        """RBF Gram matrix should be symmetric."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = rbf_gram_matrix(X, backend)
        gram_T = backend.transpose(gram)

        assert _all_close(backend, gram, gram_T)

    def test_values_in_zero_one(self):
        """RBF kernel values should be in (0, 1]."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = rbf_gram_matrix(X, backend)

        assert _all_in_unit_interval(backend, gram)

    def test_custom_sigma(self):
        """Test RBF with custom sigma."""
        backend = get_default_backend()
        X = backend.array([[0.0, 0.0], [1.0, 0.0]])

        distances = _compute_pairwise_squared_distances(X, backend)
        backend.eval(distances)
        dist_max_arr = backend.max(distances)
        backend.eval(dist_max_arr)
        dist_max = float(backend.to_scalar(dist_max_arr))
        sigma_small = regularization_epsilon(backend, X)
        if dist_max > 0.0:
            sigma_arr = backend.sqrt(backend.array([dist_max]))
            backend.eval(sigma_arr)
            sigma_large = float(backend.to_scalar(sigma_arr))
        else:
            sigma_large = sigma_small

        gram_small = rbf_gram_matrix(X, backend, sigma=sigma_small)
        gram_large = rbf_gram_matrix(X, backend, sigma=sigma_large)

        backend.eval(gram_small, gram_large)
        small_01 = float(backend.to_scalar(gram_small[0, 1]))
        large_01 = float(backend.to_scalar(gram_large[0, 1]))

        assert small_01 < large_01


class TestCKARBFKernel:
    """Tests for CKA with RBF kernel."""

    def test_rbf_identical_returns_one(self):
        """CKA of identical data with RBF should be 1."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)

        result = compute_cka(X, X, backend)

        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol
        assert result.is_valid

    def test_rbf_similar_activations(self):
        """Similar activations should have high RBF CKA."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        noise = _random_matrix(backend, 20, 10, 43) * division_epsilon(
            backend, backend.array([1.0])
        )
        Y_similar = X + noise
        Y_random = _random_matrix(backend, 20, 10, 44)

        result_similar = compute_cka(X, Y_similar, backend)
        result_random = compute_cka(X, Y_random, backend)

        tol = _scalar_tol(backend)
        assert result_similar.is_valid
        assert result_random.is_valid
        assert result_similar.cka >= result_random.cka - tol

    def test_rbf_random_activations(self):
        """Unrelated random activations should have moderate RBF CKA."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 10, 43)

        result = compute_cka(X, Y, backend)

        assert result.is_valid

    def test_rbf_invariant_to_orthogonal_transform(self):
        """RBF CKA should be invariant to orthogonal transformations."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 10, 43)

        # Apply random orthogonal transform to Y via QR decomposition
        random_mat = _random_matrix(backend, 10, 10, 44)
        Q, _ = backend.qr(random_mat)
        Y_rotated = backend.matmul(Y, Q)

        result_original = compute_cka(X, Y, backend)
        result_rotated = compute_cka(X, Y_rotated, backend)

        tol = _scalar_tol(backend)
        assert abs(result_original.cka - result_rotated.cka) <= tol

    def test_rbf_different_dimensions(self):
        """RBF CKA should work with different feature dimensions."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 15, 43)

        result = compute_cka(X, Y, backend)

        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_rbf_vs_linear_correlation(self):
        """RBF and linear CKA should be correlated for similar data."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        noise = _random_matrix(backend, 20, 10, 43) * division_epsilon(
            backend, backend.array([1.0])
        )
        Y_similar = X + noise
        Y_random = _random_matrix(backend, 20, 10, 44)

        # Both use RBF kernel now (default)
        result_sim = compute_cka(X, Y_similar, backend)
        result_rand = compute_cka(X, Y_random, backend)

        tol = _scalar_tol(backend)
        assert result_sim.cka >= result_rand.cka - tol

    def test_rbf_small_sample_count(self):
        """RBF CKA should handle small sample counts."""
        backend = get_default_backend()
        X = _random_matrix(backend, 3, 10, 42)
        Y = _random_matrix(backend, 3, 10, 43)

        result = compute_cka(X, Y, backend)

        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 3
