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
    _rbf_gram_matrix,
    compute_cka,
)


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _all_close(backend, arr1, arr2, atol: float = 1e-5) -> bool:
    """Check if two arrays are element-wise close using backend."""
    diff = backend.abs(arr1 - arr2)
    backend.eval(diff)
    max_diff = float(backend.to_numpy(backend.max(diff)))
    return max_diff <= atol


def _all_greater_than(backend, arr, threshold: float) -> bool:
    """Check if all elements are > threshold using backend."""
    backend.eval(arr)
    min_val = float(backend.to_numpy(backend.min(arr)))
    return min_val > threshold


def _all_less_equal(backend, arr, threshold: float) -> bool:
    """Check if all elements are <= threshold using backend."""
    backend.eval(arr)
    max_val = float(backend.to_numpy(backend.max(arr)))
    return max_val <= threshold


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_identical_points_zero_distance(self):
        """Identical points should have zero distance."""
        backend = get_default_backend()
        X = backend.array([[1.0, 2.0], [1.0, 2.0]])
        distances = _compute_pairwise_squared_distances(X, backend)
        backend.eval(distances)
        distances_00_1 = float(backend.to_numpy(distances[0, 1]))
        distances_10 = float(backend.to_numpy(distances[1, 0]))

        assert distances_00_1 == pytest.approx(0.0, abs=1e-6)
        assert distances_10 == pytest.approx(0.0, abs=1e-6)

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
        diag_max = float(backend.to_numpy(backend.max(backend.abs(diag))))

        assert diag_max < 1e-5

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
        dist_01 = float(backend.to_numpy(distances[0, 1]))

        # With n=2 points, falls back to Euclidean
        assert dist_01 == pytest.approx(25.0, rel=1e-5)


class TestRBFGramMatrix:
    """Tests for RBF Gram matrix computation."""

    def test_diagonal_is_one(self):
        """RBF Gram diagonal should be 1 (K(x,x) = 1)."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = _rbf_gram_matrix(X, backend)
        diag = backend.diag(gram)
        ones = backend.ones_like(diag)

        assert _all_close(backend, diag, ones)

    def test_symmetric(self):
        """RBF Gram matrix should be symmetric."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = _rbf_gram_matrix(X, backend)
        gram_T = backend.transpose(gram)

        assert _all_close(backend, gram, gram_T)

    def test_values_in_zero_one(self):
        """RBF kernel values should be in (0, 1]."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        gram = _rbf_gram_matrix(X, backend)

        assert _all_greater_than(backend, gram, 0.0)
        assert _all_less_equal(backend, gram, 1.0)

    def test_custom_sigma(self):
        """Test RBF with custom sigma."""
        backend = get_default_backend()
        X = backend.array([[0.0, 0.0], [1.0, 0.0]])

        # Small sigma -> lower similarity for distant points
        gram_small = _rbf_gram_matrix(X, backend, sigma=0.1)
        # Large sigma -> higher similarity for distant points
        gram_large = _rbf_gram_matrix(X, backend, sigma=10.0)

        backend.eval(gram_small, gram_large)
        small_01 = float(backend.to_numpy(gram_small[0, 1]))
        large_01 = float(backend.to_numpy(gram_large[0, 1]))

        assert small_01 < large_01


class TestCKARBFKernel:
    """Tests for CKA with RBF kernel."""

    def test_rbf_identical_returns_one(self):
        """CKA of identical data with RBF should be 1."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)

        result = compute_cka(X, X, backend, use_linear_kernel=False)

        assert result.cka == pytest.approx(1.0, abs=1e-3)
        assert result.is_valid

    def test_rbf_similar_activations(self):
        """Similar activations should have high RBF CKA."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        # Small perturbation
        noise = _random_matrix(backend, 20, 10, 43) * 0.1
        Y = X + noise

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert result.cka > 0.8
        assert result.is_valid

    def test_rbf_random_activations(self):
        """Unrelated random activations should have moderate RBF CKA."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 10, 43)

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        # RBF kernels can produce higher similarity due to non-linear structure
        # Random data typically produces CKA around 0.3-0.7
        assert result.cka < 0.8
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

        result_original = compute_cka(X, Y, backend, use_linear_kernel=False)
        result_rotated = compute_cka(X, Y_rotated, backend, use_linear_kernel=False)

        # Should be approximately equal (rotation invariance)
        assert result_original.cka == pytest.approx(result_rotated.cka, abs=0.01)

    def test_rbf_different_dimensions(self):
        """RBF CKA should work with different feature dimensions."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        Y = _random_matrix(backend, 20, 15, 43)

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_rbf_vs_linear_correlation(self):
        """RBF and linear CKA should be correlated for similar data."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        noise = _random_matrix(backend, 20, 10, 43) * 0.3
        Y = X + noise

        result_linear = compute_cka(X, Y, backend, use_linear_kernel=True)
        result_rbf = compute_cka(X, Y, backend, use_linear_kernel=False)

        # Both should indicate high similarity
        assert result_linear.cka > 0.5
        assert result_rbf.cka > 0.5

    def test_rbf_small_sample_count(self):
        """RBF CKA should handle small sample counts."""
        backend = get_default_backend()
        X = _random_matrix(backend, 3, 10, 42)
        Y = _random_matrix(backend, 3, 10, 43)

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 3
