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

"""Tests for CKA RBF kernel implementation."""

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import (
    _compute_pairwise_squared_distances,
    _rbf_gram_matrix,
    compute_cka,
)


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_identical_points_zero_distance(self):
        """Identical points should have zero distance."""
        backend = get_default_backend()
        X = backend.array([[1.0, 2.0], [1.0, 2.0]])
        distances = _compute_pairwise_squared_distances(X, backend)
        distances_np = backend.to_numpy(distances)

        assert distances_np[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert distances_np[1, 0] == pytest.approx(0.0, abs=1e-10)

    def test_distance_is_symmetric(self):
        """Distance matrix should be symmetric."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((10, 5)))
        distances = _compute_pairwise_squared_distances(X, backend)
        distances_np = backend.to_numpy(distances)

        assert np.allclose(distances_np, distances_np.T)

    def test_diagonal_is_zero(self):
        """Diagonal should be zero (distance to self)."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((10, 5)))
        distances = _compute_pairwise_squared_distances(X, backend)
        distances_np = backend.to_numpy(distances)

        assert np.allclose(np.diag(distances_np), 0.0)

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
        distances_np = backend.to_numpy(distances)

        # With n=2 points, falls back to Euclidean
        assert distances_np[0, 1] == pytest.approx(25.0, rel=1e-5)


class TestRBFGramMatrix:
    """Tests for RBF Gram matrix computation."""

    def test_diagonal_is_one(self):
        """RBF Gram diagonal should be 1 (K(x,x) = 1)."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((10, 5)))
        gram = _rbf_gram_matrix(X, backend)
        gram_np = backend.to_numpy(gram)

        assert np.allclose(np.diag(gram_np), 1.0)

    def test_symmetric(self):
        """RBF Gram matrix should be symmetric."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((10, 5)))
        gram = _rbf_gram_matrix(X, backend)
        gram_np = backend.to_numpy(gram)

        assert np.allclose(gram_np, gram_np.T)

    def test_values_in_zero_one(self):
        """RBF kernel values should be in (0, 1]."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((10, 5)))
        gram = _rbf_gram_matrix(X, backend)
        gram_np = backend.to_numpy(gram)

        assert np.all(gram_np > 0)
        assert np.all(gram_np <= 1.0)

    def test_custom_sigma(self):
        """Test RBF with custom sigma."""
        backend = get_default_backend()
        X = backend.array([[0.0, 0.0], [1.0, 0.0]])

        # Small sigma -> lower similarity for distant points
        gram_small = _rbf_gram_matrix(X, backend, sigma=0.1)
        gram_small_np = backend.to_numpy(gram_small)
        # Large sigma -> higher similarity for distant points
        gram_large = _rbf_gram_matrix(X, backend, sigma=10.0)
        gram_large_np = backend.to_numpy(gram_large)

        assert gram_small_np[0, 1] < gram_large_np[0, 1]


class TestCKARBFKernel:
    """Tests for CKA with RBF kernel."""

    def test_rbf_identical_returns_one(self):
        """CKA of identical data with RBF should be 1."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((20, 10)).astype(np.float32))

        result = compute_cka(X, X, backend, use_linear_kernel=False)

        assert result.cka == pytest.approx(1.0, abs=1e-3)
        assert result.is_valid

    def test_rbf_similar_activations(self):
        """Similar activations should have high RBF CKA."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((20, 10)).astype(np.float32)
        # Small perturbation
        Y_np = X_np + rng.standard_normal((20, 10)).astype(np.float32) * 0.1

        X = backend.array(X_np)
        Y = backend.array(Y_np)

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert result.cka > 0.8
        assert result.is_valid

    def test_rbf_random_activations(self):
        """Unrelated random activations should have moderate RBF CKA."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((20, 10)).astype(np.float32))
        Y = backend.array(rng.standard_normal((20, 10)).astype(np.float32))

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        # RBF kernels can produce higher similarity due to non-linear structure
        # Random data typically produces CKA around 0.3-0.7
        assert result.cka < 0.8
        assert result.is_valid

    def test_rbf_invariant_to_orthogonal_transform(self):
        """RBF CKA should be invariant to orthogonal transformations."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((20, 10)).astype(np.float32)
        Y_np = rng.standard_normal((20, 10)).astype(np.float32)

        # Apply random orthogonal transform to Y
        Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
        Y_rotated_np = Y_np @ Q.astype(np.float32)

        X = backend.array(X_np)
        Y = backend.array(Y_np)
        Y_rotated = backend.array(Y_rotated_np)

        result_original = compute_cka(X, Y, backend, use_linear_kernel=False)
        result_rotated = compute_cka(X, Y_rotated, backend, use_linear_kernel=False)

        # Should be approximately equal (rotation invariance)
        assert result_original.cka == pytest.approx(result_rotated.cka, abs=0.01)

    def test_rbf_different_dimensions(self):
        """RBF CKA should work with different feature dimensions."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((20, 10)).astype(np.float32))
        Y = backend.array(rng.standard_normal((20, 15)).astype(np.float32))

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert 0.0 <= result.cka <= 1.0
        assert result.is_valid

    def test_rbf_vs_linear_correlation(self):
        """RBF and linear CKA should be correlated for similar data."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((20, 10)).astype(np.float32)
        Y_np = X_np + rng.standard_normal((20, 10)).astype(np.float32) * 0.3

        X = backend.array(X_np)
        Y = backend.array(Y_np)

        result_linear = compute_cka(X, Y, backend, use_linear_kernel=True)
        result_rbf = compute_cka(X, Y, backend, use_linear_kernel=False)

        # Both should indicate high similarity
        assert result_linear.cka > 0.5
        assert result_rbf.cka > 0.5

    def test_rbf_small_sample_count(self):
        """RBF CKA should handle small sample counts."""
        backend = get_default_backend()
        rng = np.random.default_rng(42)
        X = backend.array(rng.standard_normal((3, 10)).astype(np.float32))
        Y = backend.array(rng.standard_normal((3, 10)).astype(np.float32))

        result = compute_cka(X, Y, backend, use_linear_kernel=False)

        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 3
