# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Tests for RBF kernel CKA implementation.

All CKA is now geodesic RBF - these tests verify the RBF Gram and CKA properties.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import (
    geodesic_squared_distances,
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
    """Check if two arrays are element-wise close."""
    diff = backend.abs(arr1 - arr2)
    backend.eval(diff)
    max_arr = backend.max(diff)
    backend.eval(max_arr)
    tol = division_epsilon(backend, diff)
    return float(backend.to_scalar(max_arr)) <= tol


class TestGeodesicDistances:
    """Tests for geodesic distance computation."""

    def test_distance_is_symmetric(self):
        """Distance matrix should be symmetric."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        distances = geodesic_squared_distances(X, backend)
        distances_T = backend.transpose(distances)

        assert _all_close(backend, distances, distances_T)

    def test_diagonal_is_zero(self):
        """Diagonal should be zero (distance to self)."""
        backend = get_default_backend()
        X = _random_matrix(backend, 10, 5, 42)
        distances = geodesic_squared_distances(X, backend)
        diag = backend.diag(distances)
        diag_max = backend.max(backend.abs(diag))
        backend.eval(diag_max)

        tol = division_epsilon(backend, distances)
        assert float(backend.to_scalar(diag_max)) <= tol


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

        min_val = backend.min(gram)
        max_val = backend.max(gram)
        backend.eval(min_val, max_val)

        tol = division_epsilon(backend, gram)
        assert float(backend.to_scalar(min_val)) >= -tol
        assert float(backend.to_scalar(max_val)) <= 1.0 + tol

    def test_custom_sigma(self):
        """Test RBF with custom sigma."""
        backend = get_default_backend()
        X = backend.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        gram_small = rbf_gram_matrix(X, backend, sigma=0.1)
        gram_large = rbf_gram_matrix(X, backend, sigma=10.0)
        backend.eval(gram_small, gram_large)

        # Larger sigma = higher off-diagonal values (more similarity)
        small_01 = float(backend.to_scalar(gram_small[0, 1]))
        large_01 = float(backend.to_scalar(gram_large[0, 1]))

        # With geodesic distances, sigma affects the RBF scaling
        # Small sigma = rapid decay = values closer to 0 for distant points
        # Large sigma = slow decay = values closer to 1 for distant points
        assert large_01 > small_01 or abs(large_01 - small_01) < 1e-10


class TestCKARBFKernel:
    """Tests for CKA with RBF kernel."""

    def test_rbf_identical_returns_one(self):
        """CKA of identical data should be ~1."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)

        result = compute_cka(X, X, backend)

        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol
        assert result.is_valid

    def test_rbf_similar_activations(self):
        """Similar activations should have high CKA."""
        backend = get_default_backend()
        X = _random_matrix(backend, 20, 10, 42)
        noise = _random_matrix(backend, 20, 10, 43) * 1e-6
        Y_similar = X + noise
        Y_random = _random_matrix(backend, 20, 10, 44)

        result_similar = compute_cka(X, Y_similar, backend)
        result_random = compute_cka(X, Y_random, backend)

        tol = _scalar_tol(backend)
        assert result_similar.is_valid
        assert result_random.is_valid
        assert result_similar.cka >= result_random.cka - tol

    def test_rbf_invariant_to_rotation(self):
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

    def test_rbf_small_sample_count(self):
        """RBF CKA should handle small sample counts."""
        backend = get_default_backend()
        X = _random_matrix(backend, 3, 10, 42)
        Y = _random_matrix(backend, 3, 10, 43)

        result = compute_cka(X, Y, backend)

        assert 0.0 <= result.cka <= 1.0
        assert result.sample_count == 3
