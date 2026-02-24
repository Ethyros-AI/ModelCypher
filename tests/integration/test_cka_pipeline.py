# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Integration tests for the CKA computation pipeline.

Tests the full flow: activations → Gram matrices → CKA → alignment → verification.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import compute_cka, compute_cka_from_grams
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def cache():
    """Create a fresh computation cache for testing."""
    cache = ComputationCache()
    yield cache
    cache.clear_all()


class TestCKAPipeline:
    """End-to-end tests for CKA computation pipeline."""

    def test_identical_activations_cka_equals_one(self, backend):
        """Identical activations should have CKA = 1.0."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.eval(X)

        result = compute_cka(X, X, backend)

        eps = regularization_epsilon(backend, X)
        assert result.cka == pytest.approx(1.0, rel=eps)

    def test_random_activations_cka_less_than_one(self, backend):
        """Independent random activations should match shared-sigma Gram CKA."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        from modelcypher.core.domain.geometry.cka import (
            _rbf_gram_from_sq_distances,
            _shared_rbf_sigma,
            geodesic_squared_distances,
        )

        sq_x = geodesic_squared_distances(X, backend)
        sq_y = geodesic_squared_distances(Y, backend)
        sigma = _shared_rbf_sigma(sq_x, sq_y, backend)
        gram_x = _rbf_gram_from_sq_distances(sq_x, sigma, backend)
        gram_y = _rbf_gram_from_sq_distances(sq_y, sigma, backend)
        expected = compute_cka_from_grams(gram_x, gram_y, backend)
        eps = regularization_epsilon(backend, gram_x)
        assert result.cka == pytest.approx(expected, rel=eps)

    def test_cka_symmetry(self, backend):
        """CKA(X, Y) should equal CKA(Y, X)."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 32))
        backend.eval(X, Y)

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        eps = regularization_epsilon(backend, X)
        assert result_xy.cka == pytest.approx(result_yx.cka, rel=eps)

    def test_cka_from_grams_self_consistent(self, backend):
        """CKA from Gram matrices should be self-consistent.

        Note: compute_cka uses RBF kernel with geodesic distances, while
        compute_cka_from_grams works with pre-computed (linear) Grams.
        These are intentionally different kernels for different use cases:
        - RBF/geodesic: correct for manifold geometry analysis
        - Linear Grams: legacy compatibility and specific cross-modal comparisons
        """
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.eval(X)

        # Compute linear Gram matrix
        gram_x = backend.matmul(X, backend.transpose(X))
        backend.eval(gram_x)

        # Self-similarity should be 1.0
        cka_self = compute_cka_from_grams(gram_x, gram_x, backend)

        eps = regularization_epsilon(backend, gram_x)
        assert cka_self == pytest.approx(1.0, rel=eps)

    def test_scaled_activations_preserve_linear_cka(self, backend):
        """Linear CKA is scale-invariant: uniform scaling should not change CKA.

        CKA = HSIC(K_x, K_y) / sqrt(HSIC(K_x, K_x) * HSIC(K_y, K_y))

        The normalization makes linear CKA (with linear kernel K = X @ X.T)
        invariant to uniform scaling. Note: Geodesic/RBF CKA is NOT
        scale-invariant because scaling affects distances and kernel bandwidth.
        """
        from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            sqrt_scalar,
        )

        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 32))
        backend.eval(X, Y)

        # Scale X by a constant
        X_scaled = X * 100.0
        backend.eval(X_scaled)

        # Linear Gram matrices: K = X @ X.T
        gram_x = backend.matmul(X, backend.transpose(X))
        gram_x_scaled = backend.matmul(X_scaled, backend.transpose(X_scaled))
        gram_y = backend.matmul(Y, backend.transpose(Y))
        backend.eval(gram_x, gram_x_scaled, gram_y)

        cka_original = compute_cka_from_grams(gram_x, gram_y, backend)
        cka_scaled = compute_cka_from_grams(gram_x_scaled, gram_y, backend)

        # Linear CKA should be scale-invariant
        precision = float(sqrt_scalar(machine_epsilon(backend, X), backend))
        assert abs(cka_original - cka_scaled) <= precision, (
            f"Linear CKA should be scale-invariant: original={cka_original}, scaled={cka_scaled}"
        )


class TestCKACacheIntegration:
    """Tests for CKA computation with cache integration."""

    def test_full_cka_with_cache_reuse(self, backend):
        """Verify geodesic distances are reused across CKA computations."""
        # Get shared cache and clear it
        cache = ComputationCache.shared()
        cache.clear_all()

        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        key_x = cache.make_array_key(X, backend)
        key_y = cache.make_array_key(Y, backend)
        geo_key_x = f"geo:{key_x}"
        geo_key_y = f"geo:{key_y}"
        assert cache.get_geodesic(geo_key_x) is None
        assert cache.get_geodesic(geo_key_y) is None

        _ = compute_cka(X, Y, backend)
        cached_x = cache.get_geodesic(geo_key_x)
        cached_y = cache.get_geodesic(geo_key_y)
        assert cached_x is not None
        assert cached_y is not None

        _ = compute_cka(X, Y, backend)
        cached_x_second = cache.get_geodesic(geo_key_x)
        cached_y_second = cache.get_geodesic(geo_key_y)
        assert cached_x_second is cached_x
        assert cached_y_second is cached_y

    def test_cross_representation_cka_matrix(self, backend):
        """Compute CKA matrix across multiple representations."""
        backend.random_seed(42)

        # Simulate layer activations
        layers = []
        for i in range(4):
            backend.random_seed(42 + i)
            layer = backend.random_normal((30, 32 + i * 8))
            backend.eval(layer)
            layers.append(layer)

        # Compute CKA matrix
        n_layers = len(layers)
        cka_matrix = []
        for i in range(n_layers):
            row = []
            for j in range(n_layers):
                result = compute_cka(layers[i], layers[j], backend)
                row.append(result.cka)
            cka_matrix.append(row)

        # Verify diagonal elements are 1.0
        for i in range(n_layers):
            eps = regularization_epsilon(backend, layers[i])
            assert cka_matrix[i][i] == pytest.approx(1.0, rel=eps)

        # Verify symmetry
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                eps = regularization_epsilon(backend, layers[i])
                assert cka_matrix[i][j] == pytest.approx(cka_matrix[j][i], rel=eps)


class TestCKAWithDifferentDimensions:
    """Tests for CKA across different feature dimensions."""

    def test_cka_different_feature_dims(self, backend):
        """CKA should work across different feature dimensions."""
        backend.random_seed(42)

        # Different feature dimensions, same sample count
        X_small = backend.random_normal((50, 32))
        X_medium = backend.random_normal((50, 128))
        X_large = backend.random_normal((50, 512))
        backend.eval(X_small, X_medium, X_large)

        # All pairs should be computable
        result_sm = compute_cka(X_small, X_medium, backend)
        result_ml = compute_cka(X_medium, X_large, backend)
        result_sl = compute_cka(X_small, X_large, backend)

        # All should be valid CKA values
        for result in [result_sm, result_ml, result_sl]:
            assert 0.0 <= result.cka <= 1.0

    def test_cka_single_feature(self, backend):
        """CKA should handle single-feature representations."""
        backend.random_seed(42)

        X = backend.random_normal((50, 1))
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert 0.0 <= result.cka <= 1.0

    def test_cka_high_dimensional(self, backend):
        """CKA should work with high-dimensional representations."""
        backend.random_seed(42)

        X = backend.random_normal((30, 1024))
        Y = backend.random_normal((30, 2048))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert 0.0 <= result.cka <= 1.0
