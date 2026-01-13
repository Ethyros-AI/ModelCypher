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

        assert result.cka == pytest.approx(1.0, rel=1e-5)

    def test_random_activations_cka_less_than_one(self, backend):
        """Independent random activations should have CKA < 1."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        result = compute_cka(X, Y, backend)

        assert result.cka < 1.0
        assert result.cka >= 0.0

    def test_cka_symmetry(self, backend):
        """CKA(X, Y) should equal CKA(Y, X)."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 32))
        backend.eval(X, Y)

        result_xy = compute_cka(X, Y, backend)
        result_yx = compute_cka(Y, X, backend)

        assert result_xy.cka == pytest.approx(result_yx.cka, rel=1e-6)

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

        assert cka_self == pytest.approx(1.0, rel=1e-5)

    def test_scaled_activations_change_cka(self, backend):
        """Geodesic CKA should change when activations are scaled.

        RBF CKA adapts sigma to the data distribution, so scaling activations
        shifts geodesic distances and changes CKA.
        """
        from modelcypher.core.domain.geometry.cka import compute_geodesic_cka
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            sqrt_scalar,
        )

        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 32))
        backend.eval(X, Y)

        # Scale X by a constant using backend element-wise multiplication
        X_scaled = X * 100.0
        backend.eval(X_scaled)

        cka_original = compute_geodesic_cka(X, Y, backend)
        cka_scaled = compute_geodesic_cka(X_scaled, Y, backend)

        precision = float(sqrt_scalar(machine_epsilon(backend, X), backend))
        assert abs(cka_original - cka_scaled) > precision


class TestCKACacheIntegration:
    """Tests for CKA computation with cache integration."""

    def test_full_cka_with_cache_reuse(self, backend):
        """Verify Gram matrices are reused across CKA computations."""
        # Get shared cache and clear it
        cache = ComputationCache.shared()
        cache.clear_all()

        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        # First CKA call - should have cache misses
        stats_before = cache.get_stats()
        _ = compute_cka(X, Y, backend)
        stats_after_first = cache.get_stats()

        # Should have at least 2 misses (gram_x and gram_y)
        first_misses = stats_after_first.misses - stats_before.misses
        assert first_misses >= 2

        # Second CKA call with same inputs - should have cache hits
        _ = compute_cka(X, Y, backend)
        stats_after_second = cache.get_stats()

        # Should have at least 2 hits
        second_hits = stats_after_second.hits - stats_after_first.hits
        assert second_hits >= 2

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
            assert cka_matrix[i][i] == pytest.approx(1.0, rel=1e-5)

        # Verify symmetry
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                assert cka_matrix[i][j] == pytest.approx(cka_matrix[j][i], rel=1e-6)


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
