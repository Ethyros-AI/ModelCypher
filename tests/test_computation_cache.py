# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for computation cache infrastructure."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.support.array_utils import array_to_list


@pytest.fixture
def cache() -> ComputationCache:
    """Create a fresh computation cache for testing."""
    cache = ComputationCache()
    yield cache
    cache.clear_all()


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


class TestGramMatrixCaching:
    """Tests for Gram matrix caching."""

    def test_gram_cache_stores_and_retrieves(self, cache: ComputationCache, backend):
        """Test that Gram matrices are cached correctly."""
        backend.random_seed(42)
        activations = backend.random_normal((50, 64))

        # First call - should miss and compute
        key = cache.make_gram_key(activations, backend)
        assert cache.get_gram(key) is None

        gram = cache.get_or_compute_gram(activations, backend)
        assert gram is not None
        assert gram.shape == (50, 50)

        # Second call - should hit cache
        stats_before = cache.get_stats()
        gram2 = cache.get_or_compute_gram(activations, backend)
        stats_after = cache.get_stats()

        assert stats_after.hits - stats_before.hits == 1
        # Verify same result
        backend.eval(gram, gram2)
        diff = backend.max(backend.abs(gram - gram2))
        backend.eval(diff)
        eps = machine_epsilon(backend, gram) * gram.shape[0]
        assert backend.to_scalar(diff) <= eps

    def test_different_inputs_different_keys(self, cache: ComputationCache, backend):
        """Test that different inputs produce different cache keys."""
        backend.random_seed(42)
        act1 = backend.random_normal((50, 64))
        act2 = backend.random_normal((50, 64))

        key1 = cache.make_gram_key(act1, backend)
        key2 = cache.make_gram_key(act2, backend)

        assert key1 != key2


class TestGeodesicCaching:
    """Tests for geodesic distance caching."""

    def test_geodesic_cache_stores_and_retrieves(self, cache: ComputationCache, backend):
        """Test that geodesic distances are cached correctly."""
        backend.random_seed(42)
        points = backend.random_normal((20, 16))

        key = cache.make_geodesic_key(points, backend, k_neighbors=5)
        assert cache.get_geodesic(key) is None

        # Manually compute and cache
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        result = rg.geodesic_distances(points, k_neighbors=5)

        # The result should now be cached by RiemannianGeometry
        # Check by calling again
        result2 = rg.geodesic_distances(points, k_neighbors=5)

        # Verify same result
        backend.eval(result.distances, result2.distances)
        diff = backend.max(backend.abs(result.distances - result2.distances))
        backend.eval(diff)
        eps = machine_epsilon(backend, result.distances) * result.distances.shape[0]
        assert backend.to_scalar(diff) <= eps


class TestSVDCaching:
    """Tests for SVD caching."""

    def test_svd_cache_stores_and_retrieves(self, cache: ComputationCache, backend):
        """Test that SVD results are cached correctly."""
        backend.random_seed(42)
        matrix = backend.random_normal((32, 32))

        # First call - should miss
        key = cache.make_svd_key(matrix, backend)
        assert cache.get_svd(key) is None

        # Compute and cache
        u, s, vt = cache.get_or_compute_svd(matrix, backend)

        # Second call - should hit
        u2, s2, vt2 = cache.get_or_compute_svd(matrix, backend)

        backend.eval(s, s2)
        diff = backend.max(backend.abs(s - s2))
        backend.eval(diff)
        eps = machine_epsilon(backend, s) * s.shape[0]
        assert backend.to_scalar(diff) <= eps


class TestCacheStatistics:
    """Tests for cache statistics."""

    def test_stats_track_hits_and_misses(self, cache: ComputationCache, backend):
        """Test that statistics correctly track hits and misses."""
        backend.random_seed(42)
        activations = backend.random_normal((20, 32))

        stats_before = cache.get_stats()
        assert stats_before.hits == 0
        assert stats_before.misses == 0

        # First call - miss
        cache.get_or_compute_gram(activations, backend)
        stats_after_first = cache.get_stats()
        assert stats_after_first.misses == 1

        # Second call - hit
        cache.get_or_compute_gram(activations, backend)
        stats_after_second = cache.get_stats()
        assert stats_after_second.hits == 1

    def test_hit_rate_calculation(self, cache: ComputationCache, backend):
        """Test that hit rate is calculated correctly."""
        backend.random_seed(42)
        activations = backend.random_normal((20, 32))

        # 1 miss, 3 hits = 75% hit rate
        cache.get_or_compute_gram(activations, backend)  # miss
        cache.get_or_compute_gram(activations, backend)  # hit
        cache.get_or_compute_gram(activations, backend)  # hit
        cache.get_or_compute_gram(activations, backend)  # hit

        stats = cache.get_stats()
        expected_hit_rate = 3 / 4
        eps = division_epsilon(backend, backend.array([1.0]))
        assert abs(stats.hit_rate - expected_hit_rate) <= eps

    def test_compute_time_saved_tracked(self, cache: ComputationCache, backend):
        """Test that compute time saved is tracked."""
        backend.random_seed(42)
        activations = backend.random_normal((100, 128))

        # First call - compute
        key = cache.make_gram_key(activations, backend)
        cache.get_or_compute_gram(activations, backend)
        entry = cache._gram_cache[key]

        # Second call - cached
        cache.get_or_compute_gram(activations, backend)

        stats = cache.get_stats()
        eps = division_epsilon(backend, backend.array([1.0]))
        assert stats.total_compute_time_saved_ms == pytest.approx(entry.compute_time_ms, rel=eps)


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    def test_lru_eviction(self, backend):
        """Test that least recently used entries are evicted."""
        # Create cache with small limit
        cache = ComputationCache(max_gram_entries=3)

        backend.random_seed(42)

        # Add 4 items (exceeds limit of 3)
        arrays = [backend.random_normal((10, 16)) for _ in range(4)]

        for arr in arrays:
            cache.get_or_compute_gram(arr, backend)

        # Cache should only have 3 entries
        sizes = cache.get_cache_sizes()
        assert sizes["gram"] == 3

        # Check evictions occurred
        stats = cache.get_stats()
        assert stats.evictions == 1

        cache.clear_all()


class TestCKACaching:
    """Tests for CKA with caching."""

    def test_cka_reuses_cached_gram(self, backend):
        """Test that CKA computations reuse cached geodesic distances."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.cka import compute_cka

        # Get the shared cache that CKA uses (don't reset - it's used at module level)
        cache = ComputationCache.shared()

        # Clear the cache to get a fresh baseline
        cache.clear_all()

        backend.random_seed(42)
        act_x = backend.random_normal((50, 64))
        act_y = backend.random_normal((50, 64))

        key_x = cache.make_array_key(act_x, backend)
        key_y = cache.make_array_key(act_y, backend)
        geo_key_x = f"geo:{key_x}"
        geo_key_y = f"geo:{key_y}"
        assert cache.get_geodesic(geo_key_x) is None
        assert cache.get_geodesic(geo_key_y) is None

        result1 = compute_cka(act_x, act_y, backend)
        cached_x = cache.get_geodesic(geo_key_x)
        cached_y = cache.get_geodesic(geo_key_y)
        assert cached_x is not None
        assert cached_y is not None

        result2 = compute_cka(act_x, act_y, backend)
        cached_x_second = cache.get_geodesic(geo_key_x)
        cached_y_second = cache.get_geodesic(geo_key_y)
        assert cached_x_second is cached_x
        assert cached_y_second is cached_y

        # Results should be the same
        tol = division_epsilon(backend, backend.array([1.0]))
        assert abs(result1.cka - result2.cka) <= tol


class TestCacheKeyCollisionResistance:
    """Tests for cache key collision resistance."""

    def test_permuted_arrays_have_different_keys(self, cache: ComputationCache, backend):
        """Permuted arrays should produce different cache keys."""
        backend.random_seed(42)

        # Create a small array (uses full hash)
        X = backend.random_normal((15, 8))  # 120 elements
        backend.eval(X)

        # Create permutation matrix
        perm = list(reversed(range(15)))
        P_list = [[0.0 for _ in range(15)] for _ in range(15)]
        for i, j in enumerate(perm):
            P_list[i][j] = 1.0
        P = backend.array(P_list)

        # Permute X
        X_perm = backend.matmul(P, X)
        backend.eval(X_perm)

        # Keys should be different
        key_x = cache.make_array_key(X, backend)
        key_x_perm = cache.make_array_key(X_perm, backend)

        assert key_x != key_x_perm, "Permuted array should have different key"

    def test_slightly_different_arrays_have_different_keys(self, cache: ComputationCache, backend):
        """Arrays differing only in later elements should have different keys."""
        backend.random_seed(42)

        # Create small array
        X = backend.random_normal((10, 10))  # 100 elements
        backend.eval(X)
        X_list = array_to_list(backend, X)
        X_list[-1][-1] += division_epsilon(backend, X)
        X_modified = backend.array(X_list)
        backend.eval(X_modified)

        key_x = cache.make_array_key(X, backend)
        key_modified = cache.make_array_key(X_modified, backend)

        assert key_x != key_modified, "Modified array should have different key"

    def test_identical_arrays_have_same_key(self, cache: ComputationCache, backend):
        """Identical arrays should produce the same cache key."""
        backend.random_seed(42)

        X = backend.random_normal((20, 10))
        backend.eval(X)

        # Copy to a new array (same content)
        X_list = array_to_list(backend, X)
        X_copy = backend.array(X_list)
        backend.eval(X_copy)

        key1 = cache.make_array_key(X, backend)
        key2 = cache.make_array_key(X_copy, backend)

        assert key1 == key2, "Identical arrays should have same key"
