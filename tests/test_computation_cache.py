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

        assert stats_after.hits > stats_before.hits
        # Verify same result
        backend.eval(gram, gram2)
        gram_np = backend.to_numpy(gram)
        gram2_np = backend.to_numpy(gram2)
        assert (gram_np == gram2_np).all()

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
        d1 = backend.to_numpy(result.distances)
        d2 = backend.to_numpy(result2.distances)
        assert (d1 == d2).all()


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
        s_np = backend.to_numpy(s)
        s2_np = backend.to_numpy(s2)
        assert (s_np == s2_np).all()


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
        assert stats.hit_rate == 0.75

    def test_compute_time_saved_tracked(self, cache: ComputationCache, backend):
        """Test that compute time saved is tracked."""
        backend.random_seed(42)
        activations = backend.random_normal((100, 128))

        # First call - compute
        cache.get_or_compute_gram(activations, backend)

        # Second call - cached
        cache.get_or_compute_gram(activations, backend)

        stats = cache.get_stats()
        assert stats.total_compute_time_saved_ms > 0


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    def test_lru_eviction(self, backend):
        """Test that least recently used entries are evicted."""
        from modelcypher.core.domain.cache import ComputationCacheConfig

        # Create cache with small limit
        config = ComputationCacheConfig(max_gram_entries=3)
        cache = ComputationCache(config)

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
        assert stats.evictions >= 1

        cache.clear_all()


class TestCKACaching:
    """Tests for CKA with caching."""

    def test_cka_reuses_cached_gram(self, backend):
        """Test that CKA computations reuse cached Gram matrices."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.cka import compute_cka

        # Get the shared cache that CKA uses (don't reset - it's used at module level)
        cache = ComputationCache.shared()

        # Clear the cache to get a fresh baseline
        cache.clear_all()

        backend.random_seed(42)
        act_x = backend.random_normal((50, 64))
        act_y = backend.random_normal((50, 64))

        # First CKA call - should have misses for Gram matrices
        stats_before = cache.get_stats()
        result1 = compute_cka(act_x, act_y, backend)

        stats_after_first = cache.get_stats()
        # Should have misses (gram_x and gram_y at minimum)
        first_misses = stats_after_first.misses
        assert first_misses >= 2, f"Expected at least 2 misses, got {first_misses}"

        # Second CKA call with same inputs - should have hits
        result2 = compute_cka(act_x, act_y, backend)

        stats_after_second = cache.get_stats()
        # Should have hits for the Gram matrices
        assert stats_after_second.hits >= 2, (
            f"Expected at least 2 hits on second call, got {stats_after_second.hits}"
        )

        # Results should be the same
        assert result1.cka == pytest.approx(result2.cka, rel=1e-6)
