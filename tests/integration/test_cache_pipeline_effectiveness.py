# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Integration tests for cache pipeline effectiveness.

Tests that the caching infrastructure provides real performance benefits
across different computation pipelines.
"""

from __future__ import annotations

import time

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import compute_cka


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def fresh_cache():
    """Create a fresh computation cache for testing."""
    cache = ComputationCache()
    yield cache
    cache.clear_all()


class TestIdCacheFastPath:
    """Tests for id()-based fast-path caching."""

    def test_same_array_skips_hashing(self, backend, fresh_cache):
        """Same array object should return cached key without rehashing."""
        backend.random_seed(42)
        X = backend.random_normal((100, 128))
        backend.eval(X)

        # First call computes hash
        key1 = fresh_cache.make_array_key(X, backend)

        # Time repeated calls - should be much faster due to id() cache
        start = time.perf_counter()
        for _ in range(100):
            key2 = fresh_cache.make_array_key(X, backend)
        elapsed = time.perf_counter() - start

        assert key1 == key2
        # 100 cached lookups should be very fast (<10ms total)
        assert elapsed < 0.1, f"Repeated lookups too slow: {elapsed*1000:.1f}ms"

    def test_different_arrays_different_keys(self, backend, fresh_cache):
        """Different array objects should produce different cache keys."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.random_seed(123)
        Y = backend.random_normal((50, 64))
        backend.eval(X, Y)

        key_x = fresh_cache.make_array_key(X, backend)
        key_y = fresh_cache.make_array_key(Y, backend)

        assert key_x != key_y

    def test_copied_array_same_key(self, backend, fresh_cache):
        """Copied arrays with same content should produce same key."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.eval(X)

        # Copy to new array
        X_np = backend.to_numpy(X)
        X_copy = backend.array(X_np.copy())
        backend.eval(X_copy)

        key_x = fresh_cache.make_array_key(X, backend)
        key_copy = fresh_cache.make_array_key(X_copy, backend)

        assert key_x == key_copy


class TestGramCacheEffectiveness:
    """Tests for Gram matrix cache effectiveness."""

    def test_cka_pipeline_cache_hits(self, backend):
        """Verify gram matrices are reused across CKA computations."""
        cache = ComputationCache.shared()
        cache.clear_all()

        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        Y = backend.random_normal((50, 32))
        backend.eval(X, Y)

        # First computation - cache miss
        stats_before = cache.get_stats()
        _ = compute_cka(X, Y, backend)
        stats_after_first = cache.get_stats()

        first_misses = stats_after_first.misses - stats_before.misses
        assert first_misses >= 2, "Expected cache misses on first CKA"

        # Second computation - cache hit
        _ = compute_cka(X, Y, backend)
        stats_after_second = cache.get_stats()

        second_hits = stats_after_second.hits - stats_after_first.hits
        assert second_hits >= 2, "Expected cache hits on second CKA"

    def test_cache_time_savings(self, backend):
        """Cache should track compute time saved."""
        cache = ComputationCache.shared()
        cache.clear_all()

        backend.random_seed(42)
        # Use larger array for measurable compute time
        X = backend.random_normal((200, 256))
        backend.eval(X)

        # First call - computes and caches
        stats_before = cache.get_stats()
        _ = cache.get_or_compute_gram(X, backend)
        stats_after_first = cache.get_stats()

        assert stats_after_first.misses > stats_before.misses

        # Second call - uses cache
        _ = cache.get_or_compute_gram(X, backend)
        stats_after_second = cache.get_stats()

        assert stats_after_second.hits > stats_after_first.hits
        assert stats_after_second.total_compute_time_saved_ms > 0


class TestLRUEvictionBehavior:
    """Tests for LRU cache eviction behavior."""

    def test_lru_eviction_order(self, backend):
        """Oldest entries should be evicted first."""
        from modelcypher.core.domain.cache import ComputationCacheConfig

        # Create cache with very small limit
        config = ComputationCacheConfig(max_gram_entries=3)
        cache = ComputationCache(config)

        backend.random_seed(42)

        # Create 5 different arrays (exceeds limit of 3)
        arrays = []
        for i in range(5):
            backend.random_seed(42 + i)
            arr = backend.random_normal((10, 16))
            backend.eval(arr)
            arrays.append(arr)

        # Add first 3 - all should be cached
        for i in range(3):
            cache.get_or_compute_gram(arrays[i], backend)

        sizes = cache.get_cache_sizes()
        assert sizes["gram"] == 3

        # Add 4th - should evict 1st
        cache.get_or_compute_gram(arrays[3], backend)
        sizes = cache.get_cache_sizes()
        assert sizes["gram"] == 3

        # Add 5th - should evict 2nd
        cache.get_or_compute_gram(arrays[4], backend)
        sizes = cache.get_cache_sizes()
        assert sizes["gram"] == 3

        # Verify evictions were tracked
        stats = cache.get_stats()
        assert stats.evictions >= 2

        cache.clear_all()

    def test_access_refreshes_lru_position(self, backend):
        """Accessing an entry should move it to end of LRU order."""
        from modelcypher.core.domain.cache import ComputationCacheConfig

        config = ComputationCacheConfig(max_gram_entries=3)
        cache = ComputationCache(config)

        backend.random_seed(42)

        # Create 4 arrays
        arrays = []
        for i in range(4):
            backend.random_seed(42 + i)
            arr = backend.random_normal((10, 16))
            backend.eval(arr)
            arrays.append(arr)

        # Add arrays[0], [1], [2]
        for i in range(3):
            cache.get_or_compute_gram(arrays[i], backend)

        # Access arrays[0] to refresh its position
        cache.get_or_compute_gram(arrays[0], backend)

        # Add arrays[3] - should evict arrays[1] (oldest non-accessed)
        cache.get_or_compute_gram(arrays[3], backend)

        # arrays[0] should still be in cache (was refreshed)
        key_0 = cache.make_gram_key(arrays[0], backend)
        assert cache.get_gram(key_0) is not None

        cache.clear_all()


class TestSVDCacheEffectiveness:
    """Tests for SVD cache effectiveness."""

    def test_svd_cache_stores_and_retrieves(self, backend, fresh_cache):
        """SVD results should be cached correctly."""
        backend.random_seed(42)
        matrix = backend.random_normal((32, 32))
        backend.eval(matrix)

        # First call - miss
        stats_before = fresh_cache.get_stats()
        u1, s1, vt1 = fresh_cache.get_or_compute_svd(matrix, backend)
        stats_after_first = fresh_cache.get_stats()

        assert stats_after_first.misses > stats_before.misses

        # Second call - hit
        u2, s2, vt2 = fresh_cache.get_or_compute_svd(matrix, backend)
        stats_after_second = fresh_cache.get_stats()

        assert stats_after_second.hits > stats_after_first.hits

        # Results should be identical
        backend.eval(s1, s2)
        s1_np = backend.to_numpy(s1)
        s2_np = backend.to_numpy(s2)
        assert (s1_np == s2_np).all()


class TestCacheClearBehavior:
    """Tests for cache clearing behavior."""

    def test_clear_all_resets_everything(self, backend, fresh_cache):
        """clear_all should reset all caches and stats."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.eval(X)

        # Populate cache
        fresh_cache.get_or_compute_gram(X, backend)
        fresh_cache.get_or_compute_gram(X, backend)  # Hit

        stats_before = fresh_cache.get_stats()
        assert stats_before.hits > 0
        assert stats_before.misses > 0

        sizes_before = fresh_cache.get_cache_sizes()
        assert sizes_before["gram"] > 0

        # Clear everything
        fresh_cache.clear_all()

        # Verify all cleared
        stats_after = fresh_cache.get_stats()
        assert stats_after.hits == 0
        assert stats_after.misses == 0
        assert stats_after.evictions == 0

        sizes_after = fresh_cache.get_cache_sizes()
        assert all(size == 0 for size in sizes_after.values())

    def test_clear_id_cache(self, backend, fresh_cache):
        """clear_id_cache should only clear the id cache."""
        backend.random_seed(42)
        X = backend.random_normal((50, 64))
        backend.eval(X)

        # Populate both caches
        fresh_cache.get_or_compute_gram(X, backend)
        _ = fresh_cache.make_array_key(X, backend)

        sizes_before = fresh_cache.get_cache_sizes()
        assert sizes_before["gram"] > 0

        # Clear only id cache
        fresh_cache.clear_id_cache()

        # Gram cache should still be populated
        sizes_after = fresh_cache.get_cache_sizes()
        assert sizes_after["gram"] > 0
