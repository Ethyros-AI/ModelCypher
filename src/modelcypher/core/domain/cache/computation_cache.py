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

"""Session-scoped computation cache for expensive tensor operations.

This module provides in-memory caching for frequently repeated high-dimensional
calculations like Gram matrices, geodesic distances, SVD decompositions, and
Fréchet means.

Unlike the disk-backed TwoLevelCache, this cache is:
- Session-scoped (cleared when the process exits)
- Memory-only (no disk persistence)
- Optimized for tensor operations that repeat within a single analysis

Usage:
    cache = ComputationCache.shared()

    # Cache a Gram matrix
    key = cache.make_gram_key(activations)
    gram = cache.get_gram(key)
    if gram is None:
        gram = backend.matmul(activations, backend.transpose(activations))
        cache.set_gram(key, gram)

    # Or use the convenience wrapper
    gram = cache.get_or_compute_gram(activations, backend)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_compute_time_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """A cached computation result with metadata."""

    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 1
    compute_time_ms: float = 0.0


@dataclass
class ComputationCacheConfig:
    """Configuration for computation cache behavior."""

    max_gram_entries: int = 50
    max_geodesic_entries: int = 20
    max_svd_entries: int = 30
    max_frechet_entries: int = 30
    max_centered_gram_entries: int = 50


class ComputationCache:
    """
    Session-scoped in-memory cache for expensive tensor computations.

    Provides separate LRU caches for different computation types:
    - Gram matrices (X @ X^T)
    - Geodesic distance matrices
    - SVD decompositions
    - Fréchet means
    - Centered Gram matrices

    Thread-safe with per-cache-type locking.
    """

    _shared_instance: "ComputationCache | None" = None
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "ComputationCache":
        """Get the shared singleton instance."""
        if cls._shared_instance is None:
            with cls._shared_lock:
                if cls._shared_instance is None:
                    cls._shared_instance = ComputationCache()
        return cls._shared_instance

    @classmethod
    def reset_shared(cls) -> None:
        """Reset the shared instance (for testing)."""
        with cls._shared_lock:
            cls._shared_instance = None

    def __init__(self, config: ComputationCacheConfig | None = None) -> None:
        """
        Initialize the computation cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or ComputationCacheConfig()

        # Separate caches for different computation types
        self._gram_cache: dict[str, CacheEntry] = {}
        self._gram_order: list[str] = []
        self._gram_lock = threading.Lock()

        self._centered_gram_cache: dict[str, CacheEntry] = {}
        self._centered_gram_order: list[str] = []
        self._centered_gram_lock = threading.Lock()

        self._geodesic_cache: dict[str, CacheEntry] = {}
        self._geodesic_order: list[str] = []
        self._geodesic_lock = threading.Lock()

        self._svd_cache: dict[str, CacheEntry] = {}
        self._svd_order: list[str] = []
        self._svd_lock = threading.Lock()

        self._frechet_cache: dict[str, CacheEntry] = {}
        self._frechet_order: list[str] = []
        self._frechet_lock = threading.Lock()

        self._stats = CacheStats()
        self._stats_lock = threading.Lock()

    # --- Key Generation ---

    def make_array_key(self, arr: "Array", backend: "Backend") -> str:
        """
        Create a hash key from an array's content.

        Uses shape + sampled values for efficiency on large arrays.

        Args:
            arr: Input array
            backend: Backend for array operations

        Returns:
            16-character hex hash
        """
        backend.eval(arr)
        shape = tuple(int(d) for d in arr.shape)
        n_elements = 1
        for d in shape:
            n_elements *= d

        # For efficiency, hash shape + sampled values instead of all values
        # Sample corners and center for large arrays
        arr_np = backend.to_numpy(arr)

        if n_elements <= 1000:
            # Small array - hash all values
            flat = arr_np.flatten()
            content = f"shape={shape}|{flat.tobytes().hex()[:64]}"
        else:
            # Large array - sample strategically
            flat = arr_np.flatten()
            samples = []
            # First 10
            samples.extend(flat[:10].tolist())
            # Last 10
            samples.extend(flat[-10:].tolist())
            # Middle 10
            mid = len(flat) // 2
            samples.extend(flat[mid - 5 : mid + 5].tolist())
            # Random-ish samples based on position (deterministic)
            step = max(1, len(flat) // 20)
            samples.extend(flat[::step][:10].tolist())
            content = f"shape={shape}|samples={samples}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def make_gram_key(
        self,
        arr: "Array",
        backend: "Backend",
        kernel_type: str = "linear",
    ) -> str:
        """Create cache key for Gram matrix computation."""
        base_key = self.make_array_key(arr, backend)
        return f"gram_{kernel_type}_{base_key}"

    def make_centered_gram_key(self, gram_key: str) -> str:
        """Create cache key for centered Gram matrix."""
        return f"centered_{gram_key}"

    def make_geodesic_key(
        self,
        arr: "Array",
        backend: "Backend",
        k_neighbors: int,
    ) -> str:
        """Create cache key for geodesic distance matrix."""
        base_key = self.make_array_key(arr, backend)
        return f"geodesic_k{k_neighbors}_{base_key}"

    def make_svd_key(
        self,
        arr: "Array",
        backend: "Backend",
        full_matrices: bool = False,
    ) -> str:
        """Create cache key for SVD computation."""
        base_key = self.make_array_key(arr, backend)
        return f"svd_full{full_matrices}_{base_key}"

    def make_frechet_key(
        self,
        arr: "Array",
        backend: "Backend",
        weights_key: str | None = None,
        k_neighbors: int | None = None,
    ) -> str:
        """Create cache key for Fréchet mean computation."""
        base_key = self.make_array_key(arr, backend)
        weights_suffix = f"_w{weights_key}" if weights_key else ""
        k_suffix = f"_k{k_neighbors}" if k_neighbors is not None else ""
        return f"frechet_{base_key}{weights_suffix}{k_suffix}"

    # --- Gram Matrix Cache ---

    def get_gram(self, key: str) -> "Array | None":
        """Get cached Gram matrix."""
        return self._get_from_cache(
            key, self._gram_cache, self._gram_order, self._gram_lock, "gram"
        )

    def set_gram(
        self, key: str, value: "Array", compute_time_ms: float = 0.0
    ) -> None:
        """Cache Gram matrix."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._gram_cache,
            self._gram_order,
            self._gram_lock,
            self._config.max_gram_entries,
        )

    def get_or_compute_gram(
        self,
        activations: "Array",
        backend: "Backend",
        kernel_type: str = "linear",
    ) -> "Array":
        """
        Get Gram matrix from cache or compute it.

        Args:
            activations: Input matrix [n_samples, n_features]
            backend: Backend for computation
            kernel_type: Type of kernel ("linear" for X @ X^T)

        Returns:
            Gram matrix [n_samples, n_samples]
        """
        key = self.make_gram_key(activations, backend, kernel_type)
        cached = self.get_gram(key)
        if cached is not None:
            return cached

        start = time.perf_counter()
        if kernel_type == "linear":
            gram = backend.matmul(activations, backend.transpose(activations))
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        backend.eval(gram)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.set_gram(key, gram, elapsed_ms)
        return gram

    # --- Centered Gram Matrix Cache ---

    def get_centered_gram(self, key: str) -> "Array | None":
        """Get cached centered Gram matrix."""
        return self._get_from_cache(
            key,
            self._centered_gram_cache,
            self._centered_gram_order,
            self._centered_gram_lock,
            "centered_gram",
        )

    def set_centered_gram(
        self, key: str, value: "Array", compute_time_ms: float = 0.0
    ) -> None:
        """Cache centered Gram matrix."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._centered_gram_cache,
            self._centered_gram_order,
            self._centered_gram_lock,
            self._config.max_centered_gram_entries,
        )

    # --- Geodesic Distance Cache ---

    def get_geodesic(self, key: str) -> Any | None:
        """Get cached geodesic distance result."""
        return self._get_from_cache(
            key,
            self._geodesic_cache,
            self._geodesic_order,
            self._geodesic_lock,
            "geodesic",
        )

    def set_geodesic(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache geodesic distance result."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._geodesic_cache,
            self._geodesic_order,
            self._geodesic_lock,
            self._config.max_geodesic_entries,
        )

    # --- SVD Cache ---

    def get_svd(self, key: str) -> tuple["Array", "Array", "Array"] | None:
        """Get cached SVD decomposition (U, S, Vt)."""
        return self._get_from_cache(
            key, self._svd_cache, self._svd_order, self._svd_lock, "svd"
        )

    def set_svd(
        self,
        key: str,
        value: tuple["Array", "Array", "Array"],
        compute_time_ms: float = 0.0,
    ) -> None:
        """Cache SVD decomposition."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._svd_cache,
            self._svd_order,
            self._svd_lock,
            self._config.max_svd_entries,
        )

    def get_or_compute_svd(
        self,
        matrix: "Array",
        backend: "Backend",
        full_matrices: bool = False,
    ) -> tuple["Array", "Array", "Array"]:
        """
        Get SVD from cache or compute it.

        Args:
            matrix: Input matrix
            backend: Backend for computation
            full_matrices: Whether to compute full matrices (may not be supported)

        Returns:
            Tuple of (U, S, Vt)
        """
        key = self.make_svd_key(matrix, backend, full_matrices)
        cached = self.get_svd(key)
        if cached is not None:
            return cached

        start = time.perf_counter()
        # Some backends don't support full_matrices param - try without if needed
        try:
            u, s, vt = backend.svd(matrix, full_matrices=full_matrices)
        except TypeError:
            u, s, vt = backend.svd(matrix)
        backend.eval(u, s, vt)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = (u, s, vt)
        self.set_svd(key, result, elapsed_ms)
        return result

    # --- Fréchet Mean Cache ---

    def get_frechet(self, key: str) -> Any | None:
        """Get cached Fréchet mean result."""
        return self._get_from_cache(
            key,
            self._frechet_cache,
            self._frechet_order,
            self._frechet_lock,
            "frechet",
        )

    def set_frechet(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache Fréchet mean result."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._frechet_cache,
            self._frechet_order,
            self._frechet_lock,
            self._config.max_frechet_entries,
        )

    # --- Internal Cache Operations ---

    def _get_from_cache(
        self,
        key: str,
        cache: dict[str, CacheEntry],
        order: list[str],
        lock: threading.Lock,
        cache_name: str,
    ) -> Any | None:
        """Get value from a specific cache."""
        with lock:
            if key in cache:
                entry = cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                # Move to end of order (most recently used)
                if key in order:
                    order.remove(key)
                order.append(key)

                with self._stats_lock:
                    self._stats.hits += 1
                    self._stats.total_compute_time_saved_ms += entry.compute_time_ms

                logger.debug("Cache hit (%s): %s", cache_name, key[:16])
                return entry.value

            with self._stats_lock:
                self._stats.misses += 1

            logger.debug("Cache miss (%s): %s", cache_name, key[:16])
            return None

    def _set_in_cache(
        self,
        key: str,
        value: Any,
        compute_time_ms: float,
        cache: dict[str, CacheEntry],
        order: list[str],
        lock: threading.Lock,
        max_entries: int,
    ) -> None:
        """Set value in a specific cache with LRU eviction."""
        with lock:
            # Create entry
            now = time.time()
            cache[key] = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                compute_time_ms=compute_time_ms,
            )
            order.append(key)

            # LRU eviction
            while len(cache) > max_entries:
                if not order:
                    break
                oldest_key = order.pop(0)
                cache.pop(oldest_key, None)
                with self._stats_lock:
                    self._stats.evictions += 1
                logger.debug("Cache eviction: %s", oldest_key[:16])

    # --- Statistics and Utilities ---

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._stats_lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_compute_time_saved_ms=self._stats.total_compute_time_saved_ms,
            )

    def clear_all(self) -> None:
        """Clear all caches."""
        with self._gram_lock:
            self._gram_cache.clear()
            self._gram_order.clear()

        with self._centered_gram_lock:
            self._centered_gram_cache.clear()
            self._centered_gram_order.clear()

        with self._geodesic_lock:
            self._geodesic_cache.clear()
            self._geodesic_order.clear()

        with self._svd_lock:
            self._svd_cache.clear()
            self._svd_order.clear()

        with self._frechet_lock:
            self._frechet_cache.clear()
            self._frechet_order.clear()

        with self._stats_lock:
            self._stats = CacheStats()

        logger.info("Cleared all computation caches")

    def get_cache_sizes(self) -> dict[str, int]:
        """Get the size of each cache."""
        return {
            "gram": len(self._gram_cache),
            "centered_gram": len(self._centered_gram_cache),
            "geodesic": len(self._geodesic_cache),
            "svd": len(self._svd_cache),
            "frechet": len(self._frechet_cache),
        }
