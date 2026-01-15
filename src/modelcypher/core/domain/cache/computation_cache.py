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

import logging
import struct
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import xxhash

from modelcypher.core.domain.geometry.numerical_stability import geodesic_svd

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


class ComputationCache:
    """
    Session-scoped in-memory cache for expensive tensor computations.

    Provides separate LRU caches for different computation types:
    - Gram matrices (X @ X^T)
    - Geodesic distance matrices
    - Geodesic null-space bases
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

    def __init__(
        self,
        max_gram_entries: int = 200,
        max_geodesic_entries: int = 1024,
        max_svd_entries: int = 128,
        max_frechet_entries: int = 1024,
        max_basis_entries: int = 256,
        max_kmin_entries: int = 1024,
        max_centered_gram_entries: int = 200,
        max_chord_entries: int = 256,
        max_stitch_entries: int = 128,
        max_pinv_entries: int = 64,
    ) -> None:
        """
        Initialize the computation cache.

        Default sizes are tuned for typical analysis sessions on systems with
        16-64 GB RAM. The sizing rationale:

        - Gram/centered_gram (200 each): O(n²) memory per entry where n=samples.
          For 1024 samples: ~8MB per entry × 200 = ~1.6GB max.
        - SVD (128): Stores U, S, Vt. For 1024×4096: ~150MB per entry × 128 = ~19GB max.
        - Geodesic/frechet/kmin (1024): Sparse or scalar results, ~1KB-1MB each.
        - Basis (256): Orthonormal bases, ~10MB per entry × 256 = ~2.5GB max.
        - Chord/stitch (256/128): Distance matrices and transforms, moderate size.
        - Pinv (64): Pseudoinverses can be large, limited count.

        To auto-derive from available memory:
            available_mb = 8000  # or detect at runtime
            entry_size_kb = 8000  # estimate for your use case
            max_entries = available_mb * 1024 // entry_size_kb

        Args:
            max_gram_entries: Maximum Gram matrix entries. Large n² matrices.
            max_geodesic_entries: Maximum geodesic distance entries. Sparse results.
            max_svd_entries: Maximum SVD entries. Stores (U, S, Vt) tuples.
            max_frechet_entries: Maximum Fréchet mean entries. Single vectors.
            max_basis_entries: Maximum geodesic basis entries. Orthonormal matrices.
            max_kmin_entries: Maximum k-min lookup entries. Scalar integers.
            max_centered_gram_entries: Maximum centered Gram entries. Like Gram.
            max_chord_entries: Maximum chord distance entries. n² dense matrices.
            max_stitch_entries: Maximum stitch transform entries. Transform matrices.
            max_pinv_entries: Maximum pseudoinverse entries. Can be very large.
        """
        self._max_gram_entries = max_gram_entries
        self._max_geodesic_entries = max_geodesic_entries
        self._max_svd_entries = max_svd_entries
        self._max_frechet_entries = max_frechet_entries
        self._max_basis_entries = max_basis_entries
        self._max_kmin_entries = max_kmin_entries
        self._max_centered_gram_entries = max_centered_gram_entries
        self._max_chord_entries = max_chord_entries
        self._max_stitch_entries = max_stitch_entries
        self._max_pinv_entries = max_pinv_entries

        # Separate LRU caches for different computation types
        # Using OrderedDict for O(1) move_to_end() and eviction
        self._gram_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._gram_lock = threading.Lock()

        self._centered_gram_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._centered_gram_lock = threading.Lock()

        self._chord_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._chord_lock = threading.Lock()

        self._stitch_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stitch_lock = threading.Lock()

        self._geodesic_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._geodesic_lock = threading.Lock()

        self._svd_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._svd_lock = threading.Lock()

        self._frechet_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._frechet_lock = threading.Lock()

        self._basis_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._basis_lock = threading.Lock()

        self._kmin_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._kmin_lock = threading.Lock()

        self._pinv_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._pinv_lock = threading.Lock()

        self._stats = CacheStats()
        self._stats_lock = threading.Lock()

        # id() → cache_key fast-path for arrays still in memory
        # Maps id(array) → (weakref or None, cache_key)
        # Weakref ensures we detect when array is GC'd and id reused
        self._id_cache: dict[int, tuple[weakref.ref | None, str]] = {}
        self._id_cache_lock = threading.Lock()
        self._max_id_cache_entries = 500

    # --- Key Generation ---

    def _backend_id(self, backend: "Backend") -> str:
        """Get a short identifier for the backend type.

        This ensures cached values are not returned to a different backend,
        which would cause type errors (e.g., MLX array passed to JAX function).
        """
        return type(backend).__name__[:3].lower()  # "mlx", "jax", "cud"

    def make_array_key(self, arr: "Array", backend: "Backend") -> str:
        """
        Create a hash key from an array's content.

        Uses id()-based fast-path caching for arrays still in memory,
        falling back to shape + sampled values hashing for new arrays.

        Args:
            arr: Input array
            backend: Backend for array operations

        Returns:
            16-character hex hash
        """
        arr_id = id(arr)

        # Fast path: check if we've seen this exact array object before
        with self._id_cache_lock:
            if arr_id in self._id_cache:
                ref, cached_key = self._id_cache[arr_id]
                # If weakref exists, verify it still points to same object
                if ref is None or ref() is arr:
                    return cached_key
                # Array was GC'd and id reused - remove stale entry
                del self._id_cache[arr_id]

        # Slow path: compute the hash
        key = self._compute_array_key(arr, backend)

        # Cache for future lookups
        self._cache_array_id(arr_id, arr, key)

        return key

    def _compute_array_key(self, arr: "Array", backend: "Backend") -> str:
        """Compute cache key by hashing array content (slow path)."""
        backend.eval(arr)
        shape = tuple(int(d) for d in arr.shape)
        dtype = str(backend.dtype(arr))
        n_elements = 1
        for d in shape:
            n_elements *= d

        # For efficiency, hash shape + sampled values instead of all values
        # Sample corners and center for large arrays
        flat = backend.reshape(arr, (-1,))
        flat_len = int(flat.shape[0])

        if n_elements <= 1000:
            # Small array - hash all values directly as bytes (not hex string!)
            # This is ~8× faster than converting to hex and back
            # Use native tolist() for O(1) extraction instead of O(n) scalar extractions
            shape_bytes = f"{shape}|dtype={dtype}".encode()
            flat_list = backend.tolist(flat)
            content_bytes = b"".join(
                struct.pack(">d", float(val)) for val in flat_list
            )
            # Hash shape + content bytes directly (avoids hex conversion overhead)
            # xxhash is ~10-50× faster than SHA256 for non-cryptographic hashing
            return xxhash.xxh64(shape_bytes + content_bytes).hexdigest()[:16]
        else:
            # Large array - sample BEFORE tolist to avoid full array conversion
            # This is the critical optimization: we only extract ~40 values instead of all
            
            # Build sample indices
            sample_indices = []
            # First 8 indices
            sample_indices.extend(range(min(8, flat_len)))
            # Last 8 indices
            sample_indices.extend(range(max(0, flat_len - 8), flat_len))
            # Middle 8 indices
            mid = flat_len // 2
            for offset in range(-4, 4):
                idx = mid + offset
                if 0 <= idx < flat_len and idx not in sample_indices:
                    sample_indices.append(idx)
            # Strided samples (deterministic "random")
            step = max(1, flat_len // 16)
            for i in range(0, flat_len, step):
                if i not in sample_indices:
                    sample_indices.append(i)
                if len(sample_indices) >= 40:
                    break
            
            # Extract only the samples we need using backend.take
            indices_arr = backend.array(sample_indices)
            samples_arr = backend.take(flat, indices_arr, axis=0)
            backend.eval(samples_arr)
            samples = backend.tolist(samples_arr)
            
            shape_bytes = f"{shape}|dtype={dtype}".encode()
            sample_bytes = b"".join(struct.pack(">d", float(val)) for val in samples)
            return xxhash.xxh64(shape_bytes + sample_bytes).hexdigest()[:16]

    def _cache_array_id(self, arr_id: int, arr: "Array", key: str) -> None:
        """Cache id(array) → key mapping with LRU eviction."""
        # Try to create a weakref for GC detection
        try:
            ref: weakref.ref | None = weakref.ref(arr)
        except TypeError:
            # Some array types don't support weakref
            ref = None

        with self._id_cache_lock:
            self._id_cache[arr_id] = (ref, key)

            # Simple LRU: if too many entries, remove oldest
            if len(self._id_cache) > self._max_id_cache_entries:
                # Remove first entry (oldest by insertion order in Python 3.7+)
                oldest_id = next(iter(self._id_cache))
                del self._id_cache[oldest_id]

    def clear_id_cache(self) -> None:
        """Clear the id() → key cache (for testing)."""
        with self._id_cache_lock:
            self._id_cache.clear()

    def make_gram_key(
        self,
        arr: "Array",
        backend: "Backend",
        kernel_type: str = "linear",
    ) -> str:
        """Create cache key for Gram matrix computation."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"gram_{bid}_{kernel_type}_{base_key}"

    def make_centered_gram_key(self, gram_key: str) -> str:
        """Create cache key for centered Gram matrix."""
        return f"centered_{gram_key}"

    def make_chord_key(self, arr: "Array", backend: "Backend") -> str:
        """Create cache key for chord (Euclidean) distance matrix.

        Unlike geodesic keys which include k_neighbors, chord distance
        is computed from raw points and can be reused across different k values.
        """
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"chord_{bid}_{base_key}"

    def make_stitch_key(
        self,
        source_gram: "Array",
        target_gram: "Array",
        backend: "Backend",
    ) -> str:
        """Create cache key for stitch (alignment) transform.

        Stitch transforms are keyed by both source and target Gram matrices,
        enabling reuse when the same alignment is needed multiple times.
        """
        source_key = self.make_array_key(source_gram, backend)
        target_key = self.make_array_key(target_gram, backend)
        bid = self._backend_id(backend)
        return f"stitch_{bid}_{source_key}_{target_key}"

    def make_geodesic_key(
        self,
        arr: "Array",
        backend: "Backend",
        k_neighbors: int,
    ) -> str:
        """Create cache key for geodesic distance matrix."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"geodesic_{bid}_k{k_neighbors}_{base_key}"

    def make_basis_key(
        self,
        arr: "Array",
        backend: "Backend",
        k_neighbors: int | None,
    ) -> str:
        """Create cache key for geodesic null-space basis."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        k_tag = "auto" if k_neighbors is None else f"k{k_neighbors}"
        return f"basis_{bid}_{k_tag}_{base_key}"

    def make_kmin_key(self, arr: "Array", backend: "Backend") -> str:
        """Create cache key for minimal connected k lookup."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"kmin_{bid}_{base_key}"

    def make_svd_key(
        self,
        arr: "Array",
        backend: "Backend",
        full_matrices: bool = False,
    ) -> str:
        """Create cache key for SVD computation."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"svd_{bid}_full{full_matrices}_{base_key}"

    def make_frechet_key(
        self,
        arr: "Array",
        backend: "Backend",
        weights_key: str | None = None,
        k_neighbors: int | None = None,
    ) -> str:
        """Create cache key for Fréchet mean computation."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        weights_suffix = f"_w{weights_key}" if weights_key else ""
        k_suffix = f"_k{k_neighbors}" if k_neighbors is not None else ""
        return f"frechet_{bid}_{base_key}{weights_suffix}{k_suffix}"

    # --- Gram Matrix Cache ---

    def get_gram(self, key: str) -> "Array | None":
        """Get cached Gram matrix."""
        return self._get_from_cache(key, self._gram_cache, self._gram_lock, "gram")

    def set_gram(
        self, key: str, value: "Array", compute_time_ms: float = 0.0
    ) -> None:
        """Cache Gram matrix."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._gram_cache,
            self._gram_lock,
            self._max_gram_entries,
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
        elif kernel_type == "geodesic_cosine":
            from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix

            gram = geodesic_cosine_matrix(activations, backend)
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
            self._centered_gram_lock,
            self._max_centered_gram_entries,
        )

    # --- Chord Distance Cache ---

    def get_chord(self, key: str) -> Any | None:
        """Get cached chord distance matrix.

        Chord distance is the pairwise Euclidean distance matrix,
        reusable across different k values for geodesic computation.
        """
        return self._get_from_cache(
            key,
            self._chord_cache,
            self._chord_lock,
            "chord",
        )

    def set_chord(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache chord distance matrix."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._chord_cache,
            self._chord_lock,
            self._max_chord_entries,
        )

    # --- Stitch Transform Cache ---

    def get_stitch(self, key: str) -> Any | None:
        """Get cached stitch (alignment) transform.

        Stitch transforms map source Gram geometry to target Gram geometry.
        Caching avoids repeated eigendecomposition for same Gram pairs.
        """
        return self._get_from_cache(
            key,
            self._stitch_cache,
            self._stitch_lock,
            "stitch",
        )

    def set_stitch(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache stitch transform."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._stitch_cache,
            self._stitch_lock,
            self._max_stitch_entries,
        )

    # --- Geodesic Distance Cache ---

    def get_geodesic(self, key: str) -> Any | None:
        """Get cached geodesic distance result."""
        return self._get_from_cache(
            key,
            self._geodesic_cache,
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
            self._geodesic_lock,
            self._max_geodesic_entries,
        )

    # --- Geodesic Basis Cache ---

    def get_basis(self, key: str) -> Any | None:
        """Get cached geodesic basis result."""
        return self._get_from_cache(
            key,
            self._basis_cache,
            self._basis_lock,
            "basis",
        )

    def set_basis(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache geodesic basis result."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._basis_cache,
            self._basis_lock,
            self._max_basis_entries,
        )

    # --- k-min Cache ---

    def get_kmin(self, key: str) -> Any | None:
        """Get cached minimum connected k."""
        return self._get_from_cache(
            key,
            self._kmin_cache,
            self._kmin_lock,
            "kmin",
        )

    def set_kmin(
        self, key: str, value: Any, compute_time_ms: float = 0.0
    ) -> None:
        """Cache minimum connected k."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._kmin_cache,
            self._kmin_lock,
            self._max_kmin_entries,
        )

    # --- Pseudoinverse Cache ---

    def make_pinv_key(self, arr: "Array", backend: "Backend") -> str:
        """Create cache key for pseudoinverse computation."""
        base_key = self.make_array_key(arr, backend)
        bid = self._backend_id(backend)
        return f"pinv_{bid}_{base_key}"

    def get_pinv(self, key: str) -> "Array | None":
        """Get cached pseudoinverse."""
        return self._get_from_cache(key, self._pinv_cache, self._pinv_lock, "pinv")

    def set_pinv(
        self, key: str, value: "Array", compute_time_ms: float = 0.0
    ) -> None:
        """Cache pseudoinverse."""
        self._set_in_cache(
            key,
            value,
            compute_time_ms,
            self._pinv_cache,
            self._pinv_lock,
            self._max_pinv_entries,
        )

    def get_or_compute_pinv(
        self,
        matrix: "Array",
        backend: "Backend",
    ) -> "Array":
        """Get pseudoinverse from cache or compute it.

        Args:
            matrix: Input matrix
            backend: Backend for computation

        Returns:
            Pseudoinverse of the matrix
        """
        import time as time_module

        key = self.make_pinv_key(matrix, backend)
        cached = self.get_pinv(key)
        if cached is not None:
            return cached

        start = time_module.perf_counter()
        pinv = backend.pinv(matrix)
        backend.eval(pinv)
        elapsed_ms = (time_module.perf_counter() - start) * 1000

        self.set_pinv(key, pinv, elapsed_ms)
        return pinv

    # --- SVD Cache ---

    def get_svd(self, key: str) -> tuple["Array", "Array", "Array"] | None:
        """Get cached SVD decomposition (U, S, Vt)."""
        return self._get_from_cache(key, self._svd_cache, self._svd_lock, "svd")

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
            self._svd_lock,
            self._max_svd_entries,
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
        # Use geodesic SVD (GPU-only) - no full_matrices param needed
        u, s, vt = geodesic_svd(backend, matrix)
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
            self._frechet_lock,
            self._max_frechet_entries,
        )

    # --- Internal Cache Operations ---

    def _get_from_cache(
        self,
        key: str,
        cache: OrderedDict[str, CacheEntry],
        lock: threading.Lock,
        cache_name: str,
    ) -> Any | None:
        """Get value from a specific cache."""
        with lock:
            if key in cache:
                entry = cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                # O(1) move to end (most recently used)
                cache.move_to_end(key)

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
        cache: OrderedDict[str, CacheEntry],
        lock: threading.Lock,
        max_entries: int,
    ) -> None:
        """Set value in a specific cache with LRU eviction."""
        with lock:
            # Create entry (automatically added at end of OrderedDict)
            now = time.time()
            cache[key] = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                compute_time_ms=compute_time_ms,
            )

            # O(1) LRU eviction - remove from front
            while len(cache) > max_entries:
                # Get oldest key (first item in OrderedDict)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
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

        with self._centered_gram_lock:
            self._centered_gram_cache.clear()

        with self._geodesic_lock:
            self._geodesic_cache.clear()

        with self._svd_lock:
            self._svd_cache.clear()

        with self._frechet_lock:
            self._frechet_cache.clear()

        with self._basis_lock:
            self._basis_cache.clear()

        with self._kmin_lock:
            self._kmin_cache.clear()

        with self._pinv_lock:
            self._pinv_cache.clear()

        with self._stats_lock:
            self._stats = CacheStats()

        with self._id_cache_lock:
            self._id_cache.clear()

        if logger.isEnabledFor(logging.INFO):
            try:
                logger.info("Cleared all computation caches")
            except ValueError:
                # Logging handlers can be closed during test teardown.
                pass

    def clear_geometry_caches(self) -> None:
        """Clear geometry-heavy caches (O(n^2) buffers)."""
        with self._gram_lock:
            self._gram_cache.clear()

        with self._centered_gram_lock:
            self._centered_gram_cache.clear()

        with self._geodesic_lock:
            self._geodesic_cache.clear()

        with self._chord_lock:
            self._chord_cache.clear()

        with self._basis_lock:
            self._basis_cache.clear()

        with self._kmin_lock:
            self._kmin_cache.clear()

        if logger.isEnabledFor(logging.INFO):
            try:
                logger.info("Cleared geometry computation caches")
            except ValueError:
                # Logging handlers can be closed during test teardown.
                pass

    def get_cache_sizes(self) -> dict[str, int]:
        """Get the size of each cache."""
        return {
            "gram": len(self._gram_cache),
            "centered_gram": len(self._centered_gram_cache),
            "geodesic": len(self._geodesic_cache),
            "svd": len(self._svd_cache),
            "frechet": len(self._frechet_cache),
            "basis": len(self._basis_cache),
            "kmin": len(self._kmin_cache),
            "pinv": len(self._pinv_cache),
        }
