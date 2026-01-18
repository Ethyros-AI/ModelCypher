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

"""Base cache infrastructure for ModelCypher.

Provides a generic two-level cache with memory and disk backing,
following patterns from TokenCounterService and RefusalDirectionCache.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import xxhash

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TwoLevelCache(Generic[T]):
    """
    Thread-safe two-level cache with memory and disk backing.

    Features:
    - Memory LRU cache for fast access
    - Disk JSON persistence for durability
    - TTL-based expiry for disk cache
    - Thread-safe with locking
    - Working Set Size (WSS) based memory limits (Denning 1968)

    Memory limit is derived from Working Set Theory (Denning 1968):
    - Tracks access patterns over a time window
    - Adjusts limit to accommodate the working set + 20% headroom
    - Avoids thrashing while not over-allocating

    Example:
        cache = TwoLevelCache(
            cache_directory=Path("~/.cache/myapp"),
            serializer=lambda x: x.to_dict(),
            deserializer=MyClass.from_dict,
        )
        cache.set("key", value)
        value = cache.get("key")
    """

    # WSS window: 60 seconds (derived from typical cache access patterns)
    # Beyond this window, items are less likely to be re-accessed soon.
    _WSS_WINDOW_SECONDS: float = 60.0

    # Minimum cache size - prevents thrashing on cold start
    _MIN_MEMORY_LIMIT: int = 10

    def __init__(
        self,
        cache_directory: Path,
        serializer: Callable[[T], dict],
        deserializer: Callable[[dict], T],
        memory_limit: int = 100,
        disk_ttl_seconds: float = 7 * 24 * 60 * 60,
        cache_version: int = 1,
    ):
        """
        Initialize the cache.

        Args:
            cache_directory: Directory for disk cache files
            serializer: Function to convert T to dict for JSON storage
            deserializer: Function to convert dict back to T
            memory_limit: Initial memory limit before WSS data is available.
                Will be adjusted based on Working Set Size (Denning 1968).
            disk_ttl_seconds: Time-to-live for disk cache entries in seconds
            cache_version: Cache format version for invalidation on schema changes
        """
        self.cache_directory = cache_directory
        self._serializer = serializer
        self._deserializer = deserializer
        self._memory_limit = memory_limit
        self._disk_ttl_seconds = disk_ttl_seconds
        self._cache_version = cache_version
        self._memory_cache: dict[str, tuple[T, float]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        # Track access times for WSS computation
        self._access_times: dict[str, float] = {}

    def _update_wss_limit(self) -> None:
        """Update memory limit based on Working Set Size (must hold lock).

        Working Set Theory (Denning 1968): cache size should accommodate
        the number of distinct items accessed within a time window.
        Beyond WSS, doubling cache yields <5% hit rate improvement.
        """
        now = time.time()
        cutoff = now - self._WSS_WINDOW_SECONDS

        # Count keys accessed within window (the working set)
        wss = sum(1 for t in self._access_times.values() if t > cutoff)

        # Set limit to WSS + 20% headroom to avoid thrashing at boundary
        # The 1.2 factor is from cache theory: small headroom prevents
        # oscillation when WSS is exactly at the limit.
        new_limit = max(self._MIN_MEMORY_LIMIT, int(wss * 1.2))

        if new_limit != self._memory_limit:
            logger.debug(
                "WSS-derived memory_limit: %d -> %d (wss=%d)",
                self._memory_limit, new_limit, wss
            )
            self._memory_limit = new_limit

    def get(self, key: str) -> T | None:
        """
        Get value from cache (memory first, then disk).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            # Track access for WSS computation
            self._access_times[key] = time.time()
            self._update_wss_limit()

            # Check memory cache
            if key in self._memory_cache:
                value, _timestamp = self._memory_cache[key]
                self._touch(key)
                logger.debug("Cache hit (memory): %s", key)
                return value

        # Check disk cache (outside lock for I/O)
        cache_file = self._cache_file(key)
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))

                # Check version
                if payload.get("version") != self._cache_version:
                    logger.debug("Cache version mismatch for %s", key)
                    return None

                # Check TTL
                cached_at = payload.get("cached_at", 0)
                if time.time() - cached_at > self._disk_ttl_seconds:
                    logger.debug("Cache expired for %s", key)
                    cache_file.unlink()
                    return None

                value = self._deserializer(payload["data"])

                # Promote to memory cache
                with self._lock:
                    self._memory_cache[key] = (value, time.time())
                    self._touch(key)
                    self._trim()

                logger.debug("Cache hit (disk): %s", key)
                return value

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning("Failed to load cache file %s: %s", key, e)
                return None

        logger.debug("Cache miss: %s", key)
        return None

    def set(self, key: str, value: T) -> None:
        """
        Set value in both memory and disk cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Track access for WSS computation
            self._access_times[key] = time.time()
            self._update_wss_limit()

            self._memory_cache[key] = (value, time.time())
            self._touch(key)
            self._trim()

        # Write to disk (outside lock for I/O)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_file(key)
        try:
            payload = {
                "version": self._cache_version,
                "cached_at": time.time(),
                "data": self._serializer(value),
            }
            cache_file.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.debug("Cached to disk: %s", key)
        except (OSError, TypeError) as e:
            logger.warning("Failed to write cache file %s: %s", key, e)

    def invalidate(self, key: str) -> None:
        """
        Remove entry from both caches.

        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            self._memory_cache.pop(key, None)
            if key in self._order:
                self._order.remove(key)

        cache_file = self._cache_file(key)
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Invalidated cache key: %s", key)

    def clear_all(self) -> None:
        """Clear all cached values from memory and disk."""
        with self._lock:
            self._memory_cache.clear()
            self._order.clear()
            self._access_times.clear()

        if self.cache_directory.exists():
            for file_path in self.cache_directory.glob("*.json"):
                try:
                    file_path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete cache file %s: %s", file_path, e)

        logger.info("Cleared all caches in %s", self.cache_directory)

    def list_cached(self) -> list[str]:
        """
        List all cached keys.

        Returns:
            List of cache keys
        """
        keys: set[str] = set()

        with self._lock:
            keys.update(self._memory_cache.keys())

        if self.cache_directory.exists():
            for file_path in self.cache_directory.glob("*.json"):
                keys.add(file_path.stem)

        return sorted(keys)

    def _cache_file(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self.cache_directory / f"{key}.json"

    def _touch(self, key: str) -> None:
        """Move key to end of LRU order (must hold lock)."""
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _trim(self) -> None:
        """Trim memory cache to limit (must hold lock)."""
        while len(self._memory_cache) > self._memory_limit:
            if not self._order:
                break
            oldest = self._order.pop(0)
            self._memory_cache.pop(oldest, None)
            logger.debug("Evicted from memory cache: %s", oldest)

        # Clean up stale access times (outside WSS window) to prevent unbounded growth
        now = time.time()
        cutoff = now - self._WSS_WINDOW_SECONDS * 2  # Keep 2x window for hysteresis
        stale_keys = [k for k, t in self._access_times.items() if t < cutoff]
        for k in stale_keys:
            del self._access_times[k]


def content_hash(data: Any) -> str:
    """
    Create a deterministic hash of data for cache keys.

    Uses xxhash for ~10-50× faster hashing than SHA256.
    Not cryptographically secure, but that's not needed for cache keys.

    Args:
        data: Data to hash (must be JSON-serializable)

    Returns:
        16-character hex hash
    """
    content = json.dumps(data, sort_keys=True, default=str)
    return xxhash.xxh64(content.encode()).hexdigest()[:16]
