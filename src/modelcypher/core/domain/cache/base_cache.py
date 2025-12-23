"""Base cache infrastructure for ModelCypher.

Provides a generic two-level cache with memory and disk backing,
following patterns from TokenCounterService and RefusalDirectionCache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache behavior."""

    memory_limit: int = 100
    """Maximum number of entries in memory cache."""

    disk_ttl_seconds: float = 7 * 24 * 60 * 60  # 7 days
    """Time-to-live for disk cache entries in seconds."""

    cache_version: int = 1
    """Cache format version for invalidation on schema changes."""


class TwoLevelCache(Generic[T]):
    """
    Thread-safe two-level cache with memory and disk backing.

    Features:
    - Memory LRU cache for fast access
    - Disk JSON persistence for durability
    - TTL-based expiry for disk cache
    - Thread-safe with locking
    - Configurable limits and TTL

    Example:
        cache = TwoLevelCache(
            cache_directory=Path("~/.cache/myapp"),
            serializer=lambda x: x.to_dict(),
            deserializer=MyClass.from_dict,
        )
        cache.set("key", value)
        value = cache.get("key")
    """

    def __init__(
        self,
        cache_directory: Path,
        serializer: Callable[[T], dict],
        deserializer: Callable[[dict], T],
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize the cache.

        Args:
            cache_directory: Directory for disk cache files
            serializer: Function to convert T to dict for JSON storage
            deserializer: Function to convert dict back to T
            config: Cache configuration (uses defaults if None)
        """
        self.cache_directory = cache_directory
        self._serializer = serializer
        self._deserializer = deserializer
        self._config = config or CacheConfig()
        self._memory_cache: dict[str, tuple[T, float]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache (memory first, then disk).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
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
                if payload.get("version") != self._config.cache_version:
                    logger.debug("Cache version mismatch for %s", key)
                    return None

                # Check TTL
                cached_at = payload.get("cached_at", 0)
                if time.time() - cached_at > self._config.disk_ttl_seconds:
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
            self._memory_cache[key] = (value, time.time())
            self._touch(key)
            self._trim()

        # Write to disk (outside lock for I/O)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_file(key)
        try:
            payload = {
                "version": self._config.cache_version,
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
        while len(self._memory_cache) > self._config.memory_limit:
            if not self._order:
                break
            oldest = self._order.pop(0)
            self._memory_cache.pop(oldest, None)
            logger.debug("Evicted from memory cache: %s", oldest)


def content_hash(data: Any) -> str:
    """
    Create a deterministic hash of data for cache keys.

    Args:
        data: Data to hash (must be JSON-serializable)

    Returns:
        16-character hex hash
    """
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
