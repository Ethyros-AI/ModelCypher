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

"""Token counter service with LRU cache.

Provides token counting with caching for dataset and model size estimation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TextTokenizer(Protocol):
    """Protocol for text tokenizers."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...


class LRUCache:
    """Thread-safe LRU cache for token counts."""

    def __init__(self, limit: int = 1000):
        """Initialize cache.

        Args:
            limit: Maximum entries in cache.
        """
        self._limit = max(1, limit)
        self._order: list[str] = []
        self._storage: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> int | None:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        with self._lock:
            if key not in self._storage:
                return None

            # Move to end (most recently used)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

            return self._storage[key]

    def set(self, key: str, value: int) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            self._storage[key] = value

            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

            self._trim()

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._order.clear()
            self._storage.clear()

    def _trim(self) -> None:
        """Trim cache to limit."""
        while len(self._storage) > self._limit:
            if not self._order:
                break
            oldest = self._order.pop(0)
            self._storage.pop(oldest, None)

    @property
    def count(self) -> int:
        """Number of entries in cache."""
        with self._lock:
            return len(self._storage)


def _sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TokenCounterConfig:
    """Configuration for token counter service."""

    cache_limit: int = 1000
    """Maximum entries in memory cache."""

    disk_cache_path: Path | None = None
    """Path for disk cache (optional)."""

    auto_save_interval: float = 120.0
    """Auto-save interval in seconds."""


class TokenCounterService:
    """Token counting service with LRU cache.

    Provides accurate token counts with caching for efficiency.
    """

    CACHE_VERSION = 3

    def __init__(
        self,
        tokenizer: TextTokenizer | None = None,
        token_estimator: Callable[[str], int] | None = None,
        config: TokenCounterConfig | None = None,
    ):
        """Initialize token counter.

        Args:
            tokenizer: Optional tokenizer for accurate counts.
            token_estimator: Optional fallback estimator (default: len/4).
            config: Configuration options.
        """
        self._tokenizer = tokenizer
        self._token_estimator = token_estimator or (lambda s: len(s) // 4)
        self._config = config or TokenCounterConfig()
        self._memory_cache = LRUCache(limit=self._config.cache_limit)
        self._disk_cache: dict[str, int] = {}
        self._disk_cache_dirty = False
        self._lock = threading.Lock()

        # Load disk cache if configured
        if self._config.disk_cache_path:
            self._load_disk_cache()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count.

        Returns:
            Token count.
        """
        if not text:
            return 0

        # Check memory cache
        cached = self._memory_cache.get(text)
        if cached is not None:
            return cached

        # Check disk cache
        content_hash = _sha256(text)
        with self._lock:
            if content_hash in self._disk_cache:
                count = self._disk_cache[content_hash]
                self._memory_cache.set(text, count)
                return count

        # Count tokens
        if self._tokenizer:
            count = len(self._tokenizer.encode(text))
        else:
            count = self._token_estimator(text)

        # Cache result
        self._memory_cache.set(text, count)
        with self._lock:
            self._disk_cache[content_hash] = count
            self._disk_cache_dirty = True

        return count

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts.

        Args:
            texts: List of texts to count.

        Returns:
            List of token counts.
        """
        return [self.count_tokens(text) for text in texts]

    def estimate_dataset_tokens(
        self,
        samples: list[str],
        sample_size: int = 100,
    ) -> tuple[int, float]:
        """Estimate total tokens in a dataset.

        Uses sampling for large datasets.

        Args:
            samples: List of sample texts.
            sample_size: Number of samples to use for estimation.

        Returns:
            Tuple of (estimated_total, average_per_sample).
        """
        if not samples:
            return 0, 0.0

        # Sample if dataset is large
        if len(samples) > sample_size:
            import random

            sample_texts = random.sample(samples, sample_size)
        else:
            sample_texts = samples

        counts = self.count_tokens_batch(sample_texts)
        avg = sum(counts) / len(counts) if counts else 0

        estimated_total = int(avg * len(samples))
        return estimated_total, avg

    def set_tokenizer(self, tokenizer: TextTokenizer) -> None:
        """Set or update the tokenizer.

        Clears caches when tokenizer changes.

        Args:
            tokenizer: New tokenizer to use.
        """
        self._tokenizer = tokenizer
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        with self._lock:
            self._disk_cache.clear()
            self._disk_cache_dirty = True
        logger.debug("Token count caches cleared")

    def save_cache_to_disk(self) -> None:
        """Save disk cache to persistent storage."""
        if not self._config.disk_cache_path:
            return

        with self._lock:
            if not self._disk_cache_dirty:
                return

            try:
                self._config.disk_cache_path.parent.mkdir(parents=True, exist_ok=True)

                payload = {
                    "version": self.CACHE_VERSION,
                    "counts": self._disk_cache,
                }

                with open(self._config.disk_cache_path, "w") as f:
                    json.dump(payload, f)

                self._disk_cache_dirty = False
                logger.debug(f"Saved {len(self._disk_cache)} token counts to disk")

            except OSError as e:
                logger.error(f"Failed to save token count cache: {e}")

    def _load_disk_cache(self) -> None:
        """Load disk cache from persistent storage."""
        if not self._config.disk_cache_path:
            return

        if not self._config.disk_cache_path.exists():
            logger.debug("No existing token count cache found, starting fresh")
            return

        try:
            with open(self._config.disk_cache_path, "r") as f:
                payload = json.load(f)

            version = payload.get("version", 0)
            if version != self.CACHE_VERSION:
                logger.debug(
                    f"Discarded token count cache due to schema mismatch "
                    f"(found v{version}, expected v{self.CACHE_VERSION})"
                )
                self._disk_cache_dirty = True
                return

            self._disk_cache = payload.get("counts", {})
            logger.debug(
                f"Loaded {len(self._disk_cache)} cached token counts from disk"
            )

        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to load token count cache: {e}")
            self._disk_cache_dirty = True

    @property
    def cache_count(self) -> int:
        """Number of entries in memory cache."""
        return self._memory_cache.count

    @property
    def disk_cache_count(self) -> int:
        """Number of entries in disk cache."""
        with self._lock:
            return len(self._disk_cache)


# Default instance (can be configured globally)
_default_service: TokenCounterService | None = None


def get_token_counter_service() -> TokenCounterService:
    """Get the default token counter service.

    Returns:
        Default service instance.
    """
    global _default_service
    if _default_service is None:
        _default_service = TokenCounterService()
    return _default_service


def set_token_counter_service(service: TokenCounterService) -> None:
    """Set the default token counter service.

    Args:
        service: Service instance to use as default.
    """
    global _default_service
    _default_service = service


def reset_token_counter_service() -> None:
    """Reset the default token counter service (for testing)."""
    global _default_service
    _default_service = None
