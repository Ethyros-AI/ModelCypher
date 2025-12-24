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

"""Memory-bounded streaming shuffler for large datasets.

Provides approximate shuffling with bounded memory usage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ShufflerEntry(Generic[T]):
    """Entry in the shuffle buffer."""

    item: T
    """The item."""

    byte_size: int
    """Estimated byte size."""


class StreamingShuffler(Generic[T]):
    """Memory-bounded streaming shuffler.

    Uses reservoir sampling with bounded memory to provide approximate
    shuffling of streaming data. Items are emitted when buffer limits
    are exceeded.

    This is useful for shuffling datasets that don't fit in memory.
    The quality of the shuffle depends on buffer size - larger buffers
    produce better shuffles but use more memory.
    """

    def __init__(
        self,
        count_limit: int = 10000,
        byte_limit: int = 100_000_000,
        size_estimator: Any = None,
    ):
        """Initialize shuffler.

        Args:
            count_limit: Maximum items in buffer.
            byte_limit: Maximum bytes in buffer.
            size_estimator: Optional function to estimate item size.
        """
        self._max_count = max(1, count_limit)
        self._max_bytes = max(1, byte_limit)
        self._buffer: list[ShufflerEntry[T]] = []
        self._buffer_bytes = 0
        self._size_estimator = size_estimator or self._default_size_estimate

    def _default_size_estimate(self, item: T) -> int:
        """Default size estimation."""
        if isinstance(item, str):
            return len(item.encode("utf-8"))
        if isinstance(item, bytes):
            return len(item)
        if isinstance(item, dict):
            # Rough estimate for dict
            import json
            try:
                return len(json.dumps(item))
            except (TypeError, ValueError):
                return 100
        return 100  # Default estimate

    def push(self, item: T) -> list[T]:
        """Push an item into the buffer.

        Args:
            item: Item to add.

        Returns:
            List of items emitted due to buffer overflow (may be empty).
        """
        byte_size = self._size_estimator(item)
        entry = ShufflerEntry(item=item, byte_size=byte_size)

        self._buffer.append(entry)
        self._buffer_bytes += byte_size

        spills: list[T] = []

        # Emit items if buffer limits exceeded
        while (
            len(self._buffer) > self._max_count
            or self._buffer_bytes > self._max_bytes
        ):
            if not self._buffer:
                break

            # Randomly select an item to emit
            swap_index = random.randint(0, len(self._buffer) - 1)
            self._buffer[swap_index], self._buffer[-1] = (
                self._buffer[-1],
                self._buffer[swap_index],
            )

            removed = self._buffer.pop()
            self._buffer_bytes -= removed.byte_size
            spills.append(removed.item)

        return spills

    def drain(self) -> list[T]:
        """Drain all remaining items in shuffled order.

        Returns:
            All remaining items, shuffled.
        """
        if not self._buffer:
            return []

        items = [entry.item for entry in self._buffer]
        random.shuffle(items)

        self._buffer.clear()
        self._buffer_bytes = 0

        return items

    @property
    def buffer_count(self) -> int:
        """Number of items in buffer."""
        return len(self._buffer)

    @property
    def buffer_bytes(self) -> int:
        """Bytes in buffer."""
        return self._buffer_bytes


def shuffle_streaming(
    items: list[T],
    count_limit: int = 10000,
    byte_limit: int = 100_000_000,
) -> list[T]:
    """Shuffle items using streaming shuffler.

    Convenience function for shuffling a list with bounded memory.

    Args:
        items: Items to shuffle.
        count_limit: Maximum items in buffer.
        byte_limit: Maximum bytes in buffer.

    Returns:
        Shuffled items.
    """
    shuffler: StreamingShuffler[T] = StreamingShuffler(
        count_limit=count_limit,
        byte_limit=byte_limit,
    )

    result: list[T] = []

    for item in items:
        spills = shuffler.push(item)
        result.extend(spills)

    result.extend(shuffler.drain())

    return result
