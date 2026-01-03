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

"""
Entropy Window: Sliding window tracker for entropy measurements during inference.

Maintains a rolling window of entropy samples to measure local statistics
without threshold-based classification.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("modelcypher.entropy.entropy_window")


# =============================================================================
# Sample and Status
# =============================================================================


@dataclass
class EntropySample:
    """Individual sample in the window."""

    entropy: float
    variance: float
    token_index: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EntropyWindowStatus:
    """Current status of the entropy window.

    Raw measurements only. Caller interprets downstream.
    """

    window_id: str
    sample_count: int
    current_entropy: float
    moving_average: float
    max_entropy: float
    min_entropy: float
    token_start: int
    token_end: int


# =============================================================================
# Entropy Window
# =============================================================================


class EntropyWindow:
    """Sliding window tracker for entropy measurements during inference.

    window_size should be derived from context by the caller (e.g., sqrt(n)
    where n is the expected sample count or baseline size).
    """

    def __init__(
        self,
        window_size: int,
        window_id: str | None = None,
    ):
        """Initialize entropy window.

        Args:
            window_size: Size of the sliding window. Caller derives this
                from context (e.g., sqrt(baseline_sample_count)).
            window_id: Optional unique identifier for this window.
        """
        self._window_size = window_size
        self.window_id = window_id or str(uuid.uuid4())
        self._samples: list[EntropySample] = []
        self._lock = asyncio.Lock()

    def add(
        self,
        entropy: float,
        variance: float,
        token_index: int,
    ) -> EntropyWindowStatus:
        """Add a new entropy sample to the window (synchronous)."""
        sample = EntropySample(
            entropy=entropy,
            variance=variance,
            token_index=token_index,
            timestamp=datetime.now(),
        )

        # Add to window, maintaining size limit
        self._samples.append(sample)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)

        return self._current_status()

    async def add_async(
        self,
        entropy: float,
        variance: float,
        token_index: int,
    ) -> EntropyWindowStatus:
        """Add a new entropy sample to the window (async)."""
        async with self._lock:
            return self.add(entropy=entropy, variance=variance, token_index=token_index)

    def add_batch(self, batch: list[tuple[float, float, int]]) -> EntropyWindowStatus:
        """Add multiple samples and return final status."""
        status = self.status()
        for entropy, variance, token_index in batch:
            status = self.add(entropy=entropy, variance=variance, token_index=token_index)
        return status

    def status(self) -> EntropyWindowStatus:
        """Return current window status without adding a new sample."""
        return self._current_status()

    def reset(self) -> None:
        """Reset the window to empty state."""
        self._samples = []

    def to_entropy_summary(self) -> dict:
        """Return summary dictionary for JSON serialization."""
        status = self.status()
        return {
            "window_id": status.window_id,
            "logit_entropy": status.current_entropy,
            "moving_average": status.moving_average,
            "sample_count": status.sample_count,
            "token_start": status.token_start,
            "token_end": status.token_end,
        }

    def _moving_average(self) -> float:
        values = [sample.entropy for sample in self._samples]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _current_status(self) -> EntropyWindowStatus:
        if not self._samples:
            return EntropyWindowStatus(
                window_id=self.window_id,
                sample_count=0,
                current_entropy=0.0,
                moving_average=0.0,
                max_entropy=0.0,
                min_entropy=0.0,
                token_start=0,
                token_end=0,
            )

        values = [sample.entropy for sample in self._samples]
        tokens = [sample.token_index for sample in self._samples]
        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self._samples),
            current_entropy=self._samples[-1].entropy,
            moving_average=self._moving_average(),
            max_entropy=max(values),
            min_entropy=min(values),
            token_start=min(tokens),
            token_end=max(tokens),
        )
