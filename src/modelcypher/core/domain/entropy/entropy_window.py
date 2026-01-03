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

    Window size is derived from the provided sample count using geometry.
    """

    def __init__(
        self,
        sample_count: int,
        window_id: str | None = None,
    ):
        """Initialize entropy window.

        Args:
            sample_count: Expected sample count used to derive window size.
            window_id: Optional unique identifier for this window.
        """
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        if sample_count <= 0:
            raise ValueError("sample_count must be positive to derive window size")
        backend = get_default_backend()
        window_size = int(sqrt_scalar(float(sample_count), backend))
        if window_size <= 0:
            raise ValueError("derived window_size must be positive")

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

        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        values = [sample.entropy for sample in self._samples]
        tokens = [sample.token_index for sample in self._samples]
        entropy_arr = backend.array(values)
        token_arr = backend.array(tokens)
        mean_arr = backend.mean(entropy_arr)
        max_arr = backend.max(entropy_arr)
        min_arr = backend.min(entropy_arr)
        token_min_arr = backend.min(token_arr)
        token_max_arr = backend.max(token_arr)
        backend.eval(mean_arr, max_arr, min_arr, token_min_arr, token_max_arr)

        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self._samples),
            current_entropy=self._samples[-1].entropy,
            moving_average=float(backend.to_scalar(mean_arr)),
            max_entropy=float(backend.to_scalar(max_arr)),
            min_entropy=float(backend.to_scalar(min_arr)),
            token_start=int(backend.to_scalar(token_min_arr)),
            token_end=int(backend.to_scalar(token_max_arr)),
        )
