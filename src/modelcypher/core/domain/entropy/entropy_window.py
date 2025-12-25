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

Maintains a rolling window of entropy samples to detect sustained uncertainty
patterns rather than transient spikes. Emits signals when circuit breaker
conditions are met.

Design:
- Window size of 20 tokens (configurable) balances responsiveness vs noise
- Tracks both instantaneous and moving average entropy
- Thread-safe via asyncio locks (or synchronous for simple use)

Ported from EntropyWindow.swift (301 lines).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("modelcypher.entropy.entropy_window")


# =============================================================================
# Entropy Level Classification
# =============================================================================


class EntropyLevel(str, Enum):
    """Classification of entropy level."""

    LOW = "low"  # < 1.5
    MODERATE = "moderate"  # 1.5 - 3.0
    HIGH = "high"  # > 3.0


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EntropyWindowConfig:
    """Configuration for the entropy window."""

    # Number of samples to maintain in the window
    window_size: int = 20

    # Minimum samples needed before computing moving average
    minimum_samples: int = 5

    # Threshold above which a single sample is considered "high"
    high_entropy_threshold: float = 3.0

    # Moving average threshold for circuit breaker
    circuit_breaker_threshold: float = 4.0

    # Number of consecutive high samples before alerting
    sustained_high_count: int = 3


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
    """Current status of the entropy window."""

    window_id: str
    sample_count: int
    current_entropy: float
    moving_average: float
    max_entropy: float
    min_entropy: float
    consecutive_high_count: int
    should_trip_circuit_breaker: bool
    level: EntropyLevel
    token_start: int
    token_end: int


# =============================================================================
# Entropy Window
# =============================================================================


class EntropyWindow:
    """
    Sliding window tracker for entropy measurements during inference.

    Usage:
        window = EntropyWindow(config=EntropyWindowConfig(window_size=20))
        status = window.add(entropy=2.45, variance=0.12, token_index=42)
        if status.should_trip_circuit_breaker:
            # Handle high entropy condition
    """

    def __init__(
        self,
        config: EntropyWindowConfig | None = None,
        window_id: str | None = None,
    ):
        """
        Initialize entropy window.

        Args:
            config: Window configuration.
            window_id: Unique identifier for this window session.
        """
        self.config = config or EntropyWindowConfig()
        self.window_id = window_id or str(uuid.uuid4())
        self._samples: list[EntropySample] = []
        self._consecutive_high_count = 0
        self._circuit_breaker_tripped = False
        self._lock = asyncio.Lock()

    def add(
        self,
        entropy: float,
        variance: float,
        token_index: int,
    ) -> EntropyWindowStatus:
        """
        Add a new entropy sample to the window (synchronous).

        Args:
            entropy: Shannon entropy value.
            variance: Top-K variance value.
            token_index: Index of the token in generation sequence.

        Returns:
            Current window status after adding the sample.
        """
        sample = EntropySample(
            entropy=entropy,
            variance=variance,
            token_index=token_index,
            timestamp=datetime.now(),
        )

        # Add to window, maintaining size limit
        self._samples.append(sample)
        if len(self._samples) > self.config.window_size:
            self._samples.pop(0)

        # Track consecutive high entropy
        if entropy >= self.config.high_entropy_threshold:
            self._consecutive_high_count += 1
        else:
            self._consecutive_high_count = 0

        # Check circuit breaker conditions
        avg = self._moving_average()
        if (
            avg >= self.config.circuit_breaker_threshold
            or self._consecutive_high_count >= self.config.sustained_high_count
        ):
            self._circuit_breaker_tripped = True

        return self._current_status()

    async def add_async(
        self,
        entropy: float,
        variance: float,
        token_index: int,
    ) -> EntropyWindowStatus:
        """Add entropy sample (async, thread-safe)."""
        async with self._lock:
            return self.add(entropy, variance, token_index)

    def status(self) -> EntropyWindowStatus:
        """Returns the current window status without adding a sample."""
        return self._current_status()

    def reset_circuit_breaker(self) -> None:
        """Resets the circuit breaker state (called after user intervention)."""
        self._circuit_breaker_tripped = False
        self._consecutive_high_count = 0

    def reset(self) -> None:
        """Clears all samples and resets state for a new generation."""
        self._samples.clear()
        self._consecutive_high_count = 0
        self._circuit_breaker_tripped = False

    def add_batch(
        self,
        batch: list[tuple[float, float, int]],
    ) -> EntropyWindowStatus:
        """
        Add multiple samples efficiently.

        Args:
            batch: List of (entropy, variance, token_index) tuples.

        Returns:
            Final status after all samples added.
        """
        for entropy, variance, token_index in batch:
            self.add(entropy, variance, token_index)
        return self._current_status()

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _moving_average(self) -> float:
        """Compute moving average of entropy values."""
        if not self._samples:
            return 0.0
        total = sum(s.entropy for s in self._samples)
        return total / len(self._samples)

    def _variance_mean(self) -> float:
        """Compute mean variance."""
        if not self._samples:
            return 0.0
        total = sum(s.variance for s in self._samples)
        return total / len(self._samples)

    def _current_status(self) -> EntropyWindowStatus:
        """Build current status from window state."""
        entropies = [s.entropy for s in self._samples]
        current = entropies[-1] if entropies else 0.0
        avg = self._moving_average()

        # Classify level
        if avg < 1.5:
            level = EntropyLevel.LOW
        elif avg < 3.0:
            level = EntropyLevel.MODERATE
        else:
            level = EntropyLevel.HIGH

        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self._samples),
            current_entropy=current,
            moving_average=avg,
            max_entropy=max(entropies) if entropies else 0.0,
            min_entropy=min(entropies) if entropies else 0.0,
            consecutive_high_count=self._consecutive_high_count,
            should_trip_circuit_breaker=self._circuit_breaker_tripped,
            level=level,
            token_start=self._samples[0].token_index if self._samples else 0,
            token_end=self._samples[-1].token_index if self._samples else 0,
        )

    # =========================================================================
    # Signal Generation
    # =========================================================================

    def to_entropy_summary(self) -> dict:
        """
        Create a summary dict from current window state.

        Returns:
            Dictionary with entropy summary data.
        """
        status = self._current_status()
        return {
            "window_id": self.window_id,
            "token_start": status.token_start,
            "token_end": status.token_end,
            "logit_entropy": status.moving_average,
            "top_k_variance": self._variance_mean(),
            "level": status.level.value,
            "sample_count": status.sample_count,
        }

    def circuit_breaker_alert(self) -> dict | None:
        """
        Create a circuit breaker alert if conditions are met.

        Returns:
            Alert dictionary if circuit breaker should trip, else None.
        """
        status = self._current_status()
        if not status.should_trip_circuit_breaker:
            return None

        return {
            "type": "circuit_breaker_tripped",
            "window_id": self.window_id,
            "current_entropy": status.current_entropy,
            "moving_average": status.moving_average,
            "consecutive_high_count": status.consecutive_high_count,
            "threshold": self.config.circuit_breaker_threshold,
            "recommended_action": "pause_and_steer",
        }
