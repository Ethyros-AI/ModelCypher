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
Metrics Ring Buffer for Entropy Visualization.

Ported 1:1 from the reference Swift implementation.

High-performance ring buffer for visualization data with:
- Constant memory usage via circular buffer
- Pre-computed domain statistics for charting
- Min-max binning for viewport-width data reduction
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from uuid import UUID, uuid4


# =============================================================================
# MetricSample
# =============================================================================


@dataclass
class MetricSample:
    """
    Lightweight metric sample for high-frequency visualization.

    All properties are stored (not computed) for performance.
    Uses float('nan') for missing values to maintain array contiguity.
    """

    id: int
    timestamp: float  # Unix timestamp
    loss: float = float("nan")
    entropy: float = float("nan")
    throughput: float = float("nan")
    gpu_memory: float = float("nan")
    active_skill_count: int = 0

    @classmethod
    def create(
        cls,
        sample_id: int,
        timestamp: float | None = None,
        date: datetime | None = None,
        loss: float = float("nan"),
        entropy: float = float("nan"),
        throughput: float = float("nan"),
        gpu_memory: float = float("nan"),
        active_skill_count: int = 0,
    ) -> "MetricSample":
        """Create a metric sample from timestamp or date."""
        if timestamp is None:
            timestamp = date.timestamp() if date else datetime.utcnow().timestamp()
        return cls(
            id=sample_id,
            timestamp=timestamp,
            loss=loss,
            entropy=entropy,
            throughput=throughput,
            gpu_memory=gpu_memory,
            active_skill_count=active_skill_count,
        )

    @property
    def has_loss(self) -> bool:
        """Whether this sample has a valid loss value."""
        return not math.isnan(self.loss)

    @property
    def has_entropy(self) -> bool:
        """Whether this sample has a valid entropy value."""
        return not math.isnan(self.entropy)

    @property
    def has_throughput(self) -> bool:
        """Whether this sample has valid throughput data."""
        return not math.isnan(self.throughput)

    @property
    def date(self) -> datetime:
        """Convert stored timestamp back to datetime."""
        return datetime.fromtimestamp(self.timestamp)

    # Traffic light classification

    class TrafficLight(str, Enum):
        """Traffic light state based on entropy level."""

        green = "green"  # entropy < 1.5: confident
        yellow = "yellow"  # 1.5 <= entropy < 3.0: elevated
        red = "red"  # entropy >= 3.0: uncertain

        @property
        def background_opacity(self) -> float:
            return {
                MetricSample.TrafficLight.green: 0.10,
                MetricSample.TrafficLight.yellow: 0.15,
                MetricSample.TrafficLight.red: 0.20,
            }[self]

    def traffic_light(
        self,
        low_threshold: float = 1.5,
        high_threshold: float = 3.0,
    ) -> TrafficLight:
        """Determine traffic light state from entropy value."""
        if not self.has_entropy:
            return MetricSample.TrafficLight.green

        if self.entropy < low_threshold:
            return MetricSample.TrafficLight.green
        elif self.entropy < high_threshold:
            return MetricSample.TrafficLight.yellow
        else:
            return MetricSample.TrafficLight.red

    @staticmethod
    def binned(samples: list["MetricSample"]) -> "MetricSample" | None:
        """
        Create a binned sample from multiple samples, preserving peak features.

        Uses min-max binning to preserve visual peaks rather than averaging.
        This maintains chart fidelity while reducing point count.

        Args:
            samples: Array of samples to bin together

        Returns:
            Single representative sample, or None if input is empty
        """
        if not samples:
            return None
        if len(samples) == 1:
            return samples[0]

        min_timestamp_sample = samples[0]
        max_skill_count = samples[0].active_skill_count

        loss_values = [s.loss for s in samples if s.has_loss]
        entropy_values = [s.entropy for s in samples if s.has_entropy]
        throughput_values = [s.throughput for s in samples if s.has_throughput]
        gpu_values = [s.gpu_memory for s in samples if not math.isnan(s.gpu_memory)]

        for sample in samples:
            if sample.timestamp < min_timestamp_sample.timestamp:
                min_timestamp_sample = sample
            max_skill_count = max(max_skill_count, sample.active_skill_count)

        def choose_extreme(values: list[float]) -> float:
            if not values:
                return float("nan")
            if len(values) == 1:
                return values[0]
            mean = sum(values) / len(values)
            min_val, max_val = min(values), max(values)
            return max_val if abs(max_val - mean) >= abs(mean - min_val) else min_val

        return MetricSample(
            id=min_timestamp_sample.id,
            timestamp=min_timestamp_sample.timestamp,
            loss=choose_extreme(loss_values),
            entropy=choose_extreme(entropy_values),
            throughput=choose_extreme(throughput_values),
            gpu_memory=choose_extreme(gpu_values),
            active_skill_count=max_skill_count,
        )


# =============================================================================
# MetricEvent
# =============================================================================


class EventType(str, Enum):
    """Event types for chart overlay markers."""

    circuit_breaker_tripped = "circuit_breaker_tripped"  # ▲ High entropy
    dpo_correction = "dpo_correction"  # ◆ User steering
    skill_activated = "skill_activated"  # ● Adapter loaded
    skill_deactivated = "skill_deactivated"  # ○ Adapter unloaded
    checkpoint_saved = "checkpoint_saved"  # ■ Training checkpoint
    memory_warning = "memory_warning"  # ⚠ Memory pressure

    @property
    def display_name(self) -> str:
        return {
            EventType.circuit_breaker_tripped: "Circuit",
            EventType.dpo_correction: "DPO",
            EventType.skill_activated: "Skill+",
            EventType.skill_deactivated: "Skill-",
            EventType.checkpoint_saved: "Checkpoint",
            EventType.memory_warning: "Memory",
        }[self]

    @property
    def symbol(self) -> str:
        return {
            EventType.circuit_breaker_tripped: "▲",
            EventType.dpo_correction: "◆",
            EventType.skill_activated: "●",
            EventType.skill_deactivated: "○",
            EventType.checkpoint_saved: "■",
            EventType.memory_warning: "⚠",
        }[self]


@dataclass
class MetricEvent:
    """Discrete event marker for overlay on metrics chart."""

    id: UUID
    timestamp: float
    event_type: EventType
    label: str | None = None
    correlation_id: UUID | None = None

    @classmethod
    def create(
        cls,
        event_type: EventType,
        timestamp: float | None = None,
        date: datetime | None = None,
        label: str | None = None,
        correlation_id: UUID | None = None,
    ) -> "MetricEvent":
        """Create an event from timestamp or date."""
        if timestamp is None:
            timestamp = date.timestamp() if date else datetime.utcnow().timestamp()
        return cls(
            id=uuid4(),
            timestamp=timestamp,
            event_type=event_type,
            label=label,
            correlation_id=correlation_id,
        )

    @property
    def symbol(self) -> str:
        """Symbol for chart annotation."""
        return self.event_type.symbol


# =============================================================================
# MetricsRingBuffer
# =============================================================================


class MetricsRingBuffer:
    """
    High-performance ring buffer for visualization data.

    Capacity: 10,000 samples (~2.7 minutes at 60Hz)

    Features:
    - Constant memory usage via circular buffer
    - Pre-computed domain statistics for charting
    - Min-max binning for viewport-width data reduction
    """

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._storage: list[MetricSample] = []
        self._write_index: int = 0
        self._total_written: int = 0
        self._next_id: int = 0

        # Domain tracking (pre-computed for charting)
        self._min_timestamp: float = float("inf")
        self._max_timestamp: float = float("-inf")
        self._max_loss: float = 0.0
        self._max_entropy: float = 0.0

    @property
    def count(self) -> int:
        """Number of samples currently in buffer."""
        return min(self._total_written, self.capacity)

    @property
    def is_empty(self) -> bool:
        """Whether the buffer is empty."""
        return self._total_written == 0

    @property
    def has_wrapped(self) -> bool:
        """Whether the buffer has wrapped around at least once."""
        return self._total_written > self.capacity

    @property
    def x_domain(self) -> tuple[float, float]:
        """Time domain for chart X axis (pre-computed, O(1))."""
        if self.is_empty:
            return (0.0, 1.0)
        return (self._min_timestamp, self._max_timestamp)

    @property
    def max_y(self) -> float:
        """Maximum Y value across loss and entropy (for chart domain)."""
        return max(self._max_loss, self._max_entropy, 1.0)

    def append(self, sample: MetricSample) -> None:
        """
        Append a new sample to the buffer.

        Note: Overwrites oldest sample when buffer is full.
        """
        # Update domain tracking
        self._min_timestamp = min(self._min_timestamp, sample.timestamp)
        self._max_timestamp = max(self._max_timestamp, sample.timestamp)
        if sample.has_loss:
            self._max_loss = max(self._max_loss, sample.loss)
        if sample.has_entropy:
            self._max_entropy = max(self._max_entropy, sample.entropy)

        # Append or overwrite
        if len(self._storage) < self.capacity:
            self._storage.append(sample)
        else:
            self._storage[self._write_index] = sample

        self._write_index = (self._write_index + 1) % self.capacity
        self._total_written += 1

    def append_values(
        self,
        timestamp: float | None = None,
        date: datetime | None = None,
        loss: float = float("nan"),
        entropy: float = float("nan"),
        throughput: float = float("nan"),
        gpu_memory: float = float("nan"),
        active_skill_count: int = 0,
    ) -> None:
        """Append a new sample with auto-generated ID."""
        if timestamp is None:
            timestamp = date.timestamp() if date else datetime.utcnow().timestamp()
        sample = MetricSample(
            id=self._next_id,
            timestamp=timestamp,
            loss=loss,
            entropy=entropy,
            throughput=throughput,
            gpu_memory=gpu_memory,
            active_skill_count=active_skill_count,
        )
        self._next_id += 1
        self.append(sample)

    def all_points(self) -> list[MetricSample]:
        """
        Return all samples in chronological order.

        Note: Creates a copy; prefer binned_points for charts.
        """
        if self.is_empty:
            return []

        if not self.has_wrapped:
            return self._storage.copy()

        # Reconstruct chronological order from ring buffer
        result: list[MetricSample] = []
        for i in range(self.capacity):
            index = (self._write_index + i) % self.capacity
            result.append(self._storage[index])
        return result

    @property
    def latest(self) -> MetricSample | None:
        """Return the most recent sample."""
        if self.is_empty:
            return None
        index = self._write_index - 1 if self._write_index > 0 else len(self._storage) - 1
        return self._storage[index]

    @property
    def oldest(self) -> MetricSample | None:
        """Return the oldest sample."""
        if self.is_empty:
            return None
        if self.has_wrapped:
            return self._storage[self._write_index]
        return self._storage[0]

    def binned_points(self, viewport_width: int) -> list[MetricSample]:
        """
        Return samples binned to viewport width for efficient chart rendering.

        Binning to viewport resolution avoids rendering thousands of invisible
        points while preserving visual peaks.

        Args:
            viewport_width: Number of horizontal pixels in chart

        Returns:
            Array of representative samples (max viewport_width points)
        """
        if self.count <= viewport_width:
            return self.all_points()

        points = self.all_points()
        bin_size = max(1, self.count // viewport_width)

        result: list[MetricSample] = []
        i = 0
        while i < len(points):
            end = min(i + bin_size, len(points))
            chunk = points[i:end]

            representative = MetricSample.binned(chunk)
            if representative:
                result.append(representative)

            i = end

        return result

    def points_in_range(self, start: float, end: float) -> list[MetricSample]:
        """Return samples within a time range."""
        return [p for p in self.all_points() if start <= p.timestamp <= end]

    @property
    def current_entropy(self) -> float:
        """Current entropy value (most recent sample)."""
        latest = self.latest
        return latest.entropy if latest else 0.0

    def average_entropy(self, window_size: int = 20) -> float:
        """Moving average of entropy over recent samples."""
        points = self.all_points()[-window_size:]
        valid_points = [p for p in points if p.has_entropy]
        if not valid_points:
            return 0.0
        return sum(p.entropy for p in valid_points) / len(valid_points)

    def average_loss(self, window_size: int = 20) -> float:
        """Moving average of loss over recent samples."""
        points = self.all_points()[-window_size:]
        valid_points = [p for p in points if p.has_loss]
        if not valid_points:
            return 0.0
        return sum(p.loss for p in valid_points) / len(valid_points)

    def reset(self) -> None:
        """Clear all samples and reset state."""
        self._storage.clear()
        self._write_index = 0
        self._total_written = 0
        self._min_timestamp = float("inf")
        self._max_timestamp = float("-inf")
        self._max_loss = 0.0
        self._max_entropy = 0.0


# =============================================================================
# EventMarkerBuffer
# =============================================================================


class EventMarkerBuffer:
    """Ring buffer specifically for event markers (lower capacity than metrics)."""

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._storage: list[MetricEvent] = []
        self._write_index: int = 0
        self._total_written: int = 0

    @property
    def count(self) -> int:
        return min(self._total_written, self.capacity)

    def append(self, event: MetricEvent) -> None:
        """Append an event to the buffer."""
        if len(self._storage) < self.capacity:
            self._storage.append(event)
        else:
            self._storage[self._write_index] = event
        self._write_index = (self._write_index + 1) % self.capacity
        self._total_written += 1

    def events_in_range(self, start: float, end: float) -> list[MetricEvent]:
        """Return events within a time range."""
        if self._total_written <= self.capacity:
            all_events = self._storage
        else:
            all_events = []
            for i in range(self.capacity):
                index = (self._write_index + i) % self.capacity
                all_events.append(self._storage[index])
        return [e for e in all_events if start <= e.timestamp <= end]

    def reset(self) -> None:
        """Clear all events and reset state."""
        self._storage.clear()
        self._write_index = 0
        self._total_written = 0
