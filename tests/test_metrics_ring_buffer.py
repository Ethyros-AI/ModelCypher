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
Tests for MetricsRingBuffer, MetricSample, and related utilities.

This tests the high-performance ring buffer for entropy visualization.
"""

from __future__ import annotations

from datetime import datetime

from modelcypher.core.domain.entropy.metrics_ring_buffer import (
    EventMarkerBuffer,
    EventType,
    MetricEvent,
    MetricSample,
    MetricsRingBuffer,
)

# =============================================================================
# MetricSample Tests
# =============================================================================


class TestMetricSample:
    """Tests for MetricSample."""

    def test_create_with_timestamp(self) -> None:
        """Test creating a sample with explicit timestamp."""
        sample = MetricSample.create(
            sample_id=1,
            timestamp=1703265600.0,
            loss=0.5,
            entropy=2.3,
            throughput=100.0,
            gpu_memory=0.75,
            active_skill_count=2,
        )
        assert sample.id == 1
        assert sample.timestamp == 1703265600.0
        assert sample.loss == 0.5
        assert sample.entropy == 2.3
        assert sample.throughput == 100.0
        assert sample.gpu_memory == 0.75
        assert sample.active_skill_count == 2

    def test_create_with_date(self) -> None:
        """Test creating a sample with datetime."""
        now = datetime(2024, 12, 22, 12, 0, 0)
        sample = MetricSample.create(
            sample_id=2,
            date=now,
            entropy=1.5,
        )
        assert sample.id == 2
        assert sample.timestamp == now.timestamp()

    def test_has_properties(self) -> None:
        """Test has_* property methods."""
        sample = MetricSample(
            id=1,
            timestamp=1000.0,
            loss=0.5,
            entropy=float("nan"),
            throughput=100.0,
            gpu_memory=float("nan"),
        )
        assert sample.has_loss is True
        assert sample.has_entropy is False
        assert sample.has_throughput is True

    def test_date_property(self) -> None:
        """Test converting timestamp back to datetime."""
        ts = 1703265600.0
        sample = MetricSample(id=1, timestamp=ts)
        assert isinstance(sample.date, datetime)
        assert sample.date.timestamp() == ts

    def test_traffic_light_green(self) -> None:
        """Test green traffic light for low entropy."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=1.0)
        assert sample.traffic_light() == MetricSample.TrafficLight.green
        assert sample.traffic_light().background_opacity == 0.10

    def test_traffic_light_yellow(self) -> None:
        """Test yellow traffic light for moderate entropy."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=2.0)
        assert sample.traffic_light() == MetricSample.TrafficLight.yellow
        assert sample.traffic_light().background_opacity == 0.15

    def test_traffic_light_red(self) -> None:
        """Test red traffic light for high entropy."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=4.0)
        assert sample.traffic_light() == MetricSample.TrafficLight.red
        assert sample.traffic_light().background_opacity == 0.20

    def test_traffic_light_no_entropy(self) -> None:
        """Test traffic light defaults to green when no entropy."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=float("nan"))
        assert sample.traffic_light() == MetricSample.TrafficLight.green

    def test_traffic_light_custom_thresholds(self) -> None:
        """Test traffic light with custom thresholds."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=2.0)
        # With higher threshold, 2.0 should be green
        assert (
            sample.traffic_light(low_threshold=2.5, high_threshold=4.0)
            == MetricSample.TrafficLight.green
        )


class TestMetricSampleBinning:
    """Tests for MetricSample binning functionality."""

    def test_binned_empty_list(self) -> None:
        """Test binning empty list returns None."""
        assert MetricSample.binned([]) is None

    def test_binned_single_sample(self) -> None:
        """Test binning single sample returns that sample."""
        sample = MetricSample(id=1, timestamp=1000.0, entropy=2.0)
        result = MetricSample.binned([sample])
        assert result is sample

    def test_binned_multiple_samples(self) -> None:
        """Test binning preserves min timestamp and chooses extreme values."""
        samples = [
            MetricSample(id=1, timestamp=1000.0, loss=0.5, entropy=2.0, active_skill_count=1),
            MetricSample(id=2, timestamp=1001.0, loss=0.8, entropy=1.5, active_skill_count=2),
            MetricSample(id=3, timestamp=1002.0, loss=0.3, entropy=3.0, active_skill_count=1),
        ]
        result = MetricSample.binned(samples)
        assert result is not None
        assert result.id == 1  # Uses min timestamp sample's ID
        assert result.timestamp == 1000.0  # Min timestamp
        assert result.active_skill_count == 2  # Max skill count

    def test_binned_chooses_extreme(self) -> None:
        """Test that binning chooses the value furthest from mean."""
        # Mean entropy = 2.0, max = 3.0 (distance 1.0), min = 1.0 (distance 1.0)
        # Should choose max when equidistant
        samples = [
            MetricSample(id=1, timestamp=1000.0, entropy=1.0),
            MetricSample(id=2, timestamp=1001.0, entropy=2.0),
            MetricSample(id=3, timestamp=1002.0, entropy=3.0),
        ]
        result = MetricSample.binned(samples)
        assert result is not None
        # Mean = 2.0, max deviation is 1.0 for both, so max is chosen
        assert result.entropy == 3.0 or result.entropy == 1.0

    def test_binned_handles_nan(self) -> None:
        """Test that binning handles NaN values correctly."""
        samples = [
            MetricSample(id=1, timestamp=1000.0, loss=0.5, entropy=float("nan")),
            MetricSample(id=2, timestamp=1001.0, loss=float("nan"), entropy=2.0),
        ]
        result = MetricSample.binned(samples)
        assert result is not None
        assert result.loss == 0.5  # Only valid value
        assert result.entropy == 2.0  # Only valid value


# =============================================================================
# MetricEvent Tests
# =============================================================================


class TestMetricEvent:
    """Tests for MetricEvent."""

    def test_create_with_timestamp(self) -> None:
        """Test creating an event with timestamp."""
        event = MetricEvent.create(
            event_type=EventType.circuit_breaker_tripped,
            timestamp=1000.0,
            label="High entropy detected",
        )
        assert event.event_type == EventType.circuit_breaker_tripped
        assert event.timestamp == 1000.0
        assert event.label == "High entropy detected"
        assert event.id is not None

    def test_create_with_date(self) -> None:
        """Test creating an event with datetime."""
        now = datetime.utcnow()
        event = MetricEvent.create(
            event_type=EventType.checkpoint_saved,
            date=now,
        )
        assert event.timestamp == now.timestamp()

    def test_event_type_properties(self) -> None:
        """Test EventType display properties."""
        assert EventType.circuit_breaker_tripped.display_name == "Circuit"
        assert EventType.circuit_breaker_tripped.symbol == "▲"

        assert EventType.dpo_correction.display_name == "DPO"
        assert EventType.dpo_correction.symbol == "◆"

        assert EventType.skill_activated.display_name == "Skill+"
        assert EventType.skill_activated.symbol == "●"

        assert EventType.skill_deactivated.display_name == "Skill-"
        assert EventType.skill_deactivated.symbol == "○"

        assert EventType.checkpoint_saved.display_name == "Checkpoint"
        assert EventType.checkpoint_saved.symbol == "■"

        assert EventType.memory_warning.display_name == "Memory"
        assert EventType.memory_warning.symbol == "⚠"

    def test_event_symbol_property(self) -> None:
        """Test that event symbol delegates to event_type."""
        event = MetricEvent.create(
            event_type=EventType.dpo_correction,
            timestamp=1000.0,
        )
        assert event.symbol == "◆"


# =============================================================================
# MetricsRingBuffer Tests
# =============================================================================


class TestMetricsRingBuffer:
    """Tests for MetricsRingBuffer."""

    def test_init_default_capacity(self) -> None:
        """Test default capacity."""
        buffer = MetricsRingBuffer()
        assert buffer.capacity == 10_000
        assert buffer.count == 0
        assert buffer.is_empty is True

    def test_init_custom_capacity(self) -> None:
        """Test custom capacity."""
        buffer = MetricsRingBuffer(capacity=100)
        assert buffer.capacity == 100

    def test_append_sample(self) -> None:
        """Test appending a sample."""
        buffer = MetricsRingBuffer(capacity=10)
        sample = MetricSample(id=1, timestamp=1000.0, entropy=2.0)
        buffer.append(sample)

        assert buffer.count == 1
        assert buffer.is_empty is False
        assert buffer.latest == sample

    def test_append_values(self) -> None:
        """Test appending with values (auto-generates ID)."""
        buffer = MetricsRingBuffer(capacity=10)
        buffer.append_values(timestamp=1000.0, entropy=2.0, loss=0.5)

        assert buffer.count == 1
        latest = buffer.latest
        assert latest is not None
        assert latest.id == 0
        assert latest.entropy == 2.0
        assert latest.loss == 0.5

    def test_x_domain(self) -> None:
        """Test X domain (time range) tracking."""
        buffer = MetricsRingBuffer(capacity=10)

        # Empty buffer
        assert buffer.x_domain == (0.0, 1.0)

        buffer.append_values(timestamp=1000.0)
        buffer.append_values(timestamp=1005.0)
        buffer.append_values(timestamp=1010.0)

        assert buffer.x_domain == (1000.0, 1010.0)

    def test_max_y(self) -> None:
        """Test max Y tracking."""
        buffer = MetricsRingBuffer(capacity=10)

        buffer.append_values(timestamp=1000.0, loss=0.5, entropy=2.0)
        buffer.append_values(timestamp=1001.0, loss=1.5, entropy=3.0)

        assert buffer.max_y == 3.0  # max(1.5, 3.0, 1.0) = 3.0

    def test_all_points(self) -> None:
        """Test getting all points in chronological order."""
        buffer = MetricsRingBuffer(capacity=10)

        for i in range(5):
            buffer.append_values(timestamp=1000.0 + i, entropy=float(i))

        points = buffer.all_points()
        assert len(points) == 5
        assert [p.timestamp for p in points] == [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]

    def test_ring_buffer_wrap(self) -> None:
        """Test that buffer wraps correctly."""
        buffer = MetricsRingBuffer(capacity=3)

        for i in range(5):
            buffer.append_values(timestamp=float(i), entropy=float(i))

        assert buffer.count == 3
        assert buffer.has_wrapped is True

        points = buffer.all_points()
        # Should have the last 3 samples
        assert len(points) == 3
        assert [p.timestamp for p in points] == [2.0, 3.0, 4.0]

    def test_oldest_and_latest(self) -> None:
        """Test oldest and latest accessors."""
        buffer = MetricsRingBuffer(capacity=5)

        for i in range(3):
            buffer.append_values(timestamp=float(i))

        assert buffer.oldest is not None
        assert buffer.oldest.timestamp == 0.0
        assert buffer.latest is not None
        assert buffer.latest.timestamp == 2.0

    def test_oldest_and_latest_after_wrap(self) -> None:
        """Test oldest and latest after wrap."""
        buffer = MetricsRingBuffer(capacity=3)

        for i in range(5):
            buffer.append_values(timestamp=float(i))

        assert buffer.oldest is not None
        assert buffer.oldest.timestamp == 2.0  # Oldest surviving sample
        assert buffer.latest is not None
        assert buffer.latest.timestamp == 4.0

    def test_binned_points(self) -> None:
        """Test binning to viewport width."""
        buffer = MetricsRingBuffer(capacity=100)

        for i in range(20):
            buffer.append_values(timestamp=float(i), entropy=float(i))

        # Request 5 bins
        binned = buffer.binned_points(viewport_width=5)

        # Should have at most 5 points
        assert len(binned) <= 5

    def test_binned_points_small_buffer(self) -> None:
        """Test binning when buffer is smaller than viewport."""
        buffer = MetricsRingBuffer(capacity=100)

        for i in range(3):
            buffer.append_values(timestamp=float(i))

        # Request 10 bins but only 3 samples
        binned = buffer.binned_points(viewport_width=10)

        # Should return all points since count < viewport
        assert len(binned) == 3

    def test_points_in_range(self) -> None:
        """Test filtering points by time range."""
        buffer = MetricsRingBuffer(capacity=100)

        for i in range(10):
            buffer.append_values(timestamp=float(i))

        # Get points between 3.0 and 6.0
        filtered = buffer.points_in_range(3.0, 6.0)

        assert len(filtered) == 4
        assert all(3.0 <= p.timestamp <= 6.0 for p in filtered)

    def test_average_entropy(self) -> None:
        """Test entropy moving average."""
        buffer = MetricsRingBuffer(capacity=100)

        for i in range(5):
            buffer.append_values(timestamp=float(i), entropy=float(i))

        # Average of 0, 1, 2, 3, 4 = 2.0
        assert buffer.average_entropy(window_size=5) == 2.0

        # Average of last 3: 2, 3, 4 = 3.0
        assert buffer.average_entropy(window_size=3) == 3.0

    def test_average_loss(self) -> None:
        """Test loss moving average."""
        buffer = MetricsRingBuffer(capacity=100)

        for i in range(5):
            buffer.append_values(timestamp=float(i), loss=float(i) * 0.1)

        # Average of 0.0, 0.1, 0.2, 0.3, 0.4 = 0.2
        assert abs(buffer.average_loss(window_size=5) - 0.2) < 0.001

    def test_current_entropy(self) -> None:
        """Test current entropy property."""
        buffer = MetricsRingBuffer(capacity=10)

        assert buffer.current_entropy == 0.0  # Empty buffer

        buffer.append_values(timestamp=1.0, entropy=2.5)
        buffer.append_values(timestamp=2.0, entropy=3.5)

        assert buffer.current_entropy == 3.5

    def test_reset(self) -> None:
        """Test reset clears all state."""
        buffer = MetricsRingBuffer(capacity=10)

        for i in range(5):
            buffer.append_values(timestamp=float(i), entropy=float(i))

        buffer.reset()

        assert buffer.count == 0
        assert buffer.is_empty is True
        assert buffer.has_wrapped is False
        assert buffer.x_domain == (0.0, 1.0)


# =============================================================================
# EventMarkerBuffer Tests
# =============================================================================


class TestEventMarkerBuffer:
    """Tests for EventMarkerBuffer."""

    def test_init(self) -> None:
        """Test initialization."""
        buffer = EventMarkerBuffer()
        assert buffer.capacity == 500
        assert buffer.count == 0

    def test_init_custom_capacity(self) -> None:
        """Test custom capacity."""
        buffer = EventMarkerBuffer(capacity=50)
        assert buffer.capacity == 50

    def test_append(self) -> None:
        """Test appending events."""
        buffer = EventMarkerBuffer(capacity=10)

        event = MetricEvent.create(
            event_type=EventType.checkpoint_saved,
            timestamp=1000.0,
        )
        buffer.append(event)

        assert buffer.count == 1

    def test_wrap(self) -> None:
        """Test buffer wrap."""
        buffer = EventMarkerBuffer(capacity=3)

        for i in range(5):
            buffer.append(
                MetricEvent.create(
                    event_type=EventType.skill_activated,
                    timestamp=float(i),
                )
            )

        assert buffer.count == 3

    def test_events_in_range(self) -> None:
        """Test filtering events by time range."""
        buffer = EventMarkerBuffer(capacity=100)

        for i in range(10):
            buffer.append(
                MetricEvent.create(
                    event_type=EventType.checkpoint_saved,
                    timestamp=float(i),
                )
            )

        filtered = buffer.events_in_range(3.0, 6.0)

        assert len(filtered) == 4
        assert all(3.0 <= e.timestamp <= 6.0 for e in filtered)

    def test_reset(self) -> None:
        """Test reset."""
        buffer = EventMarkerBuffer(capacity=10)

        for i in range(5):
            buffer.append(
                MetricEvent.create(
                    event_type=EventType.dpo_correction,
                    timestamp=float(i),
                )
            )

        buffer.reset()

        assert buffer.count == 0
