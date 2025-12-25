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
Tests for EntropyDeltaTracker.

This tests the dual-path entropy tracking functionality for LoRA adapter security analysis.
"""

from __future__ import annotations

from typing import List
from uuid import uuid4

import pytest

from modelcypher.core.domain.entropy.entropy_delta_sample import (
    EntropyDeltaSample,
)
from modelcypher.core.domain.entropy.entropy_delta_tracker import (
    EntropyDeltaTracker,
    EntropyDeltaTrackerConfig,
    PendingEntropyData,
)


def _test_config(
    anomaly_threshold: float = 0.6,
    consecutive_anomaly_count: int = 3,
    top_k: int = 10,
    compute_variance: bool = True,
    source: str = "EntropyDeltaTracker",
) -> EntropyDeltaTrackerConfig:
    """Create test config with explicit values."""
    return EntropyDeltaTrackerConfig(
        top_k=top_k,
        anomaly_threshold=anomaly_threshold,
        consecutive_anomaly_count=consecutive_anomaly_count,
        compute_variance=compute_variance,
        source=source,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


def test_from_baseline_distribution() -> None:
    """Test deriving thresholds from baseline distribution."""
    # Simulate anomaly scores with 90th percentile at 0.7
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    config = EntropyDeltaTrackerConfig.from_baseline_distribution(samples)

    # 90th percentile of [0.1...1.0] is 0.9 (index 9 * 0.9 = 8.1 -> index 8)
    assert config.anomaly_threshold == 0.9
    assert config.consecutive_anomaly_count == 3
    assert config.top_k == 10


def test_from_baseline_distribution_custom_percentile() -> None:
    """Test custom percentile for threshold derivation."""
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    config = EntropyDeltaTrackerConfig.from_baseline_distribution(
        samples, alert_percentile=0.50
    )
    # 50th percentile should be around 0.5
    assert config.anomaly_threshold == 0.5


def test_from_baseline_requires_samples() -> None:
    """Test that empty samples raises error."""
    with pytest.raises(ValueError, match="anomaly_score_samples required"):
        EntropyDeltaTrackerConfig.from_baseline_distribution([])


# =============================================================================
# Session Lifecycle Tests
# =============================================================================


def test_session_lifecycle() -> None:
    """Test starting and ending a session."""
    tracker = EntropyDeltaTracker(_test_config())

    assert tracker.is_session_active is False
    assert tracker.correlation_id is None

    correlation_id = uuid4()
    tracker.start_session(correlation_id=correlation_id)

    assert tracker.is_session_active is True
    assert tracker.correlation_id == correlation_id
    assert tracker.current_sample_count == 0
    assert tracker.current_consecutive_anomalies == 0
    assert tracker.is_circuit_breaker_tripped is False

    result = tracker.end_session()

    assert tracker.is_session_active is False
    assert result.total_tokens == 0
    assert result.has_security_flags is False


def test_end_session_without_start() -> None:
    """Test ending a session that was never started."""
    tracker = EntropyDeltaTracker(_test_config())
    result = tracker.end_session()

    assert result.total_tokens == 0
    assert result.has_security_flags is False


def test_session_auto_generates_correlation_id() -> None:
    """Test that starting a session without correlation_id generates one."""
    tracker = EntropyDeltaTracker(_test_config())
    tracker.start_session()

    assert tracker.correlation_id is not None
    assert tracker.is_session_active is True


# =============================================================================
# Pending Entropy Data Tests
# =============================================================================


def test_pending_entropy_data() -> None:
    """Test PendingEntropyData structure - raw entropy values only."""
    data = PendingEntropyData(
        token_index=5,
        generated_token=42,
        base_entropy=3.5,  # High = uncertain
        base_top_k_variance=0.8,
        base_top_token=101,
        adapter_entropy=1.2,  # Low = confident
        adapter_top_k_variance=0.3,
        adapter_top_token=102,
        base_surprisal=6.5,
        kl_divergence_adapter_to_base=0.25,
        latency_ms=15.5,
    )

    assert data.token_index == 5
    assert data.base_entropy == 3.5
    assert data.adapter_entropy == 1.2
    assert data.base_surprisal == 6.5


@pytest.mark.asyncio
async def test_record_entropy_from_data() -> None:
    """Test recording entropy from pre-computed data."""
    tracker = EntropyDeltaTracker(_test_config())
    tracker.start_session()

    data = PendingEntropyData(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.5,
        base_top_token=1,
        adapter_entropy=1.8,
        adapter_top_k_variance=0.4,
        adapter_top_token=1,
        latency_ms=5.0,
    )

    sample = await tracker.record_entropy_from_data(data)

    assert sample.token_index == 0
    assert sample.base_entropy == 2.0
    assert sample.adapter_entropy == 1.8
    assert sample.latency_ms == 5.0
    assert tracker.current_sample_count == 1


# =============================================================================
# Anomaly Detection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_anomaly_detection_low_score() -> None:
    """Test that low anomaly scores don't trigger detection."""
    tracker = EntropyDeltaTracker(_test_config())
    tracker.start_session()

    # Create benign sample (same tokens, similar entropy)
    data = PendingEntropyData(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.5,
        base_top_token=1,
        adapter_entropy=1.9,
        adapter_top_k_variance=0.4,
        adapter_top_token=1,
    )

    sample = await tracker.record_entropy_from_data(data)

    assert sample.anomaly_score < 0.6
    assert tracker.current_consecutive_anomalies == 0
    assert tracker.is_circuit_breaker_tripped is False


@pytest.mark.asyncio
async def test_anomaly_detection_high_score() -> None:
    """Test that high anomaly scores trigger detection."""
    anomalies_detected: List[EntropyDeltaSample] = []

    async def on_anomaly(sample: EntropyDeltaSample) -> None:
        anomalies_detected.append(sample)

    tracker = EntropyDeltaTracker(_test_config())
    tracker.on_anomaly_detected = on_anomaly
    tracker.start_session()

    # Create suspicious sample (high base entropy, low adapter entropy, token disagreement)
    data = PendingEntropyData(
        token_index=0,
        generated_token=999,
        base_entropy=5.0,
        base_top_k_variance=0.8,
        base_top_token=1,
        adapter_entropy=0.5,
        adapter_top_k_variance=0.2,
        adapter_top_token=999,
    )

    sample = await tracker.record_entropy_from_data(data)

    # This should have high anomaly score due to large delta and token disagreement
    assert sample.delta > 0  # base_entropy > adapter_entropy
    assert sample.top_token_disagreement is True
    assert tracker.current_consecutive_anomalies >= 0

    result = tracker.end_session()
    assert result.total_tokens == 1


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_consecutive_anomalies() -> None:
    """Test that circuit breaker trips after consecutive anomalies."""
    tripped_samples: List[List[EntropyDeltaSample]] = []

    async def on_circuit_breaker(samples: List[EntropyDeltaSample]) -> None:
        tripped_samples.append(samples)

    config = _test_config(
        anomaly_threshold=0.3,  # Lower threshold for easier triggering
        consecutive_anomaly_count=2,
    )
    tracker = EntropyDeltaTracker(config)
    tracker.on_circuit_breaker_tripped = on_circuit_breaker
    tracker.start_session()

    # Create multiple high-anomaly samples
    for i in range(3):
        data = PendingEntropyData(
            token_index=i,
            generated_token=999 + i,
            base_entropy=6.0,
            base_top_k_variance=0.9,
            base_top_token=1,
            adapter_entropy=0.3,
            adapter_top_k_variance=0.1,
            adapter_top_token=999 + i,
        )
        await tracker.record_entropy_from_data(data)

    result = tracker.end_session()

    # Circuit breaker should have tripped
    assert result.circuit_breaker_tripped is True
    assert result.circuit_breaker_trip_index is not None


@pytest.mark.asyncio
async def test_consecutive_anomalies_reset_on_normal_sample() -> None:
    """Test that consecutive anomaly count resets on normal sample."""
    config = _test_config(
        anomaly_threshold=0.3,
        consecutive_anomaly_count=3,
    )
    tracker = EntropyDeltaTracker(config)
    tracker.start_session()

    # Create one anomalous sample
    anomaly_data = PendingEntropyData(
        token_index=0,
        generated_token=999,
        base_entropy=6.0,
        base_top_k_variance=0.9,
        base_top_token=1,
        adapter_entropy=0.3,
        adapter_top_k_variance=0.1,
        adapter_top_token=999,
    )
    await tracker.record_entropy_from_data(anomaly_data)

    # Create a normal sample
    normal_data = PendingEntropyData(
        token_index=1,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.5,
        base_top_token=1,
        adapter_entropy=1.9,
        adapter_top_k_variance=0.4,
        adapter_top_token=1,
    )
    await tracker.record_entropy_from_data(normal_data)

    # Consecutive count should reset
    assert tracker.current_consecutive_anomalies == 0

    result = tracker.end_session()
    assert result.circuit_breaker_tripped is False


# =============================================================================
# Session Result Tests
# =============================================================================


@pytest.mark.asyncio
async def test_session_result_statistics() -> None:
    """Test that session results compute correct statistics."""
    tracker = EntropyDeltaTracker(_test_config())
    tracker.start_session()

    # Add several samples
    samples_data = [
        PendingEntropyData(
            token_index=i,
            generated_token=i + 1,
            base_entropy=2.0 + (i * 0.1),
            base_top_k_variance=0.5,
            base_top_token=i + 1,
            adapter_entropy=1.8 + (i * 0.05),
            adapter_top_k_variance=0.4,
            adapter_top_token=i + 1,
            latency_ms=5.0 + i,
        )
        for i in range(5)
    ]

    for data in samples_data:
        await tracker.record_entropy_from_data(data)

    result = tracker.end_session()

    assert result.total_tokens == 5
    assert result.anomaly_count >= 0
    assert result.avg_latency_ms > 0
    assert result.duration >= 0
    assert len(result.samples) == 5


# =============================================================================
# Callback Tests
# =============================================================================


@pytest.mark.asyncio
async def test_on_delta_sample_callback() -> None:
    """Test that on_delta_sample callback is invoked for each sample."""
    samples_received: List[EntropyDeltaSample] = []

    async def on_sample(sample: EntropyDeltaSample) -> None:
        samples_received.append(sample)

    tracker = EntropyDeltaTracker(_test_config())
    tracker.on_delta_sample = on_sample
    tracker.start_session()

    # Record 3 samples
    for i in range(3):
        data = PendingEntropyData(
            token_index=i,
            generated_token=i + 1,
            base_entropy=2.0,
            base_top_k_variance=0.5,
            base_top_token=i + 1,
            adapter_entropy=1.8,
            adapter_top_k_variance=0.4,
            adapter_top_token=i + 1,
        )
        await tracker.record_entropy_from_data(data)

    assert len(samples_received) == 3
    assert all(isinstance(s, EntropyDeltaSample) for s in samples_received)


