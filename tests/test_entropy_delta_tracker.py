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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    EntropyDeltaSample,
)
from modelcypher.core.domain.entropy.entropy_delta_tracker import (
    EntropyDeltaCalibration,
    EntropyDeltaTracker,
    PendingEntropyData,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
)


def _test_calibration(
    baseline_samples: list[float] | None = None,
    source: str = "EntropyDeltaTracker",
) -> EntropyDeltaCalibration:
    """Create test calibration derived from baseline samples."""
    samples = baseline_samples or [0.05, 0.1, 0.15, 0.2, 0.25]
    return EntropyDeltaCalibration.from_baseline_distribution(
        samples,
        source=source,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


def test_from_baseline_distribution() -> None:
    """Test deriving threshold from baseline distribution."""
    samples = [0.1, 0.2, 0.3, 5.0]
    calibration = EntropyDeltaCalibration.from_baseline_distribution(
        samples,
        source="calibration_test",
    )

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))
    expected = find_magnitude_gap_threshold(sorted(samples), eps=eps)
    assert abs(calibration.anomaly_threshold - expected) <= eps


def test_from_baseline_requires_samples() -> None:
    """Test that empty samples raises error."""
    with pytest.raises(ValueError, match="anomaly_score_samples required"):
        EntropyDeltaCalibration.from_baseline_distribution(
            [],
            source="calibration_test",
        )


# =============================================================================
# Session Lifecycle Tests
# =============================================================================


def test_session_lifecycle() -> None:
    """Test starting and ending a session."""
    calibration = _test_calibration()
    tracker = EntropyDeltaTracker(calibration)

    assert tracker.is_session_active is False
    assert tracker.correlation_id is None

    correlation_id = uuid4()
    tracker.start_session(correlation_id=correlation_id)

    assert tracker.is_session_active is True
    assert tracker.correlation_id == correlation_id
    assert tracker.current_sample_count == 0

    result = tracker.end_session()

    assert tracker.is_session_active is False
    assert result.total_tokens == 0
    assert result.anomaly_count == 0


def test_end_session_without_start() -> None:
    """Test ending a session that was never started."""
    tracker = EntropyDeltaTracker(_test_calibration())
    result = tracker.end_session()

    assert result.total_tokens == 0
    assert result.anomaly_count == 0


def test_session_auto_generates_correlation_id() -> None:
    """Test that starting a session without correlation_id generates one."""
    tracker = EntropyDeltaTracker(_test_calibration())
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
        base_entropy=3.5,
        base_logit_variance=0.8,
        base_top_token=101,
        adapter_entropy=1.2,
        adapter_logit_variance=0.3,
        adapter_top_token=102,
        base_logit_margin=6.5,
        kl_divergence_adapter_to_base=0.25,
        latency_ms=15.5,
    )

    assert data.token_index == 5
    assert data.base_entropy == 3.5
    assert data.adapter_entropy == 1.2
    assert data.base_logit_margin == 6.5


@pytest.mark.asyncio
async def test_record_entropy_from_data() -> None:
    """Test recording entropy from pre-computed data."""
    tracker = EntropyDeltaTracker(_test_calibration())
    tracker.start_session()

    data = PendingEntropyData(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_logit_variance=0.5,
        base_top_token=1,
        adapter_entropy=1.8,
        adapter_logit_variance=0.4,
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
async def test_anomaly_detection_callback() -> None:
    """High anomaly scores trigger callback."""
    tracker = EntropyDeltaTracker(_test_calibration())
    tracker.start_session()

    observed: List[EntropyDeltaSample] = []

    async def on_anomaly(sample: EntropyDeltaSample) -> None:
        observed.append(sample)

    tracker.on_anomaly_detected = on_anomaly

    data = PendingEntropyData(
        token_index=0,
        generated_token=1,
        base_entropy=10.0,
        base_logit_variance=0.5,
        base_top_token=1,
        adapter_entropy=0.1,
        adapter_logit_variance=0.4,
        adapter_top_token=2,
        latency_ms=5.0,
    )

    await tracker.record_entropy_from_data(data)

    assert len(observed) == 1


@pytest.mark.asyncio
async def test_session_summary_counts_anomalies() -> None:
    """Session summary counts anomalies by threshold."""
    calibration = _test_calibration()
    tracker = EntropyDeltaTracker(calibration)
    tracker.start_session()

    data = PendingEntropyData(
        token_index=0,
        generated_token=1,
        base_entropy=10.0,
        base_logit_variance=0.5,
        base_top_token=1,
        adapter_entropy=0.1,
        adapter_logit_variance=0.4,
        adapter_top_token=2,
        latency_ms=5.0,
    )
    await tracker.record_entropy_from_data(data)
    result = tracker.end_session()

    assert result.total_tokens == 1
    assert result.anomaly_count == 1
    assert result.max_anomaly_score >= calibration.anomaly_threshold
