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

"""Tests for EntropyDeltaSample - raw geometric measurements."""

from __future__ import annotations

from datetime import datetime, timedelta

from modelcypher.core.domain.adapters.signal import SystemEvent
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    BaselineDistribution,
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)


def test_entropy_delta_sample_anomaly_metrics() -> None:
    """Test anomaly metrics with raw entropy values."""
    # High base entropy (uncertain), low adapter entropy (confident), token disagreement
    sample = EntropyDeltaSample.create(
        token_index=0,
        generated_token=42,
        base_entropy=5.0,  # High = uncertain
        base_top_k_variance=1.0,
        base_top_token=1,
        adapter_entropy=1.0,  # Low = confident
        adapter_top_k_variance=0.5,
        adapter_top_token=2,  # Disagreement with base_top_token=1
        base_surprisal=7.0,
        normalized_approval_score=0.05,
        latency_ms=12.0,
    )

    assert sample.delta == 4.0
    assert sample.top_token_disagreement is True
    assert sample.has_backdoor_signature is True  # Uncertain base + confident adapter + disagreement
    assert sample.has_approval_anomaly is True
    assert sample.anomaly_score > 0.0
    assert sample.enhanced_anomaly_score >= sample.anomaly_score


def test_entropy_delta_sample_signal_payload() -> None:
    """Test signal payload contains raw measurements."""
    sample = EntropyDeltaSample.create(
        token_index=1,
        generated_token=3,
        base_entropy=1.0,
        base_top_k_variance=0.1,
        base_top_token=3,
        adapter_entropy=1.2,
        adapter_top_k_variance=0.2,
        adapter_top_token=3,
        latency_ms=5.0,
    )

    payload = sample.to_signal_payload()
    assert payload["baseEntropy"].double_value == 1.0
    assert payload["adapterEntropy"].double_value == 1.2
    assert payload["topTokenDisagreement"].bool_value is False

    signal = sample.to_anomaly_signal()
    assert signal.type.capability_string == f"system:{SystemEvent.adapter_anomaly_detected.value}"


def test_entropy_delta_session_metrics() -> None:
    """Test session result contains raw measurements."""
    now = datetime.utcnow()
    sample = EntropyDeltaSample.create(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.2,
        base_top_token=1,
        adapter_entropy=1.5,
        adapter_top_k_variance=0.1,
        adapter_top_token=1,
        latency_ms=3.0,
    )
    result = EntropyDeltaSessionResult(
        session_id=sample.id,
        correlation_id=None,
        session_start=now,
        session_end=now + timedelta(seconds=2),
        total_tokens=1,
        anomaly_count=0,
        max_anomaly_score=0.1,
        avg_delta=0.5,
        disagreement_rate=0.0,
        backdoor_signature_count=0,
        samples=[sample],
    )

    assert result.duration == 2.0
    assert result.avg_latency_ms == 3.0
    assert result.has_security_flags is False
    assert result.max_anomaly_score == 0.1


def test_baseline_distribution_z_score() -> None:
    """Test z-score computation from baseline."""
    baseline = BaselineDistribution(mean=0.5, std=0.1)

    # At mean: z=0
    assert abs(baseline.z_score(0.5)) < 0.001

    # 1 std above: z=1
    assert abs(baseline.z_score(0.6) - 1.0) < 0.001

    # 3 std above: z=3 (outlier)
    assert abs(baseline.z_score(0.8) - 3.0) < 0.001


def test_baseline_distribution_is_outlier() -> None:
    """Test outlier detection using 3σ threshold."""
    baseline = BaselineDistribution(mean=0.5, std=0.1)

    # Within 3σ: not outlier
    assert not baseline.is_outlier(0.5)
    assert not baseline.is_outlier(0.7)

    # Beyond 3σ: outlier
    assert baseline.is_outlier(0.85)  # > mean + 3*std


def test_baseline_distribution_from_samples() -> None:
    """Test computing baseline from calibration samples."""
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    baseline = BaselineDistribution.from_samples(samples)

    assert abs(baseline.mean - 0.3) < 0.001
    # std = sqrt(variance) where variance = mean of squared deviations
    expected_std = (0.02) ** 0.5  # variance = 0.02
    assert abs(baseline.std - expected_std) < 0.001


def test_sample_is_anomaly_outlier() -> None:
    """Test sample outlier detection using baseline."""
    baseline = BaselineDistribution(mean=0.1, std=0.05)

    # Normal sample (low anomaly score)
    normal = EntropyDeltaSample.create(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.2,
        base_top_token=1,
        adapter_entropy=2.0,  # Same as base
        adapter_top_k_variance=0.2,
        adapter_top_token=1,  # Same as base
    )

    # Anomalous sample (high entropy delta + disagreement)
    anomalous = EntropyDeltaSample.create(
        token_index=1,
        generated_token=2,
        base_entropy=5.0,  # High
        base_top_k_variance=0.2,
        base_top_token=1,
        adapter_entropy=0.5,  # Low
        adapter_top_k_variance=0.2,
        adapter_top_token=99,  # Different
    )

    # Normal should have low anomaly score, anomalous should have high
    assert normal.anomaly_score < anomalous.anomaly_score
