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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    BaselineDistribution,
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _ops_eps(n_ops: int, *values: float) -> float:
    """Error bound for n floating-point operations.

    For a chain of n operations, error accumulates approximately as n * eps.
    This is standard numerical analysis for multi-operation error propagation.
    """
    return n_ops * _eps(*values)


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
        base_logit_margin=7.0,
        base_rank_fraction=0.05,
        latency_ms=12.0,
    )

    assert abs(sample.delta - 4.0) <= _eps(sample.delta, 4.0)
    assert sample.top_token_disagreement is True
    assert sample.anomaly_score > _eps(sample.anomaly_score)

    eps = _eps(sample.anomaly_score, 5.0)
    expected_ratio = 4.0 / max(5.0, eps)
    assert abs(sample.anomaly_score - expected_ratio) <= eps


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
    eps = _eps(payload["baseEntropy"].double_value, payload["adapterEntropy"].double_value)
    assert abs(payload["baseEntropy"].double_value - 1.0) <= eps
    assert abs(payload["adapterEntropy"].double_value - 1.2) <= eps
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
        samples=[sample],
    )

    eps = _eps(result.duration, result.avg_latency_ms, result.max_anomaly_score)
    assert abs(result.duration - 2.0) <= eps
    assert abs(result.avg_latency_ms - 3.0) <= eps
    assert abs(result.max_anomaly_score - 0.1) <= eps


def test_baseline_distribution_z_score() -> None:
    """Test z-score computation from baseline."""
    baseline = BaselineDistribution(mean=0.5, std=0.1)

    # z_score computes (value - mean) / std, a 2-operation chain.
    # Error accumulates: use 2*eps for the two-operation error bound.

    # At mean: z=0 (use input scale since output is 0)
    assert abs(baseline.z_score(0.5)) <= _ops_eps(2, 0.5)

    # 1 std above: z=1 (use eps relative to expected output)
    assert abs(baseline.z_score(0.6) - 1.0) <= _ops_eps(2, 1.0)

    # 3 std above: z=3 (use eps relative to expected output)
    assert abs(baseline.z_score(0.8) - 3.0) <= _ops_eps(2, 3.0)


def test_baseline_distribution_from_samples() -> None:
    """Test computing baseline from calibration samples."""
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    baseline = BaselineDistribution.from_samples(samples)

    eps = _eps(baseline.mean, baseline.std)
    assert abs(baseline.mean - 0.3) <= eps
    # std = sqrt(variance) where variance = mean of squared deviations
    expected_std = (0.02) ** 0.5  # variance = 0.02
    assert abs(baseline.std - expected_std) <= eps
