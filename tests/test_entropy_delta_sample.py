from __future__ import annotations

from datetime import datetime, timedelta

from modelcypher.core.domain.adapters.signal import SystemEvent
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)
from modelcypher.core.domain.entropy.model_state import ModelState


def test_entropy_delta_sample_anomaly_metrics() -> None:
    sample = EntropyDeltaSample.create(
        token_index=0,
        generated_token=42,
        base_entropy=5.0,
        base_top_k_variance=1.0,
        base_state=ModelState.uncertain,
        base_top_token=1,
        adapter_entropy=1.0,
        adapter_top_k_variance=0.5,
        adapter_state=ModelState.confident,
        adapter_top_token=2,
        base_surprisal=7.0,
        normalized_approval_score=0.05,
        latency_ms=12.0,
    )

    assert sample.delta == 4.0
    assert sample.top_token_disagreement is True
    assert sample.has_backdoor_signature is True
    assert sample.has_approval_anomaly is True
    assert sample.anomaly_score > 0.0
    assert sample.enhanced_anomaly_score >= sample.anomaly_score
    assert sample.base_approves(threshold=0.1) is False
    assert sample.approval_anomaly_level.value == "high"


def test_entropy_delta_sample_signal_payload() -> None:
    sample = EntropyDeltaSample.create(
        token_index=1,
        generated_token=3,
        base_entropy=1.0,
        base_top_k_variance=0.1,
        base_state=ModelState.nominal,
        base_top_token=3,
        adapter_entropy=1.2,
        adapter_top_k_variance=0.2,
        adapter_state=ModelState.nominal,
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
    now = datetime.utcnow()
    sample = EntropyDeltaSample.create(
        token_index=0,
        generated_token=1,
        base_entropy=2.0,
        base_top_k_variance=0.2,
        base_state=ModelState.nominal,
        base_top_token=1,
        adapter_entropy=1.5,
        adapter_top_k_variance=0.1,
        adapter_state=ModelState.nominal,
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
    assert result.security_assessment.value == "safe"
