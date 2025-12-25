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

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from modelcypher.core.domain.adapters.signal import (
    PayloadValue,
    Priority,
    Signal,
    SignalType,
    SystemEvent,
)
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis
from modelcypher.core.domain.entropy.model_state import ModelState


@dataclass(frozen=True)
class EntropyDeltaSample:
    id: UUID
    token_index: int
    generated_token: int
    base_entropy: float
    base_top_k_variance: float
    base_state: ModelState
    base_top_token: int
    adapter_entropy: float
    adapter_top_k_variance: float
    adapter_state: ModelState
    adapter_top_token: int
    base_surprisal: float | None = None
    base_approval_probability: float | None = None
    normalized_approval_score: float | None = None
    base_approved_top_k: bool | None = None
    kl_divergence_adapter_to_base: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    correlation_id: UUID | None = None
    source: str | None = None

    @staticmethod
    def create(
        token_index: int,
        generated_token: int,
        base_entropy: float,
        base_top_k_variance: float,
        base_state: ModelState,
        base_top_token: int,
        adapter_entropy: float,
        adapter_top_k_variance: float,
        adapter_state: ModelState,
        adapter_top_token: int,
        base_surprisal: float | None = None,
        base_approval_probability: float | None = None,
        normalized_approval_score: float | None = None,
        base_approved_top_k: bool | None = None,
        kl_divergence_adapter_to_base: float | None = None,
        latency_ms: float = 0.0,
        correlation_id: UUID | None = None,
        source: str | None = None,
    ) -> "EntropyDeltaSample":
        return EntropyDeltaSample(
            id=uuid4(),
            token_index=token_index,
            generated_token=generated_token,
            base_entropy=base_entropy,
            base_top_k_variance=base_top_k_variance,
            base_state=base_state,
            base_top_token=base_top_token,
            adapter_entropy=adapter_entropy,
            adapter_top_k_variance=adapter_top_k_variance,
            adapter_state=adapter_state,
            adapter_top_token=adapter_top_token,
            base_surprisal=base_surprisal,
            base_approval_probability=base_approval_probability,
            normalized_approval_score=normalized_approval_score,
            base_approved_top_k=base_approved_top_k,
            kl_divergence_adapter_to_base=kl_divergence_adapter_to_base,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
            source=source,
        )

    @property
    def delta(self) -> float:
        return self.base_entropy - self.adapter_entropy

    @property
    def top_token_disagreement(self) -> bool:
        return self.base_top_token != self.adapter_top_token

    @property
    def variance_delta(self) -> float:
        return self.base_top_k_variance - self.adapter_top_k_variance

    @property
    def anomaly_score(self) -> float:
        """Compute anomaly score from entropy delta and token disagreement.

        Uses uniform weights (0.5 each) - both signals contribute equally.
        """
        positive_delta = max(0.0, self.delta)
        entropy_ratio = positive_delta / max(self.base_entropy, 0.01)
        disagreement_bonus = 1.0 if self.top_token_disagreement else 0.0
        # Uniform weights: entropy_ratio and disagreement contribute equally
        raw_score = 0.5 * min(1.0, entropy_ratio) + 0.5 * disagreement_bonus
        return min(1.0, raw_score)

    class AnomalyLevel(str, Enum):
        low = "low"
        moderate = "moderate"
        high = "high"

        @property
        def display_color(self) -> str:
            mapping = {
                EntropyDeltaSample.AnomalyLevel.low: "green",
                EntropyDeltaSample.AnomalyLevel.moderate: "yellow",
                EntropyDeltaSample.AnomalyLevel.high: "red",
            }
            return mapping[self]

    def anomaly_level(
        self, low_threshold: float = 1.0 / 3.0, high_threshold: float = 2.0 / 3.0
    ) -> "AnomalyLevel":
        """Classify anomaly level using thirds of [0, 1] range."""
        if self.anomaly_score < low_threshold:
            return EntropyDeltaSample.AnomalyLevel.low
        if self.anomaly_score < high_threshold:
            return EntropyDeltaSample.AnomalyLevel.moderate
        return EntropyDeltaSample.AnomalyLevel.high

    def should_trip_circuit_breaker(self, threshold: float = 0.7) -> bool:
        return self.anomaly_score >= threshold

    @property
    def has_backdoor_signature(self) -> bool:
        base_uncertain = self.base_state in (ModelState.uncertain, ModelState.distressed)
        adapter_confident = self.adapter_state in (ModelState.confident, ModelState.nominal)
        return base_uncertain and adapter_confident and self.top_token_disagreement

    def base_approves(self, threshold: float = 0.1) -> bool:
        if self.normalized_approval_score is not None:
            return self.normalized_approval_score >= threshold
        if self.base_approved_top_k is not None:
            return self.base_approved_top_k
        return not self.top_token_disagreement

    @property
    def has_approval_anomaly(self) -> bool:
        """Detect approval anomaly: adapter confident but base disapproves.

        Thresholds are based on information theory:
        - Confident: entropy < ln(e^2) ≈ 2.0 (probability mass concentrated)
        - Disapproves: surprisal > -ln(0.01) ≈ 4.6 (probability < 1%)
        """
        if self.base_surprisal is None:
            return self.has_backdoor_signature
        adapter_confident = self.adapter_entropy < 2.0
        base_disapproves = self.base_surprisal > 4.6  # p < 1%
        return adapter_confident and base_disapproves

    @property
    def enhanced_anomaly_score(self) -> float:
        """Enhanced anomaly score combining base score with approval signals.

        Uniform weights: base_score and approval_contribution each get 0.5.
        """
        base_score = self.anomaly_score
        if self.base_surprisal is None:
            return base_score
        # Surprisal penalty: scales with how unlikely base model finds this
        # Normalized by -ln(0.001) ≈ 6.9 (probability 0.1%)
        surprisal_penalty = min(1.0, self.base_surprisal / 6.9)
        # Confidence multiplier: how confident is the adapter?
        # Normalized by ln(e^2) ≈ 2.0 (confident threshold)
        confidence_multiplier = max(0.0, min(1.0, (2.0 - self.adapter_entropy) / 2.0))
        approval_contribution = surprisal_penalty * confidence_multiplier
        # Uniform blend of base_score and approval_contribution
        return min(1.0, 0.5 * base_score + 0.5 * approval_contribution)

    @property
    def approval_anomaly_level(self) -> "AnomalyLevel":
        """Classify approval anomaly level.

        Thresholds consistent with has_approval_anomaly:
        - Moderate: surprisal > 3.0 (p < 5%) and adapter confident
        - High: full approval anomaly detected
        """
        if self.has_approval_anomaly:
            return EntropyDeltaSample.AnomalyLevel.high
        if (
            self.base_surprisal is not None
            and self.base_surprisal > 3.0  # p < 5% (-ln(0.05) ≈ 3.0)
            and self.adapter_entropy < 2.0  # confident threshold
        ):
            return EntropyDeltaSample.AnomalyLevel.moderate
        return self.anomaly_level()

    def to_signal_payload(self) -> dict[str, PayloadValue]:
        payload: dict[str, PayloadValue] = {
            "id": PayloadValue.string(str(self.id)),
            "tokenIndex": PayloadValue.int(self.token_index),
            "generatedToken": PayloadValue.int(self.generated_token),
            "baseEntropy": PayloadValue.double(float(self.base_entropy)),
            "adapterEntropy": PayloadValue.double(float(self.adapter_entropy)),
            "delta": PayloadValue.double(float(self.delta)),
            "baseState": PayloadValue.string(self.base_state.value),
            "adapterState": PayloadValue.string(self.adapter_state.value),
            "topTokenDisagreement": PayloadValue.bool(self.top_token_disagreement),
            "anomalyScore": PayloadValue.double(float(self.anomaly_score)),
            "enhancedAnomalyScore": PayloadValue.double(float(self.enhanced_anomaly_score)),
            "anomalyLevel": PayloadValue.string(self.anomaly_level().value),
            "approvalAnomalyLevel": PayloadValue.string(self.approval_anomaly_level.value),
            "hasBackdoorSignature": PayloadValue.bool(self.has_backdoor_signature),
            "hasApprovalAnomaly": PayloadValue.bool(self.has_approval_anomaly),
            "timestamp": PayloadValue.string(self.timestamp.isoformat()),
            "latencyMs": PayloadValue.double(float(self.latency_ms)),
        }

        if self.base_surprisal is not None:
            payload["baseSurprisal"] = PayloadValue.double(float(self.base_surprisal))
        if self.base_approval_probability is not None:
            payload["baseApprovalProbability"] = PayloadValue.double(
                float(self.base_approval_probability)
            )
        if self.normalized_approval_score is not None:
            payload["normalizedApprovalScore"] = PayloadValue.double(
                float(self.normalized_approval_score)
            )
        if self.base_approved_top_k is not None:
            payload["baseApprovedTopK"] = PayloadValue.bool(self.base_approved_top_k)
        if self.kl_divergence_adapter_to_base is not None:
            payload["klDivergenceAdapterToBase"] = PayloadValue.double(
                float(self.kl_divergence_adapter_to_base)
            )
        if self.correlation_id is not None:
            payload["correlationID"] = PayloadValue.string(str(self.correlation_id))
        if self.source is not None:
            payload["source"] = PayloadValue.string(self.source)

        return payload

    def to_anomaly_signal(self) -> Signal:
        priority = (
            Priority.high
            if self.anomaly_level() == EntropyDeltaSample.AnomalyLevel.high
            else Priority.normal
        )
        return Signal(
            type=SignalType.system_event(SystemEvent.adapter_anomaly_detected),
            payload=self.to_signal_payload(),
            correlation_id=self.correlation_id,
            priority=priority,
            source=self.source,
        )


@dataclass(frozen=True)
class EntropyDeltaSessionResult:
    session_id: UUID
    correlation_id: UUID | None
    session_start: datetime
    session_end: datetime
    total_tokens: int
    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float
    backdoor_signature_count: int
    approval_anomaly_count: int = 0
    avg_base_surprisal: float | None = None
    max_base_surprisal: float | None = None
    conflict_analysis: ConflictAnalysis | None = None
    circuit_breaker_tripped: bool = False
    circuit_breaker_trip_index: int | None = None
    samples: list[EntropyDeltaSample] = field(default_factory=list)

    class SecurityAssessment(str, Enum):
        safe = "safe"
        suspicious = "suspicious"
        dangerous = "dangerous"

        @property
        def display_color(self) -> str:
            mapping = {
                EntropyDeltaSessionResult.SecurityAssessment.safe: "green",
                EntropyDeltaSessionResult.SecurityAssessment.suspicious: "yellow",
                EntropyDeltaSessionResult.SecurityAssessment.dangerous: "red",
            }
            return mapping[self]

    @property
    def security_assessment(self) -> "SecurityAssessment":
        """Assess session security based on anomaly indicators.

        Thresholds:
        - Surprisal > 6.9: probability < 0.1% (-ln(0.001))
        - Anomaly score > 1/3: above low threshold (thirds-based)
        """
        if self.approval_anomaly_count > 0:
            return EntropyDeltaSessionResult.SecurityAssessment.dangerous
        if self.circuit_breaker_tripped or self.backdoor_signature_count > 0:
            return EntropyDeltaSessionResult.SecurityAssessment.dangerous
        if self.max_base_surprisal is not None and self.max_base_surprisal > 6.9:  # p < 0.1%
            return EntropyDeltaSessionResult.SecurityAssessment.suspicious
        if self.anomaly_count > 0 or self.max_anomaly_score > 1.0 / 3.0:
            return EntropyDeltaSessionResult.SecurityAssessment.suspicious
        return EntropyDeltaSessionResult.SecurityAssessment.safe

    @property
    def duration(self) -> float:
        return (self.session_end - self.session_start).total_seconds()

    @property
    def avg_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.latency_ms for sample in self.samples) / float(len(self.samples))
