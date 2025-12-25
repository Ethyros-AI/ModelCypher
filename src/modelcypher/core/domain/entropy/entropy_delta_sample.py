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
Entropy Delta Sample: Raw geometric measurements of adapter-base divergence.

The anomaly_score IS the anomaly state. The max_anomaly_score IS the security
assessment. Raw measurements, not categorical bins.

For outlier detection, use BaselineDistribution.is_outlier() which applies
z-score statistics from calibration data. 3σ is not arbitrary - it's the
geometry of normal distributions (99.7% of data falls within).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from modelcypher.core.domain.adapters.signal import (
    PayloadValue,
    Priority,
    Signal,
    SignalType,
    SystemEvent,
)
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis


# =============================================================================
# Baseline Distribution for Geometry-Derived Outlier Detection
# =============================================================================


@dataclass(frozen=True)
class BaselineDistribution:
    """Baseline distribution learned from calibration data.

    The geometry of normal operation. All decisions become:
    "is this point an outlier from the learned geometry?"

    3σ threshold is not arbitrary - it's the geometry of normal distributions
    (99.7% of data falls within 3σ).
    """

    mean: float
    std: float

    def z_score(self, value: float) -> float:
        """Compute z-score: how many standard deviations from mean."""
        if self.std < 1e-10:
            return 0.0 if abs(value - self.mean) < 1e-10 else float("inf")
        return (value - self.mean) / self.std

    def is_outlier(self, value: float, sigma: float = 3.0) -> bool:
        """Check if value is an outlier (>sigma standard deviations from mean).

        Uses 3σ by default - the geometry of normal distributions.
        """
        return abs(self.z_score(value)) > sigma

    @classmethod
    def from_samples(cls, values: list[float]) -> "BaselineDistribution":
        """Compute baseline from calibration samples."""
        if not values:
            raise ValueError("Cannot compute baseline from empty samples")
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance**0.5
        return cls(mean=mean, std=std)


# =============================================================================
# Entropy Delta Sample
# =============================================================================


@dataclass(frozen=True)
class EntropyDeltaSample:
    """Raw geometric measurements of adapter-base divergence.

    The entropy and variance values ARE the cognitive state - no classification needed.
    Use anomaly_score directly or with BaselineDistribution.is_outlier() for
    geometry-derived outlier detection.

    Information-theoretic thresholds (not arbitrary):
    - Confident: entropy < ln(e²) ≈ 2.0 (probability mass concentrated)
    - Uncertain: entropy > 3.0 (high uncertainty)
    - Distress signature: high entropy + low variance (normative uncertainty)
    """

    id: UUID
    token_index: int
    generated_token: int
    base_entropy: float
    base_top_k_variance: float
    base_top_token: int
    adapter_entropy: float
    adapter_top_k_variance: float
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
        base_top_token: int,
        adapter_entropy: float,
        adapter_top_k_variance: float,
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
            base_top_token=base_top_token,
            adapter_entropy=adapter_entropy,
            adapter_top_k_variance=adapter_top_k_variance,
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
        """Entropy delta: base - adapter."""
        return self.base_entropy - self.adapter_entropy

    @property
    def top_token_disagreement(self) -> bool:
        """Whether base and adapter disagree on top token."""
        return self.base_top_token != self.adapter_top_token

    @property
    def variance_delta(self) -> float:
        """Variance delta: base - adapter."""
        return self.base_top_k_variance - self.adapter_top_k_variance

    @property
    def anomaly_score(self) -> float:
        """Raw anomaly score from entropy delta and token disagreement.

        This IS the anomaly measurement - use directly or with
        BaselineDistribution.is_outlier() for geometry-derived detection.
        """
        positive_delta = max(0.0, self.delta)
        entropy_ratio = positive_delta / max(self.base_entropy, 0.01)
        disagreement_bonus = 1.0 if self.top_token_disagreement else 0.0
        raw_score = 0.5 * min(1.0, entropy_ratio) + 0.5 * disagreement_bonus
        return min(1.0, raw_score)

    def is_anomaly_outlier(self, baseline: BaselineDistribution) -> bool:
        """Check if this sample is an anomaly outlier using geometry-derived detection.

        Uses z-score from calibration baseline. 3σ threshold is the geometry
        of normal distributions (99.7% of data falls within).
        """
        return baseline.is_outlier(self.anomaly_score)

    def anomaly_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute z-score relative to calibration baseline.

        The z-score IS the anomaly significance.
        """
        return baseline.z_score(self.anomaly_score)

    @property
    def has_backdoor_signature(self) -> bool:
        """Detect backdoor signature: base uncertain but adapter confident with disagreement.

        Uses information-theoretic thresholds:
        - Base uncertain: entropy > 3.0 (high uncertainty)
        - Adapter confident: entropy < 2.0 (probability mass concentrated)
        """
        base_uncertain = self.base_entropy > 3.0
        adapter_confident = self.adapter_entropy < 2.0
        return base_uncertain and adapter_confident and self.top_token_disagreement

    @property
    def has_approval_anomaly(self) -> bool:
        """Detect approval anomaly: adapter confident but base disapproves.

        Uses information-theoretic thresholds:
        - Confident: entropy < ln(e²) ≈ 2.0 (probability mass concentrated)
        - Disapproves: surprisal > -ln(0.01) ≈ 4.6 (probability < 1%)
        """
        if self.base_surprisal is None:
            return self.has_backdoor_signature
        adapter_confident = self.adapter_entropy < 2.0
        base_disapproves = self.base_surprisal > 4.6
        return adapter_confident and base_disapproves

    @property
    def enhanced_anomaly_score(self) -> float:
        """Enhanced anomaly score combining base score with approval signals.

        This IS the enhanced anomaly measurement.
        """
        base_score = self.anomaly_score
        if self.base_surprisal is None:
            return base_score
        # Surprisal penalty normalized by -ln(0.001) ≈ 6.9
        surprisal_penalty = min(1.0, self.base_surprisal / 6.9)
        # Confidence multiplier normalized by ln(e²) ≈ 2.0
        confidence_multiplier = max(0.0, min(1.0, (2.0 - self.adapter_entropy) / 2.0))
        approval_contribution = surprisal_penalty * confidence_multiplier
        return min(1.0, 0.5 * base_score + 0.5 * approval_contribution)

    def to_signal_payload(self) -> dict[str, PayloadValue]:
        """Convert to signal payload with raw measurements."""
        payload: dict[str, PayloadValue] = {
            "id": PayloadValue.string(str(self.id)),
            "tokenIndex": PayloadValue.int(self.token_index),
            "generatedToken": PayloadValue.int(self.generated_token),
            "baseEntropy": PayloadValue.double(float(self.base_entropy)),
            "baseVariance": PayloadValue.double(float(self.base_top_k_variance)),
            "adapterEntropy": PayloadValue.double(float(self.adapter_entropy)),
            "adapterVariance": PayloadValue.double(float(self.adapter_top_k_variance)),
            "delta": PayloadValue.double(float(self.delta)),
            "topTokenDisagreement": PayloadValue.bool(self.top_token_disagreement),
            "anomalyScore": PayloadValue.double(float(self.anomaly_score)),
            "enhancedAnomalyScore": PayloadValue.double(float(self.enhanced_anomaly_score)),
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

    def to_anomaly_signal(self, baseline: BaselineDistribution | None = None) -> Signal:
        """Create anomaly signal with priority from geometry-derived detection."""
        is_outlier = baseline.is_outlier(self.anomaly_score) if baseline else False
        priority = Priority.high if is_outlier else Priority.normal
        return Signal(
            type=SignalType.system_event(SystemEvent.adapter_anomaly_detected),
            payload=self.to_signal_payload(),
            correlation_id=self.correlation_id,
            priority=priority,
            source=self.source,
        )


# =============================================================================
# Entropy Delta Session Result
# =============================================================================


@dataclass(frozen=True)
class EntropyDeltaSessionResult:
    """Aggregated entropy delta metrics over a generation session.

    Raw measurements: max_anomaly_score, backdoor_signature_count, etc.
    These ARE the security assessment - use directly or with
    BaselineDistribution.is_outlier() for geometry-derived detection.
    """

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

    def is_security_outlier(self, baseline: BaselineDistribution) -> bool:
        """Check if this session is a security outlier using geometry-derived detection.

        Uses z-score from calibration baseline on max_anomaly_score.
        """
        return baseline.is_outlier(self.max_anomaly_score)

    def security_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute security z-score relative to calibration baseline.

        The z-score IS the security significance.
        """
        return baseline.z_score(self.max_anomaly_score)

    @property
    def has_security_flags(self) -> bool:
        """Check if any security flags are raised.

        Raw boolean: circuit breaker, backdoor signatures, or approval anomalies.
        """
        return (
            self.circuit_breaker_tripped
            or self.backdoor_signature_count > 0
            or self.approval_anomaly_count > 0
        )

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return (self.session_end - self.session_start).total_seconds()

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per token in milliseconds."""
        if not self.samples:
            return 0.0
        return sum(sample.latency_ms for sample in self.samples) / float(len(self.samples))
