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

"""Entropy Delta Sample: Raw geometric measurements of adapter-base divergence."""

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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

# =============================================================================
# Baseline Distribution for Geometry-Derived Comparisons
# =============================================================================


@dataclass(frozen=True)
class BaselineDistribution:
    """Baseline distribution learned from calibration data.

    Attributes
    ----------
    mean : float
        Mean value from calibration samples.
    std : float
        Standard deviation from calibration samples.
    """

    mean: float
    std: float

    def z_score(self, value: float) -> float:
        """Compute z-score: how many standard deviations from mean."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        if self.std < eps:
            return 0.0 if abs(value - self.mean) < eps else float("inf")
        return (value - self.mean) / self.std

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

    Notes
    -----
    Entropy distributions are model-specific. Use calibrated baselines:
    - anomaly_z_score(baseline) for normalization against calibration stats
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
        """Entropy ratio measuring base uncertainty relative to adapter confidence."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        positive_delta = max(0.0, self.delta)
        return positive_delta / max(self.base_entropy, eps)

    def anomaly_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute z-score relative to calibration baseline."""
        return baseline.z_score(self.anomaly_score)

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
        """Create anomaly signal with raw measurements."""
        return Signal(
            type=SignalType.system_event(SystemEvent.adapter_anomaly_detected),
            payload=self.to_signal_payload(),
            correlation_id=self.correlation_id,
            priority=Priority.normal,
            source=self.source,
        )


# =============================================================================
# Entropy Delta Session Result
# =============================================================================


@dataclass(frozen=True)
class EntropyDeltaSessionResult:
    """Aggregated entropy delta metrics over a generation session.

    Use raw measurements directly or with BaselineDistribution.z_score()
    for normalization against calibration stats.
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
    avg_base_surprisal: float | None = None
    max_base_surprisal: float | None = None
    conflict_analysis: ConflictAnalysis | None = None
    samples: list[EntropyDeltaSample] = field(default_factory=list)

    def security_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute security z-score relative to calibration baseline."""
        return baseline.z_score(self.max_anomaly_score)

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
