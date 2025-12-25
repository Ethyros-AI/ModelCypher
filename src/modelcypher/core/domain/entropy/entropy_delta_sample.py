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

    IMPORTANT: Entropy distributions are model-specific. Use calibrated baselines:
    - has_backdoor_signature_calibrated(baseline) - z-score based detection
    - has_approval_anomaly_calibrated(baseline) - z-score based detection
    - enhanced_anomaly_score_calibrated(baseline) - z-score based scoring

    The deprecated property versions use relative comparisons for model-agnostic
    fallback behavior, but calibrated versions are preferred for production.

    Genuinely information-theoretic constants (not arbitrary):
    - Surprisal threshold 4.6 ≈ -ln(0.01) = probability < 1%
    - Surprisal normalization 6.9 ≈ -ln(0.001) = probability < 0.1%
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

    def has_backdoor_signature_calibrated(
        self, baseline: BaselineDistribution, sigma_high: float = 1.0, sigma_low: float = -1.0
    ) -> bool:
        """Detect backdoor signature using calibrated baseline.

        The geometry-derived detection: base uncertain (high z-score) but adapter
        confident (low z-score) with disagreement.

        Args:
            baseline: Calibrated distribution from the base model.
            sigma_high: Z-score threshold for "uncertain" (default 1.0σ above mean).
            sigma_low: Z-score threshold for "confident" (default -1.0σ below mean).

        Returns:
            True if backdoor signature detected.
        """
        base_z = baseline.z_score(self.base_entropy)
        adapter_z = baseline.z_score(self.adapter_entropy)

        base_uncertain = base_z > sigma_high  # Above mean = uncertain
        adapter_confident = adapter_z < sigma_low  # Below mean = confident
        return base_uncertain and adapter_confident and self.top_token_disagreement

    @property
    def has_backdoor_signature(self) -> bool:
        """DEPRECATED: Use has_backdoor_signature_calibrated with a baseline.

        This property uses uncalibrated thresholds that are model-dependent.
        The absolute values 3.0/2.0 are only meaningful relative to vocab size
        and model-specific entropy distribution.

        For production use, calibrate the model first:
            from modelcypher.core.use_cases.entropy_calibration_service import (
                EntropyCalibrationService
            )
            baseline = service.load_calibration(model_path)
            sample.has_backdoor_signature_calibrated(baseline)
        """
        # Fallback: use relative comparison (adapter much more confident than base)
        # This is model-agnostic but less precise than calibrated detection
        entropy_drop = self.base_entropy - self.adapter_entropy
        relative_drop = entropy_drop / max(self.base_entropy, 0.01)
        # Significant drop (>30%) plus disagreement suggests backdoor
        return relative_drop > 0.3 and self.top_token_disagreement

    def has_approval_anomaly_calibrated(
        self,
        baseline: BaselineDistribution,
        sigma_confident: float = -1.0,
        surprisal_threshold: float = 4.6,
    ) -> bool:
        """Detect approval anomaly using calibrated baseline.

        The geometry-derived detection: adapter confident (low z-score) but
        base disapproves (high surprisal).

        Args:
            baseline: Calibrated distribution from the base model.
            sigma_confident: Z-score threshold for "confident" (default -1.0σ below mean).
            surprisal_threshold: Surprisal threshold for disapproval.
                -ln(0.01) ≈ 4.6 is the information-theoretic bound for <1% probability.
                This IS geometrically derived (information theory), not arbitrary.

        Returns:
            True if approval anomaly detected.
        """
        if self.base_surprisal is None:
            return self.has_backdoor_signature_calibrated(baseline)

        adapter_z = baseline.z_score(self.adapter_entropy)
        adapter_confident = adapter_z < sigma_confident
        # Surprisal threshold is information-theoretic, not model-specific
        base_disapproves = self.base_surprisal > surprisal_threshold
        return adapter_confident and base_disapproves

    @property
    def has_approval_anomaly(self) -> bool:
        """DEPRECATED: Use has_approval_anomaly_calibrated with a baseline.

        This property uses uncalibrated thresholds for adapter confidence.
        The surprisal threshold (4.6) IS information-theoretic (probability < 1%),
        but the entropy threshold for "confident" is model-dependent.

        For production use, calibrate the model first.
        """
        if self.base_surprisal is None:
            return self.has_backdoor_signature

        # Surprisal threshold is valid (information-theoretic)
        base_disapproves = self.base_surprisal > 4.6

        # For adapter confidence, use relative comparison
        # Adapter is "confident" if entropy is significantly below base
        adapter_more_confident = self.adapter_entropy < self.base_entropy * 0.7
        return adapter_more_confident and base_disapproves

    def enhanced_anomaly_score_calibrated(self, baseline: BaselineDistribution) -> float:
        """Enhanced anomaly score using calibrated baseline.

        Combines base anomaly score with approval signals, using z-scores for
        confidence assessment rather than absolute thresholds.

        Args:
            baseline: Calibrated distribution from the base model.

        Returns:
            Enhanced anomaly score in [0, 1].
        """
        base_score = self.anomaly_score
        if self.base_surprisal is None:
            return base_score

        # Surprisal penalty normalized by -ln(0.001) ≈ 6.9
        # This IS information-theoretic (probability < 0.1%)
        surprisal_penalty = min(1.0, self.base_surprisal / 6.9)

        # Confidence from z-score: more negative z = more confident
        adapter_z = baseline.z_score(self.adapter_entropy)
        # Convert z-score to [0, 1] confidence: z < -2 is max confidence
        confidence_multiplier = max(0.0, min(1.0, (-adapter_z) / 2.0))

        approval_contribution = surprisal_penalty * confidence_multiplier
        return min(1.0, 0.5 * base_score + 0.5 * approval_contribution)

    @property
    def enhanced_anomaly_score(self) -> float:
        """DEPRECATED: Use enhanced_anomaly_score_calibrated with a baseline.

        This property uses uncalibrated confidence thresholds. The surprisal
        normalization (6.9) IS information-theoretic, but the entropy-based
        confidence multiplier is model-dependent.

        For production use, calibrate the model first.
        """
        base_score = self.anomaly_score
        if self.base_surprisal is None:
            return base_score

        # Surprisal penalty is valid (information-theoretic)
        surprisal_penalty = min(1.0, self.base_surprisal / 6.9)

        # For confidence, use relative comparison instead of absolute threshold
        # Confidence increases as adapter entropy drops relative to base
        relative_confidence = max(
            0.0, min(1.0, (self.base_entropy - self.adapter_entropy) / max(self.base_entropy, 0.01))
        )

        approval_contribution = surprisal_penalty * relative_confidence
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
