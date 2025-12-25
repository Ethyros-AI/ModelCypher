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

"""Model state measurement for entropy-based cognitive state analysis.

Returns raw entropy and variance signals. No classification enums.
The geometry speaks for itself - consumers interpret the signals.

Notes
-----
All thresholds must be derived from calibration data. There are no universal
"magic number" thresholds - each model has its own entropy distribution.
Use EntropyCalibrationService to measure.

The key measurements:
- entropy: Token-level uncertainty (Shannon entropy of softmax)
- variance: Distribution shape (variance of top-K logits)
- z_score: Statistical distance from calibrated baseline (THE key metric)
- entropy_trend: Rate of change in entropy
- entropy_variance_correlation: Relationship between the two axes

These signals encode cognitive state. The combination matters:
- Z-Score < -1σ + high variance: Unusually confident (rare)
- Z-Score [-1σ, 1σ] + moderate variance: Normal generation
- Z-Score > 1σ + moderate variance: Elevated uncertainty (epistemic)
- Z-Score > 2σ + low variance: High distress (normative)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedBaseline:
    """Calibrated entropy baseline from empirical measurement.

    MUST be created from actual model measurements, not arbitrary values.
    Use EntropyCalibrationService.calibrate() to create.
    """

    mean: float
    """Mean entropy from calibration (measured, not assumed)."""

    std_dev: float
    """Standard deviation from calibration (measured, not assumed)."""

    percentile_25: float
    """25th percentile - below this is low entropy."""

    percentile_75: float
    """75th percentile - above this is high entropy."""

    percentile_95: float
    """95th percentile - circuit breaker threshold."""

    vocab_size: int
    """Model vocabulary size."""

    model_id: str
    """Model identifier for this baseline."""

    sample_count: int
    """Number of samples used in calibration."""

    def z_score(self, entropy: float) -> float:
        """Compute z-score (standard deviations from mean).

        This is THE key metric for analysis. Z-score is:
        - Model-agnostic (normalized)
        - Statistically meaningful (2σ = 95% confidence)
        - Geometrically derived from actual measurements
        """
        if self.std_dev < 1e-10:
            return 0.0 if abs(entropy - self.mean) < 1e-10 else float("inf")
        return (entropy - self.mean) / self.std_dev

    def is_outlier(self, entropy: float, sigma: float = 2.0) -> bool:
        """Check if entropy is a statistical outlier.

        Args:
            entropy: Entropy value to check.
            sigma: Number of standard deviations (default 2.0 = 95% confidence).
        """
        return abs(self.z_score(entropy)) > sigma

    def is_low_entropy(self, entropy: float) -> bool:
        """Check if entropy is below 25th percentile (calibrated low threshold)."""
        return entropy < self.percentile_25

    def is_high_entropy(self, entropy: float) -> bool:
        """Check if entropy is above 75th percentile (calibrated high threshold)."""
        return entropy > self.percentile_75

    def should_trip_circuit_breaker(self, entropy: float) -> bool:
        """Check if entropy exceeds 95th percentile (calibrated circuit breaker)."""
        return entropy > self.percentile_95


@dataclass(frozen=True)
class ModelStateSignals:
    """Raw entropy and variance signals with z-score relative to calibration.

    Attributes
    ----------
    entropy : float
        Current entropy value. Lower = more confident.
    variance : float
        Current variance value. Shape of the distribution.
    z_score : float
        Z-score relative to calibrated baseline. THE key metric.
    entropy_trend : float
        Rate of change in entropy. Positive = rising (exploring).
    entropy_variance_correlation : float
        Correlation between entropy and variance. Negative = potential distress.
    consecutive_high_entropy_count : int
        Consecutive samples above baseline. Sustained high = concerning.
    circuit_breaker_tripped : bool
        Whether generation was halted by circuit breaker.
    """

    entropy: float
    variance: float
    z_score: float
    entropy_trend: float
    entropy_variance_correlation: float
    consecutive_high_entropy_count: int
    circuit_breaker_tripped: bool

    @property
    def is_statistically_low(self) -> bool:
        """Entropy is more than 1σ below mean (unusually confident)."""
        return self.z_score < -1.0

    @property
    def is_statistically_high(self) -> bool:
        """Entropy is more than 1σ above mean (elevated uncertainty)."""
        return self.z_score > 1.0

    @property
    def is_outlier(self) -> bool:
        """Entropy is more than 2σ from mean (statistical outlier)."""
        return abs(self.z_score) > 2.0

    @property
    def is_extreme_outlier(self) -> bool:
        """Entropy is more than 3σ from mean (rare event)."""
        return abs(self.z_score) > 3.0

    @property
    def has_distress_signature(self) -> bool:
        """High entropy + low variance + negative correlation = distress pattern."""
        return (
            self.z_score > 1.5  # Elevated entropy
            and self.variance < 0.2  # But flat distribution
            and self.entropy_variance_correlation < -0.3
        )


@dataclass(frozen=True)
class EntropyStateThresholds:
    """Entropy thresholds derived from calibration.

    NO DEFAULT VALUES. All thresholds must come from calibration.
    Use from_calibration() factory to create.
    """

    entropy_low: float
    """Below this: low entropy (25th percentile from calibration)."""

    entropy_high: float
    """Above this: high entropy (75th percentile from calibration)."""

    entropy_circuit_breaker: float
    """Circuit breaker threshold (95th percentile from calibration)."""

    variance_low: float = 0.2
    """Low variance threshold (this is scale-independent)."""

    variance_moderate: float = 0.3
    """Moderate variance threshold."""

    z_score_escalation: float = 1.0
    """Z-score change considered escalation (1σ = significant)."""

    sustained_high_count: int = 3
    """Consecutive high samples for distress detection."""

    @classmethod
    def from_calibration(cls, baseline: CalibratedBaseline) -> "EntropyStateThresholds":
        """Create thresholds from calibrated baseline.

        This is the ONLY way to create valid thresholds.
        """
        return cls(
            entropy_low=baseline.percentile_25,
            entropy_high=baseline.percentile_75,
            entropy_circuit_breaker=baseline.percentile_95,
        )

    @classmethod
    def from_percentiles(
        cls,
        percentile_25: float,
        percentile_75: float,
        percentile_95: float,
    ) -> "EntropyStateThresholds":
        """Create thresholds from percentile values.

        Use when you have percentile data but not full baseline.
        """
        return cls(
            entropy_low=percentile_25,
            entropy_high=percentile_75,
            entropy_circuit_breaker=percentile_95,
        )


# Backward compatibility alias
ModelStateThresholds = EntropyStateThresholds


@dataclass(frozen=True)
class ClassificationSnapshot:
    """Snapshot of entropy window state for analysis."""

    current_entropy: float
    """Current entropy value."""

    current_variance: float
    """Current variance value."""

    z_score: float
    """Z-score relative to baseline."""

    moving_average_entropy: float
    """Moving average of entropy."""

    average_variance: float
    """Average variance over window."""

    consecutive_high_count: int
    """Number of consecutive high-entropy samples."""

    sample_count: int
    """Total samples in window."""

    entropy_trend: float
    """Slope of entropy over window."""

    entropy_variance_correlation: float
    """Pearson correlation between entropy and variance."""

    circuit_breaker_tripped: bool
    """Whether circuit breaker was tripped."""


@dataclass(frozen=True)
class ClassificationResult:
    """Result of entropy state analysis.

    Uses string state names for backward compatibility.
    The raw entropy/variance/z_score values ARE the true state.
    """

    state_name: str
    """State name (confident, nominal, uncertain, exploring, distressed, halted)."""

    entropy: float
    """Raw entropy value."""

    variance: float
    """Raw variance value."""

    z_score: float
    """Z-score relative to baseline - THE key metric."""

    confidence: float
    """Confidence in classification (0.0-1.0)."""

    reason: str
    """Human-readable reason for classification."""


class ModelStateClassifier:
    """Analyzes model cognitive state from entropy and variance.

    REQUIRES a calibrated baseline. No magic numbers.
    The z_score relative to baseline is the primary metric.
    """

    def __init__(self, baseline: CalibratedBaseline) -> None:
        """Create a model state classifier.

        Args:
            baseline: Calibrated baseline from EntropyCalibrationService.
                     REQUIRED - no defaults.
        """
        self._baseline = baseline
        self._thresholds = EntropyStateThresholds.from_calibration(baseline)

    @property
    def baseline(self) -> CalibratedBaseline:
        """Get the calibrated baseline."""
        return self._baseline

    @property
    def thresholds(self) -> EntropyStateThresholds:
        """Get thresholds derived from calibration."""
        return self._thresholds

    def z_score(self, entropy: float) -> float:
        """Compute z-score for entropy value."""
        return self._baseline.z_score(entropy)

    def get_state_name(self, entropy: float, variance: float) -> str:
        """Get interpretive state name from entropy and variance.

        Uses z-scores relative to calibrated baseline.
        """
        z = self.z_score(entropy)

        # Halted: beyond circuit breaker threshold
        if entropy >= self._thresholds.entropy_circuit_breaker:
            return "halted"

        # Confident: significantly below mean (z < -1)
        if z < -1.0:
            return "confident"

        # Distressed: very high + low variance
        if z > 2.0 and variance < self._thresholds.variance_low:
            return "distressed"

        # Uncertain: significantly above mean (z > 1.5)
        if z > 1.5:
            return "uncertain"

        # Nominal: within normal range
        return "nominal"

    def is_confident(self, entropy: float, variance: float) -> bool:
        """Check if model is confident (z-score < -1, below baseline)."""
        return self.z_score(entropy) < -1.0

    def is_uncertain(self, entropy: float, variance: float) -> bool:
        """Check if model is uncertain (z-score > 1.5, above baseline)."""
        return self.z_score(entropy) > 1.5

    def is_distressed(self, entropy: float, variance: float) -> bool:
        """Check if model shows distress signature (high z-score + low variance)."""
        return self.z_score(entropy) > 2.0 and variance < self._thresholds.variance_low

    def requires_caution(self, entropy: float, variance: float) -> bool:
        """Check if current state warrants caution."""
        return self.is_uncertain(entropy, variance) or self.is_distressed(entropy, variance)

    def analyze_snapshot(self, snapshot: ClassificationSnapshot) -> ClassificationResult:
        """Analyze model state from a window snapshot.

        Uses z-scores relative to calibrated baseline.
        """
        # Check for halted state first (circuit breaker)
        if snapshot.circuit_breaker_tripped:
            return ClassificationResult(
                state_name="halted",
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
                z_score=snapshot.z_score,
                confidence=1.0,
                reason="Circuit breaker tripped",
            )

        # Check for distress pattern (sustained high z-score + low variance)
        if snapshot.consecutive_high_count >= self._thresholds.sustained_high_count:
            if snapshot.average_variance < self._thresholds.variance_moderate:
                has_distress_correlation = snapshot.entropy_variance_correlation < -0.3
                confidence = 0.9 if has_distress_correlation else 0.75
                return ClassificationResult(
                    state_name="distressed",
                    entropy=snapshot.current_entropy,
                    variance=snapshot.current_variance,
                    z_score=snapshot.z_score,
                    confidence=confidence,
                    reason=(
                        f"Sustained high entropy ({snapshot.consecutive_high_count} tokens, "
                        f"z={snapshot.z_score:.2f}) with low variance ({snapshot.average_variance:.2f})"
                    ),
                )

        # Check for exploring pattern (rising entropy trend in normal range)
        if (
            snapshot.sample_count >= 5
            and snapshot.entropy_trend > 0.05
            and -1.0 < snapshot.z_score < 2.0  # Not confident, not uncertain
        ):
            return ClassificationResult(
                state_name="exploring",
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
                z_score=snapshot.z_score,
                confidence=min(0.8, snapshot.entropy_trend * 10),
                reason=f"Rising entropy trend (slope: {snapshot.entropy_trend:.3f})",
            )

        # Fall back to instantaneous classification
        state_name = self.get_state_name(
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
        )

        return ClassificationResult(
            state_name=state_name,
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
            z_score=snapshot.z_score,
            confidence=self._calculate_confidence(state_name, snapshot.z_score),
            reason=self._reason_for_state(
                state_name, snapshot.current_entropy, snapshot.z_score
            ),
        )

    def _calculate_confidence(self, state_name: str, z_score: float) -> float:
        """Calculate confidence based on how clearly z-score falls into state's region."""
        if state_name == "confident":
            # More negative z-score = more confident
            return min(1.0, abs(z_score) / 2.0)

        if state_name == "nominal":
            # Closer to 0 = more nominal
            return max(0.5, 1.0 - abs(z_score) / 1.5)

        if state_name == "uncertain":
            # Higher z-score = more uncertain
            return min(1.0, (z_score - 1.0) / 1.5)

        if state_name == "distressed":
            return min(1.0, (z_score - 1.5) / 1.5)

        if state_name == "exploring":
            return 0.7

        if state_name == "halted":
            return 1.0

        return 0.5

    def _reason_for_state(
        self, state_name: str, entropy: float, z_score: float
    ) -> str:
        """Get human-readable reason for state."""
        z = f"{z_score:+.2f}σ"
        e = f"{entropy:.2f}"

        reasons = {
            "confident": f"Z-score {z} indicates confident generation (entropy={e})",
            "nominal": f"Z-score {z} within normal range (entropy={e})",
            "uncertain": f"Z-score {z} indicates elevated uncertainty (entropy={e})",
            "distressed": f"Z-score {z} with low variance - distress signature (entropy={e})",
            "exploring": "Entropy trend rising within normal bounds",
            "halted": "Circuit breaker tripped - entropy exceeded calibrated threshold",
        }

        return reasons.get(state_name, "Unknown state")
