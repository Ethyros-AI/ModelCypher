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

    def is_outlier(self, entropy: float, sigma: float) -> bool:
        """Check if entropy is a statistical outlier.

        Args:
            entropy: Entropy value to check.
            sigma: Number of standard deviations from the mean.
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

@dataclass(frozen=True)
class EntropyStateThresholds:
    """Entropy thresholds derived from calibration.

    NO DEFAULT VALUES. All thresholds must come from calibration.
    """

    entropy_low: float
    """Below this: low entropy (25th percentile from calibration)."""

    entropy_high: float
    """Above this: high entropy (75th percentile from calibration)."""

    entropy_circuit_breaker: float
    """Circuit breaker threshold (95th percentile from calibration)."""

    variance_low: float
    """Low variance threshold (explicit)."""

    variance_moderate: float
    """Moderate variance threshold (explicit)."""

    z_confident: float
    """Z-score below this is confident."""

    z_uncertain: float
    """Z-score above this is uncertain."""

    z_distressed: float
    """Z-score above this is distressed (with low variance)."""

    z_extreme: float
    """Z-score above this is extreme (immediate circuit breaker)."""

    trend_min_samples: int
    """Minimum samples required for trend-based state detection."""

    trend_slope_threshold: float
    """Entropy trend slope required for exploring state."""

    distress_correlation_threshold: float
    """Entropy/variance correlation threshold for distress detection."""

    sustained_high_count: int
    """Consecutive high samples for distress detection."""


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

    Attributes
    ----------
    state_name : str
        State name (confident, nominal, uncertain, exploring, distressed, halted).
    entropy : float
        Raw entropy value.
    variance : float
        Raw variance value.
    """

    state_name: str
    entropy: float
    variance: float

    z_score: float
    """Z-score relative to baseline - THE key metric."""


class ModelStateClassifier:
    """Analyzes model cognitive state from entropy and variance.

    REQUIRES a calibrated baseline. No magic numbers.
    The z_score relative to baseline is the primary metric.
    """

    def __init__(self, baseline: CalibratedBaseline, thresholds: EntropyStateThresholds) -> None:
        """Create a model state classifier.

        Args:
            baseline: Calibrated baseline from EntropyCalibrationService.
                     REQUIRED - no defaults.
        """
        self._baseline = baseline
        self._thresholds = thresholds

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

        # Confident: below confident threshold
        if z < self._thresholds.z_confident:
            return "confident"

        # Distressed: very high + low variance
        if z > self._thresholds.z_distressed and variance < self._thresholds.variance_low:
            return "distressed"

        # Uncertain: above uncertain threshold
        if z > self._thresholds.z_uncertain:
            return "uncertain"

        # Nominal: within normal range
        return "nominal"

    def is_confident(self, entropy: float, variance: float) -> bool:
        """Check if model is confident (z-score < -1, below baseline)."""
        return self.z_score(entropy) < self._thresholds.z_confident

    def is_uncertain(self, entropy: float, variance: float) -> bool:
        """Check if model is uncertain (z-score > 1.5, above baseline)."""
        return self.z_score(entropy) > self._thresholds.z_uncertain

    def is_distressed(self, entropy: float, variance: float) -> bool:
        """Check if model shows distress signature (high z-score + low variance)."""
        return (
            self.z_score(entropy) > self._thresholds.z_distressed
            and variance < self._thresholds.variance_low
        )

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
            )

        # Check for distress pattern (sustained high z-score + low variance)
        if snapshot.consecutive_high_count >= self._thresholds.sustained_high_count:
            if snapshot.average_variance < self._thresholds.variance_moderate:
                has_distress_correlation = (
                    snapshot.entropy_variance_correlation
                    < self._thresholds.distress_correlation_threshold
                )
                return ClassificationResult(
                    state_name="distressed",
                    entropy=snapshot.current_entropy,
                    variance=snapshot.current_variance,
                    z_score=snapshot.z_score,
                )

        # Check for exploring pattern (rising entropy trend in normal range)
        if (
            snapshot.sample_count >= self._thresholds.trend_min_samples
            and snapshot.entropy_trend > self._thresholds.trend_slope_threshold
            and self._thresholds.z_confident
            < snapshot.z_score
            < self._thresholds.z_uncertain
        ):
            return ClassificationResult(
                state_name="exploring",
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
                z_score=snapshot.z_score,
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
        )
