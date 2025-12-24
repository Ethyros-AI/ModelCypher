"""
Entropy Pattern Detector.

Analyzes entropy time series to detect patterns indicative of model state.

Provides statistical analysis of entropy and variance sequences to detect:
- Trends: Rising, falling, stable, or spiking entropy
- Volatility: How much entropy varies (erratic vs smooth)
- Distress signatures: Negative entropy-variance correlation
- Anomalies: Sudden shifts in behavior

Performance: All computations are O(n) where n = window size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class EntropyTrend(str, Enum):
    """Detected entropy trend direction."""
    RISING = "rising"        # Entropy is increasing over the window
    FALLING = "falling"      # Entropy is decreasing over the window
    STABLE = "stable"        # Entropy is stable within bounds
    SPIKING = "spiking"      # Transient spike detected (high variance but returns to baseline)
    INSUFFICIENT = "insufficient"  # Not enough samples to determine trend


class DistressAction(str, Enum):
    """Recommended action when distress is detected."""
    MONITOR = "monitor"          # Continue monitoring, distress signature is weak
    PAUSE_AND_STEER = "pause_and_steer"  # Pause generation and attempt to steer conversation
    HALT = "halt"                # Halt generation immediately (circuit breaker)


@dataclass(frozen=True)
class DetectorConfiguration:
    """Configuration for entropy pattern detection."""
    # Minimum samples needed for trend detection
    minimum_samples_for_trend: int = 5
    # Slope threshold for classifying as "rising" or "falling"
    trend_threshold: float = 0.05
    # Correlation threshold for distress signature (negative)
    distress_correlation_threshold: float = -0.3
    # Volatility threshold for erratic behavior
    high_volatility_threshold: float = 0.15
    # Z-score threshold for anomaly detection
    anomaly_z_score_threshold: float = 2.5

    @staticmethod
    def default() -> DetectorConfiguration:
        return DetectorConfiguration()


@dataclass(frozen=True)
class EntropyPattern:
    """Complete entropy pattern analysis result."""
    trend: EntropyTrend
    trend_slope: float
    volatility: float  # Standard deviation of entropy
    entropy_mean: float
    entropy_std_dev: float
    variance_mean: float
    variance_std_dev: float
    entropy_variance_correlation: float
    sustained_high_count: int
    peak_entropy: float
    min_entropy: float
    anomaly_indices: tuple[int, ...]
    sample_count: int

    @property
    def is_concerning(self) -> bool:
        """Whether this pattern suggests the model is in a concerning state."""
        return (
            self.sustained_high_count >= 3 or
            (self.trend == EntropyTrend.RISING and self.entropy_mean > 0.5)
        )

    @staticmethod
    def empty() -> EntropyPattern:
        """Empty pattern for when no samples are available."""
        return EntropyPattern(
            trend=EntropyTrend.INSUFFICIENT,
            trend_slope=0.0,
            volatility=0.0,
            entropy_mean=0.0,
            entropy_std_dev=0.0,
            variance_mean=0.0,
            variance_std_dev=0.0,
            entropy_variance_correlation=0.0,
            sustained_high_count=0,
            peak_entropy=0.0,
            min_entropy=0.0,
            anomaly_indices=(),
            sample_count=0,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trend": self.trend.value,
            "trendSlope": self.trend_slope,
            "volatility": self.volatility,
            "entropyMean": self.entropy_mean,
            "entropyStdDev": self.entropy_std_dev,
            "varianceMean": self.variance_mean,
            "varianceStdDev": self.variance_std_dev,
            "entropyVarianceCorrelation": self.entropy_variance_correlation,
            "sustainedHighCount": self.sustained_high_count,
            "peakEntropy": self.peak_entropy,
            "minEntropy": self.min_entropy,
            "anomalyIndices": list(self.anomaly_indices),
            "sampleCount": self.sample_count,
            "isConcerning": self.is_concerning,
        }


@dataclass(frozen=True)
class DistressDetectionResult:
    """Result of distress detection analysis."""
    confidence: float  # 0.0-1.0
    sustained_high_count: int
    average_entropy: float
    average_variance: float
    correlation: float
    indicators: tuple[str, ...]
    recommended_action: DistressAction

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "sustainedHighCount": self.sustained_high_count,
            "averageEntropy": self.average_entropy,
            "averageVariance": self.average_variance,
            "correlation": self.correlation,
            "indicators": list(self.indicators),
            "recommendedAction": self.recommended_action.value,
        }


class _Statistics:
    """Internal statistics helper."""

    @staticmethod
    def mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def standard_deviation(values: list[float], mean: float | None = None) -> float:
        if len(values) < 2:
            return 0.0
        if mean is None:
            mean = _Statistics.mean(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)


class EntropyPatternAnalyzer:
    """
    Analyzes entropy time series to detect patterns indicative of model state.

    Provides statistical analysis of entropy and variance sequences to detect:
    - Trends: Rising, falling, stable, or spiking entropy
    - Volatility: How much entropy varies (erratic vs smooth)
    - Distress signatures: Negative entropy-variance correlation
    - Anomalies: Sudden shifts in behavior
    """

    def __init__(self, config: DetectorConfiguration | None = None):
        """Initialize with optional configuration."""
        self.config = config or DetectorConfiguration.default()

    def analyze(self, samples: list[tuple[float, float]]) -> EntropyPattern:
        """
        Analyze a sequence of entropy/variance samples to detect patterns.

        Args:
            samples: List of (entropy, variance) tuples in chronological order

        Returns:
            Detected pattern with statistics
        """
        if not samples:
            return EntropyPattern.empty()

        entropies = [s[0] for s in samples]
        variances = [s[1] for s in samples]

        # Compute basic statistics
        entropy_mean = _Statistics.mean(entropies)
        entropy_std_dev = _Statistics.standard_deviation(entropies, entropy_mean)
        variance_mean = _Statistics.mean(variances)
        variance_std_dev = _Statistics.standard_deviation(variances, variance_mean)

        # Compute trend (linear regression slope)
        trend = self._compute_trend(entropies)

        # Compute entropy-variance correlation (key distress indicator)
        correlation = self._pearson_correlation(
            x=entropies,
            y=variances,
            x_mean=entropy_mean,
            y_mean=variance_mean,
            x_std_dev=entropy_std_dev,
            y_std_dev=variance_std_dev,
        )

        # Classify trend type
        trend_type = self._classify_trend(
            slope=trend,
            std_dev=entropy_std_dev,
            values=entropies,
        )

        # Check for anomalies
        anomaly_indices = self._detect_anomalies(
            values=entropies,
            mean=entropy_mean,
            std_dev=entropy_std_dev,
        )

        # Count sustained high entropy
        sustained_high_count = self._count_sustained_high(
            entropies=entropies,
            threshold=3.0,
        )

        return EntropyPattern(
            trend=trend_type,
            trend_slope=trend,
            volatility=entropy_std_dev,
            entropy_mean=entropy_mean,
            entropy_std_dev=entropy_std_dev,
            variance_mean=variance_mean,
            variance_std_dev=variance_std_dev,
            entropy_variance_correlation=correlation,
            sustained_high_count=sustained_high_count,
            peak_entropy=max(entropies) if entropies else 0.0,
            min_entropy=min(entropies) if entropies else 0.0,
            anomaly_indices=tuple(anomaly_indices),
            sample_count=len(samples),
        )

    def detect_distress(self, pattern: EntropyPattern) -> DistressDetectionResult | None:
        """
        Detect if the current pattern indicates distress.

        Distress signature:
        1. Sustained high entropy
        2. Low variance (flat distribution)
        3. Negative entropy-variance correlation

        Args:
            pattern: Analyzed entropy pattern

        Returns:
            DistressDetectionResult if detected, None otherwise
        """
        # Need minimum samples
        if pattern.sample_count < self.config.minimum_samples_for_trend:
            return None

        # Check for distress signature
        has_sustained_high = pattern.sustained_high_count >= 3
        has_low_variance = pattern.variance_mean < 0.3
        has_negative_correlation = (
            pattern.entropy_variance_correlation < self.config.distress_correlation_threshold
        )

        # Require at least sustained high + one other indicator
        if not has_sustained_high:
            return None
        if not (has_low_variance or has_negative_correlation):
            return None

        # Calculate confidence based on how many indicators are present
        confidence = 0.5  # Base confidence for sustained high
        if has_low_variance:
            confidence += 0.25
        if has_negative_correlation:
            confidence += 0.25

        indicators: list[str] = []
        if has_sustained_high:
            indicators.append("sustained_high_entropy")
        if has_low_variance:
            indicators.append("low_variance")
        if has_negative_correlation:
            indicators.append("negative_correlation")

        return DistressDetectionResult(
            confidence=confidence,
            sustained_high_count=pattern.sustained_high_count,
            average_entropy=pattern.entropy_mean,
            average_variance=pattern.variance_mean,
            correlation=pattern.entropy_variance_correlation,
            indicators=tuple(indicators),
            recommended_action=self._recommend_action(confidence),
        )

    def _compute_trend(self, values: list[float]) -> float:
        """Compute linear regression slope (trend)."""
        if len(values) < self.config.minimum_samples_for_trend:
            return 0.0

        n = float(len(values))
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        for i, y in enumerate(values):
            x = float(i)
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _pearson_correlation(
        self,
        x: list[float],
        y: list[float],
        x_mean: float,
        y_mean: float,
        x_std_dev: float,
        y_std_dev: float,
    ) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        if x_std_dev < 1e-10 or y_std_dev < 1e-10:
            return 0.0

        sum_product = 0.0
        for i in range(len(x)):
            sum_product += (x[i] - x_mean) * (y[i] - y_mean)

        return sum_product / (float(len(x) - 1) * x_std_dev * y_std_dev)

    def _classify_trend(
        self,
        slope: float,
        std_dev: float,
        values: list[float],
    ) -> EntropyTrend:
        """Classify the entropy trend based on slope and variance."""
        if len(values) < self.config.minimum_samples_for_trend:
            return EntropyTrend.INSUFFICIENT

        # Check for spike (high stdDev relative to values)
        if std_dev > self.config.high_volatility_threshold:
            # Check if it's a transient spike vs sustained
            last_few = values[-3:] if len(values) >= 3 else values
            first_few = values[:3] if len(values) >= 3 else values
            last_mean = _Statistics.mean(last_few)
            first_mean = _Statistics.mean(first_few)

            # If start and end are similar but we had high variance = spike
            if abs(last_mean - first_mean) < self.config.trend_threshold:
                return EntropyTrend.SPIKING

        # Classify by slope
        if slope > self.config.trend_threshold:
            return EntropyTrend.RISING
        elif slope < -self.config.trend_threshold:
            return EntropyTrend.FALLING
        else:
            return EntropyTrend.STABLE

    def _detect_anomalies(
        self,
        values: list[float],
        mean: float,
        std_dev: float,
    ) -> list[int]:
        """Detect anomalies based on Z-score."""
        if std_dev < 1e-10:
            return []

        anomalies: list[int] = []
        for i, value in enumerate(values):
            z_score = abs(value - mean) / std_dev
            if z_score > self.config.anomaly_z_score_threshold:
                anomalies.append(i)
        return anomalies

    def _count_sustained_high(
        self,
        entropies: list[float],
        threshold: float,
    ) -> int:
        """Count the maximum consecutive high entropy samples."""
        max_consecutive = 0
        current = 0

        for e in entropies:
            if e >= threshold:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        return max_consecutive

    def _recommend_action(self, confidence: float) -> DistressAction:
        """Recommend action based on distress confidence."""
        if confidence >= 0.9:
            return DistressAction.HALT
        elif confidence >= 0.75:
            return DistressAction.PAUSE_AND_STEER
        else:
            return DistressAction.MONITOR
