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


class DistressAction(str, Enum):
    """Recommended action when distress is detected."""

    MONITOR = "monitor"  # Continue monitoring, distress signature is weak
    PAUSE_AND_STEER = "pause_and_steer"  # Pause generation and attempt to steer conversation
    HALT = "halt"  # Halt generation immediately (circuit breaker)


@dataclass(frozen=True)
class DetectorConfiguration:
    """Configuration for entropy pattern detection.

    Thresholds must be derived from baseline entropy measurements.
    """

    minimum_samples_for_trend: int
    trend_threshold: float
    distress_correlation_threshold: float
    high_volatility_threshold: float
    anomaly_z_score_threshold: float

    @classmethod
    def from_baseline_entropy(
        cls,
        entropy_samples: list[float],
        *,
        minimum_samples: int = 5,
    ) -> "DetectorConfiguration":
        """Derive thresholds from baseline entropy measurements."""
        if len(entropy_samples) < minimum_samples:
            raise ValueError(
                f"Need at least {minimum_samples} entropy samples for calibration"
            )

        import statistics

        mean = statistics.mean(entropy_samples)
        std = statistics.stdev(entropy_samples) if len(entropy_samples) > 1 else 0.01

        # Trend threshold: based on variance in the data
        trend_threshold = std * 0.5

        # Volatility threshold: above mean volatility is "high"
        high_volatility_threshold = std

        # Distress correlation: negative correlation threshold
        # Based on expected correlation strength
        distress_correlation_threshold = -0.3 * (std / max(mean, 0.01))

        # Z-score threshold: 2.5 std devs is statistically anomalous
        anomaly_z_score_threshold = 2.5

        return cls(
            minimum_samples_for_trend=minimum_samples,
            trend_threshold=trend_threshold,
            distress_correlation_threshold=distress_correlation_threshold,
            high_volatility_threshold=high_volatility_threshold,
            anomaly_z_score_threshold=anomaly_z_score_threshold,
        )


@dataclass(frozen=True)
class EntropyPattern:
    """Complete entropy pattern analysis result.

    Raw geometric measurements - no categorical classifications.
    """

    trend_slope: float  # Linear regression slope of entropy over time
    volatility: float  # Standard deviation of entropy
    entropy_mean: float
    entropy_std_dev: float
    variance_mean: float
    variance_std_dev: float
    entropy_variance_correlation: float
    sustained_high_count: int  # Consecutive samples above mean + std
    peak_entropy: float
    min_entropy: float
    anomaly_indices: tuple[int, ...]  # Indices with z-score > threshold
    sample_count: int

    @property
    def is_rising(self) -> bool:
        """Whether trend_slope indicates rising entropy."""
        return self.trend_slope > 0

    @property
    def is_falling(self) -> bool:
        """Whether trend_slope indicates falling entropy."""
        return self.trend_slope < 0

    @property
    def sustained_significance(self) -> float:
        """How significant the sustained high count is.

        Returns ratio of sustained_high_count to sqrt(sample_count).
        Values > 1.0 indicate statistically unlikely by chance.
        """
        if self.sample_count < 1:
            return 0.0
        threshold = max(2.0, math.sqrt(self.sample_count))
        return self.sustained_high_count / threshold

    @staticmethod
    def empty() -> EntropyPattern:
        """Empty pattern for when no samples are available."""
        return EntropyPattern(
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
            "trendSlope": self.trend_slope,
            "volatility": self.volatility,
            "entropyMean": self.entropy_mean,
            "entropyStdDev": self.entropy_std_dev,
            "varianceMean": self.variance_mean,
            "varianceStdDev": self.variance_std_dev,
            "entropyVarianceCorrelation": self.entropy_variance_correlation,
            "sustainedHighCount": self.sustained_high_count,
            "sustainedSignificance": self.sustained_significance,
            "peakEntropy": self.peak_entropy,
            "minEntropy": self.min_entropy,
            "anomalyIndices": list(self.anomaly_indices),
            "sampleCount": self.sample_count,
        }


@dataclass(frozen=True)
class DistressDetectionResult:
    """Result of distress detection analysis.

    Raw confidence and indicator measurements.
    Caller decides action based on their risk tolerance.
    """

    confidence: float  # 0.0-1.0
    sustained_high_count: int
    average_entropy: float
    average_variance: float
    correlation: float
    indicators: tuple[str, ...]

    def action_for_thresholds(
        self,
        halt_threshold: float = 0.8,
        pause_threshold: float = 0.5,
    ) -> DistressAction:
        """Map confidence to action using caller-provided thresholds.

        Args:
            halt_threshold: Confidence >= this triggers HALT
            pause_threshold: Confidence >= this triggers PAUSE_AND_STEER

        Returns:
            Recommended action based on thresholds
        """
        if self.confidence >= halt_threshold:
            return DistressAction.HALT
        elif self.confidence >= pause_threshold:
            return DistressAction.PAUSE_AND_STEER
        else:
            return DistressAction.MONITOR

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "sustainedHighCount": self.sustained_high_count,
            "averageEntropy": self.average_entropy,
            "averageVariance": self.average_variance,
            "correlation": self.correlation,
            "indicators": list(self.indicators),
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

    def __init__(self, config: DetectorConfiguration):
        """Initialize with calibrated configuration."""
        self.config = config

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

        # Check for anomalies
        anomaly_indices = self._detect_anomalies(
            values=entropies,
            mean=entropy_mean,
            std_dev=entropy_std_dev,
        )

        # Count sustained high entropy
        # "High" is defined as mean + 1 std dev (statistically elevated)
        high_threshold = entropy_mean + entropy_std_dev if entropy_std_dev > 1e-10 else entropy_mean
        sustained_high_count = self._count_sustained_high(
            entropies=entropies,
            threshold=high_threshold,
        )

        return EntropyPattern(
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

        # Check for distress signature using data-derived thresholds
        # Sustained high: significant if count > sqrt(samples)
        sustained_threshold = max(2, int(math.sqrt(pattern.sample_count)))
        has_sustained_high = pattern.sustained_high_count >= sustained_threshold

        # Low variance indicates flat distribution (model equally uncertain about all options)
        # Compare variance to entropy: if variance << entropy, distribution is uniform-like
        # Geometric relationship: for uniform distribution, variance/entropy approaches a constant
        has_low_variance = pattern.variance_mean < pattern.entropy_mean * 0.5 if pattern.entropy_mean > 1e-10 else False

        has_negative_correlation = (
            pattern.entropy_variance_correlation < self.config.distress_correlation_threshold
        )

        # Require at least sustained high + one other indicator
        if not has_sustained_high:
            return None
        if not (has_low_variance or has_negative_correlation):
            return None

        # Confidence derived from indicator strengths
        # Sustained: ratio of count to threshold (capped at 1)
        sustained_strength = min(1.0, pattern.sustained_high_count / sustained_threshold)
        # Correlation: how far below threshold (scaled to [0, 0.5])
        correlation_strength = (
            min(1.0, abs(pattern.entropy_variance_correlation - self.config.distress_correlation_threshold) / 0.5)
            if has_negative_correlation else 0.0
        )
        # Base confidence from indicator presence, weighted by strength
        num_indicators = sum([has_sustained_high, has_low_variance, has_negative_correlation])
        confidence = (sustained_strength + correlation_strength + (1.0 if has_low_variance else 0.0)) / num_indicators

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
