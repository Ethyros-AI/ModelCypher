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

Analyzes entropy time series to compute raw statistical measurements.
No threshold-based classification is performed.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    sqrt_scalar,
)


@dataclass(frozen=True)
class EntropyPattern:
    """Complete entropy pattern analysis result.

    Raw geometric measurements only.
    """

    trend_slope: float
    volatility: float
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
    def is_rising(self) -> bool:
        """Whether trend_slope indicates rising entropy."""
        return self.trend_slope > 0

    @property
    def is_falling(self) -> bool:
        """Whether trend_slope indicates falling entropy."""
        return self.trend_slope < 0

    @property
    def sustained_significance(self) -> float:
        """Ratio of sustained high count to sqrt(sample_count)."""
        if self.sample_count < 1:
            return 0.0
        _b = get_default_backend()
        return self.sustained_high_count / sqrt_scalar(float(self.sample_count), _b)

    @staticmethod
    def empty() -> "EntropyPattern":
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
    """Raw distress-related measurements."""

    sustained_high_count: int
    sustained_significance: float
    entropy_mean: float
    variance_mean: float
    entropy_variance_correlation: float
    sample_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sustainedHighCount": self.sustained_high_count,
            "sustainedSignificance": self.sustained_significance,
            "entropyMean": self.entropy_mean,
            "varianceMean": self.variance_mean,
            "entropyVarianceCorrelation": self.entropy_variance_correlation,
            "sampleCount": self.sample_count,
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
        _b = get_default_backend()
        return sqrt_scalar(variance, _b)


class EntropyPatternAnalyzer:
    """Analyze entropy sequences and return raw measurements."""

    def analyze(self, samples: list[tuple[float, float]]) -> EntropyPattern:
        """Analyze a sequence of entropy/variance samples."""
        if not samples:
            return EntropyPattern.empty()

        entropies = [s[0] for s in samples]
        variances = [s[1] for s in samples]

        entropy_mean = _Statistics.mean(entropies)
        entropy_std_dev = _Statistics.standard_deviation(entropies, entropy_mean)
        variance_mean = _Statistics.mean(variances)
        variance_std_dev = _Statistics.standard_deviation(variances, variance_mean)

        trend = self._compute_trend(entropies)
        correlation = self._pearson_correlation(
            x=entropies,
            y=variances,
            x_mean=entropy_mean,
            y_mean=variance_mean,
            x_std_dev=entropy_std_dev,
            y_std_dev=variance_std_dev,
        )

        anomaly_indices = self._detect_anomalies(
            values=entropies,
            mean=entropy_mean,
            std_dev=entropy_std_dev,
        )

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        high_threshold = entropy_mean + entropy_std_dev if entropy_std_dev > eps else entropy_mean
        sustained_high_count = self._count_sustained_high(entropies, high_threshold)

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
        """Return distress-related measurements from the pattern."""
        if pattern.sample_count == 0:
            return None
        return DistressDetectionResult(
            sustained_high_count=pattern.sustained_high_count,
            sustained_significance=pattern.sustained_significance,
            entropy_mean=pattern.entropy_mean,
            variance_mean=pattern.variance_mean,
            entropy_variance_correlation=pattern.entropy_variance_correlation,
            sample_count=pattern.sample_count,
        )

    def _compute_trend(self, values: list[float]) -> float:
        """Compute linear regression slope (trend)."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        return numerator / denominator if denominator != 0 else 0.0

    def _count_sustained_high(self, entropies: list[float], threshold: float) -> int:
        """Count consecutive samples above the threshold."""
        max_count = 0
        current_count = 0

        for entropy in entropies:
            if entropy > threshold:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _detect_anomalies(self, values: list[float], mean: float, std_dev: float) -> list[int]:
        """Detect anomalies using data-derived z-score separation."""
        if len(values) < 2:
            return []

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        if std_dev <= eps:
            return []

        z_scores = [abs((value - mean) / std_dev) for value in values]
        threshold = find_magnitude_gap_threshold(sorted(z_scores), eps=eps)
        if threshold <= 0.0:
            return []

        return [i for i, z in enumerate(z_scores) if z >= threshold]

    def _pearson_correlation(
        self,
        x: list[float],
        y: list[float],
        x_mean: float,
        y_mean: float,
        x_std_dev: float,
        y_std_dev: float,
    ) -> float:
        """Compute Pearson correlation coefficient between two lists."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        if x_std_dev == 0 or y_std_dev == 0:
            return 0.0

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = (len(x) - 1) * x_std_dev * y_std_dev

        if denominator == 0:
            return 0.0

        return numerator / denominator


__all__ = [
    "DistressDetectionResult",
    "EntropyPattern",
    "EntropyPatternAnalyzer",
]
