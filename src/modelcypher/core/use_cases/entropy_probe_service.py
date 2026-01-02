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
Entropy Probe Service.

Exposes entropy probe operations as CLI/MCP-consumable operations.
Provides pattern analysis and baseline verification for entropy monitoring.
"""

from __future__ import annotations

from datetime import datetime

from modelcypher.core.domain.entropy.baseline_verification_probe import (
    BaselineComparison,
    EntropyBaseline,
    VerificationResult,
)
from modelcypher.core.domain.entropy.entropy_pattern_detector import (
    DistressDetectionResult,
    EntropyPattern,
    EntropyPatternAnalyzer,
)


class EntropyProbeService:
    """
    Service for entropy probe operations.

    Provides pattern analysis and baseline verification for CLI/MCP consumption.
    """

    def __init__(self) -> None:
        """Initialize the service."""
        pass

    def analyze_pattern(
        self,
        samples: list[tuple[float, float]],
    ) -> EntropyPattern:
        """
        Analyze entropy/variance samples for patterns.

        Args:
            samples: List of (entropy, variance) tuples in chronological order
        Returns:
            EntropyPattern with trend, statistics, and anomaly information
        """
        analyzer = EntropyPatternAnalyzer()
        return analyzer.analyze(samples)

    def detect_distress(
        self,
        samples: list[tuple[float, float]],
    ) -> DistressDetectionResult | None:
        """
        Detect distress patterns in entropy/variance samples.

        Args:
            samples: List of (entropy, variance) tuples
        Returns:
            DistressDetectionResult if distress detected, None otherwise
        """
        analyzer = EntropyPatternAnalyzer()
        pattern = analyzer.analyze(samples)
        return analyzer.detect_distress(pattern)

    def verify_baseline(
        self,
        declared_mean: float,
        declared_std_dev: float,
        declared_max: float,
        declared_min: float,
        observed_deltas: list[float],
        *,
        base_model_id: str,
        adapter_path: str,
    ) -> VerificationResult:
        """
        Verify observed entropy deltas against declared baseline.

        Args:
            declared_mean: Declared delta mean from manifest
            declared_std_dev: Declared delta standard deviation
            declared_max: Declared maximum delta
            declared_min: Declared minimum delta
            observed_deltas: List of observed delta values
            base_model_id: Base model identifier
            adapter_path: Path to adapter (for reporting)
        Returns:
            VerificationResult with comparison metrics and statistics
        """
        declared_baseline = EntropyBaseline(
            delta_mean=declared_mean,
            delta_std_dev=declared_std_dev,
            delta_max=declared_max,
            delta_min=declared_min,
            base_model_id=base_model_id,
            sample_count=0,  # Declared baselines don't track sample count
        )

        if not observed_deltas:
            observed_baseline = EntropyBaseline(
                delta_mean=0.0,
                delta_std_dev=0.0,
                delta_max=0.0,
                delta_min=0.0,
                base_model_id=base_model_id,
                sample_count=0,
            )
        else:
            mean = sum(observed_deltas) / len(observed_deltas)
            if len(observed_deltas) > 1:
                variance = sum((d - mean) ** 2 for d in observed_deltas) / (
                    len(observed_deltas) - 1
                )
                std_dev = variance**0.5
            else:
                std_dev = 0.0
            observed_baseline = EntropyBaseline(
                delta_mean=mean,
                delta_std_dev=std_dev,
                delta_max=max(observed_deltas),
                delta_min=min(observed_deltas),
                base_model_id=base_model_id,
                sample_count=len(observed_deltas),
            )

        comparison = BaselineComparison.from_baselines(
            observed=observed_baseline,
            declared=declared_baseline,
        )

        return VerificationResult(
            adapter_path=adapter_path,
            base_model_path=base_model_id,
            declared_baseline=declared_baseline,
            observed_baseline=observed_baseline,
            comparison=comparison,
            prompt_results=(),
            total_samples=len(observed_deltas),
            verification_duration=0.0,
            timestamp=datetime.now(),
        )

    @staticmethod
    def pattern_payload(pattern: EntropyPattern) -> dict:
        """Convert pattern to CLI/MCP payload."""
        return {
            "trendSlope": pattern.trend_slope,
            "isRising": pattern.is_rising,
            "isFalling": pattern.is_falling,
            "volatility": pattern.volatility,
            "entropyMean": pattern.entropy_mean,
            "entropyStdDev": pattern.entropy_std_dev,
            "varianceMean": pattern.variance_mean,
            "varianceStdDev": pattern.variance_std_dev,
            "entropyVarianceCorrelation": pattern.entropy_variance_correlation,
            "sustainedHighCount": pattern.sustained_high_count,
            "sustainedSignificance": pattern.sustained_significance,
            "peakEntropy": pattern.peak_entropy,
            "minEntropy": pattern.min_entropy,
            "anomalyIndices": list(pattern.anomaly_indices),
            "sampleCount": pattern.sample_count,
        }

    @staticmethod
    def distress_payload(distress: DistressDetectionResult | None) -> dict:
        """Convert distress metrics to CLI/MCP payload."""
        if distress is None:
            return {
                "sustainedHighCount": 0,
                "sustainedSignificance": 0.0,
                "entropyMean": 0.0,
                "varianceMean": 0.0,
                "entropyVarianceCorrelation": 0.0,
                "sampleCount": 0,
            }
        return {
            "sustainedHighCount": distress.sustained_high_count,
            "sustainedSignificance": distress.sustained_significance,
            "entropyMean": distress.entropy_mean,
            "varianceMean": distress.variance_mean,
            "entropyVarianceCorrelation": distress.entropy_variance_correlation,
            "sampleCount": distress.sample_count,
        }

    @staticmethod
    def verification_payload(result: VerificationResult) -> dict:
        """Convert verification result to CLI/MCP payload."""
        return {
            "adapterPath": result.adapter_path,
            "baseModelPath": result.base_model_path,
            "declaredBaseline": result.declared_baseline.to_dict(),
            "observedBaseline": result.observed_baseline.to_dict(),
            "comparison": result.comparison.to_dict(),
            "totalSamples": result.total_samples,
            "verificationDuration": result.verification_duration,
            "timestamp": result.timestamp.isoformat(),
        }
