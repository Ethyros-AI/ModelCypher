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

from dataclasses import dataclass

from modelcypher.core.domain.entropy.baseline_verification_probe import (
    BaselineVerificationProbe,
    DeltaSample,
    EntropyBaseline,
    VerificationConfiguration,
    VerificationResult,
    VerificationVerdict,
)
from modelcypher.core.domain.entropy.entropy_pattern_detector import (
    DetectorConfiguration,
    DistressAction,
    DistressDetectionResult,
    EntropyPattern,
    EntropyPatternAnalyzer,
)


@dataclass(frozen=True)
class PatternAnalysisConfig:
    """Configuration for pattern analysis operations."""

    minimum_samples_for_trend: int = 5
    trend_threshold: float = 0.05
    distress_correlation_threshold: float = -0.3
    high_volatility_threshold: float = 0.15
    anomaly_z_score_threshold: float = 2.5


class EntropyProbeService:
    """
    Service for entropy probe operations.

    Provides pattern analysis and baseline verification for CLI/MCP consumption.
    """

    def __init__(self) -> None:
        """Initialize the service."""
        self.pattern_analyzer = EntropyPatternAnalyzer()

    def analyze_pattern(
        self,
        samples: list[tuple[float, float]],
        config: PatternAnalysisConfig | None = None,
    ) -> EntropyPattern:
        """
        Analyze entropy/variance samples for patterns.

        Args:
            samples: List of (entropy, variance) tuples in chronological order
            config: Optional configuration overrides

        Returns:
            EntropyPattern with trend, statistics, and anomaly information
        """
        if config:
            detector_config = DetectorConfiguration(
                minimum_samples_for_trend=config.minimum_samples_for_trend,
                trend_threshold=config.trend_threshold,
                distress_correlation_threshold=config.distress_correlation_threshold,
                high_volatility_threshold=config.high_volatility_threshold,
                anomaly_z_score_threshold=config.anomaly_z_score_threshold,
            )
            analyzer = EntropyPatternAnalyzer(detector_config)
        else:
            analyzer = self.pattern_analyzer

        return analyzer.analyze(samples)

    def detect_distress(
        self,
        samples: list[tuple[float, float]],
        config: PatternAnalysisConfig | None = None,
    ) -> DistressDetectionResult | None:
        """
        Detect distress patterns in entropy/variance samples.

        Args:
            samples: List of (entropy, variance) tuples
            config: Optional configuration overrides

        Returns:
            DistressDetectionResult if distress detected, None otherwise
        """
        pattern = self.analyze_pattern(samples, config)
        if config:
            detector_config = DetectorConfiguration(
                minimum_samples_for_trend=config.minimum_samples_for_trend,
                trend_threshold=config.trend_threshold,
                distress_correlation_threshold=config.distress_correlation_threshold,
                high_volatility_threshold=config.high_volatility_threshold,
                anomaly_z_score_threshold=config.anomaly_z_score_threshold,
            )
            analyzer = EntropyPatternAnalyzer(detector_config)
        else:
            analyzer = self.pattern_analyzer

        return analyzer.detect_distress(pattern)

    def verify_baseline(
        self,
        declared_mean: float,
        declared_std_dev: float,
        declared_max: float,
        declared_min: float,
        observed_deltas: list[float],
        base_model_id: str = "unknown",
        adapter_path: str = "unknown",
        tier: str = "default",
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
            tier: Verification tier (quick, default, thorough)

        Returns:
            VerificationResult with verdict and statistics
        """
        declared_baseline = EntropyBaseline(
            delta_mean=declared_mean,
            delta_std_dev=declared_std_dev,
            delta_max=declared_max,
            delta_min=declared_min,
            base_model_id=base_model_id,
            sample_count=0,  # Declared baselines don't track sample count
        )

        observed_samples = [
            DeltaSample(token_index=i, delta=d, anomaly_score=0.0)
            for i, d in enumerate(observed_deltas)
        ]

        # Tier determines statistical stringency
        if tier == "quick":
            # Less strict: higher z-scores (less sensitive), fewer samples
            config = VerificationConfiguration.with_statistical_thresholds(
                failure_z_score=3.5,
                suspicious_z_score=2.5,
                minimum_sample_count=50,
            )
        elif tier == "thorough":
            # More strict: lower z-scores (more sensitive), include adversarial
            config = VerificationConfiguration.with_statistical_thresholds(
                failure_z_score=2.5,
                suspicious_z_score=1.5,
                include_adversarial=True,
                minimum_sample_count=200,
            )
        else:
            # Standard: 99.7% confidence for failure, 95% for suspicious
            config = VerificationConfiguration.with_statistical_thresholds(
                failure_z_score=3.0,
                suspicious_z_score=2.0,
                minimum_sample_count=100,
            )

        probe = BaselineVerificationProbe(config)
        return probe.quick_verify_sync(
            declared_baseline=declared_baseline,
            observed_samples=observed_samples,
            adapter_path=adapter_path,
            base_model_path=base_model_id,
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
        """Convert distress detection result to CLI/MCP payload.

        Raw confidence - caller uses action_for_thresholds() to decide action.
        """
        if distress is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "indicators": [],
            }
        return {
            "detected": True,
            "confidence": distress.confidence,
            "sustainedHighCount": distress.sustained_high_count,
            "averageEntropy": distress.average_entropy,
            "averageVariance": distress.average_variance,
            "correlation": distress.correlation,
            "indicators": list(distress.indicators),
        }

    @staticmethod
    def verification_payload(result: VerificationResult) -> dict:
        """Convert verification result to CLI/MCP payload."""
        return {
            "verdict": result.verdict.value,
            "adapterPath": result.adapter_path,
            "baseModelPath": result.base_model_path,
            "declaredBaseline": result.declared_baseline.to_dict(),
            "observedBaseline": result.observed_baseline.to_dict(),
            "comparison": result.comparison.to_dict(),
            "totalSamples": result.total_samples,
            "adversarialFlagCount": len(result.adversarial_flags),
            "verificationDuration": result.verification_duration,
            "timestamp": result.timestamp.isoformat(),
            "summary": result.summary,
            "status": (
                "verified"
                if result.verdict == VerificationVerdict.VERIFIED
                else "suspicious"
                if result.verdict == VerificationVerdict.SUSPICIOUS
                else "failed"
            ),
        }
