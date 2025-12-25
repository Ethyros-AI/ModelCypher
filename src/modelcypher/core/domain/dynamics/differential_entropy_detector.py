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
Differential Entropy Detector for Unsafe Prompt Pattern Detection.

Ported 1:1 from the reference Swift implementation.

Two-pass entropy differential detector that detects unsafe prompt patterns
by measuring entropy changes under intensity modifiers.

Research Basis:
- Based on Phase 7 Linguistic Thermodynamics research (2025-12)
- Unsafe prompts consistently show entropy COOLING under intensity modifiers
- Detection rule: ΔH(caps) < -0.1 achieves 100% recall, 0.89 F1
- Benign prompts show mixed/heating patterns (71% heat, 29% cool)

Detection Algorithm:
1. Measure H(baseline) - entropy with unmodified prompt
2. Measure H(intensity) - entropy with CAPS modifier
3. Compute ΔH = H(intensity) - H(baseline)
4. If ΔH < threshold → unsafe pattern detected
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable

# =============================================================================
# LinguisticModifier
# =============================================================================


class LinguisticModifier(str, Enum):
    """Linguistic modifiers for prompt perturbation."""

    baseline = "baseline"
    caps = "caps"
    emphasis = "emphasis"
    hedging = "hedging"
    urgency = "urgency"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DifferentialEntropyConfig:
    """Configuration for differential entropy detection.

    Thresholds must be explicitly provided or derived from calibration data.
    No arbitrary defaults.
    """

    delta_h_threshold: float
    """Threshold for unsafe pattern detection. ΔH below this is unsafe."""

    minimum_baseline_entropy: float
    """Minimum baseline entropy to consider valid measurement."""

    # Which modifier to compare against baseline.
    comparison_modifier: LinguisticModifier = LinguisticModifier.caps

    # Maximum tokens to generate for measurement.
    max_tokens: int = 30

    # Temperature for generation (0.0 = greedy for consistency).
    temperature: float = 0.7

    # Top-K for entropy calculation.
    top_k: int = 10

    @classmethod
    def from_calibration_results(
        cls,
        unsafe_delta_h_samples: list[float],
        benign_delta_h_samples: list[float],
        baseline_entropies: list[float],
        target_recall: float = 0.95,
    ) -> "DifferentialEntropyConfig":
        """Derive thresholds from calibration data.

        Args:
            unsafe_delta_h_samples: Delta-H values from known unsafe prompts
            benign_delta_h_samples: Delta-H values from known benign prompts
            baseline_entropies: Baseline entropy values from calibration
            target_recall: Target recall rate for unsafe detection (default 95%)

        Returns:
            Configuration with thresholds derived from calibration data.
        """
        if not unsafe_delta_h_samples or not benign_delta_h_samples:
            raise ValueError("Both unsafe and benign samples required for calibration")

        # Sort unsafe samples to find threshold at target recall
        sorted_unsafe = sorted(unsafe_delta_h_samples)
        # Threshold where target_recall of unsafe prompts are below threshold
        recall_idx = int(len(sorted_unsafe) * target_recall)
        recall_idx = min(recall_idx, len(sorted_unsafe) - 1)
        delta_h_threshold = sorted_unsafe[recall_idx]

        # Minimum baseline entropy from calibration data
        sorted_baseline = sorted(baseline_entropies)
        # Use 1st percentile as minimum
        min_idx = max(0, len(sorted_baseline) // 100)
        minimum_baseline = sorted_baseline[min_idx] if sorted_baseline else 0.01

        return cls(
            delta_h_threshold=delta_h_threshold,
            minimum_baseline_entropy=minimum_baseline,
        )


# =============================================================================
# Detection Result
# =============================================================================


@dataclass(frozen=True)
class DetectionResult:
    """Result from differential entropy detection.

    Contains raw measurements only. Caller determines classification
    via is_unsafe_for_threshold() using their calibrated thresholds.
    """

    # Mean entropy from baseline measurement.
    baseline_entropy: float

    # Mean entropy from intensity (CAPS) measurement.
    intensity_entropy: float

    # Entropy delta: H(intensity) - H(baseline).
    # Negative = entropy cooling (potential unsafe pattern).
    # Positive = entropy heating (benign pattern).
    delta_h: float

    # Timestamp of detection.
    timestamp: datetime

    # Processing time in seconds.
    processing_time: float

    # Token count from baseline measurement.
    baseline_token_count: int

    # Token count from intensity measurement.
    intensity_token_count: int

    @property
    def is_cooling(self) -> bool:
        """Whether entropy decreased (cooling pattern)."""
        return self.delta_h < 0

    @property
    def is_heating(self) -> bool:
        """Whether entropy increased (heating pattern)."""
        return self.delta_h > 0

    def is_unsafe_for_threshold(
        self,
        delta_h_threshold: float,
        minimum_baseline_entropy: float,
    ) -> bool:
        """Check if result indicates unsafe pattern for given thresholds.

        Args:
            delta_h_threshold: Delta-H below which pattern is unsafe (typically negative).
            minimum_baseline_entropy: Minimum baseline entropy for valid measurement.

        Returns:
            True if delta_h <= threshold AND baseline_entropy >= minimum.
        """
        if self.baseline_entropy < minimum_baseline_entropy:
            return False  # Indeterminate - can't make determination
        return self.delta_h <= delta_h_threshold

    def is_valid_measurement(self, minimum_baseline_entropy: float) -> bool:
        """Check if baseline entropy is sufficient for valid measurement."""
        return self.baseline_entropy >= minimum_baseline_entropy

    def threshold_ratio(self, delta_h_threshold: float) -> float:
        """Ratio of delta_h to threshold - measures distance from decision boundary.

        Returns:
            |delta_h| / |threshold|. Values > 1.0 mean past threshold.
        """
        if abs(delta_h_threshold) < 1e-10:
            return 0.0
        return abs(self.delta_h) / abs(delta_h_threshold)


# =============================================================================
# Batch Statistics
# =============================================================================


@dataclass(frozen=True)
class BatchDetectionStatistics:
    """Aggregate statistics for batch detection.

    Raw statistics only. Caller computes counts using their thresholds.
    """

    total: int
    cooling_count: int
    """Count of results with delta_h < 0 (entropy cooling)."""

    heating_count: int
    """Count of results with delta_h > 0 (entropy heating)."""

    mean_delta_h: float
    std_delta_h: float
    min_delta_h: float
    max_delta_h: float
    total_processing_time: float

    @property
    def cooling_rate(self) -> float:
        """Rate of cooling patterns (delta_h < 0)."""
        return self.cooling_count / self.total if self.total > 0 else 0.0

    @property
    def heating_rate(self) -> float:
        """Rate of heating patterns (delta_h > 0)."""
        return self.heating_count / self.total if self.total > 0 else 0.0

    def unsafe_count_for_threshold(
        self,
        results: list[DetectionResult],
        delta_h_threshold: float,
        minimum_baseline_entropy: float,
    ) -> int:
        """Count results that would be classified as unsafe for given thresholds."""
        return sum(
            1
            for r in results
            if r.is_unsafe_for_threshold(delta_h_threshold, minimum_baseline_entropy)
        )

    @staticmethod
    def compute(results: list[DetectionResult]) -> "BatchDetectionStatistics":
        """Compute aggregate statistics from detection results."""
        import math

        total = len(results)
        if total == 0:
            return BatchDetectionStatistics(
                total=0,
                cooling_count=0,
                heating_count=0,
                mean_delta_h=0.0,
                std_delta_h=0.0,
                min_delta_h=0.0,
                max_delta_h=0.0,
                total_processing_time=0.0,
            )

        cooling_count = sum(1 for r in results if r.is_cooling)
        heating_count = sum(1 for r in results if r.is_heating)

        delta_h_values = [r.delta_h for r in results]
        mean_delta_h = sum(delta_h_values) / total
        variance = sum((d - mean_delta_h) ** 2 for d in delta_h_values) / total
        std_delta_h = math.sqrt(variance)
        min_delta_h = min(delta_h_values)
        max_delta_h = max(delta_h_values)
        total_processing_time = sum(r.processing_time for r in results)

        return BatchDetectionStatistics(
            total=total,
            cooling_count=cooling_count,
            heating_count=heating_count,
            mean_delta_h=mean_delta_h,
            std_delta_h=std_delta_h,
            min_delta_h=min_delta_h,
            max_delta_h=max_delta_h,
            total_processing_time=total_processing_time,
        )


# =============================================================================
# Variant Measurement
# =============================================================================


@dataclass
class VariantMeasurement:
    """Measurement result from a single prompt variant."""

    mean_entropy: float
    token_count: int
    entropies: list[float] = field(default_factory=list)


# =============================================================================
# Differential Entropy Detector
# =============================================================================


class DifferentialEntropyDetector:
    """
    Two-pass entropy differential detector for unsafe prompt pattern detection.

    Based on Phase 7 Linguistic Thermodynamics research:
    - Unsafe prompts consistently show entropy COOLING under intensity modifiers
    - Detection rule: ΔH(caps) < threshold achieves high recall

    Returns raw measurements. Caller uses is_unsafe_for_threshold() with
    calibrated thresholds to make classification decisions.

    Usage:
        config = DifferentialEntropyConfig.from_calibration_results(...)
        detector = DifferentialEntropyDetector(config)
        result = await detector.detect(prompt, measure_fn)
        if result.is_unsafe_for_threshold(config.delta_h_threshold, config.minimum_baseline_entropy):
            # Handle unsafe pattern
    """

    def __init__(self, config: DifferentialEntropyConfig):
        """Initialize detector with explicit configuration.

        Args:
            config: Detection thresholds. Use from_calibration_results() to
                derive from labeled calibration data.
        """
        self.config = config

    async def detect(
        self,
        prompt: str,
        measure_fn: Callable[[str], Awaitable[VariantMeasurement]],
    ) -> DetectionResult:
        """
        Measure differential entropy for a prompt.

        Performs two-pass measurement:
        1. Generate with baseline prompt → capture H(baseline)
        2. Generate with intensity modifier → capture H(intensity)
        3. Compute ΔH = H(intensity) - H(baseline)

        Args:
            prompt: The prompt to analyze.
            measure_fn: Async function that measures entropy for a prompt variant.

        Returns:
            Detection result with raw measurements.
            Use is_unsafe_for_threshold() to check against calibrated thresholds.
        """
        start_time = time.perf_counter()

        # Create prompt variants
        baseline_prompt = prompt
        intensity_prompt = self._apply_modifier(prompt, self.config.comparison_modifier)

        # Measure baseline
        baseline_measurement = await measure_fn(baseline_prompt)

        # Measure intensity variant
        intensity_measurement = await measure_fn(intensity_prompt)

        processing_time = time.perf_counter() - start_time

        # Compute delta
        delta_h = intensity_measurement.mean_entropy - baseline_measurement.mean_entropy

        return DetectionResult(
            baseline_entropy=baseline_measurement.mean_entropy,
            intensity_entropy=intensity_measurement.mean_entropy,
            delta_h=delta_h,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            baseline_token_count=baseline_measurement.token_count,
            intensity_token_count=intensity_measurement.token_count,
        )

    async def detect_batch(
        self,
        prompts: list[str],
        measure_fn: Callable[[str], Awaitable[VariantMeasurement]],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> list[DetectionResult]:
        """
        Batch detection for multiple prompts.

        Args:
            prompts: Array of prompts to analyze.
            measure_fn: Async function that measures entropy for a prompt variant.
            progress_fn: Optional progress callback (index, total).

        Returns:
            Array of detection results.
        """
        results: list[DetectionResult] = []

        for index, prompt in enumerate(prompts):
            result = await self.detect(prompt=prompt, measure_fn=measure_fn)
            results.append(result)

            if progress_fn:
                progress_fn(index + 1, len(prompts))

        return results

    def detect_from_measurements(
        self,
        baseline_entropy: float,
        baseline_token_count: int,
        intensity_entropy: float,
        intensity_token_count: int,
    ) -> DetectionResult:
        """
        Create detection result from pre-computed measurements.

        Useful when entropy has already been measured externally.

        Args:
            baseline_entropy: Mean entropy from baseline.
            baseline_token_count: Token count from baseline.
            intensity_entropy: Mean entropy from intensity measurement.
            intensity_token_count: Token count from intensity measurement.

        Returns:
            Detection result with raw measurements.
        """
        delta_h = intensity_entropy - baseline_entropy

        return DetectionResult(
            baseline_entropy=baseline_entropy,
            intensity_entropy=intensity_entropy,
            delta_h=delta_h,
            timestamp=datetime.utcnow(),
            processing_time=0.0,
            baseline_token_count=baseline_token_count,
            intensity_token_count=intensity_token_count,
        )

    def _apply_modifier(self, prompt: str, modifier: LinguisticModifier) -> str:
        """Apply a linguistic modifier to a prompt."""
        if modifier == LinguisticModifier.baseline:
            return prompt
        elif modifier == LinguisticModifier.caps:
            return prompt.upper()
        elif modifier == LinguisticModifier.emphasis:
            return f"IMPORTANT: {prompt}"
        elif modifier == LinguisticModifier.hedging:
            return f"Perhaps, maybe, {prompt.lower()}"
        elif modifier == LinguisticModifier.urgency:
            return f"URGENT! {prompt} NOW!"
        else:
            return prompt

