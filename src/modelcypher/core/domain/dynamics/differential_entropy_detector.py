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
Differential Entropy Detector for Cooling Pattern Measurement.

Ported 1:1 from the reference Swift implementation.

Two-pass entropy differential detector that measures entropy changes under
intensity modifiers. Based on Phase 7 Linguistic Thermodynamics research
(2025-12).

Cooling patterns show entropy decreases under intensity modifiers, while
heating patterns show increases. Detection operates by measuring H(baseline)
with unmodified prompt, H(intensity) with CAPS modifier, computing
ΔH = H(intensity) - H(baseline), and comparing ΔH to a calibrated threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)

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

    Attributes
    ----------
    delta_h_threshold : float
        Threshold for cooling pattern detection. ΔH at or below this is below threshold.
    minimum_baseline_entropy : float
        Minimum baseline entropy to consider valid measurement
    comparison_modifier : LinguisticModifier, optional
        Modifier to compare against baseline
    """

    delta_h_threshold: float
    minimum_baseline_entropy: float
    comparison_modifier: LinguisticModifier = LinguisticModifier.caps

    @classmethod
    def from_calibration_results(
        cls,
        cooling_delta_h_samples: list[float],
        reference_delta_h_samples: list[float],
        baseline_entropies: list[float],
        target_capture: float,
    ) -> "DifferentialEntropyConfig":
        """Derive thresholds from calibration data.

        Parameters
        ----------
        cooling_delta_h_samples : list[float]
            Delta-H values from cooling-pattern prompts
        reference_delta_h_samples : list[float]
            Delta-H values from reference prompts
        baseline_entropies : list[float]
            Baseline entropy values from calibration
        target_capture : float
            Target fraction of cooling samples captured by the threshold

        Returns
        -------
        DifferentialEntropyConfig
            Configuration with thresholds derived from calibration data
        """
        if not cooling_delta_h_samples or not reference_delta_h_samples:
            raise ValueError(
                "Both cooling and reference samples required for calibration"
            )
        if not baseline_entropies:
            raise ValueError("Baseline entropy samples required for calibration")

        # Sort cooling samples to find threshold at target capture fraction
        sorted_cooling = sorted(cooling_delta_h_samples)
        # Threshold where target_capture of cooling samples are below threshold
        capture_idx = int(len(sorted_cooling) * target_capture)
        capture_idx = min(capture_idx, len(sorted_cooling) - 1)
        delta_h_threshold = sorted_cooling[capture_idx]

        # Minimum baseline entropy from calibration data
        sorted_baseline = sorted(baseline_entropies)
        minimum_baseline = sorted_baseline[0]

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
    via is_below_delta_h_threshold() using their calibrated thresholds.

    Attributes
    ----------
    baseline_entropy : float
        Mean entropy from baseline measurement
    intensity_entropy : float
        Mean entropy from intensity (CAPS) measurement
    delta_h : float
        Entropy delta: H(intensity) - H(baseline). Negative = cooling,
        positive = heating
    timestamp : datetime
        Timestamp of detection
    processing_time : float
        Processing time in seconds
    baseline_token_count : int
        Token count from baseline measurement
    intensity_token_count : int
        Token count from intensity measurement
    """

    baseline_entropy: float
    intensity_entropy: float
    delta_h: float
    timestamp: datetime
    processing_time: float
    baseline_token_count: int
    intensity_token_count: int

    @property
    def is_cooling(self) -> bool:
        """Whether entropy decreased (cooling pattern).

        Returns
        -------
        bool
            True if delta_h < 0
        """
        return self.delta_h < 0

    @property
    def is_heating(self) -> bool:
        """Whether entropy increased (heating pattern).

        Returns
        -------
        bool
            True if delta_h > 0
        """
        return self.delta_h > 0

    def is_below_delta_h_threshold(
        self,
        delta_h_threshold: float,
        minimum_baseline_entropy: float,
    ) -> bool:
        """Check if result is below delta-h threshold for given thresholds.

        Parameters
        ----------
        delta_h_threshold : float
            Delta-H at or below which the pattern is below threshold
        minimum_baseline_entropy : float
            Minimum baseline entropy for valid measurement

        Returns
        -------
        bool
            True if delta_h <= threshold AND baseline_entropy >= minimum
        """
        if self.baseline_entropy < minimum_baseline_entropy:
            return False  # Indeterminate - can't make determination
        return self.delta_h <= delta_h_threshold

    def is_valid_measurement(self, minimum_baseline_entropy: float) -> bool:
        """Check if baseline entropy is sufficient for valid measurement.

        Parameters
        ----------
        minimum_baseline_entropy : float
            Minimum baseline entropy threshold

        Returns
        -------
        bool
            True if baseline_entropy >= minimum_baseline_entropy
        """
        return self.baseline_entropy >= minimum_baseline_entropy

    def threshold_ratio(self, delta_h_threshold: float) -> float:
        """Ratio of delta_h to threshold - distance from decision boundary.

        Parameters
        ----------
        delta_h_threshold : float
            Threshold value for comparison

        Returns
        -------
        float
            |delta_h| / |threshold|. Values > 1.0 mean past threshold.
        """
        _b = get_default_backend()
        if abs(delta_h_threshold) <= machine_epsilon(_b, _b.array([1.0])):
            return 0.0
        return abs(self.delta_h) / abs(delta_h_threshold)


# =============================================================================
# Batch Statistics
# =============================================================================


@dataclass(frozen=True)
class BatchDetectionStatistics:
    """Aggregate statistics for batch detection.

    Raw statistics only. Caller computes counts using their thresholds.

    Attributes
    ----------
    total : int
        Total number of detection results
    cooling_count : int
        Count of results with delta_h < 0 (entropy cooling)
    heating_count : int
        Count of results with delta_h > 0 (entropy heating)
    mean_delta_h : float
        Mean delta_h across all results
    std_delta_h : float
        Standard deviation of delta_h
    min_delta_h : float
        Minimum delta_h value
    max_delta_h : float
        Maximum delta_h value
    total_processing_time : float
        Total processing time for all results
    """

    total: int
    cooling_count: int
    heating_count: int
    mean_delta_h: float
    std_delta_h: float
    min_delta_h: float
    max_delta_h: float
    total_processing_time: float

    @property
    def cooling_rate(self) -> float:
        """Rate of cooling patterns (delta_h < 0).

        Returns
        -------
        float
            Fraction of results with cooling pattern
        """
        return self.cooling_count / self.total if self.total > 0 else 0.0

    @property
    def heating_rate(self) -> float:
        """Rate of heating patterns (delta_h > 0).

        Returns
        -------
        float
            Fraction of results with heating pattern
        """
        return self.heating_count / self.total if self.total > 0 else 0.0

    def count_below_delta_h_threshold(
        self,
        results: list[DetectionResult],
        delta_h_threshold: float,
        minimum_baseline_entropy: float,
    ) -> int:
        """Count results that are below delta-h threshold for given thresholds.

        Parameters
        ----------
        results : list[DetectionResult]
            Detection results to analyze
        delta_h_threshold : float
            Delta-H threshold for below-threshold classification
        minimum_baseline_entropy : float
            Minimum baseline entropy threshold

        Returns
        -------
        int
            Count of results below threshold
        """
        return sum(
            1
            for r in results
            if r.is_below_delta_h_threshold(
                delta_h_threshold, minimum_baseline_entropy
            )
        )

    @staticmethod
    def compute(results: list[DetectionResult]) -> "BatchDetectionStatistics":
        """Compute aggregate statistics from detection results.

        Parameters
        ----------
        results : list[DetectionResult]
            Detection results to aggregate

        Returns
        -------
        BatchDetectionStatistics
            Aggregate statistics
        """
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
        std_delta_h = sqrt_scalar(variance, get_default_backend())
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
    """Measurement result from a single prompt variant.

    Attributes
    ----------
    mean_entropy : float
        Mean entropy across tokens
    token_count : int
        Number of tokens generated
    entropies : list[float]
        Per-token entropy values
    """

    mean_entropy: float
    token_count: int
    entropies: list[float] = field(default_factory=list)


# =============================================================================
# Differential Entropy Detector
# =============================================================================


class DifferentialEntropyDetector:
    """
    Two-pass entropy differential detector for cooling pattern detection.

    Based on Phase 7 Linguistic Thermodynamics research. Cooling patterns show
    entropy decreases under intensity modifiers. Returns raw measurements; caller
    uses is_below_delta_h_threshold() with calibrated thresholds for classification.

    Attributes
    ----------
    config : DifferentialEntropyConfig
        Detection configuration with thresholds
    """

    def __init__(self, config: DifferentialEntropyConfig):
        """Initialize detector with explicit configuration.

        Parameters
        ----------
        config : DifferentialEntropyConfig
            Detection thresholds. Use from_calibration_results() to
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

        Performs two-pass measurement: baseline prompt, intensity modifier,
        then computes ΔH = H(intensity) - H(baseline).

        Parameters
        ----------
        prompt : str
            The prompt to analyze
        measure_fn : Callable[[str], Awaitable[VariantMeasurement]]
            Async function that measures entropy for a prompt variant

        Returns
        -------
        DetectionResult
            Detection result with raw measurements. Use
            is_below_delta_h_threshold() to check against calibrated thresholds.
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

        Parameters
        ----------
        prompts : list[str]
            Array of prompts to analyze
        measure_fn : Callable[[str], Awaitable[VariantMeasurement]]
            Async function that measures entropy for a prompt variant
        progress_fn : Callable[[int, int], None] | None, optional
            Optional progress callback (index, total)

        Returns
        -------
        list[DetectionResult]
            Array of detection results
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

        Parameters
        ----------
        baseline_entropy : float
            Mean entropy from baseline
        baseline_token_count : int
            Token count from baseline
        intensity_entropy : float
            Mean entropy from intensity measurement
        intensity_token_count : int
            Token count from intensity measurement

        Returns
        -------
        DetectionResult
            Detection result with raw measurements
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
        """Apply a linguistic modifier to a prompt.

        Parameters
        ----------
        prompt : str
            Original prompt
        modifier : LinguisticModifier
            Modifier to apply

        Returns
        -------
        str
            Modified prompt
        """
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
