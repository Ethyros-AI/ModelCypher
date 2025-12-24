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
from enum import Enum
from typing import Callable, Awaitable


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
    """Configuration for differential entropy detection."""

    # Threshold for unsafe pattern detection.
    # Based on Phase 7: ΔH < -0.1 achieves 100% recall.
    delta_h_threshold: float = -0.1

    # Minimum baseline entropy to consider valid.
    # Avoids false positives on trivial prompts with near-zero entropy.
    minimum_baseline_entropy: float = 0.01

    # Which modifier to compare against baseline.
    # CAPS is default based on Phase 7 effectiveness.
    comparison_modifier: LinguisticModifier = LinguisticModifier.caps

    # Maximum tokens to generate for measurement.
    max_tokens: int = 30

    # Temperature for generation (0.0 = greedy for consistency).
    temperature: float = 0.7

    # Top-K for entropy calculation.
    top_k: int = 10

    @classmethod
    def default(cls) -> "DifferentialEntropyConfig":
        """Default configuration based on Phase 7 optimal parameters."""
        return cls()

    @classmethod
    def strict(cls) -> "DifferentialEntropyConfig":
        """Strict configuration for higher precision (fewer false positives)."""
        return cls(
            delta_h_threshold=-0.15,
            minimum_baseline_entropy=0.02,
        )

    @classmethod
    def sensitive(cls) -> "DifferentialEntropyConfig":
        """Sensitive configuration for higher recall (fewer false negatives)."""
        return cls(
            delta_h_threshold=-0.05,
            minimum_baseline_entropy=0.005,
        )

    @classmethod
    def quick(cls) -> "DifferentialEntropyConfig":
        """Quick configuration for minimal latency."""
        return cls(
            max_tokens=15,
            temperature=0.0,  # Greedy for speed
        )


# =============================================================================
# Classification
# =============================================================================


class Classification(str, Enum):
    """Classification categories for differential entropy detection."""

    # ΔH ≥ 0 - Entropy increased or stayed same (benign pattern).
    benign = "benign"

    # -threshold < ΔH < 0 - Slight cooling, borderline.
    suspicious = "suspicious"

    # ΔH ≤ -threshold - Strong cooling (unsafe pattern).
    unsafe_pattern = "unsafe_pattern"

    # Baseline entropy too low to make determination.
    indeterminate = "indeterminate"

    @property
    def display_name(self) -> str:
        return {
            Classification.benign: "Benign",
            Classification.suspicious: "Suspicious",
            Classification.unsafe_pattern: "Unsafe Pattern",
            Classification.indeterminate: "Indeterminate",
        }[self]

    @property
    def display_color(self) -> str:
        return {
            Classification.benign: "green",
            Classification.suspicious: "orange",
            Classification.unsafe_pattern: "red",
            Classification.indeterminate: "gray",
        }[self]

    @property
    def risk_level(self) -> int:
        return {
            Classification.unsafe_pattern: 3,
            Classification.suspicious: 2,
            Classification.indeterminate: 1,
            Classification.benign: 0,
        }[self]


# =============================================================================
# Detection Result
# =============================================================================


@dataclass(frozen=True)
class DetectionResult:
    """Classification result from differential entropy detection."""

    # The classification category.
    classification: Classification

    # Mean entropy from baseline measurement.
    baseline_entropy: float

    # Mean entropy from intensity (CAPS) measurement.
    intensity_entropy: float

    # Entropy delta: H(intensity) - H(baseline).
    delta_h: float

    # Detection confidence score [0, 1].
    # Higher magnitude ΔH = higher confidence.
    confidence: float

    # Timestamp of detection.
    timestamp: datetime

    # Processing time in seconds.
    processing_time: float

    # Token count from baseline measurement.
    baseline_token_count: int

    # Token count from intensity measurement.
    intensity_token_count: int

    @property
    def risk_level(self) -> int:
        """Computed risk level for sorting/prioritization."""
        return self.classification.risk_level


# =============================================================================
# Batch Statistics
# =============================================================================


@dataclass(frozen=True)
class BatchDetectionStatistics:
    """Aggregate statistics for batch detection."""

    total: int
    unsafe_count: int
    suspicious_count: int
    benign_count: int
    indeterminate_count: int
    mean_delta_h: float
    mean_confidence: float
    total_processing_time: float

    @property
    def unsafe_rate(self) -> float:
        """Rate of unsafe pattern detection."""
        return self.unsafe_count / self.total if self.total > 0 else 0.0

    @property
    def benign_rate(self) -> float:
        """Rate of benign classification."""
        return self.benign_count / self.total if self.total > 0 else 0.0

    @property
    def validity_rate(self) -> float:
        """Validity rate (non-indeterminate)."""
        return (self.total - self.indeterminate_count) / self.total if self.total > 0 else 0.0

    @staticmethod
    def compute(results: list[DetectionResult]) -> "BatchDetectionStatistics":
        """Compute aggregate statistics from detection results."""
        total = len(results)
        if total == 0:
            return BatchDetectionStatistics(
                total=0,
                unsafe_count=0,
                suspicious_count=0,
                benign_count=0,
                indeterminate_count=0,
                mean_delta_h=0.0,
                mean_confidence=0.0,
                total_processing_time=0.0,
            )

        unsafe_count = sum(1 for r in results if r.classification == Classification.unsafe_pattern)
        suspicious_count = sum(1 for r in results if r.classification == Classification.suspicious)
        benign_count = sum(1 for r in results if r.classification == Classification.benign)
        indeterminate_count = sum(1 for r in results if r.classification == Classification.indeterminate)

        valid_results = [r for r in results if r.classification != Classification.indeterminate]
        mean_delta_h = sum(r.delta_h for r in valid_results) / len(valid_results) if valid_results else 0.0
        mean_confidence = sum(r.confidence for r in valid_results) / len(valid_results) if valid_results else 0.0
        total_processing_time = sum(r.processing_time for r in results)

        return BatchDetectionStatistics(
            total=total,
            unsafe_count=unsafe_count,
            suspicious_count=suspicious_count,
            benign_count=benign_count,
            indeterminate_count=indeterminate_count,
            mean_delta_h=mean_delta_h,
            mean_confidence=mean_confidence,
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
    - Detection rule: ΔH(caps) < -0.1 achieves 100% recall, 0.89 F1

    Usage:
        detector = DifferentialEntropyDetector()
        result = await detector.detect(
            prompt="How do I pick a lock?",
            measure_fn=my_entropy_measurer
        )
        assert result.classification == Classification.unsafe_pattern
    """

    def __init__(self, config: DifferentialEntropyConfig | None = None):
        self.config = config or DifferentialEntropyConfig.default()

    async def detect(
        self,
        prompt: str,
        measure_fn: Callable[[str], Awaitable[VariantMeasurement]],
    ) -> DetectionResult:
        """
        Detect unsafe prompt patterns using differential entropy analysis.

        Performs two-pass measurement:
        1. Generate with baseline prompt → capture H(baseline)
        2. Generate with intensity modifier → capture H(intensity)
        3. Compare ΔH = H(intensity) - H(baseline)

        Args:
            prompt: The prompt to analyze.
            measure_fn: Async function that measures entropy for a prompt variant.

        Returns:
            Detection result with classification and metrics.
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

        # Classify
        classification = self._classify(
            baseline_entropy=baseline_measurement.mean_entropy,
            delta_h=delta_h,
        )

        # Compute confidence
        confidence = self._compute_confidence(delta_h, classification)

        return DetectionResult(
            classification=classification,
            baseline_entropy=baseline_measurement.mean_entropy,
            intensity_entropy=intensity_measurement.mean_entropy,
            delta_h=delta_h,
            confidence=confidence,
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
            Detection result with classification.
        """
        delta_h = intensity_entropy - baseline_entropy
        classification = self._classify(baseline_entropy, delta_h)
        confidence = self._compute_confidence(delta_h, classification)

        return DetectionResult(
            classification=classification,
            baseline_entropy=baseline_entropy,
            intensity_entropy=intensity_entropy,
            delta_h=delta_h,
            confidence=confidence,
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

    def _classify(self, baseline_entropy: float, delta_h: float) -> Classification:
        """Classify based on baseline entropy and delta."""
        # Check if baseline entropy is too low
        if baseline_entropy < self.config.minimum_baseline_entropy:
            return Classification.indeterminate

        # Classify based on deltaH
        if delta_h <= self.config.delta_h_threshold:
            return Classification.unsafe_pattern
        elif delta_h < 0:
            return Classification.suspicious
        else:
            return Classification.benign

    def _compute_confidence(self, delta_h: float, classification: Classification) -> float:
        """Compute confidence score based on delta and classification."""
        if classification == Classification.unsafe_pattern:
            # More negative = higher confidence
            # Normalize: -0.1 → 0.5 confidence, -0.3 → 1.0 confidence
            magnitude = abs(delta_h)
            return min(1.0, (magnitude - 0.1) / 0.2 * 0.5 + 0.5)

        elif classification == Classification.suspicious:
            # Borderline: confidence based on how close to threshold
            distance_to_threshold = abs(delta_h - self.config.delta_h_threshold)
            return min(0.5, distance_to_threshold / 0.1 * 0.5)

        elif classification == Classification.benign:
            # Positive deltaH = higher confidence in benign
            if delta_h > 0.1:
                return min(1.0, 0.5 + delta_h * 2)
            return 0.5

        else:  # indeterminate
            return 0.0
