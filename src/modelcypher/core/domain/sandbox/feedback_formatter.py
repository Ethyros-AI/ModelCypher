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

"""Format geometric metrics as human-readable feedback.

The model learns to interpret its own geometry through structured text feedback.
This module converts raw metrics (expansion_ratio, peak layer, entropy) into
interpretable text the model can "see" and learn from.

Key insight: By formatting geometry as text, the model can:
1. Observe patterns between geometry and correctness
2. Learn to predict geometry before generating
3. Develop intuition for "what good geometry feels like"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EntropyPattern(Enum):
    """Classification of entropy trajectory patterns."""

    EXPAND_COMPRESS = "expand_compress"  # Expected healthy pattern
    FLAT = "flat"  # No expansion phase
    MONOTONIC_INCREASE = "monotonic_increase"  # Never compresses
    MONOTONIC_DECREASE = "monotonic_decrease"  # Only compresses
    IRREGULAR = "irregular"  # No clear pattern


@dataclass
class GeometricFeedback:
    """Structured geometric feedback for model self-study.

    Attributes:
        expansion_ratio: Peak/final dimension ratio (target: 1.0 = flat trajectory)
        peak_layer: Layer with maximum activation/entropy
        n_layers: Total number of layers
        entropy_pattern: Classification of the entropy trajectory
        expansion_rate: Rate of activation growth to peak
        compression_rate: Rate of activation reduction from peak
        interpretation: Human-readable interpretation of the geometry
    """

    expansion_ratio: float
    peak_layer: float
    n_layers: int
    entropy_pattern: EntropyPattern
    expansion_rate: float
    compression_rate: float
    interpretation: str

    @property
    def peak_layer_fraction(self) -> float:
        """Peak position as fraction of total layers (0.0 = early, 1.0 = late)."""
        return self.peak_layer / self.n_layers if self.n_layers > 0 else 0.5

    @property
    def expansion_ratio_distance(self) -> float:
        """Distance from target expansion_ratio = 1.0."""
        return abs(self.expansion_ratio - 1.0)


def classify_entropy_pattern(
    trajectory: list[float],
    monotonicity_threshold: float = 0.3,
) -> EntropyPattern:
    """Classify the entropy trajectory pattern.

    Args:
        trajectory: List of entropy values per layer
        monotonicity_threshold: Threshold for classifying as monotonic

    Returns:
        EntropyPattern classification
    """
    if not trajectory or len(trajectory) < 3:
        return EntropyPattern.FLAT

    # Find peak
    peak_idx = max(range(len(trajectory)), key=lambda i: trajectory[i])
    peak_val = trajectory[peak_idx]
    initial = trajectory[0]
    final = trajectory[-1]

    # Check for flat pattern (minimal variation)
    val_range = max(trajectory) - min(trajectory)
    mean_val = sum(trajectory) / len(trajectory)
    if val_range < 0.1 * mean_val:  # Less than 10% variation
        return EntropyPattern.FLAT

    # Check for expand-compress (peak in middle, lower at ends)
    has_expansion = peak_idx > 0 and peak_val > initial
    has_compression = peak_idx < len(trajectory) - 1 and peak_val > final

    if has_expansion and has_compression:
        return EntropyPattern.EXPAND_COMPRESS

    # Check for monotonic patterns
    increasing = sum(
        1 for i in range(len(trajectory) - 1) if trajectory[i + 1] > trajectory[i]
    )
    decreasing = sum(
        1 for i in range(len(trajectory) - 1) if trajectory[i + 1] < trajectory[i]
    )
    n_transitions = len(trajectory) - 1

    if increasing / n_transitions > (1 - monotonicity_threshold):
        return EntropyPattern.MONOTONIC_INCREASE
    if decreasing / n_transitions > (1 - monotonicity_threshold):
        return EntropyPattern.MONOTONIC_DECREASE

    return EntropyPattern.IRREGULAR


def _interpret_expansion_ratio(expansion_ratio: float) -> str:
    """Generate interpretation text for expansion ratio.

    Interpretations based on empirical observations:
    - expansion_ratio = 1.0: Flat trajectory (peak = final)
    - expansion_ratio < 0.8: Compression dominant
    - expansion_ratio > 1.4: Expansion dominant
    """
    if 0.9 <= expansion_ratio <= 1.1:
        return "BALANCED - flat trajectory"
    elif expansion_ratio < 0.8:
        return "COMPRESSED - narrow processing"
    elif expansion_ratio < 0.9:
        return "SLIGHTLY_COMPRESSED - processing somewhat narrow"
    elif expansion_ratio > 1.4:
        return "EXPANDED - over-expansion, reasoning may be unfocused"
    elif expansion_ratio > 1.1:
        return "SLIGHTLY_EXPANDED - processing somewhat broad"
    else:
        return "NEAR_BALANCED - close to flat trajectory"


def _interpret_peak_layer(peak_fraction: float) -> str:
    """Generate interpretation text for peak layer position.

    Interpretations:
    - Early peak (< 0.3): Minimal expansion phase
    - Middle peak (0.3-0.7): Healthy expand-compress cycle
    - Late peak (> 0.7): Minimal compression phase
    """
    if peak_fraction < 0.3:
        return "EARLY - minimal expansion phase"
    elif peak_fraction > 0.7:
        return "LATE - minimal compression phase"
    else:
        return "MIDDLE - healthy expand-compress cycle"


def _interpret_pattern(pattern: EntropyPattern) -> str:
    """Generate interpretation text for entropy pattern."""
    interpretations = {
        EntropyPattern.EXPAND_COMPRESS: "expand-compress cycle detected",
        EntropyPattern.FLAT: "no expand-compress cycle detected",
        EntropyPattern.MONOTONIC_INCREASE: "only expansion, no compression",
        EntropyPattern.MONOTONIC_DECREASE: "only compression, no expansion",
        EntropyPattern.IRREGULAR: "irregular pattern",
    }
    return interpretations.get(pattern, "unknown pattern")


def _generate_interpretation(feedback: GeometricFeedback) -> str:
    """Generate full interpretation text based on all geometric signals.

    Combines expansion_ratio, peak position, and entropy pattern into actionable
    interpretation. The model learns to use these interpretations to adjust
    its reasoning approach.
    """
    lines = []

    # Primary signal: expansion_ratio
    ratio_interp = _interpret_expansion_ratio(feedback.expansion_ratio)
    if "BALANCED" in ratio_interp:
        lines.append("Processing geometry is balanced.")
    elif "COMPRESSED" in ratio_interp:
        lines.append("Processing was shallow. Consider explicit step-by-step reasoning.")
    elif "EXPANDED" in ratio_interp:
        lines.append("Processing was over-expanded. Consider focusing on core question.")

    # Secondary signal: peak position
    peak_interp = _interpret_peak_layer(feedback.peak_layer_fraction)
    if "EARLY" in peak_interp:
        lines.append("Peak came early - minimal exploration before converging.")
    elif "LATE" in peak_interp:
        lines.append("Peak came late - minimal compression before output.")

    # Tertiary signal: entropy pattern
    if feedback.entropy_pattern == EntropyPattern.FLAT:
        lines.append("Flat entropy suggests intuitive (System 1) processing.")
    elif feedback.entropy_pattern == EntropyPattern.EXPAND_COMPRESS:
        lines.append("Expand-compress cycle suggests deliberate (System 2) processing.")

    return " ".join(lines)


def format_geometric_feedback(
    expansion_ratio: float,
    peak_layer: float,
    n_layers: int,
    expansion_rate: float,
    compression_rate: float,
    entropy_trajectory: list[float] | None = None,
) -> GeometricFeedback:
    """Create structured geometric feedback from raw metrics.

    Args:
        expansion_ratio: Peak/final dimension ratio
        peak_layer: Layer index of peak activation/entropy
        n_layers: Total number of model layers
        expansion_rate: Rate of activation growth to peak
        compression_rate: Rate of activation reduction from peak
        entropy_trajectory: Optional list of per-layer entropy values

    Returns:
        GeometricFeedback with interpretation
    """
    # Classify entropy pattern
    if entropy_trajectory:
        pattern = classify_entropy_pattern(entropy_trajectory)
    else:
        # Infer pattern from expansion_ratio if no trajectory provided
        if expansion_ratio < 0.8:
            pattern = EntropyPattern.FLAT
        elif expansion_ratio > 1.4:
            pattern = EntropyPattern.MONOTONIC_INCREASE
        else:
            pattern = EntropyPattern.EXPAND_COMPRESS

    feedback = GeometricFeedback(
        expansion_ratio=expansion_ratio,
        peak_layer=peak_layer,
        n_layers=n_layers,
        entropy_pattern=pattern,
        expansion_rate=expansion_rate,
        compression_rate=compression_rate,
        interpretation="",  # Will be set below
    )

    # Generate interpretation
    interpretation = _generate_interpretation(feedback)

    # Return with interpretation filled in
    return GeometricFeedback(
        expansion_ratio=expansion_ratio,
        peak_layer=peak_layer,
        n_layers=n_layers,
        entropy_pattern=pattern,
        expansion_rate=expansion_rate,
        compression_rate=compression_rate,
        interpretation=interpretation,
    )


def format_feedback_text(feedback: GeometricFeedback) -> str:
    """Format GeometricFeedback as structured text for model consumption.

    This is the text format the model "sees" during self-study. It learns
    to interpret these signals and correlate them with response quality.

    Example output:
        === GEOMETRIC FEEDBACK ===
        expansion_ratio: 1.05 (BALANCED - flat trajectory)
        peak_layer: 15/16 (LATE - minimal compression phase)
        entropy_pattern: flat (no expand-compress cycle detected)
        interpretation: Processing geometry is balanced.
        ===========================
    """
    ratio_interp = _interpret_expansion_ratio(feedback.expansion_ratio)
    peak_interp = _interpret_peak_layer(feedback.peak_layer_fraction)
    pattern_interp = _interpret_pattern(feedback.entropy_pattern)

    lines = [
        "=== GEOMETRIC FEEDBACK ===",
        f"expansion_ratio: {feedback.expansion_ratio:.3f} ({ratio_interp})",
        f"peak_layer: {feedback.peak_layer:.1f}/{feedback.n_layers} ({peak_interp})",
        f"entropy_pattern: {feedback.entropy_pattern.value} ({pattern_interp})",
        f"interpretation: {feedback.interpretation}",
        "===========================",
    ]
    return "\n".join(lines)


def format_comparison_text(
    approaches: list[tuple[str, GeometricFeedback, str]],
) -> str:
    """Format multiple approaches with their geometry for comparison.

    Args:
        approaches: List of (approach_name, feedback, response) tuples

    Returns:
        Formatted comparison text for model consumption
    """
    lines = ["=== GEOMETRIC COMPARISON ===", ""]

    for i, (name, feedback, response) in enumerate(approaches, 1):
        ratio_interp = _interpret_expansion_ratio(feedback.expansion_ratio)
        lines.extend([
            f"--- Approach {i}: {name} ---",
            f"Response: {response[:100]}...",
            f"expansion_ratio: {feedback.expansion_ratio:.3f} ({ratio_interp})",
            f"peak_layer: {feedback.peak_layer:.1f}/{feedback.n_layers}",
            f"pattern: {feedback.entropy_pattern.value}",
            "",
        ])

    # Add summary
    best_idx = min(
        range(len(approaches)),
        key=lambda i: abs(approaches[i][1].expansion_ratio - 1.0),
    )
    best_name = approaches[best_idx][0]
    lines.extend([
        "--- Summary ---",
        f"Best geometry: {best_name} (closest to expansion_ratio = 1.0)",
        "============================",
    ])

    return "\n".join(lines)
