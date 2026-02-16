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

"""Ridge cross detector for behavioral basin transitions.

Detects transitions between behavioral attractor basins and computes
transition statistics.

Notes
-----
Ridge crossing refers to the model escaping from the caution/refusal
attractor basin into the solution basin.

Key metric:
    Ridge Cross Rate = freq(outcome in {attempted, solved} | modifier)
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_math import EntropyMath
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
from modelcypher.experimental.thermo.linguistic_thermodynamics import (
    AttractorBasin,
    BehavioralOutcome,
    LinguisticModifier,
    ThermoMeasurement,
)


@dataclass(frozen=True)
class RidgeCrossEvent:
    """A detected transition between behavioral basins.

    Attributes
    ----------
    from_basin : AttractorBasin
        Basin the model started in.
    to_basin : AttractorBasin
        Basin the model transitioned to.
    trigger_modifier : LinguisticModifier
        Modifier that triggered transition.
    delta_h : float
        Entropy delta that enabled the crossing.
    from_outcome : BehavioralOutcome
        Behavioral outcome before (baseline).
    to_outcome : BehavioralOutcome
        Behavioral outcome after (variant).
    """

    from_basin: AttractorBasin
    to_basin: AttractorBasin
    trigger_modifier: LinguisticModifier
    delta_h: float
    from_outcome: BehavioralOutcome
    to_outcome: BehavioralOutcome

    @property
    def is_solution_crossing(self) -> bool:
        """Whether this crossed from caution/refusal basin to solution basin."""
        from_caution = self.from_basin in (AttractorBasin.REFUSAL, AttractorBasin.CAUTION)
        to_solution = self.to_basin in (AttractorBasin.TRANSITION, AttractorBasin.SOLUTION)
        return from_caution and to_solution

    @property
    def is_refusal_crossing(self) -> bool:
        """Whether this crossed from solution/transition basin to refusal basin."""
        from_solution = self.from_basin in (AttractorBasin.TRANSITION, AttractorBasin.SOLUTION)
        to_refusal = self.to_basin in (AttractorBasin.REFUSAL, AttractorBasin.CAUTION)
        return from_solution and to_refusal

    @property
    def description(self) -> str:
        """Human-readable description."""
        if self.is_solution_crossing:
            direction = "→ solution"
        elif self.is_refusal_crossing:
            direction = "→ refusal"
        else:
            direction = "→"
        return (
            f"{self.trigger_modifier.display_name}: "
            f"{self.from_basin.value} {direction} {self.to_basin.value} "
            f"(delta_H={self.delta_h:.3f})"
        )


@dataclass(frozen=True)
class RidgeCrossRateStats:
    """Statistics for ridge cross rate of a modifier."""

    modifier: LinguisticModifier
    rate: float
    sample_count: int
    crossed_count: int
    confidence_interval_lower: float
    confidence_interval_upper: float

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% confidence interval."""
        return (self.confidence_interval_lower, self.confidence_interval_upper)

    @property
    def display_string(self) -> str:
        """Human-readable display string."""
        return (
            f"{self.rate * 100:.1f}% "
            f"[{self.confidence_interval_lower * 100:.1f}%, "
            f"{self.confidence_interval_upper * 100:.1f}%] "
            f"(n={self.sample_count})"
        )


@dataclass(frozen=True)
class TransitionAnalysis:
    """Analysis of energy landscape based on observed transitions."""

    events: list[RidgeCrossEvent]
    solution_crossings: int
    most_effective_modifier: LinguisticModifier | None
    mean_successful_delta_h: float | None
    threshold_delta_h: float | None
    summary: str


class RidgeCrossDetector:
    """Detects transitions between behavioral attractor basins.

    Returns ALL transitions - caller determines significance from geometry.
    Transitions are sorted by delta_H magnitude.
    """

    def detect_crossings(
        self,
        baseline: ThermoMeasurement,
        variants: list[ThermoMeasurement],
    ) -> list[RidgeCrossEvent]:
        """Detect all ridge crossings by comparing variants to baseline.

        Returns ALL basin transitions sorted by |delta_H|.
        Caller interprets significance from the geometry.

        Args:
            baseline: Baseline measurement (no modifier).
            variants: Variant measurements to compare against baseline.

        Returns:
            List of ridge crossing events, sorted by |delta_H| descending.
        """
        baseline_basin = baseline.behavioral_outcome.basin
        events: list[RidgeCrossEvent] = []

        for variant in variants:
            # Skip baseline itself
            if variant.modifier == LinguisticModifier.BASELINE:
                continue

            variant_basin = variant.behavioral_outcome.basin

            # Compute delta_H
            delta_h = variant.delta_h
            if delta_h is None:
                delta_h = EntropyMath.compute_delta_h(
                    variant.mean_entropy,
                    baseline.mean_entropy,
                )
            if delta_h is None:
                continue

            # Record ALL transitions - geometry determines significance
            if variant_basin != baseline_basin:
                event = RidgeCrossEvent(
                    from_basin=baseline_basin,
                    to_basin=variant_basin,
                    trigger_modifier=variant.modifier,
                    delta_h=delta_h,
                    from_outcome=baseline.behavioral_outcome,
                    to_outcome=variant.behavioral_outcome,
                )
                events.append(event)

        # Sort by |delta_H| descending - most significant first
        events.sort(key=lambda e: abs(e.delta_h), reverse=True)
        return events

    def ridge_cross_rate(
        self,
        modifier: LinguisticModifier,
        measurements: list[ThermoMeasurement],
    ) -> float:
        """Compute ridge cross rate for a specific modifier.

        Ridge cross rate = P(outcome in {attempted, solved} | modifier)

        Args:
            modifier: The modifier to analyze.
            measurements: Measurements for this modifier across different prompts.

        Returns:
            Ridge cross rate [0, 1].
        """
        if not measurements:
            return 0.0

        crossed_count = sum(1 for m in measurements if m.ridge_crossed)
        return crossed_count / len(measurements)

    def ridge_cross_rates(
        self,
        measurements_by_modifier: dict[LinguisticModifier, list[ThermoMeasurement]],
    ) -> dict[LinguisticModifier, RidgeCrossRateStats]:
        """Compute ridge cross rates for all modifiers.

        Args:
            measurements_by_modifier: Measurements grouped by modifier.

        Returns:
            Dictionary mapping modifiers to their rate statistics.
        """
        stats: dict[LinguisticModifier, RidgeCrossRateStats] = {}

        for modifier, measurements in measurements_by_modifier.items():
            if not measurements:
                continue

            crossed_count = sum(1 for m in measurements if m.ridge_crossed)
            rate = crossed_count / len(measurements)

            # Compute Wilson score confidence interval (95%)
            n = float(len(measurements))
            p = rate
            z = 1.96  # 95% CI

            denominator = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denominator
            _b = get_default_backend()
            spread = z * sqrt_scalar((p * (1 - p) + z * z / (4 * n)) / n, _b) / denominator

            lower_bound = max(0.0, center - spread)
            upper_bound = min(1.0, center + spread)

            stats[modifier] = RidgeCrossRateStats(
                modifier=modifier,
                rate=rate,
                sample_count=len(measurements),
                crossed_count=crossed_count,
                confidence_interval_lower=lower_bound,
                confidence_interval_upper=upper_bound,
            )

        return stats

    def compute_effect_size(
        self,
        baseline_measurements: list[ThermoMeasurement],
        variant_measurements: list[ThermoMeasurement],
    ) -> float | None:
        """Compute effect size (Cohen's d) for a modifier's impact.

        Args:
            baseline_measurements: Baseline entropy measurements.
            variant_measurements: Variant entropy measurements.

        Returns:
            Cohen's d effect size, or None if insufficient data.
        """
        if not baseline_measurements or not variant_measurements:
            return None

        baseline_entropies = [m.mean_entropy for m in baseline_measurements]
        variant_entropies = [m.mean_entropy for m in variant_measurements]

        return self._cohens_d(variant_entropies, baseline_entropies)

    def _cohens_d(self, group1: list[float], group2: list[float]) -> float:
        """Compute Cohen's d effect size between two groups."""
        if not group1 or not group2:
            return 0.0

        mean1 = sum(group1) / len(group1)
        mean2 = sum(group2) / len(group2)

        var1 = sum((x - mean1) ** 2 for x in group1) / max(1, len(group1) - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / max(1, len(group2) - 1)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        _b = get_default_backend()
        pooled_std = sqrt_scalar(pooled_var, _b) if pooled_var > 0 else 1.0

        return (mean1 - mean2) / pooled_std

    def analyze_transitions(
        self,
        baseline: ThermoMeasurement,
        variants: list[ThermoMeasurement],
    ) -> TransitionAnalysis:
        """Perform comprehensive transition analysis.

        Args:
            baseline: Baseline measurement.
            variants: Variant measurements.

        Returns:
            TransitionAnalysis with summary statistics.
        """
        events = self.detect_crossings(baseline, variants)

        # Count solution crossings
        solution_crossings = sum(1 for e in events if e.is_solution_crossing)

        # Group by modifier
        events_by_modifier: dict[LinguisticModifier, list[RidgeCrossEvent]] = {}
        for event in events:
            modifier = event.trigger_modifier
            if modifier not in events_by_modifier:
                events_by_modifier[modifier] = []
            events_by_modifier[modifier].append(event)

        # Find most effective modifier
        most_effective: LinguisticModifier | None = None
        max_solution_count = 0
        for modifier, modifier_events in events_by_modifier.items():
            solution_count = sum(1 for e in modifier_events if e.is_solution_crossing)
            if solution_count > max_solution_count:
                max_solution_count = solution_count
                most_effective = modifier

        # Compute mean delta_H for successful crossings
        successful_delta_hs = [e.delta_h for e in events if e.is_solution_crossing]
        mean_successful_delta_h: float | None = None
        if successful_delta_hs:
            mean_successful_delta_h = sum(successful_delta_hs) / len(successful_delta_hs)

        # Find threshold (minimum delta_H that achieved crossing)
        threshold_delta_h = min(successful_delta_hs) if successful_delta_hs else None

        # Generate summary
        summary_lines = [
            "Transition Analysis Summary",
            f"Total crossings detected: {len(events)}",
            f"Solution crossings: {solution_crossings}",
        ]

        if most_effective:
            summary_lines.append(f"Most effective modifier: {most_effective.display_name}")

        if mean_successful_delta_h is not None:
            summary_lines.append(f"Mean delta_H for success: {mean_successful_delta_h:.4f}")

        if threshold_delta_h is not None:
            summary_lines.append(f"Threshold delta_H: {threshold_delta_h:.4f}")

        return TransitionAnalysis(
            events=list(events),
            solution_crossings=solution_crossings,
            most_effective_modifier=most_effective,
            mean_successful_delta_h=mean_successful_delta_h,
            threshold_delta_h=threshold_delta_h,
            summary="\n".join(summary_lines),
        )
