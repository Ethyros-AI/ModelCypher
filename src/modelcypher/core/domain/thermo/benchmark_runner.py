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

"""Thermodynamic benchmark runner for modifier effectiveness analysis.

Framework for comparative analysis of modifier effectiveness across
prompt corpora with raw statistical measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
from modelcypher.core.domain.thermo.linguistic_calorimeter import (
    LinguisticCalorimeter,
)
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    LinguisticModifier,
    ThermoMeasurement,
)

# =============================================================================
# Result Types
# =============================================================================


@dataclass
class SignificanceResult:
    """Result of statistical significance testing."""

    t_statistic: float
    degrees_of_freedom: float


@dataclass
class EffectSizeResult:
    """Cohen's d effect size with standard error."""

    cohens_d: float
    standard_error: float


@dataclass
class ModifierStats:
    """Statistics for a single modifier."""

    modifier: LinguisticModifier
    sample_size: int
    mean_entropy: float
    std_entropy: float
    mean_delta_h: float | None
    ridge_cross_rate: float
    significance: SignificanceResult | None
    effect_size: EffectSizeResult | None


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all modifiers."""

    corpus_size: int
    modifiers: list[ModifierStats]
    baseline_mean: float
    baseline_std: float
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Benchmark Runner
# =============================================================================


class ThermoBenchmarkRunner:
    """Run and analyze thermodynamic experiments.

    Compares modifier effectiveness across a prompt corpus with
    statistical rigor.

    Parameters
    ----------
    calorimeter : LinguisticCalorimeter | None
        LinguisticCalorimeter instance. If None, creates simulated.
    """

    def __init__(
        self,
        calorimeter: LinguisticCalorimeter | None = None,
    ):
        """Initialize the benchmark runner.

        Args:
            calorimeter: LinguisticCalorimeter instance. If None, creates simulated.
        """
        self.calorimeter = calorimeter or LinguisticCalorimeter(simulated=True)

    def run_modifier_comparison(
        self,
        prompts: list[str],
        modifiers: list[LinguisticModifier] | None = None,
        temperature: float | None = None,
    ) -> BenchmarkResult:
        """Compare modifier effects across prompt corpus.

        Args:
            prompts: List of prompts to test.
            modifiers: Modifiers to compare. Defaults to all.
            temperature: Sampling temperature scale. If None, uses identity scaling.

        Returns:
            BenchmarkResult with statistics for each modifier.
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        if modifiers is None:
            modifiers = list(LinguisticModifier)

        # Ensure baseline is included
        if LinguisticModifier.BASELINE not in modifiers:
            modifiers = [LinguisticModifier.BASELINE] + list(modifiers)

        # Collect measurements per modifier
        measurements_by_modifier: dict[LinguisticModifier, list[ThermoMeasurement]] = {
            m: [] for m in modifiers
        }

        for prompt in prompts:
            prompt_measurements = self.calorimeter.measure_with_modifiers(
                prompt=prompt,
                modifiers=modifiers,
                temperature=temperature,
            )
            for m in prompt_measurements:
                measurements_by_modifier[m.modifier].append(m)

        # Compute baseline statistics
        baseline_measurements = measurements_by_modifier[LinguisticModifier.BASELINE]
        baseline_entropies = [m.mean_entropy for m in baseline_measurements]
        baseline_mean = sum(baseline_entropies) / len(baseline_entropies)
        baseline_std = self._std(baseline_entropies)

        # Compute stats for each modifier
        modifier_stats = []
        for modifier in modifiers:
            measurements = measurements_by_modifier[modifier]
            entropies = [m.mean_entropy for m in measurements]
            delta_hs = [m.delta_h for m in measurements if m.delta_h is not None]

            mean_entropy = sum(entropies) / len(entropies)
            std_entropy = self._std(entropies)
            mean_delta_h = sum(delta_hs) / len(delta_hs) if delta_hs else None

            # Ridge cross rate
            ridge_crosses = sum(1 for m in measurements if m.ridge_crossed)
            ridge_rate = ridge_crosses / len(measurements)

            # Statistical significance vs baseline
            significance = None
            effect_size = None
            if modifier != LinguisticModifier.BASELINE:
                significance = self.statistical_significance(
                    baseline_entropies,
                    entropies,
                )
                effect_size = self._compute_effect_size(
                    baseline_entropies,
                    entropies,
                )

            stats = ModifierStats(
                modifier=modifier,
                sample_size=len(measurements),
                mean_entropy=mean_entropy,
                std_entropy=std_entropy,
                mean_delta_h=mean_delta_h,
                ridge_cross_rate=ridge_rate,
                significance=significance,
                effect_size=effect_size,
            )
            modifier_stats.append(stats)

        return BenchmarkResult(
            corpus_size=len(prompts),
            modifiers=modifier_stats,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
        )

    def statistical_significance(
        self,
        baseline: list[float],
        treatment: list[float],
    ) -> SignificanceResult:
        """Welch's t-test for entropy difference significance.

        Uses Welch's t-test which does not assume equal variances.

        Args:
            baseline: Baseline entropy measurements.
            treatment: Treatment (modified) entropy measurements.

        Returns:
            SignificanceResult with test statistics.
        """
        n1 = len(baseline)
        n2 = len(treatment)

        if n1 < 2 or n2 < 2:
            return SignificanceResult(
                t_statistic=0.0,
                degrees_of_freedom=0.0,
            )

        mean1 = sum(baseline) / n1
        mean2 = sum(treatment) / n2

        var1 = sum((x - mean1) ** 2 for x in baseline) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment) / (n2 - 1)

        # Welch's t-statistic
        _b = get_default_backend()
        se = sqrt_scalar(var1 / n1 + var2 / n2, _b)
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else 1.0

        return SignificanceResult(
            t_statistic=t_stat,
            degrees_of_freedom=df,
        )

    def _compute_effect_size(
        self,
        baseline: list[float],
        treatment: list[float],
    ) -> EffectSizeResult:
        """Compute Cohen's d effect size with 95% CI.

        Args:
            baseline: Baseline measurements.
            treatment: Treatment measurements.

        Returns:
            EffectSizeResult with Cohen's d and confidence interval.
        """
        n1 = len(baseline)
        n2 = len(treatment)

        if n1 < 2 or n2 < 2:
            return EffectSizeResult(
                cohens_d=0.0,
                standard_error=0.0,
            )

        mean1 = sum(baseline) / n1
        mean2 = sum(treatment) / n2

        var1 = sum((x - mean1) ** 2 for x in baseline) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment) / (n2 - 1)

        # Pooled standard deviation
        _b = get_default_backend()
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = sqrt_scalar(pooled_var, _b) if pooled_var > 0 else 1.0

        # Cohen's d
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        # Standard error for Cohen's d
        # SE(d) ≈ sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2)))
        se_d = sqrt_scalar((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)), _b)

        return EffectSizeResult(
            cohens_d=d,
            standard_error=se_d,
        )

    def effect_size_analysis(
        self, result: BenchmarkResult
    ) -> dict[LinguisticModifier, EffectSizeResult]:
        """Extract effect sizes for all modifiers.

        Args:
            result: Benchmark result.

        Returns:
            Dict mapping modifier to effect size.
        """
        return {
            stats.modifier: stats.effect_size
            for stats in result.modifiers
            if stats.effect_size is not None
        }

    def generate_report(self, result: BenchmarkResult) -> str:
        """Generate markdown report with measurement tables.

        Args:
            result: Benchmark result.

        Returns:
            Markdown formatted report with raw measurements.
        """
        lines = [
            "# Thermodynamic Benchmark Report",
            "",
            f"**Generated**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Corpus Size**: {result.corpus_size} prompts",
            "",
            "## Summary",
            "",
            f"- **Baseline Mean Entropy**: {result.baseline_mean:.4f} ± {result.baseline_std:.4f}",
            "",
            "## Modifier Comparison",
            "",
            "| Modifier | Mean H | Δ H | Ridge Rate | Effect Size | t-stat |",
            "|----------|--------|-----|------------|-------------|--------|",
        ]

        for stats in result.modifiers:
            delta_h = f"{stats.mean_delta_h:.4f}" if stats.mean_delta_h is not None else "—"
            effect = f"d={stats.effect_size.cohens_d:.3f}" if stats.effect_size else "—"
            t_stat = f"{stats.significance.t_statistic:.3f}" if stats.significance else "—"

            lines.append(
                f"| {stats.modifier.display_name} | {stats.mean_entropy:.4f} | "
                f"{delta_h} | {stats.ridge_cross_rate:.1%} | {effect} | {t_stat} |"
            )

        lines.extend(
            [
                "",
                "## Statistical Details",
                "",
            ]
        )

        for stats in result.modifiers:
            if stats.significance:
                lines.extend(
                    [
                        f"### {stats.modifier.display_name}",
                        "",
                        f"- t-statistic: {stats.significance.t_statistic:.4f}",
                        f"- df: {stats.significance.degrees_of_freedom:.1f}",
                        "",
                    ]
                )
                if stats.effect_size:
                    lines.extend(
                        [
                            f"- Cohen's d: {stats.effect_size.cohens_d:.4f}",
                            f"- SE(d): {stats.effect_size.standard_error:.4f}",
                            "",
                        ]
                    )

        # NOTE: "Recommendations" section removed per "No Vibes" rule.
        # Raw measurements (effect sizes, test statistics, confidence intervals) are
        # provided above. Caller interprets meaning based on their context.

        return "\n".join(lines)

    def _std(self, values: list[float]) -> float:
        """Compute sample standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return sqrt_scalar(variance, get_default_backend())
