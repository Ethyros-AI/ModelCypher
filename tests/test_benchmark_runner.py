"""Tests for ThermoBenchmarkRunner."""
from __future__ import annotations

import pytest

from modelcypher.core.domain.thermo.benchmark_runner import (
    BenchmarkResult,
    EffectSizeResult,
    ModifierStats,
    SignificanceResult,
    ThermoBenchmarkRunner,
)
from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
from modelcypher.core.domain.thermo.linguistic_thermodynamics import LinguisticModifier


class TestThermoBenchmarkRunner:
    """Tests for ThermoBenchmarkRunner."""

    @pytest.fixture
    def runner(self) -> ThermoBenchmarkRunner:
        """Create a benchmark runner with simulated calorimeter."""
        calorimeter = LinguisticCalorimeter(simulated=True)
        return ThermoBenchmarkRunner(calorimeter=calorimeter)

    def test_run_modifier_comparison_returns_result(self, runner: ThermoBenchmarkRunner) -> None:
        """Should return a BenchmarkResult."""
        prompts = [
            "What is 2+2?",
            "Explain gravity.",
            "Write a haiku.",
        ]
        modifiers = [
            LinguisticModifier.BASELINE,
            LinguisticModifier.CAPS,
        ]

        result = runner.run_modifier_comparison(
            prompts=prompts,
            modifiers=modifiers,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.corpus_size == 3
        assert len(result.modifiers) == 2

    def test_run_modifier_comparison_includes_baseline(self, runner: ThermoBenchmarkRunner) -> None:
        """Should always include baseline even if not specified."""
        prompts = ["What is 2+2?"]
        modifiers = [LinguisticModifier.CAPS]  # No baseline specified

        result = runner.run_modifier_comparison(
            prompts=prompts,
            modifiers=modifiers,
        )

        # Baseline should be added automatically
        modifier_values = [s.modifier for s in result.modifiers]
        assert LinguisticModifier.BASELINE in modifier_values

    def test_run_modifier_comparison_computes_statistics(self, runner: ThermoBenchmarkRunner) -> None:
        """Should compute statistics for each modifier."""
        prompts = ["What is 2+2?", "Explain light."]
        modifiers = [
            LinguisticModifier.BASELINE,
            LinguisticModifier.CAPS,
        ]

        result = runner.run_modifier_comparison(
            prompts=prompts,
            modifiers=modifiers,
        )

        caps_stats = next(s for s in result.modifiers if s.modifier == LinguisticModifier.CAPS)

        assert caps_stats.sample_size == 2
        assert caps_stats.mean_entropy > 0
        assert caps_stats.significance is not None
        assert caps_stats.effect_size is not None

    def test_run_modifier_comparison_empty_prompts_raises(self, runner: ThermoBenchmarkRunner) -> None:
        """Should raise on empty prompts list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            runner.run_modifier_comparison(prompts=[])

    def test_generate_report_produces_markdown(self, runner: ThermoBenchmarkRunner) -> None:
        """Should generate markdown report."""
        prompts = ["What is 2+2?", "Explain gravity."]

        result = runner.run_modifier_comparison(prompts=prompts)
        report = runner.generate_report(result)

        assert "# Thermodynamic Benchmark Report" in report
        assert "## Summary" in report
        assert "## Modifier Comparison" in report
        assert "Baseline Mean Entropy" in report


class TestStatisticalSignificance:
    """Tests for statistical significance testing."""

    @pytest.fixture
    def runner(self) -> ThermoBenchmarkRunner:
        """Create a benchmark runner."""
        return ThermoBenchmarkRunner()

    def test_welch_t_test_identical_samples(self, runner: ThermoBenchmarkRunner) -> None:
        """Identical samples should not be significant."""
        baseline = [2.0, 2.0, 2.0, 2.0, 2.0]
        treatment = [2.0, 2.0, 2.0, 2.0, 2.0]

        result = runner.statistical_significance(baseline, treatment)

        assert isinstance(result, SignificanceResult)
        assert result.t_statistic == 0.0
        assert not result.is_significant

    def test_welch_t_test_different_samples(self, runner: ThermoBenchmarkRunner) -> None:
        """Very different samples should be significant."""
        baseline = [1.0, 1.1, 1.0, 0.9, 1.1]
        treatment = [5.0, 5.1, 5.0, 4.9, 5.1]

        result = runner.statistical_significance(baseline, treatment)

        assert result.t_statistic != 0.0
        assert result.is_significant
        assert result.p_value < 0.05

    def test_welch_t_test_small_sample_not_significant(self, runner: ThermoBenchmarkRunner) -> None:
        """Small samples should fail gracefully."""
        baseline = [1.0]  # Too small
        treatment = [2.0]

        result = runner.statistical_significance(baseline, treatment)

        assert result.p_value == 1.0
        assert not result.is_significant


class TestEffectSize:
    """Tests for Cohen's d effect size calculation."""

    @pytest.fixture
    def runner(self) -> ThermoBenchmarkRunner:
        """Create a benchmark runner."""
        return ThermoBenchmarkRunner()

    def test_cohens_d_zero_difference(self, runner: ThermoBenchmarkRunner) -> None:
        """Same means should have zero effect size."""
        baseline = [2.0, 2.1, 1.9, 2.0, 2.0]
        treatment = [2.0, 2.1, 1.9, 2.0, 2.0]

        result = runner._compute_effect_size(baseline, treatment)

        assert isinstance(result, EffectSizeResult)
        assert abs(result.cohens_d) < 0.1
        assert result.interpretation == "negligible"

    def test_cohens_d_large_effect(self, runner: ThermoBenchmarkRunner) -> None:
        """Large difference should have large effect size."""
        baseline = [1.0, 1.1, 1.0, 0.9, 1.0]
        treatment = [3.0, 3.1, 3.0, 2.9, 3.0]

        result = runner._compute_effect_size(baseline, treatment)

        assert abs(result.cohens_d) > 0.8
        assert result.interpretation == "large"

    def test_cohens_d_small_effect(self, runner: ThermoBenchmarkRunner) -> None:
        """Small difference should have small effect size."""
        # Use larger variance and smaller mean difference to get small effect
        baseline = [1.5, 2.0, 2.5, 2.0, 2.0]  # mean ~2.0, std ~0.35
        treatment = [1.65, 2.15, 2.65, 2.15, 2.15]  # mean ~2.15, std ~0.35
        # d ≈ 0.15 / 0.35 ≈ 0.43 (small effect)

        result = runner._compute_effect_size(baseline, treatment)

        assert 0.2 <= abs(result.cohens_d) < 0.5
        assert result.interpretation == "small"

    def test_cohens_d_confidence_interval(self, runner: ThermoBenchmarkRunner) -> None:
        """Should compute 95% CI."""
        baseline = [2.0, 2.1, 1.9, 2.0, 2.0, 2.1]
        treatment = [3.0, 3.1, 2.9, 3.0, 3.0, 3.1]

        result = runner._compute_effect_size(baseline, treatment)

        assert result.ci_lower < result.cohens_d < result.ci_upper


class TestModifierStats:
    """Tests for ModifierStats dataclass."""

    def test_modifier_stats_fields(self) -> None:
        """Should hold all required fields."""
        stats = ModifierStats(
            modifier=LinguisticModifier.CAPS,
            sample_size=10,
            mean_entropy=2.5,
            std_entropy=0.3,
            mean_delta_h=-0.2,
            ridge_cross_rate=0.4,
            significance=None,
            effect_size=None,
        )

        assert stats.modifier == LinguisticModifier.CAPS
        assert stats.sample_size == 10
        assert stats.mean_entropy == 2.5
        assert stats.ridge_cross_rate == 0.4


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_timestamp(self) -> None:
        """Should have timestamp."""
        from datetime import datetime

        result = BenchmarkResult(
            corpus_size=5,
            modifiers=[],
            baseline_mean=2.0,
            baseline_std=0.3,
            best_modifier=LinguisticModifier.CAPS,
            best_effect_size=-0.5,
        )

        assert isinstance(result.timestamp, datetime)
