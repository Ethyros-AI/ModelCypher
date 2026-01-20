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

"""Tests for ThermoBenchmarkRunner."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.thermo.benchmark_runner import (
    BenchmarkResult,
    EffectSizeResult,
    ModifierStats,
    SignificanceResult,
    ThermoBenchmarkRunner,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
from modelcypher.core.domain.thermo.linguistic_thermodynamics import LinguisticModifier


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


class DummyTokenizer:
    def __init__(self, vocab_size: int = 16, model_max_length: int = 32) -> None:
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.eos_token_id = vocab_size - 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text.strip():
            return [0] if add_special_tokens else []
        tokens = []
        for part in text.split():
            token_id = sum(ord(ch) for ch in part) % (self.vocab_size - 1)
            tokens.append(token_id)
        return tokens or ([0] if add_special_tokens else [])

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"<t{token_id}>" for token_id in token_ids)


class DummyModel:
    def __init__(self, backend, vocab_size: int) -> None:
        self._backend = backend
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        seq_len = int(input_ids.shape[1])
        vocab = self._backend.arange(self._vocab_size)
        vocab = vocab + 0.0
        logits = self._backend.tile(vocab, (seq_len, 1))
        return self._backend.expand_dims(logits, axis=0)


def _make_calorimeter() -> LinguisticCalorimeter:
    backend = get_default_backend()
    tokenizer = DummyTokenizer()
    model = DummyModel(backend, tokenizer.vocab_size)
    return LinguisticCalorimeter(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
    )


class TestThermoBenchmarkRunner:
    """Tests for ThermoBenchmarkRunner."""

    @pytest.fixture
    def runner(self) -> ThermoBenchmarkRunner:
        """Create a benchmark runner with dummy calorimeter."""
        calorimeter = _make_calorimeter()
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

    def test_run_modifier_comparison_computes_statistics(
        self, runner: ThermoBenchmarkRunner
    ) -> None:
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
        assert caps_stats.mean_entropy >= -_eps()
        assert caps_stats.significance is not None
        assert caps_stats.effect_size is not None

    def test_run_modifier_comparison_empty_prompts_raises(
        self, runner: ThermoBenchmarkRunner
    ) -> None:
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
        assert "| Modifier |" in report
        assert "t-stat" in report


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
        assert abs(result.t_statistic) <= _eps()

    def test_welch_t_test_different_samples(self, runner: ThermoBenchmarkRunner) -> None:
        """Very different samples should be significant."""
        baseline = [1.0, 1.1, 1.0, 0.9, 1.1]
        treatment = [5.0, 5.1, 5.0, 4.9, 5.1]

        result = runner.statistical_significance(baseline, treatment)
        baseline_result = runner.statistical_significance(baseline, baseline)

        assert abs(result.t_statistic) >= _eps()
        assert abs(result.t_statistic) >= abs(baseline_result.t_statistic) + _eps()

    def test_welch_t_test_small_sample_not_significant(self, runner: ThermoBenchmarkRunner) -> None:
        """Small samples should fail gracefully."""
        baseline = [1.0]  # Too small
        treatment = [2.0]

        result = runner.statistical_significance(baseline, treatment)

        assert abs(result.t_statistic) <= _eps()
        assert result.degrees_of_freedom == 0.0


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
        assert abs(result.cohens_d) <= _eps()

    def test_cohens_d_large_effect(self, runner: ThermoBenchmarkRunner) -> None:
        """Large difference should have large effect size."""
        baseline = [1.0, 1.1, 1.0, 0.9, 1.0]
        treatment = [3.0, 3.1, 3.0, 2.9, 3.0]

        result = runner._compute_effect_size(baseline, treatment)

        mean1 = sum(baseline) / len(baseline)
        mean2 = sum(treatment) / len(treatment)
        var1 = sum((x - mean1) ** 2 for x in baseline) / (len(baseline) - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment) / (len(treatment) - 1)
        pooled_var = ((len(baseline) - 1) * var1 + (len(treatment) - 1) * var2) / (
            len(baseline) + len(treatment) - 2
        )
        backend = get_default_backend()
        pooled_std = sqrt_scalar(pooled_var, backend) if pooled_var > 0 else 1.0
        expected_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        assert abs(result.cohens_d - expected_d) <= _eps()

    def test_cohens_d_small_effect(self, runner: ThermoBenchmarkRunner) -> None:
        """Small difference should have small effect size."""
        # Use larger variance and smaller mean difference to get small effect
        baseline = [1.5, 2.0, 2.5, 2.0, 2.0]  # mean ~2.0, std ~0.35
        treatment = [1.65, 2.15, 2.65, 2.15, 2.15]  # mean ~2.15, std ~0.35
        # d ≈ 0.15 / 0.35 ≈ 0.43 (small effect)

        result = runner._compute_effect_size(baseline, treatment)

        mean1 = sum(baseline) / len(baseline)
        mean2 = sum(treatment) / len(treatment)
        var1 = sum((x - mean1) ** 2 for x in baseline) / (len(baseline) - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment) / (len(treatment) - 1)
        pooled_var = ((len(baseline) - 1) * var1 + (len(treatment) - 1) * var2) / (
            len(baseline) + len(treatment) - 2
        )
        backend = get_default_backend()
        pooled_std = sqrt_scalar(pooled_var, backend) if pooled_var > 0 else 1.0
        expected_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        assert abs(result.cohens_d - expected_d) <= _eps()

    def test_cohens_d_standard_error(self, runner: ThermoBenchmarkRunner) -> None:
        """Should compute standard error."""
        baseline = [2.0, 2.1, 1.9, 2.0, 2.0, 2.1]
        treatment = [3.0, 3.1, 2.9, 3.0, 3.0, 3.1]

        result = runner._compute_effect_size(baseline, treatment)

        mean1 = sum(baseline) / len(baseline)
        mean2 = sum(treatment) / len(treatment)
        var1 = sum((x - mean1) ** 2 for x in baseline) / (len(baseline) - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment) / (len(treatment) - 1)
        pooled_var = ((len(baseline) - 1) * var1 + (len(treatment) - 1) * var2) / (
            len(baseline) + len(treatment) - 2
        )
        backend = get_default_backend()
        pooled_std = sqrt_scalar(pooled_var, backend) if pooled_var > 0 else 1.0
        expected_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        se_d = sqrt_scalar(
            (len(baseline) + len(treatment)) / (len(baseline) * len(treatment))
            + expected_d**2 / (2 * (len(baseline) + len(treatment))),
            backend,
        )
        assert abs(result.standard_error - se_d) <= _eps()


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
        )

        assert isinstance(result.timestamp, datetime)
