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

"""Tests for LinguisticCalorimeter."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.thermo.linguistic_calorimeter import (
    BaselineMeasurements,
    EntropyMeasurement,
    EntropyTrajectory,
    LinguisticCalorimeter,
)
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    EntropyDirection,
    LinguisticModifier,
    PromptLanguage,
)


class TestLinguisticCalorimeterSimulated:
    """Tests for simulated calorimeter mode."""

    def test_create_simulated_calorimeter(self) -> None:
        """Should create a simulated calorimeter."""
        cal = LinguisticCalorimeter(simulated=True)
        assert cal.simulated is True
        assert cal.model_path is None

    def test_measure_entropy_returns_measurement(self) -> None:
        """Should return an EntropyMeasurement."""
        cal = LinguisticCalorimeter(simulated=True)
        result = cal.measure_entropy("What is 2+2?", temperature=1.0)

        assert isinstance(result, EntropyMeasurement)
        assert result.prompt == "What is 2+2?"
        assert result.mean_entropy > 0
        assert result.first_token_entropy > 0
        assert len(result.entropy_trajectory) > 0
        assert result.token_count > 0

    def test_measure_entropy_temperature_affects_entropy(self) -> None:
        """Higher temperature should yield higher entropy in simulation."""
        cal = LinguisticCalorimeter(simulated=True)

        low_temp = cal.measure_entropy("Test prompt", temperature=0.5)
        high_temp = cal.measure_entropy("Test prompt", temperature=2.0)

        # Higher temperature should produce higher entropy
        assert high_temp.mean_entropy > low_temp.mean_entropy

    def test_measure_with_modifiers_returns_list(self) -> None:
        """Should return measurements for each modifier."""
        cal = LinguisticCalorimeter(simulated=True)
        modifiers = [
            LinguisticModifier.BASELINE,
            LinguisticModifier.CAPS,
            LinguisticModifier.URGENT,
        ]

        results = cal.measure_with_modifiers(
            prompt="What is 2+2?",
            modifiers=modifiers,
        )

        assert len(results) == 3
        assert all(r.modifier in modifiers for r in results)

    def test_measure_with_modifiers_includes_baseline_comparison(self) -> None:
        """Non-baseline modifiers should have delta_h calculated."""
        cal = LinguisticCalorimeter(simulated=True)
        modifiers = [
            LinguisticModifier.BASELINE,
            LinguisticModifier.CAPS,
        ]

        results = cal.measure_with_modifiers(
            prompt="What is 2+2?",
            modifiers=modifiers,
        )

        baseline = next(r for r in results if r.modifier == LinguisticModifier.BASELINE)
        caps = next(r for r in results if r.modifier == LinguisticModifier.CAPS)

        assert baseline.delta_h is None  # Baseline has no delta
        assert caps.delta_h is not None

    def test_measure_with_modifiers_multilingual(self) -> None:
        """Should use localized modifiers for non-English languages."""
        cal = LinguisticCalorimeter(simulated=True)
        modifiers = [LinguisticModifier.BASELINE, LinguisticModifier.URGENT]

        results = cal.measure_with_modifiers(
            prompt="What is 2+2?",
            modifiers=modifiers,
            language=PromptLanguage.CHINESE,
        )

        assert len(results) == 2
        # Verify Chinese modifier was applied (contains Chinese text)
        urgent = next(r for r in results if r.modifier == LinguisticModifier.URGENT)
        assert "紧急" in urgent.prompt.full_prompt  # Chinese for "urgent"

    def test_establish_baseline_returns_stats(self) -> None:
        """Should compute baseline statistics from corpus."""
        cal = LinguisticCalorimeter(simulated=True)
        corpus = [
            "What is 2+2?",
            "Explain photosynthesis.",
            "Write a poem.",
        ]

        result = cal.establish_baseline(corpus)

        assert isinstance(result, BaselineMeasurements)
        assert result.corpus_size == 3
        assert result.mean_first_token_entropy > 0
        assert result.mean_generation_entropy > 0
        assert result.std_first_token_entropy >= 0
        assert result.std_generation_entropy >= 0
        assert 50 in result.percentiles

    def test_track_generation_entropy_returns_trajectory(self) -> None:
        """Should return token-level entropy trajectory."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.track_generation_entropy(
            prompt="Tell me a story.",
            max_tokens=20,
        )

        assert isinstance(result, EntropyTrajectory)
        assert len(result.per_token_entropy) > 0
        assert len(result.per_token_variance) > 0
        assert len(result.tokens) > 0
        assert len(result.cumulative_entropy) > 0
        assert isinstance(result.entropy_trend, EntropyDirection)


class TestEntropyMeasurement:
    """Tests for EntropyMeasurement dataclass."""

    def test_measurement_fields(self) -> None:
        """Should hold all required fields."""
        from datetime import datetime

        measurement = EntropyMeasurement(
            prompt="Test",
            first_token_entropy=2.0,
            mean_entropy=2.0,
            entropy_variance=0.1,
            entropy_trajectory=[2.0, 2.1],
            top_k_concentration=0.5,
            token_count=2,
            generated_text="Hi",
            stop_reason="length",
            temperature=1.0,
            measurement_time=0.5,
        )

        assert measurement.prompt == "Test"
        assert measurement.first_token_entropy == 2.0
        assert measurement.mean_entropy == 2.0
        assert measurement.token_count == 2
        assert isinstance(measurement.timestamp, datetime)

    def test_entropy_trajectory_stored(self) -> None:
        """Should store and retrieve entropy trajectory."""
        trajectory = [2.0, 2.1, 2.2, 2.0, 1.9]
        measurement = EntropyMeasurement(
            prompt="Test",
            first_token_entropy=trajectory[0],
            mean_entropy=sum(trajectory) / len(trajectory),
            entropy_variance=0.1,
            entropy_trajectory=trajectory,
            top_k_concentration=0.5,
            token_count=len(trajectory),
            generated_text="Output",
            stop_reason="stop",
            temperature=1.0,
            measurement_time=0.3,
        )

        assert len(measurement.entropy_trajectory) == 5
        assert measurement.entropy_trajectory[0] == 2.0


class TestBaselineMeasurements:
    """Tests for BaselineMeasurements dataclass."""

    def test_baseline_fields(self) -> None:
        """Should hold all required fields."""
        baseline = BaselineMeasurements(
            corpus_size=10,
            mean_first_token_entropy=2.5,
            std_first_token_entropy=0.3,
            mean_generation_entropy=2.2,
            std_generation_entropy=0.25,
            percentiles={25: 2.0, 50: 2.2, 75: 2.4, 95: 2.8},
        )

        assert baseline.corpus_size == 10
        assert baseline.mean_first_token_entropy == 2.5
        assert baseline.std_first_token_entropy == 0.3
        assert baseline.percentiles[50] == 2.2

    def test_percentiles_stored(self) -> None:
        """Should store percentile values."""
        percentiles = {25: 1.5, 50: 2.0, 75: 2.5, 95: 3.5}
        baseline = BaselineMeasurements(
            corpus_size=100,
            mean_first_token_entropy=2.0,
            std_first_token_entropy=0.5,
            mean_generation_entropy=2.0,
            std_generation_entropy=0.4,
            percentiles=percentiles,
        )

        assert baseline.percentiles[25] == 1.5
        assert baseline.percentiles[95] == 3.5


class TestEntropyTrajectory:
    """Tests for EntropyTrajectory dataclass."""

    def test_trajectory_fields(self) -> None:
        """Should hold all required fields."""
        trajectory = EntropyTrajectory(
            prompt="Test",
            per_token_entropy=[2.0, 2.1, 2.0, 1.9, 1.8],
            per_token_variance=[0.5, 0.4, 0.3, 0.2, 0.1],
            tokens=["token_0", "token_1", "token_2", "token_3", "token_4"],
            cumulative_entropy=[2.0, 2.05, 2.03, 2.0, 1.96],
            entropy_trend=EntropyDirection.DECREASE,
            inflection_points=[2],
        )

        assert trajectory.prompt == "Test"
        assert len(trajectory.per_token_entropy) == 5
        assert trajectory.entropy_trend == EntropyDirection.DECREASE

    def test_inflection_points_stored(self) -> None:
        """Should store inflection point indices."""
        trajectory = EntropyTrajectory(
            prompt="Test",
            per_token_entropy=[2.0, 2.5, 2.0, 2.5, 2.0],  # Oscillating
            per_token_variance=[0.1] * 5,
            tokens=[f"t{i}" for i in range(5)],
            cumulative_entropy=[2.0, 2.25, 2.17, 2.25, 2.2],
            entropy_trend=EntropyDirection.NEUTRAL,
            inflection_points=[1, 2, 3],
        )

        assert len(trajectory.inflection_points) == 3
        assert 1 in trajectory.inflection_points

    def test_entropy_trend_values(self) -> None:
        """Should accept valid entropy direction values."""
        for trend in [
            EntropyDirection.INCREASE,
            EntropyDirection.DECREASE,
            EntropyDirection.NEUTRAL,
        ]:
            trajectory = EntropyTrajectory(
                prompt="Test",
                per_token_entropy=[2.0],
                per_token_variance=[0.1],
                tokens=["t0"],
                cumulative_entropy=[2.0],
                entropy_trend=trend,
                inflection_points=[],
            )
            assert trajectory.entropy_trend == trend


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestEntropyMathInvariants:
    """Tests for mathematical invariants in entropy computation.

    These tests verify that the calorimeter's computations satisfy
    fundamental mathematical properties, even in simulated mode.
    """

    def test_entropy_always_non_negative(self) -> None:
        """Entropy should always be >= 0.

        Mathematical property: Shannon entropy H = -Σ p log(p) ≥ 0
        """
        cal = LinguisticCalorimeter(simulated=True)

        for prompt in ["", "a", "test", "A" * 1000, "!@#$%^&*()"]:
            result = cal.measure_entropy(prompt)
            assert result.mean_entropy >= 0, f"Entropy negative for '{prompt[:20]}'"
            assert result.first_token_entropy >= 0

    def test_variance_always_non_negative(self) -> None:
        """Entropy variance should always be >= 0.

        Mathematical property: Variance = E[(X - μ)²] ≥ 0
        """
        cal = LinguisticCalorimeter(simulated=True)

        for prompt in ["test", "another test", "A" * 500]:
            result = cal.measure_entropy(prompt)
            assert result.entropy_variance >= 0

    def test_trajectory_entropy_all_positive(self) -> None:
        """All entropy values in trajectory should be > 0."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.measure_entropy("Test prompt", max_tokens=50)

        for i, entropy in enumerate(result.entropy_trajectory):
            assert entropy > 0, f"Trajectory[{i}] = {entropy} is not positive"

    def test_mean_entropy_is_trajectory_average(self) -> None:
        """Mean entropy should equal the average of the trajectory."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.measure_entropy("Test prompt", max_tokens=20)

        if result.entropy_trajectory:
            expected_mean = sum(result.entropy_trajectory) / len(result.entropy_trajectory)
            assert result.mean_entropy == pytest.approx(expected_mean, rel=1e-6)

    def test_first_token_entropy_matches_trajectory_start(self) -> None:
        """First token entropy should match trajectory[0]."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.measure_entropy("Test prompt")

        if result.entropy_trajectory:
            assert result.first_token_entropy == pytest.approx(
                result.entropy_trajectory[0], rel=1e-6
            )


class TestVarianceComputation:
    """Tests for variance computation correctness."""

    def test_variance_formula_correctness(self) -> None:
        """Variance should follow the sample variance formula.

        Var = Σ(x - mean)² / (n - 1)
        """
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.measure_entropy("Test prompt", max_tokens=10)

        if len(result.entropy_trajectory) > 1:
            # Manually compute variance
            mean = sum(result.entropy_trajectory) / len(result.entropy_trajectory)
            squared_diff_sum = sum((e - mean) ** 2 for e in result.entropy_trajectory)
            expected_var = squared_diff_sum / (len(result.entropy_trajectory) - 1)

            assert result.entropy_variance == pytest.approx(expected_var, rel=1e-6)

    def test_single_point_variance_is_zero(self) -> None:
        """Variance of single point should be 0."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.measure_entropy("Test", max_tokens=1)

        # With only 1 token, variance should be 0
        assert result.entropy_variance == 0.0


class TestBaselineStatistics:
    """Tests for baseline statistics computation."""

    def test_baseline_mean_is_corpus_average(self) -> None:
        """Mean should equal average of individual measurements."""
        cal = LinguisticCalorimeter(simulated=True)
        corpus = ["prompt one", "prompt two", "prompt three"]

        # Measure individually
        measurements = [cal.measure_entropy(p) for p in corpus]
        expected_mean = sum(m.mean_entropy for m in measurements) / len(measurements)

        # Clear cache and compute baseline
        cal._baseline_cache.clear()
        baseline = cal.establish_baseline(corpus)

        assert baseline.mean_generation_entropy == pytest.approx(expected_mean, rel=1e-6)

    def test_baseline_std_is_population_std(self) -> None:
        """Std dev should follow the population formula.

        std = sqrt(Σ(x - mean)² / n)
        """
        import math

        cal = LinguisticCalorimeter(simulated=True)
        corpus = ["a", "bb", "ccc", "dddd", "eeeee"]

        measurements = [cal.measure_entropy(p) for p in corpus]
        entropies = [m.mean_entropy for m in measurements]

        mean = sum(entropies) / len(entropies)
        expected_std = math.sqrt(sum((e - mean) ** 2 for e in entropies) / len(entropies))

        cal._baseline_cache.clear()
        baseline = cal.establish_baseline(corpus)

        assert baseline.std_generation_entropy == pytest.approx(expected_std, rel=1e-6)

    def test_percentile_ordering(self) -> None:
        """Percentiles should be monotonically increasing."""
        cal = LinguisticCalorimeter(simulated=True)
        corpus = [f"prompt {i}" for i in range(20)]

        baseline = cal.establish_baseline(corpus)

        assert baseline.percentiles[25] <= baseline.percentiles[50]
        assert baseline.percentiles[50] <= baseline.percentiles[75]
        assert baseline.percentiles[75] <= baseline.percentiles[95]

    def test_baseline_empty_corpus_raises(self) -> None:
        """Empty corpus should raise ValueError."""
        cal = LinguisticCalorimeter(simulated=True)

        with pytest.raises(ValueError, match="Corpus cannot be empty"):
            cal.establish_baseline([])


class TestTrajectoryAnalysis:
    """Tests for entropy trajectory analysis."""

    def test_cumulative_entropy_is_running_average(self) -> None:
        """Cumulative entropy should be running average.

        cumulative[i] = sum(trajectory[0:i+1]) / (i + 1)
        """
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.track_generation_entropy("Test prompt", max_tokens=10)

        for i, cum in enumerate(result.cumulative_entropy):
            expected = sum(result.per_token_entropy[: i + 1]) / (i + 1)
            assert cum == pytest.approx(expected, rel=1e-6)

    def test_per_token_variance_is_sliding_window(self) -> None:
        """Per-token variance should use 3-token sliding window."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.track_generation_entropy("Test prompt", max_tokens=10)

        window_size = 3
        for i in range(len(result.per_token_variance)):
            start = max(0, i - window_size + 1)
            window = result.per_token_entropy[start : i + 1]

            if len(window) > 1:
                mean_w = sum(window) / len(window)
                expected_var = sum((x - mean_w) ** 2 for x in window) / len(window)
            else:
                expected_var = 0.0

            assert result.per_token_variance[i] == pytest.approx(expected_var, rel=1e-6)

    def test_trend_detection_increasing(self) -> None:
        """Should detect INCREASE trend when second half > first half."""
        trajectory = EntropyTrajectory(
            prompt="Test",
            per_token_entropy=[1.0, 1.1, 1.2, 2.0, 2.1, 2.2],  # Clear increase
            per_token_variance=[0.1] * 6,
            tokens=[f"t{i}" for i in range(6)],
            cumulative_entropy=[1.0, 1.05, 1.1, 1.325, 1.48, 1.6],
            entropy_trend=EntropyDirection.INCREASE,
            inflection_points=[],
        )

        # Verify the trend is INCREASE
        first_half = trajectory.per_token_entropy[:3]
        second_half = trajectory.per_token_entropy[3:]
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        assert second_mean > first_mean + 0.1
        assert trajectory.entropy_trend == EntropyDirection.INCREASE

    def test_trend_detection_decreasing(self) -> None:
        """Should detect DECREASE trend when second half < first half."""
        cal = LinguisticCalorimeter(simulated=True)

        # Simulated mode has built-in decay (cooling effect)
        result = cal.track_generation_entropy("Test prompt", max_tokens=40)

        # With decay, second half should have lower mean
        mid = len(result.per_token_entropy) // 2
        first_mean = sum(result.per_token_entropy[:mid]) / mid if mid > 0 else 0
        second_mean = (
            sum(result.per_token_entropy[mid:]) / len(result.per_token_entropy[mid:])
            if mid < len(result.per_token_entropy)
            else 0
        )

        # The simulated mode has decay, so second half should be lower
        if first_mean - second_mean > 0.1:
            assert result.entropy_trend == EntropyDirection.DECREASE

    def test_inflection_points_are_valid_indices(self) -> None:
        """Inflection points should be valid indices in trajectory."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.track_generation_entropy("Test prompt", max_tokens=30)

        for inflection in result.inflection_points:
            assert 0 < inflection < len(result.per_token_entropy) - 1


class TestBehavioralClassification:
    """Tests for behavioral outcome classification."""

    def test_low_entropy_is_solved(self) -> None:
        """Low entropy should classify as SOLVED."""
        from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
            BehavioralOutcome,
        )

        cal = LinguisticCalorimeter(simulated=True)

        # Manual classification test
        outcome = cal._classify_outcome(entropy=2.0, variance=0.1)
        assert outcome == BehavioralOutcome.SOLVED

    def test_high_entropy_low_variance_is_refused(self) -> None:
        """High entropy + low variance should be REFUSED."""
        from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
            BehavioralOutcome,
        )

        cal = LinguisticCalorimeter(simulated=True)

        outcome = cal._classify_outcome(entropy=4.5, variance=0.05)
        assert outcome == BehavioralOutcome.REFUSED

    def test_high_entropy_high_variance_is_hedged(self) -> None:
        """High entropy + high variance should be HEDGED."""
        from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
            BehavioralOutcome,
        )

        cal = LinguisticCalorimeter(simulated=True)

        outcome = cal._classify_outcome(entropy=4.5, variance=0.5)
        assert outcome == BehavioralOutcome.HEDGED

    def test_medium_entropy_is_attempted(self) -> None:
        """Medium entropy should be ATTEMPTED."""
        from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
            BehavioralOutcome,
        )

        cal = LinguisticCalorimeter(simulated=True)

        outcome = cal._classify_outcome(entropy=3.5, variance=0.2)
        assert outcome == BehavioralOutcome.ATTEMPTED


class TestModelStateClassification:
    """Tests for model state classification."""

    def test_very_low_entropy_is_confident(self) -> None:
        """Entropy < 1.5 should be 'confident'."""
        cal = LinguisticCalorimeter(simulated=True)

        state = cal._classify_model_state(1.0)
        assert state == "confident"

    def test_low_entropy_is_normal(self) -> None:
        """Entropy 1.5-3.0 should be 'normal'."""
        cal = LinguisticCalorimeter(simulated=True)

        state = cal._classify_model_state(2.0)
        assert state == "normal"

    def test_medium_entropy_is_uncertain(self) -> None:
        """Entropy 3.0-4.0 should be 'uncertain'."""
        cal = LinguisticCalorimeter(simulated=True)

        state = cal._classify_model_state(3.5)
        assert state == "uncertain"

    def test_high_entropy_is_distressed(self) -> None:
        """Entropy >= 4.0 should be 'distressed'."""
        cal = LinguisticCalorimeter(simulated=True)

        state = cal._classify_model_state(5.0)
        assert state == "distressed"


class TestTemperatureEffects:
    """Tests for temperature effects on entropy."""

    def test_higher_temperature_increases_entropy(self) -> None:
        """Higher temperature should yield higher entropy.

        Physical property: T↑ → S↑ (higher temp = more disorder)
        """
        cal = LinguisticCalorimeter(simulated=True)

        results = []
        for temp in [0.1, 0.5, 1.0, 2.0, 3.0]:
            result = cal.measure_entropy("Test prompt", temperature=temp)
            results.append((temp, result.mean_entropy))

        # Verify monotonic increase
        for i in range(len(results) - 1):
            assert results[i + 1][1] >= results[i][1], (
                f"Entropy should increase with temp: "
                f"T={results[i][0]}→{results[i + 1][0]}, "
                f"H={results[i][1]:.2f}→{results[i + 1][1]:.2f}"
            )

    def test_temperature_effect_magnitude(self) -> None:
        """Temperature change should produce measurable entropy change.

        Simulated mode uses: temp_effect = (temperature - 1.0) * 0.5
        """
        cal = LinguisticCalorimeter(simulated=True)

        low = cal.measure_entropy("Test", temperature=0.5)
        high = cal.measure_entropy("Test", temperature=2.0)

        # Delta should be approximately (2.0 - 0.5) * 0.5 = 0.75
        delta = high.mean_entropy - low.mean_entropy
        assert 0.5 < delta < 1.0


class TestPropertyBasedInvariants:
    """Property-based tests using hypothesis."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "",
            "a",
            "test",
            "A" * 100,
            "Hello, world!",
            "12345",
            "!@#$%",
            "Mixed 123 !@#",
        ],
    )
    def test_entropy_positive_for_all_prompts(self, prompt: str) -> None:
        """Entropy should be positive for any prompt."""
        cal = LinguisticCalorimeter(simulated=True)
        result = cal.measure_entropy(prompt)

        assert result.mean_entropy > 0
        assert result.first_token_entropy > 0

    @pytest.mark.parametrize("max_tokens", [1, 5, 10, 20, 50])
    def test_trajectory_length_matches_max_tokens(self, max_tokens: int) -> None:
        """Trajectory length should match (or be <= ) max_tokens."""
        cal = LinguisticCalorimeter(simulated=True)
        result = cal.measure_entropy("Test", max_tokens=max_tokens)

        # In simulated mode: trajectory_len = min(max_tokens, 20)
        expected_len = min(max_tokens, 20)
        assert len(result.entropy_trajectory) == expected_len

    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 1.5, 2.0])
    def test_all_temperatures_produce_valid_output(self, temp: float) -> None:
        """All temperature values should produce valid measurements."""
        cal = LinguisticCalorimeter(simulated=True)
        result = cal.measure_entropy("Test prompt", temperature=temp)

        assert result.mean_entropy > 0
        assert result.entropy_variance >= 0
        assert result.temperature == temp
