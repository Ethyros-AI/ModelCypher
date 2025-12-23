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
        for trend in [EntropyDirection.INCREASE, EntropyDirection.DECREASE, EntropyDirection.NEUTRAL]:
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
