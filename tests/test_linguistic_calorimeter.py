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
    BehavioralOutcome,
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
        assert result.sample_size == 3
        assert result.mean_entropy > 0
        assert result.std_entropy >= 0
        assert len(result.samples) == 3

    def test_track_generation_entropy_returns_trajectory(self) -> None:
        """Should return token-level entropy trajectory."""
        cal = LinguisticCalorimeter(simulated=True)

        result = cal.track_generation_entropy(
            prompt="Tell me a story.",
            max_tokens=20,
        )

        assert isinstance(result, EntropyTrajectory)
        assert len(result.token_entropies) > 0
        assert len(result.token_variances) > 0
        assert result.final_entropy > 0


class TestEntropyMeasurement:
    """Tests for EntropyMeasurement dataclass."""

    def test_entropy_trend_neutral_for_short(self) -> None:
        """Short trajectories should have neutral trend."""
        measurement = EntropyMeasurement(
            prompt="Test",
            first_token_entropy=2.0,
            mean_entropy=2.0,
            entropy_variance=0.1,
            entropy_trajectory=[2.0, 2.1],
            top_k_variance=0.5,
            token_count=2,
            generated_text="Hi",
        )

        from modelcypher.core.domain.thermo.linguistic_thermodynamics import EntropyDirection
        assert measurement.entropy_trend == EntropyDirection.NEUTRAL

    def test_entropy_trend_decreasing(self) -> None:
        """Should detect decreasing entropy trend."""
        measurement = EntropyMeasurement(
            prompt="Test",
            first_token_entropy=3.0,
            mean_entropy=2.0,
            entropy_variance=0.5,
            entropy_trajectory=[3.0, 2.8, 2.6, 2.4, 2.0, 1.8],
            top_k_variance=0.5,
            token_count=6,
            generated_text="Hi there",
        )

        from modelcypher.core.domain.thermo.linguistic_thermodynamics import EntropyDirection
        assert measurement.entropy_trend == EntropyDirection.DECREASE

    def test_entropy_trend_increasing(self) -> None:
        """Should detect increasing entropy trend."""
        measurement = EntropyMeasurement(
            prompt="Test",
            first_token_entropy=1.0,
            mean_entropy=2.5,
            entropy_variance=0.5,
            entropy_trajectory=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            top_k_variance=0.5,
            token_count=6,
            generated_text="Hi there",
        )

        from modelcypher.core.domain.thermo.linguistic_thermodynamics import EntropyDirection
        assert measurement.entropy_trend == EntropyDirection.INCREASE


class TestBaselineMeasurements:
    """Tests for BaselineMeasurements dataclass."""

    def test_percentile_thresholds(self) -> None:
        """Should compute percentile thresholds."""
        # Create mock samples with known entropy values
        samples = [
            EntropyMeasurement(
                prompt=f"Prompt {i}",
                first_token_entropy=float(i),
                mean_entropy=float(i),
                entropy_variance=0.1,
                entropy_trajectory=[float(i)],
                top_k_variance=0.5,
                token_count=1,
                generated_text="",
            )
            for i in range(1, 11)
        ]

        baseline = BaselineMeasurements(
            mean_entropy=5.5,
            std_entropy=2.87,
            samples=samples,
        )

        p10, p90 = baseline.percentile_thresholds()
        # For [1..10], 10th percentile ~ 1.9, 90th ~ 9.1
        assert 1.0 <= p10 <= 2.5
        assert 8.5 <= p90 <= 10.0

    def test_is_anomalous(self) -> None:
        """Should detect anomalous entropy values."""
        baseline = BaselineMeasurements(
            mean_entropy=2.0,
            std_entropy=0.5,
            samples=[],
        )

        # Within 2 std devs
        assert not baseline.is_anomalous(2.5)
        assert not baseline.is_anomalous(1.5)

        # Beyond 2 std devs
        assert baseline.is_anomalous(3.5)  # > mean + 2*std
        assert baseline.is_anomalous(0.5)  # < mean - 2*std


class TestEntropyTrajectory:
    """Tests for EntropyTrajectory dataclass."""

    def test_detect_phase_transition(self) -> None:
        """Should detect sudden entropy changes."""
        trajectory = EntropyTrajectory(
            prompt="Test",
            token_entropies=[2.0, 2.1, 2.0, 5.0, 5.2, 5.1],  # Jump at token 3
            token_variances=[0.5] * 6,
            generated_text="Test output",
        )

        transitions = trajectory.detect_phase_transitions(threshold=1.0)
        assert len(transitions) == 1
        assert transitions[0] == 3  # Index of transition

    def test_entropy_stability(self) -> None:
        """Should compute rolling stability measure."""
        # Stable trajectory
        stable = EntropyTrajectory(
            prompt="Test",
            token_entropies=[2.0, 2.01, 2.02, 2.01, 2.0],
            token_variances=[0.5] * 5,
            generated_text="Test",
        )

        # Unstable trajectory
        unstable = EntropyTrajectory(
            prompt="Test",
            token_entropies=[2.0, 3.0, 1.5, 4.0, 2.5],
            token_variances=[0.5] * 5,
            generated_text="Test",
        )

        assert stable.entropy_stability > unstable.entropy_stability
