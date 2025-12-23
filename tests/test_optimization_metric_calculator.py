"""Tests for OptimizationMetricCalculator.

Tests the entropy dynamics measurement system that quantifies
how linguistic modifiers affect model behavior.
"""

import pytest
from datetime import datetime

from modelcypher.core.domain.dynamics.optimization_metric_calculator import (
    OptimizationMetricCalculator,
    OptimizationMetricConfig,
    OptimizationMeasurement,
    OptimizationResult,
)
from modelcypher.core.domain.entropy.entropy_tracker import ModelState


class TestOptimizationMetricConfig:
    """Tests for OptimizationMetricConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = OptimizationMetricConfig.default()
        assert config.temperature == 0.0
        assert config.max_tokens == 100
        assert config.top_k == 10
        assert config.capture_trajectory is True
        assert config.use_refusal_detector is True

    def test_custom_config(self):
        """Custom config should accept provided values."""
        config = OptimizationMetricConfig(
            temperature=0.7,
            max_tokens=200,
            top_k=50,
            capture_trajectory=False,
            use_refusal_detector=False,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 200
        assert config.top_k == 50
        assert config.capture_trajectory is False
        assert config.use_refusal_detector is False


class TestOptimizationMeasurement:
    """Tests for OptimizationMeasurement dataclass."""

    def test_measurement_fields(self):
        """Measurement should have expected fields."""
        measurement = OptimizationMeasurement(
            modifier="safety_prefix",
            full_prompt="Be careful: test prompt",
            response="Test response",
            mean_entropy=1.5,
            entropy_variance=0.1,
            first_token_entropy=1.2,
            delta_h=0.3,
            entropy_trajectory=[1.2, 1.5, 1.6],
        )

        assert measurement.modifier == "safety_prefix"
        assert measurement.mean_entropy == 1.5
        assert measurement.entropy_variance == 0.1
        assert measurement.first_token_entropy == 1.2
        assert measurement.delta_h == 0.3
        assert len(measurement.entropy_trajectory) == 3


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_result_baseline_property(self):
        """Result should find baseline measurement."""
        baseline = OptimizationMeasurement(
            modifier="baseline",
            full_prompt="test",
            response="response",
            mean_entropy=1.0,
            entropy_variance=0.1,
            first_token_entropy=1.0,
        )
        variant = OptimizationMeasurement(
            modifier="variant",
            full_prompt="modified test",
            response="modified response",
            mean_entropy=1.2,
            entropy_variance=0.15,
            first_token_entropy=1.1,
        )

        result = OptimizationResult(
            base_prompt="test",
            measurements=[baseline, variant],
        )

        assert result.baseline is not None
        assert result.baseline.modifier == "baseline"

    def test_result_no_baseline(self):
        """Result without baseline should return None."""
        variant = OptimizationMeasurement(
            modifier="variant",
            full_prompt="test",
            response="response",
            mean_entropy=1.0,
            entropy_variance=0.1,
            first_token_entropy=1.0,
        )

        result = OptimizationResult(
            base_prompt="test",
            measurements=[variant],
        )

        assert result.baseline is None


class TestOptimizationMetricCalculator:
    """Tests for OptimizationMetricCalculator."""

    @pytest.fixture
    def calculator(self):
        """Default calculator."""
        return OptimizationMetricCalculator()

    def test_calculate_statistics_empty(self, calculator):
        """Empty trajectory should return zeros."""
        stats = calculator.calculate_statistics([])
        assert stats["mean_entropy"] == 0
        assert stats["entropy_variance"] == 0
        assert stats["first_token_entropy"] == 0

    def test_calculate_statistics_single_value(self, calculator):
        """Single value trajectory should work."""
        stats = calculator.calculate_statistics([2.5])
        assert stats["mean_entropy"] == 2.5
        assert stats["entropy_variance"] == 0.0
        assert stats["first_token_entropy"] == 2.5

    def test_calculate_statistics_known_values(self, calculator):
        """Known trajectory should give expected statistics."""
        trajectory = [1.0, 2.0, 3.0]  # Mean = 2.0, Var = 1.0
        stats = calculator.calculate_statistics(trajectory)

        assert stats["mean_entropy"] == 2.0
        assert abs(stats["entropy_variance"] - 1.0) < 0.01
        assert stats["first_token_entropy"] == 1.0

    def test_calculate_statistics_uniform(self, calculator):
        """Uniform trajectory should have zero variance."""
        trajectory = [1.5, 1.5, 1.5, 1.5]
        stats = calculator.calculate_statistics(trajectory)

        assert stats["mean_entropy"] == 1.5
        assert stats["entropy_variance"] == 0.0
        assert stats["first_token_entropy"] == 1.5

    def test_measure_variant_basic(self, calculator):
        """measure_variant should compute all fields."""
        measurement = calculator.measure_variant(
            modifier="test_modifier",
            full_prompt="Test prompt",
            response="This is a long enough test response for classification.",
            entropy_trajectory=[1.0, 1.5, 2.0],
            model_state=ModelState.NOMINAL,
            baseline_entropy=None,
        )

        assert measurement.modifier == "test_modifier"
        assert measurement.full_prompt == "Test prompt"
        assert measurement.mean_entropy == 1.5
        assert measurement.first_token_entropy == 1.0
        assert measurement.delta_h is None  # No baseline

    def test_measure_variant_with_baseline(self, calculator):
        """measure_variant should compute delta_h with baseline."""
        measurement = calculator.measure_variant(
            modifier="variant",
            full_prompt="Modified prompt",
            response="This is a test response that exceeds minimum length threshold.",
            entropy_trajectory=[1.5, 1.8, 2.0],  # Mean = 1.77
            model_state=ModelState.NOMINAL,
            baseline_entropy=1.5,
        )

        assert measurement.delta_h is not None
        expected_delta = 1.7666666666666666 - 1.5
        assert abs(measurement.delta_h - expected_delta) < 0.01

    def test_measure_variant_captures_trajectory(self, calculator):
        """measure_variant should capture trajectory when configured."""
        trajectory = [1.0, 1.5, 2.0, 2.5]
        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="This is a long enough test response for classification.",
            entropy_trajectory=trajectory,
            model_state=ModelState.NOMINAL,
        )

        assert len(measurement.entropy_trajectory) == 4
        assert measurement.entropy_trajectory == trajectory

    def test_measure_variant_no_trajectory_capture(self):
        """measure_variant should not capture trajectory when disabled."""
        config = OptimizationMetricConfig(capture_trajectory=False)
        calculator = OptimizationMetricCalculator(config)

        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="This is a long enough test response for classification.",
            entropy_trajectory=[1.0, 1.5, 2.0],
            model_state=ModelState.NOMINAL,
        )

        assert len(measurement.entropy_trajectory) == 0

    def test_measure_variant_classifies_outcome(self, calculator):
        """measure_variant should classify the outcome."""
        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="I cannot help with that request. I must decline.",
            entropy_trajectory=[1.0, 1.2, 1.1],
            model_state=ModelState.NOMINAL,
        )

        assert measurement.outcome is not None
        # The refusal keywords should trigger refused outcome
        assert measurement.outcome.outcome.value == "refused"


class TestCalculateStatisticsEdgeCases:
    """Edge case tests for calculate_statistics."""

    @pytest.fixture
    def calculator(self):
        return OptimizationMetricCalculator()

    def test_very_large_values(self, calculator):
        """Should handle large entropy values."""
        trajectory = [1e10, 2e10, 3e10]
        stats = calculator.calculate_statistics(trajectory)

        assert stats["mean_entropy"] == 2e10
        assert stats["first_token_entropy"] == 1e10

    def test_very_small_values(self, calculator):
        """Should handle very small entropy values."""
        trajectory = [1e-10, 2e-10, 3e-10]
        stats = calculator.calculate_statistics(trajectory)

        assert stats["mean_entropy"] == 2e-10
        assert stats["first_token_entropy"] == 1e-10

    def test_negative_values(self, calculator):
        """Should handle negative values (edge case)."""
        trajectory = [-1.0, 0.0, 1.0]
        stats = calculator.calculate_statistics(trajectory)

        assert stats["mean_entropy"] == 0.0
        assert stats["first_token_entropy"] == -1.0


class TestMeasureVariantWithModelStates:
    """Tests for measure_variant with different model states."""

    @pytest.fixture
    def calculator(self):
        return OptimizationMetricCalculator()

    def test_halted_state(self, calculator):
        """HALTED state should affect classification."""
        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="This is a normal looking response without refusal patterns.",
            entropy_trajectory=[1.0, 1.2, 1.1],
            model_state=ModelState.HALTED,
        )

        assert measurement.outcome is not None
        # HALTED should contribute to refusal classification
        assert measurement.outcome.outcome.value == "refused"

    def test_distressed_state(self, calculator):
        """DISTRESSED state should affect classification."""
        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="This is a normal looking response without refusal patterns.",
            entropy_trajectory=[1.0, 1.2, 1.1],
            model_state=ModelState.DISTRESSED,
        )

        assert measurement.outcome is not None
        # DISTRESSED should contribute to refusal classification
        assert measurement.outcome.outcome.value == "refused"

    def test_confident_state(self, calculator):
        """CONFIDENT state with solution should be SOLVED."""
        measurement = calculator.measure_variant(
            modifier="test",
            full_prompt="Test",
            response="Here's how to solve this problem. Step 1: Configure settings. Step 2: Run the command.",
            entropy_trajectory=[0.5, 0.6, 0.5],  # Low entropy
            model_state=ModelState.CONFIDENT,
        )

        assert measurement.outcome is not None
        # Low entropy + solution indicators should be SOLVED
        assert measurement.outcome.outcome.value == "solved"
