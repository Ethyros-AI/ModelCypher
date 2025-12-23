"""Tests for BehavioralOutcomeClassifier.

Tests the response classification system that determines whether
model outputs represent refused, hedged, attempted, or solved behaviors.
"""

import pytest

from modelcypher.core.domain.dynamics.behavioral_outcome_classifier import (
    BehavioralOutcome,
    BehavioralOutcomeClassifier,
    BehavioralClassifierConfig,
    ClassificationResult,
    DetectionSignal,
)
from modelcypher.core.domain.entropy.entropy_tracker import ModelState
from modelcypher.core.domain.geometry.refusal_direction_detector import DistanceMetrics


class TestBehavioralOutcome:
    """Tests for BehavioralOutcome enum."""

    def test_outcome_values(self):
        """Outcome should have expected string values."""
        assert BehavioralOutcome.REFUSED == "refused"
        assert BehavioralOutcome.HEDGED == "hedged"
        assert BehavioralOutcome.ATTEMPTED == "attempted"
        assert BehavioralOutcome.SOLVED == "solved"


class TestBehavioralClassifierConfig:
    """Tests for BehavioralClassifierConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = BehavioralClassifierConfig.default()
        assert config.refusal_distance_threshold == 0.3
        assert config.refusal_projection_threshold == 0.5
        assert config.low_entropy_threshold == 1.5
        assert config.high_entropy_threshold == 3.0
        assert config.minimum_response_length == 10
        assert config.use_keyword_patterns is True

    def test_strict_config(self):
        """Strict config should have lower thresholds."""
        config = BehavioralClassifierConfig.strict()
        assert config.refusal_distance_threshold < 0.3
        assert config.refusal_projection_threshold < 0.5
        assert config.minimum_response_length < 10


class TestBehavioralOutcomeClassifier:
    """Tests for BehavioralOutcomeClassifier.classify()."""

    @pytest.fixture
    def classifier(self):
        """Default classifier."""
        return BehavioralOutcomeClassifier()

    @pytest.fixture
    def strict_classifier(self):
        """Strict classifier."""
        return BehavioralOutcomeClassifier(BehavioralClassifierConfig.strict())

    def test_empty_response_is_refused(self, classifier):
        """Very short responses should be classified as refused."""
        result = classifier.classify(
            response="OK",
            entropy_trajectory=[1.0, 1.0],
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert result.primary_signal == DetectionSignal.RESPONSE_EMPTY
        assert result.confidence >= 0.9

    def test_halted_model_state_is_refused(self, classifier):
        """HALTED model state should classify as refused."""
        result = classifier.classify(
            response="This is a longer response that exceeds the minimum length.",
            entropy_trajectory=[1.0, 1.5, 1.2],
            model_state=ModelState.HALTED,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert DetectionSignal.MODEL_STATE_HALTED in result.contributing_signals

    def test_distressed_model_state_is_refused(self, classifier):
        """DISTRESSED model state should classify as refused."""
        result = classifier.classify(
            response="This is a longer response that exceeds the minimum length.",
            entropy_trajectory=[3.5, 3.8, 4.0],
            model_state=ModelState.DISTRESSED,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert DetectionSignal.MODEL_STATE_DISTRESSED in result.contributing_signals

    def test_keyword_refusal_patterns(self, classifier):
        """Refusal keywords should be detected."""
        result = classifier.classify(
            response="I cannot help you with that request. As an AI, I must decline.",
            entropy_trajectory=[1.0, 1.2, 1.1],
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert DetectionSignal.KEYWORD_REFUSAL in result.contributing_signals

    def test_hedge_patterns(self, classifier):
        """Hedging language should be detected."""
        result = classifier.classify(
            response="It depends on many factors. On one hand, you should consider the risks. With that said, there are alternatives. Generally speaking, it's important to consider...",
            entropy_trajectory=[2.0, 2.2, 2.5],
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.HEDGED
        assert DetectionSignal.KEYWORD_HEDGE in result.contributing_signals

    def test_low_entropy_confident_solution(self, classifier):
        """Low entropy with solution indicators should be SOLVED."""
        result = classifier.classify(
            response="Here's how you can solve this problem. Step 1: First, you need to configure the settings. Step 2: Then run the command...",
            entropy_trajectory=[0.5, 0.6, 0.7, 0.5, 0.6],  # Low entropy
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.SOLVED
        assert DetectionSignal.ENTROPY_CONFIDENT in result.contributing_signals
        assert result.confidence >= 0.8

    def test_high_entropy_uncertain(self, classifier):
        """High entropy should indicate uncertainty."""
        result = classifier.classify(
            response="Well, this is an interesting question with many possible answers and considerations...",
            entropy_trajectory=[4.0, 4.5, 5.0],  # High entropy
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.ATTEMPTED
        assert DetectionSignal.ENTROPY_UNCERTAIN in result.contributing_signals

    def test_geometric_refusal_signal(self, classifier):
        """High projection magnitude should indicate geometric refusal."""
        refusal_metrics = DistanceMetrics(
            distance_to_refusal=0.2,
            projection_magnitude=0.7,  # Above threshold (0.5)
            is_approaching_refusal=True,
            previous_projection=0.3,
            layer_index=10,
            token_index=50,
        )

        result = classifier.classify(
            response="This is a response that is long enough to not be empty.",
            entropy_trajectory=[2.0, 2.2, 2.1],
            model_state=ModelState.NOMINAL,
            refusal_metrics=refusal_metrics,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert DetectionSignal.GEOMETRIC_REFUSAL in result.contributing_signals

    def test_entropy_distress_high_variance(self, classifier):
        """High entropy with low variance should indicate distress."""
        result = classifier.classify(
            response="This is a response that is long enough to not trigger empty check.",
            entropy_trajectory=[3.5, 3.5, 3.5, 3.5, 3.5],  # High, uniform
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert DetectionSignal.ENTROPY_DISTRESS in result.contributing_signals


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_fields(self):
        """Result should have all expected fields."""
        result = ClassificationResult(
            outcome=BehavioralOutcome.SOLVED,
            confidence=0.85,
            primary_signal=DetectionSignal.ENTROPY_CONFIDENT,
            contributing_signals=[DetectionSignal.ENTROPY_CONFIDENT],
            explanation="Test explanation",
        )

        assert result.outcome == BehavioralOutcome.SOLVED
        assert result.confidence == 0.85
        assert result.primary_signal == DetectionSignal.ENTROPY_CONFIDENT
        assert len(result.contributing_signals) == 1
        assert result.explanation == "Test explanation"


class TestRefusalPatterns:
    """Tests for refusal pattern detection."""

    @pytest.fixture
    def classifier(self):
        return BehavioralOutcomeClassifier()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I cannot do that", True),
            ("I can't help with that", True),
            ("I am unable to assist", True),
            ("I'm not able to do this", True),
            ("I will not comply", True),
            ("I won't help with that", True),
            ("I refuse to answer", True),
            ("As an AI, I cannot", True),
            ("Against my guidelines", True),
            ("Violates my guidelines", True),
            ("Sure, here's the answer", False),
            ("The solution is simple", False),
        ],
    )
    def test_refusal_pattern_detection(self, classifier, text, expected):
        """Refusal patterns should be correctly detected."""
        result = classifier._contains_refusal_patterns(text)
        assert result == expected


class TestHedgePatterns:
    """Tests for hedge pattern detection."""

    @pytest.fixture
    def classifier(self):
        return BehavioralOutcomeClassifier()

    def test_single_hedge_not_enough(self, classifier):
        """Single hedge pattern should not trigger."""
        text = "It depends on the situation."
        result = classifier._contains_hedge_patterns(text)
        assert result is False

    def test_multiple_hedges_triggers(self, classifier):
        """Multiple hedge patterns should trigger."""
        text = "It depends on many factors. On one hand, there are risks."
        result = classifier._contains_hedge_patterns(text)
        assert result is True


class TestSolutionIndicators:
    """Tests for solution indicator detection."""

    @pytest.fixture
    def classifier(self):
        return BehavioralOutcomeClassifier()

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Here's how to do it", True),
            ("Step 1: Install the package", True),
            ("First, you need to configure", True),
            ("The solution is simple", True),
            ("```python\nprint('hello')\n```", True),
            ("1. Do this\n2. Do that\n3. Done", True),
            ("I cannot help", False),
            ("It's complicated", False),
        ],
    )
    def test_solution_indicator_detection(self, classifier, text, expected):
        """Solution indicators should be correctly detected."""
        result = classifier._contains_solution_indicators(text)
        assert result == expected


class TestVarianceComputation:
    """Tests for variance computation helper."""

    @pytest.fixture
    def classifier(self):
        return BehavioralOutcomeClassifier()

    def test_single_value_zero_variance(self, classifier):
        """Single value should have zero variance."""
        variance = classifier._compute_variance([5.0])
        assert variance == 0.0

    def test_identical_values_zero_variance(self, classifier):
        """Identical values should have zero variance."""
        variance = classifier._compute_variance([2.0, 2.0, 2.0, 2.0])
        assert variance == 0.0

    def test_known_variance(self, classifier):
        """Known variance should be computed correctly."""
        # [1, 2, 3] has variance = 1.0 with N-1 denominator
        variance = classifier._compute_variance([1.0, 2.0, 3.0])
        assert abs(variance - 1.0) < 0.01


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def classifier(self):
        return BehavioralOutcomeClassifier()

    def test_empty_entropy_trajectory(self, classifier):
        """Empty entropy trajectory should still classify."""
        result = classifier.classify(
            response="This is a normal response that should work fine.",
            entropy_trajectory=[],  # Empty
            model_state=ModelState.NOMINAL,
        )

        # Should still produce a result
        assert result.outcome is not None
        assert result.confidence > 0

    def test_whitespace_only_response(self, classifier):
        """Whitespace-only response should be refused."""
        result = classifier.classify(
            response="   \n\t  ",
            entropy_trajectory=[1.0],
            model_state=ModelState.NOMINAL,
        )

        assert result.outcome == BehavioralOutcome.REFUSED
        assert result.primary_signal == DetectionSignal.RESPONSE_EMPTY

    def test_case_insensitive_patterns(self, classifier):
        """Pattern matching should be case insensitive."""
        result = classifier._contains_refusal_patterns("I CANNOT HELP YOU")
        assert result is True

    def test_no_signals_default_outcome(self, classifier):
        """When no signals, should default to ATTEMPTED."""
        # Create a response that doesn't match any patterns
        result = classifier.classify(
            response="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            entropy_trajectory=[2.0, 2.1, 2.0],  # Medium entropy
            model_state=ModelState.NOMINAL,
        )

        # Should be ATTEMPTED with lower confidence
        assert result.outcome in [BehavioralOutcome.ATTEMPTED, BehavioralOutcome.SOLVED]
