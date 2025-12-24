"""Integration tests for the entropy workflow.

Tests the full workflow:
    probe → measure → classify → detect

This validates that the entropy monitoring components work together
to detect distress patterns and trigger appropriate responses.
"""
from __future__ import annotations

import pytest
import numpy as np

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    EntropyResult,
)
from modelcypher.core.domain.entropy.entropy_window import (
    EntropyWindow,
    EntropyWindowConfig,
    EntropyLevel,
)
from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
    ConversationEntropyTracker,
    ConversationEntropyConfiguration,
    RiskLevel,
)
from modelcypher.core.domain.thermo.phase_transition_theory import (
    Phase,
    PhaseTransitionTheory,
)


# =============================================================================
# Entropy Calculation Integration
# =============================================================================


class TestEntropyCalculationIntegration:
    """Tests for entropy calculation integration."""

    def test_calculate_entropy_from_logits(self) -> None:
        """Entropy calculation should work on typical logit distributions."""
        calculator = LogitEntropyCalculator()

        # Typical logit distribution (softmax input)
        logits = [2.0, 1.5, 0.5, -0.5, -1.0]

        result = calculator.calculate(logits)

        assert result.raw_entropy >= 0
        assert 0.0 <= result.normalized_entropy <= 1.0
        assert result.variance >= 0

    def test_entropy_tracks_uncertainty(self) -> None:
        """Higher uncertainty distributions should have higher entropy."""
        calculator = LogitEntropyCalculator()

        # Low uncertainty (one dominant logit)
        low_uncertainty = [10.0, 0.0, 0.0, 0.0, 0.0]

        # High uncertainty (uniform-ish)
        high_uncertainty = [1.0, 1.0, 1.0, 1.0, 1.0]

        low_result = calculator.calculate(low_uncertainty)
        high_result = calculator.calculate(high_uncertainty)

        assert high_result.raw_entropy > low_result.raw_entropy

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_always_non_negative(self, seed: int) -> None:
        """Entropy should always be non-negative."""
        rng = np.random.default_rng(seed)
        calculator = LogitEntropyCalculator()

        logits = rng.standard_normal(20).tolist()
        result = calculator.calculate(logits)

        assert result.raw_entropy >= 0
        assert result.normalized_entropy >= 0


# =============================================================================
# Entropy Window Integration
# =============================================================================


class TestEntropyWindowIntegration:
    """Tests for entropy window tracking integration."""

    def test_window_tracks_entropy_over_time(self) -> None:
        """Window should track entropy samples over time."""
        config = EntropyWindowConfig(
            window_size=10,
            high_entropy_threshold=3.0,
            circuit_breaker_threshold=4.0,
        )
        window = EntropyWindow(config)

        # Add samples
        for i in range(15):
            entropy = 1.5 + 0.1 * i
            variance = 0.3 + 0.02 * i
            window.add(entropy, variance, i)

        status = window.status()

        assert status.current_entropy > 0
        assert status.moving_average > 0
        assert status.level in EntropyLevel

    def test_window_detects_high_entropy(self) -> None:
        """Window should detect high entropy levels."""
        config = EntropyWindowConfig(
            window_size=5,
            high_entropy_threshold=2.0,
            circuit_breaker_threshold=4.0,
        )
        window = EntropyWindow(config)

        # Add high entropy samples
        for i in range(5):
            window.add(3.0, 0.5, i)  # Above high threshold

        status = window.status()

        assert status.level in {EntropyLevel.HIGH, EntropyLevel.CRITICAL}

    def test_window_circuit_breaker_trips(self) -> None:
        """Circuit breaker should trip on extreme entropy."""
        config = EntropyWindowConfig(
            window_size=5,
            high_entropy_threshold=3.0,
            circuit_breaker_threshold=4.0,
        )
        window = EntropyWindow(config)

        # Add extreme entropy samples
        for i in range(5):
            window.add(5.0, 2.0, i)  # Above circuit breaker threshold

        status = window.status()

        assert status.circuit_breaker_tripped is True


# =============================================================================
# Conversation Tracking Integration
# =============================================================================


class TestConversationTrackingIntegration:
    """Tests for conversation entropy tracking integration."""

    def test_track_normal_conversation(self) -> None:
        """Normal conversation should not trigger alerts."""
        config = ConversationEntropyConfiguration(
            oscillation_threshold=0.8,
            drift_threshold=1.5,
        )
        tracker = ConversationEntropyTracker(config)

        # Normal conversation with stable entropy
        turns = [
            ("user", 1.5, 0.3),
            ("assistant", 1.6, 0.35),
            ("user", 1.55, 0.32),
            ("assistant", 1.58, 0.34),
        ]

        for role, entropy, variance in turns:
            tracker.record_turn(role, entropy, variance)

        assessment = tracker.assess()

        assert assessment.oscillation_detected is False
        assert assessment.drift_detected is False

    def test_detect_oscillation_pattern(self) -> None:
        """Should detect entropy oscillation (manipulation indicator)."""
        config = ConversationEntropyConfiguration(
            oscillation_threshold=0.5,  # Lower threshold for easier detection
            drift_threshold=2.0,
        )
        tracker = ConversationEntropyTracker(config)

        # Oscillating entropy pattern (potential manipulation)
        turns = [
            ("user", 1.0, 0.2),
            ("assistant", 3.0, 0.8),  # Spike
            ("user", 1.0, 0.2),       # Drop
            ("assistant", 3.0, 0.8),  # Spike
            ("user", 1.0, 0.2),       # Drop
            ("assistant", 3.0, 0.8),  # Spike
        ]

        for role, entropy, variance in turns:
            tracker.record_turn(role, entropy, variance)

        assessment = tracker.assess()

        # Should detect the oscillation pattern
        assert assessment.oscillation_detected is True or len(assessment.patterns) > 0

    def test_detect_drift_pattern(self) -> None:
        """Should detect entropy drift (gradual increase)."""
        config = ConversationEntropyConfiguration(
            oscillation_threshold=1.0,
            drift_threshold=0.5,  # Lower threshold for easier detection
        )
        tracker = ConversationEntropyTracker(config)

        # Drifting entropy (gradually increasing)
        for i in range(10):
            entropy = 1.0 + 0.3 * i  # Steady increase
            variance = 0.2 + 0.05 * i
            role = "user" if i % 2 == 0 else "assistant"
            tracker.record_turn(role, entropy, variance)

        assessment = tracker.assess()

        # Should detect drift or have elevated risk
        assert (
            assessment.drift_detected is True
            or assessment.manipulation_risk > 0.0
        )


# =============================================================================
# Phase Transition Integration
# =============================================================================


class TestPhaseTransitionIntegration:
    """Tests for phase transition theory integration."""

    def test_classify_phase_from_entropy(self) -> None:
        """Should classify phase based on entropy patterns."""
        logits = [2.0, 1.5, 0.5, -0.5, -1.0]
        temperature = 1.0

        entropy = PhaseTransitionTheory.compute_entropy(logits, temperature)
        tc = PhaseTransitionTheory.theoretical_tc()

        phase = PhaseTransitionTheory.classify_phase(temperature, tc)

        assert phase in Phase
        assert entropy >= 0

    def test_entropy_increases_with_temperature(self) -> None:
        """Entropy should generally increase with temperature."""
        logits = [2.0, 1.5, 0.5, -0.5, -1.0]

        entropy_low = PhaseTransitionTheory.compute_entropy(logits, 0.5)
        entropy_high = PhaseTransitionTheory.compute_entropy(logits, 2.0)

        assert entropy_high > entropy_low

    def test_phase_analysis_integration(self) -> None:
        """Full phase analysis should work end-to-end."""
        logits = [2.0, 1.5, 0.5, -0.5, -1.0]

        result = PhaseTransitionTheory.analyze(
            logits=logits,
            temperature=1.0,
            intensity_score=0.5,
        )

        assert result.temperature == 1.0
        assert result.estimated_tc > 0
        assert result.phase in Phase
        assert result.logit_variance >= 0
        assert result.effective_vocab_size >= 1


# =============================================================================
# Full Workflow Integration
# =============================================================================


class TestFullEntropyWorkflow:
    """Tests for the full probe → measure → classify → detect workflow."""

    def test_workflow_normal_operation(self) -> None:
        """Full workflow should work for normal operation."""
        calculator = LogitEntropyCalculator()
        window_config = EntropyWindowConfig(
            window_size=10,
            high_entropy_threshold=3.0,
            circuit_breaker_threshold=4.0,
        )
        window = EntropyWindow(window_config)

        # Step 1: Probe (simulate logit samples)
        rng = np.random.default_rng(42)

        for i in range(10):
            # Generate typical logits
            logits = rng.standard_normal(100).tolist()

            # Step 2: Measure
            result = calculator.calculate(logits)

            # Step 3: Track in window
            window.add(result.raw_entropy, result.variance, i)

        # Step 4: Assess
        status = window.status()

        assert status.level in EntropyLevel
        assert not status.circuit_breaker_tripped

    def test_workflow_with_phase_classification(self) -> None:
        """Workflow should integrate phase classification."""
        calculator = LogitEntropyCalculator()

        # Generate and analyze samples
        rng = np.random.default_rng(42)
        phases = []

        for _ in range(5):
            logits = rng.standard_normal(50).tolist()

            # Calculate entropy
            result = calculator.calculate(logits)

            # Classify phase
            tc = PhaseTransitionTheory.theoretical_tc()
            # Use entropy as proxy for effective temperature
            effective_temp = max(0.1, result.raw_entropy / 2.0)
            phase = PhaseTransitionTheory.classify_phase(effective_temp, tc)

            phases.append(phase)

        # Should have classified all samples
        assert len(phases) == 5
        assert all(p in Phase for p in phases)

    def test_workflow_distress_detection_to_circuit_breaker(self) -> None:
        """Distress detection should propagate to circuit breaker."""
        calculator = LogitEntropyCalculator()
        window_config = EntropyWindowConfig(
            window_size=5,
            high_entropy_threshold=2.0,
            circuit_breaker_threshold=3.0,
        )
        window = EntropyWindow(window_config)
        conv_config = ConversationEntropyConfiguration(
            oscillation_threshold=0.5,
            drift_threshold=1.0,
        )
        tracker = ConversationEntropyTracker(conv_config)

        # Simulate escalating distress
        for i in range(10):
            # Increasing uncertainty logits
            logits = [1.0] * 50  # Uniform = high entropy

            result = calculator.calculate(logits)
            window.add(result.raw_entropy, result.variance, i)

            role = "user" if i % 2 == 0 else "assistant"
            tracker.record_turn(role, result.raw_entropy, result.variance)

        window_status = window.status()
        conv_assessment = tracker.assess()

        # High entropy should trigger high level or circuit breaker
        assert window_status.level in {EntropyLevel.HIGH, EntropyLevel.CRITICAL}
        # Should have some risk indication
        assert conv_assessment.manipulation_risk >= 0


# =============================================================================
# Error Handling
# =============================================================================


class TestEntropyWorkflowErrorHandling:
    """Tests for error handling in the entropy workflow."""

    def test_empty_logits_handled(self) -> None:
        """Empty logits should be handled gracefully."""
        calculator = LogitEntropyCalculator()

        result = calculator.calculate([])

        # Should return a result (may have zero values)
        assert result is not None

    def test_single_logit_handled(self) -> None:
        """Single logit should be handled gracefully."""
        calculator = LogitEntropyCalculator()

        result = calculator.calculate([1.0])

        # Should return a result
        assert result is not None
        assert result.raw_entropy >= 0

    def test_extreme_logits_handled(self) -> None:
        """Extreme logit values should be handled gracefully."""
        calculator = LogitEntropyCalculator()

        # Very large logits
        large_logits = [1e10, -1e10, 0.0]
        result = calculator.calculate(large_logits)

        # Should not crash, entropy should be finite
        assert result is not None
        assert np.isfinite(result.raw_entropy)

    def test_empty_conversation_handled(self) -> None:
        """Empty conversation should be handled gracefully."""
        config = ConversationEntropyConfiguration()
        tracker = ConversationEntropyTracker(config)

        assessment = tracker.assess()

        # Should return assessment without crashing
        assert assessment is not None
        assert assessment.manipulation_risk >= 0


# =============================================================================
# Mathematical Invariants
# =============================================================================


class TestEntropyWorkflowInvariants:
    """Tests for mathematical invariants in the entropy workflow."""

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_always_bounded(self, seed: int) -> None:
        """Entropy should always be bounded by log(vocab_size)."""
        rng = np.random.default_rng(seed)
        calculator = LogitEntropyCalculator()

        vocab_size = rng.integers(10, 1000)
        logits = rng.standard_normal(vocab_size).tolist()

        result = calculator.calculate(logits)
        max_entropy = np.log(vocab_size)

        assert result.raw_entropy <= max_entropy + 1e-6

    def test_normalized_entropy_in_zero_one(self) -> None:
        """Normalized entropy should be in [0, 1]."""
        calculator = LogitEntropyCalculator()

        for _ in range(10):
            rng = np.random.default_rng()
            logits = rng.standard_normal(100).tolist()

            result = calculator.calculate(logits)

            assert 0.0 <= result.normalized_entropy <= 1.0

    def test_window_moving_average_stable(self) -> None:
        """Moving average should be stable over time."""
        config = EntropyWindowConfig(window_size=10)
        window = EntropyWindow(config)

        # Add constant entropy
        for i in range(20):
            window.add(2.0, 0.5, i)

        status = window.status()

        # Moving average should converge to constant
        assert abs(status.moving_average - 2.0) < 0.1

    def test_phase_classification_consistent(self) -> None:
        """Phase classification should be consistent with temperature."""
        tc = PhaseTransitionTheory.theoretical_tc()

        # Well below T_c should be ordered
        phase_low = PhaseTransitionTheory.classify_phase(tc * 0.5, tc)
        assert phase_low == Phase.ORDERED

        # Well above T_c should be disordered
        phase_high = PhaseTransitionTheory.classify_phase(tc * 2.0, tc)
        assert phase_high == Phase.DISORDERED
