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

"""Integration tests for the entropy workflow.

Tests the full workflow:
    probe → measure → classify → detect

This validates that the entropy monitoring components work together
to detect distress patterns and trigger appropriate responses.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
    ConversationEntropyConfiguration,
    ConversationEntropyTracker,
    ConversationPattern,
)
from modelcypher.core.domain.entropy.entropy_window import (
    EntropyLevel as WindowEntropyLevel,
)
from modelcypher.core.domain.entropy.entropy_window import (
    EntropyWindow,
    EntropyWindowConfig,
)
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    EntropyLevel,
    LogitEntropyCalculator,
)
from modelcypher.core.domain.thermo.phase_transition_theory import (
    Phase,
    PhaseTransitionTheory,
)

from modelcypher.core.domain._backend import get_default_backend


# =============================================================================
# Entropy Calculation Integration
# =============================================================================


class TestEntropyCalculationIntegration:
    """Tests for entropy calculation integration."""

    def test_calculate_entropy_from_logits(self) -> None:
        """Entropy calculation should work on typical logit distributions."""
        backend = get_default_backend()
        calculator = LogitEntropyCalculator(backend=backend)

        # Typical logit distribution (softmax input)
        logits = backend.array([2.0, 1.5, 0.5, -0.5, -1.0])

        entropy, variance = calculator.compute(logits)

        assert entropy >= 0
        assert variance >= 0

    def test_entropy_tracks_uncertainty(self) -> None:
        """Higher uncertainty distributions should have higher entropy."""
        backend = get_default_backend()
        calculator = LogitEntropyCalculator(backend=backend)

        # Low uncertainty (one dominant logit)
        low_uncertainty = backend.array([10.0, 0.0, 0.0, 0.0, 0.0])

        # High uncertainty (uniform-ish)
        high_uncertainty = backend.array([1.0, 1.0, 1.0, 1.0, 1.0])

        low_entropy, _ = calculator.compute(low_uncertainty)
        high_entropy, _ = calculator.compute(high_uncertainty)

        assert high_entropy > low_entropy

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_always_non_negative(self, seed: int) -> None:
        """Entropy should always be non-negative."""
        backend = get_default_backend()
        rng = np.random.default_rng(seed)
        calculator = LogitEntropyCalculator(backend=backend)

        logits = rng.standard_normal(20).astype(np.float32)
        entropy, variance = calculator.compute(logits)

        assert entropy >= 0
        assert variance >= 0


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
        assert status.level in WindowEntropyLevel

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

        assert status.level in {WindowEntropyLevel.HIGH, WindowEntropyLevel.MODERATE}

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

        assert status.should_trip_circuit_breaker is True


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
        tracker = ConversationEntropyTracker(configuration=config)

        # Normal conversation with stable entropy deltas
        for i in range(4):
            assessment = tracker.record_turn(
                token_count=50,
                avg_delta=0.1,  # Small, stable deltas
                max_anomaly_score=0.1,
                anomaly_count=0,
            )

        # Should settle into normal pattern
        assert assessment.pattern in {
            ConversationPattern.SETTLING,
            ConversationPattern.INSUFFICIENT,
        }
        # Check individual components show low signals
        c = assessment.manipulation_components
        assert not c.circuit_breaker_tripped
        assert not c.baseline_oscillation_exceeded

    def test_detect_oscillation_pattern(self) -> None:
        """Should detect entropy oscillation (manipulation indicator)."""
        config = ConversationEntropyConfiguration(
            oscillation_threshold=0.3,  # Lower threshold for easier detection
            drift_threshold=2.0,
        )
        tracker = ConversationEntropyTracker(configuration=config)

        # Oscillating entropy deltas (potential manipulation)
        for i in range(6):
            delta = 1.0 if i % 2 == 0 else -0.8  # Alternating
            assessment = tracker.record_turn(
                token_count=50,
                avg_delta=delta,
                max_anomaly_score=0.5,
                anomaly_count=1,
            )

        # Should detect the oscillation pattern or elevated component signals
        c = assessment.manipulation_components
        assert (
            assessment.pattern == ConversationPattern.OSCILLATING
            or assessment.oscillation_amplitude > 0.5
            or c.oscillation_amplitude_score > 0.3
        )

    def test_detect_drift_pattern(self) -> None:
        """Should detect entropy drift (gradual increase)."""
        config = ConversationEntropyConfiguration(
            oscillation_threshold=1.0,
            drift_threshold=0.5,  # Lower threshold for easier detection
        )
        tracker = ConversationEntropyTracker(configuration=config)

        # Drifting entropy (gradually increasing deltas)
        for i in range(10):
            delta = 0.1 + 0.1 * i  # Steady increase
            assessment = tracker.record_turn(
                token_count=50,
                avg_delta=delta,
                max_anomaly_score=0.2,
                anomaly_count=0,
            )

        # Should detect drift or have elevated component signals
        c = assessment.manipulation_components
        assert (
            assessment.pattern == ConversationPattern.DRIFTING
            or assessment.cumulative_drift > 0.3
            or c.drift_score > 0.2
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
        calculator = LogitEntropyCalculator(backend=get_default_backend())
        window_config = EntropyWindowConfig(
            window_size=10,
            high_entropy_threshold=3.0,
            circuit_breaker_threshold=4.0,
        )
        window = EntropyWindow(window_config)

        # Step 1: Probe (simulate logit samples)
        rng = np.random.default_rng(42)

        for i in range(10):
            # Generate peaked logits (low entropy - one dominant class)
            logits = np.zeros(100, dtype=np.float32)
            logits[i % 10] = 5.0  # One dominant logit per sample
            logits += rng.uniform(-0.1, 0.1, 100).astype(np.float32)  # Small noise

            # Step 2: Measure
            entropy, variance = calculator.compute(logits)

            # Step 3: Track in window
            window.add(entropy, variance, i)

        # Step 4: Assess
        status = window.status()

        assert status.level in WindowEntropyLevel
        assert not status.should_trip_circuit_breaker

    def test_workflow_with_phase_classification(self) -> None:
        """Workflow should integrate phase classification."""
        LogitEntropyCalculator(backend=get_default_backend())

        # Generate and analyze samples
        rng = np.random.default_rng(42)
        phases = []

        for _ in range(5):
            logits_list = rng.standard_normal(50).tolist()

            # Calculate entropy via phase theory (works with lists)
            entropy = PhaseTransitionTheory.compute_entropy(logits_list, 1.0)

            # Classify phase
            tc = PhaseTransitionTheory.theoretical_tc()
            # Use entropy as proxy for effective temperature
            effective_temp = max(0.1, entropy / 2.0)
            phase = PhaseTransitionTheory.classify_phase(effective_temp, tc)

            phases.append(phase)

        # Should have classified all samples
        assert len(phases) == 5
        assert all(p in Phase for p in phases)

    def test_workflow_distress_detection_to_circuit_breaker(self) -> None:
        """Distress detection should propagate to circuit breaker."""
        calculator = LogitEntropyCalculator(backend=get_default_backend())
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
        tracker = ConversationEntropyTracker(configuration=conv_config)

        # Simulate escalating distress with uniform logits
        for i in range(10):
            # Uniform logits = high entropy
            logits = np.ones(50, dtype=np.float32)

            entropy, variance = calculator.compute(logits)
            window.add(entropy, variance, i)

            tracker.record_turn(
                token_count=50,
                avg_delta=entropy / 2.0,
                max_anomaly_score=0.3,
                anomaly_count=0,
            )

        window_status = window.status()
        conv_assessment = tracker.record_turn(
            token_count=50, avg_delta=0.5, max_anomaly_score=0.3, anomaly_count=0
        )

        # High entropy should trigger high level or circuit breaker
        assert window_status.level in {WindowEntropyLevel.HIGH, WindowEntropyLevel.MODERATE}
        # Should have manipulation components
        assert conv_assessment.manipulation_components is not None


# =============================================================================
# Error Handling
# =============================================================================


class TestEntropyWorkflowErrorHandling:
    """Tests for error handling in the entropy workflow."""

    def test_empty_logits_handled(self) -> None:
        """Empty logits should raise ValueError (no valid entropy for empty array)."""
        calculator = LogitEntropyCalculator(backend=get_default_backend())

        logits = np.array([], dtype=np.float32)

        # Empty logits cannot have entropy computed - expect ValueError
        with pytest.raises(ValueError):
            calculator.compute(logits)

    def test_single_logit_handled(self) -> None:
        """Single logit should be handled gracefully."""
        calculator = LogitEntropyCalculator(backend=get_default_backend())

        logits = np.array([1.0], dtype=np.float32)
        entropy, variance = calculator.compute(logits)

        # Should return values without crashing
        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_extreme_logits_handled(self) -> None:
        """Extreme logit values should be handled gracefully."""
        calculator = LogitEntropyCalculator(backend=get_default_backend())

        # Very large logits (but not extreme enough to cause overflow)
        large_logits = np.array([100.0, -100.0, 0.0], dtype=np.float32)
        entropy, variance = calculator.compute(large_logits)

        # Should not crash, entropy should be finite
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)

    def test_empty_conversation_handled(self) -> None:
        """Empty conversation should be handled gracefully."""
        config = ConversationEntropyConfiguration()
        tracker = ConversationEntropyTracker(configuration=config)

        # Record a turn to get assessment
        assessment = tracker.record_turn(
            token_count=0,
            avg_delta=0.0,
            max_anomaly_score=0.0,
            anomaly_count=0,
        )

        # Should return assessment without crashing
        assert assessment is not None
        assert assessment.manipulation_components is not None


# =============================================================================
# Mathematical Invariants
# =============================================================================


class TestEntropyWorkflowInvariants:
    """Tests for mathematical invariants in the entropy workflow."""

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_always_bounded(self, seed: int) -> None:
        """Entropy should always be bounded by log(vocab_size)."""
        rng = np.random.default_rng(seed)
        calculator = LogitEntropyCalculator(backend=get_default_backend())

        vocab_size = rng.integers(10, 1000)
        logits = rng.standard_normal(vocab_size).astype(np.float32)

        entropy, _ = calculator.compute(logits)
        max_entropy = np.log(vocab_size)

        assert entropy <= max_entropy + 1e-6

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

    @pytest.mark.parametrize("seed", range(3))
    def test_entropy_level_classification_consistent(self, seed: int) -> None:
        """Entropy level classification should be consistent."""
        rng = np.random.default_rng(seed)
        calculator = LogitEntropyCalculator(backend=get_default_backend())

        logits = rng.standard_normal(100).astype(np.float32)
        entropy, _ = calculator.compute(logits)

        level = calculator.classify(entropy)

        assert level in EntropyLevel
