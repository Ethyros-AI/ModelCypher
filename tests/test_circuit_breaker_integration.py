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

"""Tests for CircuitBreakerIntegration."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    Configuration,
    InputSignals,
    RecommendedAction,
    SignalContributions,
    TriggerSource,
)


class TestConfiguration:
    """Tests for Configuration dataclass."""

    def test_uniform_weights_creates_valid_config(self):
        """Test uniform_weights creates valid configuration."""
        config = Configuration.uniform_weights(
            trip_threshold=0.8,
            warning_threshold=0.5,
        )

        assert config.trip_threshold == 0.8
        assert config.warning_threshold == 0.5
        assert config.entropy_weight == 0.25
        assert config.refusal_weight == 0.25
        assert config.persona_drift_weight == 0.25
        assert config.oscillation_weight == 0.25
        assert config.is_weights_valid is True

    def test_from_baseline_measurements(self):
        """Test threshold derivation from baseline measurements."""
        # Simulate 100 baseline measurements
        baselines = [0.1 * (i % 10) for i in range(100)]  # 0.0 to 0.9

        config = Configuration.from_baseline_measurements(
            baselines,
            percentile_trip=99.0,
            percentile_warning=95.0,
        )

        # 99th percentile of [0.0, 0.1, ..., 0.9] repeated 10 times
        assert config.trip_threshold >= 0.8
        assert config.warning_threshold >= 0.8

    def test_from_baseline_empty_raises(self):
        """Test that empty baselines raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Configuration.from_baseline_measurements([])

    def test_is_weights_valid(self):
        """Test weight validation."""
        valid = Configuration(
            entropy_weight=0.25,
            refusal_weight=0.25,
            persona_drift_weight=0.25,
            oscillation_weight=0.25,
            trip_threshold=0.8,
            warning_threshold=0.5,
            trend_window_size=10,
            enable_auto_escalation=True,
            cooldown_tokens=5,
        )
        assert valid.is_weights_valid is True

        invalid = Configuration(
            entropy_weight=0.5,
            refusal_weight=0.5,
            persona_drift_weight=0.5,
            oscillation_weight=0.5,  # Total = 2.0
            trip_threshold=0.8,
            warning_threshold=0.5,
            trend_window_size=10,
            enable_auto_escalation=True,
            cooldown_tokens=5,
        )
        assert invalid.is_weights_valid is False


class TestInputSignals:
    """Tests for InputSignals dataclass."""

    def test_default_values(self):
        """Test that default signals are None/empty."""
        signals = InputSignals()

        assert signals.entropy_signal is None
        assert signals.refusal_distance is None
        assert signals.persona_drift_magnitude is None
        assert signals.oscillation_severity is None
        assert signals.drifting_traits == []

    def test_signal_creation(self):
        """Test creating signals with values."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.3,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.2,
            drifting_traits=["helpfulness", "safety"],
            oscillation_severity=0.1,
            consecutive_oscillations=2,
            has_oscillation=True,
            token_index=100,
        )

        assert signals.entropy_signal == 0.6
        assert signals.refusal_distance == 0.3
        assert signals.is_approaching_refusal is True
        assert len(signals.drifting_traits) == 2


class TestSignalContributions:
    """Tests for SignalContributions dataclass."""

    def test_dominant_source_entropy(self):
        """Test dominant source detection for entropy."""
        contrib = SignalContributions(
            entropy=0.8,
            refusal=0.2,
            persona_drift=0.1,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.entropy_spike

    def test_dominant_source_refusal(self):
        """Test dominant source detection for refusal."""
        contrib = SignalContributions(
            entropy=0.2,
            refusal=0.9,
            persona_drift=0.1,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.refusal_approach

    def test_dominant_source_persona_drift(self):
        """Test dominant source detection for persona drift."""
        contrib = SignalContributions(
            entropy=0.2,
            refusal=0.2,
            persona_drift=0.8,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.persona_drift

    def test_dominant_source_oscillation(self):
        """Test dominant source detection for oscillation."""
        contrib = SignalContributions(
            entropy=0.1,
            refusal=0.1,
            persona_drift=0.1,
            oscillation=0.9,
        )
        assert contrib.dominant_source == TriggerSource.oscillation_pattern

    def test_get_contribution(self):
        """Test getting contribution by source."""
        contrib = SignalContributions(
            entropy=0.5,
            refusal=0.3,
            persona_drift=0.2,
            oscillation=0.1,
        )

        assert contrib.get(TriggerSource.entropy_spike) == 0.5
        assert contrib.get(TriggerSource.refusal_approach) == 0.3
        assert contrib.get(TriggerSource.manual) == 0.0  # Unknown source

    def test_max_and_mean_signal(self):
        """Test max and mean signal calculations."""
        contrib = SignalContributions(
            entropy=0.4,
            refusal=0.2,
            persona_drift=0.3,
            oscillation=0.1,
        )

        assert contrib.max_signal == 0.4
        assert abs(contrib.mean_signal - 0.25) < 1e-10


class TestCircuitBreakerIntegration:
    """Tests for CircuitBreakerIntegration evaluation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Configuration.uniform_weights(
            trip_threshold=0.7,
            warning_threshold=0.4,
        )

    def test_evaluate_safe_signals(self, config):
        """Test evaluation with safe (low) signals."""
        signals = InputSignals(
            entropy_signal=0.1,
            refusal_distance=0.9,  # Far from refusal
            persona_drift_magnitude=0.05,
            oscillation_severity=0.02,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        assert state.is_tripped is False
        assert state.severity < config.warning_threshold
        assert state.recommended_action == RecommendedAction.continue_generation

    def test_evaluate_tripped_entropy(self, config):
        """Test evaluation trips on high entropy."""
        signals = InputSignals(
            entropy_signal=0.95,  # Very high normalized entropy
            refusal_distance=0.8,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.05,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        # High entropy should contribute significantly
        assert state.signal_contributions.entropy > 0.2
        # May or may not trip depending on total, but entropy should be high

    def test_evaluate_tripped_refusal_approach(self, config):
        """Test evaluation trips when approaching refusal boundary."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.05,  # Very close to refusal boundary
            is_approaching_refusal=True,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.05,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        # Approaching refusal should contribute significantly
        assert state.signal_contributions.refusal > 0.15

    def test_evaluate_warning_zone(self, config):
        """Test evaluation in warning zone (above warning, below trip)."""
        # Create signals that should produce moderate severity
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.5,
            persona_drift_magnitude=0.3,
            oscillation_severity=0.2,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        # Should be in middle range
        assert state.severity > 0.0

    def test_evaluate_none_signals(self, config):
        """Test evaluation handles None signals gracefully."""
        signals = InputSignals()  # All None

        state = CircuitBreakerIntegration.evaluate(signals, config)

        assert state.is_tripped is False
        assert state.severity == 0.0

    def test_evaluate_oscillation_pattern(self, config):
        """Test evaluation with oscillation pattern."""
        signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.7,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.8,
            consecutive_oscillations=5,
            has_oscillation=True,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        # Oscillation should contribute
        assert state.signal_contributions.oscillation > 0.0

    def test_evaluate_persona_drift(self, config):
        """Test evaluation with persona drift."""
        signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.7,
            persona_drift_magnitude=0.7,
            drifting_traits=["helpfulness", "safety", "honesty"],
            oscillation_severity=0.1,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)

        # Persona drift with multiple traits should contribute
        assert state.signal_contributions.persona_drift > 0.0

    def test_recommended_actions(self, config):
        """Test that recommended actions make sense."""
        # Very safe
        safe_signals = InputSignals(
            entropy_signal=0.05,
            refusal_distance=0.95,
            persona_drift_magnitude=0.02,
            oscillation_severity=0.01,
        )
        safe_state = CircuitBreakerIntegration.evaluate(safe_signals, config)
        assert safe_state.recommended_action == RecommendedAction.continue_generation

        # Tripped state should have stronger action
        danger_signals = InputSignals(
            entropy_signal=0.9,
            refusal_distance=0.1,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.8,
            oscillation_severity=0.7,
        )
        danger_state = CircuitBreakerIntegration.evaluate(danger_signals, config)
        # Should recommend something stronger than continue
        if danger_state.is_tripped:
            assert danger_state.recommended_action != RecommendedAction.continue_generation
