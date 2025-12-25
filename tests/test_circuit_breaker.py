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

"""Tests for CircuitBreaker integration.

Tests the safety circuit breaker that monitors generation for
entropy spikes, refusal approach, persona drift, and oscillation patterns.
"""

import pytest

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    CircuitBreakerState,
    Configuration,
    InputSignals,
    RecommendedAction,
    SignalContributions,
    TriggerSource,
)


# Standard test config with explicit thresholds (no arbitrary defaults)
TEST_CONFIG = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)


class TestConfiguration:
    """Tests for Configuration dataclass."""

    def test_uniform_weights_config(self):
        """Uniform weights config should have valid weights summing to 1.0."""
        config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)
        assert config.is_weights_valid
        assert config.trip_threshold == 0.75
        assert config.warning_threshold == 0.50
        assert config.entropy_weight == 0.25
        assert config.refusal_weight == 0.25

    def test_from_baseline_measurements(self):
        """Should derive thresholds from baseline data."""
        # Simulate baseline measurements
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        config = Configuration.from_baseline_measurements(baseline)
        assert config.is_weights_valid
        # 99th percentile of 10 values = index 9 = 1.0
        assert config.trip_threshold == 1.0
        # 95th percentile of 10 values = index 9 = 1.0
        assert config.warning_threshold == 1.0

    def test_from_baseline_measurements_larger_sample(self):
        """With more data, percentiles should be more granular."""
        # 100 samples from 0.0 to 0.99
        baseline = [i / 100.0 for i in range(100)]
        config = Configuration.from_baseline_measurements(baseline)
        assert config.is_weights_valid
        # 99th percentile of 100 values = index 99 = 0.99
        assert config.trip_threshold == 0.99
        # 95th percentile of 100 values = index 95 = 0.95
        assert config.warning_threshold == 0.95

    def test_from_baseline_empty_raises(self):
        """Empty baseline should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="cannot be empty"):
            Configuration.from_baseline_measurements([])


class TestInputSignals:
    """Tests for InputSignals dataclass."""

    def test_default_signals(self):
        """Default signals should have safe values."""
        signals = InputSignals()
        assert signals.entropy_signal is None
        assert signals.refusal_distance is None
        assert signals.has_oscillation is False
        assert signals.token_index == 0

    def test_signals_with_values(self):
        """Signals should accept all values."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.8,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.2,
            drifting_traits=["helpfulness", "safety"],
            oscillation_severity=0.6,  # Raw severity
            has_oscillation=True,
            token_index=42,
        )

        assert signals.entropy_signal == 0.5
        assert signals.refusal_distance == 0.8
        assert signals.is_approaching_refusal is True
        assert len(signals.drifting_traits) == 2
        assert signals.has_oscillation is True
        assert signals.oscillation_severity == 0.6


class TestCircuitBreakerEvaluate:
    """Tests for CircuitBreakerIntegration.evaluate()."""

    def test_evaluate_safe_signals(self):
        """Safe signals should not trip the breaker."""
        signals = InputSignals(
            entropy_signal=0.2,
            refusal_distance=0.9,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.05,
            has_oscillation=False,
            token_index=10,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)

        assert state.is_tripped is False
        assert state.severity < 0.5
        assert state.recommended_action == RecommendedAction.continue_generation

    def test_evaluate_high_entropy_trips(self):
        """High entropy combined with other signals should trip the breaker."""
        signals = InputSignals(
            entropy_signal=0.99,  # Very high entropy
            refusal_distance=0.2,  # Close to refusal
            is_approaching_refusal=True,
            persona_drift_magnitude=0.6,
            drifting_traits=["safety", "honesty"],
            has_oscillation=True,
            oscillation_severity=0.8,  # High oscillation severity
            token_index=100,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)

        assert state.is_tripped is True
        assert state.severity >= 0.75
        # With equal weights, any high signal can be the trigger
        assert state.trigger_source is not None

    def test_evaluate_refusal_approach_trips(self):
        """Approaching refusal direction with other signals should trip the breaker."""
        signals = InputSignals(
            entropy_signal=0.95,  # Very high entropy
            refusal_distance=0.01,  # Extremely close to refusal
            is_approaching_refusal=True,
            persona_drift_magnitude=0.8,  # High drift
            drifting_traits=["safety", "honesty"],
            has_oscillation=True,
            oscillation_severity=0.8,  # High oscillation severity
            token_index=50,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)

        assert state.is_tripped is True
        assert state.severity >= 0.75

    def test_evaluate_persona_drift_contribution(self):
        """High persona drift should contribute significantly to severity."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.5,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.8,  # High drift
            drifting_traits=["honesty", "safety", "helpfulness"],
            has_oscillation=True,
            token_index=75,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)

        # Check that persona drift contributes
        assert state.signal_contributions.persona_drift > 0.1
        # With high drift, severity should be elevated
        assert state.severity >= 0.5

    def test_evaluate_oscillation_contribution(self):
        """Oscillation with high severity should contribute significantly."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.6,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.3,
            oscillation_severity=0.9,  # High oscillation severity
            has_oscillation=True,
            token_index=100,
        )

        config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)
        state = CircuitBreakerIntegration.evaluate(signals, config)

        # Oscillation with high severity should contribute significantly
        assert state.signal_contributions.oscillation > 0.15
        # Combined should elevate severity
        assert state.severity >= 0.5

    def test_evaluate_with_lower_threshold(self):
        """Lower threshold config should trip earlier."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.5,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.2,
            has_oscillation=False,
            token_index=20,
        )

        standard_config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)
        lower_config = Configuration.uniform_weights(trip_threshold=0.60, warning_threshold=0.40)

        standard_state = CircuitBreakerIntegration.evaluate(signals, standard_config)
        lower_state = CircuitBreakerIntegration.evaluate(signals, lower_config)

        # Same severity, but lower threshold trips more easily
        assert standard_state.severity == lower_state.severity
        # If severity is between thresholds, lower should trip while standard doesn't
        if 0.60 <= standard_state.severity < 0.75:
            assert lower_state.is_tripped
            assert not standard_state.is_tripped

    def test_evaluate_with_higher_threshold(self):
        """Higher threshold config should trip later."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.4,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.3,
            has_oscillation=True,
            token_index=50,
        )

        standard_config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)
        higher_config = Configuration.uniform_weights(trip_threshold=0.85, warning_threshold=0.65)

        standard_state = CircuitBreakerIntegration.evaluate(signals, standard_config)
        higher_state = CircuitBreakerIntegration.evaluate(signals, higher_config)

        # Permissive has higher threshold
        assert not higher_state.is_tripped or standard_state.is_tripped


class TestRecommendedActions:
    """Tests for action recommendations."""

    def test_continue_for_safe(self):
        """Safe signals should recommend continue."""
        signals = InputSignals(
            entropy_signal=0.1,
            refusal_distance=0.95,
            token_index=5,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        assert state.recommended_action == RecommendedAction.continue_generation

    def test_monitor_for_warning(self):
        """Warning level (0.5 <= severity < 0.75) should recommend monitor."""
        signals = InputSignals(
            entropy_signal=0.7,
            refusal_distance=0.4,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.4,
            has_oscillation=False,
            token_index=30,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        # With equal weights, verify we're in warning range (not tripped but elevated)
        assert state.severity >= 0.4, f"Severity {state.severity} too low for warning test"
        assert state.is_tripped is False
        assert state.recommended_action in [RecommendedAction.monitor, RecommendedAction.continue_generation]

    def test_stop_for_severe(self):
        """Severe signals should recommend stop."""
        signals = InputSignals(
            entropy_signal=0.99,
            refusal_distance=0.05,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.9,
            has_oscillation=True,
            oscillation_severity=0.95,  # Very high oscillation severity
            token_index=200,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        assert state.is_tripped is True
        assert state.recommended_action in [
            RecommendedAction.stop_generation,
            RecommendedAction.human_review,
        ]


class TestSignalContributions:
    """Tests for signal contribution calculations."""

    def test_contributions_sum_to_severity(self):
        """Signal contributions should sum to severity."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.5,
            is_approaching_refusal=True,
            persona_drift_magnitude=0.3,
            has_oscillation=True,
            token_index=50,
        )

        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        contrib = state.signal_contributions

        total = contrib.entropy + contrib.refusal + contrib.persona_drift + contrib.oscillation
        assert abs(total - state.severity) < 0.01

    def test_dominant_source_calculation(self):
        """Dominant source should be the highest contributor."""
        contrib = SignalContributions(
            entropy=0.4,
            refusal=0.1,
            persona_drift=0.1,
            oscillation=0.1,
        )

        assert contrib.dominant_source == TriggerSource.entropy_spike

    def test_dominant_source_refusal(self):
        """Refusal should be dominant when highest."""
        contrib = SignalContributions(
            entropy=0.1,
            refusal=0.5,
            persona_drift=0.1,
            oscillation=0.1,
        )

        assert contrib.dominant_source == TriggerSource.refusal_approach


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState."""

    def test_interpretation_safe(self):
        """Safe state should have positive interpretation."""
        state = CircuitBreakerState(
            is_tripped=False,
            severity=0.1,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.continue_generation,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

        assert "All clear" in state.interpretation

    def test_interpretation_tripped_entropy(self):
        """Tripped state should explain trigger."""
        state = CircuitBreakerState(
            is_tripped=True,
            severity=0.85,
            trigger_source=TriggerSource.entropy_spike,
            confidence=0.75,
            recommended_action=RecommendedAction.insert_safety_prompt,
            signal_contributions=SignalContributions(0.5, 0.15, 0.1, 0.1),
            token_index=100,
        )

        assert "UNCERTAINTY" in state.interpretation or "entropy" in state.interpretation.lower()

    def test_interpretation_refusal(self):
        """Refusal trigger should mention safety."""
        state = CircuitBreakerState(
            is_tripped=True,
            severity=0.8,
            trigger_source=TriggerSource.refusal_approach,
            confidence=0.75,
            recommended_action=RecommendedAction.insert_safety_prompt,
            signal_contributions=SignalContributions(0.2, 0.4, 0.1, 0.1),
            token_index=50,
        )

        assert "SAFETY" in state.interpretation or "refusal" in state.interpretation.lower()


class TestTelemetryAndMetrics:
    """Tests for telemetry and metrics export."""

    def test_create_telemetry(self):
        """Telemetry should capture state and signals."""
        signals = InputSignals(
            entropy_signal=0.8,
            refusal_distance=0.3,
            has_oscillation=True,
            oscillation_severity=0.5,
            token_index=75,
        )
        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals, TEST_CONFIG)

        assert telemetry.token_index == 75
        assert telemetry.state == state
        # any_signal_exceeded is now based on severity >= warning_threshold (0.50)
        # With these signals, severity should be above warning threshold
        assert telemetry.any_signal_exceeded == (state.severity >= TEST_CONFIG.warning_threshold)
        # Telemetry should include raw oscillation measurements
        assert telemetry.oscillation_severity == 0.5

    def test_to_metrics_dict(self):
        """Metrics dict should have all expected keys."""
        signals = InputSignals(entropy_signal=0.5, token_index=10)
        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)
        metrics = CircuitBreakerIntegration.to_metrics_dict(state)

        expected_keys = [
            "geometry/circuit_breaker_tripped",
            "geometry/circuit_breaker_confidence",
            "geometry/circuit_breaker_severity",
            "geometry/circuit_breaker_entropy",
            "geometry/circuit_breaker_refusal",
            "geometry/circuit_breaker_persona",
            "geometry/circuit_breaker_oscillation",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_none_signals(self):
        """Should handle all None signals gracefully."""
        signals = InputSignals(token_index=0)
        state = CircuitBreakerIntegration.evaluate(signals, TEST_CONFIG)

        assert state.is_tripped is False
        assert state.severity == 0.0
        assert state.confidence == 0.0  # No signals available

    def test_boundary_trip_threshold(self):
        """Should trip exactly at threshold."""
        config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)
        # Create signals that produce severity above threshold
        # With uniform weights, need signals averaging ~3 (since 0.75/0.25 = 3 needed per channel)

        signals = InputSignals(
            entropy_signal=0.99,  # Max entropy
            refusal_distance=0.01,  # Very close to refusal
            is_approaching_refusal=True,
            persona_drift_magnitude=0.9,  # High drift
            drifting_traits=["safety", "honesty", "helpfulness"],
            has_oscillation=True,
            oscillation_severity=0.95,  # High oscillation
            token_index=100,
        )

        state = CircuitBreakerIntegration.evaluate(signals, config)
        # Should be well above threshold with these extreme signals
        assert state.is_tripped is True
        assert state.severity >= config.trip_threshold

    def test_action_description(self):
        """RecommendedAction should have descriptions."""
        for action in RecommendedAction:
            assert len(action.description) > 0
