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

Tests the safety signal aggregation that measures entropy, refusal proximity,
persona drift, and oscillation magnitude.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    CircuitBreakerState,
    InputSignals,
    SignalContributions,
    TriggerSource,
)


@pytest.fixture
def backend():
    """Provide backend for dtype-derived tolerances."""
    return get_default_backend()


def _eps(backend) -> float:
    return machine_epsilon(backend, backend.array([1.0]))


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
            oscillation_severity=0.6,
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

    def test_evaluate_safe_signals(self, backend):
        """Safe signals produce low severity."""
        signals = InputSignals(
            entropy_signal=0.2,
            refusal_distance=0.9,
            persona_drift_magnitude=0.05,
            has_oscillation=False,
            token_index=10,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        assert state.severity > 0.0
        assert state.dominant_source == TriggerSource.entropy_spike
        expected = (0.2 + 0.1 + 0.05 + 0.0) / 4.0
        assert abs(state.severity - expected) <= _eps(backend)

    def test_evaluate_high_entropy(self, backend):
        """High entropy dominates severity."""
        signals = InputSignals(
            entropy_signal=0.99,
            refusal_distance=0.2,
            persona_drift_magnitude=0.6,
            oscillation_severity=0.8,
            has_oscillation=True,
            token_index=100,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        assert state.dominant_source == TriggerSource.entropy_spike
        expected = (0.99 + 0.8 + 0.6 + 0.8) / 4.0
        assert abs(state.severity - expected) <= _eps(backend)

    def test_evaluate_refusal_approach(self, backend):
        """Refusal proximity dominates when closest to boundary."""
        signals = InputSignals(
            entropy_signal=0.1,
            refusal_distance=0.01,
            persona_drift_magnitude=0.2,
            oscillation_severity=0.3,
            has_oscillation=True,
            token_index=50,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        assert state.dominant_source == TriggerSource.refusal_approach
        expected = (0.1 + 0.99 + 0.2 + 0.3) / 4.0
        assert abs(state.severity - expected) <= _eps(backend)

    def test_evaluate_persona_drift_contribution(self, backend):
        """Persona drift contributes directly to severity."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.5,
            persona_drift_magnitude=0.8,
            has_oscillation=True,
            token_index=75,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        assert abs(state.signal_contributions.persona_drift - 0.8) <= _eps(backend)
        expected = (0.6 + 0.5 + 0.8 + 0.0) / 4.0
        assert abs(state.severity - expected) <= _eps(backend)

    def test_evaluate_oscillation_contribution(self, backend):
        """Oscillation contributes directly to severity."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.6,
            persona_drift_magnitude=0.3,
            oscillation_severity=0.9,
            has_oscillation=True,
            token_index=100,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        assert abs(state.signal_contributions.oscillation - 0.9) <= _eps(backend)
        expected = (0.5 + 0.4 + 0.3 + 0.9) / 4.0
        assert abs(state.severity - expected) <= _eps(backend)


class TestSignalContributions:
    """Tests for signal contribution calculations."""

    def test_contributions_sum_to_severity(self, backend):
        """Signal contributions mean should equal severity."""
        signals = InputSignals(
            entropy_signal=0.5,
            refusal_distance=0.5,
            persona_drift_magnitude=0.3,
            has_oscillation=True,
            token_index=50,
        )

        state = CircuitBreakerIntegration.evaluate(signals)
        contrib = state.signal_contributions

        total = contrib.entropy + contrib.refusal + contrib.persona_drift + contrib.oscillation
        assert abs((total / 4.0) - state.severity) <= _eps(backend)

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

    def test_state_properties(self):
        """State should expose raw measurements."""
        state = CircuitBreakerState(
            severity=0.1,
            dominant_source=None,
            confidence=0.75,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

        assert state.severity == 0.1
        assert state.dominant_source is None
        assert state.confidence == 0.75


class TestTelemetryAndMetrics:
    """Tests for telemetry and metrics export."""

    def test_create_telemetry(self):
        """Telemetry should capture state and signals."""
        signals = InputSignals(
            entropy_signal=0.8,
            refusal_distance=0.3,
            oscillation_severity=0.5,
            has_oscillation=True,
            token_index=75,
        )
        state = CircuitBreakerIntegration.evaluate(signals)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals)

        assert telemetry.token_index == 75
        assert telemetry.state == state
        assert telemetry.oscillation_severity == 0.5

    def test_to_metrics_dict(self):
        """Metrics dict should have all expected keys."""
        signals = InputSignals(entropy_signal=0.5, token_index=10)
        state = CircuitBreakerIntegration.evaluate(signals)
        metrics = CircuitBreakerIntegration.to_metrics_dict(state)

        expected_keys = [
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
        state = CircuitBreakerIntegration.evaluate(signals)

        assert state.severity == 0.0
        assert state.confidence == 0.0
        assert state.dominant_source is None
