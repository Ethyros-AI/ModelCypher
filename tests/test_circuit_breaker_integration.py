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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    InputSignals,
    SignalContributions,
    TriggerSource,
)


def _eps():
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


class TestInputSignals:
    """Tests for InputSignals dataclass."""

    def test_default_values(self):
        """Default signals are None/empty."""
        signals = InputSignals()

        assert signals.entropy_signal is None
        assert signals.refusal_distance is None
        assert signals.persona_drift_magnitude is None
        assert signals.oscillation_severity is None
        assert signals.drifting_traits == []

    def test_signal_creation(self):
        """Signals accept values."""
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
        """Dominant source detection for entropy."""
        contrib = SignalContributions(
            entropy=0.8,
            refusal=0.2,
            persona_drift=0.1,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.entropy_spike

    def test_dominant_source_refusal(self):
        """Dominant source detection for refusal."""
        contrib = SignalContributions(
            entropy=0.2,
            refusal=0.9,
            persona_drift=0.1,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.refusal_approach

    def test_dominant_source_persona_drift(self):
        """Dominant source detection for persona drift."""
        contrib = SignalContributions(
            entropy=0.2,
            refusal=0.2,
            persona_drift=0.8,
            oscillation=0.1,
        )
        assert contrib.dominant_source == TriggerSource.persona_drift

    def test_dominant_source_oscillation(self):
        """Dominant source detection for oscillation."""
        contrib = SignalContributions(
            entropy=0.1,
            refusal=0.1,
            persona_drift=0.1,
            oscillation=0.9,
        )
        assert contrib.dominant_source == TriggerSource.oscillation_pattern

    def test_get_contribution(self):
        """Getting contribution by source."""
        contrib = SignalContributions(
            entropy=0.5,
            refusal=0.3,
            persona_drift=0.2,
            oscillation=0.1,
        )

        assert contrib.get(TriggerSource.entropy_spike) == 0.5
        assert contrib.get(TriggerSource.refusal_approach) == 0.3
        assert contrib.get(TriggerSource.manual) == 0.0

    def test_max_and_mean_signal(self):
        """Max and mean signal calculations."""
        contrib = SignalContributions(
            entropy=0.4,
            refusal=0.2,
            persona_drift=0.3,
            oscillation=0.1,
        )

        assert contrib.max_signal == 0.4
        assert abs(contrib.mean_signal - 0.25) <= _eps()


class TestEvaluation:
    """Tests for evaluating signals."""

    def test_evaluate_returns_measurements(self):
        """Evaluate returns raw measurements."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.4,
            persona_drift_magnitude=0.2,
            oscillation_severity=0.1,
            has_oscillation=True,
            token_index=42,
        )

        state = CircuitBreakerIntegration.evaluate(signals)
        expected = (0.6 + 0.6 + 0.2 + 0.1) / 4.0
        assert abs(state.severity - expected) <= _eps()
        assert state.dominant_source == TriggerSource.entropy_spike
        assert state.confidence == 1.0

    def test_evaluate_missing_signals(self):
        """Missing signals reduce confidence."""
        signals = InputSignals(
            entropy_signal=0.6,
            token_index=10,
        )

        state = CircuitBreakerIntegration.evaluate(signals)
        assert state.confidence == 0.25
        assert abs(state.severity - (0.6 / 4.0)) <= _eps()


class TestMetrics:
    """Tests for metric output."""

    def test_to_metrics_dict(self):
        """Metrics include raw measurements."""
        signals = InputSignals(
            entropy_signal=0.6,
            refusal_distance=0.4,
            persona_drift_magnitude=0.2,
            oscillation_severity=0.1,
            token_index=10,
        )
        state = CircuitBreakerIntegration.evaluate(signals)
        metrics = CircuitBreakerIntegration.to_metrics_dict(state)

        assert metrics["geometry/circuit_breaker_confidence"] == 1.0
        assert metrics["geometry/circuit_breaker_entropy"] == 0.6
