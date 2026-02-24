"""Tests for circuit breaker signal aggregation logic."""

import pytest

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    CircuitBreakerState,
    CircuitBreakerTelemetry,
    InputSignals,
    SignalContributions,
    TriggerSource,
)

# ---------------------------------------------------------------------------
# evaluate -- all-None baseline
# ---------------------------------------------------------------------------

class TestEvaluateAllNone:
    """With no signals provided, everything should be zero."""

    def test_all_contributions_zero(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        c = state.signal_contributions
        assert c.entropy == 0.0
        assert c.refusal == 0.0
        assert c.persona_drift == 0.0
        assert c.oscillation == 0.0

    def test_severity_zero(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        assert state.severity == 0.0

    def test_confidence_zero(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        assert state.confidence == 0.0

    def test_dominant_source_none_when_severity_zero(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        assert state.dominant_source is None


# ---------------------------------------------------------------------------
# evaluate -- single-signal cases
# ---------------------------------------------------------------------------

class TestEvaluateSingleSignal:
    """Each signal in isolation."""

    def test_entropy_only(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(entropy_signal=0.8)
        )
        assert state.signal_contributions.entropy == 0.8
        assert state.signal_contributions.refusal == 0.0
        assert state.confidence == pytest.approx(0.25)
        # severity = mean of (0.8, 0, 0, 0)
        assert state.severity == pytest.approx(0.2)

    def test_refusal_only_approaching(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(refusal_distance=0.3, is_approaching_refusal=True)
        )
        # contribution = 1.0 - 0.3 = 0.7
        assert state.signal_contributions.refusal == pytest.approx(0.7)
        assert state.confidence == pytest.approx(0.25)

    def test_refusal_only_not_approaching(self):
        """is_approaching_refusal is not used to gate the contribution."""
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(refusal_distance=0.3, is_approaching_refusal=False)
        )
        # Contribution is still 1.0 - distance regardless of approach flag
        assert state.signal_contributions.refusal == pytest.approx(0.7)

    def test_persona_drift_only(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(persona_drift_magnitude=0.6)
        )
        assert state.signal_contributions.persona_drift == pytest.approx(0.6)
        assert state.confidence == pytest.approx(0.25)

    def test_oscillation_only(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(
                oscillation_severity=0.9,
                has_oscillation=True,
                consecutive_oscillations=3,
            )
        )
        assert state.signal_contributions.oscillation == pytest.approx(0.9)
        assert state.confidence == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# evaluate -- all signals present
# ---------------------------------------------------------------------------

class TestEvaluateAllSignals:
    """All four signal categories provided."""

    def test_confidence_one(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(
                entropy_signal=0.5,
                refusal_distance=0.5,
                persona_drift_magnitude=0.5,
                oscillation_severity=0.5,
            )
        )
        assert state.confidence == pytest.approx(1.0)

    def test_severity_is_mean(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(
                entropy_signal=0.2,
                refusal_distance=0.6,  # contribution = 0.4
                persona_drift_magnitude=0.3,
                oscillation_severity=0.1,
            )
        )
        # mean of (0.2, 0.4, 0.3, 0.1) = 0.25
        assert state.severity == pytest.approx(0.25)

    def test_dominant_source_picks_highest(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(
                entropy_signal=0.1,
                refusal_distance=0.0,  # contribution = 1.0
                persona_drift_magnitude=0.3,
                oscillation_severity=0.5,
            )
        )
        assert state.dominant_source == TriggerSource.refusal_approach


# ---------------------------------------------------------------------------
# evaluate -- clamping out-of-range inputs
# ---------------------------------------------------------------------------

class TestEvaluateClamping:
    """Values outside [0, 1] are clamped."""

    def test_entropy_above_one_clamped(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(entropy_signal=1.5)
        )
        assert state.signal_contributions.entropy == pytest.approx(1.0)

    def test_entropy_below_zero_clamped(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(entropy_signal=-0.5)
        )
        assert state.signal_contributions.entropy == pytest.approx(0.0)

    def test_persona_drift_above_one_clamped(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(persona_drift_magnitude=2.0)
        )
        assert state.signal_contributions.persona_drift == pytest.approx(1.0)

    def test_oscillation_severity_above_one_clamped(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(oscillation_severity=5.0, has_oscillation=True)
        )
        assert state.signal_contributions.oscillation == pytest.approx(1.0)

    def test_refusal_contribution_clamped(self):
        """refusal_distance can be negative, pushing 1-d above 1.0."""
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(refusal_distance=-0.5)
        )
        # 1.0 - (-0.5) = 1.5, clamped to 1.0
        assert state.signal_contributions.refusal == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SignalContributions -- properties and get()
# ---------------------------------------------------------------------------

class TestSignalContributions:
    """Computed properties on SignalContributions."""

    def test_dominant_source_entropy(self):
        c = SignalContributions(entropy=0.9, refusal=0.1, persona_drift=0.2, oscillation=0.3)
        assert c.dominant_source == TriggerSource.entropy_spike

    def test_dominant_source_refusal(self):
        c = SignalContributions(entropy=0.1, refusal=0.9, persona_drift=0.2, oscillation=0.3)
        assert c.dominant_source == TriggerSource.refusal_approach

    def test_dominant_source_persona(self):
        c = SignalContributions(entropy=0.1, refusal=0.2, persona_drift=0.9, oscillation=0.3)
        assert c.dominant_source == TriggerSource.persona_drift

    def test_dominant_source_oscillation(self):
        c = SignalContributions(entropy=0.1, refusal=0.2, persona_drift=0.3, oscillation=0.9)
        assert c.dominant_source == TriggerSource.oscillation_pattern

    def test_max_signal(self):
        c = SignalContributions(entropy=0.1, refusal=0.7, persona_drift=0.3, oscillation=0.5)
        assert c.max_signal == pytest.approx(0.7)

    def test_mean_signal(self):
        c = SignalContributions(entropy=0.2, refusal=0.4, persona_drift=0.6, oscillation=0.8)
        assert c.mean_signal == pytest.approx(0.5)

    def test_get_entropy(self):
        c = SignalContributions(entropy=0.42, refusal=0.0, persona_drift=0.0, oscillation=0.0)
        assert c.get(TriggerSource.entropy_spike) == pytest.approx(0.42)

    def test_get_refusal(self):
        c = SignalContributions(entropy=0.0, refusal=0.55, persona_drift=0.0, oscillation=0.0)
        assert c.get(TriggerSource.refusal_approach) == pytest.approx(0.55)

    def test_get_persona_drift(self):
        c = SignalContributions(entropy=0.0, refusal=0.0, persona_drift=0.77, oscillation=0.0)
        assert c.get(TriggerSource.persona_drift) == pytest.approx(0.77)

    def test_get_oscillation(self):
        c = SignalContributions(entropy=0.0, refusal=0.0, persona_drift=0.0, oscillation=0.33)
        assert c.get(TriggerSource.oscillation_pattern) == pytest.approx(0.33)

    def test_get_combined_signals_returns_zero(self):
        c = SignalContributions(entropy=0.5, refusal=0.5, persona_drift=0.5, oscillation=0.5)
        assert c.get(TriggerSource.combined_signals) == 0.0

    def test_get_manual_returns_zero(self):
        c = SignalContributions(entropy=0.5, refusal=0.5, persona_drift=0.5, oscillation=0.5)
        assert c.get(TriggerSource.manual) == 0.0

    def test_all_zero_dominant_is_entropy(self):
        """When all are 0.0, max picks entropy (first in comparison order)."""
        c = SignalContributions(entropy=0.0, refusal=0.0, persona_drift=0.0, oscillation=0.0)
        # max(0,0,0,0) == 0.0 == self.entropy -> entropy_spike
        assert c.dominant_source == TriggerSource.entropy_spike


# ---------------------------------------------------------------------------
# to_metrics_dict
# ---------------------------------------------------------------------------

class TestToMetricsDict:
    """Metrics dictionary has correct keys and values."""

    def test_keys(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        d = CircuitBreakerIntegration.to_metrics_dict(state)
        expected_keys = {
            "geometry/circuit_breaker_confidence",
            "geometry/circuit_breaker_severity",
            "geometry/circuit_breaker_entropy",
            "geometry/circuit_breaker_refusal",
            "geometry/circuit_breaker_persona",
            "geometry/circuit_breaker_oscillation",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_state(self):
        state = CircuitBreakerIntegration.evaluate(
            InputSignals(entropy_signal=0.6, refusal_distance=0.2)
        )
        d = CircuitBreakerIntegration.to_metrics_dict(state)
        assert d["geometry/circuit_breaker_entropy"] == pytest.approx(0.6)
        assert d["geometry/circuit_breaker_refusal"] == pytest.approx(0.8)
        assert d["geometry/circuit_breaker_confidence"] == pytest.approx(0.5)

    def test_all_values_are_floats(self):
        state = CircuitBreakerIntegration.evaluate(InputSignals())
        d = CircuitBreakerIntegration.to_metrics_dict(state)
        for v in d.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# create_telemetry
# ---------------------------------------------------------------------------

class TestCreateTelemetry:
    """Telemetry snapshot populates all fields."""

    def test_fields_populated(self):
        signals = InputSignals(
            entropy_signal=0.5,
            oscillation_severity=0.3,
            consecutive_oscillations=2,
            token_index=42,
        )
        state = CircuitBreakerIntegration.evaluate(signals)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals)

        assert telemetry.token_index == 42
        assert telemetry.state is state
        assert telemetry.combined_severity == state.severity
        assert telemetry.oscillation_severity == pytest.approx(0.3)
        assert telemetry.consecutive_oscillations == 2

    def test_timestamp_from_state(self):
        signals = InputSignals()
        state = CircuitBreakerIntegration.evaluate(signals)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals)
        assert telemetry.timestamp == state.timestamp

    def test_none_oscillation_severity(self):
        signals = InputSignals()
        state = CircuitBreakerIntegration.evaluate(signals)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals)
        assert telemetry.oscillation_severity is None


# ---------------------------------------------------------------------------
# InputSignals defaults
# ---------------------------------------------------------------------------

class TestInputSignalsDefaults:
    """Default field values on InputSignals."""

    def test_defaults(self):
        s = InputSignals()
        assert s.entropy_signal is None
        assert s.refusal_distance is None
        assert s.is_approaching_refusal is None
        assert s.persona_drift_magnitude is None
        assert s.drifting_traits == []
        assert s.oscillation_severity is None
        assert s.consecutive_oscillations == 0
        assert s.has_oscillation is False
        assert s.token_index == 0

    def test_timestamp_auto_set(self):
        s = InputSignals()
        assert s.timestamp is not None
