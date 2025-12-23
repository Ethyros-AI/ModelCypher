
import math
from modelcypher.core.domain.entropy.geometric_alignment import (
    GeometricAlignmentSystem, GASConfig, SentinelSample, InterventionLevel, DipClassification
)
from modelcypher.core.domain.safety.circuit_breaker import (
    CircuitBreakerIntegration, CircuitBreakerConfig, InputSignals, TriggerSource, RecommendedAction
)
from modelcypher.core.domain.geometry.refusal_direction_detector import DistanceMetrics


def test_geometric_alignment_sentinel():
    """Test GeometricAlignmentSystem sentinel detects spikes and dips correctly."""
    config = GASConfig.default()
    session = GeometricAlignmentSystem.Session(config)

    # Test 1: Stable entropy (no spike)
    decision = session.observe(entropy=2.0, token_index=0)
    assert not decision.sentinel.is_spike
    assert decision.sentinel.dip_classification == DipClassification.NONE

    # Test 2: Spike detection (delta > 1.0)
    decision = session.observe(entropy=3.5, token_index=1)  # Delta +1.5
    assert decision.sentinel.is_spike
    assert decision.sentinel.delta_h == 1.5

    # Test 3: Pseudo-dip (drop > 0.3 but entropy > ceiling)
    # Ceiling is 4.0. Let's go up first
    session.observe(entropy=5.0, token_index=2)
    decision = session.observe(entropy=4.5, token_index=3)  # Delta -0.5
    assert decision.sentinel.dip_classification == DipClassification.PSEUDO_DIP

    # Test 4: True dip (drop > 0.3 and entropy < ceiling)
    decision = session.observe(entropy=3.0, token_index=4)  # Delta -1.5, Entropy 3.0 (<4.0)
    assert decision.sentinel.dip_classification == DipClassification.TRUE_DIP


def test_geometric_alignment_director():
    """Test GeometricAlignmentSystem director escalates intervention levels on oscillation."""
    config = GASConfig.default()
    # Lower threshold for testing
    config.oscillator.consecutive_oscillations_for_termination = 2
    session = GeometricAlignmentSystem.Session(config)

    # Simulate oscillation to trigger levels
    # Need sign changes: + - + -
    entropies = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
    for i, e in enumerate(entropies):
        decision = session.observe(entropy=e, token_index=i)

    # Should have some level escalation by now due to sign changes
    assert decision.level >= InterventionLevel.LEVEL_1_GENTLE


def test_circuit_breaker_integration():
    """Test CircuitBreakerIntegration evaluates safety signals correctly."""
    config = CircuitBreakerConfig.default()

    # Case 1: Safe state
    signals_safe = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.9,  # Far from refusal
        persona_drift_magnitude=0.1
    )
    state_safe = CircuitBreakerIntegration.evaluate(signals_safe, config)
    assert not state_safe.is_tripped
    assert state_safe.recommended_action == RecommendedAction.CONTINUE

    # Case 2: Tripped by Refusal Approach
    # Use conservative config for safety testing
    safety_config = CircuitBreakerConfig(
        refusal_weight=0.5,
        trip_threshold=0.4
    )
    signals_refusal = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.1,  # Very close => ~1.0 contribution base
        is_approaching_refusal=True,
        persona_drift_magnitude=0.1
    )
    state_refusal = CircuitBreakerIntegration.evaluate(signals_refusal, safety_config)

    # Refusal contrib: (0.9 + 0.2) -> 1.0 * 0.5 = 0.5
    # Severity >= 0.5 > 0.4 -> Trip
    assert state_refusal.is_tripped
    assert state_refusal.trigger_source == TriggerSource.REFUSAL_APPROACH
    assert state_refusal.recommended_action == RecommendedAction.INSERT_SAFETY_PROMPT

    # Case 3: Combined signals trip
    signals_combined = InputSignals(
        entropy_signal=0.8,  # Very high (>0.7), contrib ~0.23
        refusal_distance=0.3,  # Close -> 0.7, contrib ~0.175
        is_approaching_refusal=True,  # +0.2 bonus -> 0.9, contrib ~0.225
        persona_drift_magnitude=0.6,  # High -> 1.0, contrib ~0.2
        has_oscillation=True  # +0.5 -> contrib ~0.1
    )
    # Total ~ 0.23 + 0.225 + 0.2 + 0.1 = 0.755 > 0.75

    state_combined = CircuitBreakerIntegration.evaluate(signals_combined, config)
    assert state_combined.is_tripped
    assert state_combined.trigger_source == TriggerSource.COMBINED_SIGNALS
