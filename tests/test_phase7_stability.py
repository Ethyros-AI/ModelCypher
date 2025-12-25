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

"""Tests for Phase 7 stability: GAS and CircuitBreaker integration.

Uses pure geometry API - raw measurements, no classification.
"""

from modelcypher.core.domain.entropy.geometric_alignment import (
    GASConfig,
    GeometricAlignmentSystem,
)
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    Configuration,
    InputSignals,
    RecommendedAction,
    TriggerSource,
)


def test_geometric_alignment_sentinel():
    """Test GeometricAlignmentSystem sentinel detects spikes and dips correctly."""
    config = GASConfig.default()
    session = GeometricAlignmentSystem.Session(config)

    # Test 1: Stable entropy (no spike)
    decision = session.observe(entropy=2.0, token_index=0)
    assert not decision.sentinel.is_spike
    assert not decision.sentinel.is_any_dip  # No dip (no negative delta)

    # Test 2: Spike detection (delta > 1.0)
    decision = session.observe(entropy=3.5, token_index=1)  # Delta +1.5
    assert decision.sentinel.is_spike
    assert decision.sentinel.delta_h == 1.5

    # Test 3: Pseudo-dip (drop > 0.3 but entropy > ceiling)
    # Ceiling is 4.0. Let's go up first
    session.observe(entropy=5.0, token_index=2)
    decision = session.observe(entropy=4.5, token_index=3)  # Delta -0.5
    assert decision.sentinel.is_pseudo_dip  # Negative delta but above ceiling

    # Test 4: True dip (drop > 0.3 and entropy < ceiling)
    decision = session.observe(entropy=3.0, token_index=4)  # Delta -1.5, Entropy 3.0 (<4.0)
    assert decision.sentinel.is_true_dip  # Negative delta and below ceiling


def test_geometric_alignment_oscillation_pattern():
    """Test GeometricAlignmentSystem pattern detection for oscillations."""
    config = GASConfig.default()
    session = GeometricAlignmentSystem.Session(config)

    # Simulate oscillation to trigger patterns: high-low-high-low-high
    # These sign changes should be detected
    entropies = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
    for i, e in enumerate(entropies):
        decision = session.observe(entropy=e, token_index=i)

    # Should have sign changes detected in the pattern
    assert decision.pattern.window_sign_changes > 0
    # Severity should be elevated due to oscillation
    assert decision.pattern.severity > 0


def test_circuit_breaker_integration():
    """Test CircuitBreakerIntegration evaluates safety signals correctly."""
    config = Configuration.uniform_weights(trip_threshold=0.75, warning_threshold=0.50)

    # Case 1: Safe state
    signals_safe = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.9,  # Far from refusal
        persona_drift_magnitude=0.1,
    )
    state_safe = CircuitBreakerIntegration.evaluate(signals_safe, config)
    assert not state_safe.is_tripped
    assert state_safe.recommended_action == RecommendedAction.continue_generation

    # Case 2: Tripped by Refusal Approach
    # Use low threshold config for safety testing
    safety_config = Configuration(
        entropy_weight=0.25,
        refusal_weight=0.5,
        persona_drift_weight=0.15,
        oscillation_weight=0.10,
        trip_threshold=0.4,
        warning_threshold=0.25,
        trend_window_size=10,
        enable_auto_escalation=True,
        cooldown_tokens=5,
    )
    signals_refusal = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.1,  # Very close => high contribution
        is_approaching_refusal=True,
        persona_drift_magnitude=0.1,
    )
    state_refusal = CircuitBreakerIntegration.evaluate(signals_refusal, safety_config)

    # Refusal contribution should dominate and trip
    assert state_refusal.is_tripped
    assert state_refusal.trigger_source == TriggerSource.refusal_approach

    # Case 3: Combined signals trip
    signals_combined = InputSignals(
        entropy_signal=0.95,  # Very high
        refusal_distance=0.1,  # Very close to refusal
        is_approaching_refusal=True,
        persona_drift_magnitude=0.85,  # High drift
        has_oscillation=True,
        oscillation_severity=0.9,  # High oscillation
    )

    state_combined = CircuitBreakerIntegration.evaluate(signals_combined, config)
    assert state_combined.is_tripped
    assert state_combined.trigger_source == TriggerSource.combined_signals
