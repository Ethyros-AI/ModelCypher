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

from modelcypher.core.domain.entropy.geometric_alignment import GeometricAlignmentSystem
import math
import sys
from modelcypher.core.domain.safety.calibration.geometric_alignment_calibration import (
    GeometricAlignmentCalibration,
)
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    InputSignals,
    TriggerSource,
)


def _div_eps() -> float:
    return math.sqrt(sys.float_info.epsilon)


def _calibration() -> GeometricAlignmentCalibration:
    base_samples = [2.0, 2.1, 2.2, 2.3, 3.2, 4.8, 4.9, 5.0]
    samples = base_samples * 4
    return GeometricAlignmentCalibration.from_entropy_samples("test-model", samples)


def test_geometric_alignment_sentinel():
    """Test GeometricAlignmentSystem sentinel detects spikes and dips correctly."""
    calibration = _calibration()
    session = GeometricAlignmentSystem.Session(calibration)
    ceiling = calibration.sentinel_thresholds.entropy_ceiling
    assert ceiling > 3.0
    assert ceiling < 4.5

    # Test 1: Stable entropy (no spike)
    decision = session.observe(entropy=2.0, token_index=0)
    assert decision.sentinel.entropy == 2.0
    assert decision.sentinel.delta_h == 0.0

    # Test 2: Spike detection (delta > 1.0)
    decision = session.observe(entropy=3.5, token_index=1)  # Delta +1.5
    assert decision.sentinel.delta_h == 1.5
    assert decision.sentinel.is_negative_delta is False

    # Test 3: Pseudo-dip (drop below but still above ceiling)
    session.observe(entropy=5.0, token_index=2)
    decision = session.observe(entropy=4.5, token_index=3)  # Delta -0.5
    assert decision.sentinel.is_negative_delta is True

    # Test 4: True dip (drop below ceiling)
    decision = session.observe(entropy=3.0, token_index=4)  # Delta -1.5, Entropy 3.0 (< ceiling)
    assert decision.sentinel.delta_h == -1.5


def test_geometric_alignment_oscillation_pattern():
    """Test GeometricAlignmentSystem pattern detection for oscillations."""
    calibration = _calibration()
    session = GeometricAlignmentSystem.Session(calibration)

    # Simulate oscillation to trigger patterns: high-low-high-low-high
    # These sign changes should be detected
    entropies = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
    for i, e in enumerate(entropies):
        decision = session.observe(entropy=e, token_index=i)

    # Should have sign changes detected in the pattern
    assert decision.pattern.window_sign_changes > 0


def test_circuit_breaker_integration():
    """Test CircuitBreakerIntegration evaluates safety signals correctly."""
    eps = _div_eps()

    signals_safe = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.9,
        persona_drift_magnitude=0.1,
    )
    state_safe = CircuitBreakerIntegration.evaluate(signals_safe)

    expected_refusal = 1.0 - 0.9
    expected_severity = (0.2 + expected_refusal + 0.1 + 0.0) / 4.0
    assert abs(state_safe.severity - expected_severity) < eps
    assert state_safe.dominant_source == TriggerSource.entropy_spike
    assert abs(state_safe.signal_contributions.refusal - expected_refusal) < eps
    assert abs(state_safe.confidence - 0.75) < eps

    signals_refusal = InputSignals(
        entropy_signal=0.2,
        refusal_distance=0.1,
        persona_drift_magnitude=0.1,
    )
    state_refusal = CircuitBreakerIntegration.evaluate(signals_refusal)
    assert state_refusal.dominant_source == TriggerSource.refusal_approach
