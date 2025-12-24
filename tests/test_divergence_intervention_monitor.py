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

"""Tests for divergence intervention monitor.

Tests verify that DivergenceInterventionMonitor correctly:
1. Detects divergence conditions and triggers intervention callbacks
2. Detects overfitting conditions (entropy collapse)
3. Tracks regime state transitions

Note: The monitor uses hardcoded heuristics for regime detection,
not the injected RegimeStateDetector. Tests verify the heuristic behavior.
"""

import pytest
from unittest.mock import Mock

from modelcypher.core.domain.dynamics.monitoring import DivergenceInterventionMonitor
from modelcypher.core.domain.dynamics.regime_state_detector import (
    RegimeStateDetector,
    RegimeState,
)


class TestDivergenceInterventionMonitor:
    """Tests for DivergenceInterventionMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor for testing."""
        # Note: The detector is stored but monitor_step uses internal heuristics
        detector = Mock(spec=RegimeStateDetector)
        return DivergenceInterventionMonitor(detector)

    def test_normal_training_does_not_trigger_intervention(self, monitor):
        """Normal training values should not trigger any intervention."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Simulate normal training progression
        for step in range(1, 50):
            monitor.monitor_step(
                step=step,
                loss=5.0 - (step * 0.05),  # Loss decreasing
                grad_norm=1.0,
                entropy=5.0,  # Moderate entropy
            )

        callback.assert_not_called()

    def test_loss_explosion_triggers_divergence_intervention(self, monitor):
        """Rapidly increasing loss should trigger divergence detection."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Loss explodes above threshold
        monitor.monitor_step(step=100, loss=15.0, grad_norm=50.0, entropy=120.0)

        callback.assert_called_once()
        message = callback.call_args[0][0]
        assert "DIVERGENCE" in message
        assert "15.00" in message  # Should include the loss value

    def test_entropy_explosion_triggers_disordered_state(self, monitor):
        """Very high entropy indicates disordered state."""
        monitor.monitor_step(step=50, loss=5.0, grad_norm=2.0, entropy=150.0)

        assert monitor.last_state == RegimeState.DISORDERED

    def test_entropy_collapse_after_warmup_triggers_overfitting(self, monitor):
        """Near-zero entropy after sufficient training indicates model collapse."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # After 100+ steps, very low entropy indicates overfitting
        monitor.monitor_step(step=200, loss=0.1, grad_norm=0.01, entropy=0.005)

        callback.assert_called_once()
        message = callback.call_args[0][0]
        assert "OVERFITTING" in message or "collapsed" in message.lower()

    def test_early_low_entropy_does_not_trigger_overfitting(self, monitor):
        """Low entropy early in training is expected, not overfitting."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Early in training (step < 100), low entropy is normal
        monitor.monitor_step(step=10, loss=0.5, grad_norm=0.5, entropy=0.005)

        callback.assert_not_called()

    def test_state_transitions_are_tracked(self, monitor):
        """Monitor should track regime state transitions correctly."""
        assert monitor.last_state is None

        # Start with ordered state
        monitor.monitor_step(step=10, loss=2.0, grad_norm=1.0, entropy=0.05)
        assert monitor.last_state == RegimeState.ORDERED

        # Transition to disordered
        monitor.monitor_step(step=20, loss=12.0, grad_norm=10.0, entropy=150.0)
        assert monitor.last_state == RegimeState.DISORDERED

    def test_intervention_callback_receives_step_number(self, monitor):
        """Intervention message should include step number for debugging."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        monitor.monitor_step(step=42, loss=15.0, grad_norm=50.0, entropy=120.0)

        message = callback.call_args[0][0]
        assert "42" in message

    def test_intervention_logs_warning(self, monitor, caplog):
        """Intervention should log warning even without callback."""
        import logging
        with caplog.at_level(logging.WARNING):
            monitor.monitor_step(step=100, loss=15.0, grad_norm=50.0, entropy=120.0)

        assert "INTERVENTION" in caplog.text
