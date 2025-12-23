"""Tests for divergence intervention monitor."""

import pytest
from unittest.mock import Mock, MagicMock

from modelcypher.core.domain.dynamics.monitoring import DivergenceInterventionMonitor
from modelcypher.core.domain.dynamics.regime_state_detector import (
    RegimeStateDetector,
    RegimeState,
)


class TestDivergenceInterventionMonitor:
    """Tests for DivergenceInterventionMonitor class."""

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector."""
        return Mock(spec=RegimeStateDetector)

    @pytest.fixture
    def monitor(self, mock_regime_detector):
        """Create a monitor with mock detector."""
        return DivergenceInterventionMonitor(mock_regime_detector)

    def test_initialization(self, mock_regime_detector):
        """Test monitor initializes correctly."""
        monitor = DivergenceInterventionMonitor(mock_regime_detector)

        assert monitor.regime_detector is mock_regime_detector
        assert monitor.metric_calculator is not None
        assert monitor.intervention_callback is None
        assert monitor.last_state is None

    def test_set_intervention_callback(self, monitor):
        """Test setting intervention callback."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        assert monitor.intervention_callback is callback

    def test_monitor_step_with_normal_values(self, monitor):
        """Test monitoring with normal training values."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Normal values should not trigger intervention
        monitor.monitor_step(step=50, loss=2.5, grad_norm=1.0, entropy=5.0)

        callback.assert_not_called()
        assert monitor.last_state == RegimeState.ORDERED

    def test_monitor_step_detects_divergence_high_loss(self, monitor, capsys):
        """Test that high loss triggers divergence detection."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # High loss + disordered state triggers intervention
        monitor.monitor_step(step=100, loss=15.0, grad_norm=50.0, entropy=120.0)

        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert "DIVERGENCE DETECTED" in call_args

    def test_monitor_step_detects_overfitting(self, monitor):
        """Test that very low entropy triggers overfitting detection."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Very low entropy after many steps indicates collapse
        monitor.monitor_step(step=200, loss=0.5, grad_norm=0.01, entropy=0.005)

        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert "OVERFITTING DETECTED" in call_args

    def test_monitor_step_no_overfitting_early_steps(self, monitor):
        """Test that low entropy doesn't trigger on early steps."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        # Low entropy but early in training (step < 100)
        monitor.monitor_step(step=50, loss=0.5, grad_norm=0.01, entropy=0.005)

        # Should not trigger overfitting intervention on early steps
        # The implementation checks step > 100
        callback.assert_not_called()

    def test_monitor_step_prints_intervention(self, monitor, capsys):
        """Test that intervention prints message."""
        monitor.monitor_step(step=100, loss=15.0, grad_norm=50.0, entropy=120.0)

        captured = capsys.readouterr()
        assert "INTERVENTION TRIGGERED" in captured.out

    def test_monitor_step_updates_last_state(self, monitor):
        """Test that last_state is updated after each step."""
        assert monitor.last_state is None

        monitor.monitor_step(step=10, loss=2.0, grad_norm=1.0, entropy=5.0)
        assert monitor.last_state == RegimeState.ORDERED

        monitor.monitor_step(step=20, loss=15.0, grad_norm=10.0, entropy=150.0)
        assert monitor.last_state == RegimeState.DISORDERED

    def test_monitor_step_without_callback(self, monitor, capsys):
        """Test monitoring without a callback set."""
        # Should not raise even without callback
        monitor.monitor_step(step=100, loss=15.0, grad_norm=50.0, entropy=120.0)

        # Should still print intervention message
        captured = capsys.readouterr()
        assert "INTERVENTION TRIGGERED" in captured.out

    def test_ordered_state_threshold(self, monitor):
        """Test ordered state detection with low entropy."""
        monitor.monitor_step(step=10, loss=1.0, grad_norm=0.5, entropy=0.05)
        assert monitor.last_state == RegimeState.ORDERED

    def test_disordered_state_high_entropy(self, monitor):
        """Test disordered state detection with high entropy."""
        monitor.monitor_step(step=10, loss=5.0, grad_norm=5.0, entropy=150.0)
        assert monitor.last_state == RegimeState.DISORDERED

    def test_disordered_state_high_loss(self, monitor):
        """Test disordered state detection with high loss."""
        monitor.monitor_step(step=10, loss=15.0, grad_norm=5.0, entropy=50.0)
        assert monitor.last_state == RegimeState.DISORDERED

    def test_callback_receives_step_info(self, monitor):
        """Test that callback receives step number in message."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        monitor.monitor_step(step=42, loss=15.0, grad_norm=50.0, entropy=120.0)

        call_args = callback.call_args[0][0]
        assert "step 42" in call_args
