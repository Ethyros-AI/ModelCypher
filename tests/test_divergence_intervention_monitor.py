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

Tests verify that DivergenceInterventionMonitor:
- Emits raw measurements (loss/entropy/grad_norm) and deltas
- Tracks the latest raw values without classification
- Forwards measurements to callbacks for external decisions
"""

from unittest.mock import Mock

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.dynamics.monitoring import (
    DivergenceInterventionMonitor,
    DivergenceMeasurement,
)
from modelcypher.core.domain.dynamics.regime_state_detector import RegimeStateDetector
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


class TestDivergenceInterventionMonitor:
    """Tests for DivergenceInterventionMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor for testing."""
        detector = Mock(spec=RegimeStateDetector)
        return DivergenceInterventionMonitor(detector)

    def test_monitor_step_returns_measurement(self, monitor):
        """Monitor step returns raw measurements and no deltas on first step."""
        measurement = monitor.monitor_step(
            step=1, loss=2.0, grad_norm=1.0, entropy=0.5
        )

        assert isinstance(measurement, DivergenceMeasurement)
        assert measurement.step == 1
        assert abs(measurement.loss - 2.0) <= _eps()
        assert abs(measurement.grad_norm - 1.0) <= _eps()
        assert abs(measurement.entropy - 0.5) <= _eps()
        assert measurement.loss_delta is None
        assert measurement.entropy_delta is None
        assert abs(monitor.last_loss - 2.0) <= _eps()
        assert abs(monitor.last_entropy - 0.5) <= _eps()

    def test_monitor_step_computes_deltas(self, monitor):
        """Monitor computes deltas between successive steps."""
        monitor.monitor_step(step=1, loss=2.0, grad_norm=1.0, entropy=0.5)
        measurement = monitor.monitor_step(
            step=2, loss=3.5, grad_norm=1.5, entropy=0.2
        )

        assert abs(measurement.loss_delta - 1.5) <= _eps()
        assert abs(measurement.entropy_delta + 0.3) <= _eps()

    def test_callback_receives_measurement(self, monitor):
        """Callback receives raw measurements for external decisions."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        measurement = monitor.monitor_step(
            step=42, loss=4.0, grad_norm=2.0, entropy=1.25
        )

        callback.assert_called_once_with(measurement)
        assert callback.call_args[0][0].step == 42

    def test_callback_receives_sequence(self, monitor):
        """Callback receives each step measurement in order."""
        callback = Mock()
        monitor.set_intervention_callback(callback)

        for step in range(1, 4):
            monitor.monitor_step(step=step, loss=step * 1.0, grad_norm=1.0, entropy=0.5)

        assert callback.call_count == 3
        steps = [call_args[0][0].step for call_args in callback.call_args_list]
        assert steps == [1, 2, 3]
