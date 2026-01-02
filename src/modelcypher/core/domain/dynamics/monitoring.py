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

from dataclasses import dataclass
from typing import Callable

from .regime_state_detector import RegimeStateDetector


@dataclass(frozen=True)
class DivergenceMeasurement:
    """Raw divergence-related measurements for a single step."""

    step: int
    loss: float
    grad_norm: float
    entropy: float
    loss_delta: float | None
    entropy_delta: float | None


class DivergenceInterventionMonitor:
    """Monitor optimization metrics and emit raw measurements."""

    def __init__(self, regime_detector: RegimeStateDetector):
        self.regime_detector = regime_detector
        self.intervention_callback: Callable[[DivergenceMeasurement], None] | None = None
        self.last_loss: float | None = None
        self.last_entropy: float | None = None

    def set_intervention_callback(
        self, callback: Callable[[DivergenceMeasurement], None]
    ) -> None:
        self.intervention_callback = callback

    def monitor_step(
        self, step: int, loss: float, grad_norm: float, entropy: float
    ) -> DivergenceMeasurement:
        """Monitor training step and emit raw measurements."""
        loss_delta = None if self.last_loss is None else loss - self.last_loss
        entropy_delta = (
            None if self.last_entropy is None else entropy - self.last_entropy
        )

        measurement = DivergenceMeasurement(
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            entropy=entropy,
            loss_delta=loss_delta,
            entropy_delta=entropy_delta,
        )

        self.last_loss = loss
        self.last_entropy = entropy

        if self.intervention_callback:
            self.intervention_callback(measurement)

        return measurement
