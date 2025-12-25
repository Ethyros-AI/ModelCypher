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

import logging
from typing import Callable

from .regime_state_detector import RegimeStateDetector

logger = logging.getLogger(__name__)


class DivergenceInterventionMonitor:
    """
    Monitors optimization metrics and intervenes based on raw geometric measurements.

    Uses continuous metrics (loss, entropy, grad_norm) directly instead of
    binning into discrete categories.

    Attributes
    ----------
    regime_detector : RegimeStateDetector
        Detector for regime state analysis
    intervention_callback : Callable[[str], None] | None
        Optional callback triggered on intervention
    last_loss : float | None
        Most recent loss value
    last_entropy : float | None
        Most recent entropy value
    """

    def __init__(self, regime_detector: RegimeStateDetector):
        self.regime_detector = regime_detector
        self.intervention_callback: Callable[[str], None] | None = None
        # Track raw metrics - no enum needed
        self.last_loss: float | None = None
        self.last_entropy: float | None = None

    def set_intervention_callback(self, callback: Callable[[str], None]):
        self.intervention_callback = callback

    def monitor_step(self, step: int, loss: float, grad_norm: float, entropy: float):
        """Monitor training step using raw metrics.

        Interventions are triggered based on continuous thresholds that
        indicate training instability.

        Parameters
        ----------
        step : int
            Current training step
        loss : float
            Current loss value
        grad_norm : float
            Current gradient norm
        entropy : float
            Current entropy measurement
        """
        # Check for divergence: high loss indicates explosion
        if loss > 8.0:
            self._trigger_intervention(f"DIVERGENCE DETECTED at step {step}: Loss={loss:.2f}")

        # Check for collapse: very low entropy indicates mode collapse
        elif step > 100 and entropy < 0.01:
            self._trigger_intervention(
                f"COLLAPSE DETECTED at step {step}: Model has collapsed (Entropy={entropy:.4f})"
            )

        # Track for trend analysis
        self.last_loss = loss
        self.last_entropy = entropy

    def _trigger_intervention(self, reason: str):
        logger.warning("INTERVENTION TRIGGERED: %s", reason)
        if self.intervention_callback:
            self.intervention_callback(reason)
