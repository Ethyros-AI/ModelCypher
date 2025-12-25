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
    Monitors optimization metrics and intervenes (early stopping/warning)
    based on raw geometric measurements.

    Uses continuous metrics (loss, entropy, grad_norm) directly instead of
    binning into ORDERED/DISORDERED categories. The raw values ARE the signal.
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

        The raw values (loss, entropy, grad_norm) are the signal - no need to
        classify them into ORDERED/DISORDERED categories.

        Interventions are triggered based on continuous thresholds that
        indicate training instability.
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
