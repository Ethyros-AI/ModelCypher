import logging
from typing import Callable, Optional

from .regime_state_detector import RegimeStateDetector, RegimeState

logger = logging.getLogger(__name__)


class DivergenceInterventionMonitor:
    """
    Monitors optimization metrics and intervenes (early stopping/warning)
    if the model enters a DIVERGENT regime.
    """

    def __init__(self, regime_detector: RegimeStateDetector):
        self.regime_detector = regime_detector
        self.intervention_callback: Optional[Callable[[str], None]] = None
        self.last_state: Optional[RegimeState] = None

    def set_intervention_callback(self, callback: Callable[[str], None]):
        self.intervention_callback = callback

    def monitor_step(self, step: int, loss: float, grad_norm: float, entropy: float):
        # Check for divergence based on loss and entropy thresholds
        # (entropy_trajectory stats not needed for this heuristic check)
        
        current_state = RegimeState.ORDERED
        if loss > 10.0 or entropy > 100.0:
            current_state = RegimeState.DISORDERED # Proxy for divergent
        elif entropy < 0.1:
            current_state = RegimeState.ORDERED # Proxy for overfitting/collapsed
            
        # 3. Trigger Interventions
        if current_state == RegimeState.DISORDERED and loss > 8.0:
             self._trigger_intervention(f"DIVERGENCE DETECTED at step {step}: Loss={loss:.2f}")
             
        elif current_state == RegimeState.ORDERED and step > 100 and entropy < 0.01:
             self._trigger_intervention(f"OVERFITTING DETECTED at step {step}: Model has collapsed (Entropy={entropy:.4f})")
             
        self.last_state = current_state

    def _trigger_intervention(self, reason: str):
        logger.warning("INTERVENTION TRIGGERED: %s", reason)
        if self.intervention_callback:
            self.intervention_callback(reason)
