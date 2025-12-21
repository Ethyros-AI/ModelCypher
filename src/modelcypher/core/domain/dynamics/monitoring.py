from typing import Callable, Optional
from .metrics import OptimizationMetricCalculator, OptimizationState
from .regimes import RegimeStateDetector, OptimizationRegime

class DivergenceInterventionMonitor:
    """
    Monitors optimization metrics and intervenes (early stopping/warning) 
    if the model enters a DIVERGENT regime.
    """
    
    def __init__(self, regime_detector: RegimeStateDetector):
        self.regime_detector = regime_detector
        self.metric_calculator = OptimizationMetricCalculator()
        self.intervention_callback: Optional[Callable[[str], None]] = None
        
    def set_intervention_callback(self, callback: Callable[[str], None]):
        self.intervention_callback = callback
        
    def monitor_step(self, step: int, loss: float, grad_norm: float, entropy: float):
        # 1. Calculate Standard Metrics
        state = self.metric_calculator.calculate_metrics(loss, grad_norm, entropy)
        
        # 2. Update Regime State
        transition_event = self.regime_detector.update(state.perplexity, None, step)
        
        # 3. Check for Divergence
        current_regime = self.regime_detector.current_regime
        
        if current_regime == OptimizationRegime.DIVERGENT:
             self._trigger_intervention(f"DIVERGENCE DETECTED at step {step}: Perplexity={state.perplexity:.2f}")
             
        elif current_regime == OptimizationRegime.OVERFITTING and step > 100:
             self._trigger_intervention(f"OVERFITTING DETECTED at step {step}: Model has collapsed to low entropy.")

    def _trigger_intervention(self, reason: str):
        print(f"!!! INTERVENTION TRIGGERED: {reason} !!!")
        if self.intervention_callback:
            self.intervention_callback(reason)
