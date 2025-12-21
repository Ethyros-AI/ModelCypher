from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker

class OptimizationRegime(str, Enum):
    OVERFITTING = "overfitting" # Low entropy/variance
    STABLE = "stable"           # Healthy learning
    VOLATILE = "volatile"       # High variance, exploring
    DIVERGENT = "divergent"     # Exploding gradients/entropy

@dataclass
class RegimeTransitionEvent:
    from_regime: OptimizationRegime
    to_regime: OptimizationRegime
    step: int

class RegimeStateDetector:
    """
    Detects the current optimization regime of the model based on metrics.
    Replaces metaphors with standard ML regime classifications.
    """
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.state_history: List[OptimizationRegime] = []
        self.current_regime: OptimizationRegime = OptimizationRegime.STABLE
        
    def assess_regime(self, perplexity: float, entropy_delta: float) -> OptimizationRegime:
        """
        Determines the current optimization regime.
        """
        if perplexity > 100.0 or entropy_delta > 5.0:
            return OptimizationRegime.DIVERGENT
        elif perplexity > 10.0:
            return OptimizationRegime.VOLATILE
        elif perplexity < 0.8:
            return OptimizationRegime.OVERFITTING
        else:
            return OptimizationRegime.STABLE
            
    def update(self, perplexity: float, entropy_tracker: Optional[EntropyDeltaTracker], step: int) -> Optional[RegimeTransitionEvent]:
        # Using 0.0 placeholder for delta if tracker not provided, similar to previous logic
        entropy_delta = 0.0 
        
        new_regime = self.assess_regime(perplexity, entropy_delta)
        
        event = None
        if new_regime != self.current_regime:
            event = RegimeTransitionEvent(
                from_regime=self.current_regime,
                to_regime=new_regime,
                step=step
            )
            self.current_regime = new_regime
            
        self.state_history.append(new_regime)
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
            
        return event
