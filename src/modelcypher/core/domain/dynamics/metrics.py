from dataclasses import dataclass
import mlx.core as mx
import math
from typing import Dict, Any

@dataclass
class OptimizationState:
    perplexity: float
    total_energy: float # scalar optimized metric
    gradient_norm: float
    loss: float
    is_stable: bool

class OptimizationMetricCalculator:
    """
    Calculates standard high-dimensional optimization metrics.
    - Perplexity: exp(entropy)
    - Gradient Norm: L2 norm of update vector
    - Loss: Function value (Potential)
    """
    
    # Thresholds for divergence detection
    CRITICAL_PERPLEXITY_THRESHOLD = 100.0 
    STABLE_PERPLEXITY_RANGE = (0.5, 5.0)

    def calculate_metrics(self, loss: float, gradient_norm: float, entropy: float) -> OptimizationState:
        """
        Computes optimization state vector.
        """
        perplexity = math.exp(entropy) if entropy < 100 else float('inf')
        
        # Total "Energy" in optimization landscape (Hamiltonian view)
        # H = T + V (Kinetic + Potential)
        # Kinetic ~ Gradient Norm^2
        # Potential ~ Loss
        kinetic = gradient_norm ** 2
        potential = loss
        total_energy = kinetic + potential
        
        is_stable = (self.STABLE_PERPLEXITY_RANGE[0] <= perplexity <= self.STABLE_PERPLEXITY_RANGE[1])
        
        return OptimizationState(
            perplexity=perplexity,
            total_energy=total_energy,
            gradient_norm=gradient_norm,
            loss=loss,
            is_stable=is_stable
        )

    def analyze_stability(self, state: OptimizationState) -> str:
        if state.perplexity > self.CRITICAL_PERPLEXITY_THRESHOLD:
            return "DIVERGENT"
        elif state.perplexity < self.STABLE_PERPLEXITY_RANGE[0]:
            return "OVERFITTING"
        elif state.is_stable:
            return "STABLE"
        else:
            return "VOLATILE"
