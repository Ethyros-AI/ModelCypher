from dataclasses import dataclass
import mlx.core as mx
import math
from typing import Dict, Any

@dataclass
class EnergyState:
    temperature: float
    internal_energy: float
    kinetic_energy: float  # e.g. gradient norm magnitude
    potential_energy: float # e.g. loss magnitude
    is_stable: bool

class LinguisticCalorimeter:
    """
    Measures the 'thermodynamic' state of the model during training/inference.
    Metaphor:
    - Temperature ~ Perplexity / Entropy
    - Kinetic Energy ~ Gradient Norm of recent updates
    - Potential Energy ~ Current Loss
    """
    
    # Constants ported from Swift
    CRITICAL_TEMP_THRESHOLD = 100.0 # High perplexity/entropy
    STABLE_TEMP_RANGE = (0.5, 5.0)

    def measure_state(self, loss: float, gradient_norm: float, entropy: float) -> EnergyState:
        """
        Calculates the thermodynamic state.
        """
        # Temperature is analogous to Perplexity (exp(entropy)) or raw entropy scaling
        # Here we use a calibrated entropy scale.
        temperature = math.exp(entropy) if entropy < 100 else float('inf')
        
        # Kinetic Energy is the "speed" of learning (gradient magnitude)
        kinetic = gradient_norm ** 2
        
        # Potential Energy is the "height" in the loss landscape
        potential = loss
        
        # Internal Energy U = KE + PE
        internal = kinetic + potential
        
        is_stable = (self.STABLE_TEMP_RANGE[0] <= temperature <= self.STABLE_TEMP_RANGE[1])
        
        return EnergyState(
            temperature=temperature,
            internal_energy=internal,
            kinetic_energy=kinetic,
            potential_energy=potential,
            is_stable=is_stable
        )

    def analyze_stability(self, state: EnergyState) -> str:
        if state.temperature > self.CRITICAL_TEMP_THRESHOLD:
            return "PLASMA: Model is Hallucinating/ Diverging"
        elif state.temperature < self.STABLE_TEMP_RANGE[0]:
            return "SOLID: Model is Frozen/ Overfitting"
        elif state.is_stable:
            return "LIQUID: Optimal Learning State"
        else:
            return "GAS: High Volatility"
