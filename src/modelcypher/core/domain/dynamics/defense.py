from typing import Callable, Optional
from .calorimetry import LinguisticCalorimeter, EnergyState
from .phase_transition import PhaseTransitionTheory, MatterState

class EntropyDefenseMonitor:
    """
    Active monitor that triggers countermeasures if the model enters dangerous thermodynamic states.
    """
    
    def __init__(self, transition_theory: PhaseTransitionTheory):
        self.transition_theory = transition_theory
        self.calorimeter = LinguisticCalorimeter()
        self.defense_protocol: Optional[Callable[[str], None]] = None
        
    def set_defense_protocol(self, callback: Callable[[str], None]):
        self.defense_protocol = callback
        
    def monitor_step(self, step: int, loss: float, grad_norm: float, entropy: float):
        # 1. Measure Energy
        energy_state = self.calorimeter.measure_state(loss, grad_norm, entropy)
        
        # 2. Update Phase Theory
        # Note: In a full system, entropy_delta would come from a tracker.
        # Here we pass 0.0 as a placeholder for delta or simple flux.
        transition_event = self.transition_theory.update(energy_state.temperature, None, step)
        
        # 3. Check for Danger
        current_state = self.transition_theory.current_state
        
        if current_state == MatterState.PLASMA:
             self._trigger_defense(f"PLASMA DETECTED at step {step}: Temp={energy_state.temperature:.2f}")
             
        elif current_state == MatterState.SOLID and step > 100:
             # If solid for too long after warm-up
             self._trigger_defense(f"FROZEN STATE at step {step}: Model is not learning.")

    def _trigger_defense(self, reason: str):
        print(f"!!! ENTROPY DEFENSE ACTIVATED: {reason} !!!")
        if self.defense_protocol:
            self.defense_protocol(reason)
