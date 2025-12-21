from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker, EntropySample

class MatterState(str, Enum):
    SOLID = "solid"   # Frozen, deterministic, potential overfitting
    LIQUID = "liquid" # Flexible, adaptive, measuring reality (Good)
    GAS = "gas"       # Chaotic, high variance, exploring (Okay in bursts)
    PLASMA = "plasma" # Divergent, broken (Bad)

@dataclass
class PhaseTransitionEvent:
    from_state: MatterState
    to_state: MatterState
    step: int
    energy_delta: float

class PhaseTransitionTheory:
    """
    Detects macroscopic state changes in the model's behavior over time.
    """
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.state_history: List[MatterState] = []
        self.current_state: MatterState = MatterState.LIQUID
        
    def assess_state(self, temperature: float, entropy_delta: float) -> MatterState:
        """
        Determines the current state of matter based on temperature and entropy flux.
        """
        # Heuristics ported from Swift
        if temperature > 100.0 or entropy_delta > 5.0:
            return MatterState.PLASMA
        elif temperature > 10.0:
            return MatterState.GAS
        elif temperature < 0.8:
            return MatterState.SOLID
        else:
            return MatterState.LIQUID
            
    def update(self, temperature: float, entropy_tracker: EntropyDeltaTracker, step: int) -> Optional[PhaseTransitionEvent]:
        current_entropy_delta = 0.0
        # In a real scenario, we'd query the tracker for the latest drift
        # For this port, we assume entropy_delta is derived or passed appropriately.
        # Let's use a simpler signature for now or assume tracker has latest.
        
        # Simplified logic for pure port without coupling too tightly to tracker mechanics yet
        # We need the *rate of change* of entropy.
        
        new_state = self.assess_state(temperature, current_entropy_delta)
        
        event = None
        if new_state != self.current_state:
            event = PhaseTransitionEvent(
                from_state=self.current_state,
                to_state=new_state,
                step=step,
                energy_delta=0.0 # Placeholder
            )
            self.current_state = new_state
            
        self.state_history.append(new_state)
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
            
        return event
