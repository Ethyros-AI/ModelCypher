"""
Circuit Breaker Integration.

Unified safety signal combining multiple geometric and semantic indicators.
Based on Zou 2024 (NeurIPS): "Circuit Breakers for Language Model Harm Reduction"

Synthesizes signals from:
- Semantic Entropy Probe (SEP)
- Refusal Direction Detector
- Persona Vector Monitor
- Geometric Alignment System (GAS)

Ported from TrainingCypher/Domain/Safety/CircuitBreakerIntegration.swift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from modelcypher.core.domain.entropy.geometric_alignment import GeometricAlignmentSystem, InterventionLevel, Decision
from modelcypher.core.domain.geometry.refusal_direction_detector import DistanceMetrics
from modelcypher.core.domain.entropy.sep_probe import SEPProbe  # Assuming this exists from Phase 2
# Note: PersonaVectorMonitor might be missing or mocked if not yet ported. 
# We'll use a simple placeholder if strict typing is needed, but typed as Optional generic for now.
# from modelcypher.core.domain.persona_vector_monitor import TrainingDriftMetrics 


class TriggerSource(str, Enum):
    ENTROPY_SPIKE = "entropy_spike"
    REFUSAL_APPROACH = "refusal_approach"
    PERSONA_DRIFT = "persona_drift"
    OSCILLATION_PATTERN = "oscillation_pattern"
    COMBINED_SIGNALS = "combined_signals"
    MANUAL = "manual"


class RecommendedAction(str, Enum):
    CONTINUE = "continue"
    MONITOR = "monitor"
    REDUCE_TEMPERATURE = "reduce_temperature"
    INSERT_SAFETY_PROMPT = "insert_safety_prompt"
    STOP_GENERATION = "stop_generation"
    HUMAN_REVIEW = "human_review"
    
    @property
    def description(self) -> str:
        if self == RecommendedAction.CONTINUE: return "Continue normally"
        if self == RecommendedAction.MONITOR: return "Monitor more closely"
        if self == RecommendedAction.REDUCE_TEMPERATURE: return "Reduce sampling temperature"
        if self == RecommendedAction.INSERT_SAFETY_PROMPT: return "Insert safety system prompt"
        if self == RecommendedAction.STOP_GENERATION: return "Stop generation"
        if self == RecommendedAction.HUMAN_REVIEW: return "Stop and request human review"
        return "Unknown action"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker evaluation."""
    entropy_weight: float = 0.35
    refusal_weight: float = 0.25
    persona_drift_weight: float = 0.20
    oscillation_weight: float = 0.20
    trip_threshold: float = 0.75
    warning_threshold: float = 0.50
    trend_window_size: int = 10
    enable_auto_escalation: bool = True
    cooldown_tokens: int = 5

    @classmethod
    def default(cls) -> "CircuitBreakerConfig":
        return cls()
        
    @property
    def is_weights_valid(self) -> bool:
        total = self.entropy_weight + self.refusal_weight + self.persona_drift_weight + self.oscillation_weight
        return abs(total - 1.0) < 0.01


@dataclass
class InputSignals:
    """Input signals for circuit breaker evaluation."""
    entropy_signal: Optional[float] = None
    refusal_distance: Optional[float] = None
    is_approaching_refusal: Optional[bool] = None
    persona_drift_magnitude: Optional[float] = None
    drifting_traits: List[str] = field(default_factory=list)
    gas_level: Optional[InterventionLevel] = None
    has_oscillation: bool = False
    token_index: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_components(
        cls,
        sep_prediction: Optional[Any] = None, # SEPProbe.PredictionResult
        refusal_metrics: Optional[DistanceMetrics] = None,
        persona_drift: Optional[Any] = None, # TrainingDriftMetrics
        gas_decision: Optional[Decision] = None,
        token_index: int = 0
    ) -> "InputSignals":
        entropy = getattr(sep_prediction, 'predicted_entropy', None) if sep_prediction else None
        
        drift_mag = getattr(persona_drift, 'overall_drift_magnitude', None) if persona_drift else None
        traits = getattr(persona_drift, 'drifting_traits', []) if persona_drift else []
        
        return cls(
            entropy_signal=entropy,
            refusal_distance=refusal_metrics.distance_to_refusal if refusal_metrics else None,
            is_approaching_refusal=refusal_metrics.is_approaching_refusal if refusal_metrics else None,
            persona_drift_magnitude=drift_mag,
            drifting_traits=traits,
            gas_level=gas_decision.level if gas_decision else None,
            has_oscillation=gas_decision.pattern.is_unstable if gas_decision else False,
            token_index=token_index
        )


@dataclass
class SignalContributions:
    entropy: float
    refusal: float
    persona_drift: float
    oscillation: float

    @property
    def dominant_source(self) -> TriggerSource:
        m = max(self.entropy, self.refusal, self.persona_drift, self.oscillation)
        if m == self.entropy: return TriggerSource.ENTROPY_SPIKE
        if m == self.refusal: return TriggerSource.REFUSAL_APPROACH
        if m == self.persona_drift: return TriggerSource.PERSONA_DRIFT
        return TriggerSource.OSCILLATION_PATTERN


@dataclass
class CircuitBreakerState:
    is_tripped: bool
    severity: float
    trigger_source: Optional[TriggerSource]
    confidence: float
    recommended_action: RecommendedAction
    signal_contributions: SignalContributions
    token_index: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def status_emoji(self) -> str:
        if not self.is_tripped:
            if self.severity < 0.25: return "🟢"
            if self.severity < 0.50: return "🟡"
            return "🟠"
        return "🔴"


@dataclass
class CircuitBreakerTelemetry:
    token_index: int
    timestamp: datetime
    state: CircuitBreakerState
    gas_level: Optional[InterventionLevel]
    combined_severity: float
    any_signal_exceeded: bool


class CircuitBreakerIntegration:
    """Unified safety enforcement integration."""

    @staticmethod
    def evaluate(
        signals: InputSignals,
        configuration: CircuitBreakerConfig = CircuitBreakerConfig.default(),
        previous_state: Optional[CircuitBreakerState] = None
    ) -> CircuitBreakerState:
        
        # 1. Compute Contributions
        entropy_contrib = CircuitBreakerIntegration._compute_entropy_contribution(
            signals.entropy_signal, configuration.entropy_weight
        )
        
        refusal_contrib = CircuitBreakerIntegration._compute_refusal_contribution(
            signals.refusal_distance, signals.is_approaching_refusal, configuration.refusal_weight
        )
        
        persona_contrib = CircuitBreakerIntegration._compute_persona_contribution(
            signals.persona_drift_magnitude, signals.drifting_traits, configuration.persona_drift_weight
        )
        
        oscillation_contrib = CircuitBreakerIntegration._compute_oscillation_contribution(
            signals.has_oscillation, signals.gas_level, configuration.oscillation_weight
        )
        
        contributions = SignalContributions(
            entropy=entropy_contrib,
            refusal=refusal_contrib,
            persona_drift=persona_contrib,
            oscillation=oscillation_contrib
        )
        
        # 2. Combined Severity
        severity = entropy_contrib + refusal_contrib + persona_contrib + oscillation_contrib
        is_tripped = severity >= configuration.trip_threshold
        
        # 3. Trigger Source
        trigger_source = None
        if is_tripped:
            dom = contributions.dominant_source
            if dom == TriggerSource.ENTROPY_SPIKE and entropy_contrib > 0.3:
                trigger_source = TriggerSource.ENTROPY_SPIKE
            elif dom == TriggerSource.REFUSAL_APPROACH and refusal_contrib > 0.2:
                trigger_source = TriggerSource.REFUSAL_APPROACH
            elif dom == TriggerSource.PERSONA_DRIFT and persona_contrib > 0.2:
                trigger_source = TriggerSource.PERSONA_DRIFT
            elif dom == TriggerSource.OSCILLATION_PATTERN and oscillation_contrib > 0.2:
                trigger_source = TriggerSource.OSCILLATION_PATTERN
            else:
                trigger_source = TriggerSource.COMBINED_SIGNALS
        
        # 4. Confidence
        confidence = CircuitBreakerIntegration._compute_confidence(signals)
        
        # 5. Action
        action = CircuitBreakerIntegration._determine_action(
            severity, is_tripped, configuration, signals
        )
        
        return CircuitBreakerState(
            is_tripped=is_tripped,
            severity=severity,
            trigger_source=trigger_source,
            confidence=confidence,
            recommended_action=action,
            signal_contributions=contributions,
            token_index=signals.token_index
        )

    @staticmethod
    def create_telemetry(state: CircuitBreakerState, signals: InputSignals) -> CircuitBreakerTelemetry:
        any_exceeded = (
            (signals.entropy_signal or 0) > 0.7 or
            (signals.refusal_distance or 1.0) < 0.3 or
            (signals.persona_drift_magnitude or 0) > 0.3 or
            signals.has_oscillation
        )
        
        return CircuitBreakerTelemetry(
            token_index=state.token_index,
            timestamp=state.timestamp,
            state=state,
            gas_level=signals.gas_level,
            combined_severity=state.severity,
            any_signal_exceeded=any_exceeded
        )

    # --- Private Helpers ---

    @staticmethod
    def _compute_entropy_contribution(entropy: Optional[float], weight: float) -> float:
        if entropy is None: return 0.0
        # Scale: 0-0.7 -> 0-0.5, 0.7-1.0 -> 0.5-1.0
        scaled = 0.0
        if entropy < 0.7:
            scaled = entropy * 0.71
        else:
            scaled = 0.5 + (entropy - 0.7) * 1.67
        return min(scaled, 1.0) * weight

    @staticmethod
    def _compute_refusal_contribution(distance: Optional[float], is_approaching: Optional[bool], weight: float) -> float:
        if distance is None: return 0.0
        # Invert: close = high concern
        base = 1.0 - distance
        bonus = 0.2 if is_approaching else 0.0
        return min(base + bonus, 1.0) * weight

    @staticmethod
    def _compute_persona_contribution(drift: Optional[float], traits: List[str], weight: float) -> float:
        if drift is None: return 0.0
        scaled = min(drift * 2.0, 1.0)
        penalty = min(len(traits) * 0.1, 0.3)
        return min(scaled + penalty, 1.0) * weight

    @staticmethod
    def _compute_oscillation_contribution(has_oscillation: bool, level: Optional[InterventionLevel], weight: float) -> float:
        contrib = 0.0
        if has_oscillation:
            contrib += 0.5
        
        if level is not None:
             # Using value for int mapping 
             # 0=0, 1=0.1, 2=0.3, 3=0.5, 4=0.7 from Swift
             if level == InterventionLevel.LEVEL_1_GENTLE: contrib += 0.1
             elif level == InterventionLevel.LEVEL_2_CLARIFY: contrib += 0.3
             elif level == InterventionLevel.LEVEL_3_HARD: contrib += 0.5
             elif level == InterventionLevel.LEVEL_4_TERMINATE: contrib += 0.7
             
        return min(contrib, 1.0) * weight

    @staticmethod
    def _compute_confidence(signals: InputSignals) -> float:
        available = 0
        total = 4
        if signals.entropy_signal is not None: available += 1
        if signals.refusal_distance is not None: available += 1
        if signals.persona_drift_magnitude is not None: available += 1
        if signals.gas_level is not None: available += 1
        if total == 0: return 0.0
        return float(available) / total

    @staticmethod
    def _determine_action(
        severity: float, 
        is_tripped: bool, 
        config: CircuitBreakerConfig, 
        signals: InputSignals
    ) -> RecommendedAction:
        if not is_tripped:
            if severity >= config.warning_threshold:
                return RecommendedAction.MONITOR
            return RecommendedAction.CONTINUE
            
        if severity >= 0.95:
            return RecommendedAction.HUMAN_REVIEW
        
        if severity >= 0.85 or signals.gas_level == InterventionLevel.LEVEL_4_TERMINATE:
            return RecommendedAction.STOP_GENERATION
            
        if severity >= 0.75 or signals.is_approaching_refusal:
            return RecommendedAction.INSERT_SAFETY_PROMPT
            
        return RecommendedAction.REDUCE_TEMPERATURE
