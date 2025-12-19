from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class InterventionLevel(str, Enum):
    level0_continue = "level0Continue"
    level1_gentle = "level1Gentle"
    level2_clarify = "level2Clarify"
    level3_hard = "level3Hard"
    level4_terminate = "level4Terminate"


@dataclass(frozen=True)
class Configuration:
    entropy_weight: float
    refusal_weight: float
    persona_drift_weight: float
    oscillation_weight: float
    trip_threshold: float
    warning_threshold: float
    trend_window_size: int
    enable_auto_escalation: bool
    cooldown_tokens: int

    @staticmethod
    def default() -> "Configuration":
        return Configuration(
            entropy_weight=0.35,
            refusal_weight=0.25,
            persona_drift_weight=0.20,
            oscillation_weight=0.20,
            trip_threshold=0.75,
            warning_threshold=0.50,
            trend_window_size=10,
            enable_auto_escalation=True,
            cooldown_tokens=5,
        )

    @staticmethod
    def conservative() -> "Configuration":
        return Configuration(
            entropy_weight=0.30,
            refusal_weight=0.35,
            persona_drift_weight=0.20,
            oscillation_weight=0.15,
            trip_threshold=0.60,
            warning_threshold=0.40,
            trend_window_size=15,
            enable_auto_escalation=True,
            cooldown_tokens=10,
        )

    @staticmethod
    def permissive() -> "Configuration":
        return Configuration(
            entropy_weight=0.40,
            refusal_weight=0.20,
            persona_drift_weight=0.20,
            oscillation_weight=0.20,
            trip_threshold=0.85,
            warning_threshold=0.65,
            trend_window_size=8,
            enable_auto_escalation=False,
            cooldown_tokens=3,
        )

    @property
    def is_weights_valid(self) -> bool:
        total = self.entropy_weight + self.refusal_weight + self.persona_drift_weight + self.oscillation_weight
        return abs(total - 1.0) < 0.01


@dataclass(frozen=True)
class InputSignals:
    entropy_signal: Optional[float] = None
    refusal_distance: Optional[float] = None
    is_approaching_refusal: Optional[bool] = None
    persona_drift_magnitude: Optional[float] = None
    drifting_traits: list[str] = field(default_factory=list)
    gas_level: Optional[InterventionLevel] = None
    has_oscillation: bool = False
    token_index: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TriggerSource(str, Enum):
    entropy_spike = "entropySpike"
    refusal_approach = "refusalApproach"
    persona_drift = "personaDrift"
    oscillation_pattern = "oscillationPattern"
    combined_signals = "combinedSignals"
    manual = "manual"


class RecommendedAction(str, Enum):
    continue_generation = "continue"
    monitor = "monitor"
    reduce_temperature = "reduceTemperature"
    insert_safety_prompt = "insertSafetyPrompt"
    stop_generation = "stopGeneration"
    human_review = "humanReview"

    @property
    def description(self) -> str:
        if self is RecommendedAction.continue_generation:
            return "Continue normally"
        if self is RecommendedAction.monitor:
            return "Monitor more closely"
        if self is RecommendedAction.reduce_temperature:
            return "Reduce sampling temperature"
        if self is RecommendedAction.insert_safety_prompt:
            return "Insert safety system prompt"
        if self is RecommendedAction.stop_generation:
            return "Stop generation"
        return "Stop and request human review"


@dataclass(frozen=True)
class SignalContributions:
    entropy: float
    refusal: float
    persona_drift: float
    oscillation: float

    @property
    def dominant_source(self) -> TriggerSource:
        max_value = max(self.entropy, self.refusal, self.persona_drift, self.oscillation)
        if max_value == self.entropy:
            return TriggerSource.entropy_spike
        if max_value == self.refusal:
            return TriggerSource.refusal_approach
        if max_value == self.persona_drift:
            return TriggerSource.persona_drift
        return TriggerSource.oscillation_pattern


@dataclass(frozen=True)
class CircuitBreakerState:
    is_tripped: bool
    severity: float
    trigger_source: Optional[TriggerSource]
    confidence: float
    recommended_action: RecommendedAction
    signal_contributions: SignalContributions
    token_index: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def interpretation(self) -> str:
        if not self.is_tripped:
            if self.severity < 0.25:
                return "All clear - generation proceeding normally"
            if self.severity < 0.50:
                return "Low concern - monitoring signals"
            return "Elevated concern - close monitoring recommended"

        source = self.trigger_source
        if source is None:
            return "Circuit breaker tripped - unknown trigger"
        if source is TriggerSource.entropy_spike:
            return "HIGH UNCERTAINTY: Semantic entropy spike detected - potential hallucination"
        if source is TriggerSource.refusal_approach:
            return "SAFETY CONCERN: Model approaching refusal direction"
        if source is TriggerSource.persona_drift:
            return "ALIGNMENT DRIFT: Persona vectors shifting from baseline"
        if source is TriggerSource.oscillation_pattern:
            return "INSTABILITY: Oscillation pattern detected in generation"
        if source is TriggerSource.combined_signals:
            return "MULTIPLE CONCERNS: Combined safety signals exceeded threshold"
        return "MANUAL TRIP: Circuit breaker manually activated"


@dataclass(frozen=True)
class CircuitBreakerTelemetry:
    token_index: int
    timestamp: datetime
    state: CircuitBreakerState
    gas_level: Optional[InterventionLevel]
    combined_severity: float
    any_signal_exceeded: bool


class CircuitBreakerIntegration:
    @staticmethod
    def evaluate(
        signals: InputSignals,
        configuration: Configuration | None = None,
        previous_state: CircuitBreakerState | None = None,
    ) -> CircuitBreakerState:
        config = configuration or Configuration.default()

        entropy_contribution = CircuitBreakerIntegration._compute_entropy_contribution(
            signals.entropy_signal, config.entropy_weight
        )
        refusal_contribution = CircuitBreakerIntegration._compute_refusal_contribution(
            signals.refusal_distance, signals.is_approaching_refusal, config.refusal_weight
        )
        persona_contribution = CircuitBreakerIntegration._compute_persona_contribution(
            signals.persona_drift_magnitude, signals.drifting_traits, config.persona_drift_weight
        )
        oscillation_contribution = CircuitBreakerIntegration._compute_oscillation_contribution(
            signals.has_oscillation, signals.gas_level, config.oscillation_weight
        )

        contributions = SignalContributions(
            entropy=entropy_contribution,
            refusal=refusal_contribution,
            persona_drift=persona_contribution,
            oscillation=oscillation_contribution,
        )
        severity = entropy_contribution + refusal_contribution + persona_contribution + oscillation_contribution
        is_tripped = severity >= config.trip_threshold

        trigger_source: Optional[TriggerSource]
        if is_tripped:
            dominant = contributions.dominant_source
            if dominant is TriggerSource.entropy_spike and entropy_contribution > 0.3:
                trigger_source = TriggerSource.entropy_spike
            elif dominant is TriggerSource.refusal_approach and refusal_contribution > 0.2:
                trigger_source = TriggerSource.refusal_approach
            elif dominant is TriggerSource.persona_drift and persona_contribution > 0.2:
                trigger_source = TriggerSource.persona_drift
            elif dominant is TriggerSource.oscillation_pattern and oscillation_contribution > 0.2:
                trigger_source = TriggerSource.oscillation_pattern
            else:
                trigger_source = TriggerSource.combined_signals
        else:
            trigger_source = None

        confidence = CircuitBreakerIntegration._compute_confidence(signals)
        action = CircuitBreakerIntegration._determine_action(severity, is_tripped, config, signals)

        return CircuitBreakerState(
            is_tripped=is_tripped,
            severity=severity,
            trigger_source=trigger_source,
            confidence=confidence,
            recommended_action=action,
            signal_contributions=contributions,
            token_index=signals.token_index,
            timestamp=signals.timestamp,
        )

    @staticmethod
    def create_telemetry(state: CircuitBreakerState, signals: InputSignals) -> CircuitBreakerTelemetry:
        any_exceeded = (
            (signals.entropy_signal or 0) > 0.7
            or (signals.refusal_distance or 1.0) < 0.3
            or (signals.persona_drift_magnitude or 0) > 0.3
            or signals.has_oscillation
        )
        return CircuitBreakerTelemetry(
            token_index=state.token_index,
            timestamp=state.timestamp,
            state=state,
            gas_level=signals.gas_level,
            combined_severity=state.severity,
            any_signal_exceeded=any_exceeded,
        )

    @staticmethod
    def to_metrics_dict(state: CircuitBreakerState) -> dict[str, float]:
        return {
            "geometry/circuit_breaker_tripped": 1.0 if state.is_tripped else 0.0,
            "geometry/circuit_breaker_confidence": float(state.confidence),
            "geometry/circuit_breaker_severity": float(state.severity),
            "geometry/circuit_breaker_entropy": float(state.signal_contributions.entropy),
            "geometry/circuit_breaker_refusal": float(state.signal_contributions.refusal),
            "geometry/circuit_breaker_persona": float(state.signal_contributions.persona_drift),
            "geometry/circuit_breaker_oscillation": float(state.signal_contributions.oscillation),
        }

    @staticmethod
    def _compute_entropy_contribution(entropy: Optional[float], weight: float) -> float:
        if entropy is None:
            return 0.0
        if entropy < 0.7:
            scaled = entropy * 0.71
        else:
            scaled = 0.5 + (entropy - 0.7) * 1.67
        return min(scaled, 1.0) * weight

    @staticmethod
    def _compute_refusal_contribution(
        distance: Optional[float],
        is_approaching: Optional[bool],
        weight: float,
    ) -> float:
        if distance is None:
            return 0.0
        base_contribution = 1.0 - distance
        approach_bonus = 0.2 if is_approaching else 0.0
        scaled = min(base_contribution + approach_bonus, 1.0)
        return scaled * weight

    @staticmethod
    def _compute_persona_contribution(
        drift_magnitude: Optional[float],
        drifting_traits: list[str],
        weight: float,
    ) -> float:
        if drift_magnitude is None:
            return 0.0
        drift_scaled = min(drift_magnitude * 2.0, 1.0)
        trait_penalty = min(len(drifting_traits) * 0.1, 0.3)
        return min(drift_scaled + trait_penalty, 1.0) * weight

    @staticmethod
    def _compute_oscillation_contribution(
        has_oscillation: bool,
        gas_level: Optional[InterventionLevel],
        weight: float,
    ) -> float:
        contribution = 0.0
        if has_oscillation:
            contribution += 0.5

        if gas_level is not None:
            if gas_level is InterventionLevel.level1_gentle:
                contribution += 0.1
            elif gas_level is InterventionLevel.level2_clarify:
                contribution += 0.3
            elif gas_level is InterventionLevel.level3_hard:
                contribution += 0.5
            elif gas_level is InterventionLevel.level4_terminate:
                contribution += 0.7

        return min(contribution, 1.0) * weight

    @staticmethod
    def _compute_confidence(signals: InputSignals) -> float:
        total_signals = 4
        available = 0
        if signals.entropy_signal is not None:
            available += 1
        if signals.refusal_distance is not None:
            available += 1
        if signals.persona_drift_magnitude is not None:
            available += 1
        if signals.gas_level is not None:
            available += 1
        return float(available) / float(total_signals)

    @staticmethod
    def _determine_action(
        severity: float,
        is_tripped: bool,
        configuration: Configuration,
        signals: InputSignals,
    ) -> RecommendedAction:
        if not is_tripped:
            if severity >= configuration.warning_threshold:
                return RecommendedAction.monitor
            return RecommendedAction.continue_generation

        if severity >= 0.95:
            return RecommendedAction.human_review
        if severity >= 0.85 or signals.gas_level is InterventionLevel.level4_terminate:
            return RecommendedAction.stop_generation
        if severity >= 0.75 or signals.is_approaching_refusal:
            return RecommendedAction.insert_safety_prompt
        return RecommendedAction.reduce_temperature
