"""Compatibility wrapper for circuit breaker integration."""
from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.safety import circuit_breaker_integration as core

InputSignals = core.InputSignals
TriggerSource = core.TriggerSource
RecommendedAction = core.RecommendedAction
CircuitBreakerState = core.CircuitBreakerState


@dataclass(frozen=True)
class CircuitBreakerConfig:
    entropy_weight: float = core.Configuration.default().entropy_weight
    refusal_weight: float = core.Configuration.default().refusal_weight
    persona_drift_weight: float = core.Configuration.default().persona_drift_weight
    oscillation_weight: float = core.Configuration.default().oscillation_weight
    trip_threshold: float = core.Configuration.default().trip_threshold
    warning_threshold: float = core.Configuration.default().warning_threshold
    trend_window_size: int = core.Configuration.default().trend_window_size
    enable_auto_escalation: bool = core.Configuration.default().enable_auto_escalation
    cooldown_tokens: int = core.Configuration.default().cooldown_tokens

    @staticmethod
    def default() -> "CircuitBreakerConfig":
        return CircuitBreakerConfig()

    @staticmethod
    def conservative() -> "CircuitBreakerConfig":
        base = core.Configuration.conservative()
        return CircuitBreakerConfig(
            entropy_weight=base.entropy_weight,
            refusal_weight=base.refusal_weight,
            persona_drift_weight=base.persona_drift_weight,
            oscillation_weight=base.oscillation_weight,
            trip_threshold=base.trip_threshold,
            warning_threshold=base.warning_threshold,
            trend_window_size=base.trend_window_size,
            enable_auto_escalation=base.enable_auto_escalation,
            cooldown_tokens=base.cooldown_tokens,
        )

    @staticmethod
    def permissive() -> "CircuitBreakerConfig":
        base = core.Configuration.permissive()
        return CircuitBreakerConfig(
            entropy_weight=base.entropy_weight,
            refusal_weight=base.refusal_weight,
            persona_drift_weight=base.persona_drift_weight,
            oscillation_weight=base.oscillation_weight,
            trip_threshold=base.trip_threshold,
            warning_threshold=base.warning_threshold,
            trend_window_size=base.trend_window_size,
            enable_auto_escalation=base.enable_auto_escalation,
            cooldown_tokens=base.cooldown_tokens,
        )

    def to_core(self) -> core.Configuration:
        return core.Configuration(
            entropy_weight=self.entropy_weight,
            refusal_weight=self.refusal_weight,
            persona_drift_weight=self.persona_drift_weight,
            oscillation_weight=self.oscillation_weight,
            trip_threshold=self.trip_threshold,
            warning_threshold=self.warning_threshold,
            trend_window_size=self.trend_window_size,
            enable_auto_escalation=self.enable_auto_escalation,
            cooldown_tokens=self.cooldown_tokens,
        )


class CircuitBreakerIntegration:
    @staticmethod
    def evaluate(
        signals: core.InputSignals,
        configuration: CircuitBreakerConfig | core.Configuration | None = None,
        previous_state: core.CircuitBreakerState | None = None,
    ) -> core.CircuitBreakerState:
        core_config = configuration.to_core() if isinstance(configuration, CircuitBreakerConfig) else configuration
        return core.CircuitBreakerIntegration.evaluate(signals, core_config, previous_state)


# Backwards-compatible enum aliases expected by tests.
RecommendedAction.CONTINUE = core.RecommendedAction.continue_generation
RecommendedAction.INSERT_SAFETY_PROMPT = core.RecommendedAction.insert_safety_prompt
TriggerSource.REFUSAL_APPROACH = core.TriggerSource.refusal_approach
TriggerSource.COMBINED_SIGNALS = core.TriggerSource.combined_signals
