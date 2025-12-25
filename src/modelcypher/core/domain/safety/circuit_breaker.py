# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

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
    """Circuit breaker configuration. All fields required - no arbitrary defaults."""

    entropy_weight: float
    refusal_weight: float
    persona_drift_weight: float
    oscillation_weight: float
    trip_threshold: float
    warning_threshold: float
    trend_window_size: int = 10
    enable_auto_escalation: bool = True
    cooldown_tokens: int = 5

    @staticmethod
    def uniform_weights(
        trip_threshold: float,
        warning_threshold: float,
        trend_window_size: int = 10,
        enable_auto_escalation: bool = True,
        cooldown_tokens: int = 5,
    ) -> "CircuitBreakerConfig":
        """Create config with uniform weights. Thresholds must be provided."""
        return CircuitBreakerConfig(
            entropy_weight=0.25,
            refusal_weight=0.25,
            persona_drift_weight=0.25,
            oscillation_weight=0.25,
            trip_threshold=trip_threshold,
            warning_threshold=warning_threshold,
            trend_window_size=trend_window_size,
            enable_auto_escalation=enable_auto_escalation,
            cooldown_tokens=cooldown_tokens,
        )

    @staticmethod
    def from_baseline_measurements(
        baseline_severities: list[float],
        percentile_trip: float = 99.0,
        percentile_warning: float = 95.0,
        trend_window_size: int = 10,
        enable_auto_escalation: bool = True,
        cooldown_tokens: int = 5,
    ) -> "CircuitBreakerConfig":
        """Derive thresholds from baseline severity measurements."""
        core_config = core.Configuration.from_baseline_measurements(
            baseline_severities=baseline_severities,
            percentile_trip=percentile_trip,
            percentile_warning=percentile_warning,
            trend_window_size=trend_window_size,
            enable_auto_escalation=enable_auto_escalation,
            cooldown_tokens=cooldown_tokens,
        )
        return CircuitBreakerConfig(
            entropy_weight=core_config.entropy_weight,
            refusal_weight=core_config.refusal_weight,
            persona_drift_weight=core_config.persona_drift_weight,
            oscillation_weight=core_config.oscillation_weight,
            trip_threshold=core_config.trip_threshold,
            warning_threshold=core_config.warning_threshold,
            trend_window_size=core_config.trend_window_size,
            enable_auto_escalation=core_config.enable_auto_escalation,
            cooldown_tokens=core_config.cooldown_tokens,
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
        configuration: CircuitBreakerConfig | core.Configuration,
        previous_state: core.CircuitBreakerState | None = None,
    ) -> core.CircuitBreakerState:
        core_config = (
            configuration.to_core()
            if isinstance(configuration, CircuitBreakerConfig)
            else configuration
        )
        return core.CircuitBreakerIntegration.evaluate(signals, core_config, previous_state)


# Backwards-compatible enum aliases expected by tests.
RecommendedAction.CONTINUE = core.RecommendedAction.continue_generation
RecommendedAction.INSERT_SAFETY_PROMPT = core.RecommendedAction.insert_safety_prompt
TriggerSource.REFUSAL_APPROACH = core.TriggerSource.refusal_approach
TriggerSource.COMBINED_SIGNALS = core.TriggerSource.combined_signals
