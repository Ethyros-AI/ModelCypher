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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class InterventionLevel(str, Enum):
    """DEPRECATED: Use raw oscillation measurements instead.

    This enum exists for backward compatibility. New code should use:
    - oscillation_severity: float (0-1)
    - consecutive_oscillations: int
    - has_oscillation: bool

    These raw measurements ARE the signal. Classification destroys information.
    """

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
    def uniform_weights(
        trip_threshold: float,
        warning_threshold: float,
        trend_window_size: int = 10,
        enable_auto_escalation: bool = True,
        cooldown_tokens: int = 5,
    ) -> "Configuration":
        """Create config with uniform weights.

        All signals contribute equally. Thresholds must be explicitly provided
        by the caller - no arbitrary defaults.

        Args:
            trip_threshold: Severity at which circuit breaker trips (must be provided)
            warning_threshold: Severity at which warning is issued (must be provided)
            trend_window_size: Number of tokens to track for trend detection
            enable_auto_escalation: Whether to auto-escalate on trend
            cooldown_tokens: Tokens to wait before de-escalating
        """
        return Configuration(
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
    ) -> "Configuration":
        """Derive thresholds from baseline severity measurements.

        Args:
            baseline_severities: Measured severities from representative samples
            percentile_trip: Percentile for trip threshold (default: 99th)
            percentile_warning: Percentile for warning threshold (default: 95th)
            trend_window_size: Number of tokens to track for trend detection
            enable_auto_escalation: Whether to auto-escalate on trend
            cooldown_tokens: Tokens to wait before de-escalating

        Raises:
            ValueError: If baseline_severities is empty
        """
        if not baseline_severities:
            raise ValueError("baseline_severities cannot be empty")

        sorted_severities = sorted(baseline_severities)
        n = len(sorted_severities)

        trip_idx = min(int(n * percentile_trip / 100), n - 1)
        warning_idx = min(int(n * percentile_warning / 100), n - 1)

        return Configuration(
            entropy_weight=0.25,
            refusal_weight=0.25,
            persona_drift_weight=0.25,
            oscillation_weight=0.25,
            trip_threshold=sorted_severities[trip_idx],
            warning_threshold=sorted_severities[warning_idx],
            trend_window_size=trend_window_size,
            enable_auto_escalation=enable_auto_escalation,
            cooldown_tokens=cooldown_tokens,
        )

    @property
    def is_weights_valid(self) -> bool:
        total = (
            self.entropy_weight
            + self.refusal_weight
            + self.persona_drift_weight
            + self.oscillation_weight
        )
        return abs(total - 1.0) < 0.01


@dataclass(frozen=True)
class InputSignals:
    """Input signals for circuit breaker evaluation.

    All signal values should be normalized to [0, 1] range:
    - 0 = safe/nominal/no concern
    - 1 = maximum risk/concern

    ## Entropy Signal

    The `entropy_signal` MUST be normalized entropy in [0, 1], NOT raw
    Shannon entropy. Raw entropy from LogitEntropyCalculator is in
    [0, ln(vocab_size)] ≈ [0, 10.5] for 32K vocab.

    To convert raw entropy to normalized:
    ```python
    from modelcypher.core.domain.entropy.logit_entropy_calculator import (
        LogitEntropyCalculator,
    )
    normalized = LogitEntropyCalculator.normalize_entropy(raw_entropy, vocab_size)
    ```

    ## Refusal Distance

    Distance to refusal boundary in embedding space, normalized [0, 1]:
    - 0 = at refusal boundary (maximum risk)
    - 1 = far from refusal (safe)
    """

    entropy_signal: float | None = None
    """Normalized entropy [0, 1]. Use LogitEntropyCalculator.normalize_entropy()."""

    refusal_distance: float | None = None
    """Distance to refusal boundary [0, 1]. 0 = at boundary, 1 = far from boundary."""

    is_approaching_refusal: bool | None = None
    """Whether trajectory is moving toward refusal boundary."""

    persona_drift_magnitude: float | None = None
    """Magnitude of persona drift [0, 1]."""

    drifting_traits: list[str] = field(default_factory=list)
    """List of persona traits that are drifting."""

    gas_level: InterventionLevel | None = None
    """Current GAS (Guardrail Awareness System) intervention level."""

    has_oscillation: bool = False
    """Whether oscillation pattern is detected."""

    token_index: int = 0
    """Current token index in generation."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """Timestamp of signal measurement."""


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
    """Raw signal values for circuit breaker evaluation.

    All values are in [0, 1] range representing raw measurements, not weighted.
    Consumers decide how to combine or interpret these signals.
    """

    entropy: float
    """Raw entropy signal [0, 1]. 0 = low entropy, 1 = high entropy."""

    refusal: float
    """Refusal proximity signal [0, 1]. 0 = far from refusal, 1 = at refusal boundary."""

    persona_drift: float
    """Persona drift signal [0, 1]. 0 = stable persona, 1 = maximum drift."""

    oscillation: float
    """Oscillation pattern signal [0, 1]. 0 = stable, 1 = severe oscillation."""

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

    def get(self, source: TriggerSource) -> float:
        """Get contribution value for a specific trigger source."""
        if source is TriggerSource.entropy_spike:
            return self.entropy
        if source is TriggerSource.refusal_approach:
            return self.refusal
        if source is TriggerSource.persona_drift:
            return self.persona_drift
        if source is TriggerSource.oscillation_pattern:
            return self.oscillation
        return 0.0

    @property
    def max_signal(self) -> float:
        """Maximum signal value across all contributions."""
        return max(self.entropy, self.refusal, self.persona_drift, self.oscillation)

    @property
    def mean_signal(self) -> float:
        """Mean signal value across all contributions."""
        return (self.entropy + self.refusal + self.persona_drift + self.oscillation) / 4.0


@dataclass(frozen=True)
class CircuitBreakerState:
    is_tripped: bool
    severity: float
    trigger_source: TriggerSource | None
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
    gas_level: InterventionLevel | None
    combined_severity: float
    any_signal_exceeded: bool


class CircuitBreakerIntegration:
    @staticmethod
    def evaluate(
        signals: InputSignals,
        configuration: Configuration,
        previous_state: CircuitBreakerState | None = None,
    ) -> CircuitBreakerState:
        config = configuration

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
        severity = (
            entropy_contribution
            + refusal_contribution
            + persona_contribution
            + oscillation_contribution
        )
        is_tripped = severity >= config.trip_threshold

        trigger_source: TriggerSource | None
        if is_tripped:
            # When tripped, the dominant signal IS the trigger source
            # No arbitrary minimum thresholds - we already know severity >= trip_threshold
            dominant = contributions.dominant_source
            # Check if any single signal contributes majority (>50%) of total
            total = severity if severity > 0 else 1.0
            dominant_contrib = contributions.get(dominant)
            if dominant_contrib / total > 0.5:
                trigger_source = dominant
            else:
                # No single signal dominates - combined effect
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
    def create_telemetry(
        state: CircuitBreakerState, signals: InputSignals, config: Configuration
    ) -> CircuitBreakerTelemetry:
        # Signal is "exceeded" if combined severity is above warning threshold
        # This uses the config-derived threshold, not arbitrary values
        any_exceeded = state.severity >= config.warning_threshold
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
    def _compute_entropy_contribution(entropy: float | None, weight: float) -> float:
        """Compute weighted entropy contribution to circuit breaker score.

        Args:
            entropy: Normalized entropy in [0, 1]. Values outside this range
                indicate incorrect normalization (likely raw entropy was passed).
            weight: Weight for this signal in the aggregate score.

        Returns:
            Weighted contribution to circuit breaker severity.
        """
        if entropy is None:
            return 0.0

        # Validate range - log warning if out of [0, 1]
        if entropy < 0.0 or entropy > 1.0:
            logger.warning(
                "entropy_signal %.3f is outside expected [0, 1] range. "
                "Raw Shannon entropy should be normalized using "
                "LogitEntropyCalculator.normalize_entropy(raw_entropy, vocab_size). "
                "Clamping to [0, 1].",
                entropy,
            )
            entropy = max(0.0, min(1.0, entropy))

        # Direct pass-through - no arbitrary piecewise scaling
        # Entropy is already normalized [0, 1] upstream
        return entropy * weight

    @staticmethod
    def _compute_refusal_contribution(
        distance: float | None,
        is_approaching: bool | None,
        weight: float,
    ) -> float:
        if distance is None:
            return 0.0
        base_contribution = 1.0 - distance
        # No arbitrary bonus - geometry speaks for itself
        return min(base_contribution, 1.0) * weight

    @staticmethod
    def _compute_persona_contribution(
        drift_magnitude: float | None,
        drifting_traits: list[str],
        weight: float,
    ) -> float:
        """Compute persona drift contribution.

        Uses drift_magnitude directly - no arbitrary scaling.
        Trait count adds logarithmic penalty (principled: diminishing returns).
        """
        if drift_magnitude is None:
            return 0.0
        import math

        # Direct pass-through for magnitude
        base = min(drift_magnitude, 1.0)
        # Trait penalty: log(1 + n) / log(1 + max_reasonable) gives diminishing returns
        # At 0 traits: 0, at 1 trait: 0.5, at 3 traits: 0.79, at 10 traits: 0.96
        trait_bonus = math.log(1 + len(drifting_traits)) / math.log(11) if drifting_traits else 0.0
        # Combine: magnitude + small trait contribution (max trait contribution = 0.2)
        combined = base + trait_bonus * 0.2
        return min(combined, 1.0) * weight

    @staticmethod
    def _compute_oscillation_contribution(
        has_oscillation: bool,
        gas_level: InterventionLevel | None,
        weight: float,
    ) -> float:
        """Compute oscillation contribution.

        GAS levels are ordinal (0-4), so contribution is level / max_level.
        Oscillation detection is binary, contributing 0.5 if present.
        """
        # GAS level as fraction of max (level4 = 1.0)
        gas_contribution = 0.0
        if gas_level is not None:
            level_map = {
                InterventionLevel.level0_continue: 0,
                InterventionLevel.level1_gentle: 1,
                InterventionLevel.level2_clarify: 2,
                InterventionLevel.level3_hard: 3,
                InterventionLevel.level4_terminate: 4,
            }
            gas_contribution = level_map.get(gas_level, 0) / 4.0

        # Oscillation is binary - 0.5 if present (midpoint)
        oscillation_contribution = 0.5 if has_oscillation else 0.0

        # Combine: max of the two signals (don't double-count related signals)
        combined = max(gas_contribution, oscillation_contribution)
        return combined * weight

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
        """Determine action based on severity relative to thresholds.

        Action escalation is based on how far severity exceeds trip_threshold,
        not fixed absolute values. The range [trip, 1.0] is divided into
        action zones proportionally.
        """
        if not is_tripped:
            if severity >= configuration.warning_threshold:
                return RecommendedAction.monitor
            return RecommendedAction.continue_generation

        # GAS level4 always triggers stop
        if signals.gas_level is InterventionLevel.level4_terminate:
            return RecommendedAction.stop_generation

        # Action based on how far into the danger zone we are
        # Severity in [trip_threshold, 1.0] -> zones: reduce_temp, safety_prompt, stop, human_review
        trip = configuration.trip_threshold
        danger_range = 1.0 - trip
        if danger_range <= 0:
            # Trip threshold is 1.0, so any trip is maximum severity
            return RecommendedAction.human_review

        danger_fraction = (severity - trip) / danger_range

        # Equal-sized zones in danger range
        if danger_fraction >= 0.75:
            return RecommendedAction.human_review
        if danger_fraction >= 0.50 or signals.is_approaching_refusal:
            return RecommendedAction.stop_generation
        if danger_fraction >= 0.25:
            return RecommendedAction.insert_safety_prompt
        return RecommendedAction.reduce_temperature
