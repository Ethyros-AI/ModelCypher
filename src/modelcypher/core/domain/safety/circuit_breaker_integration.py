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

"""Circuit breaker integration - raw signal aggregation.

Aggregates multiple safety signals (entropy, refusal distance, persona drift,
oscillation) into a combined severity score. Uses caller-provided thresholds
for trip/warning decisions.

No arbitrary defaults. No classification. The geometry speaks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    """Circuit breaker configuration with signal weights and thresholds.

    All thresholds must be explicitly provided by the caller.
    No arbitrary defaults - derive from your baseline measurements.
    """

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
    ) -> Configuration:
        """Create config with uniform weights.

        All signals contribute equally. Thresholds must be explicitly provided.
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
    ) -> Configuration:
        """Derive thresholds from baseline severity measurements.

        Args:
            baseline_severities: Measured severities from representative samples
            percentile_trip: Percentile for trip threshold (default: 99th)
            percentile_warning: Percentile for warning threshold (default: 95th)

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

    All signal values are normalized to [0, 1]:
    - 0 = safe/nominal/no concern
    - 1 = maximum risk/concern

    ## Entropy Signal

    The `entropy_signal` MUST be normalized entropy in [0, 1], NOT raw
    Shannon entropy. Raw entropy is in [0, ln(vocab_size)] ≈ [0, 10.5].

    To convert raw entropy:
    ```python
    normalized = LogitEntropyCalculator.normalize_entropy(raw_entropy, vocab_size)
    ```

    ## Refusal Distance

    Distance to refusal boundary in embedding space, normalized [0, 1]:
    - 0 = at refusal boundary (maximum risk)
    - 1 = far from refusal (safe)
    """

    entropy_signal: float | None = None
    """Normalized entropy [0, 1]."""

    refusal_distance: float | None = None
    """Distance to refusal boundary [0, 1]. 0 = at boundary, 1 = far."""

    is_approaching_refusal: bool | None = None
    """Whether trajectory is moving toward refusal boundary."""

    persona_drift_magnitude: float | None = None
    """Magnitude of persona drift [0, 1]."""

    drifting_traits: list[str] = field(default_factory=list)
    """List of persona traits that are drifting."""

    oscillation_severity: float | None = None
    """Oscillation pattern severity [0, 1]. Raw measurement."""

    consecutive_oscillations: int = 0
    """Count of consecutive unstable windows. Raw measurement."""

    has_oscillation: bool = False
    """Whether oscillation pattern is detected."""

    token_index: int = 0
    """Current token index in generation."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """Timestamp of signal measurement."""


class TriggerSource(str, Enum):
    """Source that triggered the circuit breaker."""

    entropy_spike = "entropySpike"
    refusal_approach = "refusalApproach"
    persona_drift = "personaDrift"
    oscillation_pattern = "oscillationPattern"
    combined_signals = "combinedSignals"
    manual = "manual"


class RecommendedAction(str, Enum):
    """Recommended action based on circuit breaker state."""

    continue_generation = "continue"
    monitor = "monitor"
    reduce_temperature = "reduceTemperature"
    insert_safety_prompt = "insertSafetyPrompt"
    stop_generation = "stopGeneration"
    human_review = "humanReview"

    @property
    def description(self) -> str:
        descriptions = {
            RecommendedAction.continue_generation: "Continue normally",
            RecommendedAction.monitor: "Monitor more closely",
            RecommendedAction.reduce_temperature: "Reduce sampling temperature",
            RecommendedAction.insert_safety_prompt: "Insert safety system prompt",
            RecommendedAction.stop_generation: "Stop generation",
            RecommendedAction.human_review: "Stop and request human review",
        }
        return descriptions.get(self, "Unknown action")


@dataclass(frozen=True)
class SignalContributions:
    """Raw signal values for circuit breaker evaluation.

    All values are in [0, 1] representing raw measurements, not weighted.
    """

    entropy: float
    """Raw entropy signal [0, 1]."""

    refusal: float
    """Refusal proximity signal [0, 1]. 0 = far, 1 = at boundary."""

    persona_drift: float
    """Persona drift signal [0, 1]."""

    oscillation: float
    """Oscillation pattern signal [0, 1]."""

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
        mapping = {
            TriggerSource.entropy_spike: self.entropy,
            TriggerSource.refusal_approach: self.refusal,
            TriggerSource.persona_drift: self.persona_drift,
            TriggerSource.oscillation_pattern: self.oscillation,
        }
        return mapping.get(source, 0.0)

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
    """Current state of the circuit breaker."""

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

        interpretations = {
            TriggerSource.entropy_spike: (
                "HIGH UNCERTAINTY: Semantic entropy spike detected - potential hallucination"
            ),
            TriggerSource.refusal_approach: (
                "SAFETY CONCERN: Model approaching refusal direction"
            ),
            TriggerSource.persona_drift: (
                "ALIGNMENT DRIFT: Persona vectors shifting from baseline"
            ),
            TriggerSource.oscillation_pattern: (
                "INSTABILITY: Oscillation pattern detected in generation"
            ),
            TriggerSource.combined_signals: (
                "MULTIPLE CONCERNS: Combined safety signals exceeded threshold"
            ),
            TriggerSource.manual: "MANUAL TRIP: Circuit breaker manually activated",
        }
        return interpretations.get(source, "Circuit breaker tripped")


@dataclass(frozen=True)
class CircuitBreakerTelemetry:
    """Telemetry snapshot for circuit breaker state."""

    token_index: int
    timestamp: datetime
    state: CircuitBreakerState
    combined_severity: float
    any_signal_exceeded: bool
    oscillation_severity: float | None
    consecutive_oscillations: int


class CircuitBreakerIntegration:
    """Evaluates safety signals and determines circuit breaker state."""

    @staticmethod
    def evaluate(
        signals: InputSignals,
        configuration: Configuration,
        previous_state: CircuitBreakerState | None = None,
    ) -> CircuitBreakerState:
        """Evaluate signals and return circuit breaker state."""
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
            signals.oscillation_severity,
            signals.consecutive_oscillations,
            signals.has_oscillation,
            config.oscillation_weight,
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

        trigger_source: TriggerSource | None = None
        if is_tripped:
            dominant = contributions.dominant_source
            total = severity if severity > 0 else 1.0
            dominant_contrib = contributions.get(dominant)
            if dominant_contrib / total > 0.5:
                trigger_source = dominant
            else:
                trigger_source = TriggerSource.combined_signals

        confidence = CircuitBreakerIntegration._compute_confidence(signals)
        action = CircuitBreakerIntegration._determine_action(
            severity, is_tripped, config, signals
        )

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
        """Create telemetry snapshot from current state."""
        any_exceeded = state.severity >= config.warning_threshold
        return CircuitBreakerTelemetry(
            token_index=state.token_index,
            timestamp=state.timestamp,
            state=state,
            combined_severity=state.severity,
            any_signal_exceeded=any_exceeded,
            oscillation_severity=signals.oscillation_severity,
            consecutive_oscillations=signals.consecutive_oscillations,
        )

    @staticmethod
    def to_metrics_dict(state: CircuitBreakerState) -> dict[str, float]:
        """Convert state to metrics dictionary."""
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
        """Compute weighted entropy contribution.

        Args:
            entropy: Normalized entropy in [0, 1].
            weight: Weight for this signal.

        Returns:
            Weighted contribution.
        """
        if entropy is None:
            return 0.0

        if entropy < 0.0 or entropy > 1.0:
            logger.warning(
                "entropy_signal %.3f is outside expected [0, 1] range. "
                "Use LogitEntropyCalculator.normalize_entropy(). Clamping.",
                entropy,
            )
            entropy = max(0.0, min(1.0, entropy))

        return entropy * weight

    @staticmethod
    def _compute_refusal_contribution(
        distance: float | None,
        is_approaching: bool | None,
        weight: float,
    ) -> float:
        """Compute weighted refusal contribution."""
        if distance is None:
            return 0.0
        base_contribution = 1.0 - distance
        return min(base_contribution, 1.0) * weight

    @staticmethod
    def _compute_persona_contribution(
        drift_magnitude: float | None,
        drifting_traits: list[str],
        weight: float,
    ) -> float:
        """Compute persona drift contribution.

        Uses drift_magnitude directly. Trait count adds logarithmic penalty
        (diminishing returns is principled, not arbitrary).
        """
        if drift_magnitude is None:
            return 0.0

        base = min(drift_magnitude, 1.0)
        # log(1 + n) / log(11) gives diminishing returns: 0 traits -> 0, 10 traits -> 1
        trait_bonus = math.log(1 + len(drifting_traits)) / math.log(11) if drifting_traits else 0.0
        # Trait contribution capped at 0.2 of weight
        combined = base + trait_bonus * 0.2
        return min(combined, 1.0) * weight

    @staticmethod
    def _compute_oscillation_contribution(
        oscillation_severity: float | None,
        consecutive_oscillations: int,
        has_oscillation: bool,
        weight: float,
    ) -> float:
        """Compute oscillation contribution from raw measurements.

        Args:
            oscillation_severity: Direct severity measurement [0, 1]
            consecutive_oscillations: Count of consecutive unstable windows
            has_oscillation: Whether oscillation is detected
            weight: Weight for this signal
        """
        if oscillation_severity is not None:
            # Direct pass-through with consecutive bonus
            # Each consecutive window adds 0.1 up to 0.3 max
            consecutive_bonus = min(consecutive_oscillations * 0.1, 0.3)
            raw_contribution = min(oscillation_severity + consecutive_bonus, 1.0)
            return raw_contribution * weight

        # Fallback if no severity provided - binary oscillation signal
        oscillation_contribution = 0.5 if has_oscillation else 0.0
        return oscillation_contribution * weight

    @staticmethod
    def _compute_confidence(signals: InputSignals) -> float:
        """Compute confidence based on signal availability."""
        total_signals = 4
        available = 0
        if signals.entropy_signal is not None:
            available += 1
        if signals.refusal_distance is not None:
            available += 1
        if signals.persona_drift_magnitude is not None:
            available += 1
        if signals.oscillation_severity is not None:
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

        Action zones are proportional to the range [trip_threshold, 1.0].
        """
        if not is_tripped:
            if severity >= configuration.warning_threshold:
                return RecommendedAction.monitor
            return RecommendedAction.continue_generation

        # Severe oscillation triggers stop
        if signals.oscillation_severity is not None and signals.oscillation_severity >= 0.9:
            return RecommendedAction.stop_generation

        # Action based on position in danger zone
        trip = configuration.trip_threshold
        danger_range = 1.0 - trip
        if danger_range <= 0:
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
