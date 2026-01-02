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
oscillation) into a combined severity magnitude. Returns raw measurements only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


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
    """Dominant source among available signals."""

    entropy_spike = "entropySpike"
    refusal_approach = "refusalApproach"
    persona_drift = "personaDrift"
    oscillation_pattern = "oscillationPattern"
    combined_signals = "combinedSignals"
    manual = "manual"


@dataclass(frozen=True)
class SignalContributions:
    """Raw signal values for circuit breaker evaluation.

    All values are in [0, 1] representing raw measurements.
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

    severity: float
    dominant_source: TriggerSource | None
    confidence: float
    signal_contributions: SignalContributions
    token_index: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class CircuitBreakerTelemetry:
    """Telemetry snapshot for circuit breaker state."""

    token_index: int
    timestamp: datetime
    state: CircuitBreakerState
    combined_severity: float
    oscillation_severity: float | None
    consecutive_oscillations: int


class CircuitBreakerIntegration:
    """Evaluates safety signals and reports raw measurements."""

    @staticmethod
    def evaluate(signals: InputSignals) -> CircuitBreakerState:
        """Evaluate signals and return circuit breaker state."""
        entropy_contribution = CircuitBreakerIntegration._compute_entropy_contribution(
            signals.entropy_signal
        )
        refusal_contribution = CircuitBreakerIntegration._compute_refusal_contribution(
            signals.refusal_distance, signals.is_approaching_refusal
        )
        persona_contribution = CircuitBreakerIntegration._compute_persona_contribution(
            signals.persona_drift_magnitude, signals.drifting_traits
        )
        oscillation_contribution = CircuitBreakerIntegration._compute_oscillation_contribution(
            signals.oscillation_severity,
            signals.consecutive_oscillations,
            signals.has_oscillation,
        )

        contributions = SignalContributions(
            entropy=entropy_contribution,
            refusal=refusal_contribution,
            persona_drift=persona_contribution,
            oscillation=oscillation_contribution,
        )

        severity = contributions.mean_signal
        dominant_source = contributions.dominant_source if severity > 0 else None
        confidence = CircuitBreakerIntegration._compute_confidence(signals)

        return CircuitBreakerState(
            severity=severity,
            dominant_source=dominant_source,
            confidence=confidence,
            signal_contributions=contributions,
            token_index=signals.token_index,
            timestamp=signals.timestamp,
        )

    @staticmethod
    def create_telemetry(
        state: CircuitBreakerState, signals: InputSignals
    ) -> CircuitBreakerTelemetry:
        """Create telemetry snapshot from current state."""
        return CircuitBreakerTelemetry(
            token_index=state.token_index,
            timestamp=state.timestamp,
            state=state,
            combined_severity=state.severity,
            oscillation_severity=signals.oscillation_severity,
            consecutive_oscillations=signals.consecutive_oscillations,
        )

    @staticmethod
    def to_metrics_dict(state: CircuitBreakerState) -> dict[str, float]:
        """Convert state to metrics dictionary."""
        return {
            "geometry/circuit_breaker_confidence": float(state.confidence),
            "geometry/circuit_breaker_severity": float(state.severity),
            "geometry/circuit_breaker_entropy": float(state.signal_contributions.entropy),
            "geometry/circuit_breaker_refusal": float(state.signal_contributions.refusal),
            "geometry/circuit_breaker_persona": float(state.signal_contributions.persona_drift),
            "geometry/circuit_breaker_oscillation": float(state.signal_contributions.oscillation),
        }

    @staticmethod
    def _compute_entropy_contribution(entropy: float | None) -> float:
        """Compute entropy contribution from raw signal."""
        if entropy is None:
            return 0.0

        if entropy < 0.0 or entropy > 1.0:
            logger.warning(
                "entropy_signal %.3f is outside expected [0, 1] range. "
                "Use LogitEntropyCalculator.normalize_entropy(). Clamping.",
                entropy,
            )
            entropy = max(0.0, min(1.0, entropy))

        return entropy

    @staticmethod
    def _compute_refusal_contribution(
        distance: float | None,
        is_approaching: bool | None,
    ) -> float:
        """Compute refusal contribution from distance to boundary."""
        if distance is None:
            return 0.0
        base_contribution = 1.0 - distance
        return min(base_contribution, 1.0)

    @staticmethod
    def _compute_persona_contribution(
        drift_magnitude: float | None,
        drifting_traits: list[str],
    ) -> float:
        """Compute persona drift contribution from raw measurement."""
        if drift_magnitude is None:
            return 0.0

        return min(drift_magnitude, 1.0)

    @staticmethod
    def _compute_oscillation_contribution(
        oscillation_severity: float | None,
        consecutive_oscillations: int,
        has_oscillation: bool,
    ) -> float:
        """Compute oscillation contribution from raw measurements."""
        if oscillation_severity is not None:
            return min(oscillation_severity, 1.0)

        return 0.0

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
