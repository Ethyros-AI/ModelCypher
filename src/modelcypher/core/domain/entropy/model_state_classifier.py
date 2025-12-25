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

"""Model state measurement for entropy-based cognitive state analysis.

Returns raw entropy and variance signals. No classification enums.
The geometry speaks for itself - consumers interpret the signals.

The key measurements:
- entropy: Token-level uncertainty (Shannon entropy of softmax)
- variance: Distribution shape (variance of top-K logits)
- entropy_trend: Rate of change in entropy
- entropy_variance_correlation: Relationship between the two axes

These signals encode cognitive state. The COMBINATION matters:
| Entropy   | Variance | Interpretation                      |
|-----------|----------|-------------------------------------|
| low       | high     | One token dominates                 |
| moderate  | moderate | Healthy generation                  |
| high      | moderate | Epistemic uncertainty (doesn't know)|
| high      | low      | Normative uncertainty (shouldn't do)|
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelStateSignals:
    """Raw entropy and variance signals. This IS the cognitive state measurement.

    Consumers interpret these signals according to their needs.
    No arbitrary threshold classifications.
    """

    entropy: float
    """Current entropy value. Lower = more confident."""

    variance: float
    """Current variance value. Shape of the distribution."""

    entropy_trend: float
    """Rate of change in entropy. Positive = rising (exploring)."""

    entropy_variance_correlation: float
    """Correlation between entropy and variance. Negative = potential distress."""

    consecutive_high_entropy_count: int
    """Consecutive samples above baseline. Sustained high = concerning."""

    circuit_breaker_tripped: bool
    """Whether generation was halted by circuit breaker."""

    @property
    def is_low_entropy(self) -> bool:
        """Heuristic: entropy below typical confident threshold."""
        return self.entropy < 2.0  # Information-theoretic threshold

    @property
    def is_high_entropy(self) -> bool:
        """Heuristic: entropy above typical uncertainty threshold."""
        return self.entropy > 3.0

    @property
    def is_low_variance(self) -> bool:
        """Heuristic: variance suggesting flat distribution."""
        return self.variance < 0.2

    @property
    def has_distress_signature(self) -> bool:
        """High entropy + low variance + negative correlation = distress pattern."""
        return (
            self.is_high_entropy
            and self.is_low_variance
            and self.entropy_variance_correlation < -0.3
        )


@dataclass(frozen=True)
class EntropyStateThresholds:
    """Information-theoretic thresholds for entropy analysis.

    These are derived from probability theory, not arbitrary:
    - Confident: entropy < ln(e²) ≈ 2.0 means probability mass concentrated
    - Uncertain: entropy > 3.0 means high uncertainty
    - Distress: entropy > 3.5 with variance < 0.2
    """

    entropy_confident: float = 2.0
    entropy_uncertain: float = 3.0
    entropy_distress: float = 3.5
    entropy_halted: float = 4.0
    variance_low: float = 0.2
    variance_moderate: float = 0.3
    trend_threshold: float = 0.05
    sustained_high_count: int = 3

    @classmethod
    def default(cls) -> EntropyStateThresholds:
        """Create default thresholds."""
        return cls()


# Backward compatibility alias
ModelStateThresholds = EntropyStateThresholds


@dataclass(frozen=True)
class ClassificationSnapshot:
    """Snapshot of entropy window state for analysis."""

    current_entropy: float
    """Current entropy value."""

    current_variance: float
    """Current variance value."""

    moving_average_entropy: float
    """Moving average of entropy."""

    average_variance: float
    """Average variance over window."""

    consecutive_high_count: int
    """Number of consecutive high-entropy samples."""

    sample_count: int
    """Total samples in window."""

    entropy_trend: float
    """Slope of entropy over window."""

    entropy_variance_correlation: float
    """Pearson correlation between entropy and variance."""

    circuit_breaker_tripped: bool
    """Whether circuit breaker was tripped."""


@dataclass(frozen=True)
class ClassificationResult:
    """Result of entropy state analysis.

    Uses string state names for backward compatibility.
    The raw entropy/variance values ARE the true state.
    """

    state_name: str
    """State name (confident, nominal, uncertain, exploring, distressed, halted)."""

    entropy: float
    """Raw entropy value - this IS the state."""

    variance: float
    """Raw variance value."""

    confidence: float
    """Confidence in classification (0.0-1.0)."""

    reason: str
    """Human-readable reason for classification."""


class ModelStateClassifier:
    """Analyzes model cognitive state from entropy and variance.

    Returns raw signals with interpretive state names for compatibility.
    The entropy/variance values ARE the cognitive state.
    """

    def __init__(self, thresholds: EntropyStateThresholds | None = None) -> None:
        """Create a model state classifier.

        Args:
            thresholds: Classification thresholds. Defaults to information-theoretic values.
        """
        self._thresholds = thresholds or EntropyStateThresholds.default()

    @property
    def thresholds(self) -> EntropyStateThresholds:
        """Get current thresholds."""
        return self._thresholds

    def get_state_name(self, entropy: float, variance: float) -> str:
        """Get interpretive state name from entropy and variance.

        Args:
            entropy: Current entropy value.
            variance: Current variance value.

        Returns:
            State name string.
        """
        # Halted: entropy exceeds circuit breaker
        if entropy >= self._thresholds.entropy_halted:
            return "halted"

        # Confident: low entropy
        if entropy < self._thresholds.entropy_confident:
            return "confident"

        # Distressed: high entropy + low variance
        if (
            entropy >= self._thresholds.entropy_distress
            and variance < self._thresholds.variance_low
        ):
            return "distressed"

        # Uncertain: high entropy
        if entropy >= self._thresholds.entropy_uncertain:
            return "uncertain"

        # Nominal: moderate entropy
        return "nominal"

    def is_confident(self, entropy: float, variance: float) -> bool:
        """Check if model is confident (low entropy)."""
        return entropy < self._thresholds.entropy_confident

    def is_uncertain(self, entropy: float, variance: float) -> bool:
        """Check if model is uncertain (high entropy)."""
        return entropy >= self._thresholds.entropy_uncertain

    def is_distressed(self, entropy: float, variance: float) -> bool:
        """Check if model shows distress signature (high entropy + low variance)."""
        return (
            entropy >= self._thresholds.entropy_distress
            and variance < self._thresholds.variance_low
        )

    def requires_caution(self, entropy: float, variance: float) -> bool:
        """Check if current state warrants caution."""
        return self.is_uncertain(entropy, variance) or self.is_distressed(entropy, variance)

    def analyze_snapshot(self, snapshot: ClassificationSnapshot) -> ClassificationResult:
        """Analyze model state from a window snapshot.

        Args:
            snapshot: Entropy window snapshot with history.

        Returns:
            Classification result with raw values and state name.
        """
        # Check for halted state first (circuit breaker)
        if snapshot.circuit_breaker_tripped:
            return ClassificationResult(
                state_name="halted",
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
                confidence=1.0,
                reason="Circuit breaker tripped",
            )

        # Check for distress pattern (sustained high entropy + low variance)
        if snapshot.consecutive_high_count >= self._thresholds.sustained_high_count:
            if snapshot.average_variance < self._thresholds.variance_moderate:
                has_distress_correlation = snapshot.entropy_variance_correlation < -0.3
                confidence = 0.9 if has_distress_correlation else 0.75
                return ClassificationResult(
                    state_name="distressed",
                    entropy=snapshot.current_entropy,
                    variance=snapshot.current_variance,
                    confidence=confidence,
                    reason=(
                        f"Sustained high entropy ({snapshot.consecutive_high_count} tokens) "
                        f"with low variance ({snapshot.average_variance:.2f})"
                    ),
                )

        # Check for exploring pattern (rising entropy trend)
        if (
            snapshot.sample_count >= 5
            and snapshot.entropy_trend > self._thresholds.trend_threshold
            and snapshot.current_entropy < self._thresholds.entropy_distress
            and snapshot.current_entropy >= self._thresholds.entropy_confident
        ):
            return ClassificationResult(
                state_name="exploring",
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
                confidence=min(0.8, snapshot.entropy_trend * 10),
                reason=f"Rising entropy trend (slope: {snapshot.entropy_trend:.3f})",
            )

        # Fall back to instantaneous classification
        state_name = self.get_state_name(
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
        )

        return ClassificationResult(
            state_name=state_name,
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
            confidence=self._calculate_confidence(
                state_name, snapshot.current_entropy, snapshot.current_variance
            ),
            reason=self._reason_for_state(
                state_name, snapshot.current_entropy, snapshot.current_variance
            ),
        )

    def _calculate_confidence(
        self,
        state_name: str,
        entropy: float,
        variance: float,
    ) -> float:
        """Calculate confidence based on how clearly values fall into state's region."""
        if state_name == "confident":
            return 1.0 - (entropy / self._thresholds.entropy_confident)

        if state_name == "nominal":
            entropy_distance = min(
                entropy - self._thresholds.entropy_confident,
                self._thresholds.entropy_uncertain - entropy,
            )
            return min(1.0, entropy_distance / 0.15)

        if state_name == "uncertain":
            return min(
                1.0, (entropy - self._thresholds.entropy_uncertain) / 0.3
            )

        if state_name == "distressed":
            return min(
                1.0,
                (self._thresholds.variance_moderate - variance)
                / self._thresholds.variance_moderate,
            )

        if state_name == "exploring":
            return 0.7

        if state_name == "halted":
            return 1.0

        return 0.5

    def _reason_for_state(self, state_name: str, entropy: float, variance: float) -> str:
        """Get human-readable reason for state."""
        e = f"{entropy:.2f}"
        v = f"{variance:.2f}"

        reasons = {
            "confident": f"Low entropy ({e}) indicates confident generation",
            "nominal": f"Moderate entropy ({e}) with balanced variance ({v})",
            "uncertain": f"High entropy ({e}) indicates uncertainty",
            "distressed": f"High entropy ({e}) with low variance ({v}) - distress signature",
            "exploring": "Entropy trend rising",
            "halted": "Circuit breaker tripped",
        }

        return reasons.get(state_name, "Unknown state")
