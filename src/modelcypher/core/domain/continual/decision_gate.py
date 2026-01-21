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

"""
Decision Gate - Geometry-derived metacognitive routing.

This gate makes routing decisions based on running statistics derived from
the entropy trajectory. No hardcoded thresholds - all decisions are based
on z-scores relative to observed baseline behavior.

The key insight: A perpetually curious AI should THINK MORE when exploring
novel manifold regions (high entropy, positive derivative) and EMIT when
confident (converging entropy). Safety boundaries trigger CLARIFY.

Policy:
    EMIT: entropy z-score < sqrt(eps) OR derivative < 0 (confident/converging)
    THINK_MORE: entropy z-score > 2σ AND derivative > 0 (uncertain/diverging)
    CLARIFY: refusal_distance < sqrt(eps) (approaching safety boundary)

All thresholds derived from machine precision (sqrt(eps)) or running statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.entropy_analyzer import EntropyState

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class DecisionAction(Enum):
    """Possible actions from the decision gate."""

    EMIT = "emit"  # Emit the token
    THINK_MORE = "think_more"  # Re-run computation without emitting
    CLARIFY = "clarify"  # Request clarification from user


@dataclass(frozen=True)
class Decision:
    """Output from the decision gate.

    Attributes:
        action: The chosen action (EMIT, THINK_MORE, CLARIFY)
        confidence: Confidence in the decision [0, 1]
        action_logits: Raw logits for each action [3]
        thinking_steps_used: How many extra thinking steps have been used
        thinking_budget_remaining: How many more thinking steps allowed
        entropy_zscore: Z-score of current entropy vs baseline
        derivative_zscore: Z-score of current derivative vs baseline
    """

    action: DecisionAction
    confidence: float
    action_logits: tuple[float, float, float]  # (emit, think_more, clarify)
    thinking_steps_used: int
    thinking_budget_remaining: int
    entropy_zscore: float = 0.0
    derivative_zscore: float = 0.0

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "action_logits": {
                "emit": self.action_logits[0],
                "think_more": self.action_logits[1],
                "clarify": self.action_logits[2],
            },
            "thinking_steps_used": self.thinking_steps_used,
            "thinking_budget_remaining": self.thinking_budget_remaining,
            "entropy_zscore": self.entropy_zscore,
            "derivative_zscore": self.derivative_zscore,
        }


class DecisionGate:
    """Metacognitive gate for generation routing decisions.

    Uses running statistics to derive decisions geometrically:
    - Tracks entropy baseline via Welford's algorithm
    - Computes z-scores relative to observed behavior
    - Decisions based on statistical significance, not heuristics

    The thinking budget is derived from model geometry to prevent infinite loops.
    Budget exhaustion forces EMIT regardless of entropy state.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        """Initialize the decision gate.

        Args:
            backend: Compute backend.
            hidden_dim: Model hidden dimension for geometry-derived budget.
                If None, uses a minimal default.
        """
        self._backend = backend or get_default_backend()
        self._hidden_dim = hidden_dim

        # Derive thinking budget from geometry:
        # - Entropy converges exponentially in well-conditioned systems
        # - Number of iterations ~ log2(dimension) for convergence
        # - Minimum of 2 to allow at least one re-evaluation
        if hidden_dim is not None:
            import math
            self._thinking_budget = max(2, int(math.log2(hidden_dim)))
        else:
            self._thinking_budget = 2  # Minimal default

        self._thinking_steps_used = 0

        # Running statistics for entropy (Welford's algorithm)
        self._entropy_count = 0
        self._entropy_mean = 0.0
        self._entropy_m2 = 0.0

        # Running statistics for derivative
        self._derivative_count = 0
        self._derivative_mean = 0.0
        self._derivative_m2 = 0.0

        # Safety boundary distance (set externally by safety components)
        self._refusal_distance: float | None = None

        # Cache machine epsilon
        self._sqrt_eps: float | None = None

    def reset(self) -> None:
        """Reset internal counters for new generation."""
        self._thinking_steps_used = 0
        # Note: We preserve running statistics across generations
        # to build a stable baseline. Only reset thinking counter.

    def reset_statistics(self) -> None:
        """Reset all running statistics. Call when model changes."""
        self._entropy_count = 0
        self._entropy_mean = 0.0
        self._entropy_m2 = 0.0
        self._derivative_count = 0
        self._derivative_mean = 0.0
        self._derivative_m2 = 0.0
        self._refusal_distance = None

    def set_refusal_distance(self, distance: float) -> None:
        """Set the current distance to safety boundary.

        Called by safety components to inform the gate about
        proximity to refusal regions. Lower = closer to danger.

        Args:
            distance: Distance to refusal boundary in activation space.
        """
        self._refusal_distance = distance

    def decide(
        self,
        entropy_state: EntropyState,
        hidden_state: Array | None = None,
    ) -> Decision:
        """Make a routing decision based on geometry-derived statistics.

        The decision follows this priority:
        1. Safety: CLARIFY if approaching refusal boundary
        2. Budget: EMIT if thinking budget exhausted
        3. Confidence: EMIT if entropy low/converging
        4. Exploration: THINK_MORE if entropy high/diverging

        Args:
            entropy_state: Current entropy analysis from EntropyAnalyzer.
            hidden_state: Optional hidden state for future safety checks.

        Returns:
            Decision with action and diagnostic metadata.
        """
        # Ensure sqrt_eps is computed
        if self._sqrt_eps is None:
            self._sqrt_eps = self._compute_sqrt_eps()

        # Update running statistics
        self._update_entropy_stats(entropy_state.entropy)
        self._update_derivative_stats(entropy_state.entropy_derivative)

        # Compute z-scores
        entropy_zscore = self._compute_zscore(
            entropy_state.entropy,
            self._entropy_mean,
            self._get_entropy_std(),
        )
        derivative_zscore = self._compute_zscore(
            entropy_state.entropy_derivative,
            self._derivative_mean,
            self._get_derivative_std(),
        )

        # Priority 1: Safety boundary
        if self._refusal_distance is not None:
            if self._refusal_distance < self._sqrt_eps:
                return self._make_decision(
                    DecisionAction.CLARIFY,
                    entropy_zscore,
                    derivative_zscore,
                    reason="approaching_safety_boundary",
                )

        # Priority 2: Budget exhausted
        if self._thinking_steps_used >= self._thinking_budget:
            return self._make_decision(
                DecisionAction.EMIT,
                entropy_zscore,
                derivative_zscore,
                reason="budget_exhausted",
            )

        # Priority 3: Confident/converging → EMIT
        # Emit if entropy is within normal range (z < sqrt_eps)
        # OR if entropy is decreasing (derivative < 0)
        if abs(entropy_zscore) < self._sqrt_eps or entropy_state.entropy_derivative < 0:
            return self._make_decision(
                DecisionAction.EMIT,
                entropy_zscore,
                derivative_zscore,
                reason="confident",
            )

        # Priority 4: Uncertain/diverging → THINK_MORE
        # Think more if entropy is significantly above baseline
        # AND derivative is positive (not converging)
        # Threshold derived from geometry: sqrt(log2(hidden_dim))
        # - Larger models can tolerate more variance before needing extra thought
        # - Scales smoothly: 768-dim → 3.1, 4096-dim → 3.5
        if self._hidden_dim is not None:
            import math
            zscore_threshold = math.sqrt(math.log2(max(2, self._hidden_dim)))
        else:
            zscore_threshold = 2.0  # Fallback for unknown dimension
        if entropy_zscore > zscore_threshold and entropy_state.entropy_derivative > 0:
            self._thinking_steps_used += 1
            return self._make_decision(
                DecisionAction.THINK_MORE,
                entropy_zscore,
                derivative_zscore,
                reason="exploring",
            )

        # Default: EMIT
        return self._make_decision(
            DecisionAction.EMIT,
            entropy_zscore,
            derivative_zscore,
            reason="default",
        )

    def _make_decision(
        self,
        action: DecisionAction,
        entropy_zscore: float,
        derivative_zscore: float,
        reason: str = "",
    ) -> Decision:
        """Construct a Decision object with computed confidence.

        Confidence is derived from the inverse of entropy z-score:
        - Low |z| = high confidence
        - High |z| = low confidence
        """
        # Confidence decreases with increasing z-score
        # Using sigmoid-like transform: 1 / (1 + |z|)
        confidence = 1.0 / (1.0 + abs(entropy_zscore))

        # Compute action logits (higher = more preferred)
        # These are informational, not used for sampling
        emit_logit = -abs(entropy_zscore)  # Prefer emit when entropy normal
        think_logit = entropy_zscore if entropy_zscore > 0 else -10.0  # Think when high
        clarify_logit = -10.0  # Only via safety
        if self._refusal_distance is not None and self._sqrt_eps is not None:
            clarify_logit = -(self._refusal_distance / self._sqrt_eps)

        return Decision(
            action=action,
            confidence=confidence,
            action_logits=(emit_logit, think_logit, clarify_logit),
            thinking_steps_used=self._thinking_steps_used,
            thinking_budget_remaining=self._thinking_budget - self._thinking_steps_used,
            entropy_zscore=entropy_zscore,
            derivative_zscore=derivative_zscore,
        )

    def _compute_sqrt_eps(self) -> float:
        """Compute sqrt(machine_epsilon) for the backend's dtype."""
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        b = self._backend
        eps = float(machine_epsilon(b, b.array([1.0])))
        return eps ** 0.5

    def _update_entropy_stats(self, entropy: float) -> None:
        """Update running entropy statistics via Welford's algorithm."""
        self._entropy_count += 1
        if self._entropy_count == 1:
            self._entropy_mean = entropy
            self._entropy_m2 = 0.0
            return

        delta = entropy - self._entropy_mean
        self._entropy_mean += delta / self._entropy_count
        delta2 = entropy - self._entropy_mean
        self._entropy_m2 += delta * delta2

    def _update_derivative_stats(self, derivative: float) -> None:
        """Update running derivative statistics via Welford's algorithm."""
        self._derivative_count += 1
        if self._derivative_count == 1:
            self._derivative_mean = derivative
            self._derivative_m2 = 0.0
            return

        delta = derivative - self._derivative_mean
        self._derivative_mean += delta / self._derivative_count
        delta2 = derivative - self._derivative_mean
        self._derivative_m2 += delta * delta2

    def _get_entropy_std(self) -> float:
        """Get standard deviation of entropy from running stats."""
        if self._entropy_count < 2:
            return 0.0
        variance = self._entropy_m2 / (self._entropy_count - 1)
        return variance ** 0.5

    def _get_derivative_std(self) -> float:
        """Get standard deviation of derivative from running stats."""
        if self._derivative_count < 2:
            return 0.0
        variance = self._derivative_m2 / (self._derivative_count - 1)
        return variance ** 0.5

    def _compute_zscore(self, value: float, mean: float, std: float) -> float:
        """Compute z-score with numerical stability.

        Returns 0.0 if std is too small (insufficient variation).
        """
        if self._sqrt_eps is None:
            self._sqrt_eps = self._compute_sqrt_eps()

        # Threshold based on sqrt(eps) - values below this are noise
        if std < self._sqrt_eps:
            return 0.0
        return (value - mean) / std

    @property
    def thinking_steps_used(self) -> int:
        """Number of thinking steps used this generation."""
        return self._thinking_steps_used

    @property
    def entropy_baseline(self) -> float:
        """Current entropy baseline (running mean)."""
        return self._entropy_mean

    @property
    def entropy_std(self) -> float:
        """Current entropy standard deviation."""
        return self._get_entropy_std()
