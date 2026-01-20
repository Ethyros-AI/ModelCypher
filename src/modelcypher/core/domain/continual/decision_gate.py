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
Decision Gate - Routes between emit/think_more/clarify based on entropy state.

The gate makes metacognitive decisions about generation:
- EMIT: Confidence is sufficient, emit the token
- THINK_MORE: Uncertainty suggests extra computation would help
- CLARIFY: Uncertainty is too high, request clarification

The gate is parameterized by learnable weights, not hardcoded thresholds.
This allows training via:
1. RL with reward for correct answers
2. Distillation from a larger "teacher" model
3. Self-play with verification

Architecture:
    Input: [entropy, entropy_normalized, dH/dt, d²H/dt², variance, thinking_budget_remaining]
    Output: Decision (EMIT | THINK_MORE | CLARIFY) with confidence

The gate uses a small MLP to map entropy state to action logits, then samples
or takes argmax depending on inference mode.

Math:
    features = [H, H_norm, dH/dt, d²H/dt², var, budget]
    logits = W2 @ relu(W1 @ features + b1) + b2
    action = argmax(logits) or sample(softmax(logits))

References:
    - SOFAI-LM: Metacognition Architecture (arXiv:2508.17959)
    - SpecEE: Speculative Early Exiting (ACM 2024)
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
    """

    action: DecisionAction
    confidence: float
    action_logits: tuple[float, float, float]  # (emit, think_more, clarify)
    thinking_steps_used: int
    thinking_budget_remaining: int

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
        }


class DecisionGate:
    """Metacognitive gate for generation routing decisions.

    The gate is a small MLP that maps entropy state to action decisions.
    Parameters can be learned via RL, distillation, or self-play.

    The gate respects a "thinking budget" to prevent infinite loops.
    Once the budget is exhausted, it always returns EMIT.

    Usage:
        gate = DecisionGate(max_thinking_steps=5)

        for step in generation:
            state = entropy_analyzer.analyze(logits)
            decision = gate.decide(state)

            if decision.action == DecisionAction.EMIT:
                emit_token()
            elif decision.action == DecisionAction.THINK_MORE:
                re_run_transformer()
            else:
                emit_clarification_prompt()

        gate.reset()  # For next generation
    """

    # Feature indices for the input vector
    _FEAT_ENTROPY = 0
    _FEAT_ENTROPY_NORM = 1
    _FEAT_DERIVATIVE = 2
    _FEAT_ACCELERATION = 3
    _FEAT_VARIANCE = 4
    _FEAT_BUDGET = 5
    _NUM_FEATURES = 6

    # Action indices
    _ACT_EMIT = 0
    _ACT_THINK = 1
    _ACT_CLARIFY = 2
    _NUM_ACTIONS = 3

    def __init__(
        self,
        max_thinking_steps: int = 5,
        hidden_dim: int = 16,
        temperature: float = 1.0,
        deterministic: bool = True,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the decision gate.

        Args:
            max_thinking_steps: Maximum extra thinking steps per token.
                Prevents infinite loops.
            hidden_dim: Hidden dimension of the MLP.
            temperature: Softmax temperature for sampling mode.
            deterministic: If True, use argmax. If False, sample.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._max_thinking_steps = max_thinking_steps
        self._hidden_dim = hidden_dim
        self._temperature = temperature
        self._deterministic = deterministic

        # Thinking budget tracking
        self._thinking_steps_used = 0

        # Initialize MLP weights
        # Xavier initialization for better gradient flow
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights with Xavier initialization."""
        b = self._backend

        # Layer 1: [num_features, hidden_dim]
        # Xavier: std = sqrt(2 / (fan_in + fan_out))
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        std1 = sqrt_scalar(2.0 / (self._NUM_FEATURES + self._hidden_dim), b)
        self._w1 = b.random_normal((self._NUM_FEATURES, self._hidden_dim)) * std1
        self._b1 = b.zeros((self._hidden_dim,))

        # Layer 2: [hidden_dim, num_actions]
        std2 = sqrt_scalar(2.0 / (self._hidden_dim + self._NUM_ACTIONS), b)
        self._w2 = b.random_normal((self._hidden_dim, self._NUM_ACTIONS)) * std2
        self._b2 = b.zeros((self._NUM_ACTIONS,))

        # Initialize biases to favor EMIT initially (stable starting point)
        # This makes the gate conservative - it emits by default
        emit_bias = b.array([1.0, 0.0, 0.0])  # Slight bias toward emit
        self._b2 = emit_bias

        b.eval(self._w1, self._b1, self._w2, self._b2)

    def reset(self) -> None:
        """Reset thinking budget for new generation."""
        self._thinking_steps_used = 0

    def decide(
        self,
        entropy_state: EntropyState,
        hidden_state: Array | None = None,
    ) -> Decision:
        """Make a decision based on entropy state.

        Args:
            entropy_state: Current entropy analysis from EntropyAnalyzer.
            hidden_state: Optional hidden state for richer features.
                Not used in basic gate, but available for extensions.

        Returns:
            Decision with action, confidence, and diagnostics.
        """
        b = self._backend

        # Check if thinking budget exhausted
        budget_remaining = self._max_thinking_steps - self._thinking_steps_used
        if budget_remaining <= 0:
            # Force emit when budget exhausted
            return Decision(
                action=DecisionAction.EMIT,
                confidence=1.0,
                action_logits=(1.0, 0.0, 0.0),
                thinking_steps_used=self._thinking_steps_used,
                thinking_budget_remaining=0,
            )

        # Build feature vector
        # Normalize budget to [0, 1] range
        budget_normalized = budget_remaining / self._max_thinking_steps

        features = b.array(
            [
                entropy_state.entropy,
                entropy_state.entropy_normalized,
                entropy_state.entropy_derivative,
                entropy_state.entropy_acceleration,
                entropy_state.logit_variance,
                budget_normalized,
            ]
        )

        # Forward pass through MLP
        # Layer 1: ReLU(W1 @ x + b1)
        hidden = b.matmul(features[None, :], self._w1)[0] + self._b1
        hidden = b.maximum(hidden, b.zeros_like(hidden))  # ReLU

        # Layer 2: W2 @ hidden + b2
        logits = b.matmul(hidden[None, :], self._w2)[0] + self._b2

        b.eval(logits)

        # Convert to probabilities
        if self._deterministic:
            # Argmax for deterministic inference
            logits_list = b.tolist(logits)
            action_idx = logits_list.index(max(logits_list))
            confidence = 1.0  # Deterministic = full confidence in chosen action
        else:
            # Softmax with temperature for sampling
            scaled_logits = logits / self._temperature
            max_logit = b.max(scaled_logits)
            exp_logits = b.exp(scaled_logits - max_logit)
            probs = exp_logits / b.sum(exp_logits)
            b.eval(probs)

            # Sample from distribution
            probs_list = b.tolist(probs)
            import random

            action_idx = random.choices(range(3), weights=probs_list)[0]
            confidence = probs_list[action_idx]

        # Map index to action
        action = [DecisionAction.EMIT, DecisionAction.THINK_MORE, DecisionAction.CLARIFY][
            action_idx
        ]

        # Update thinking budget if we chose to think more
        if action == DecisionAction.THINK_MORE:
            self._thinking_steps_used += 1

        return Decision(
            action=action,
            confidence=confidence,
            action_logits=tuple(b.tolist(logits)),  # type: ignore
            thinking_steps_used=self._thinking_steps_used,
            thinking_budget_remaining=budget_remaining
            - (1 if action == DecisionAction.THINK_MORE else 0),
        )

    def get_parameters(self) -> dict[str, Array]:
        """Get learnable parameters for training.

        Returns:
            Dictionary of parameter name to tensor.
        """
        return {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
        }

    def set_parameters(self, params: dict[str, Array]) -> None:
        """Set parameters from training.

        Args:
            params: Dictionary of parameter name to tensor.
        """
        if "w1" in params:
            self._w1 = params["w1"]
        if "b1" in params:
            self._b1 = params["b1"]
        if "w2" in params:
            self._w2 = params["w2"]
        if "b2" in params:
            self._b2 = params["b2"]

    def compute_loss(
        self,
        entropy_state: EntropyState,
        target_action: DecisionAction,
    ) -> Array:
        """Compute cross-entropy loss for training.

        Args:
            entropy_state: Input entropy state.
            target_action: Ground truth action.

        Returns:
            Scalar loss tensor.
        """
        b = self._backend

        # Build feature vector
        budget_normalized = (
            self._max_thinking_steps - self._thinking_steps_used
        ) / self._max_thinking_steps

        features = b.array(
            [
                entropy_state.entropy,
                entropy_state.entropy_normalized,
                entropy_state.entropy_derivative,
                entropy_state.entropy_acceleration,
                entropy_state.logit_variance,
                budget_normalized,
            ]
        )

        # Forward pass
        hidden = b.matmul(features[None, :], self._w1)[0] + self._b1
        hidden = b.maximum(hidden, b.zeros_like(hidden))
        logits = b.matmul(hidden[None, :], self._w2)[0] + self._b2

        # Cross-entropy loss
        # target one-hot
        target_idx = [DecisionAction.EMIT, DecisionAction.THINK_MORE, DecisionAction.CLARIFY].index(
            target_action
        )

        # Softmax
        max_logit = b.max(logits)
        exp_logits = b.exp(logits - max_logit)
        log_sum_exp = b.log(b.sum(exp_logits))
        log_probs = logits - max_logit - log_sum_exp

        # Negative log likelihood of target
        target_log_prob = b.take(log_probs, b.array([target_idx]), axis=0)
        loss = -target_log_prob

        return loss

    @property
    def max_thinking_steps(self) -> int:
        """Maximum thinking steps allowed."""
        return self._max_thinking_steps

    @property
    def thinking_steps_used(self) -> int:
        """Thinking steps used so far."""
        return self._thinking_steps_used
