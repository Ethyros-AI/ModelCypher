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
Decision Gate - Emits by default without user-tuned thresholds.

The gate reports a deterministic decision and leaves policy choices to the
caller. This avoids heuristic thresholds or learned parameters in core logic.
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

    This implementation avoids heuristic thresholds and user-configurable
    parameters by emitting a neutral, deterministic decision payload.
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()
        self._thinking_steps_used = 0

    def reset(self) -> None:
        """Reset internal counters for new generation."""
        self._thinking_steps_used = 0

    def decide(
        self,
        entropy_state: EntropyState,
        hidden_state: Array | None = None,
    ) -> Decision:
        """Return a deterministic emit decision without thresholds.

        Args:
            entropy_state: Current entropy analysis from EntropyAnalyzer.
            hidden_state: Optional hidden state (unused).

        Returns:
            Decision with action EMIT and neutral diagnostics.
        """
        return Decision(
            action=DecisionAction.EMIT,
            confidence=1.0,
            action_logits=(0.0, 0.0, 0.0),
            thinking_steps_used=self._thinking_steps_used,
            thinking_budget_remaining=0,
        )
