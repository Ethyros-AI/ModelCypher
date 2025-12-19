from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ModelState(str, Enum):
    confident = "confident"
    nominal = "nominal"
    uncertain = "uncertain"
    exploring = "exploring"
    distressed = "distressed"
    halted = "halted"

    @property
    def display_name(self) -> str:
        mapping = {
            ModelState.confident: "Confident",
            ModelState.nominal: "Normal",
            ModelState.uncertain: "Uncertain",
            ModelState.exploring: "Exploring",
            ModelState.distressed: "Distressed",
            ModelState.halted: "Halted",
        }
        return mapping[self]

    @property
    def display_color(self) -> str:
        mapping = {
            ModelState.confident: "green",
            ModelState.nominal: "blue",
            ModelState.uncertain: "yellow",
            ModelState.exploring: "orange",
            ModelState.distressed: "red",
            ModelState.halted: "gray",
        }
        return mapping[self]

    @property
    def symbol_name(self) -> str:
        mapping = {
            ModelState.confident: "checkmark.circle.fill",
            ModelState.nominal: "circle.fill",
            ModelState.uncertain: "questionmark.circle.fill",
            ModelState.exploring: "arrow.triangle.branch",
            ModelState.distressed: "exclamationmark.triangle.fill",
            ModelState.halted: "stop.circle.fill",
        }
        return mapping[self]

    @property
    def explanation(self) -> str:
        mapping = {
            ModelState.confident: "Model is generating with high confidence in well-known territory.",
            ModelState.nominal: "Normal generation with healthy token diversity.",
            ModelState.uncertain: "Model is uncertain - consider this output less reliable.",
            ModelState.exploring: "Model is venturing into less familiar territory.",
            ModelState.distressed: "Model shows signs of being pushed into aversive territory.",
            ModelState.halted: "Generation paused due to safety threshold.",
        }
        return mapping[self]

    @property
    def requires_caution(self) -> bool:
        if self in (ModelState.confident, ModelState.nominal):
            return False
        return True

    @property
    def severity_level(self) -> int:
        mapping = {
            ModelState.confident: 0,
            ModelState.nominal: 1,
            ModelState.uncertain: 2,
            ModelState.exploring: 3,
            ModelState.distressed: 4,
            ModelState.halted: 5,
        }
        return mapping[self]


@dataclass(frozen=True)
class StateTransition:
    from_state: ModelState
    to_state: ModelState
    token_index: int
    entropy: float
    variance: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None

    @property
    def is_escalation(self) -> bool:
        return self.to_state.severity_level > self.from_state.severity_level

    @property
    def is_recovery(self) -> bool:
        return self.to_state.severity_level < self.from_state.severity_level

    @property
    def severity_delta(self) -> int:
        return self.to_state.severity_level - self.from_state.severity_level

    @property
    def description(self) -> str:
        if self.is_escalation:
            direction = "escalated"
        elif self.is_recovery:
            direction = "recovered"
        else:
            direction = "changed"
        return (
            f"State {direction} from {self.from_state.display_name} to "
            f"{self.to_state.display_name} at token {self.token_index}"
        )
