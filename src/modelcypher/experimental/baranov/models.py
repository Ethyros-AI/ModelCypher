"""Core data models for Baranov Sleeping-LLM replication.

EXPERIMENTAL: Not validated for production use.

Data models:
    FactTriple      -- Immutable (subject, relation, object) with unique fact_id.
    EditStatus      -- State enum for weight edits.
    EditState       -- Immutable snapshot of a weight-edit lifecycle.
    ConsolidationStage -- Per-fact consolidation progression entry.

All dataclasses are frozen and fully hashable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Edit lifecycle
# ---------------------------------------------------------------------------


class EditStatus(str, Enum):
    """Status of a weight edit in its lifecycle.

    State machine:
        pending -> applied | failed
        applied -> consolidated | rolled_back | failed
        rolled_back -> pending
        consolidated -> (terminal)
        failed -> (terminal)
    """

    pending = "pending"
    applied = "applied"
    consolidated = "consolidated"
    rolled_back = "rolled_back"
    failed = "failed"


VALID_TRANSITIONS: dict[EditStatus, frozenset[EditStatus]] = {
    EditStatus.pending: frozenset({EditStatus.applied, EditStatus.failed}),
    EditStatus.applied: frozenset(
        {EditStatus.consolidated, EditStatus.rolled_back, EditStatus.failed},
    ),
    EditStatus.consolidated: frozenset(),
    EditStatus.rolled_back: frozenset({EditStatus.pending}),
    EditStatus.failed: frozenset(),
}


# ---------------------------------------------------------------------------
# FactTriple
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactTriple:
    """An atomic fact as (subject, relation, object) with a unique identifier.

    Frozen and hashable -- safe for use in sets and as dict keys.
    """

    subject: str
    relation: str
    object: str
    fact_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "fact_id": self.fact_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> FactTriple:
        return cls(
            subject=data["subject"],
            relation=data["relation"],
            object=data["object"],
            fact_id=data["fact_id"],
        )


# ---------------------------------------------------------------------------
# EditState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditState:
    """Immutable snapshot of a weight-edit lifecycle.

    All collection fields use immutable types (tuples) so the frozen
    dataclass is genuinely immutable and hashable.

    ``metrics`` stores key-value pairs as a sorted tuple of 2-tuples
    instead of a dict. Use the ``metrics_dict`` property for convenient
    dict access, and the ``from_metrics_dict`` classmethod for ergonomic
    construction.
    """

    edit_id: str
    fact_ids: tuple[str, ...]
    layer_ids: tuple[int, ...]
    status: EditStatus
    metrics: tuple[tuple[str, float], ...]

    @property
    def metrics_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict (read-only convenience)."""
        return dict(self.metrics)

    @classmethod
    def from_metrics_dict(
        cls,
        *,
        edit_id: str,
        fact_ids: tuple[str, ...],
        layer_ids: tuple[int, ...],
        status: EditStatus,
        metrics_dict: dict[str, float],
    ) -> EditState:
        """Construct an EditState from a dict of metrics.

        The dict is converted to a sorted tuple of pairs for immutability.
        """
        return cls(
            edit_id=edit_id,
            fact_ids=fact_ids,
            layer_ids=layer_ids,
            status=status,
            metrics=tuple(sorted(metrics_dict.items())),
        )

    def transition_to(self, new_status: EditStatus) -> EditState:
        """Return a new EditState with updated status.

        Raises ``ValueError`` if the transition is not allowed by the
        state machine defined in ``VALID_TRANSITIONS``.
        """
        allowed = VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            valid_names = sorted(s.value for s in allowed)
            raise ValueError(
                f"Invalid transition: {self.status.value} -> {new_status.value}. "
                f"Valid targets from {self.status.value}: {valid_names}",
            )
        return EditState(
            edit_id=self.edit_id,
            fact_ids=self.fact_ids,
            layer_ids=self.layer_ids,
            status=new_status,
            metrics=self.metrics,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "fact_ids": list(self.fact_ids),
            "layer_ids": list(self.layer_ids),
            "status": self.status.value,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditState:
        metrics_raw = data.get("metrics", {})
        return cls(
            edit_id=data["edit_id"],
            fact_ids=tuple(data["fact_ids"]),
            layer_ids=tuple(data["layer_ids"]),
            status=EditStatus(data["status"]),
            metrics=tuple(sorted(metrics_raw.items())),
        )


# ---------------------------------------------------------------------------
# ConsolidationStage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsolidationStage:
    """A single entry in a fact's consolidation progression history.

    ``transfer_weight`` MUST come from measured transfer strength -- not
    from a hardcoded schedule.  The replication protocol explicitly rejects
    fixed schedules such as ``[1.0, 0.5, 0.1, 0.0]`` (claim C17, rejected).
    The caller is responsible for providing a measured value.

    Attributes
    ----------
    stage_index:
        Zero-based stage ordinal in the progression.
    transfer_weight:
        Measured transfer strength at this stage.
    passed:
        Whether the fact passed recall evaluation at this stage.
    """

    stage_index: int
    transfer_weight: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "transfer_weight": self.transfer_weight,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsolidationStage:
        return cls(
            stage_index=data["stage_index"],
            transfer_weight=data["transfer_weight"],
            passed=data["passed"],
        )


__all__ = [
    "ConsolidationStage",
    "EditState",
    "EditStatus",
    "FactTriple",
    "VALID_TRANSITIONS",
]
