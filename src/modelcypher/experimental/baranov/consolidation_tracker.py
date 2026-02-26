"""Per-fact consolidation stage tracker with rollback safety.

EXPERIMENTAL: Not validated for production use.

``FactConsolidationTracker`` manages per-fact stage progression for the
Baranov replication Track C (Sleep Convergence + Per-fact Consolidation).

Key invariants:
    - Each fact has an ordered history of ``ConsolidationStage`` entries.
    - Advancement requires the current stage to have ``passed=True``.
    - Retreat (rollback) is always allowed from any stage > 0.
    - ``transfer_weight`` for new stages always comes from the caller
      (measured value), NEVER from a hardcoded schedule.
"""

from __future__ import annotations

from modelcypher.experimental.baranov.models import ConsolidationStage


class FactConsolidationTracker:
    """Tracks per-fact consolidation stage progression with rollback safety."""

    def __init__(self) -> None:
        self._history: dict[str, list[ConsolidationStage]] = {}

    def initialize_fact(
        self,
        fact_id: str,
        initial_transfer_weight: float,
    ) -> ConsolidationStage:
        """Register a fact at stage 0 with a measured transfer weight.

        Raises ``ValueError`` if the fact is already registered.
        """
        if fact_id in self._history:
            raise ValueError(f"Fact {fact_id!r} is already initialized.")
        stage = ConsolidationStage(
            stage_index=0,
            transfer_weight=initial_transfer_weight,
            passed=False,
        )
        self._history[fact_id] = [stage]
        return stage

    def advance(
        self,
        fact_id: str,
        measured_transfer_weight: float,
        passed: bool,
    ) -> ConsolidationStage:
        """Advance a fact to the next consolidation stage.

        Requires the current stage to have ``passed=True``.
        The new stage's ``transfer_weight`` is the caller-provided
        *measured_transfer_weight* -- no internal schedule.

        Raises ``ValueError`` if the fact is unknown or current stage
        has not passed.
        """
        history = self._history.get(fact_id)
        if history is None:
            raise ValueError(f"Unknown fact {fact_id!r}. Call initialize_fact first.")
        current = history[-1]
        if not current.passed:
            raise ValueError(
                f"Cannot advance fact {fact_id!r}: current stage "
                f"{current.stage_index} has not passed.",
            )
        new_stage = ConsolidationStage(
            stage_index=current.stage_index + 1,
            transfer_weight=measured_transfer_weight,
            passed=passed,
        )
        history.append(new_stage)
        return new_stage

    def retreat(
        self,
        fact_id: str,
        measured_transfer_weight: float,
    ) -> ConsolidationStage:
        """Retreat a fact to the previous consolidation stage (rollback).

        Always allowed from any stage > 0. The retreated stage is
        recorded with ``passed=False``.

        Raises ``ValueError`` if the fact is unknown or already at stage 0.
        """
        history = self._history.get(fact_id)
        if history is None:
            raise ValueError(f"Unknown fact {fact_id!r}. Call initialize_fact first.")
        current = history[-1]
        if current.stage_index == 0:
            raise ValueError(
                f"Cannot retreat fact {fact_id!r}: already at stage 0.",
            )
        new_stage = ConsolidationStage(
            stage_index=current.stage_index - 1,
            transfer_weight=measured_transfer_weight,
            passed=False,
        )
        history.append(new_stage)
        return new_stage

    def mark_passed(self, fact_id: str) -> ConsolidationStage:
        """Mark the current stage of a fact as passed.

        Returns a new stage entry with ``passed=True`` at the same
        ``stage_index`` and ``transfer_weight``.

        Raises ``ValueError`` if the fact is unknown or already passed.
        """
        history = self._history.get(fact_id)
        if history is None:
            raise ValueError(f"Unknown fact {fact_id!r}.")
        current = history[-1]
        if current.passed:
            raise ValueError(
                f"Fact {fact_id!r} stage {current.stage_index} is already passed.",
            )
        new_stage = ConsolidationStage(
            stage_index=current.stage_index,
            transfer_weight=current.transfer_weight,
            passed=True,
        )
        history.append(new_stage)
        return new_stage

    def get_current_stage(self, fact_id: str) -> ConsolidationStage | None:
        """Return the most recent stage for *fact_id*, or ``None``."""
        history = self._history.get(fact_id)
        return history[-1] if history else None

    def get_history(self, fact_id: str) -> tuple[ConsolidationStage, ...]:
        """Return the full stage history for *fact_id*."""
        return tuple(self._history.get(fact_id, []))

    def get_all_fact_ids(self) -> frozenset[str]:
        """Return all registered fact IDs."""
        return frozenset(self._history.keys())


__all__ = ["FactConsolidationTracker"]
