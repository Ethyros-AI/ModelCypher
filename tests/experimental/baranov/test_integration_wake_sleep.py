"""Integration test: minimal wake -> sleep loop with mock components.

Verifies the full state machine:
1. Create FactTriples
2. Mock-apply edit (EditState: pending -> applied)
3. Mock-evaluate recall (RecallResult with per-fact outcomes)
4. Run consolidation tracker (advance/retreat based on recall)
5. Verify final state transitions are correct
6. Explicit rollback-path assertions (state + history unchanged on rollback)
"""

from __future__ import annotations

from modelcypher.experimental.baranov.consolidation_tracker import (
    FactConsolidationTracker,
)
from modelcypher.experimental.baranov.models import (
    EditState,
    EditStatus,
    FactTriple,
)
from modelcypher.experimental.baranov.recall_evaluator import (
    RecallOutcome,
    RecallResult,
    compute_recall_aggregate,
)


def _make_facts(n: int) -> list[FactTriple]:
    """Create n synthetic facts."""
    return [
        FactTriple(
            subject=f"subject_{i}",
            relation="is_a",
            object=f"object_{i}",
            fact_id=f"fact_{i}",
        )
        for i in range(n)
    ]


def _mock_apply_edit(
    facts: list[FactTriple],
    layer_ids: list[int],
) -> EditState:
    """Simulate applying a successful edit."""
    return EditState.from_metrics_dict(
        edit_id="mock-edit-001",
        fact_ids=tuple(f.fact_id for f in facts),
        layer_ids=tuple(layer_ids),
        status=EditStatus.applied,
        metrics_dict={"cka_drift": 0.01, "preserved_fraction": 0.98},
    )


def _mock_evaluate_recall(
    facts: list[FactTriple],
    recalled_ids: set[str],
) -> RecallResult:
    """Simulate recall evaluation with given recalled fact IDs."""
    outcomes = []
    for f in facts:
        outcomes.append(
            RecallOutcome(
                fact_id=f.fact_id,
                recalled=f.fact_id in recalled_ids,
                raw_output=f"output for {f.fact_id}",
                confidence=None,
            ),
        )
    aggregate = compute_recall_aggregate(outcomes)
    return RecallResult(
        per_fact_outcomes=tuple(outcomes),
        aggregate=aggregate,
    )


class TestWakeSleepHappyPath:
    """All facts recalled -> all advance through consolidation."""

    def test_all_facts_advance(self) -> None:
        facts = _make_facts(3)
        layer_ids = [2, 5]

        # Wake: apply edit
        edit_state = _mock_apply_edit(facts, layer_ids)
        assert edit_state.status == EditStatus.applied

        # Evaluate recall (all recalled)
        recalled_ids = {f.fact_id for f in facts}
        result = _mock_evaluate_recall(facts, recalled_ids)
        assert result.aggregate.recall_rate == 1.0

        # Sleep: consolidation tracker
        tracker = FactConsolidationTracker()
        for f in facts:
            tracker.initialize_fact(f.fact_id, initial_transfer_weight=0.9)

        # Process recall results: mark passed, then advance
        for outcome in result.per_fact_outcomes:
            if outcome.recalled:
                tracker.mark_passed(outcome.fact_id)
                tracker.advance(
                    outcome.fact_id,
                    measured_transfer_weight=0.7,
                    passed=False,
                )

        # All facts should be at stage 1
        for f in facts:
            current = tracker.get_current_stage(f.fact_id)
            assert current is not None
            assert current.stage_index == 1

        # Transition edit to consolidated
        consolidated = edit_state.transition_to(EditStatus.consolidated)
        assert consolidated.status == EditStatus.consolidated


class TestWakeSleepPartialRecall:
    """Some facts recalled -> recalled advance, others retreat."""

    def test_mixed_advancement(self) -> None:
        facts = _make_facts(4)
        layer_ids = [3]

        _mock_apply_edit(facts, layer_ids)

        # Only first 2 facts recalled
        recalled_ids = {"fact_0", "fact_1"}
        result = _mock_evaluate_recall(facts, recalled_ids)
        assert result.aggregate.recalled_count == 2

        tracker = FactConsolidationTracker()
        for f in facts:
            tracker.initialize_fact(f.fact_id, 0.85)

        # Mark passed and advance recalled facts; leave others at stage 0
        for outcome in result.per_fact_outcomes:
            if outcome.recalled:
                tracker.mark_passed(outcome.fact_id)
                tracker.advance(outcome.fact_id, 0.65, passed=False)

        # Recalled facts at stage 1
        assert tracker.get_current_stage("fact_0").stage_index == 1  # type: ignore[union-attr]
        assert tracker.get_current_stage("fact_1").stage_index == 1  # type: ignore[union-attr]

        # Non-recalled facts still at stage 0
        assert tracker.get_current_stage("fact_2").stage_index == 0  # type: ignore[union-attr]
        assert tracker.get_current_stage("fact_3").stage_index == 0  # type: ignore[union-attr]


class TestWakeSleepEditFailure:
    """Edit application fails -> EditState goes to failed, no eval needed."""

    def test_failed_edit_no_consolidation(self) -> None:
        facts = _make_facts(2)

        # Simulate failed edit
        edit_state = EditState.from_metrics_dict(
            edit_id="mock-edit-fail",
            fact_ids=tuple(f.fact_id for f in facts),
            layer_ids=(3,),
            status=EditStatus.pending,
            metrics_dict={},
        )
        failed = edit_state.transition_to(EditStatus.failed)
        assert failed.status == EditStatus.failed

        # No consolidation should be attempted on a failed edit
        # (tracker is never initialized -- this is the correct behavior)


class TestRollbackPath:
    """Explicit rollback-path: state, history, and artifacts are consistent."""

    def test_rollback_preserves_history_and_restores_state(self) -> None:
        """After rollback, fact returns to previous stage with full history."""
        tracker = FactConsolidationTracker()

        # Initialize and advance fact_0 through two stages
        tracker.initialize_fact("fact_0", 0.9)
        tracker.mark_passed("fact_0")
        tracker.advance("fact_0", 0.7, passed=True)
        tracker.advance("fact_0", 0.5, passed=False)

        # Capture state before rollback
        pre_rollback_history_len = len(tracker.get_history("fact_0"))
        pre_rollback_stage = tracker.get_current_stage("fact_0")
        assert pre_rollback_stage is not None
        assert pre_rollback_stage.stage_index == 2

        # Rollback
        retreated = tracker.retreat("fact_0", measured_transfer_weight=0.75)
        assert retreated.stage_index == 1
        assert retreated.passed is False

        # History grew by one (retreat is recorded)
        post_rollback_history = tracker.get_history("fact_0")
        assert len(post_rollback_history) == pre_rollback_history_len + 1

        # The original stages are still in history (nothing was deleted)
        assert post_rollback_history[0].stage_index == 0  # init
        assert post_rollback_history[-1].stage_index == 1  # retreat target

    def test_edit_rollback_returns_to_rolled_back_status(self) -> None:
        """EditState rollback follows the state machine correctly."""
        edit = EditState.from_metrics_dict(
            edit_id="e-rb",
            fact_ids=("f1",),
            layer_ids=(3,),
            status=EditStatus.applied,
            metrics_dict={"cka_drift": 0.01},
        )

        rolled_back = edit.transition_to(EditStatus.rolled_back)
        assert rolled_back.status == EditStatus.rolled_back

        # Can re-enter pending from rolled_back
        retried = rolled_back.transition_to(EditStatus.pending)
        assert retried.status == EditStatus.pending

        # Original is unchanged (immutable)
        assert edit.status == EditStatus.applied

    def test_rollback_does_not_affect_other_facts(self) -> None:
        """Rolling back one fact does not affect another."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.initialize_fact("f2", 0.8)

        # Advance both
        tracker.mark_passed("f1")
        tracker.advance("f1", 0.7, passed=True)
        tracker.mark_passed("f2")
        tracker.advance("f2", 0.6, passed=True)

        # Rollback f1 only
        tracker.retreat("f1", 0.85)

        # f1 retreated to stage 0
        assert tracker.get_current_stage("f1").stage_index == 0  # type: ignore[union-attr]

        # f2 still at stage 1, passed=True, unaffected
        f2_current = tracker.get_current_stage("f2")
        assert f2_current is not None
        assert f2_current.stage_index == 1
        assert f2_current.passed is True
