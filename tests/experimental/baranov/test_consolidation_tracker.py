"""Unit tests for FactConsolidationTracker."""

from __future__ import annotations

import pytest

from modelcypher.experimental.baranov.consolidation_tracker import (
    FactConsolidationTracker,
)
from modelcypher.experimental.baranov.models import ConsolidationStage


class TestFactConsolidationTracker:
    def test_initialize_and_get_stage(self) -> None:
        """Initialize a fact and verify it starts at stage 0, passed=False."""
        tracker = FactConsolidationTracker()
        stage = tracker.initialize_fact("f1", initial_transfer_weight=0.85)
        assert stage.stage_index == 0
        assert stage.transfer_weight == 0.85
        assert stage.passed is False

        current = tracker.get_current_stage("f1")
        assert current == stage

    def test_initialize_duplicate_raises(self) -> None:
        """Cannot initialize the same fact_id twice."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.5)
        with pytest.raises(ValueError, match="already initialized"):
            tracker.initialize_fact("f1", 0.5)

    def test_advance_requires_passed(self) -> None:
        """Cannot advance a fact whose current stage has not passed."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.8)
        with pytest.raises(ValueError, match="has not passed"):
            tracker.advance("f1", measured_transfer_weight=0.6, passed=True)

    def test_advance_after_mark_passed(self) -> None:
        """After marking passed, advance succeeds."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.8)
        tracker.mark_passed("f1")
        stage = tracker.advance("f1", measured_transfer_weight=0.6, passed=False)
        assert stage.stage_index == 1
        assert stage.transfer_weight == 0.6
        assert stage.passed is False

    def test_advance_increments_stage_index(self) -> None:
        """Each advance increments stage_index by 1."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")

        s1 = tracker.advance("f1", 0.7, passed=True)
        assert s1.stage_index == 1
        s2 = tracker.advance("f1", 0.5, passed=True)
        assert s2.stage_index == 2

    def test_advance_uses_measured_transfer_weight(self) -> None:
        """The transfer_weight on the new stage is the argument, not a schedule."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")

        # Arbitrary, non-schedule weights
        s1 = tracker.advance("f1", 0.314159, passed=True)
        assert s1.transfer_weight == 0.314159
        s2 = tracker.advance("f1", 0.271828, passed=False)
        assert s2.transfer_weight == 0.271828

    def test_retreat_always_allowed(self) -> None:
        """Retreat works from any stage > 0, regardless of passed status."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")
        tracker.advance("f1", 0.7, passed=False)  # stage 1, not passed

        stage = tracker.retreat("f1", measured_transfer_weight=0.85)
        assert stage.stage_index == 0
        assert stage.passed is False
        assert stage.transfer_weight == 0.85

    def test_retreat_from_stage_zero_raises(self) -> None:
        """Cannot retreat below stage 0."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        with pytest.raises(ValueError, match="already at stage 0"):
            tracker.retreat("f1", 0.9)

    def test_retreat_sets_passed_false(self) -> None:
        """Retreated stage always has passed=False."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")
        tracker.advance("f1", 0.7, passed=True)  # stage 1, passed

        retreated = tracker.retreat("f1", 0.85)
        assert retreated.passed is False

    def test_history_preserved(self) -> None:
        """Full history of stage transitions is accessible."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")
        tracker.advance("f1", 0.7, passed=True)
        tracker.advance("f1", 0.5, passed=False)

        history = tracker.get_history("f1")
        assert len(history) == 4  # init, mark_passed, advance, advance
        assert history[0].stage_index == 0
        assert history[1].stage_index == 0  # mark_passed replays stage 0
        assert history[2].stage_index == 1
        assert history[3].stage_index == 2

    def test_rollback_then_re_advance(self) -> None:
        """After retreat, can advance again with new measured weight."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")
        tracker.advance("f1", 0.7, passed=True)

        # Retreat from stage 1 -> stage 0
        tracker.retreat("f1", 0.85)
        current = tracker.get_current_stage("f1")
        assert current is not None
        assert current.stage_index == 0
        assert current.passed is False

        # Mark passed and re-advance with different weight
        tracker.mark_passed("f1")
        new_stage = tracker.advance("f1", 0.65, passed=False)
        assert new_stage.stage_index == 1
        assert new_stage.transfer_weight == 0.65

    def test_get_all_fact_ids(self) -> None:
        """get_all_fact_ids returns all registered facts."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.initialize_fact("f2", 0.8)
        assert tracker.get_all_fact_ids() == frozenset({"f1", "f2"})

    def test_unknown_fact_raises(self) -> None:
        """Operations on unregistered facts raise ValueError."""
        tracker = FactConsolidationTracker()
        with pytest.raises(ValueError, match="Unknown fact"):
            tracker.advance("f999", 0.5, passed=True)
        with pytest.raises(ValueError, match="Unknown fact"):
            tracker.retreat("f999", 0.5)

    def test_get_current_stage_unknown_returns_none(self) -> None:
        """get_current_stage returns None for unknown facts."""
        tracker = FactConsolidationTracker()
        assert tracker.get_current_stage("f999") is None

    def test_mark_passed_already_passed_raises(self) -> None:
        """Cannot mark a stage passed if it already is."""
        tracker = FactConsolidationTracker()
        tracker.initialize_fact("f1", 0.9)
        tracker.mark_passed("f1")
        with pytest.raises(ValueError, match="already passed"):
            tracker.mark_passed("f1")
