"""Unit tests for Baranov replication data models."""

from __future__ import annotations

import dataclasses

import pytest

from modelcypher.experimental.baranov.models import (
    VALID_TRANSITIONS,
    ConsolidationStage,
    EditState,
    EditStatus,
    FactTriple,
)

# ---------------------------------------------------------------------------
# FactTriple
# ---------------------------------------------------------------------------


class TestFactTriple:
    def test_immutable(self) -> None:
        """FactTriple fields cannot be reassigned (frozen=True)."""
        f = FactTriple(subject="Paris", relation="capital_of", object="France", fact_id="f1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.subject = "London"  # type: ignore[misc]

    def test_hashable(self) -> None:
        """FactTriple can be used in sets and as dict keys."""
        f1 = FactTriple(subject="Paris", relation="capital_of", object="France", fact_id="f1")
        f2 = FactTriple(subject="Paris", relation="capital_of", object="France", fact_id="f1")
        assert f1 == f2
        assert hash(f1) == hash(f2)
        assert len({f1, f2}) == 1

    def test_distinct_fact_ids(self) -> None:
        """Different fact_ids produce distinct objects."""
        f1 = FactTriple(subject="Paris", relation="capital_of", object="France", fact_id="f1")
        f2 = FactTriple(subject="Paris", relation="capital_of", object="France", fact_id="f2")
        assert f1 != f2

    def test_round_trip_serialization(self) -> None:
        """as_dict -> from_dict preserves all fields."""
        original = FactTriple(subject="Berlin", relation="capital_of", object="Germany", fact_id="f42")
        restored = FactTriple.from_dict(original.as_dict())
        assert restored == original

    def test_as_dict_keys(self) -> None:
        """as_dict returns expected keys."""
        f = FactTriple(subject="a", relation="b", object="c", fact_id="d")
        d = f.as_dict()
        assert set(d.keys()) == {"subject", "relation", "object", "fact_id"}


# ---------------------------------------------------------------------------
# EditStatus + transitions
# ---------------------------------------------------------------------------


class TestEditStatus:
    def test_all_statuses_in_transitions(self) -> None:
        """Every EditStatus value has an entry in VALID_TRANSITIONS."""
        for status in EditStatus:
            assert status in VALID_TRANSITIONS

    def test_terminal_states_have_no_exits(self) -> None:
        """consolidated and failed are terminal (no outgoing transitions)."""
        assert len(VALID_TRANSITIONS[EditStatus.consolidated]) == 0
        assert len(VALID_TRANSITIONS[EditStatus.failed]) == 0

    def test_rollback_can_retry(self) -> None:
        """rolled_back -> pending is allowed (retry path)."""
        assert EditStatus.pending in VALID_TRANSITIONS[EditStatus.rolled_back]


# ---------------------------------------------------------------------------
# EditState
# ---------------------------------------------------------------------------


class TestEditState:
    def _make_pending(self) -> EditState:
        return EditState.from_metrics_dict(
            edit_id="e1",
            fact_ids=("f1", "f2"),
            layer_ids=(3, 5),
            status=EditStatus.pending,
            metrics_dict={"cka_drift": 0.02},
        )

    def test_immutable(self) -> None:
        """EditState fields cannot be reassigned."""
        es = self._make_pending()
        with pytest.raises(dataclasses.FrozenInstanceError):
            es.edit_id = "e2"  # type: ignore[misc]

    def test_hashable(self) -> None:
        """EditState is hashable (all fields are immutable types)."""
        es1 = self._make_pending()
        es2 = self._make_pending()
        assert hash(es1) == hash(es2)
        assert es1 == es2

    def test_metrics_dict_property(self) -> None:
        """metrics_dict returns the metrics as a dict."""
        es = self._make_pending()
        assert es.metrics_dict == {"cka_drift": 0.02}

    def test_valid_transitions(self) -> None:
        """All valid transitions produce a new EditState with updated status."""
        es = self._make_pending()
        applied = es.transition_to(EditStatus.applied)
        assert applied.status == EditStatus.applied
        assert applied.edit_id == es.edit_id

        consolidated = applied.transition_to(EditStatus.consolidated)
        assert consolidated.status == EditStatus.consolidated

    def test_transition_pending_to_failed(self) -> None:
        es = self._make_pending()
        failed = es.transition_to(EditStatus.failed)
        assert failed.status == EditStatus.failed

    def test_applied_to_rolled_back(self) -> None:
        es = self._make_pending().transition_to(EditStatus.applied)
        rb = es.transition_to(EditStatus.rolled_back)
        assert rb.status == EditStatus.rolled_back

    def test_rolled_back_to_pending(self) -> None:
        es = (
            self._make_pending()
            .transition_to(EditStatus.applied)
            .transition_to(EditStatus.rolled_back)
        )
        pending = es.transition_to(EditStatus.pending)
        assert pending.status == EditStatus.pending

    def test_invalid_transition_raises(self) -> None:
        """Skip-transition (pending -> consolidated) raises ValueError."""
        es = self._make_pending()
        with pytest.raises(ValueError, match="Invalid transition"):
            es.transition_to(EditStatus.consolidated)

    def test_terminal_transition_raises(self) -> None:
        """Transition from terminal state raises ValueError."""
        es = self._make_pending().transition_to(EditStatus.failed)
        with pytest.raises(ValueError, match="Invalid transition"):
            es.transition_to(EditStatus.pending)

    def test_transition_preserves_data(self) -> None:
        """transition_to returns a new object; original is unchanged."""
        original = self._make_pending()
        applied = original.transition_to(EditStatus.applied)
        assert original.status == EditStatus.pending
        assert applied.status == EditStatus.applied
        assert original.fact_ids == applied.fact_ids
        assert original.metrics == applied.metrics

    def test_round_trip_serialization(self) -> None:
        """as_dict -> from_dict preserves all fields."""
        original = self._make_pending().transition_to(EditStatus.applied)
        restored = EditState.from_dict(original.as_dict())
        assert restored.edit_id == original.edit_id
        assert restored.status == original.status
        assert restored.fact_ids == original.fact_ids
        assert restored.layer_ids == original.layer_ids
        assert dict(restored.metrics) == dict(original.metrics)


# ---------------------------------------------------------------------------
# ConsolidationStage
# ---------------------------------------------------------------------------


class TestConsolidationStage:
    def test_frozen(self) -> None:
        """ConsolidationStage is immutable."""
        stage = ConsolidationStage(stage_index=0, transfer_weight=0.73, passed=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stage.passed = True  # type: ignore[misc]

    def test_hashable(self) -> None:
        """ConsolidationStage is hashable."""
        s1 = ConsolidationStage(stage_index=0, transfer_weight=0.73, passed=False)
        s2 = ConsolidationStage(stage_index=0, transfer_weight=0.73, passed=False)
        assert hash(s1) == hash(s2)

    def test_transfer_weight_accepts_arbitrary_floats(self) -> None:
        """transfer_weight is not restricted to a fixed schedule."""
        for w in [0.0, 0.123456789, 0.999, 1.0, 0.0001]:
            stage = ConsolidationStage(stage_index=0, transfer_weight=w, passed=False)
            assert stage.transfer_weight == w

    def test_round_trip_serialization(self) -> None:
        """as_dict -> from_dict round-trip."""
        original = ConsolidationStage(stage_index=2, transfer_weight=0.42, passed=True)
        restored = ConsolidationStage.from_dict(original.as_dict())
        assert restored == original
