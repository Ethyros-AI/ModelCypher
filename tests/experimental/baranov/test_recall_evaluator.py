"""Unit tests for recall evaluator interface and aggregate computation."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.statistics import clopper_pearson_interval
from modelcypher.experimental.baranov.recall_evaluator import (
    RecallAggregate,
    RecallOutcome,
    compute_recall_aggregate,
)


def _make_outcomes(
    total: int,
    recalled: int,
) -> list[RecallOutcome]:
    """Helper: create *total* outcomes with the first *recalled* marked True."""
    outcomes = []
    for i in range(total):
        outcomes.append(
            RecallOutcome(
                fact_id=f"f{i}",
                recalled=i < recalled,
                raw_output=f"output_{i}",
                confidence=None,
            ),
        )
    return outcomes


class TestComputeRecallAggregate:
    def test_basic_aggregate(self) -> None:
        """3/5 recalled produces correct rate."""
        agg = compute_recall_aggregate(_make_outcomes(5, 3))
        assert agg.total == 5
        assert agg.recalled_count == 3
        assert agg.recall_rate == pytest.approx(0.6)

    def test_empty_outcomes(self) -> None:
        """Empty list returns zero-valued aggregate with no CI."""
        agg = compute_recall_aggregate([])
        assert agg.total == 0
        assert agg.recalled_count == 0
        assert agg.recall_rate == 0.0
        assert agg.confidence_interval is None

    def test_perfect_recall(self) -> None:
        """N/N recall: rate=1.0, CI upper=1.0."""
        agg = compute_recall_aggregate(_make_outcomes(10, 10))
        assert agg.recall_rate == pytest.approx(1.0)
        assert agg.confidence_interval is not None
        assert agg.confidence_interval[1] == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        """0/N recall: rate=0.0, CI lower=0.0."""
        agg = compute_recall_aggregate(_make_outcomes(10, 0))
        assert agg.recall_rate == pytest.approx(0.0)
        assert agg.confidence_interval is not None
        assert agg.confidence_interval[0] == pytest.approx(0.0)

    def test_ci_matches_shared_function(self) -> None:
        """CI matches the shared clopper_pearson_interval directly."""
        outcomes = _make_outcomes(20, 12)
        alpha = 1.0 / 20  # default: 1/n
        expected_ci = clopper_pearson_interval(
            n_correct=12, n_total=20, alpha=alpha,
        )
        agg = compute_recall_aggregate(outcomes)
        assert agg.confidence_interval is not None
        assert agg.confidence_interval[0] == pytest.approx(expected_ci[0])
        assert agg.confidence_interval[1] == pytest.approx(expected_ci[1])

    def test_alpha_derived_from_n(self) -> None:
        """When confidence_level=None, alpha=1/n (not 0.05 or other constant)."""
        # With n=10, alpha=0.1. With n=100, alpha=0.01.
        # The CI width should differ.
        agg_10 = compute_recall_aggregate(_make_outcomes(10, 5))
        agg_100 = compute_recall_aggregate(_make_outcomes(100, 50))
        assert agg_10.confidence_interval is not None
        assert agg_100.confidence_interval is not None
        width_10 = agg_10.confidence_interval[1] - agg_10.confidence_interval[0]
        width_100 = agg_100.confidence_interval[1] - agg_100.confidence_interval[0]
        # Larger sample with tighter alpha should give narrower CI
        assert width_100 < width_10

    def test_explicit_confidence_level(self) -> None:
        """Explicit confidence_level overrides the 1/n default."""
        outcomes = _make_outcomes(10, 5)
        agg = compute_recall_aggregate(outcomes, confidence_level=0.95)
        expected_ci = clopper_pearson_interval(
            n_correct=5, n_total=10, alpha=0.05,
        )
        assert agg.confidence_interval is not None
        assert agg.confidence_interval[0] == pytest.approx(expected_ci[0])
        assert agg.confidence_interval[1] == pytest.approx(expected_ci[1])


class TestRecallOutcome:
    def test_as_dict(self) -> None:
        o = RecallOutcome(fact_id="f1", recalled=True, raw_output="Paris", confidence=0.9)
        d = o.as_dict()
        assert d["fact_id"] == "f1"
        assert d["recalled"] is True
        assert d["confidence"] == 0.9


class TestRecallAggregate:
    def test_as_dict_with_ci(self) -> None:
        agg = RecallAggregate(total=10, recalled_count=7, recall_rate=0.7, confidence_interval=(0.35, 0.93))
        d = agg.as_dict()
        assert d["confidence_interval"] == [0.35, 0.93]

    def test_as_dict_without_ci(self) -> None:
        agg = RecallAggregate(total=0, recalled_count=0, recall_rate=0.0, confidence_interval=None)
        d = agg.as_dict()
        assert d["confidence_interval"] is None
