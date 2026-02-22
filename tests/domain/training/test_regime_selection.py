# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for automatic regime selection from baseline online evaluation."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.online_eval import OnlineEvalResult
from modelcypher.core.domain.training.regime_selection import (
    _clopper_pearson_interval,
    select_training_regime,
)


def _make_baseline_result(
    *,
    per_type_correct: dict[str, int],
    per_type_total: dict[str, int],
) -> OnlineEvalResult:
    n_total = sum(per_type_total.values())
    n_correct = sum(per_type_correct.get(problem_type, 0) for problem_type in per_type_total)
    per_type_accuracy = {
        problem_type: (
            per_type_correct.get(problem_type, 0) / float(total) if total > 0 else 0.0
        )
        for problem_type, total in per_type_total.items()
    }
    accuracy = n_correct / float(n_total) if n_total > 0 else 0.0
    return OnlineEvalResult(
        epoch=0,
        accuracy=accuracy,
        n_correct=n_correct,
        n_total=n_total,
        correct_ids=frozenset(),
        baseline_n_correct=n_correct,
        baseline_accuracy=accuracy,
        n_lost=0,
        n_gained=0,
        degraded=False,
        per_type_accuracy=per_type_accuracy,
        per_type_correct=dict(per_type_correct),
        per_type_total=dict(per_type_total),
    )


def test_all_zero_baseline_selects_ce_only():
    per_type_total = {
        "syllogistic_chain": 5,
        "contrapositive": 5,
        "set_exception": 5,
        "multi_step_arithmetic": 5,
        "compositional_binding": 5,
    }
    per_type_correct = {problem_type: 0 for problem_type in per_type_total}
    baseline = _make_baseline_result(
        per_type_correct=per_type_correct,
        per_type_total=per_type_total,
    )

    decision = select_training_regime(baseline)

    assert decision.global_regime == "ce"
    assert decision.use_ce is True
    assert decision.use_outcome_training is False
    assert decision.use_entropy_regularization is False
    assert all(entry.regime == "ce" for entry in decision.per_type.values())


def test_all_perfect_baseline_selects_outcome_regime():
    per_type_total = {
        "syllogistic_chain": 20,
        "contrapositive": 20,
        "set_exception": 20,
        "multi_step_arithmetic": 20,
        "compositional_binding": 20,
    }
    per_type_correct = dict(per_type_total)
    baseline = _make_baseline_result(
        per_type_correct=per_type_correct,
        per_type_total=per_type_total,
    )

    decision = select_training_regime(baseline)

    assert decision.global_regime in {"reinforce", "reinforce_entropy"}
    assert decision.use_outcome_training is True
    assert all(entry.regime in {"reinforce", "reinforce_entropy"} for entry in decision.per_type.values())


def test_mixed_baseline_selects_hybrid():
    baseline = _make_baseline_result(
        per_type_correct={
            "syllogistic_chain": 0,
            "contrapositive": 18,
            "set_exception": 0,
            "multi_step_arithmetic": 4,
            "compositional_binding": 0,
        },
        per_type_total={
            "syllogistic_chain": 20,
            "contrapositive": 20,
            "set_exception": 20,
            "multi_step_arithmetic": 20,
            "compositional_binding": 20,
        },
    )

    decision = select_training_regime(baseline)

    assert decision.global_regime == "hybrid"
    assert decision.use_outcome_training is True
    assert decision.per_type["syllogistic_chain"].regime == "ce"
    assert decision.per_type["compositional_binding"].regime == "ce"


def test_small_n_is_conservative_for_binary_types():
    baseline_small = _make_baseline_result(
        per_type_correct={
            "syllogistic_chain": 1,
            "contrapositive": 1,
            "set_exception": 1,
            "multi_step_arithmetic": 1,
            "compositional_binding": 1,
        },
        per_type_total={
            "syllogistic_chain": 1,
            "contrapositive": 1,
            "set_exception": 1,
            "multi_step_arithmetic": 1,
            "compositional_binding": 1,
        },
    )

    decision_small = select_training_regime(baseline_small)

    assert decision_small.global_regime == "reinforce_entropy"
    assert decision_small.use_entropy_regularization is True
    assert decision_small.per_type["syllogistic_chain"].regime == "reinforce_entropy"
    assert decision_small.per_type["contrapositive"].regime == "reinforce_entropy"
    assert decision_small.per_type["set_exception"].regime == "reinforce_entropy"


def test_large_n_produces_sharper_confidence_bounds():
    baseline_small = _make_baseline_result(
        per_type_correct={
            "syllogistic_chain": 1,
            "contrapositive": 1,
            "set_exception": 1,
            "multi_step_arithmetic": 1,
            "compositional_binding": 1,
        },
        per_type_total={
            "syllogistic_chain": 1,
            "contrapositive": 1,
            "set_exception": 1,
            "multi_step_arithmetic": 1,
            "compositional_binding": 1,
        },
    )
    baseline_large = _make_baseline_result(
        per_type_correct={
            "syllogistic_chain": 20,
            "contrapositive": 20,
            "set_exception": 20,
            "multi_step_arithmetic": 20,
            "compositional_binding": 20,
        },
        per_type_total={
            "syllogistic_chain": 20,
            "contrapositive": 20,
            "set_exception": 20,
            "multi_step_arithmetic": 20,
            "compositional_binding": 20,
        },
    )

    decision_small = select_training_regime(baseline_small)
    decision_large = select_training_regime(baseline_large)

    small_lower = decision_small.per_type["syllogistic_chain"].ci_lower
    large_lower = decision_large.per_type["syllogistic_chain"].ci_lower
    assert large_lower > small_lower
    assert decision_large.per_type["syllogistic_chain"].regime == "reinforce"


def test_ci_boundary_equal_to_chance_uses_reinforce():
    baseline = _make_baseline_result(
        per_type_correct={"contrapositive": 10},
        per_type_total={"contrapositive": 20},
    )
    initial = select_training_regime(baseline)
    boundary = initial.per_type["contrapositive"].ci_lower

    decision = select_training_regime(
        baseline,
        problem_type_chance_rates={"contrapositive": boundary},
    )

    assert decision.per_type["contrapositive"].ci_lower == pytest.approx(boundary)
    assert decision.per_type["contrapositive"].regime == "reinforce"


def test_derivation_log_contains_inputs_and_decision():
    baseline = _make_baseline_result(
        per_type_correct={"contrapositive": 12},
        per_type_total={"contrapositive": 20},
    )

    decision = select_training_regime(baseline)

    assert len(decision.derivation_log) >= 3
    assert any("confidence derivation" in line for line in decision.derivation_log)
    assert any("contrapositive: k=12, n=20" in line for line in decision.derivation_log)
    assert any("global decision:" in line for line in decision.derivation_log)


def test_unknown_problem_type_defaults_to_zero_chance():
    baseline = _make_baseline_result(
        per_type_correct={"novel_reasoning": 1},
        per_type_total={"novel_reasoning": 8},
    )

    decision = select_training_regime(baseline)
    unknown = decision.per_type["novel_reasoning"]

    assert unknown.chance_rate == 0.0
    assert unknown.regime == "reinforce"


def test_select_training_regime_is_deterministic():
    baseline = _make_baseline_result(
        per_type_correct={
            "syllogistic_chain": 2,
            "contrapositive": 2,
            "set_exception": 2,
            "multi_step_arithmetic": 2,
            "compositional_binding": 2,
        },
        per_type_total={
            "syllogistic_chain": 5,
            "contrapositive": 5,
            "set_exception": 5,
            "multi_step_arithmetic": 5,
            "compositional_binding": 5,
        },
    )

    first = select_training_regime(baseline)
    second = select_training_regime(baseline)

    assert first == second
    assert first.derivation_log == second.derivation_log


def test_clopper_pearson_matches_known_tabulated_values():
    lower, upper = _clopper_pearson_interval(
        n_correct=5,
        n_total=10,
        alpha=0.1,
    )
    assert lower == pytest.approx(0.22244110100812936)
    assert upper == pytest.approx(0.7775588989918706)
