# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for online correctness evaluation (Layer 1)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.online_eval import (
    OnlineEvalResult,
    evaluate_correctness,
)


def _make_problem(problem_id, correct_answer="yes", problem_type="test"):
    """Create a StarProblem with a simple substring verifier."""
    from modelcypher.core.domain.star.problem_generator import StarProblem

    def verifier(response: str) -> bool:
        return correct_answer.lower() in response.lower()

    return StarProblem(
        problem_id=problem_id,
        problem_type=problem_type,
        prompt="Test prompt",
        correct_answer=correct_answer,
        difficulty=1,
        verification_fn="substring_match",
        _verifier=verifier,
    )


# Arbitrary token limit for tests — not training, just generation cap.
_TEST_MAX_TOKENS = 64


class TestEvaluateCorrectness:
    """Test online eval with mocked generation and deterministic verifiers."""

    def test_baseline_measurement(self):
        """First call (baseline_correct_ids=None) establishes baseline."""
        problems = [
            _make_problem("p1", "yes"),
            _make_problem("p2", "no"),
        ]

        def gen_fn(prompt, max_tokens):
            return "yes"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=0,
            baseline_correct_ids=None,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.epoch == 0
        assert result.n_correct == 1  # only p1 matches
        assert result.n_total == 2
        assert result.accuracy == 0.5
        assert "p1" in result.correct_ids
        assert "p2" not in result.correct_ids
        assert result.degraded is False  # baseline can't be degraded
        assert result.degraded_raw is False
        assert result.degraded_significant is False
        assert result.degraded == result.degraded_significant
        assert result.alpha == pytest.approx(0.5)
        assert result.n_lost == 0
        assert result.n_gained == 0

    def test_degradation_detected_when_significant(self):
        """Large drops should be significant under CP non-overlap."""
        problems = [_make_problem(f"p{i}", "yes") for i in range(25)]
        baseline_ids = frozenset(f"p{i}" for i in range(24))

        call_count = [0]

        def gen_fn(prompt, max_tokens):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 10:
                return "yes"
            return "no"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=1,
            baseline_correct_ids=baseline_ids,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.degraded_raw is True
        assert result.degraded_significant is True
        assert result.degraded is True
        assert result.degraded == result.degraded_significant
        assert result.n_correct == 10
        assert result.baseline_n_correct == 24

    def test_degradation_raw_drop_not_significant_for_24_to_21_of_25(self):
        problems = [_make_problem(f"p{i}", "yes") for i in range(25)]
        baseline_ids = frozenset(f"p{i}" for i in range(24))

        call_count = [0]

        def gen_fn(prompt, max_tokens):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 21:
                return "yes"
            return "no"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=1,
            baseline_correct_ids=baseline_ids,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.n_correct == 21
        assert result.baseline_n_correct == 24
        assert result.degraded_raw is True
        assert result.degraded_significant is False
        assert result.degraded is False
        assert result.degraded == result.degraded_significant

    def test_improvement_detected(self):
        """When more problems are correct than baseline, degraded=False."""
        problems = [
            _make_problem("p1", "yes"),
            _make_problem("p2", "yes"),
        ]
        baseline_ids = frozenset(["p1"])

        def gen_fn(prompt, max_tokens):
            return "yes"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=1,
            baseline_correct_ids=baseline_ids,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.degraded is False
        assert result.degraded_raw is False
        assert result.degraded_significant is False
        assert result.degraded == result.degraded_significant
        assert result.n_correct == 2
        assert result.baseline_n_correct == 1
        assert result.n_gained == 1  # p2 newly correct
        assert result.n_lost == 0

    def test_per_type_accuracy(self):
        """Accuracy is tracked per problem_type."""
        problems = [
            _make_problem("p1", "yes", problem_type="math"),
            _make_problem("p2", "yes", problem_type="math"),
            _make_problem("p3", "yes", problem_type="logic"),
        ]

        def gen_fn(prompt, max_tokens):
            return "yes"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=0,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert "math" in result.per_type_accuracy
        assert "logic" in result.per_type_accuracy
        assert result.per_type_accuracy["math"] == 1.0
        assert result.per_type_accuracy["logic"] == 1.0

    def test_generation_failure_counts_as_incorrect(self):
        """Failed generation → incorrect."""
        problems = [_make_problem("p1", "yes")]

        def gen_fn(prompt, max_tokens):
            raise RuntimeError("generation failed")

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=0,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.n_correct == 0
        assert result.n_total == 1

    def test_exact_set_drop_tracks_raw_but_not_significance(self):
        """A one-problem drop is recorded in raw counters even if not significant."""
        problems = [_make_problem(f"p{i}", "yes") for i in range(25)]
        baseline_ids = frozenset(f"p{i}" for i in range(25))

        call_count = [0]

        def gen_fn(prompt, max_tokens):
            call_count[0] += 1
            if call_count[0] == 22:
                return "no"
            return "yes"

        result = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=1,
            baseline_correct_ids=baseline_ids,
            max_tokens=_TEST_MAX_TOKENS,
        )

        assert result.n_lost == 1
        assert result.degraded_raw is True
        assert result.degraded_significant is False
        assert result.degraded is False


class TestOnlineEvalResult:

    def test_to_dict(self):
        result = OnlineEvalResult(
            epoch=1,
            accuracy=0.5,
            n_correct=5,
            n_total=10,
            correct_ids=frozenset(["a", "b"]),
            baseline_n_correct=6,
            baseline_accuracy=0.6,
            n_lost=1,
            n_gained=0,
            degraded=True,
            degraded_raw=True,
            degraded_significant=True,
            alpha=0.2,
            current_ci_lower=0.3,
            current_ci_upper=0.7,
            baseline_ci_lower=0.4,
            baseline_ci_upper=0.8,
        )
        d = result.to_dict()
        assert d["epoch"] == 1
        assert d["accuracy"] == 0.5
        assert d["degraded"] is True
        assert d["degraded_raw"] is True
        assert d["degraded_significant"] is True
        assert d["alpha"] == pytest.approx(0.2)
        assert d["n_lost"] == 1
