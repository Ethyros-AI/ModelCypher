# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for REINFORCE outcome objective (Layer 3)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.outcome_objective import (
    OutcomeCollectionResult,
    OutcomeCompletion,
    collect_outcomes,
    compute_group_advantages,
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


# =============================================================================
# compute_group_advantages tests
# =============================================================================


class TestComputeGroupAdvantages:
    """Test Williams (1992) advantage baseline: A_i = r_i - mean(r)."""

    def test_empty_rewards(self):
        assert compute_group_advantages([]) == []

    def test_all_correct(self):
        """All correct → all advantages zero (no gradient)."""
        rewards = [1.0, 1.0, 1.0]
        advantages = compute_group_advantages(rewards)
        assert all(a == 0.0 for a in advantages)

    def test_all_incorrect(self):
        """All incorrect → all advantages zero (no gradient)."""
        rewards = [0.0, 0.0, 0.0]
        advantages = compute_group_advantages(rewards)
        assert all(a == 0.0 for a in advantages)

    def test_mixed_outcomes(self):
        """Mixed outcomes → nonzero advantages."""
        rewards = [1.0, 0.0, 1.0]
        advantages = compute_group_advantages(rewards)
        # mean = 2/3 ≈ 0.6667
        # A_0 = 1 - 2/3 = 1/3
        # A_1 = 0 - 2/3 = -2/3
        # A_2 = 1 - 2/3 = 1/3
        assert len(advantages) == 3
        assert advantages[0] > 0  # correct gets positive advantage
        assert advantages[1] < 0  # incorrect gets negative advantage
        assert advantages[2] > 0  # correct gets positive advantage

    def test_advantages_sum_to_zero(self):
        """Group advantages always sum to zero (unbiased estimator)."""
        rewards = [1.0, 0.0, 1.0, 0.0]
        advantages = compute_group_advantages(rewards)
        assert abs(sum(advantages)) < 1e-10

    def test_single_reward(self):
        """Single reward → zero advantage."""
        rewards = [1.0]
        advantages = compute_group_advantages(rewards)
        assert advantages == [0.0]

    def test_binary_rewards_symmetric(self):
        """50/50 split → symmetric ±0.5 advantages."""
        rewards = [1.0, 0.0]
        advantages = compute_group_advantages(rewards)
        assert abs(advantages[0] - 0.5) < 1e-10
        assert abs(advantages[1] - (-0.5)) < 1e-10


# =============================================================================
# collect_outcomes tests
# =============================================================================


class TestCollectOutcomes:
    """Test outcome collection with mocked generation and verification."""

    def test_all_correct_zero_signal(self):
        """When model gets all variants correct, signal density = 0."""
        problems = [_make_problem("p1", "yes")]

        def gen_fn(prompt, max_tokens):
            return "The answer is yes."

        def tok_fn(text):
            return list(range(len(text.split())))

        result = collect_outcomes(problems, gen_fn, tok_fn, n_variants=3, max_tokens=64)

        assert result.n_problems == 1
        assert result.n_correct == 3
        assert result.n_incorrect == 0
        assert result.n_mixed_problems == 0
        assert result.signal_density == 0.0
        # All advantages should be zero
        assert all(c.advantage == 0.0 for c in result.completions)

    def test_mixed_outcomes_nonzero_signal(self):
        """When some variants are correct and some not, signal > 0."""
        problems = [_make_problem("p1", "yes")]
        call_count = [0]

        def gen_fn(prompt, max_tokens):
            call_count[0] += 1
            # First variant correct, second and third incorrect
            if call_count[0] % 3 == 1:
                return "yes"
            return "no"

        def tok_fn(text):
            return list(range(len(text.split())))

        result = collect_outcomes(problems, gen_fn, tok_fn, n_variants=3, max_tokens=64)

        assert result.n_problems == 1
        assert result.n_correct == 1
        assert result.n_incorrect == 2
        assert result.n_mixed_problems == 1
        assert result.signal_density == 1.0
        # Correct completion should have positive advantage
        correct_comps = [c for c in result.completions if c.correct]
        incorrect_comps = [c for c in result.completions if not c.correct]
        assert all(c.advantage > 0 for c in correct_comps)
        assert all(c.advantage < 0 for c in incorrect_comps)

    def test_generation_failure_counts_as_incorrect(self):
        """Failed generation → empty response → incorrect."""
        problems = [_make_problem("p1", "yes")]

        def gen_fn(prompt, max_tokens):
            raise RuntimeError("generation failed")

        def tok_fn(text):
            return [0]

        result = collect_outcomes(problems, gen_fn, tok_fn, n_variants=2, max_tokens=64)

        assert result.n_correct == 0
        assert result.n_incorrect == 2

    def test_n_variants_controls_completions(self):
        """n_variants controls how many completions per problem."""
        problems = [_make_problem("p1", "yes")]

        def gen_fn(prompt, max_tokens):
            return "yes"

        def tok_fn(text):
            return [0, 1, 2]

        result = collect_outcomes(problems, gen_fn, tok_fn, n_variants=2, max_tokens=64)
        assert len(result.completions) == 2

        result3 = collect_outcomes(problems, gen_fn, tok_fn, n_variants=3, max_tokens=64)
        assert len(result3.completions) == 3


# =============================================================================
# OutcomeCollectionResult tests
# =============================================================================


class TestOutcomeCollectionResult:

    def test_signal_density_zero_problems(self):
        result = OutcomeCollectionResult(
            completions=[],
            n_problems=0,
            n_correct=0,
            n_incorrect=0,
            n_mixed_problems=0,
            mean_advantage=0.0,
        )
        assert result.signal_density == 0.0

    def test_signal_density_calculation(self):
        result = OutcomeCollectionResult(
            completions=[],
            n_problems=10,
            n_correct=5,
            n_incorrect=5,
            n_mixed_problems=3,
            mean_advantage=0.0,
        )
        assert abs(result.signal_density - 0.3) < 1e-10

    def test_signal_density_gate_below_threshold(self):
        """Signal density below gate threshold should gate out REINFORCE.

        This tests the gating logic used in mlx_training_adapter.py:
        when signal_density < gate, active_completions is emptied.
        """
        result = OutcomeCollectionResult(
            completions=[
                OutcomeCompletion("p1", 0, [1, 2, 3], True, advantage=0.0),
                OutcomeCompletion("p1", 1, [1, 2, 3], True, advantage=0.0),
                OutcomeCompletion("p1", 2, [1, 2, 3], False, advantage=0.0),
            ],
            n_problems=10,
            n_correct=20,
            n_incorrect=10,
            n_mixed_problems=2,  # 2/10 = 0.2 signal density
            mean_advantage=0.0,
        )
        gate = 0.5
        # Gate logic: if signal_density < gate, skip REINFORCE
        assert result.signal_density < gate
        assert result.signal_density == pytest.approx(0.2)

    def test_signal_density_gate_above_threshold(self):
        """Signal density above gate should allow REINFORCE."""
        result = OutcomeCollectionResult(
            completions=[],
            n_problems=10,
            n_correct=5,
            n_incorrect=5,
            n_mixed_problems=7,  # 7/10 = 0.7 signal density
            mean_advantage=0.0,
        )
        gate = 0.5
        assert result.signal_density >= gate
        assert result.signal_density == pytest.approx(0.7)
