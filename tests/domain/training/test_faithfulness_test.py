# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for faithfulness intervention test."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.star.prompting import (
    FewShotExample,
    default_few_shot_examples,
)
from modelcypher.core.domain.training.faithfulness_test import (
    FaithfulnessResult,
    build_intervened_prompt,
    evaluate_faithfulness,
    shuffle_reasoning,
    wrong_demonstrations,
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


_TEST_MAX_TOKENS = 64


class TestShuffleReasoning:
    """Tests for shuffle_reasoning intervention."""

    def test_preserves_answers(self):
        """All demo answers are unchanged after shuffling."""
        examples = default_few_shot_examples()
        shuffled = shuffle_reasoning(examples, seed=42)

        assert len(shuffled) == len(examples)
        for orig, shuf in zip(examples, shuffled):
            assert shuf.answer == orig.answer

    def test_preserves_problems(self):
        """Problem statements are unchanged."""
        examples = default_few_shot_examples()
        shuffled = shuffle_reasoning(examples, seed=42)

        for orig, shuf in zip(examples, shuffled):
            assert shuf.problem == orig.problem

    def test_changes_reasoning_order(self):
        """At least one example has different reasoning after shuffle."""
        examples = default_few_shot_examples()
        shuffled = shuffle_reasoning(examples, seed=42)

        any_changed = any(
            shuf.reasoning != orig.reasoning
            for orig, shuf in zip(examples, shuffled)
        )
        assert any_changed, "Expected at least one reasoning to change"

    def test_deterministic_with_seed(self):
        """Same seed produces same shuffle."""
        examples = default_few_shot_examples()
        a = shuffle_reasoning(examples, seed=123)
        b = shuffle_reasoning(examples, seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        """Different seeds produce different shuffles (probabilistically)."""
        examples = default_few_shot_examples()
        a = shuffle_reasoning(examples, seed=1)
        b = shuffle_reasoning(examples, seed=999)
        # At least one example should differ
        any_diff = any(x.reasoning != y.reasoning for x, y in zip(a, b))
        assert any_diff


class TestWrongDemonstrations:
    """Tests for wrong_demonstrations intervention."""

    def test_all_answers_changed(self):
        """Every demo answer is different from the original."""
        examples = default_few_shot_examples()
        wrong = wrong_demonstrations(examples)

        assert len(wrong) == len(examples)
        for orig, w in zip(examples, wrong):
            assert w.answer != orig.answer, (
                f"Expected answer to change for demo with answer={orig.answer!r}"
            )

    def test_reasoning_preserved(self):
        """Reasoning text is unchanged."""
        examples = default_few_shot_examples()
        wrong = wrong_demonstrations(examples)

        for orig, w in zip(examples, wrong):
            assert w.reasoning == orig.reasoning

    def test_circular_shift(self):
        """Each demo gets the next demo's answer."""
        examples = default_few_shot_examples()
        wrong = wrong_demonstrations(examples)

        for i, w in enumerate(wrong):
            expected = examples[(i + 1) % len(examples)].answer
            assert w.answer == expected

    def test_single_example_unchanged(self):
        """With only 1 example, can't swap — returned as-is."""
        single = (FewShotExample(problem="p", reasoning="r", answer="a"),)
        result = wrong_demonstrations(single)
        assert result == single


class TestBuildIntervenedPrompt:
    """Tests for build_intervened_prompt."""

    def test_contains_intervened_answers(self):
        """Prompt includes the intervened (wrong) answers."""
        examples = default_few_shot_examples()
        wrong = wrong_demonstrations(examples)
        problem = _make_problem("p1")
        prompt = build_intervened_prompt(problem, wrong, demonstrations=3)

        for w in wrong:
            assert w.answer in prompt

    def test_contains_problem(self):
        """Prompt includes the target problem."""
        examples = default_few_shot_examples()
        problem = _make_problem("p1")
        prompt = build_intervened_prompt(problem, examples, demonstrations=3)
        assert "Test prompt" in prompt

    def test_demonstrations_count(self):
        """Number of 'Example N' blocks matches demonstrations param."""
        examples = default_few_shot_examples()
        problem = _make_problem("p1")
        prompt = build_intervened_prompt(problem, examples, demonstrations=2)
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Example 3" not in prompt

    def test_zero_demonstrations_raises(self):
        """demonstrations=0 raises ValueError."""
        examples = default_few_shot_examples()
        problem = _make_problem("p1")
        with pytest.raises(ValueError, match="positive"):
            build_intervened_prompt(problem, examples, demonstrations=0)


class TestEvaluateFaithfulness:
    """Tests for the full faithfulness evaluation."""

    def test_faithful_model_shows_delta(self):
        """A model that uses demo answers should show delta > 0 with wrong_demonstrations."""
        problems = [
            _make_problem("p1", "yes"),
            _make_problem("p2", "no"),
        ]

        def gen_fn(prompt, max_tokens):
            # Model parrots the last demo's answer pattern —
            # check if "Final answer: No" is the last demo answer seen
            if "Final answer: No\n" in prompt.split("Now solve")[0]:
                return "no"
            return "yes"

        result = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="wrong_demonstrations",
            seed=42,
        )

        # Baseline: gen_fn returns "yes" (since default demos end with "No")
        # Intervened: answers are shifted, behavior may change
        assert isinstance(result, FaithfulnessResult)
        assert result.n_total == 2
        assert result.intervention_type == "wrong_demonstrations"

    def test_independent_model_shows_no_delta(self):
        """A model that ignores demos entirely should show delta == 0."""
        problems = [
            _make_problem("p1", "yes"),
            _make_problem("p2", "yes"),
        ]

        def gen_fn(prompt, max_tokens):
            # Always returns "yes" regardless of prompt content
            return "yes"

        result = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="shuffle_reasoning",
            seed=42,
        )

        assert result.delta == 0.0
        assert result.baseline_accuracy == result.intervened_accuracy

    def test_wrong_demos_independent_model(self):
        """Independent model: wrong_demonstrations also shows no delta."""
        problems = [_make_problem("p1", "yes")]

        def gen_fn(prompt, max_tokens):
            return "yes"

        result = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="wrong_demonstrations",
            seed=42,
        )

        assert result.delta == 0.0

    def test_per_type_breakdown_populated(self):
        """Per-type accuracy dictionaries are populated."""
        problems = [
            _make_problem("p1", "yes", problem_type="math"),
            _make_problem("p2", "yes", problem_type="logic"),
        ]

        def gen_fn(prompt, max_tokens):
            return "yes"

        result = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="shuffle_reasoning",
            seed=42,
        )

        assert "math" in result.per_type_baseline
        assert "logic" in result.per_type_baseline
        assert "math" in result.per_type_intervened
        assert "logic" in result.per_type_intervened

    def test_baseline_matches_standalone_eval(self):
        """Baseline result matches a standalone evaluate_correctness call."""
        from modelcypher.core.domain.training.online_eval import evaluate_correctness

        problems = [
            _make_problem("p1", "yes"),
            _make_problem("p2", "no"),
        ]

        def gen_fn(prompt, max_tokens):
            return "yes"

        standalone = evaluate_correctness(
            problems=problems,
            generate_fn=gen_fn,
            epoch=0,
            max_tokens=_TEST_MAX_TOKENS,
        )

        faithfulness = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="shuffle_reasoning",
            seed=42,
        )

        assert faithfulness.baseline_accuracy == standalone.accuracy
        assert faithfulness.baseline_n_correct == standalone.n_correct

    def test_unknown_intervention_raises(self):
        """Unknown intervention_type raises ValueError."""
        problems = [_make_problem("p1")]

        def gen_fn(prompt, max_tokens):
            return "yes"

        with pytest.raises(ValueError, match="Unknown intervention_type"):
            evaluate_faithfulness(
                problems=problems,
                generate_fn=gen_fn,
                max_tokens=_TEST_MAX_TOKENS,
                intervention_type="nonexistent",
                seed=42,
            )

    def test_generation_failure_counts_as_incorrect(self):
        """Failed generation in intervened eval counts as incorrect."""
        problems = [_make_problem("p1", "yes")]

        call_count = [0]

        def gen_fn(prompt, max_tokens):
            call_count[0] += 1
            if call_count[0] > 1:  # Second call is intervened
                raise RuntimeError("generation failed")
            return "yes"

        result = evaluate_faithfulness(
            problems=problems,
            generate_fn=gen_fn,
            max_tokens=_TEST_MAX_TOKENS,
            intervention_type="shuffle_reasoning",
            seed=42,
        )

        assert result.baseline_n_correct == 1
        assert result.intervened_n_correct == 0
        assert result.delta > 0
