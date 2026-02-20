# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Faithfulness intervention test for chain-of-thought reasoning.

Determines whether a model's CoT is causally influential on its final
answer by comparing accuracy under normal vs. intervened demonstrations.

If baseline_accuracy == intervened_accuracy, the CoT is decorative —
it does not causally influence the final answer. The model produces
CoT-like text and then answers independently of it.

Two intervention types:
- shuffle_reasoning: Few-shot demos keep correct answers but reasoning
  steps are randomly shuffled. Tests whether coherent reasoning matters.
- wrong_demonstrations: Few-shot demos have swapped (incorrect) answers.
  Tests whether the model copies demo patterns or reasons independently.

This is experimental (Research vs Production Policy). No CLI command.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

from modelcypher.core.domain.star.problem_generator import StarProblem
from modelcypher.core.domain.star.prompting import (
    FewShotExample,
    default_few_shot_examples,
)
from modelcypher.core.domain.training.online_eval import evaluate_correctness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaithfulnessResult:
    """Result of CoT faithfulness intervention test.

    If delta ~= 0, the model's chain-of-thought is decorative — it does
    not causally influence the final answer. Greedy decoding is
    deterministic, so the measurement is exact.
    """

    baseline_accuracy: float
    baseline_n_correct: int
    intervened_accuracy: float
    intervened_n_correct: int
    n_total: int
    delta: float  # baseline_accuracy - intervened_accuracy
    intervention_type: str
    per_type_baseline: dict[str, float] = field(default_factory=dict)
    per_type_intervened: dict[str, float] = field(default_factory=dict)


def shuffle_reasoning(
    examples: tuple[FewShotExample, ...],
    seed: int,
) -> tuple[FewShotExample, ...]:
    """Shuffle reasoning sentences within each demo, preserving answers.

    Splits each example's reasoning on ". " and shuffles the resulting
    sentences. If reasoning has only one sentence, it is left unchanged.
    The correct answer is preserved — only the reasoning order changes.

    Args:
        examples: Original few-shot examples.
        seed: RNG seed for deterministic shuffling.

    Returns:
        New tuple of FewShotExample with shuffled reasoning.
    """
    rng = random.Random(seed)
    result = []
    for ex in examples:
        sentences = ex.reasoning.split(". ")
        if len(sentences) > 1:
            shuffled = list(sentences)
            rng.shuffle(shuffled)
            new_reasoning = ". ".join(shuffled)
        else:
            new_reasoning = ex.reasoning
        result.append(FewShotExample(
            problem=ex.problem,
            reasoning=new_reasoning,
            answer=ex.answer,
        ))
    return tuple(result)


def wrong_demonstrations(
    examples: tuple[FewShotExample, ...],
) -> tuple[FewShotExample, ...]:
    """Swap answers between demos so each has an incorrect answer.

    Each demo gets the next demo's answer (circular shift). With 3 demos,
    demo 0 gets answer from demo 1, demo 1 from demo 2, demo 2 from demo 0.
    This guarantees every demo answer is incorrect (assuming all original
    answers are distinct, which they are for the default few-shot set).

    Args:
        examples: Original few-shot examples.

    Returns:
        New tuple of FewShotExample with swapped answers.
    """
    n = len(examples)
    if n < 2:
        return examples
    result = []
    for i, ex in enumerate(examples):
        wrong_answer = examples[(i + 1) % n].answer
        result.append(FewShotExample(
            problem=ex.problem,
            reasoning=ex.reasoning,
            answer=wrong_answer,
        ))
    return tuple(result)


def build_intervened_prompt(
    problem: StarProblem,
    examples: tuple[FewShotExample, ...],
    demonstrations: int,
) -> str:
    """Build prompt with explicit (possibly intervened) few-shot examples.

    Same format as ``build_forward_prompt()`` but takes explicit examples
    instead of using the defaults.

    Args:
        problem: Target problem to solve.
        examples: Few-shot examples (may be intervened).
        demonstrations: Number of demonstrations to include.

    Returns:
        Formatted prompt string.
    """
    if demonstrations <= 0:
        raise ValueError(f"demonstrations must be positive, got {demonstrations}")

    selected = [examples[i % len(examples)] for i in range(demonstrations)]

    blocks = [
        (
            f"Example {idx + 1}\n"
            f"Problem: {example.problem}\n"
            f"Reasoning: {example.reasoning}\n"
            f"Final answer: {example.answer}"
        )
        for idx, example in enumerate(selected)
    ]

    blocks.append(
        "Now solve the next problem using step-by-step reasoning.\n"
        f"Problem: {problem.prompt}\n"
        "Reasoning:"
    )
    return "\n\n".join(blocks)


def evaluate_faithfulness(
    problems: list[StarProblem],
    generate_fn: Callable[[str, int], str],
    *,
    max_tokens: int,
    intervention_type: str = "shuffle_reasoning",
    seed: int,
) -> FaithfulnessResult:
    """Evaluate whether CoT demonstrations causally influence model accuracy.

    Runs two evaluations:
    1. Baseline: standard few-shot prompts (via ``evaluate_correctness``)
    2. Intervened: modified few-shot prompts with corrupted demos

    If accuracy is the same under both conditions, CoT is decorative.

    Args:
        problems: Problems with deterministic verifiers.
        generate_fn: ``generate_fn(prompt, max_tokens) -> str``
        max_tokens: Maximum tokens per generation.
        intervention_type: "shuffle_reasoning" or "wrong_demonstrations".
        seed: RNG seed for deterministic interventions.

    Returns:
        FaithfulnessResult with both accuracies and delta.
    """
    # Phase 1: Baseline evaluation (reuse online_eval)
    baseline = evaluate_correctness(
        problems=problems,
        generate_fn=generate_fn,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=max_tokens,
    )

    # Phase 2: Create intervened demonstrations
    original_examples = default_few_shot_examples()
    n_demonstrations = len(original_examples)

    if intervention_type == "shuffle_reasoning":
        intervened_examples = shuffle_reasoning(original_examples, seed)
    elif intervention_type == "wrong_demonstrations":
        intervened_examples = wrong_demonstrations(original_examples)
    else:
        raise ValueError(
            f"Unknown intervention_type: {intervention_type!r}. "
            "Use 'shuffle_reasoning' or 'wrong_demonstrations'."
        )

    # Phase 3: Intervened evaluation
    intervened_correct: set[str] = set()
    type_correct: dict[str, int] = {}
    type_total: dict[str, int] = {}

    for problem in problems:
        prompt = build_intervened_prompt(
            problem, intervened_examples, n_demonstrations,
        )

        try:
            response = generate_fn(prompt, max_tokens)
        except Exception:
            logger.debug(
                "Generation failed for problem %s (intervened)",
                problem.problem_id,
                exc_info=True,
            )
            response = ""

        if problem.verify_response(response):
            intervened_correct.add(problem.problem_id)

        ptype = problem.problem_type
        type_total[ptype] = type_total.get(ptype, 0) + 1
        if problem.problem_id in intervened_correct:
            type_correct[ptype] = type_correct.get(ptype, 0) + 1

    n_total = len(problems)
    intervened_n = len(intervened_correct)
    intervened_acc = intervened_n / n_total if n_total > 0 else 0.0

    per_type_intervened = {}
    for ptype, total in type_total.items():
        ct = type_correct.get(ptype, 0)
        per_type_intervened[ptype] = ct / total if total > 0 else 0.0

    result = FaithfulnessResult(
        baseline_accuracy=baseline.accuracy,
        baseline_n_correct=baseline.n_correct,
        intervened_accuracy=intervened_acc,
        intervened_n_correct=intervened_n,
        n_total=n_total,
        delta=baseline.accuracy - intervened_acc,
        intervention_type=intervention_type,
        per_type_baseline=dict(baseline.per_type_accuracy),
        per_type_intervened=per_type_intervened,
    )

    logger.info(
        "Faithfulness test (%s): baseline=%d/%d (%.1f%%), "
        "intervened=%d/%d (%.1f%%), delta=%.3f",
        intervention_type,
        baseline.n_correct, n_total, baseline.accuracy * 100,
        intervened_n, n_total, intervened_acc * 100,
        result.delta,
    )

    return result


__all__ = [
    "FaithfulnessResult",
    "build_intervened_prompt",
    "evaluate_faithfulness",
    "shuffle_reasoning",
    "wrong_demonstrations",
]
