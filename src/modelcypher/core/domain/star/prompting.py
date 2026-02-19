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

"""Prompt builders for STaR generation and rationalization."""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.star.problem_generator import StarProblem


@dataclass(frozen=True)
class FewShotExample:
    """Demonstration sample for reasoning prompt construction."""

    problem: str
    reasoning: str
    answer: str


def default_few_shot_examples() -> tuple[FewShotExample, ...]:
    """Deterministic few-shot reasoning demonstrations."""
    return (
        FewShotExample(
            problem=(
                "All lumenaks are brightors. All brightors are chromens. "
                "Daxa is a lumenak. Is Daxa a chromen? Answer Yes or No."
            ),
            reasoning=(
                "From the premises, lumenaks are a subset of brightors and "
                "brightors are a subset of chromens. Since Daxa is a lumenak, "
                "Daxa must also be a chromen."
            ),
            answer="Yes",
        ),
        FewShotExample(
            problem=(
                "color(alpha)=red. color(beta)=color(alpha). "
                "color(gamma)=color(beta). What is color(gamma)?"
            ),
            reasoning=(
                "color(alpha) is red. beta inherits color(alpha), so beta is red. "
                "gamma inherits color(beta), so gamma is also red."
            ),
            answer="red",
        ),
        FewShotExample(
            problem=(
                "Rule: if an item has property P, then it has marker Q. "
                "Object T does not have marker Q. "
                "Can we conclude Object T has property P? Answer Yes or No."
            ),
            reasoning=(
                "The rule P -> Q means Q is required for P. "
                "Object T lacks Q, so Object T cannot satisfy P."
            ),
            answer="No",
        ),
    )


def build_forward_prompt(problem: StarProblem, demonstrations: int = 3) -> str:
    """Build forward-generation prompt with chain-of-thought demonstrations."""
    if demonstrations <= 0:
        raise ValueError(f"demonstrations must be positive, got {demonstrations}")

    demo_pool = default_few_shot_examples()
    selected = [demo_pool[i % len(demo_pool)] for i in range(demonstrations)]

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


def build_rationalization_prompt(problem: StarProblem, demonstrations: int = 2) -> str:
    """Build rationalization prompt conditioned on the known correct answer."""
    if demonstrations <= 0:
        raise ValueError(f"demonstrations must be positive, got {demonstrations}")

    demo_pool = default_few_shot_examples()
    selected = [demo_pool[i % len(demo_pool)] for i in range(demonstrations)]

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
        "Given the correct answer, explain why it is correct.\n"
        f"Problem: {problem.prompt}\n"
        f"Known correct answer: {problem.correct_answer}\n"
        "Reasoning:"
    )
    return "\n\n".join(blocks)


__all__ = [
    "FewShotExample",
    "build_forward_prompt",
    "build_rationalization_prompt",
    "default_few_shot_examples",
]
