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

"""Programmatic STaR problem generator with deterministic verification.

Every generated problem has:
- A programmatically computed correct answer
- A deterministic verifier function (no LLM-as-judge)
- Difficulty metadata derived from structural complexity
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Callable

Verifier = Callable[[str], bool]

_YES_WORDS = {"yes", "true", "correct"}
_NO_WORDS = {"no", "false", "incorrect"}
_ANSWER_PATTERNS = (
    r"(?:final\s+answer|answer|therefore|thus|conclusion)\s*[:\-]\s*(.+)",
    r"(?:the\s+answer\s+is)\s+(.+)",
)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic matching."""
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9.\-+\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_final_answer(response: str) -> str:
    """Extract the most likely final answer segment from a response."""
    stripped = response.strip()
    if not stripped:
        return ""

    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def parse_yes_no(text: str) -> bool | None:
    """Parse yes/no answer from free-form text."""
    normalized = normalize_text(text)
    if not normalized:
        return None
    words = normalized.split()
    for word in words:
        if word in _YES_WORDS:
            return True
        if word in _NO_WORDS:
            return False
    return None


def parse_last_decimal(text: str) -> Decimal | None:
    """Parse the last decimal-like number in text."""
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def make_yes_no_verifier(expected: str) -> Verifier:
    """Build a deterministic yes/no verifier."""
    expected_bool = parse_yes_no(expected)

    def _verify(candidate: str) -> bool:
        observed = parse_yes_no(candidate)
        return observed is not None and observed == expected_bool

    return _verify


def make_exact_token_verifier(expected: str) -> Verifier:
    """Build verifier for exact symbolic/value answers."""
    expected_norm = normalize_text(expected)

    def _verify(candidate: str) -> bool:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            return False
        return expected_norm in candidate_norm

    return _verify


def make_integer_verifier(expected: str) -> Verifier:
    """Build deterministic integer verifier."""
    expected_val = parse_last_decimal(expected)

    def _verify(candidate: str) -> bool:
        observed = parse_last_decimal(candidate)
        return observed is not None and expected_val is not None and observed == expected_val

    return _verify


@dataclass(frozen=True)
class StarProblem:
    """Single STaR generation problem with deterministic verifier."""

    problem_id: str
    prompt: str
    correct_answer: str
    problem_type: str
    difficulty: int
    verification_fn: str
    _verifier: Verifier = field(repr=False, compare=False)

    def verify_response(self, response: str) -> bool:
        """Verify response deterministically.

        Tries answer extraction first, then falls back to whole response.
        """
        extracted = extract_final_answer(response)
        if extracted and self._verifier(extracted):
            return True
        return self._verifier(response)

    def to_problem_record(self) -> dict[str, str | int]:
        """Serialize required problem fields."""
        return {
            "problem_id": self.problem_id,
            "prompt": self.prompt,
            "correct_answer": self.correct_answer,
            "problem_type": self.problem_type,
            "difficulty": self.difficulty,
            "verification_fn": self.verification_fn,
        }

    @classmethod
    def from_problem_record(cls, record: dict[str, object]) -> "StarProblem":
        """Rebuild a StarProblem from ``to_problem_record`` output."""
        required = (
            "problem_id",
            "prompt",
            "correct_answer",
            "problem_type",
            "difficulty",
            "verification_fn",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(
                "Problem record missing required fields: "
                + ", ".join(sorted(missing)),
            )

        verification_fn = str(record["verification_fn"])
        correct_answer = str(record["correct_answer"])
        if verification_fn == "verify_yes_no":
            verifier = make_yes_no_verifier(correct_answer)
        elif verification_fn == "verify_exact_token":
            verifier = make_exact_token_verifier(correct_answer)
        elif verification_fn == "verify_integer":
            verifier = make_integer_verifier(correct_answer)
        else:
            raise ValueError(
                f"Unsupported verification_fn in problem record: {verification_fn}",
            )

        return cls(
            problem_id=str(record["problem_id"]),
            prompt=str(record["prompt"]),
            correct_answer=correct_answer,
            problem_type=str(record["problem_type"]),
            difficulty=int(record["difficulty"]),
            verification_fn=verification_fn,
            _verifier=verifier,
        )


class StarProblemGenerator:
    """Programmatic STaR problem generator.

    Generates structurally diverse problems that remain programmatically
    verifiable across reasoning categories.
    """

    PROBLEM_TYPES: tuple[str, ...] = (
        "syllogistic_chain",
        "compositional_binding",
        "multi_step_arithmetic",
        "contrapositive",
        "set_exception",
    )

    _SYLLABLES: tuple[str, ...] = (
        "ka", "re", "mi", "zo", "lu", "ta", "ni", "vo", "sa", "tri",
        "bel", "dor", "ven", "pha", "cor", "lin", "mur", "xel",
    )
    _ATTRS: tuple[str, ...] = (
        "color", "shape", "material", "signal", "owner", "status",
    )
    _OBJECTS: tuple[str, ...] = (
        "modules", "cells", "tokens", "samples", "units", "packets",
    )
    _EXCEPTION_PROPERTIES: tuple[str, ...] = (
        "can emit blue pulses",
        "can solve lattice puzzles",
        "can navigate by star maps",
        "can compress symbolic traces",
        "can decode axial markers",
    )

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._token_counter = 0

    def generate(self, n_problems: int) -> list[StarProblem]:
        """Generate ``n_problems`` deterministic, unique problems."""
        if n_problems <= 0:
            raise ValueError(f"n_problems must be positive, got {n_problems}")

        problems: list[StarProblem] = []
        seen_prompts: set[str] = set()

        for idx in range(n_problems):
            problem_type = self.PROBLEM_TYPES[idx % len(self.PROBLEM_TYPES)]
            while True:
                problem = self._generate_single(problem_type, idx)
                if problem.prompt not in seen_prompts:
                    seen_prompts.add(problem.prompt)
                    problems.append(problem)
                    break

        return problems

    def _generate_single(self, problem_type: str, idx: int) -> StarProblem:
        if problem_type == "syllogistic_chain":
            return self._generate_syllogistic_chain(idx)
        if problem_type == "compositional_binding":
            return self._generate_compositional_binding(idx)
        if problem_type == "multi_step_arithmetic":
            return self._generate_multi_step_arithmetic(idx)
        if problem_type == "contrapositive":
            return self._generate_contrapositive(idx)
        if problem_type == "set_exception":
            return self._generate_set_exception(idx)
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    def _generate_syllogistic_chain(self, idx: int) -> StarProblem:
        chain_length = self._rng.randint(2, 5)
        concepts = [self._coin_term("class") for _ in range(chain_length + 1)]
        subject = self._coin_term("entity").capitalize()
        use_negation = bool(self._rng.randint(0, 1))

        premises = [f"All {concepts[i]} are {concepts[i + 1]}." for i in range(chain_length)]
        if use_negation:
            premises[-1] = f"No {concepts[-2]} are {concepts[-1]}."
            answer = "No"
        else:
            answer = "Yes"

        prompt = (
            " ".join(premises)
            + f" {subject} is a {concepts[0]}."
            + f" Is {subject} a {concepts[-1]}? Answer Yes or No."
        )

        difficulty = chain_length + (1 if use_negation else 0)
        verifier = make_yes_no_verifier(answer)

        return StarProblem(
            problem_id=f"star_syllogistic_chain_{idx:06d}",
            prompt=prompt,
            correct_answer=answer,
            problem_type="syllogistic_chain",
            difficulty=difficulty,
            verification_fn="verify_yes_no",
            _verifier=verifier,
        )

    def _generate_compositional_binding(self, idx: int) -> StarProblem:
        binding_depth = self._rng.randint(2, 5)
        attr = self._rng.choice(self._ATTRS)
        value = self._coin_term("value")
        entities = [self._coin_term("node") for _ in range(binding_depth + 1)]

        statements = [f"{attr}({entities[0]}) = {value}."]
        for i in range(1, binding_depth + 1):
            statements.append(f"{attr}({entities[i]}) = {attr}({entities[i - 1]}).")

        prompt = (
            " ".join(statements)
            + f" What is {attr}({entities[-1]})?"
        )

        verifier = make_exact_token_verifier(value)

        return StarProblem(
            problem_id=f"star_compositional_binding_{idx:06d}",
            prompt=prompt,
            correct_answer=value,
            problem_type="compositional_binding",
            difficulty=binding_depth,
            verification_fn="verify_exact_token",
            _verifier=verifier,
        )

    def _generate_multi_step_arithmetic(self, idx: int) -> StarProblem:
        crates = self._rng.randint(3, 9)
        per_crate = self._rng.randint(4, 15)
        start_total = crates * per_crate
        used = self._rng.randint(1, max(1, start_total // 3))
        packs = self._rng.randint(2, 6)
        per_pack = self._rng.randint(3, 12)
        final_total = start_total - used + packs * per_pack

        actor = self._coin_term("analyst").capitalize()
        item = self._rng.choice(self._OBJECTS)
        prompt = (
            f"{actor} is auditing {item}. "
            f"There are {crates} crates with {per_crate} {item} each. "
            f"{actor} uses {used} {item}, then receives {packs} new packs "
            f"with {per_pack} {item} in each pack. "
            f"How many {item} are there now?"
        )

        answer = str(final_total)
        verifier = make_integer_verifier(answer)

        return StarProblem(
            problem_id=f"star_multi_step_arithmetic_{idx:06d}",
            prompt=prompt,
            correct_answer=answer,
            problem_type="multi_step_arithmetic",
            difficulty=3,
            verification_fn="verify_integer",
            _verifier=verifier,
        )

    def _generate_contrapositive(self, idx: int) -> StarProblem:
        prop_a = self._coin_term("prop")
        prop_b = self._coin_term("mark")
        subject = self._coin_term("specimen").capitalize()
        prompt = (
            f"Rule: if an item has property {prop_a}, then it has marker {prop_b}. "
            f"{subject} does not have marker {prop_b}. "
            f"Can we conclude that {subject} has property {prop_a}? "
            "Answer Yes or No."
        )

        answer = "No"
        verifier = make_yes_no_verifier(answer)

        return StarProblem(
            problem_id=f"star_contrapositive_{idx:06d}",
            prompt=prompt,
            correct_answer=answer,
            problem_type="contrapositive",
            difficulty=2,
            verification_fn="verify_yes_no",
            _verifier=verifier,
        )

    def _generate_set_exception(self, idx: int) -> StarProblem:
        group = self._coin_term("group") + "s"
        exception_group = self._coin_term("exception") + "s"
        subject = self._coin_term("subject").capitalize()
        property_text = self._rng.choice(self._EXCEPTION_PROPERTIES)
        subject_is_exception = bool(self._rng.randint(0, 1))

        if subject_is_exception:
            subject_fact = f"{subject} is a {exception_group[:-1]}."
            answer = "No"
        else:
            subject_fact = (
                f"{subject} is a {group[:-1]} and {subject} is not a {exception_group[:-1]}."
            )
            answer = "Yes"

        prompt = (
            f"All {group} {property_text}. "
            f"All {exception_group} are {group}. "
            f"No {exception_group} {property_text}. "
            f"{subject_fact} "
            f"Does {subject} {property_text}? Answer Yes or No."
        )

        difficulty = 4 + (1 if subject_is_exception else 0)
        verifier = make_yes_no_verifier(answer)

        return StarProblem(
            problem_id=f"star_set_exception_{idx:06d}",
            prompt=prompt,
            correct_answer=answer,
            problem_type="set_exception",
            difficulty=difficulty,
            verification_fn="verify_yes_no",
            _verifier=verifier,
        )

    def _coin_term(self, prefix: str) -> str:
        """Generate deterministic nonce-like symbolic term."""
        stem = (
            self._rng.choice(self._SYLLABLES)
            + self._rng.choice(self._SYLLABLES)
            + self._rng.choice(self._SYLLABLES)
        )
        token = f"{prefix}_{stem}_{self._token_counter:06d}"
        self._token_counter += 1
        return token.lower()


__all__ = [
    "StarProblem",
    "StarProblemGenerator",
    "extract_final_answer",
    "normalize_text",
    "parse_yes_no",
]
