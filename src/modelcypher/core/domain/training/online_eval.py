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

"""Online correctness evaluation during training.

Replaces the lying proxies (PPL, CKA, budget) with actual inference
correctness measurement. Generates greedy completions, verifies them
via deterministic verifiers, and reports exact counts.

Greedy decoding is deterministic. Same model + same prompt = same output.
There is no sampling variability. If accuracy drops by even one problem,
that is a real, measured change — not noise. No statistical tests needed.

This is Layer 1 of the outcome-based training objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from modelcypher.core.domain.star.problem_generator import StarProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineEvalResult:
    """Result of online correctness evaluation at an epoch boundary.

    Greedy decoding is deterministic: same model state + same prompt =
    same output. The accuracy measurement is exact, not a sample estimate.
    Degradation is measured by direct count comparison to baseline.
    """

    epoch: int
    accuracy: float
    n_correct: int
    n_total: int
    correct_ids: frozenset[str]  # problem IDs answered correctly
    baseline_n_correct: int
    baseline_accuracy: float
    n_lost: int             # problems correct at baseline, wrong now
    n_gained: int           # problems wrong at baseline, correct now
    degraded: bool          # n_correct < baseline_n_correct
    per_type_accuracy: dict[str, float] = field(default_factory=dict)
    per_type_correct: dict[str, int] = field(default_factory=dict)
    per_type_total: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "baseline_n_correct": self.baseline_n_correct,
            "baseline_accuracy": self.baseline_accuracy,
            "n_lost": self.n_lost,
            "n_gained": self.n_gained,
            "degraded": self.degraded,
            "per_type_accuracy": dict(self.per_type_accuracy),
            "per_type_correct": dict(self.per_type_correct),
            "per_type_total": dict(self.per_type_total),
        }


def evaluate_correctness(
    problems: list[StarProblem],
    generate_fn,
    epoch: int,
    baseline_correct_ids: frozenset[str] | None = None,
    *,
    max_tokens: int,
) -> OnlineEvalResult:
    """Evaluate model correctness on a set of problems.

    Greedy decoding is deterministic — the accuracy measurement is exact.
    Degradation = ``n_correct < len(baseline_correct_ids)``.

    For baseline measurement, pass ``baseline_correct_ids=None``.
    The returned ``correct_ids`` can then be used as the baseline for
    subsequent calls.

    Parameters
    ----------
    problems : list[StarProblem]
        Problems with deterministic verifiers.
    generate_fn : callable
        ``generate_fn(prompt: str, max_tokens: int) -> str``
        Generates a greedy completion.
    epoch : int
        Current epoch number.
    baseline_correct_ids : frozenset[str] or None
        Problem IDs the model answered correctly at baseline.
        None for the baseline measurement itself.
    max_tokens : int
        Maximum tokens to generate per problem. Should match the training
        ``seq_length`` — the caller derives this, not this function.
    """
    from modelcypher.core.domain.star.prompting import build_forward_prompt

    n_total = len(problems)
    correct_ids: set[str] = set()
    type_correct: dict[str, int] = {}
    type_total: dict[str, int] = {}

    for problem in problems:
        prompt = build_forward_prompt(problem, demonstrations=3)

        try:
            response = generate_fn(prompt, max_tokens)
        except Exception:
            logger.debug(
                "Generation failed for problem %s", problem.problem_id,
                exc_info=True,
            )
            response = ""

        if problem.verify_response(response):
            correct_ids.add(problem.problem_id)

        ptype = problem.problem_type
        type_total[ptype] = type_total.get(ptype, 0) + 1
        if problem.problem_id in correct_ids:
            type_correct[ptype] = type_correct.get(ptype, 0) + 1

    n_correct = len(correct_ids)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # Baseline comparison
    if baseline_correct_ids is not None:
        baseline_n = len(baseline_correct_ids)
        lost = baseline_correct_ids - correct_ids
        gained = correct_ids - baseline_correct_ids
    else:
        baseline_n = n_correct  # first measurement IS the baseline
        lost = frozenset()
        gained = frozenset()

    baseline_acc = baseline_n / n_total if n_total > 0 else 0.0

    # Per-type accuracy
    per_type_acc = {}
    for ptype, total in type_total.items():
        ct = type_correct.get(ptype, 0)
        per_type_acc[ptype] = ct / total if total > 0 else 0.0

    result = OnlineEvalResult(
        epoch=epoch,
        accuracy=accuracy,
        n_correct=n_correct,
        n_total=n_total,
        correct_ids=frozenset(correct_ids),
        baseline_n_correct=baseline_n,
        baseline_accuracy=baseline_acc,
        n_lost=len(lost),
        n_gained=len(gained),
        degraded=(n_correct < baseline_n) if baseline_correct_ids is not None else False,
        per_type_accuracy=per_type_acc,
        per_type_correct=type_correct,
        per_type_total=type_total,
    )

    logger.info(
        "Online eval epoch %d: %d/%d correct (%.1f%%), "
        "baseline=%d/%d, lost=%d, gained=%d, degraded=%s",
        epoch, n_correct, n_total, accuracy * 100,
        baseline_n, n_total, len(lost), len(gained), result.degraded,
    )
    for ptype in sorted(per_type_acc):
        logger.info(
            "  %s: %d/%d (%.1f%%)",
            ptype,
            type_correct.get(ptype, 0),
            type_total[ptype],
            per_type_acc[ptype] * 100,
        )

    return result


def create_eval_problem_set(
    n_problems: int = 100,
    seed: int = 42,
) -> list[StarProblem]:
    """Create a held-out evaluation problem set.

    Uses StarProblemGenerator with a fixed seed for reproducibility.
    The seed is deliberately different from training problem seeds
    to ensure no overlap.
    """
    from modelcypher.core.domain.star.problem_generator import StarProblemGenerator

    generator = StarProblemGenerator(seed=seed)
    return generator.generate(n_problems)


__all__ = [
    "OnlineEvalResult",
    "create_eval_problem_set",
    "evaluate_correctness",
]
