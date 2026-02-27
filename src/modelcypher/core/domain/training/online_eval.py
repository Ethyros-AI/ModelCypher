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

Replaces proxy metrics with direct inference correctness measurements.
Generates greedy completions, verifies them via deterministic verifiers,
and reports count-level outcomes plus Clopper-Pearson uncertainty bounds.

Greedy decoding remains deterministic for a fixed model state and prompt.
The degradation gate is significance-based: a raw count drop is only treated
as degradation when the current Clopper-Pearson upper bound is strictly below
the baseline lower bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from modelcypher.core.domain.statistics import (
    binomial_degradation_is_significant,
    clopper_pearson_interval,
)
from modelcypher.core.domain.star.problem_generator import StarProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineEvalResult:
    """Result of online correctness evaluation at an epoch boundary."""

    epoch: int
    accuracy: float
    n_correct: int
    n_total: int
    correct_ids: frozenset[str]  # problem IDs answered correctly
    baseline_n_correct: int
    baseline_accuracy: float
    n_lost: int             # problems correct at baseline, wrong now
    n_gained: int           # problems wrong at baseline, correct now
    degraded: bool          # significance-based degradation decision
    per_type_accuracy: dict[str, float] = field(default_factory=dict)
    per_type_correct: dict[str, int] = field(default_factory=dict)
    per_type_total: dict[str, int] = field(default_factory=dict)
    alpha: float | None = None
    current_ci_lower: float | None = None
    current_ci_upper: float | None = None
    baseline_ci_lower: float | None = None
    baseline_ci_upper: float | None = None
    degraded_raw: bool = False
    degraded_significant: bool = False

    def to_dict(self) -> dict:
        payload = {
            "epoch": self.epoch,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "baseline_n_correct": self.baseline_n_correct,
            "baseline_accuracy": self.baseline_accuracy,
            "n_lost": self.n_lost,
            "n_gained": self.n_gained,
            "degraded": self.degraded,
            "degraded_raw": self.degraded_raw,
            "degraded_significant": self.degraded_significant,
            "per_type_accuracy": dict(self.per_type_accuracy),
            "per_type_correct": dict(self.per_type_correct),
            "per_type_total": dict(self.per_type_total),
        }
        if self.alpha is not None:
            payload["alpha"] = self.alpha
            payload["current_ci_lower"] = self.current_ci_lower
            payload["current_ci_upper"] = self.current_ci_upper
            payload["baseline_ci_lower"] = self.baseline_ci_lower
            payload["baseline_ci_upper"] = self.baseline_ci_upper
        return payload


def evaluate_correctness(
    problems: list[StarProblem],
    generate_fn,
    epoch: int,
    baseline_correct_ids: frozenset[str] | None = None,
    *,
    max_tokens: int,
) -> OnlineEvalResult:
    """Evaluate model correctness on a set of problems.

    Greedy decoding is deterministic for fixed model state and prompt.
    Degradation is significance-based using exact Clopper-Pearson intervals.

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
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )

    n_total = len(problems)
    correct_ids: set[str] = set()
    type_correct: dict[str, int] = {}
    type_total: dict[str, int] = {}
    n_demonstrations = len(default_few_shot_examples())

    for problem in problems:
        prompt = build_forward_prompt(problem, demonstrations=n_demonstrations)

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
    degraded_raw = (n_correct < baseline_n) if baseline_correct_ids is not None else False

    alpha = (1.0 / float(n_total)) if n_total > 1 else None
    current_ci_lower = None
    current_ci_upper = None
    baseline_ci_lower = None
    baseline_ci_upper = None
    degraded_significant = False

    if alpha is not None:
        if baseline_correct_ids is not None:
            degraded_significant, current_ci, baseline_ci = (
                binomial_degradation_is_significant(
                    baseline_n_correct=baseline_n,
                    current_n_correct=n_correct,
                    n_total=n_total,
                    alpha=alpha,
                )
            )
            current_ci_lower, current_ci_upper = current_ci
            baseline_ci_lower, baseline_ci_upper = baseline_ci
        else:
            current_ci = clopper_pearson_interval(
                n_correct=n_correct, n_total=n_total, alpha=alpha,
            )
            baseline_ci = clopper_pearson_interval(
                n_correct=baseline_n, n_total=n_total, alpha=alpha,
            )
            current_ci_lower, current_ci_upper = current_ci
            baseline_ci_lower, baseline_ci_upper = baseline_ci
    elif baseline_correct_ids is not None:
        # n_total <= 1: CP alpha=1/n is undefined; fall back to exact raw count.
        degraded_significant = degraded_raw

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
        degraded=degraded_significant,
        per_type_accuracy=per_type_acc,
        per_type_correct=type_correct,
        per_type_total=type_total,
        alpha=alpha,
        current_ci_lower=current_ci_lower,
        current_ci_upper=current_ci_upper,
        baseline_ci_lower=baseline_ci_lower,
        baseline_ci_upper=baseline_ci_upper,
        degraded_raw=degraded_raw,
        degraded_significant=degraded_significant,
    )

    logger.info(
        "Online eval epoch %d: %d/%d correct (%.1f%%), "
        "baseline=%d/%d, lost=%d, gained=%d, degraded_raw=%s, degraded_significant=%s",
        epoch, n_correct, n_total, accuracy * 100,
        baseline_n, n_total, len(lost), len(gained),
        result.degraded_raw, result.degraded_significant,
    )
    if result.alpha is not None:
        logger.info(
            "  CI(alpha=%.6f): current=[%.4f, %.4f], baseline=[%.4f, %.4f]",
            result.alpha,
            result.current_ci_lower or 0.0,
            result.current_ci_upper or 0.0,
            result.baseline_ci_lower or 0.0,
            result.baseline_ci_upper or 0.0,
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
    *,
    n_problems: int,
    seed: int,
) -> list[StarProblem]:
    """Create a held-out evaluation problem set.

    Uses StarProblemGenerator with a fixed seed for reproducibility.
    Both n_problems and seed are required — n_problems is a compute budget
    choice (not derivable), and seed should be derived from the training
    seed by the caller (e.g., ``seed=training_seed + 1``).
    """
    from modelcypher.core.domain.star.problem_generator import StarProblemGenerator

    generator = StarProblemGenerator(seed=seed)
    return generator.generate(n_problems)


def compute_answer_margin(
    problems: list[StarProblem],
    collect_logits_fn,
    backend,
) -> dict[str, float]:
    """Compute top-1 minus top-2 logit gap at last prompt token per problem.

    The margin measures decision-boundary confidence: positive values mean
    the model's top prediction is well-separated from alternatives.  Near-zero
    margins indicate fragile predictions that training perturbation can flip.

    Parameters
    ----------
    problems : list[StarProblem]
        Evaluation problems.
    collect_logits_fn : callable
        ``collect_logits_fn(prompt: str) -> Array[vocab_size]``
        Returns raw logits at the last token position.
    backend : Backend
        Backend protocol for array operations.

    Returns
    -------
    dict mapping problem_id -> margin (logits[0] - logits[1]).
    """
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )

    n_demonstrations = len(default_few_shot_examples())
    margins: dict[str, float] = {}

    for problem in problems:
        prompt = build_forward_prompt(problem, demonstrations=n_demonstrations)
        try:
            logits = collect_logits_fn(prompt)
            # Sort descending to get top-1 and top-2
            sorted_logits = backend.sort(logits)
            backend.eval(sorted_logits)
            # backend.sort returns ascending; top-1 is last, top-2 is second-to-last
            n_vocab = int(sorted_logits.shape[0])
            if n_vocab >= 2:
                top1 = float(backend.to_scalar(sorted_logits[n_vocab - 1]))
                top2 = float(backend.to_scalar(sorted_logits[n_vocab - 2]))
                margins[problem.problem_id] = top1 - top2
            else:
                margins[problem.problem_id] = 0.0
        except Exception:
            logger.debug(
                "Margin computation failed for problem %s",
                problem.problem_id,
                exc_info=True,
            )
            margins[problem.problem_id] = 0.0

    return margins


__all__ = [
    "OnlineEvalResult",
    "compute_answer_margin",
    "create_eval_problem_set",
    "evaluate_correctness",
]
