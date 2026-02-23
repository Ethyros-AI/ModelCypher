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

"""REINFORCE outcome objective for training.

Layer 3 of the outcome-based training objective. Rewards correct final
answers instead of matching trace format token-by-token.

Key difference from CE:
- CE: L = -log P(y_target | x) — increases logit magnitude of target token
- REINFORCE: L = -A * log π(y | x) — increases correct, DECREASES incorrect

Advantage baseline: A_i = r_i - mean(r_group)
This is the optimal constant baseline for variance reduction of the
REINFORCE gradient estimator (Williams, 1992). No std normalization —
gradient magnitude is handled by calibrate_geometric_weights().

NB-LoRA replaces KL regularization. Standard GRPO uses β * KL(π || π_ref)
to prevent drift. NB-LoRA's spectral budget (||BA||₂/σ_k < 1) bounds drift
by construction. No reference model needed. No β to tune.

References:
- Williams (1992), "Simple statistical gradient-following algorithms
  for connectionist reinforcement learning", Machine Learning 8(3-4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from modelcypher.core.domain.star.problem_generator import StarProblem

logger = logging.getLogger(__name__)


@dataclass
class OutcomeCompletion:
    """A single completion with its verification result and advantage."""

    problem_id: str
    prompt_variant: int
    tokens: list[int]
    correct: bool
    advantage: float = 0.0
    response_start: int = 0  # Token index where response begins


@dataclass
class OutcomeCollectionResult:
    """Result of Phase A: outcome collection across problems."""

    completions: list[OutcomeCompletion]
    n_problems: int
    n_correct: int
    n_incorrect: int
    n_mixed_problems: int
    mean_advantage: float

    @property
    def signal_density(self) -> float:
        """Fraction of problems with mixed outcomes (nonzero gradient)."""
        return self.n_mixed_problems / self.n_problems if self.n_problems > 0 else 0.0


def compute_group_advantages(
    rewards: list[float],
) -> list[float]:
    """Compute per-completion advantages within a problem group.

    Advantage = r_i - mean(r)

    This is the optimal constant baseline for variance reduction of the
    REINFORCE gradient estimator (Williams, 1992, Theorem 1). The mean
    baseline minimizes Var(A_i * ∇log π) among all constant baselines.

    Gradient magnitude is handled by MASS step sizing which measures
    actual gradient norms and bounds displacement per step.

    When all rewards are identical (all correct or all incorrect),
    advantages are zero — these completions contribute zero gradient.
    REINFORCE automatically focuses on the learning frontier.
    """
    if not rewards:
        return []

    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


def collect_outcomes(
    problems: list[StarProblem],
    generate_fn,
    tokenize_fn,
    *,
    n_variants: int,
    max_tokens: int,
) -> OutcomeCollectionResult:
    """Phase A: Generate completions and compute advantages.

    For each problem, generates n_variants completions using different
    prompt framings (1-shot through n_variants-shot). Each is verified
    deterministically. Advantages are computed per-problem group.

    n_variants should equal ``len(default_few_shot_examples())`` — the number
    of unique demonstrations available. With that many variants, each prompt
    has a distinct set of demonstrations (1-shot, 2-shot, ..., n-shot).
    Beyond that count, demonstrations cycle and repeat.

    Parameters
    ----------
    problems : list[StarProblem]
        Problems with deterministic verifiers.
    generate_fn : callable
        ``generate_fn(prompt: str, max_tokens: int) -> str``
        Greedy generation function.
    tokenize_fn : callable
        ``tokenize_fn(text: str) -> list[int]``
        Tokenizer encode function.
    n_variants : int
        Number of prompt variants per problem. Derived from
        ``len(default_few_shot_examples())`` by the caller.
    max_tokens : int
        Maximum tokens per generation. Should match training ``seq_length``.

    Returns
    -------
    OutcomeCollectionResult
    """
    from modelcypher.core.domain.star.prompting import build_forward_prompt

    all_completions: list[OutcomeCompletion] = []
    n_mixed = 0

    for problem in problems:
        group_rewards: list[float] = []
        group_completions: list[OutcomeCompletion] = []

        for variant_idx in range(n_variants):
            # Vary number of demonstrations: 1-shot through n_variants-shot
            n_demos = variant_idx + 1
            prompt = build_forward_prompt(problem, demonstrations=n_demos)

            try:
                response = generate_fn(prompt, max_tokens)
            except Exception:
                logger.debug(
                    "Generation failed for %s variant %d",
                    problem.problem_id, variant_idx,
                    exc_info=True,
                )
                response = ""

            correct = problem.verify_response(response)
            reward = 1.0 if correct else 0.0
            group_rewards.append(reward)

            # Tokenize prompt and full sequence separately to track
            # the response boundary. Williams (1992): REINFORCE gradient
            # is over actions (response), not states (prompt).
            prompt_with_sep = prompt + " "
            prompt_tokens = tokenize_fn(prompt_with_sep)
            response_start = len(prompt_tokens)
            full_text = prompt_with_sep + response
            tokens = tokenize_fn(full_text)

            group_completions.append(OutcomeCompletion(
                problem_id=problem.problem_id,
                prompt_variant=variant_idx,
                tokens=tokens,
                correct=correct,
                response_start=response_start,
            ))

        # Compute advantages for this problem group
        advantages = compute_group_advantages(group_rewards)
        has_mixed = any(a != 0.0 for a in advantages)
        if has_mixed:
            n_mixed += 1

        for comp, adv in zip(group_completions, advantages):
            comp.advantage = adv

        all_completions.extend(group_completions)

    n_correct = sum(1 for c in all_completions if c.correct)
    n_incorrect = len(all_completions) - n_correct
    mean_adv = (
        sum(c.advantage for c in all_completions) / len(all_completions)
        if all_completions else 0.0
    )

    result = OutcomeCollectionResult(
        completions=all_completions,
        n_problems=len(problems),
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_mixed_problems=n_mixed,
        mean_advantage=mean_adv,
    )

    logger.info(
        "Outcome collection: %d problems, %d completions, "
        "%d correct, %d incorrect, %d mixed (%.1f%% signal density)",
        result.n_problems,
        len(result.completions),
        result.n_correct,
        result.n_incorrect,
        result.n_mixed_problems,
        result.signal_density * 100,
    )

    return result


__all__ = [
    "OutcomeCollectionResult",
    "OutcomeCompletion",
    "collect_outcomes",
    "compute_group_advantages",
]
