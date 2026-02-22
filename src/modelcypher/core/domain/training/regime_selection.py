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

"""Automatic training-regime selection from baseline correctness measurements.

Regime selection is derived from measured baseline behavior:
- CE when a problem type has zero demonstrated capability.
- REINFORCE+entropy when capability exists but confidence remains below chance.
- REINFORCE when confidence lower-bound is above chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from scipy.stats import beta

from modelcypher.core.domain.training.online_eval import OnlineEvalResult

DEFAULT_PROBLEM_TYPE_CHANCE_RATES: dict[str, float] = {
    "syllogistic_chain": 0.5,
    "contrapositive": 0.5,
    "set_exception": 0.5,
    "multi_step_arithmetic": 0.0,
    "compositional_binding": 0.0,
}


@dataclass(frozen=True)
class PerTypeRegime:
    """Per-problem-type training regime classification."""

    problem_type: str
    n_correct: int
    n_total: int
    observed_accuracy: float
    ci_lower: float
    ci_upper: float
    chance_rate: float
    regime: str  # "ce" | "reinforce_entropy" | "reinforce"
    rationale: str


@dataclass(frozen=True)
class TrainingRegime:
    """Global and per-type regime selection derived from baseline behavior."""

    global_regime: str  # "ce" | "reinforce" | "reinforce_entropy" | "hybrid"
    per_type: dict[str, PerTypeRegime]
    use_ce: bool
    use_outcome_training: bool
    use_entropy_regularization: bool
    confidence_level: float
    n_total: int
    derivation_log: tuple[str, ...]


def _clopper_pearson_interval(
    *,
    n_correct: int,
    n_total: int,
    alpha: float,
) -> tuple[float, float]:
    """Compute exact Clopper-Pearson interval for Binomial(n_total, p)."""
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")
    if n_correct < 0 or n_correct > n_total:
        raise ValueError(
            f"n_correct must satisfy 0 <= n_correct <= n_total, got {n_correct}",
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")

    lower = (
        0.0
        if n_correct == 0
        else float(beta.ppf(alpha / 2.0, n_correct, n_total - n_correct + 1))
    )
    upper = (
        1.0
        if n_correct == n_total
        else float(beta.ppf(1.0 - alpha / 2.0, n_correct + 1, n_total - n_correct))
    )
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(
            "Clopper-Pearson interval returned non-finite bounds: "
            f"lower={lower}, upper={upper}",
        )
    return lower, upper


def select_training_regime(
    baseline_result: OnlineEvalResult,
    *,
    problem_type_chance_rates: Mapping[str, float] | None = None,
) -> TrainingRegime:
    """Derive training regime from baseline online-eval measurements.

    Parameters
    ----------
    baseline_result:
        Baseline online correctness measurement.
    problem_type_chance_rates:
        Optional overrides for chance rates by problem type.
    """
    n_total = int(baseline_result.n_total)
    if n_total <= 1:
        raise ValueError(
            "Regime selection requires n_total > 1 to derive confidence "
            f"(received n_total={n_total}).",
        )

    alpha = 1.0 / float(n_total)
    confidence_level = 1.0 - alpha

    chance_rates = dict(DEFAULT_PROBLEM_TYPE_CHANCE_RATES)
    if problem_type_chance_rates is not None:
        chance_rates.update(problem_type_chance_rates)

    per_type: dict[str, PerTypeRegime] = {}
    derivation_log: list[str] = [
        (
            "confidence derivation: "
            f"n_total={n_total}, alpha=1/n_total={alpha:.10f}, "
            f"confidence_level={confidence_level:.10f}"
        ),
    ]

    for problem_type in sorted(baseline_result.per_type_total):
        n_type = int(baseline_result.per_type_total.get(problem_type, 0))
        if n_type <= 0:
            continue
        n_correct = int(baseline_result.per_type_correct.get(problem_type, 0))
        observed_accuracy = n_correct / float(n_type)
        chance_rate = float(chance_rates.get(problem_type, 0.0))

        ci_lower, ci_upper = _clopper_pearson_interval(
            n_correct=n_correct,
            n_total=n_type,
            alpha=alpha,
        )

        if n_correct == 0:
            regime = "ce"
            rationale = "no demonstrated baseline capability (k=0)"
        elif ci_lower < chance_rate:
            regime = "reinforce_entropy"
            rationale = (
                "partial capability detected but confidence lower bound remains "
                "below chance"
            )
        else:
            regime = "reinforce"
            rationale = "confidence lower bound is at or above chance"

        per_type[problem_type] = PerTypeRegime(
            problem_type=problem_type,
            n_correct=n_correct,
            n_total=n_type,
            observed_accuracy=observed_accuracy,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            chance_rate=chance_rate,
            regime=regime,
            rationale=rationale,
        )
        derivation_log.append(
            (
                f"{problem_type}: k={n_correct}, n={n_type}, "
                f"acc={observed_accuracy:.6f}, "
                f"ci=[{ci_lower:.6f}, {ci_upper:.6f}], "
                f"chance={chance_rate:.6f}, regime={regime}"
            ),
        )

    if not per_type:
        raise ValueError(
            "Baseline result has no per-type totals; cannot derive regime.",
        )

    regimes = [entry.regime for entry in per_type.values()]
    has_ce = any(regime == "ce" for regime in regimes)
    has_reinforce_entropy = any(regime == "reinforce_entropy" for regime in regimes)
    has_reinforce = any(regime == "reinforce" for regime in regimes)

    if all(regime == "ce" for regime in regimes):
        global_regime = "ce"
        use_outcome_training = False
        use_entropy_regularization = False
    elif has_ce:
        global_regime = "hybrid"
        use_outcome_training = True
        # Hybrid runs REINFORCE alongside CE; entropy reg is always on
        # when REINFORCE is active to prevent mode collapse during
        # exploration (1-shot RLVR, NeurIPS 2025: entropy loss is critical).
        use_entropy_regularization = True
    elif has_reinforce_entropy:
        global_regime = "reinforce_entropy"
        use_outcome_training = True
        use_entropy_regularization = True
    elif has_reinforce:
        global_regime = "reinforce"
        use_outcome_training = True
        use_entropy_regularization = False
    else:
        raise ValueError(f"Unexpected regime set: {sorted(set(regimes))}")

    derivation_log.append(
        (
            "global decision: "
            f"global_regime={global_regime}, use_ce=True, "
            f"use_outcome_training={use_outcome_training}, "
            f"use_entropy_regularization={use_entropy_regularization}"
        ),
    )

    return TrainingRegime(
        global_regime=global_regime,
        per_type=per_type,
        use_ce=True,
        use_outcome_training=use_outcome_training,
        use_entropy_regularization=use_entropy_regularization,
        confidence_level=confidence_level,
        n_total=n_total,
        derivation_log=tuple(derivation_log),
    )


__all__ = [
    "DEFAULT_PROBLEM_TYPE_CHANCE_RATES",
    "PerTypeRegime",
    "TrainingRegime",
    "select_training_regime",
]
