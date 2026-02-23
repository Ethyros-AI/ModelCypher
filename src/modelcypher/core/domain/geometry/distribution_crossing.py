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

"""Find Bayes-optimal decision boundaries between two empirical distributions.

Given samples from two groups (e.g. safe vs attack), find the value x* where
the empirical CDFs cross: CDF_a(x*) = 1 - CDF_b(x*). This is the point
that minimizes total classification error (Neyman-Pearson optimal boundary
for equal priors).

Bootstrap resampling provides a confidence interval on the crossing location.
If the CI is wide relative to the group separation, the boundary is unstable
and should not be promoted.

Pure Python — zero framework dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CrossingResult:
    """Result of empirical distribution crossing detection."""

    boundary: float
    """Crossing point x* where CDF_a(x*) = 1 - CDF_b(x*)."""

    ci_lower: float
    """Bootstrap lower bound on boundary."""

    ci_upper: float
    """Bootstrap upper bound on boundary."""

    false_alarm_rate: float
    """P(classify as group_b | actually group_a) at boundary."""

    miss_rate: float
    """P(classify as group_a | actually group_b) at boundary."""

    auroc: float
    """Area under ROC curve (Mann-Whitney U statistic)."""

    is_stable: bool
    """True if CI width < Cohen's d pooled std (boundary is well-resolved)."""

    n_a: int
    """Number of samples in group a."""

    n_b: int
    """Number of samples in group b."""


def find_distribution_crossing(
    group_a: list[float],
    group_b: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> CrossingResult:
    """Find the Bayes-optimal decision boundary between two groups.

    Algorithm:
    1. Merge and sort all values to create evaluation grid.
    2. At each grid point x, compute CDF_a(x) and 1 - CDF_b(x).
    3. Crossing is where |CDF_a(x) - (1 - CDF_b(x))| is minimized.
    4. Bootstrap: resample both groups, re-find crossing. CI from
       percentile method.

    Args:
        group_a: Samples from distribution A (e.g. safe prompts).
        group_b: Samples from distribution B (e.g. attack prompts).
        n_bootstrap: Number of bootstrap resamples for CI.
        confidence: Confidence level for bootstrap CI.
        seed: RNG seed for reproducibility.

    Returns:
        CrossingResult with boundary, CI, error rates, and stability.

    Raises:
        ValueError: If either group has fewer than 2 samples.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            f"Both groups need >= 2 samples (got {len(group_a)}, {len(group_b)})"
        )

    boundary = _find_crossing_point(group_a, group_b)
    false_alarm = _empirical_rate_above(group_a, boundary)
    miss = _empirical_rate_below(group_b, boundary)
    auroc = _mann_whitney_auroc(group_a, group_b)

    rng = random.Random(seed if seed is not None else 42)
    bootstrap_boundaries: list[float] = []
    for _ in range(n_bootstrap):
        resample_a = rng.choices(group_a, k=len(group_a))
        resample_b = rng.choices(group_b, k=len(group_b))
        bootstrap_boundaries.append(_find_crossing_point(resample_a, resample_b))

    bootstrap_boundaries.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap) - 1)
    ci_lower = bootstrap_boundaries[lo_idx]
    ci_upper = bootstrap_boundaries[hi_idx]

    # Stability: CI width < pooled std (the groups are well-separated
    # relative to the uncertainty in the boundary location)
    ci_width = ci_upper - ci_lower
    pooled_std = _pooled_std(group_a, group_b)
    # Guard: if pooled_std is zero (identical samples), boundary is trivially stable
    is_stable = ci_width < pooled_std if pooled_std > 0 else True

    return CrossingResult(
        boundary=boundary,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        false_alarm_rate=false_alarm,
        miss_rate=miss,
        auroc=auroc,
        is_stable=is_stable,
        n_a=len(group_a),
        n_b=len(group_b),
    )


def _find_crossing_point(group_a: list[float], group_b: list[float]) -> float:
    """Find x where CDF_a(x) = 1 - CDF_b(x) via grid search on merged values."""
    sorted_a = sorted(group_a)
    sorted_b = sorted(group_b)
    n_a = len(sorted_a)
    n_b = len(sorted_b)

    # Evaluation grid: all unique values from both groups
    grid = sorted(set(sorted_a + sorted_b))
    if not grid:
        return 0.0

    best_x = grid[0]
    best_diff = float("inf")

    # Pointers for computing CDFs incrementally
    ia = 0
    ib = 0
    for x in grid:
        # CDF_a(x) = fraction of group_a <= x
        while ia < n_a and sorted_a[ia] <= x:
            ia += 1
        cdf_a = ia / n_a

        # 1 - CDF_b(x) = fraction of group_b > x
        while ib < n_b and sorted_b[ib] <= x:
            ib += 1
        one_minus_cdf_b = 1.0 - ib / n_b

        diff = abs(cdf_a - one_minus_cdf_b)
        if diff < best_diff:
            best_diff = diff
            best_x = x

    return best_x


def _empirical_rate_above(values: list[float], threshold: float) -> float:
    """Fraction of values strictly above threshold."""
    if not values:
        return 0.0
    return sum(1 for v in values if v > threshold) / len(values)


def _empirical_rate_below(values: list[float], threshold: float) -> float:
    """Fraction of values at or below threshold."""
    if not values:
        return 0.0
    return sum(1 for v in values if v <= threshold) / len(values)


def _mann_whitney_auroc(group_a: list[float], group_b: list[float]) -> float:
    """AUROC via Mann-Whitney U statistic.

    AUROC = P(random sample from B > random sample from A).
    This equals U / (n_a * n_b) where U counts concordant pairs.
    """
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a == 0 or n_b == 0:
        return 0.5

    u = 0
    ties = 0
    for b_val in group_b:
        for a_val in group_a:
            if b_val > a_val:
                u += 1
            elif b_val == a_val:
                ties += 1

    return (u + 0.5 * ties) / (n_a * n_b)


def _pooled_std(group_a: list[float], group_b: list[float]) -> float:
    """Pooled standard deviation of two groups."""
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    return math.sqrt(pooled_var) if pooled_var > 0 else 0.0
