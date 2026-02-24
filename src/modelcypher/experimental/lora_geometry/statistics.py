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

"""Statistical analysis using Backend protocol only. No NumPy/SciPy.

Implements:
- Bootstrap confidence intervals
- Permutation tests
- Pearson and Spearman correlation
- Kendall tau (monotonicity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval.

    Attributes:
        lower: Lower bound of CI.
        upper: Upper bound of CI.
        point_estimate: Point estimate (e.g., mean).
        resamples: Number of bootstrap resamples used.
        alpha: Significance level (CI is 1-2*alpha).
    """

    lower: float
    upper: float
    point_estimate: float
    resamples: int
    alpha: float


@dataclass(frozen=True)
class CorrelationResult:
    """Result of correlation computation.

    Attributes:
        r: Correlation coefficient.
        ci: Bootstrap confidence interval for r.
        n: Sample size.
    """

    r: float
    ci: BootstrapCI | None
    n: int


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a permutation test.

    Attributes:
        observed_stat: The observed test statistic.
        permutation_stats: Distribution of statistic under null.
        p_value: Fraction of permutations >= observed (descriptive only).
        percentile_5: 5th percentile of permutation distribution.
        percentile_50: Median of permutation distribution.
        percentile_95: 95th percentile of permutation distribution.
        n_permutations: Number of unique permutations generated.
    """

    observed_stat: float
    permutation_stats: list[float]
    p_value: float
    percentile_5: float
    percentile_50: float
    percentile_95: float
    n_permutations: int


def compute_bootstrap_ci(
    values: list[float],
    statistic_fn: callable | None = None,
    n_resamples: int | None = None,
    alpha: float | None = None,
    backend: "Backend | None" = None,
) -> BootstrapCI:
    """Compute bootstrap confidence interval using Backend operations.

    Args:
        values: List of values to bootstrap.
        statistic_fn: Function to compute statistic (default: mean).
        n_resamples: Number of resamples (default: sample_size per plan).
        alpha: Significance level (default: derived from precision).
        backend: Compute backend.

    Returns:
        BootstrapCI with lower, upper, and metadata.
    """
    if backend is None:
        backend = get_default_backend()

    n = len(values)
    if n < 2:
        point = values[0] if values else 0.0
        return BootstrapCI(
            lower=point,
            upper=point,
            point_estimate=point,
            resamples=0,
            alpha=0.05,
        )

    # Default: mean statistic
    if statistic_fn is None:

        def statistic_fn(v):
            return sum(v) / len(v)

    # Resamples = sample_size (per plan: "resamples follow sample size")
    if n_resamples is None:
        n_resamples = n

    # Alpha derived from precision (per plan)
    if alpha is None:
        arr = backend.array(values)
        eps = division_epsilon(backend, arr)
        alpha = max(float(eps), 1.0 / float(n_resamples))

    # Point estimate
    point_estimate = statistic_fn(values)

    # Bootstrap resampling using backend
    arr = backend.array(values)
    bootstrap_stats: list[float] = []

    for _ in range(n_resamples):
        # Random indices with replacement
        indices = backend.random_randint(0, n, shape=(n,))
        sample = backend.take(arr, indices)
        backend.eval(sample)
        sample_list = backend.tolist(sample)
        stat = statistic_fn(sample_list)
        bootstrap_stats.append(stat)

    if len(bootstrap_stats) < 2:
        return BootstrapCI(
            lower=point_estimate,
            upper=point_estimate,
            point_estimate=point_estimate,
            resamples=len(bootstrap_stats),
            alpha=alpha,
        )

    # Sort and extract quantiles
    bootstrap_stats.sort()
    lower_idx = int(len(bootstrap_stats) * alpha)
    upper_idx = int(len(bootstrap_stats) * (1.0 - alpha))

    # Ensure valid indices
    lower_idx = max(0, min(lower_idx, len(bootstrap_stats) - 1))
    upper_idx = max(0, min(upper_idx, len(bootstrap_stats) - 1))

    return BootstrapCI(
        lower=bootstrap_stats[lower_idx],
        upper=bootstrap_stats[upper_idx],
        point_estimate=point_estimate,
        resamples=len(bootstrap_stats),
        alpha=alpha,
    )


def compute_pearson_correlation(
    x: list[float],
    y: list[float],
    with_ci: bool = False,
    n_bootstrap: int | None = None,
    backend: "Backend | None" = None,
) -> CorrelationResult:
    """Compute Pearson correlation coefficient using Backend operations.

    Formula: r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]

    Args:
        x: First variable.
        y: Second variable (must have same length as x).
        with_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap resamples.
        backend: Compute backend.

    Returns:
        CorrelationResult with r and optional CI.
    """
    if backend is None:
        backend = get_default_backend()

    n = min(len(x), len(y))
    if n < 2:
        return CorrelationResult(r=0.0, ci=None, n=n)

    x_arr = backend.array(x[:n])
    y_arr = backend.array(y[:n])
    backend.eval(x_arr, y_arr)

    # Compute means
    x_mean = backend.sum(x_arr) / n
    y_mean = backend.sum(y_arr) / n

    # Compute deviations
    x_dev = x_arr - x_mean
    y_dev = y_arr - y_mean

    # Compute covariance and variances
    cov = backend.sum(x_dev * y_dev)
    var_x = backend.sum(x_dev * x_dev)
    var_y = backend.sum(y_dev * y_dev)

    # Compute r
    denom = backend.sqrt(var_x * var_y)
    eps = machine_epsilon(backend, x_arr)

    backend.eval(cov, denom)
    cov_val = float(backend.to_scalar(cov))
    denom_val = float(backend.to_scalar(denom))

    if denom_val < eps:
        r = 0.0
    else:
        r = cov_val / denom_val
        # Clamp to [-1, 1]
        r = max(-1.0, min(1.0, r))

    ci = None
    if with_ci:
        # Bootstrap CI for correlation
        if n_bootstrap is None:
            n_bootstrap = n

        def corr_stat(indices):
            x_s = [x[i] for i in indices]
            y_s = [y[i] for i in indices]
            return compute_pearson_correlation(x_s, y_s, with_ci=False, backend=backend).r

        bootstrap_rs: list[float] = []
        for _ in range(n_bootstrap):
            indices = backend.random_randint(0, n, shape=(n,))
            backend.eval(indices)
            idx_list = [int(i) for i in backend.tolist(indices)]
            r_boot = corr_stat(idx_list)
            bootstrap_rs.append(r_boot)

        if bootstrap_rs:
            bootstrap_rs.sort()
            alpha = max(eps, 1.0 / n_bootstrap)
            lower_idx = max(0, int(len(bootstrap_rs) * alpha))
            upper_idx = min(len(bootstrap_rs) - 1, int(len(bootstrap_rs) * (1 - alpha)))
            ci = BootstrapCI(
                lower=bootstrap_rs[lower_idx],
                upper=bootstrap_rs[upper_idx],
                point_estimate=r,
                resamples=len(bootstrap_rs),
                alpha=alpha,
            )

    return CorrelationResult(r=r, ci=ci, n=n)


def compute_spearman_correlation(
    x: list[float],
    y: list[float],
    with_ci: bool = False,
    n_bootstrap: int | None = None,
    backend: "Backend | None" = None,
) -> CorrelationResult:
    """Compute Spearman rank correlation using Backend operations.

    Spearman ρ is Pearson r applied to ranks.

    Args:
        x: First variable.
        y: Second variable.
        with_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap resamples.
        backend: Compute backend.

    Returns:
        CorrelationResult with ρ and optional CI.
    """
    if backend is None:
        backend = get_default_backend()

    n = min(len(x), len(y))
    if n < 2:
        return CorrelationResult(r=0.0, ci=None, n=n)

    # Compute ranks using backend sort
    def compute_ranks(values: list[float]) -> list[float]:
        """Compute ranks with average tie handling."""
        n = len(values)
        arr = backend.array(values)
        indices = backend.argsort(arr)
        backend.eval(indices)
        sorted_indices = backend.tolist(indices)

        ranks = [0.0] * n
        i = 0
        while i < n:
            # Find all tied values
            j = i + 1
            while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            # Average rank for ties
            avg_rank = (i + j + 1) / 2.0  # +1 for 1-based ranks
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            i = j
        return ranks

    x_ranks = compute_ranks(x[:n])
    y_ranks = compute_ranks(y[:n])

    # Pearson on ranks
    return compute_pearson_correlation(
        x_ranks, y_ranks, with_ci=with_ci, n_bootstrap=n_bootstrap, backend=backend
    )


def compute_kendall_tau(
    x: list[float],
    y: list[float],
    backend: "Backend | None" = None,
) -> float:
    """Compute Kendall's tau-b rank correlation.

    Measures monotonicity. Range: [-1, 1].
    τ = (concordant - discordant) / sqrt((n0 - n1)(n0 - n2))
    where n0 = n(n-1)/2, n1 = ties in x, n2 = ties in y.

    Args:
        x: First variable.
        y: Second variable.
        backend: Compute backend (not used heavily, but for consistency).

    Returns:
        Kendall's tau-b coefficient.
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]

            if dx == 0 and dy == 0:
                ties_xy += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    n0 = n * (n - 1) // 2
    n1 = ties_x + ties_xy
    n2 = ties_y + ties_xy

    denom_sq = (n0 - n1) * (n0 - n2)
    if denom_sq <= 0:
        return 0.0

    tau = (concordant - discordant) / (denom_sq**0.5)
    return max(-1.0, min(1.0, tau))


def compute_permutation_test(
    group1_values: list[float],
    group2_values: list[float],
    statistic_fn: callable | None = None,
    n_permutations: int | None = None,
    backend: "Backend | None" = None,
) -> PermutationTestResult:
    """Perform permutation test for difference between two groups.

    Default statistic: difference of means.

    Args:
        group1_values: Values from first group.
        group2_values: Values from second group.
        statistic_fn: Function(g1, g2) -> float. Default: mean(g1) - mean(g2).
        n_permutations: Number of permutations (default: 1000 or all if small).
        backend: Compute backend.

    Returns:
        PermutationTestResult with observed stat, distribution, and percentiles.
    """
    if backend is None:
        backend = get_default_backend()

    n1 = len(group1_values)
    n2 = len(group2_values)
    n_total = n1 + n2

    if n1 == 0 or n2 == 0:
        return PermutationTestResult(
            observed_stat=0.0,
            permutation_stats=[],
            p_value=1.0,
            percentile_5=0.0,
            percentile_50=0.0,
            percentile_95=0.0,
            n_permutations=0,
        )

    # Default statistic: difference of means
    if statistic_fn is None:

        def statistic_fn(g1, g2):
            m1 = sum(g1) / len(g1) if g1 else 0.0
            m2 = sum(g2) / len(g2) if g2 else 0.0
            return m1 - m2

    # Observed statistic
    observed = statistic_fn(group1_values, group2_values)

    # Combined data
    combined = group1_values + group2_values
    backend.array(combined)

    # Number of permutations
    if n_permutations is None:
        # Default: 1000 or fewer if combinatorics allow
        # C(n, n1) is the number of unique permutations
        from math import comb

        max_unique = comb(n_total, n1)
        n_permutations = min(1000, max_unique)

    # Generate permutations
    perm_stats: list[float] = []
    seen_perms: set[tuple[int, ...]] = set()

    for _ in range(n_permutations * 2):  # Try more to get unique
        if len(perm_stats) >= n_permutations:
            break

        # Random permutation
        perm = backend.randperm(n_total)
        backend.eval(perm)
        perm_list = tuple(int(i) for i in backend.tolist(perm))

        # Only first n1 indices matter for uniqueness
        key = tuple(sorted(perm_list[:n1]))
        if key in seen_perms:
            continue
        seen_perms.add(key)

        # Split by permutation
        g1_perm = [combined[perm_list[i]] for i in range(n1)]
        g2_perm = [combined[perm_list[i]] for i in range(n1, n_total)]

        stat = statistic_fn(g1_perm, g2_perm)
        perm_stats.append(stat)

    if not perm_stats:
        return PermutationTestResult(
            observed_stat=observed,
            permutation_stats=[],
            p_value=1.0,
            percentile_5=0.0,
            percentile_50=0.0,
            percentile_95=0.0,
            n_permutations=0,
        )

    # p-value: fraction >= observed (two-tailed: use abs)
    n_extreme = sum(1 for s in perm_stats if abs(s) >= abs(observed))
    p_value = n_extreme / len(perm_stats)

    # Percentiles
    perm_stats_sorted = sorted(perm_stats)
    n_stats = len(perm_stats_sorted)

    def percentile(p):
        idx = int(n_stats * p)
        return perm_stats_sorted[min(idx, n_stats - 1)]

    return PermutationTestResult(
        observed_stat=observed,
        permutation_stats=perm_stats,
        p_value=p_value,
        percentile_5=percentile(0.05),
        percentile_50=percentile(0.50),
        percentile_95=percentile(0.95),
        n_permutations=len(seen_perms),
    )
