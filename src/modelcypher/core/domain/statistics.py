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

"""Two-group hypothesis testing, correlation, and curve fitting.

Pure Python — zero framework dependencies.

Extracted from rl_geometric_attractor_experiment.py, rl_processing_geometry_experiment.py,
and rl_spectral_anatomy_experiment.py where these functions were duplicated.

Note: ``core/support/statistics.py`` has ``cohens_d(observed, null_mean, null_std)``
for single-value vs null distribution. The functions here operate on two independent
sample groups — different use case, no overlap.
"""

from __future__ import annotations

import math
import random

from scipy.stats import beta

# ---------------------------------------------------------------------------
# Exact confidence intervals
# ---------------------------------------------------------------------------


def clopper_pearson_interval(
    *,
    n_correct: int,
    n_total: int,
    alpha: float,
) -> tuple[float, float]:
    """Compute exact Clopper-Pearson interval for Binomial(n_total, p).

    Reference: Clopper & Pearson (1934), "The use of confidence or fiducial
    limits illustrated in the case of the binomial," Biometrika 26(4):404-413.

    Parameters
    ----------
    n_correct:
        Number of successes observed.
    n_total:
        Number of trials.
    alpha:
        Significance level (e.g. 0.05 for 95% CI). The interval is
        [alpha/2, 1 - alpha/2] in probability.

    Returns
    -------
    (lower, upper):
        Exact confidence bounds for the true success rate p.
    """
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


def confidence_intervals_overlap(
    interval_a: tuple[float, float],
    interval_b: tuple[float, float],
) -> bool:
    """Return True when two closed intervals overlap."""
    lower_a, upper_a = interval_a
    lower_b, upper_b = interval_b
    if lower_a > upper_a:
        raise ValueError(
            f"interval_a must satisfy lower <= upper, got {interval_a}",
        )
    if lower_b > upper_b:
        raise ValueError(
            f"interval_b must satisfy lower <= upper, got {interval_b}",
        )
    return (upper_a >= lower_b) and (upper_b >= lower_a)


def binomial_degradation_is_significant(
    *,
    baseline_n_correct: int,
    current_n_correct: int,
    n_total: int,
    alpha: float,
) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    """Significance test for online-eval degradation via CP non-overlap.

    Uses exact Clopper-Pearson intervals for both baseline and current
    correctness counts. Degradation is significant iff the current upper bound
    is strictly below the baseline lower bound.
    """
    baseline_ci = clopper_pearson_interval(
        n_correct=baseline_n_correct,
        n_total=n_total,
        alpha=alpha,
    )
    current_ci = clopper_pearson_interval(
        n_correct=current_n_correct,
        n_total=n_total,
        alpha=alpha,
    )
    significant = current_ci[1] < baseline_ci[0]
    return significant, current_ci, baseline_ci


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def safe_mean(values: list[float]) -> float:
    """Mean of *values*, filtering NaN."""
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def safe_std(values: list[float]) -> float:
    """Population std of *values*, filtering NaN."""
    valid = [v for v in values if v == v]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def mean_trajectory(trajectories: list[list[float]]) -> list[float]:
    """Element-wise mean across multiple trajectories (ragged-safe, NaN-safe)."""
    valid = [t for t in trajectories if t]
    if not valid:
        return []
    max_len = max(len(t) for t in valid)
    result = []
    for i in range(max_len):
        vals = [t[i] for t in valid if i < len(t) and t[i] == t[i]]
        result.append(sum(vals) / len(vals) if vals else float("nan"))
    return result


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------


def cohens_d_two_groups(group1: list[float], group2: list[float]) -> float:
    """Cohen's d between two independent groups (pooled std)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

    if pooled_var <= 0:
        return 0.0
    return (m1 - m2) / math.sqrt(pooled_var)


def cohens_d_bootstrap_ci(
    group1: list[float],
    group2: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Bootstrap CI on Cohen's d between two groups.

    Parameters
    ----------
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int | None
        RNG seed for reproducibility. Ignored if *rng* is provided.
    rng : random.Random | None
        Existing RNG instance. If ``None``, one is created from *seed*.
    """
    if len(group1) < 2 or len(group2) < 2:
        return (0.0, 0.0)

    if rng is None:
        rng = random.Random(seed if seed is not None else 42)
    alpha = 1.0 - confidence

    d_samples = []
    for _ in range(n_bootstrap):
        g1 = rng.choices(group1, k=len(group1))
        g2 = rng.choices(group2, k=len(group2))
        d_samples.append(cohens_d_two_groups(g1, g2))

    d_samples.sort()
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return (d_samples[lo_idx], d_samples[hi_idx])


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------


def permutation_test_p_value(
    group1: list[float],
    group2: list[float],
    n_permutations: int = 1000,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> float:
    """Two-sided permutation test for difference of means.

    Parameters
    ----------
    seed : int | None
        RNG seed. Ignored if *rng* is provided.
    rng : random.Random | None
        Existing RNG instance. If ``None``, one is created from *seed*.
    """
    if not group1 or not group2:
        return 1.0

    if rng is None:
        rng = random.Random(seed if seed is not None else 42)

    observed = abs(sum(group1) / len(group1) - sum(group2) / len(group2))
    combined = group1 + group2
    n1 = len(group1)
    n_extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_diff = abs(sum(perm_g1) / len(perm_g1) - sum(perm_g2) / len(perm_g2))
        if perm_diff >= observed:
            n_extreme += 1

    return (n_extreme + 1) / (n_permutations + 1)


def levene_test_statistic(
    groups: list[list[float]],
) -> tuple[float, float]:
    """Brown-Forsythe variant of Levene's test for equality of variances.

    Uses median instead of mean for robustness.

    Returns
    -------
    (test_statistic, approximate_p_value)
    """
    k = len(groups)
    if k < 2:
        return (0.0, 1.0)

    ns = [len(g) for g in groups]
    N = sum(ns)

    if any(n < 2 for n in ns):
        return (0.0, 1.0)

    # Brown-Forsythe: use median instead of mean
    medians = []
    for g in groups:
        sg = sorted(g)
        mid = len(sg) // 2
        if len(sg) % 2 == 0:
            medians.append((sg[mid - 1] + sg[mid]) / 2)
        else:
            medians.append(sg[mid])

    # Absolute deviations from group medians
    z_groups: list[list[float]] = []
    for g, med in zip(groups, medians):
        z_groups.append([abs(x - med) for x in g])

    # Group means of deviations
    z_means = [sum(z) / len(z) for z in z_groups]
    z_grand_mean = sum(sum(z) for z in z_groups) / N

    # Between-group sum of squares
    ss_between = sum(n * (zm - z_grand_mean) ** 2 for n, zm in zip(ns, z_means))

    # Within-group sum of squares
    ss_within = 0.0
    for z, zm in zip(z_groups, z_means):
        ss_within += sum((zi - zm) ** 2 for zi in z)

    if ss_within <= 0:
        return (0.0, 1.0)

    df1 = k - 1
    df2 = N - k
    if df2 <= 0:
        return (0.0, 1.0)

    F = (ss_between / df1) / (ss_within / df2)
    p = _f_distribution_p_value(F, df1, df2)

    return (F, p)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------


def fit_linear(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """OLS: y = alpha * x + beta.

    Returns (alpha, beta, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    mx = sum(x) / n
    my = sum(y) / n
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_yy = sum((yi - my) ** 2 for yi in y)

    # Python float is IEEE 754 float64: eps = 2^-52 ≈ 2.2e-16.
    # Use eps_f64 as zero-variance guard for OLS denominator.
    _eps_f64 = math.ldexp(1.0, -52)
    if ss_xx < _eps_f64:
        return 0.0, my, 0.0

    alpha = ss_xy / ss_xx
    beta = my - alpha * mx

    ss_res = sum((yi - (alpha * xi + beta)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - ss_res / ss_yy if ss_yy > _eps_f64 else 0.0

    return alpha, beta, r_squared


def fit_exponential(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = a * exp(-b * x) via log-linear OLS.

    Returns (a, b, r_squared) where r_squared is on the original scale.
    Only uses positive y values.
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if yi > 0]
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0

    x_pos = [p[0] for p in pairs]
    y_pos = [p[1] for p in pairs]
    log_y = [math.log(yi) for yi in y_pos]

    alpha, beta, _ = fit_linear(x_pos, log_y)
    a = math.exp(beta)
    b = -alpha

    # R² on original scale
    y_pred = [a * math.exp(-b * xi) for xi in x_pos]
    my = sum(y_pos) / len(y_pos)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y_pos, y_pred))
    ss_tot = sum((yi - my) ** 2 for yi in y_pos)
    _eps_f64 = math.ldexp(1.0, -52)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > _eps_f64 else 0.0

    return a, b, r_squared


def fit_inverse(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = alpha / x + beta via OLS on y vs 1/x.

    Returns (alpha, beta, r_squared).
    """
    _eps_f64 = math.ldexp(1.0, -52)
    pairs = [(xi, yi) for xi, yi in zip(x, y) if abs(xi) > _eps_f64]
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0

    inv_x = [1.0 / p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    return fit_linear(inv_x, y_vals)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _f_distribution_p_value(F: float, df1: int, df2: int) -> float:
    """P(F > f) = I_x(df2/2, df1/2) where x = df2 / (df2 + df1 * f)."""
    if F <= 0:
        return 1.0

    x = df2 / (df2 + df1 * F)
    a = df2 / 2.0
    b = df1 / 2.0

    return _regularized_incomplete_beta(x, a, b)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via trapezoidal rule."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    n_steps = 10000
    dt = x / n_steps
    total = 0.0

    log_beta_val = _log_beta(a, b)

    for i in range(n_steps):
        t0 = i * dt
        t1 = (i + 1) * dt

        vals = []
        for t in (t0, t1):
            if t <= 0:
                if a >= 1:
                    vals.append(0.0)
                else:
                    vals.append(float("inf"))
                continue
            if t >= 1:
                if b >= 1:
                    vals.append(0.0)
                else:
                    vals.append(float("inf"))
                continue
            log_val = (a - 1) * math.log(t) + (b - 1) * math.log(1 - t) - log_beta_val
            vals.append(math.exp(log_val))

        total += 0.5 * (vals[0] + vals[1]) * dt

    return max(0.0, min(1.0, total))


def _log_beta(a: float, b: float) -> float:
    """log(Beta(a, b)) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
