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

"""Spectral changepoint detection via piecewise linear regression.

Detects slope breaks in ordered value sequences (e.g. sorted var_top1 across
layers, or log-singular-values). The algorithm fits two OLS lines on either
side of each candidate split point and selects the split that minimizes total
residual sum of squares (Bai & Perron 1998, single changepoint case).

Bootstrap resampling tests whether the changepoint location is stable. If
the modal changepoint appears in fewer than 50% of resamples, the break is
not robust and should be marked INCONCLUSIVE.

Pure Python — zero framework dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from modelcypher.core.domain.statistics import fit_linear


@dataclass(frozen=True)
class ChangePointResult:
    """Result of spectral changepoint detection."""

    k: int
    """Index of the changepoint (split between k and k+1)."""

    strength: float
    """Ratio of changepoint gap to background residual. Higher = more significant."""

    rss_reduction: float
    """Fractional reduction in RSS from single-line to two-line fit."""

    ci_lower: int
    """Bootstrap lower bound on changepoint index."""

    ci_upper: int
    """Bootstrap upper bound on changepoint index."""

    is_stable: bool
    """True if modal changepoint appears in >50% of bootstrap resamples."""

    frequency: float
    """Fraction of bootstrap resamples where modal changepoint was selected."""

    n_values: int
    """Number of values in the input sequence."""


def detect_spectral_changepoint(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    use_log: bool = False,
) -> ChangePointResult:
    """Detect a single changepoint in an ordered value sequence.

    Algorithm (Bai & Perron 1998, single break):
    1. For each candidate split k in [2, n-3], fit two OLS lines:
       - Line 1 on values[0:k+1] vs indices
       - Line 2 on values[k+1:] vs indices
    2. Pick k that minimizes total RSS (residual sum of squares).
    3. Strength = gap at changepoint / median background residual.
    4. Bootstrap: resample values (preserving order via index resampling),
       re-detect changepoint.

    Args:
        values: Ordered sequence (e.g. sorted var_top1, or singular values).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for CI on changepoint index.
        seed: RNG seed for reproducibility.
        use_log: If True, work on log(values) (for singular value spectra).

    Returns:
        ChangePointResult with changepoint index, strength, and stability.

    Raises:
        ValueError: If fewer than 5 values (need at least 2 per segment + 1 split).
    """
    n = len(values)
    if n < 5:
        raise ValueError(f"Need >= 5 values for changepoint detection (got {n})")

    work = values
    if use_log:
        # Filter non-positive values for log transform
        _eps_f64 = math.ldexp(1.0, -52)
        work = [math.log(max(v, _eps_f64)) for v in values]

    k_best, rss_best = _find_best_split(work)

    # Strength: ratio of the gap at changepoint to the background residual level
    rss_single = _single_line_rss(work)
    rss_reduction = 1.0 - rss_best / rss_single if rss_single > 0 else 0.0

    # Gap at changepoint: predicted value difference at the split
    strength = _changepoint_strength(work, k_best)

    # Residual bootstrap: fit two-segment model, compute residuals,
    # shuffle residuals and add back to fitted values, re-detect changepoint.
    # This preserves the ordered structure while testing noise sensitivity.
    x_all = list(range(n))
    left_alpha, left_beta, _ = fit_linear(
        [float(i) for i in x_all[: k_best + 1]], work[: k_best + 1]
    )
    right_alpha, right_beta, _ = fit_linear(
        [float(i) for i in x_all[k_best + 1 :]], work[k_best + 1 :]
    )
    fitted = []
    residuals = []
    for i in range(n):
        if i <= k_best:
            f = left_alpha * i + left_beta
        else:
            f = right_alpha * i + right_beta
        fitted.append(f)
        residuals.append(work[i] - f)

    rng = random.Random(seed if seed is not None else 42)
    bootstrap_ks: list[int] = []
    for _ in range(n_bootstrap):
        # Shuffle residuals and add back to fitted values
        shuffled_res = residuals[:]
        rng.shuffle(shuffled_res)
        resampled = [fitted[i] + shuffled_res[i] for i in range(n)]
        try:
            bk, _ = _find_best_split(resampled)
            bootstrap_ks.append(bk)
        except ValueError:
            continue

    if not bootstrap_ks:
        return ChangePointResult(
            k=k_best,
            strength=strength,
            rss_reduction=rss_reduction,
            ci_lower=k_best,
            ci_upper=k_best,
            is_stable=False,
            frequency=0.0,
            n_values=n,
        )

    # Modal changepoint and its frequency
    from collections import Counter

    counts = Counter(bootstrap_ks)
    modal_k, modal_count = counts.most_common(1)[0]
    frequency = modal_count / len(bootstrap_ks)

    bootstrap_ks.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2 * len(bootstrap_ks)))
    hi_idx = min(len(bootstrap_ks) - 1, int((1 - alpha / 2) * len(bootstrap_ks)) - 1)
    ci_lower = bootstrap_ks[lo_idx]
    ci_upper = bootstrap_ks[hi_idx]

    return ChangePointResult(
        k=k_best,
        strength=strength,
        rss_reduction=rss_reduction,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_stable=frequency > 0.5,
        frequency=frequency,
        n_values=n,
    )


def _find_best_split(values: list[float]) -> tuple[int, float]:
    """Find the split index k that minimizes total RSS of two-segment fit.

    Returns (k, total_rss).
    """
    n = len(values)
    if n < 5:
        raise ValueError(f"Need >= 5 values (got {n})")

    x_all = list(range(n))
    best_k = 2
    best_rss = float("inf")

    # Try each split: left = [0, k], right = (k, n-1]
    # Minimum segment size = 2 for OLS
    for k in range(1, n - 2):
        left_x = x_all[: k + 1]
        left_y = values[: k + 1]
        right_x = x_all[k + 1 :]
        right_y = values[k + 1 :]

        if len(left_x) < 2 or len(right_x) < 2:
            continue

        rss_left = _segment_rss(left_x, left_y)
        rss_right = _segment_rss(right_x, right_y)
        total_rss = rss_left + rss_right

        if total_rss < best_rss:
            best_rss = total_rss
            best_k = k

    return best_k, best_rss


def _segment_rss(x: list[float], y: list[float]) -> float:
    """Residual sum of squares for OLS fit on segment."""
    alpha, beta, _ = fit_linear([float(xi) for xi in x], y)
    return sum((yi - (alpha * xi + beta)) ** 2 for xi, yi in zip(x, y))


def _single_line_rss(values: list[float]) -> float:
    """RSS of a single OLS line through all values."""
    x = [float(i) for i in range(len(values))]
    return _segment_rss(x, values)


def _changepoint_strength(values: list[float], k: int) -> float:
    """Ratio of gap at changepoint to median absolute residual.

    Strength measures how much the changepoint stands out above
    the background noise level. Higher = more significant.
    """
    n = len(values)
    x = [float(i) for i in range(n)]

    # Fit two-segment model
    left_alpha, left_beta, _ = fit_linear(x[: k + 1], values[: k + 1])
    right_alpha, right_beta, _ = fit_linear(x[k + 1 :], values[k + 1 :])

    # Gap at changepoint: difference between right prediction and left prediction at k
    left_pred_at_k = left_alpha * k + left_beta
    right_pred_at_k = right_alpha * (k + 1) + right_beta
    gap = abs(right_pred_at_k - left_pred_at_k)

    # Median absolute residual (background noise level)
    residuals = []
    for i in range(k + 1):
        pred = left_alpha * i + left_beta
        residuals.append(abs(values[i] - pred))
    for i in range(k + 1, n):
        pred = right_alpha * i + right_beta
        residuals.append(abs(values[i] - pred))

    residuals.sort()
    if not residuals:
        return 0.0
    median_residual = residuals[len(residuals) // 2]

    if median_residual <= 0:
        return float("inf") if gap > 0 else 0.0
    return gap / median_residual
