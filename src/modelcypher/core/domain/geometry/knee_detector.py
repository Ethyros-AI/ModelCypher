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

"""Knee detection via discrete curvature extremum.

Given a monotone curve y(x), the knee is the point of maximum absolute
discrete curvature (second derivative). This identifies where the curve
transitions from one regime to another — e.g. where coherence transitions
from "inside the manifold" to "outside".

Bootstrap resampling on y-values provides a CI on the knee location.

Pure Python — zero framework dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KneeResult:
    """Result of knee detection on a monotone curve."""

    x_knee: float
    """X-coordinate of the knee point."""

    y_knee: float
    """Y-coordinate at the knee point."""

    curvature: float
    """Absolute discrete curvature at the knee."""

    ci_lower: float
    """Bootstrap lower bound on x_knee."""

    ci_upper: float
    """Bootstrap upper bound on x_knee."""

    is_stable: bool
    """True if modal knee index appears in >50% of bootstrap resamples."""

    frequency: float
    """Fraction of resamples selecting the modal knee index."""

    n_points: int
    """Number of points in the input curve."""


def detect_knee(
    x: list[float],
    y: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> KneeResult:
    """Find the knee point of a curve via maximum absolute curvature.

    Algorithm:
    1. Normalize x and y to [0, 1] to make curvature scale-invariant.
    2. Compute discrete second derivative at each interior point.
    3. Knee = point with maximum absolute second derivative.
    4. Bootstrap: add noise to y-values (resample residuals from local
       linear fit), re-detect knee.

    Args:
        x: X-coordinates (must be sorted, strictly increasing).
        y: Y-coordinates (same length as x).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for CI.
        seed: RNG seed for reproducibility.

    Returns:
        KneeResult with knee location, curvature, CI, and stability.

    Raises:
        ValueError: If fewer than 4 points (need interior points for 2nd deriv).
    """
    n = len(x)
    if n < 4 or len(y) != n:
        raise ValueError(
            f"Need >= 4 points with matching x, y lengths (got {n}, {len(y)})"
        )

    knee_idx = _find_knee_index(x, y)

    # Bootstrap stability: resample with replacement from y-values at each x
    rng = random.Random(seed if seed is not None else 42)
    bootstrap_indices: list[int] = []
    for _ in range(n_bootstrap):
        # Resample y-values with replacement (paired bootstrap on residuals)
        resampled_y = [y[rng.randint(0, n - 1)] for _ in range(n)]
        # Sort resampled_y to maintain monotonicity assumption
        resampled_y.sort(reverse=(y[0] > y[-1]) if n > 1 else False)
        try:
            bk = _find_knee_index(x, resampled_y)
            bootstrap_indices.append(bk)
        except (ValueError, ZeroDivisionError):
            continue

    if not bootstrap_indices:
        return KneeResult(
            x_knee=x[knee_idx],
            y_knee=y[knee_idx],
            curvature=_curvature_at(x, y, knee_idx),
            ci_lower=x[knee_idx],
            ci_upper=x[knee_idx],
            is_stable=False,
            frequency=0.0,
            n_points=n,
        )

    from collections import Counter

    counts = Counter(bootstrap_indices)
    modal_idx, modal_count = counts.most_common(1)[0]
    frequency = modal_count / len(bootstrap_indices)

    # CI on index → map back to x
    bootstrap_indices.sort()
    alpha = 1.0 - confidence
    lo = max(0, int(alpha / 2 * len(bootstrap_indices)))
    hi = min(
        len(bootstrap_indices) - 1,
        int((1 - alpha / 2) * len(bootstrap_indices)) - 1,
    )
    ci_lower_idx = bootstrap_indices[lo]
    ci_upper_idx = bootstrap_indices[hi]

    return KneeResult(
        x_knee=x[knee_idx],
        y_knee=y[knee_idx],
        curvature=_curvature_at(x, y, knee_idx),
        ci_lower=x[ci_lower_idx],
        ci_upper=x[ci_upper_idx],
        is_stable=frequency > 0.5,
        frequency=frequency,
        n_points=n,
    )


def _find_knee_index(x: list[float], y: list[float]) -> int:
    """Find index with maximum absolute discrete curvature."""
    n = len(x)
    best_idx = 1
    best_curv = 0.0

    # Normalize to [0,1] for scale-invariant curvature
    x_min, x_max = x[0], x[-1]
    y_min = min(y)
    y_max = max(y)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    xn = [(xi - x_min) / x_range for xi in x]
    yn = [(yi - y_min) / y_range for yi in y]

    for i in range(1, n - 1):
        curv = abs(_curvature_at(xn, yn, i))
        if curv > best_curv:
            best_curv = curv
            best_idx = i

    return best_idx


def _curvature_at(x: list[float], y: list[float], i: int) -> float:
    """Discrete curvature (second derivative) at index i.

    Uses the standard central difference formula:
    f''(x_i) ≈ (f(x_{i+1}) - 2*f(x_i) + f(x_{i-1})) / h^2
    adjusted for non-uniform spacing.
    """
    if i <= 0 or i >= len(x) - 1:
        return 0.0

    h1 = x[i] - x[i - 1]
    h2 = x[i + 1] - x[i]

    if h1 <= 0 or h2 <= 0:
        return 0.0

    # Second derivative for non-uniform spacing:
    # f'' ≈ 2 * (f_{i+1}/(h2*(h1+h2)) - f_i/(h1*h2) + f_{i-1}/(h1*(h1+h2)))
    return 2.0 * (
        y[i + 1] / (h2 * (h1 + h2))
        - y[i] / (h1 * h2)
        + y[i - 1] / (h1 * (h1 + h2))
    )
