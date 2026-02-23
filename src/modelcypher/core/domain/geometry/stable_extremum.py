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

"""Bootstrap-stable extremum detection.

Tests whether the argmin (or argmax) of a vector is stable under bootstrap
resampling. If the minimum location shifts across resamples, it is not a
reliable geometric event and should be marked INCONCLUSIVE.

Also provides inflection detection: find where the first derivative changes
sign (local minimum in the derivative), tested for bootstrap stability.

Pure Python — zero framework dependencies.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class StableExtremumResult:
    """Result of bootstrap-stable extremum detection."""

    index: int
    """Index of the extremum in the input vector."""

    value: float
    """Value at the extremum."""

    frequency: float
    """Fraction of bootstrap resamples selecting this index."""

    is_stable: bool
    """True if frequency > 0.5 (modal extremum in majority of resamples)."""

    ci_range: tuple[int, int]
    """(min_index, max_index) observed across bootstrap resamples."""

    n_values: int
    """Length of input vector."""


def find_stable_minimum(
    values: list[float],
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> StableExtremumResult:
    """Find the argmin of values and test bootstrap stability.

    Algorithm:
    1. Find argmin of the input vector.
    2. Bootstrap: resample values with replacement (preserving index
       correspondence), find argmin of each resample.
    3. If modal argmin appears in >50% of resamples, it is stable.

    The key insight: per-layer measurements (like intrinsic dimension)
    have measurement noise. If the minimum is not robust to resampling,
    it's not a reliable geometric event.

    Args:
        values: Per-index measurements (e.g. per-layer intrinsic dimension).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        StableExtremumResult with index, value, frequency, stability.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot find minimum of empty list")

    n = len(values)
    min_idx = _argmin(values)

    if n == 1:
        return StableExtremumResult(
            index=0,
            value=values[0],
            frequency=1.0,
            is_stable=True,
            ci_range=(0, 0),
            n_values=1,
        )

    rng = random.Random(seed if seed is not None else 42)
    bootstrap_mins: list[int] = []

    for _ in range(n_bootstrap):
        # Pairs bootstrap: resample (index, value) pairs with replacement.
        # Then find which ORIGINAL layer index has the minimum value among
        # the resampled layers. This preserves the (layer, measurement)
        # pairing — we're testing "if I resampled which layers to include,
        # would the same layer still have the minimum?"
        sampled_indices = [rng.randint(0, n - 1) for _ in range(n)]
        min_layer = min(sampled_indices, key=lambda i: values[i])
        bootstrap_mins.append(min_layer)

    counts = Counter(bootstrap_mins)
    modal_idx, modal_count = counts.most_common(1)[0]
    frequency = modal_count / len(bootstrap_mins)

    all_indices = sorted(set(bootstrap_mins))
    ci_range = (all_indices[0], all_indices[-1])

    # Use the actual argmin (not the modal bootstrap), but report stability
    # based on whether the actual argmin is also the modal bootstrap result.
    # If they differ, the minimum is ambiguous.
    actual_frequency = counts.get(min_idx, 0) / len(bootstrap_mins)

    return StableExtremumResult(
        index=min_idx,
        value=values[min_idx],
        frequency=actual_frequency,
        is_stable=actual_frequency > 0.5,
        ci_range=ci_range,
        n_values=n,
    )


def find_stable_inflection(
    values: list[float],
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> StableExtremumResult:
    """Find the inflection point (derivative sign change) and test stability.

    An inflection in a per-layer trajectory marks a transition event —
    e.g. where spectral entropy stops decreasing and starts increasing
    (the re-expansion point after the highway).

    Algorithm:
    1. Compute first differences: d[i] = values[i+1] - values[i].
    2. Find where d changes sign (negative → positive = local minimum).
    3. If multiple sign changes, pick the one with largest absolute
       second difference (sharpest inflection).
    4. Bootstrap stability test.

    Args:
        values: Per-index measurements (e.g. per-layer spectral entropy).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        StableExtremumResult with inflection index and stability.

    Raises:
        ValueError: If fewer than 3 values.
    """
    n = len(values)
    if n < 3:
        raise ValueError(f"Need >= 3 values for inflection detection (got {n})")

    inflection_idx = _find_inflection(values)

    rng = random.Random(seed if seed is not None else 42)
    bootstrap_inflections: list[int] = []

    for _ in range(n_bootstrap):
        resampled = [values[rng.randint(0, n - 1)] for _ in range(n)]
        # Sort to preserve the trajectory structure (monotone segments)
        # No — don't sort. The inflection depends on the ordering.
        # Instead, resample the measurement noise: for each position,
        # draw from nearby positions.
        # Use block bootstrap: resample contiguous blocks to preserve local structure.
        block_size = max(1, n // 4)
        block_start = rng.randint(0, n - 1)
        resampled_trajectory = []
        for i in range(n):
            # Draw from a window around position i
            window_lo = max(0, i - block_size // 2)
            window_hi = min(n - 1, i + block_size // 2)
            resampled_trajectory.append(values[rng.randint(window_lo, window_hi)])

        bi = _find_inflection(resampled_trajectory)
        if bi >= 0:
            bootstrap_inflections.append(bi)

    if not bootstrap_inflections:
        return StableExtremumResult(
            index=max(0, inflection_idx),
            value=values[max(0, inflection_idx)],
            frequency=0.0,
            is_stable=False,
            ci_range=(0, n - 1),
            n_values=n,
        )

    counts = Counter(bootstrap_inflections)
    actual_frequency = counts.get(inflection_idx, 0) / len(bootstrap_inflections)

    all_indices = sorted(set(bootstrap_inflections))
    ci_range = (all_indices[0], all_indices[-1])

    return StableExtremumResult(
        index=max(0, inflection_idx),
        value=values[max(0, inflection_idx)],
        frequency=actual_frequency,
        is_stable=actual_frequency > 0.5,
        ci_range=ci_range,
        n_values=n,
    )


def _argmin(values: list[float]) -> int:
    """Index of minimum value. Ties broken by first occurrence."""
    min_val = values[0]
    min_idx = 0
    for i, v in enumerate(values):
        if v < min_val:
            min_val = v
            min_idx = i
    return min_idx


def _find_inflection(values: list[float]) -> int:
    """Find sharpest inflection (derivative sign change: negative → positive).

    Returns index of the inflection, or -1 if none found.
    """
    n = len(values)
    if n < 3:
        return -1

    # First differences
    diffs = [values[i + 1] - values[i] for i in range(n - 1)]

    # Find sign changes (negative → positive = local minimum in trajectory)
    best_idx = -1
    best_sharpness = 0.0

    for i in range(len(diffs) - 1):
        if diffs[i] < 0 and diffs[i + 1] > 0:
            # Sign change at index i+1
            sharpness = abs(diffs[i + 1] - diffs[i])
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_idx = i + 1

    return best_idx
