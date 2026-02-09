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

"""Data-derived early stopping for geometric training.

Pure Python — zero framework dependencies.

The stopping criterion compares epoch-windowed loss means against the
standard error of the difference, which is a MEASUREMENT of the data's
own noise floor. No fixed threshold constants.

Machine epsilon is retained as a lower bound for the degenerate case
where per-batch variance -> 0 (perfectly uniform dataset).
"""

from __future__ import annotations

import math

# sqrt(eps) for float32: numerical significance threshold
_SQRT_EPS = math.sqrt(math.ldexp(1.0, -23))  # ~3.45e-4


def check_loss_stable(
    losses: list[tuple[int, float, float]],
    window: int | None = None,
    numeric_floor: float = _SQRT_EPS,
) -> tuple[bool, float]:
    """Check if loss has converged: |delta_epoch_mean| < standard error of the difference.

    Compares mean of last ``window`` losses to mean of previous ``window`` losses.
    If ``window`` is None, it is derived from the observed trajectory by splitting
    the available history into two equal windows.
    Threshold is the standard error of the difference of the two epoch means:

        SE_diff = sqrt(var_recent/N + var_earlier/N)

    This is a MEASUREMENT of the data's own noise floor, not a fixed constant.
    On homogeneous data (40 samples), per-batch variance is low -> tight threshold.
    On diverse data (724 samples), per-batch variance is high -> threshold accounts
    for batch-to-batch noise that would mask convergence.

    Args:
        losses: List of (iteration, loss_value, tokens_per_sec) tuples.
        window: Number of recent entries to compare. If None, derived from
            the available trajectory length.
        numeric_floor: Numerical lower bound for distinguishability.

    Returns:
        (is_stable, threshold) -- threshold is the SE_diff actually used.
    """
    resolved_window = window if window is not None else len(losses) // 2
    if resolved_window < 2:
        return False, 0.0

    if len(losses) < 2 * resolved_window:
        return False, 0.0

    recent = [entry[1] for entry in losses[-resolved_window:]]
    earlier = [entry[1] for entry in losses[-2 * resolved_window : -resolved_window]]

    n = len(recent)
    mean_recent = sum(recent) / n
    mean_earlier = sum(earlier) / n

    # Variance of each epoch's per-batch losses
    var_recent = sum((x - mean_recent) ** 2 for x in recent) / n
    var_earlier = sum((x - mean_earlier) ** 2 for x in earlier) / n

    # Standard error of the difference of two means
    se_diff = math.sqrt((var_recent + var_earlier) / n)

    # Use data-derived SE, but never below machine epsilon
    threshold = max(se_diff, numeric_floor)

    return abs(mean_recent - mean_earlier) < threshold, threshold


__all__ = [
    "_SQRT_EPS",
    "check_loss_stable",
]
