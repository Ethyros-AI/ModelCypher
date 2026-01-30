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

"""NumPy-compatible dtype-derived epsilon utilities.

Provides the same functionality as epsilon_utils.py but for direct NumPy usage
without requiring a Backend instance. Use these in code that operates on NumPy
arrays directly (e.g., scipy.linalg operations).

All epsilons and tolerances are derived from array dtype, not arbitrary constants.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "np_machine_epsilon",
    "np_division_epsilon",
    "np_svd_rank_threshold",
    "np_safe_log_epsilon",
    "np_tiny_value",
]


def np_machine_epsilon(arr: NDArray) -> float:
    """Get machine epsilon for the array's dtype.

    This is the smallest value such that 1.0 + epsilon != 1.0.
    """
    return float(np.finfo(arr.dtype).eps)


def np_division_epsilon(arr: NDArray) -> float:
    """Get epsilon for safe division operations.

    Uses sqrt(eps) to provide numerical headroom.
    """
    eps = np.finfo(arr.dtype).eps
    return float(np.sqrt(eps))


def np_svd_rank_threshold(arr: NDArray, max_dim: int, largest_sv: float | None = None) -> float:
    """Get threshold for determining numerical rank from SVD.

    Uses the standard formula: max_dim * eps * largest_singular_value.
    Singular values below this threshold are considered zero.

    Args:
        arr: The array being decomposed (for dtype).
        max_dim: Maximum dimension of the matrix.
        largest_sv: Optional largest singular value. If None, returns max_dim * eps.

    Returns:
        Threshold scaled by matrix size and precision.
    """
    eps = np.finfo(arr.dtype).eps
    threshold = float(max_dim) * eps
    if largest_sv is not None:
        threshold *= largest_sv
    return threshold


def np_safe_log_epsilon(arr: NDArray) -> float:
    """Get epsilon for safe logarithm operations.

    Uses tiny value to prevent log(0) while maintaining precision.
    """
    return float(np.finfo(arr.dtype).tiny)


def np_tiny_value(arr: NDArray) -> float:
    """Get the smallest positive usable number for the dtype."""
    return float(np.finfo(arr.dtype).tiny)
