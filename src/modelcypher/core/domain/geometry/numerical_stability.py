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

"""Dynamic numerical stability utilities.

All epsilons and tolerances are derived from tensor precision, not arbitrary constants.
Use these functions instead of hardcoded values like 1e-8 or 1e-10.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def machine_epsilon(backend: Backend, array: Array) -> float:
    """Get machine epsilon for the array's dtype.

    This is the smallest value such that 1.0 + epsilon != 1.0.
    Use for general numerical stability in comparisons.
    """
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for safe division operations.

    Scaled up from machine epsilon to provide numerical headroom.
    Use when dividing to prevent division by zero.
    """
    return backend.finfo(array.dtype).eps * 1e3


def regularization_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for matrix regularization.

    Uses sqrt(eps) which is the standard choice for regularization
    in numerical linear algebra (Tikhonov regularization, ridge).
    """
    return math.sqrt(backend.finfo(array.dtype).eps)


def condition_threshold(backend: Backend, array: Array) -> float:
    """Get threshold for condition number checks.

    Returns 1/eps, the inverse of machine epsilon.
    Matrices with condition number above this are numerically singular.
    """
    return 1.0 / backend.finfo(array.dtype).eps


def svd_rank_threshold(backend: Backend, array: Array, max_dim: int) -> float:
    """Get threshold for determining numerical rank from SVD.

    Uses the standard formula: max_dim * eps * largest_singular_value.
    Singular values below this threshold are considered zero.

    Args:
        backend: The compute backend.
        array: The array being decomposed (for dtype).
        max_dim: Maximum dimension of the matrix.

    Returns:
        Threshold scaled by matrix size and precision.
    """
    eps = backend.finfo(array.dtype).eps
    return float(max_dim) * eps


def tiny_value(backend: Backend, array: Array) -> float:
    """Get the smallest positive usable number for the dtype.

    Use as a floor when values must remain positive.
    """
    return backend.finfo(array.dtype).tiny


def safe_log_epsilon(backend: Backend, array: Array) -> float:
    """Get epsilon for safe logarithm operations.

    Uses tiny value to prevent log(0) while maintaining precision.
    """
    return backend.finfo(array.dtype).tiny
