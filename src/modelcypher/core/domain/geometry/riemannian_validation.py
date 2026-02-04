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

"""Vectorized validation helpers for Riemannian geometry operations.

All operations use backend ops instead of Python loops for GPU efficiency.

Core validation functions (count_nan, count_inf, count_nonfinite, validate_array_numerics)
are imported from _primitives.validation - no duplicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import core validation from canonical location
from modelcypher.core.domain.geometry.numerical_stability import (
    ArrayNumerics,
    count_inf,
    count_nan,
    count_nonfinite,
    validate_array_numerics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def count_finite(arr: "Array", backend: "Backend") -> int:
    """Count finite values in array using vectorized backend operation.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        Count of finite (not NaN, not Inf) values in the array.
    """
    finite_mask = backend.isfinite(arr)
    count = backend.sum(finite_mask)
    backend.eval(count)
    return int(float(backend.to_scalar(count)))


def has_nan(arr: "Array", backend: "Backend") -> bool:
    """Check if array contains any NaN values.

    More efficient than count_nan() > 0 for early exit.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if any NaN values present.
    """
    nan_mask = backend.isnan(arr)
    any_nan = backend.max(nan_mask)  # max of bool array = any True
    backend.eval(any_nan)
    return bool(backend.to_scalar(any_nan))


def has_inf(arr: "Array", backend: "Backend") -> bool:
    """Check if array contains any infinite values.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if any +/- infinity values present.
    """
    inf_mask = backend.isinf(arr)
    any_inf = backend.max(inf_mask)
    backend.eval(any_inf)
    return bool(backend.to_scalar(any_inf))


def all_finite(arr: "Array", backend: "Backend") -> bool:
    """Check if all values in array are finite.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if all values are finite (no NaN, no Inf).
    """
    finite_mask = backend.isfinite(arr)
    all_ok = backend.min(finite_mask)  # min of bool array = all True
    backend.eval(all_ok)
    return bool(backend.to_scalar(all_ok))


def derive_k_neighbors(points: "Array", backend: "Backend") -> int:
    """Derive k for k-NN graph from geodesic connectivity.

    Returns the minimum k that yields a connected k-NN graph. This avoids
    Euclidean-distance heuristics and ties k directly to the manifold's
    intrinsic connectivity.
    """
    from modelcypher.core.domain.geometry.riemannian_core import (
        _get_riemannian_geometry,
    )

    n = int(points.shape[0])
    if n <= 1:
        return 0
    if n == 2:
        return 1

    rg = _get_riemannian_geometry(backend)
    k_min, _, _ = rg._minimum_connected_k(points)
    return max(1, min(k_min, n - 1))


def safe_arithmetic_mean(values: "list[float] | tuple[float, ...]") -> float:
    """Compute arithmetic mean, returning 0.0 for empty sequences.

    This is a simple utility for scalar values. For embeddings on
    curved manifolds, use frechet_mean() instead.

    Args:
        values: Sequence of float values.

    Returns:
        Arithmetic mean, or 0.0 if empty.
    """
    if not values:
        return 0.0
    vals = list(values) if not isinstance(values, list) else values
    return sum(vals) / len(vals)


def set_matrix_element(
    backend: "Backend",
    matrix: "Array",
    i: int,
    j: int,
    value: float,
) -> "Array":
    """Set a single element in a matrix using backend ops.

    This is inefficient for many updates but works on any backend.
    For building sparse adjacency, we accept this cost to stay on GPU.

    Args:
        backend: Backend instance.
        matrix: Matrix to update [n, m].
        i: Row index.
        j: Column index.
        value: Value to set.

    Returns:
        Updated matrix with element (i, j) set to value.
    """
    # Create a mask: 1 at (i,j), 0 elsewhere
    n = matrix.shape[0]
    m = matrix.shape[1]

    # Create row and column index arrays
    row_idx = backend.arange(n)
    col_idx = backend.arange(m)

    # Broadcast to create masks
    row_mask = row_idx == i  # [n]
    col_mask = col_idx == j  # [m]

    # Outer product for 2D mask
    row_mask_2d = backend.reshape(row_mask, (n, 1))
    col_mask_2d = backend.reshape(col_mask, (1, m))

    # Element-wise AND via multiplication (both are boolean-like 0/1)
    # Convert to float for multiplication
    mask_dtype = matrix.dtype if hasattr(matrix, "dtype") else backend.array([1.0]).dtype
    row_float = backend.astype(row_mask_2d, mask_dtype)
    col_float = backend.astype(col_mask_2d, mask_dtype)
    mask = row_float * col_float  # [n, m], 1.0 at (i,j), 0.0 elsewhere

    # Update: matrix * (1 - mask) + value * mask
    result = matrix * (1.0 - mask) + value * mask
    return result


__all__ = [
    "ArrayNumerics",
    "count_nan",
    "count_inf",
    "count_finite",
    "count_nonfinite",
    "has_nan",
    "has_inf",
    "all_finite",
    "derive_k_neighbors",
    "safe_arithmetic_mean",
    "set_matrix_element",
    "validate_array_numerics",
]
