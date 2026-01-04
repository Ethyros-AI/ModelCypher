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

"""Hungarian Algorithm (Kuhn-Munkres) for optimal bipartite assignment.

This module provides the canonical implementation of the Hungarian algorithm
for all optimal assignment problems in ModelCypher. Use this module instead
of implementing the algorithm locally.

Features:
- GPU-accelerated implementation via Backend protocol
- Session-scoped caching for repeated calls with same cost matrix
- O(N³) time complexity

Usage:
    from modelcypher.core.domain.geometry.hungarian import hungarian_assignment

    # With Backend array (GPU)
    assignment = hungarian_assignment(cost_matrix, backend)

    # With Python list (CPU fallback for small matrices)
    assignment = hungarian_assignment_list(cost_matrix)

References:
    - Kuhn, H. W. (1955). "The Hungarian Method for the Assignment Problem."
      Naval Research Logistics Quarterly 2(1-2):83-97.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import is_finite

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Session-scoped cache for Hungarian results
# Key: (cost_matrix_hash, n) -> assignment array
_hungarian_cache: dict[tuple[int, int], list[int]] = {}
_CACHE_MAX_SIZE = 128


def _hash_cost_matrix(cost: "Array", backend: "Backend") -> int:
    """Compute hash of cost matrix for caching.

    Uses multiple statistics to reduce collisions between matrices
    with different structure but similar aggregate values.
    """
    flat = backend.reshape(cost, (-1,))
    n = int(cost.shape[0])

    # Combine multiple statistics for collision resistance
    sum_arr = backend.sum(flat)
    sum_sq_arr = backend.sum(flat * flat)
    # Sample diagonal and corners for structure
    diag = backend.diag(cost)
    diag_sum = backend.sum(diag)
    # First and last elements provide positional info
    first_elem = flat[0] if flat.shape[0] > 0 else backend.array([0.0])
    last_elem = flat[-1] if flat.shape[0] > 0 else backend.array([0.0])
    backend.eval(sum_arr, sum_sq_arr, diag_sum, first_elem, last_elem)

    hash_tuple = (
        float(backend.to_scalar(sum_arr)),
        float(backend.to_scalar(sum_sq_arr)),
        float(backend.to_scalar(diag_sum)),
        float(backend.to_scalar(first_elem)),
        float(backend.to_scalar(last_elem)),
        n,
    )
    return hash(hash_tuple)


def hungarian_assignment(
    cost_matrix: "Array",
    backend: "Backend | None" = None,
    use_cache: bool = True,
) -> "Array":
    """Optimal bipartite assignment using Hungarian algorithm (GPU-accelerated).

    Finds the minimum-cost assignment in O(N³) time.
    For maximize similarity, caller should convert: cost = max_sim - sim.

    Args:
        cost_matrix: Square [N, N] cost matrix where cost[i][j] = cost of assigning i→j.
        backend: Backend for array operations (auto-detected if None).
        use_cache: Whether to use session-scoped caching for repeated calls.

    Returns:
        Assignment array where assignment[i] = j means source i maps to target j.
    """
    b = backend or get_default_backend()
    cost = b.array(cost_matrix, dtype="float32")
    b.eval(cost)

    n = int(cost.shape[0])
    if n == 0:
        return b.zeros((0,), dtype="int32")

    # Check cache
    if use_cache:
        cache_key = _hash_cost_matrix(cost, b)
        if cache_key in _hungarian_cache:
            cached = _hungarian_cache[cache_key]
            return b.array(cached, dtype="int32")

    # Run algorithm
    result = _hungarian_backend_impl(cost, b)
    b.eval(result)

    # Cache result
    if use_cache:
        if len(_hungarian_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest entries (FIFO)
            keys_to_remove = list(_hungarian_cache.keys())[: _CACHE_MAX_SIZE // 2]
            for k in keys_to_remove:
                del _hungarian_cache[k]
        _hungarian_cache[cache_key] = [int(x) for x in b.tolist(result)]

    return result


def hungarian_assignment_list(cost_matrix: list[list[float]]) -> list[int]:
    """Optimal bipartite assignment using Hungarian algorithm (CPU/list-based).

    Use this for small matrices or when Backend is not available.
    For larger matrices, prefer hungarian_assignment() with Backend.

    Args:
        cost_matrix: Square [N, N] cost matrix where cost[i][j] = cost of assigning i→j.

    Returns:
        Assignment list where assignment[i] = j means source i maps to target j.
    """
    n = len(cost_matrix)
    if n == 0:
        return []

    INF = float("inf")

    # u[i] = potential for row i, v[j] = potential for column j
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    # p[j] = row assigned to column j (1-indexed, 0 = unassigned)
    p = [0] * (n + 1)
    # way[j] = previous column in augmenting path
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        # Start augmenting path from row i
        p[0] = i
        j0 = 0  # Current column (0 = virtual)
        minv = [INF] * (n + 1)  # minv[j] = min reduced cost to reach column j
        used = [False] * (n + 1)

        while p[j0] != 0:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    # Reduced cost from row i0 to column j
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Prevent infinite loop on impossible matching
            if delta == INF:
                break

            # Update potentials
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

        # Trace back augmenting path and update assignment
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # Extract assignment: result[i] = j means row i assigned to column j
    result = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0 and p[j] <= n:
            result[p[j] - 1] = j - 1

    return result


def _hungarian_backend_impl(cost: "Array", b: "Backend") -> "Array":
    """Hungarian algorithm implementation using Backend ops."""
    n = int(cost.shape[0])
    if n == 0:
        return b.zeros((0,), dtype="int32")

    size = n + 1
    u = b.zeros((size,), dtype="float32")
    v = b.zeros((size,), dtype="float32")
    p = b.zeros((size,), dtype="int32")
    way = b.zeros((n,), dtype="int32")

    col_idx = b.arange(1, n + 1, dtype="int32")
    row_idx = b.arange(size, dtype="int32")
    row_idx_matrix = b.reshape(row_idx, (1, size))

    inf_vec = b.full((n,), float("inf"), dtype="float32")
    zeros = b.zeros((n,), dtype="float32")
    ones = b.ones((n,), dtype="float32")

    for i in range(1, n + 1):
        p = b.where(row_idx == 0, b.full(p.shape, i, dtype="int32"), p)
        j0 = 0
        minv = inf_vec
        used = zeros
        way = b.zeros((n,), dtype="int32")

        while True:
            if j0 > 0:
                used = b.where(col_idx == j0, ones, used)

            p_j0 = b.take(p, b.array([j0], dtype="int32"), axis=0)
            b.eval(p_j0)
            i0 = int(b.to_scalar(p_j0))
            if i0 == 0:
                break

            row = b.take(cost, b.array([i0 - 1], dtype="int32"), axis=0)
            row = b.reshape(row, (n,))
            u_i0 = b.take(u, b.array([i0], dtype="int32"), axis=0)
            v_cols = b.take(v, col_idx, axis=0)
            cur = row - u_i0 - v_cols

            unused_mask = used == 0
            update = unused_mask * (cur < minv)
            minv = b.where(update, cur, minv)
            way = b.where(update, b.full(way.shape, j0, dtype="int32"), way)

            masked_minv = b.where(unused_mask, minv, inf_vec)
            delta = b.min(masked_minv)
            b.eval(delta)
            delta_val = float(b.to_scalar(delta))
            if not is_finite(delta_val, b):
                break

            j1_idx = b.argmin(masked_minv)
            b.eval(j1_idx)
            j1 = int(b.to_scalar(j1_idx)) + 1

            p_cols = b.take(p, col_idx, axis=0)
            p_cols_row = b.reshape(p_cols, (n, 1))
            used_col = b.reshape(used, (n, 1))
            row_match = b.astype(p_cols_row == row_idx_matrix, "float32")
            counts = b.sum(row_match * used_col, axis=0)

            p0 = b.take(p, b.array([0], dtype="int32"), axis=0)
            p0_mask = row_idx == p0
            p0_one = b.astype(p0_mask, "float32")
            u = u + (counts + p0_one) * delta

            v0 = b.take(v, b.array([0], dtype="int32"), axis=0)
            v0 = v0 - delta
            v_cols = v_cols - used * delta
            v = b.concatenate([v0, v_cols], axis=0)

            minv = b.where(unused_mask, minv - delta, minv)
            j0 = j1

        while True:
            if j0 == 0:
                break
            way_j0 = b.take(way, b.array([j0 - 1], dtype="int32"), axis=0)
            b.eval(way_j0)
            j1 = int(b.to_scalar(way_j0))
            p_j1 = b.take(p, b.array([j1], dtype="int32"), axis=0)
            b.eval(p_j1)
            p_val = int(b.to_scalar(p_j1))
            p = b.where(row_idx == j0, b.full(p.shape, p_val, dtype="int32"), p)
            j0 = j1

    cols = b.take(p, col_idx, axis=0)
    rows = cols - 1
    row_idx_small = b.arange(n, dtype="int32")
    rows_matrix = b.reshape(rows, (n, 1))
    row_idx_matrix = b.reshape(row_idx_small, (1, n))
    one_hot = b.astype(rows_matrix == row_idx_matrix, "float32")
    col_vals = b.reshape(col_idx - 1, (n, 1))
    result = b.sum(one_hot * col_vals, axis=0)
    return b.astype(result, "int32")


def clear_hungarian_cache() -> None:
    """Clear the session-scoped Hungarian result cache."""
    _hungarian_cache.clear()


__all__ = [
    "hungarian_assignment",
    "hungarian_assignment_list",
    "clear_hungarian_cache",
]
