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

"""Minimum Spanning Tree computation for point cloud graphs.

Uses Prim's algorithm with Backend operations for GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import infinity_threshold

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class MSTEdge:
    """An edge in the minimum spanning tree."""

    i: int
    j: int
    weight: float


@dataclass(frozen=True)
class MSTResult:
    """Result of MST computation."""

    edges: list[MSTEdge]
    component_count: int
    max_degree: int


def compute_mst(
    dist_matrix: "Array",
    backend: "Backend",
) -> MSTResult:
    """Compute minimum spanning tree (or forest) using Prim's algorithm.

    Args:
        dist_matrix: Pairwise distance matrix [n, n].
        backend: Backend for tensor operations.

    Returns:
        MSTResult with edges, component count, and max vertex degree.
    """
    n = int(dist_matrix.shape[0])

    if n == 0:
        return MSTResult(edges=[], component_count=0, max_degree=0)

    if n == 1:
        return MSTResult(edges=[], component_count=1, max_degree=0)

    inf_thresh = infinity_threshold(backend, dist_matrix)
    inf_val = float(inf_thresh * 2.0)

    # Add large value to diagonal to exclude self-loops
    eye = backend.eye(n)
    dist_no_self = dist_matrix + eye * inf_val
    backend.eval(dist_no_self)

    index = backend.arange(n, dtype="int32")
    ones_int = backend.ones((n,), dtype="int32")
    inf_vec = backend.full((n,), inf_val)

    selected = backend.zeros((n,), dtype="int32")
    min_dist = inf_vec
    min_parent = backend.full((n,), -1, dtype="int32")

    edges: list[MSTEdge] = []
    degrees = [0] * n
    component_count = 0

    while True:
        # Check if all vertices are selected
        unselected_mask = selected == 0
        unselected_count_arr = backend.sum(backend.astype(unselected_mask, "int32"))
        backend.eval(unselected_count_arr)
        if int(backend.to_scalar(unselected_count_arr)) == 0:
            break

        # Start new component from smallest unselected index
        idx_candidates = backend.where(
            unselected_mask, index, backend.full(index.shape, n, dtype="int32")
        )
        root_idx_arr = backend.argmin(idx_candidates)
        backend.eval(root_idx_arr)
        root_idx = int(backend.to_scalar(root_idx_arr))
        component_count += 1

        # Mark root as selected
        selected = backend.where(index == root_idx, ones_int, selected)

        # Initialize distances from root
        root_row = backend.take(dist_no_self, backend.array([root_idx], dtype="int32"), axis=0)
        root_row = backend.reshape(root_row, (n,))
        unselected_mask = selected == 0
        min_dist = backend.where(unselected_mask, root_row, inf_vec)
        min_parent = backend.where(
            unselected_mask,
            backend.full(min_parent.shape, root_idx, dtype="int32"),
            min_parent,
        )

        # Grow tree from root using Prim's algorithm
        while True:
            masked = backend.where(selected == 0, min_dist, inf_vec)
            min_val_arr = backend.min(masked)
            backend.eval(min_val_arr)
            min_val = float(backend.to_scalar(min_val_arr))

            if min_val >= inf_thresh:
                # No more reachable vertices
                break

            # Find vertex with minimum distance
            idx_arr = backend.argmin(masked)
            backend.eval(idx_arr)
            idx = int(backend.to_scalar(idx_arr))

            # Get parent of this vertex
            parent_arr = backend.take(min_parent, backend.array([idx], dtype="int32"), axis=0)
            backend.eval(parent_arr)
            parent_idx = int(backend.to_scalar(parent_arr))
            if parent_idx < 0:
                break

            # Record MST edge
            i, j = (parent_idx, idx) if parent_idx < idx else (idx, parent_idx)
            edges.append(MSTEdge(i=i, j=j, weight=min_val))
            degrees[parent_idx] += 1
            degrees[idx] += 1

            # Mark vertex as selected
            selected = backend.where(index == idx, ones_int, selected)

            # Update minimum distances for remaining vertices
            row = backend.take(dist_no_self, backend.array([idx], dtype="int32"), axis=0)
            row = backend.reshape(row, (n,))
            better = backend.astype(row < min_dist, "int32") * backend.astype(
                selected == 0, "int32"
            )
            min_dist = backend.where(better > 0, row, min_dist)
            min_parent = backend.where(
                better > 0,
                backend.full(min_parent.shape, idx, dtype="int32"),
                min_parent,
            )

    max_degree = max(degrees) if degrees else 0

    return MSTResult(
        edges=edges,
        component_count=component_count,
        max_degree=max_degree,
    )


def minimum_k_from_mst(
    dist_matrix: "Array",
    backend: "Backend",
) -> tuple[int, MSTResult]:
    """Find minimum k for k-NN graph connectivity from MST.

    The maximum degree in the MST gives the minimum k needed for
    the k-NN graph to be connected (assuming symmetric edges).

    Args:
        dist_matrix: Pairwise distance matrix [n, n].
        backend: Backend for tensor operations.

    Returns:
        Tuple of (minimum k, MSTResult).
    """
    mst_result = compute_mst(dist_matrix, backend)

    # For a connected graph, the MST max degree gives minimum k
    # But we need at least 1 neighbor
    k = max(1, mst_result.max_degree)

    return k, mst_result
