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

"""Riemannian interpolation mixin for geodesic path operations.

This mixin provides geodesic interpolation methods for RiemannianGeometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_inf,
    regularization_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

    from modelcypher.core.domain.geometry.riemannian_types import GeodesicDistanceResult


class RiemannianInterpolationMixin:
    """Mixin providing geodesic interpolation methods.

    Requires the base class to have:
    - self._backend: Backend instance
    - self.geodesic_distances(points, k_neighbors) method
    - self._find_nearest_point(points, query, geo_result) method
    """

    _backend: "Backend"

    def geodesic_interpolation(
        self,
        p1: "Array",
        p2: "Array",
        t: float,
        points_context: "Array | None" = None,
    ) -> "Array":
        """
        Interpolate along the geodesic between two points.

        For t=0 returns p1, for t=1 returns p2.

        If points_context is provided, uses the graph structure to find
        the geodesic path and selects the nearest path vertex for the
        requested arc length. The geodesic is the shortest path on the
        k-NN graph - exact for the discrete manifold.

        Algorithm:
            1. Project p1, p2 onto the discrete manifold (find nearest points)
            2. Reconstruct shortest path from geodesic distance matrix
            3. Compute cumulative arc lengths along path
            4. Select the nearest path vertex at arc length parameter t

        Args:
            p1: Start point [d]
            p2: End point [d]
            t: Interpolation parameter in [0, 1]
            points_context: Optional context point cloud for geodesic estimation

        Returns:
            Interpolated point [d]
        """
        backend = self._backend
        p1 = backend.array(p1)
        p2 = backend.array(p2)
        backend.eval(p1, p2)

        # Edge cases
        if t <= 0.0:
            return p1
        if t >= 1.0:
            return p2

        if points_context is None:
            # NO EUCLIDEAN FALLBACK - geodesic requires manifold context
            raise ValueError(
                "Geodesic interpolation requires points_context to define the manifold. "
                "Without context, there is no manifold structure and geodesic is undefined. "
                "Provide a point cloud that defines the discrete manifold."
            )

        points_context = backend.array(points_context)
        backend.eval(points_context)
        n = int(points_context.shape[0])

        if n < 2:
            raise ValueError(
                f"Geodesic interpolation requires at least 2 context points to define "
                f"the manifold structure. Got {n} points."
            )

        # 1. Compute geodesic distances
        geo_result = self.geodesic_distances(points_context)

        # 2. Project p1 and p2 onto the discrete manifold
        idx1 = self._find_nearest_point(points_context, p1, geo_result=geo_result)
        idx2 = self._find_nearest_point(points_context, p2, geo_result=geo_result)

        if idx1 == idx2:
            # Same projection onto manifold - geodesic distance is zero
            # Return the projection point (both p1 and p2 map to same manifold point)
            return points_context[idx1]

        # 3. Reconstruct geodesic path
        path_indices = self._reconstruct_geodesic_path(
            geo_result.distances, idx1, idx2
        )

        if len(path_indices) <= 1:
            # Path reconstruction failed - this indicates disconnected components
            raise ValueError(
                f"Failed to reconstruct geodesic path from index {idx1} to {idx2}. "
                f"This indicates the manifold has disconnected components. "
                f"Increase k_neighbors to improve graph connectivity."
            )

        # 4. Compute cumulative arc lengths along path
        arc_lengths = self._compute_path_arc_lengths(geo_result.distances, path_indices)
        total_length = arc_lengths[-1]

        # Use precision-aware threshold for near-zero detection
        eps = division_epsilon(backend, points_context)
        if total_length < eps:
            # Path has zero length - all points on path are coincident
            # Return the first path point (they're all the same)
            return points_context[path_indices[0]]

        target_length = t * total_length

        # 5. Find segment and interpolate
        return self._interpolate_along_path(
            points_context, path_indices, arc_lengths, target_length
        )

    def _reconstruct_geodesic_path(
        self,
        geo_dist: "Array",
        start_idx: int,
        end_idx: int,
    ) -> list[int]:
        """
        Reconstruct the shortest path from geodesic distance matrix.

        Uses the property that for any point k on the shortest path from i to j:
            d(i, k) + d(k, j) = d(i, j)

        This is the triangle equality (not inequality) that holds
        for points on the geodesic.

        Args:
            geo_dist: Geodesic distance matrix [n, n]
            start_idx: Starting point index
            end_idx: Ending point index

        Returns:
            List of indices forming the path from start to end (inclusive)
        """
        backend = self._backend
        backend.eval(geo_dist)
        n = int(geo_dist.shape[0])

        row = backend.take(geo_dist, backend.array([start_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        cell = backend.take(row, backend.array([end_idx]), axis=0)
        cell = backend.squeeze(cell)
        backend.eval(cell)
        total_dist = float(backend.to_scalar(cell))

        if is_inf(total_dist, backend):
            # Disconnected - no path exists
            return [start_idx]

        # Use precision-aware threshold for near-zero detection
        eps = division_epsilon(backend, geo_dist)
        if total_dist < eps:
            # Same point
            return [start_idx]

        # Greedy path reconstruction: at each step, find the next point on the path
        path = [start_idx]
        current = start_idx
        # Use precision-aware tolerance for floating point comparison
        tolerance = regularization_epsilon(backend, geo_dist) * total_dist
        col_end = backend.take(geo_dist, backend.array([end_idx]), axis=1)
        col_end = backend.squeeze(col_end, axis=1)
        index = backend.arange(n)
        ones_vec = backend.ones((n,))
        zeros_vec = backend.zeros((n,))
        visited = backend.where(index == start_idx, ones_vec, zeros_vec)
        inf_val = float(backend.finfo().max)
        inf_vec = backend.full((n,), inf_val)
        backend.eval(col_end, index, visited, ones_vec, zeros_vec)

        while current != end_idx:
            # Find next point: must satisfy triangle equality
            # d(current, next) + d(next, end) ≈ d(current, end)
            row = backend.take(geo_dist, backend.array([current]), axis=0)
            row = backend.squeeze(row, axis=0)
            dist_to_end_arr = backend.take(row, backend.array([end_idx]), axis=0)
            path_through = row + col_end
            diff = backend.abs(path_through - dist_to_end_arr)
            finite_mask = backend.isfinite(row) & backend.isfinite(col_end)
            valid_mask = finite_mask & (diff <= tolerance) & (visited < 1)  # Not yet visited
            candidate_count_arr = backend.sum(backend.astype(valid_mask, "int32"))
            backend.eval(candidate_count_arr, dist_to_end_arr)
            candidate_count = int(backend.to_scalar(candidate_count_arr))

            if candidate_count == 0:
                best_next = end_idx
            else:
                masked = backend.where(valid_mask, row, inf_vec)
                best_idx_arr = backend.argmin(masked)
                backend.eval(best_idx_arr)
                best_next = int(backend.to_scalar(best_idx_arr))

            path.append(best_next)
            visited = backend.where(index == best_next, ones_vec, visited)
            current = best_next

            # Safety: prevent infinite loops
            if len(path) > n:
                break

        return path

    def _compute_path_arc_lengths(
        self,
        geo_dist: "Array",
        path_indices: list[int],
    ) -> list[float]:
        """
        Compute cumulative arc lengths along a path.

        Uses geodesic distances between consecutive nodes in the path.
        This is exact for the discrete manifold representation.

        Args:
            geo_dist: Geodesic distance matrix [n, n]
            path_indices: Indices forming the path

        Returns:
            List of cumulative arc lengths (first element is 0)
        """
        backend = self._backend

        if len(path_indices) <= 1:
            return [0.0]

        arc_lengths = [0.0]
        for i in range(len(path_indices) - 1):
            start_idx = path_indices[i]
            end_idx = path_indices[i + 1]
            row = backend.take(geo_dist, backend.array([start_idx]), axis=0)
            row = backend.squeeze(row, axis=0)
            cell = backend.take(row, backend.array([end_idx]), axis=0)
            cell = backend.squeeze(cell)
            backend.eval(cell)
            segment_length = float(backend.to_scalar(cell))
            arc_lengths.append(arc_lengths[-1] + segment_length)

        return arc_lengths

    def _interpolate_along_path(
        self,
        points: "Array",
        path_indices: list[int],
        arc_lengths: list[float],
        target_length: float,
    ) -> "Array":
        """
        Select the discrete path point nearest the target arc length.

        The discrete manifold is defined by graph vertices. Returning the
        nearest vertex avoids introducing off-manifold approximations.

        Args:
            points: Point cloud [n, d]
            path_indices: Indices forming the path
            arc_lengths: Cumulative arc lengths at each path point
            target_length: Target arc length for interpolation

        Returns:
            Interpolated point [d]
        """
        backend = self._backend

        if not arc_lengths:
            return points[path_indices[0]]

        best_idx = 0
        best_delta = abs(arc_lengths[0] - target_length)
        for i in range(1, len(arc_lengths)):
            delta = abs(arc_lengths[i] - target_length)
            if delta < best_delta:
                best_delta = delta
                best_idx = i

        return points[path_indices[best_idx]]


__all__ = ["RiemannianInterpolationMixin"]
