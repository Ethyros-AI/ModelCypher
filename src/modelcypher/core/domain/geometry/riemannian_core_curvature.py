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

"""Local curvature estimation for RiemannianGeometry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_types import CurvatureEstimate

from .precision import _promote_precision
from .riemannian_core_utils import _count_mask

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


class RiemannianCurvatureMixin:
    """Local curvature estimation helpers.

    Activation-space only — weight space is Euclidean (falsification 2026-02-23).
    """

    def estimate_local_curvature(
        self,
        points: "Array",
        center_idx: int,
        k_neighbors: int | None = None,
        domain: GeometryDomain = GeometryDomain.ACTIVATION,
    ) -> CurvatureEstimate:
        """
        Estimate local sectional curvature at a point.

        Uses geodesic defect (geodesic vs chord length) for 2D/3D probes,
        with sign reported only when the defect exceeds a matched flat-data
        null computed at the same (n, d, k). For 4D+, returns the fallback
        scalar estimate but leaves sign unknown until a matched null is wired.

        Args:
            points: Point cloud [n, d]
            center_idx: Index of the center point
            k_neighbors: Number of neighbors (if None, derived from geometry)

        Returns:
            CurvatureEstimate with estimated sectional curvature

        Raises:
            ValueError: If domain is WEIGHT (weight space is Euclidean).
        """
        if domain == GeometryDomain.WEIGHT:
            raise ValueError(
                "Curvature estimation is not meaningful for weight-space tensors. "
                "Weight space is Euclidean (falsification 2026-02-23). "
                "Use spectral analysis (SVD, spectral_capacity) for weight geometry."
            )
        backend = self._backend
        points = _promote_precision(backend.array(points), backend)
        backend.eval(points)

        n = int(points.shape[0])
        d = int(points.shape[1]) if len(points.shape) > 1 else 1

        if n < 3:
            return CurvatureEstimate(
                sectional_curvature=0.0,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        # Get geodesic distances (k=None triggers connectivity-based selection)
        geo_result = self.geodesic_distances(points, k_neighbors=k_neighbors)
        # Use the actual k from the result (may differ from input if None was passed)
        k_neighbors = geo_result.k_neighbors

        # Look at the k nearest neighbors of the center point (geodesic order)
        center_geo = geo_result.distances[center_idx]
        inf_val = geo_result.inf_value
        center_geo = backend.where(
            backend.arange(0, n) == center_idx,
            backend.full((n,), inf_val),
            center_geo,
        )
        kth = max(0, min(k_neighbors - 1, n - 1))
        partitioned = backend.argpartition(center_geo, kth)
        neighbors = partitioned[:k_neighbors]
        backend.eval(neighbors)

        if d > 3:
            from modelcypher.core.domain.geometry.manifold_curvature import (
                SectionalCurvatureEstimator,
            )

            center = points[center_idx]
            neighbor_pts = backend.take(points, neighbors, axis=0)
            backend.eval(center, neighbor_pts)
            estimator = SectionalCurvatureEstimator()
            local = estimator.estimate_local_curvature(center, neighbor_pts)
            return CurvatureEstimate(
                sectional_curvature=local.mean_sectional,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        # 2D/3D: compute geodesic defect against chord distances.
        center = points[center_idx]
        neighbor_pts = backend.take(points, neighbors, axis=0)
        diffs = neighbor_pts - center
        center_chord_k = backend.sqrt(backend.sum(diffs * diffs, axis=1))
        backend.eval(center_chord_k)

        # Compute precision-aware epsilon
        eps = division_epsilon(backend, center_chord_k)

        center_geo_k = backend.take(center_geo, neighbors)
        backend.eval(center_geo_k)

        # Compute geodesic defect: (geodesic - chord) / chord
        valid_mask = center_chord_k > eps
        valid_count_arr = _count_mask(valid_mask, backend, dtype_source=center_chord_k)
        backend.eval(valid_count_arr)
        valid_count_val = int(backend.to_scalar(valid_count_arr))

        if valid_count_val == 0:
            return CurvatureEstimate(
                sectional_curvature=0.0,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        defects = (center_geo_k - center_chord_k) / center_chord_k
        defects = backend.where(valid_mask, defects, backend.zeros_like(defects))
        sum_defects = backend.sum(defects)
        mean_defect_arr = sum_defects / max(valid_count_val, 1)
        backend.eval(mean_defect_arr)
        mean_defect = float(backend.to_scalar(mean_defect_arr))

        # Estimate curvature from defect
        # For a sphere of radius R, geodesic/chord ≈ 1 + K*r²/6 for small r
        # where K = 1/R² is the sectional curvature
        # So defect ≈ K*r²/6, giving K ≈ 6*defect/r²

        avg_radius_arr = backend.mean(center_chord_k)
        backend.eval(avg_radius_arr)
        avg_radius = float(backend.to_scalar(avg_radius_arr))
        if avg_radius > eps:
            # Rough curvature estimate
            sectional_curvature = 6.0 * mean_defect / (avg_radius * avg_radius)
        else:
            sectional_curvature = 0.0

        null_min, null_max = self._matched_flat_defect_range(
            n=n,
            d=d,
            center_idx=center_idx,
            k_neighbors=k_neighbors,
            scale=max(avg_radius, eps),
        )
        sign_slack = division_epsilon(backend, backend.array([mean_defect, null_min, null_max]))
        is_positive = mean_defect > null_max + sign_slack
        is_negative = mean_defect < null_min - sign_slack

        return CurvatureEstimate(
            sectional_curvature=sectional_curvature,
            is_positive=is_positive,
            is_negative=is_negative,
            confidence=0.0,
        )

    def _matched_flat_defect_range(
        self,
        *,
        n: int,
        d: int,
        center_idx: int,
        k_neighbors: int,
        scale: float,
    ) -> tuple[float, float]:
        """Return min/max geodesic defect for a matched flat hyperplane sample."""
        backend = self._backend
        flat_points = self._flat_hyperplane_sample(
            n=n,
            d=d,
            center_idx=center_idx,
            scale=scale,
        )
        geo_result = self.geodesic_distances(flat_points, k_neighbors=k_neighbors)
        center_geo = geo_result.distances[center_idx]
        inf_val = geo_result.inf_value
        center_geo = backend.where(
            backend.arange(0, n) == center_idx,
            backend.full((n,), inf_val),
            center_geo,
        )
        kth = max(0, min(k_neighbors - 1, n - 1))
        partitioned = backend.argpartition(center_geo, kth)
        neighbors = partitioned[:k_neighbors]
        center = flat_points[center_idx]
        neighbor_pts = backend.take(flat_points, neighbors, axis=0)
        diffs = neighbor_pts - center
        chords = backend.sqrt(backend.sum(diffs * diffs, axis=1))
        geodesics = backend.take(center_geo, neighbors)
        backend.eval(chords, geodesics)

        eps = division_epsilon(backend, chords)
        valid_mask = chords > eps
        valid_count_arr = _count_mask(valid_mask, backend, dtype_source=chords)
        backend.eval(valid_count_arr)
        if int(backend.to_scalar(valid_count_arr)) == 0:
            return 0.0, 0.0

        defects = (geodesics - chords) / chords
        defects = backend.where(valid_mask, defects, backend.zeros_like(defects))
        valid_large = backend.where(valid_mask, defects, backend.full(defects.shape, -inf_val))
        valid_small = backend.where(valid_mask, defects, backend.full(defects.shape, inf_val))
        null_max_arr = backend.max(valid_large)
        null_min_arr = backend.min(valid_small)
        backend.eval(null_min_arr, null_max_arr)
        return float(backend.to_scalar(null_min_arr)), float(backend.to_scalar(null_max_arr))

    def _flat_hyperplane_sample(
        self,
        *,
        n: int,
        d: int,
        center_idx: int,
        scale: float,
    ) -> "Array":
        """Build deterministic uniform points on a flat hyperplane."""
        if d < 1:
            raise ValueError("Point dimension must be positive")
        width = math.ceil(math.sqrt(max(n, 1)))
        midpoint = (width - 1) / 2.0
        coords: list[list[float]] = []
        for idx in range(n):
            row = idx // width
            col = idx % width
            point = [0.0] * d
            if d == 1:
                point[0] = (idx - (n - 1) / 2.0) * scale
            else:
                point[0] = (col - midpoint) * scale
                point[1] = (row - midpoint) * scale
            coords.append(point)

        origin_idx = min(
            range(n),
            key=lambda idx: sum(abs(value) for value in coords[idx]),
        )
        coords[center_idx], coords[origin_idx] = coords[origin_idx], coords[center_idx]
        arr = self._backend.array(coords)
        self._backend.eval(arr)
        return arr
