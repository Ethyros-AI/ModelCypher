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

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_types import CurvatureEstimate

from .riemannian_core_utils import _count_mask, _promote_precision

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

        Uses geodesic defect (geodesic vs chord length) for 2D/3D probes.
        For 4D+, falls back to geodesic-only curvature estimation.

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
            std_val = sqrt_scalar(local.variance_sectional, backend)
            confidence = 1.0 / (1.0 + std_val) if std_val > 0 else 1.0
            return CurvatureEstimate(
                sectional_curvature=local.mean_sectional,
                is_positive=local.mean_sectional > 0,
                is_negative=local.mean_sectional < 0,
                confidence=confidence,
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

        if valid_count_val > 1:
            diff = defects - mean_defect_arr
            diff = backend.where(valid_mask, diff, backend.zeros_like(diff))
            variance = backend.sum(diff * diff) / valid_count_val
            std_defect_arr = backend.sqrt(variance)
            backend.eval(std_defect_arr)
            std_defect = float(backend.to_scalar(std_defect_arr))
        else:
            std_defect = 0.0

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

        # Confidence based on consistency of defects
        confidence = 1.0 / (1.0 + std_defect) if std_defect > 0 else 1.0

        return CurvatureEstimate(
            sectional_curvature=sectional_curvature,
            is_positive=sectional_curvature > 0,
            is_negative=sectional_curvature < 0,
            confidence=confidence,
        )
