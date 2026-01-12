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

"""Covariance and log-map utilities for RiemannianGeometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_types import GeodesicDistanceResult
from modelcypher.core.domain.geometry.riemannian_validation import count_inf, count_nan

from .riemannian_core_utils import _promote_precision

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


class RiemannianCovarianceMixin:
    """Riemannian covariance and log-map computations."""

    def riemannian_covariance(
        self,
        points: "Array",
        mean: "Array | None" = None,
        geo_result: GeodesicDistanceResult | None = None,
        geo_from_mean: "Array | None" = None,
    ) -> "Array":
        """
        Compute covariance matrix in the tangent space at the Fréchet mean.

        On a Riemannian manifold, covariance is computed by:
        1. Finding the Fréchet mean μ
        2. Mapping all points to the tangent space at μ via Log_μ
        3. Computing covariance in the tangent space

        Args:
            points: Point cloud [n, d]
            mean: Precomputed Fréchet mean (computed if None)

        Returns:
            Covariance matrix [d, d] in the tangent space
        """
        backend = self._backend
        points = _promote_precision(backend.array(points), backend)
        backend.eval(points)

        n = int(points.shape[0])
        d = int(points.shape[1])

        if n <= 1:
            return backend.zeros((d, d))

        # Compute Fréchet mean if not provided
        if mean is None:
            result = self.frechet_mean(points)
            mean = result.mean

        # Get geodesic distances for proper scaling
        if geo_result is None:
            geo_result = self.geodesic_distances(points)

        # Map points to tangent space at mean
        # Log_μ(x) = (x - μ) * (geodesic_dist / chord_dist)
        tangent_vectors = self._log_map_approximate(
            points,
            mean,
            geo_result,
            geo_from_base=geo_from_mean,
        )

        # Standard covariance in tangent space
        tangent_mean = backend.mean(tangent_vectors, axis=0, keepdims=True)
        centered = tangent_vectors - tangent_mean

        cov = backend.matmul(backend.transpose(centered), centered) / (n - 1)

        return cov

    def log_map(
        self,
        points: "Array",
        base: "Array",
        geo_result: GeodesicDistanceResult | None = None,
        geo_from_base: "Array | None" = None,
    ) -> "Array":
        """Map points to the tangent space at a base point."""
        points = _promote_precision(self._backend.array(points), self._backend)
        base = _promote_precision(self._backend.array(base), self._backend)
        if geo_result is None:
            geo_result = self.geodesic_distances(points)
        return self._log_map_approximate(
            points,
            base,
            geo_result,
            geo_from_base=geo_from_base,
        )

    def _log_map_approximate(
        self,
        points: "Array",
        mean: "Array",
        geo_result: GeodesicDistanceResult,
        geo_from_base: "Array | None" = None,
    ) -> "Array":
        """
        Compute logarithmic map from mean to all points.

        log_μ(x) = (x - μ) * (geodesic_dist / chord_dist)

        Raises:
            ValueError: If scale contains inf or nan values.
        """
        backend = self._backend

        # Attach mean to the k-NN graph for exact discrete geodesics
        if geo_from_base is None:
            geo_from_mean = self._geodesic_distances_from_query(
                points, mean, geo_result=geo_result
            )
        else:
            geo_from_mean = geo_from_base

        # Compute geodesic/chord ratio
        diff = points - backend.reshape(mean, (1, -1))
        chord_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        eps = division_epsilon(backend, chord_dist)
        chord_safe = backend.maximum(chord_dist, backend.full(chord_dist.shape, eps))
        scale = geo_from_mean / chord_safe

        backend.eval(scale)

        # Check for numerical issues
        inf_count = count_inf(scale, backend)
        nan_count = count_nan(scale, backend)

        if inf_count > 0 or nan_count > 0:
            raise ValueError(
                f"Log map scale contains {inf_count} inf and {nan_count} nan values. "
                f"This indicates disconnected manifold components or coincident points. "
                f"Increase k_neighbors to improve graph connectivity."
            )

        # Scaled tangent vectors
        scale_col = backend.reshape(scale, (-1, 1))
        log_vectors = diff * scale_col

        return log_vectors
