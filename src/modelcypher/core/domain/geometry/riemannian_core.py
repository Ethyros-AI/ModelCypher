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

"""Riemannian geometry core operations for representation spaces.

Provides Fréchet mean, geodesic distances, and curvature-aware covariance.

References:
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - Tenenbaum et al. (2000) "Isomap" - geodesic distance via graph
    - Sra & Hosseini (2015) "Conic Geometric Optimization on the Manifold"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.riemannian_interpolation import (
    RiemannianInterpolationMixin,
)
from modelcypher.core.domain.geometry.riemannian_sampling import (
    RiemannianSamplingMixin,
)

from .riemannian_core_covariance import RiemannianCovarianceMixin
from .riemannian_core_curvature import RiemannianCurvatureMixin
from .riemannian_core_geodesic import RiemannianGeodesicMixin
from .riemannian_core_mean import RiemannianMeanMixin

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

# Cache RiemannianGeometry instances by backend for reuse of internal caches.
_RG_CACHE: dict[int, "RiemannianGeometry"] = {}

# Session-scoped cache for geodesic distances and Fréchet means
_cache = ComputationCache.shared()


def _get_riemannian_geometry(backend: "Backend") -> "RiemannianGeometry":
    key = id(backend)
    cached = _RG_CACHE.get(key)
    if cached is None or getattr(cached, "_backend", None) is not backend:
        cached = RiemannianGeometry(backend)
        _RG_CACHE[key] = cached
    return cached


class RiemannianGeometry(
    RiemannianMeanMixin,
    RiemannianGeodesicMixin,
    RiemannianCovarianceMixin,
    RiemannianCurvatureMixin,
    RiemannianSamplingMixin,
    RiemannianInterpolationMixin,
):
    """
    Riemannian geometry operations for representation spaces.

    This class provides curvature-aware alternatives to chord-based operations:
    - Fréchet mean instead of arithmetic mean
    - Geodesic distance instead of chord distance
    - Riemannian covariance instead of ambient covariance

    Inherits from:
    - RiemannianMeanMixin: Fréchet mean computation
    - RiemannianGeodesicMixin: k-NN geodesic distances
    - RiemannianCovarianceMixin: covariance/log-map utilities
    - RiemannianCurvatureMixin: local curvature estimates
    - RiemannianSamplingMixin: farthest_point_sampling, directional_coverage
    - RiemannianInterpolationMixin: geodesic_interpolation
    """

    def __new__(cls, backend: "Backend | None" = None) -> "RiemannianGeometry":
        resolved_backend = backend or get_default_backend()
        key = id(resolved_backend)
        cached = _RG_CACHE.get(key)
        if cached is not None and getattr(cached, "_backend", None) is resolved_backend:
            return cached
        instance = super().__new__(cls)
        _RG_CACHE[key] = instance
        return instance

    def __init__(self, backend: "Backend | None" = None) -> None:
        resolved_backend = backend or get_default_backend()
        if getattr(self, "_backend", None) is resolved_backend:
            return
        self._backend = resolved_backend


__all__ = [
    "RiemannianGeometry",
    "_get_riemannian_geometry",
    "_RG_CACHE",
    "_cache",
]
