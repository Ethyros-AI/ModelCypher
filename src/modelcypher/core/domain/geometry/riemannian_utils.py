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

"""
Riemannian geometry for high-dimensional representation spaces.

This module provides the main API for Riemannian geometry operations.
It re-exports classes and functions from the implementation modules:

- RiemannianGeometry: Main class for geodesic and Fréchet mean computation
- Result types: FrechetMeanResult, GeodesicDistanceResult, CurvatureEstimate, etc.
- Validation helpers: count_nan, count_inf, all_finite, etc.
- Convenience functions: frechet_mean, geodesic_distance_matrix, etc.

Mathematical Background:
    On a Riemannian manifold (M, g), the geodesic distance d(p, q) is the
    length of the shortest path between p and q. The Fréchet mean minimizes:

        μ = argmin_{p ∈ M} Σᵢ d²(p, xᵢ)

    For discrete point clouds, the manifold is represented by a k-NN graph.
    Geodesic distance = shortest path on this graph. This is exact for the
    discrete manifold structure.

References:
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - Tenenbaum et al. (2000) "Isomap" - geodesic distance via graph
    - Sra & Hosseini (2015) "Conic Geometric Optimization on the Manifold"

Research Connections:
    Geodesic distance is the correct metric for neural representations because
    curvature is inherent in high-dimensional spaces. This aligns with the
    Platonic Representation Hypothesis (Huh et al., ICML 2024).

    See also: docs/RESEARCH-CONNECTIONS.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export from core module (main class)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_core import (
    RiemannianGeometry,
    _cache,
    _get_riemannian_geometry,
    _RG_CACHE,
)

# Re-export result types
from modelcypher.core.domain.geometry.riemannian_types import (
    CurvatureEstimate,
    DirectionalCoverage,
    FarthestPointSamplingResult,
    FrechetMeanResult,
    GeodesicDistanceResult,
)

# Re-export validation helpers
from modelcypher.core.domain.geometry.riemannian_validation import (
    all_finite,
    count_finite,
    count_inf,
    count_nan,
    count_nonfinite,
    derive_k_neighbors,
    has_inf,
    has_nan,
    safe_arithmetic_mean,
    set_matrix_element,
)



if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Convenience functions
# =============================================================================


def frechet_mean(
    points: "Array",
    weights: "Array | None" = None,
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    max_k_neighbors: int | None = None,
) -> "Array":
    """
    Compute the Fréchet mean of a point set.

    Convenience function that returns just the mean point.

    Args:
        points: Point cloud [n, d]
        weights: Optional weights [n]
        backend: Backend to use
        k_neighbors: Optional fixed k for geodesic graph connectivity
        max_k_neighbors: Optional upper bound for adaptive k retries

    Returns:
        Fréchet mean point [d]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.frechet_mean(
        points,
        weights,
        k_neighbors=k_neighbors,
        max_k_neighbors=max_k_neighbors,
    )
    return result.mean


def geodesic_distance_matrix(
    points: "Array",
    k_neighbors: int | None = None,
    backend: "Backend | None" = None,
) -> "Array":
    """
    Compute pairwise geodesic distances.

    Convenience function that returns just the distance matrix.

    Args:
        points: Point cloud [n, d]
        k_neighbors: Number of neighbors for graph construction
        backend: Backend to use

    Returns:
        Geodesic distance matrix [n, n]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.geodesic_distances(points, k_neighbors)
    return result.distances


def farthest_point_sampling(
    points: "Array",
    n_samples: int,
    seed_idx: int = 0,
    k_neighbors: int | None = None,
    backend: "Backend | None" = None,
) -> list[int]:
    """
    Select points via geodesic farthest point sampling.

    Convenience function that returns just the selected indices.

    Args:
        points: Point cloud [n, d]
        n_samples: Number of points to select
        seed_idx: Starting point index
        k_neighbors: k for geodesic graph
        backend: Backend to use

    Returns:
        List of selected point indices
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.farthest_point_sampling(points, n_samples, seed_idx, k_neighbors)
    return result.selected_indices


def find_sparse_direction(
    point_idx: int,
    points: "Array",
    k: int = 10,
    backend: "Backend | None" = None,
) -> "Array":
    """
    Find the most under-sampled direction at a point.

    Convenience function that returns just the sparse direction vector.

    Args:
        point_idx: Index of the center point
        points: Point cloud [n, d]
        k: Number of neighbors to analyze
        backend: Backend to use

    Returns:
        Unit vector in the most sparse direction [d]
    """
    if backend is None:
        backend = get_default_backend()

    rg = _get_riemannian_geometry(backend)
    result = rg.directional_coverage(point_idx, points, k=k)
    return result.sparse_direction


__all__ = [
    # Main class
    "RiemannianGeometry",
    # Result types
    "FrechetMeanResult",
    "GeodesicDistanceResult",
    "CurvatureEstimate",
    "DirectionalCoverage",
    "FarthestPointSamplingResult",
    # Validation helpers
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

    # Convenience functions
    "frechet_mean",
    "geodesic_distance_matrix",
    "farthest_point_sampling",
    "find_sparse_direction",
    # Private (for internal use)
    "_get_riemannian_geometry",
    "_RG_CACHE",
    "_cache",
]
