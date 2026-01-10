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

"""Result dataclasses for Riemannian geometry operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


@dataclass(frozen=True)
class FrechetMeanResult:
    """Result of Fréchet mean computation."""

    mean: "Array"
    iterations: int
    converged: bool
    final_variance: float  # Sum of squared geodesic distances to mean


@dataclass(frozen=True)
class GeodesicDistanceResult:
    """Result of geodesic distance computation."""

    distances: "Array"  # [n, n] pairwise geodesic distance matrix
    adjacency: "Array"  # [n, n] k-NN adjacency with large sentinel for no-edge
    inf_value: float  # Sentinel for disconnected pairs in adjacency/distances
    k_neighbors: int
    connected: bool  # Whether the graph is fully connected


@dataclass(frozen=True)
class CurvatureEstimate:
    """Local curvature estimate at a point."""

    sectional_curvature: float  # Estimated sectional curvature
    is_positive: bool  # Positive curvature (sphere-like)
    is_negative: bool  # Negative curvature (hyperbolic-like)
    confidence: float  # Confidence in the estimate [0, 1]


@dataclass(frozen=True)
class DirectionalCoverage:
    """Results of directional sparsity analysis in tangent space.

    Identifies the most under-sampled direction at a point by analyzing
    the angular distribution of neighbors on the tangent sphere.
    """

    sparse_direction: "Array"  # Unit vector in most sparse direction [d]
    max_gap_angle: float  # Largest angular gap (radians)
    coverage_uniformity: float  # 0 = highly non-uniform, 1 = fully uniform
    neighbor_directions: "Array"  # Normalized tangent directions to neighbors [k, d]
    point_idx: int  # Index of the analyzed point


@dataclass(frozen=True)
class FarthestPointSamplingResult:
    """Results of geodesic farthest point sampling.

    FPS selects points that maximize minimum geodesic distance to the
    already-selected set, providing optimal coverage of the manifold.
    """

    selected_indices: list[int]  # Indices of selected points
    min_distances: "Array"  # Final min-distance-to-selected for each point
    coverage_radius: float  # Maximum min-distance (radius of coverage)


@dataclass(frozen=True)
class ManifoldContext:
    """Context for geodesic operations on a discrete manifold.

    All geodesic operations require this context - no silent chord fallbacks.
    The k-NN graph structure captures manifold curvature that Euclidean/chord
    distance systematically ignores in high dimensions (4D+).

    Attributes:
        geo_result: Pre-computed geodesic distance result with k-NN graph
        points: The point cloud defining this manifold [n, d]
        sigma: RBF kernel bandwidth (derived from median distance if None)
    """

    geo_result: GeodesicDistanceResult
    points: "Array"
    sigma: float | None = None

    @property
    def n_points(self) -> int:
        """Number of points in the manifold."""
        return int(self.points.shape[0])

    @property
    def k_neighbors(self) -> int:
        """Number of neighbors used in k-NN graph."""
        return self.geo_result.k_neighbors

    @property
    def is_connected(self) -> bool:
        """Whether the k-NN graph is fully connected."""
        return self.geo_result.connected

    @property
    def distances(self) -> "Array":
        """Pairwise geodesic distance matrix [n, n]."""
        return self.geo_result.distances


__all__ = [
    "FrechetMeanResult",
    "GeodesicDistanceResult",
    "CurvatureEstimate",
    "DirectionalCoverage",
    "FarthestPointSamplingResult",
    "ManifoldContext",
]
