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
Manifold coverage analysis and filling.

This module orchestrates sparse region detection and filling strategies
for point clouds on Riemannian manifolds. It integrates:

1. **Density Analysis**: Identifies low-density (sparse) points using
   geodesic-aware density estimation.

2. **Directional Sparsity**: For each point, identifies the most
   under-sampled direction in tangent space.

3. **Local Dimension Analysis**: Detects dimension-collapsed regions
   where the manifold is locally lower-dimensional.

4. **Filling Strategies**: Proposes new points to improve coverage via:
   - Farthest Point Sampling (FPS): maximize geodesic coverage
   - Tangent exploration: step in sparse directions

Mathematical Framework:
    The manifold M is represented as a discrete point cloud with k-NN
    connectivity defining the geodesic structure. Coverage is measured
    as the maximum geodesic distance from any point to the nearest
    selected point (coverage radius).

    For directional analysis, we project neighbors onto the tangent
    sphere S^{m-1} at each point and find the largest angular gap.

References:
    - Farthest Point Sampling: Eldar et al. (1997)
    - Tangent Space Analysis: Pennec (2006)
    - Intrinsic Dimension: Facco et al. (2017)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageMetrics:
    """Global coverage statistics for a point cloud."""

    coverage_radius: float  # Max min-distance to FPS-selected points
    mean_density: float  # Average local density
    density_variance: float  # Variance in local density
    mean_gap_angle: float  # Average max angular gap in tangent space
    dimension_uniformity: float  # 1 - (std/mean) of local dimensions
    n_sparse_points: int  # Number of low-density points
    n_dimension_deficient: int  # Number of dimension-collapsed points


@dataclass
class CoverageAnalysis:
    """Complete coverage analysis of a point cloud on its manifold.

    Combines density, directional, and dimensional analysis to provide
    a comprehensive picture of manifold coverage.
    """

    # Sparse point identification
    sparse_points: list[int]  # Indices of low-density points
    sparse_directions: dict[int, "Array"]  # Point -> most sparse tangent direction

    # Local dimension analysis
    local_dimensions: "Array"  # Per-point intrinsic dimension [n]
    modal_dimension: float  # Most common dimension
    dimension_deficient: list[int]  # Points with collapsed local dimension

    # Proposed fills
    proposed_fills: list["Array"]  # New points to improve coverage

    # Global metrics
    metrics: CoverageMetrics

    # Configuration
    k_neighbors: int
    density_percentile: float  # Threshold for sparse point detection


@dataclass
class CoverageConfig:
    """Configuration for coverage analysis."""

    k_neighbors: int = 10  # k for geodesic/density computation
    density_percentile: float = 0.2  # Points below this percentile are sparse
    dimension_threshold: float = 0.8  # Deficiency threshold for local dimension
    n_fps_samples: int = 10  # Number of FPS samples for coverage radius
    n_fill_proposals: int = 5  # Number of new points to propose
    fill_step_size: float = 0.1  # Step size for tangent exploration


class ManifoldCoverage:
    """
    Coverage analysis and filling for manifold point clouds.

    Integrates:
    - RiemannianGeometry for geodesics and FPS
    - IntrinsicDimension for local ID estimation
    - RiemannianDensityEstimator for density analysis

    Example:
        ```python
        backend = get_default_backend()
        coverage = ManifoldCoverage(backend)

        # Analyze coverage
        analysis = coverage.analyze(points)
        print(f"Coverage radius: {analysis.metrics.coverage_radius}")
        print(f"Sparse points: {len(analysis.sparse_points)}")

        # Get proposed fill points
        fills = coverage.propose_fills(points, n_proposals=5)
        ```
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        points: "Array",
        config: CoverageConfig | None = None,
    ) -> CoverageAnalysis:
        """
        Analyze coverage of a point cloud on its manifold.

        Performs comprehensive analysis including:
        1. Density estimation to find sparse points
        2. Directional analysis to find sparse directions
        3. Local dimension analysis to find collapsed regions
        4. FPS-based coverage radius computation
        5. Proposes fill points to improve coverage

        Args:
            points: Point cloud [n, d]
            config: Analysis configuration (uses defaults if None)

        Returns:
            CoverageAnalysis with complete coverage information
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        if config is None:
            config = CoverageConfig()

        n = int(points.shape[0])

        if n < 3:
            # Not enough points for meaningful analysis
            return CoverageAnalysis(
                sparse_points=[],
                sparse_directions={},
                local_dimensions=backend.zeros((n,)),
                modal_dimension=0.0,
                dimension_deficient=[],
                proposed_fills=[],
                metrics=CoverageMetrics(
                    coverage_radius=0.0,
                    mean_density=0.0,
                    density_variance=0.0,
                    mean_gap_angle=0.0,
                    dimension_uniformity=0.0,
                    n_sparse_points=0,
                    n_dimension_deficient=0,
                ),
                k_neighbors=config.k_neighbors,
                density_percentile=config.density_percentile,
            )

        # 1. Density analysis - find sparse points
        sparse_points, densities = self._find_sparse_points(
            points,
            k=config.k_neighbors,
            percentile=config.density_percentile,
        )

        # 2. Directional analysis - find sparse directions for sparse points
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        sparse_directions: dict[int, "Array"] = {}
        gap_angles: list[float] = []

        for idx in sparse_points:
            coverage = rg.directional_coverage(idx, points, k=config.k_neighbors)
            sparse_directions[idx] = coverage.sparse_direction
            gap_angles.append(coverage.max_gap_angle)

        # 3. Local dimension analysis
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )

        id_estimator = IntrinsicDimension(backend)
        dim_map = id_estimator.local_dimension_map(
            points,
            k=config.k_neighbors,
            deficiency_threshold=config.dimension_threshold,
        )

        # 4. FPS for coverage radius
        fps_result = rg.farthest_point_sampling(
            points,
            n_samples=min(config.n_fps_samples, n),
            k_neighbors=config.k_neighbors,
        )

        # 5. Propose fill points
        proposed_fills = self._propose_fills(
            points,
            sparse_points,
            sparse_directions,
            n_proposals=config.n_fill_proposals,
            step_size=config.fill_step_size,
        )

        # Compute metrics
        density_np = backend.to_numpy(densities).flatten()
        valid_densities = [d for d in density_np if not math.isinf(d) and not math.isnan(d)]

        mean_density = sum(valid_densities) / len(valid_densities) if valid_densities else 0.0
        density_var = (
            sum((d - mean_density) ** 2 for d in valid_densities) / len(valid_densities)
            if len(valid_densities) > 1
            else 0.0
        )

        mean_gap = sum(gap_angles) / len(gap_angles) if gap_angles else 0.0

        # Dimension uniformity: 1 - coefficient of variation
        if dim_map.mean_dimension > 0:
            cv = dim_map.std_dimension / dim_map.mean_dimension
            dim_uniformity = max(0.0, 1.0 - cv)
        else:
            dim_uniformity = 0.0

        metrics = CoverageMetrics(
            coverage_radius=fps_result.coverage_radius,
            mean_density=mean_density,
            density_variance=density_var,
            mean_gap_angle=mean_gap,
            dimension_uniformity=dim_uniformity,
            n_sparse_points=len(sparse_points),
            n_dimension_deficient=len(dim_map.deficient_indices),
        )

        return CoverageAnalysis(
            sparse_points=sparse_points,
            sparse_directions=sparse_directions,
            local_dimensions=dim_map.dimensions,
            modal_dimension=dim_map.modal_dimension,
            dimension_deficient=dim_map.deficient_indices,
            proposed_fills=proposed_fills,
            metrics=metrics,
            k_neighbors=config.k_neighbors,
            density_percentile=config.density_percentile,
        )

    def propose_fills(
        self,
        points: "Array",
        n_proposals: int = 5,
        step_size: float = 0.1,
        k: int = 10,
    ) -> list["Array"]:
        """
        Propose new points to improve manifold coverage.

        Uses a combination of:
        1. Identify sparse points (low density)
        2. For each sparse point, step in the sparsest tangent direction

        Args:
            points: Point cloud [n, d]
            n_proposals: Number of points to propose
            step_size: Distance to step in sparse direction
            k: Number of neighbors for analysis

        Returns:
            List of proposed new points
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n < 3:
            return []

        # Find sparse points
        sparse_points, _ = self._find_sparse_points(points, k=k)

        if len(sparse_points) == 0:
            # No sparse points - use FPS to find coverage gaps
            from modelcypher.core.domain.geometry.riemannian_utils import (
                RiemannianGeometry,
            )

            rg = RiemannianGeometry(backend)
            fps_result = rg.farthest_point_sampling(
                points, n_samples=min(n_proposals, n), k_neighbors=k
            )
            # Use the points with highest min-distance as bases
            sparse_points = fps_result.selected_indices[-n_proposals:]

        # Get sparse directions
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        sparse_directions: dict[int, "Array"] = {}

        for idx in sparse_points[:n_proposals]:
            coverage = rg.directional_coverage(idx, points, k=k)
            sparse_directions[idx] = coverage.sparse_direction

        return self._propose_fills(
            points,
            list(sparse_directions.keys()),
            sparse_directions,
            n_proposals=n_proposals,
            step_size=step_size,
        )

    def farthest_point_sample(
        self,
        points: "Array",
        n_samples: int,
        k: int = 10,
    ) -> list[int]:
        """
        Geodesic farthest point sampling.

        Convenience method wrapping RiemannianGeometry.farthest_point_sampling.

        Args:
            points: Point cloud [n, d]
            n_samples: Number of points to select
            k: Number of neighbors for geodesic graph

        Returns:
            List of selected point indices
        """
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(self._backend)
        result = rg.farthest_point_sampling(points, n_samples, k_neighbors=k)
        return result.selected_indices

    def poisson_disk_sample(
        self,
        points: "Array",
        min_dist: float,
        k: int = 10,
    ) -> list[int]:
        """
        Blue noise sampling with minimum geodesic distance.

        Selects a maximal subset of points such that all pairwise
        geodesic distances are >= min_dist. Uses greedy selection.

        Args:
            points: Point cloud [n, d]
            min_dist: Minimum geodesic distance between selected points
            k: Number of neighbors for geodesic graph

        Returns:
            List of selected point indices
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n == 0:
            return []

        # Compute geodesic distances
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        geo_np = backend.to_numpy(geo_result.distances)

        # Greedy selection
        selected: list[int] = []
        available = set(range(n))

        while available:
            # Select the first available point
            # (could randomize or use density-based priority)
            idx = min(available)
            selected.append(idx)

            # Remove points within min_dist
            to_remove = set()
            for other in available:
                if other != idx and geo_np[idx, other] < min_dist:
                    to_remove.add(other)

            available.discard(idx)
            available -= to_remove

        return selected

    def _find_sparse_points(
        self,
        points: "Array",
        k: int = 10,
        percentile: float = 0.2,
    ) -> tuple[list[int], "Array"]:
        """Find points in low-density regions.

        Uses k-NN radius as a proxy for inverse density:
        larger k-NN radius = lower density = sparser region.

        Returns indices of points below the given density percentile.
        """
        backend = self._backend
        n = int(points.shape[0])

        if n < 2:
            return [], backend.zeros((n,))

        # Compute geodesic distances
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        geo_np = backend.to_numpy(geo_result.distances)

        # k-NN radius for each point (distance to k-th neighbor)
        radii: list[float] = []
        for i in range(n):
            dists = sorted(geo_np[i, :].tolist())
            # dists[0] = 0 (self), dists[k] = k-th neighbor distance
            if len(dists) > k:
                radii.append(dists[k])
            elif len(dists) > 1:
                radii.append(dists[-1])
            else:
                radii.append(0.0)

        # Density = 1 / radius (larger radius = lower density)
        densities: list[float] = []
        for r in radii:
            if r > 1e-12:
                densities.append(1.0 / r)
            else:
                densities.append(float("inf"))

        # Find threshold for sparse points
        finite_densities = [d for d in densities if not math.isinf(d)]
        if len(finite_densities) == 0:
            return [], backend.array(densities)

        sorted_densities = sorted(finite_densities)
        threshold_idx = int(len(sorted_densities) * percentile)
        threshold = sorted_densities[max(0, threshold_idx)]

        # Select sparse points (density below threshold)
        sparse = [i for i, d in enumerate(densities) if d <= threshold]

        return sparse, backend.array(densities)

    def _propose_fills(
        self,
        points: "Array",
        sparse_indices: list[int],
        sparse_directions: dict[int, "Array"],
        n_proposals: int,
        step_size: float,
    ) -> list["Array"]:
        """Propose new points by stepping in sparse directions."""
        backend = self._backend
        proposals: list["Array"] = []

        for idx in sparse_indices[:n_proposals]:
            if idx not in sparse_directions:
                continue

            base = points[idx]
            direction = sparse_directions[idx]

            # Step in sparse direction
            proposed = base + step_size * direction
            proposals.append(proposed)

        return proposals


# Convenience functions


def analyze_coverage(
    points: "Array",
    k: int = 10,
    backend: "Backend | None" = None,
) -> CoverageAnalysis:
    """
    Analyze manifold coverage of a point cloud.

    Convenience function for quick coverage analysis.

    Args:
        points: Point cloud [n, d]
        k: Number of neighbors for analysis
        backend: Backend to use

    Returns:
        CoverageAnalysis with complete coverage information
    """
    b = backend or get_default_backend()
    mc = ManifoldCoverage(b)
    return mc.analyze(points, CoverageConfig(k_neighbors=k))


def find_sparse_regions(
    points: "Array",
    k: int = 10,
    percentile: float = 0.2,
    backend: "Backend | None" = None,
) -> list[int]:
    """
    Find points in sparse (low-density) regions.

    Convenience function that returns just the sparse point indices.

    Args:
        points: Point cloud [n, d]
        k: Number of neighbors for density estimation
        percentile: Density percentile threshold (default 0.2 = bottom 20%)
        backend: Backend to use

    Returns:
        List of indices of points in sparse regions
    """
    b = backend or get_default_backend()
    mc = ManifoldCoverage(b)
    sparse, _ = mc._find_sparse_points(points, k=k, percentile=percentile)
    return sparse
