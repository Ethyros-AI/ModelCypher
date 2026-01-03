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
Local density estimation for manifold visualization.

Computes per-point density in the projected space for:
- Point cloud sizing (denser = smaller markers)
- Volumetric cloud rendering (density isosurfaces)
- Concept cluster identification

Uses k-NN distance to estimate local density:
    density(x) ∝ 1 / volume(k-NN ball)

This is a direct geometric measurement, not a statistical estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class DensityResult:
    """Result of density estimation.

    All values are raw geometric measurements:
    - densities: Per-point local density (higher = more crowded)
    - radii: k-NN ball radius for each point
    - neighbors: k-NN indices for each point

    Attributes:
        densities: Per-point density values [n_points]
        radii: Distance to k-th nearest neighbor [n_points]
        neighbors: k-NN indices [n_points, k]
        k_neighbors: Number of neighbors used
    """

    densities: "Array"
    radii: "Array"
    neighbors: "Array"
    k_neighbors: int


@dataclass
class DensityConfiguration:
    """Configuration for density estimation.

    Attributes:
        k_neighbors: Number of neighbors for density estimation.
            When None, derived as sqrt(n) clamped to [3, n-1].
        normalize: Whether to normalize densities to [0, 1]
    """

    k_neighbors: int | None = None
    normalize: bool = True


class DensityEstimator:
    """
    Estimate local density from k-NN distances.

    The density at each point is inversely proportional to the volume
    of its k-NN ball:

        density(x) ∝ k / V(B_k(x))

    where B_k(x) is the ball containing the k nearest neighbors.
    In d dimensions: V ∝ r^d, so density ∝ 1/r^d.

    For visualization, we use 3D projected space, so d=3.

    Usage:
        estimator = DensityEstimator(backend)
        result = estimator.compute(points_3d, k=10)

        # For point sizing: larger density = smaller marker
        sizes = 1 / (result.densities + epsilon)

        # For isosurface: threshold density values
        dense_mask = result.densities > threshold
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize the density estimator.

        Args:
            backend: Backend for tensor operations (defaults to MLX)
        """
        self.backend = backend or get_default_backend()

    def compute(
        self,
        points: "Array",
        config: DensityConfiguration | None = None,
    ) -> DensityResult:
        """
        Compute local density for each point.

        Uses k-NN distance to estimate local density.
        Density is inversely proportional to the volume of the k-NN ball.

        Args:
            points: Point cloud [n_points, d]
            config: Density estimation configuration

        Returns:
            DensityResult with per-point densities and k-NN info

        Raises:
            ValueError: If points is too small for k-NN
        """
        b = self.backend
        config = config or DensityConfiguration()

        n_points, d = points.shape

        # Derive k from sqrt(n) when not specified
        if config.k_neighbors is not None:
            k = config.k_neighbors
        else:
            # sqrt(n) scaling with clamping to [3, n-1]
            import math
            k = max(3, min(n_points - 1, int(math.sqrt(n_points))))

        if n_points <= k:
            raise ValueError(
                f"Need more than {k} points for density estimation, "
                f"got {n_points}"
            )

        # Compute pairwise squared distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y^T
        points_sq = b.sum(points ** 2, axis=1, keepdims=True)  # [n, 1]
        dot = b.matmul(points, b.transpose(points))  # [n, n]
        dist_sq = points_sq + b.transpose(points_sq) - 2 * dot  # [n, n]

        # Ensure non-negative (numerical precision)
        dist_sq = b.maximum(dist_sq, 0.0)
        b.eval(dist_sq)

        # Find k+1 nearest neighbors (including self at distance 0)
        # argsort gives indices in ascending order
        sorted_indices = b.argsort(dist_sq, axis=1)  # [n, n]
        b.eval(sorted_indices)

        # Exclude self (index 0), take next k neighbors
        neighbors = sorted_indices[:, 1:k+1]  # [n, k]
        b.eval(neighbors)

        # Get distance to k-th nearest neighbor (for radius)
        # Sort the distances and take the k-th column (k+1 to skip self)
        sorted_dist_sq = b.sort(dist_sq, axis=1)  # [n, n]
        b.eval(sorted_dist_sq)

        # k-th neighbor is at index k (0=self, 1=1st neighbor, ..., k=k-th neighbor)
        radii_sq = sorted_dist_sq[:, k]  # [n]
        b.eval(radii_sq)

        radius_eps = division_epsilon(b, radii_sq)
        radii = b.sqrt(radii_sq + radius_eps)
        b.eval(radii)

        # Compute density as 1 / r^d (volume of d-dimensional ball)
        # Actually we use k / r^d for proper scaling
        density_eps = division_epsilon(b, radii)
        densities = k / (radii ** d + density_eps)
        b.eval(densities)

        # Normalize to [0, 1] if requested
        if config.normalize:
            min_density = b.min(densities)
            max_density = b.max(densities)
            b.eval(min_density, max_density)

            range_density = max_density - min_density
            range_eps = division_epsilon(b, range_density)
            if float(b.to_scalar(range_density)) > range_eps:
                densities = (densities - min_density) / range_density
                b.eval(densities)
            else:
                # All same density, normalize to 0.5
                densities = b.ones_like(densities) * 0.5
                b.eval(densities)

        return DensityResult(
            densities=densities,
            radii=radii,
            neighbors=neighbors,
            k_neighbors=k,
        )

    def compute_grid_density(
        self,
        points: "Array",
        grid_size: int = 20,
        config: DensityConfiguration | None = None,
    ) -> tuple["Array", "Array", "Array", "Array"]:
        """
        Compute density on a 3D grid for volumetric rendering.

        Useful for creating isosurface or volumetric cloud visualizations.

        Args:
            points: Point cloud [n_points, 3]
            grid_size: Number of grid cells per dimension
            config: Density configuration

        Returns:
            Tuple of (X, Y, Z, density) where:
            - X, Y, Z are meshgrid coordinates [grid_size, grid_size, grid_size]
            - density is the density at each grid point

        Raises:
            ValueError: If points is not 3D
        """
        b = self.backend
        config = config or DensityConfiguration()

        if points.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got {points.shape[1]}D")

        n_points = points.shape[0]

        # Get bounding box with padding
        min_vals = b.min(points, axis=0)
        max_vals = b.max(points, axis=0)
        b.eval(min_vals, max_vals)

        # Add 10% padding in backend space
        padding = (max_vals - min_vals) * 0.1
        min_pad = min_vals - padding
        max_pad = max_vals + padding
        b.eval(min_pad, max_pad)

        # Create grid - use tolist() for efficient extraction of small arrays
        min_pad_list = b.tolist(min_pad)
        max_pad_list = b.tolist(max_pad)
        x = b.linspace(float(min_pad_list[0]), float(max_pad_list[0]), grid_size)
        y = b.linspace(float(min_pad_list[1]), float(max_pad_list[1]), grid_size)
        z = b.linspace(float(min_pad_list[2]), float(max_pad_list[2]), grid_size)
        b.eval(x, y, z)

        # Meshgrid
        X, Y, Z = b.meshgrid(x, y, z, indexing='ij')
        b.eval(X, Y, Z)

        # Flatten grid for distance computation
        grid_points = b.stack([
            b.reshape(X, (-1,)),
            b.reshape(Y, (-1,)),
            b.reshape(Z, (-1,)),
        ], axis=1)  # [grid_size^3, 3]
        b.eval(grid_points)

        # Compute distances from each grid point to all data points
        # [grid_size^3, n_points]
        grid_sq = b.sum(grid_points ** 2, axis=1, keepdims=True)
        points_sq = b.sum(points ** 2, axis=1, keepdims=True)
        dot = b.matmul(grid_points, b.transpose(points))
        dist_sq = grid_sq + b.transpose(points_sq) - 2 * dot
        dist_sq = b.maximum(dist_sq, 0.0)
        b.eval(dist_sq)

        # Find k-th nearest neighbor distance for each grid point
        sorted_dists = b.sort(dist_sq, axis=1)
        b.eval(sorted_dists)

        # Derive k from sqrt(n) when not specified
        if config.k_neighbors is not None:
            k = min(config.k_neighbors, n_points - 1)
        else:
            import math
            k = max(3, min(n_points - 1, int(math.sqrt(n_points))))
        kth_dist_sq = sorted_dists[:, k]
        distance_eps = division_epsilon(b, kth_dist_sq)
        kth_dist = b.sqrt(kth_dist_sq + distance_eps)
        b.eval(kth_dist)

        # Density = k / r^3
        eps = division_epsilon(b, kth_dist)
        density_flat = k / (kth_dist ** 3 + eps)
        b.eval(density_flat)

        # Reshape to grid
        density = b.reshape(density_flat, (grid_size, grid_size, grid_size))
        b.eval(density)

        return X, Y, Z, density

    def local_density(self, points: "Array", k: int = 10) -> "Array":
        """
        Convenience method for getting normalized per-point densities.

        Args:
            points: Point cloud [n_points, d]
            k: Number of neighbors

        Returns:
            Normalized density values [n_points] in range [0, 1]
        """
        config = DensityConfiguration(k_neighbors=k, normalize=True)
        result = self.compute(points, config)
        return result.densities
