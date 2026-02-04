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
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_distance_matrix

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


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - k_neighbors: derived from Berry & Sauer 2016: k >= ceil(log(n))
# - No normalization: return raw geometric measurements
# =============================================================================


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
            backend: Backend for tensor operations (defaults to system-selected backend)
        """
        self.backend = backend or get_default_backend()

    def compute(
        self,
        points: "Array",
    ) -> DensityResult:
        """
        Compute local density for each point.

        Uses k-NN distance to estimate local density.
        Density is inversely proportional to the volume of the k-NN ball.

        All parameters are derived from the data:
        - k_neighbors: Berry & Sauer 2016 criterion: k >= ceil(log(n))

        Args:
            points: Point cloud [n_points, d]

        Returns:
            DensityResult with per-point densities and k-NN info

        Raises:
            ValueError: If points is too small for k-NN
        """
        b = self.backend
        import math

        n_points, d = points.shape

        # Derive k from Berry & Sauer 2016: k >= ceil(log(n))
        # This ensures the k-NN graph is connected with high probability
        k = max(1, min(n_points - 1, int(math.ceil(math.log(n_points)))))

        if n_points <= k:
            raise ValueError(
                f"Need more than {k} points for density estimation, "
                f"got {n_points}"
            )

        # Compute pairwise geodesic distances on the manifold
        dist = geodesic_distance_matrix(points, backend=b)
        b.eval(dist)

        # Find k nearest neighbors (excluding self) without full sort.
        max_dist_arr = b.max(dist)
        b.eval(max_dist_arr)
        max_dist = float(b.to_scalar(max_dist_arr))
        eps = division_epsilon(b, dist)
        base = max(max_dist, eps)
        inf_val = min(base / eps, b.finfo(dist.dtype).max)
        dist_no_self = dist + b.eye(n_points) * inf_val

        kth = max(0, min(k - 1, n_points - 2))
        partitioned = b.argpartition(dist_no_self, kth, axis=1)
        neighbors = partitioned[:, :k]  # [n, k] unsorted
        neighbor_dists = b.take_along_axis(dist, neighbors, axis=1)
        radii = b.max(neighbor_dists, axis=1)  # distance to k-th neighbor
        b.eval(neighbors, radii)

        # Compute density as 1 / r^d (volume of d-dimensional ball)
        # Actually we use k / r^d for proper scaling
        density_eps = division_epsilon(b, radii)
        densities = k / (radii ** d + density_eps)
        b.eval(densities)

        # Return raw geometric measurements - no normalization
        # (normalization is presentation, not geometry)

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
    ) -> tuple["Array", "Array", "Array", "Array"]:
        """
        Compute density on a 3D grid for volumetric rendering.

        Useful for creating isosurface or volumetric cloud visualizations.

        All parameters are derived from the data:
        - k_neighbors: Berry & Sauer 2016 criterion: k >= ceil(log(n))

        Args:
            points: Point cloud [n_points, 3]
            grid_size: Number of grid cells per dimension

        Returns:
            Tuple of (X, Y, Z, density) where:
            - X, Y, Z are meshgrid coordinates [grid_size, grid_size, grid_size]
            - density is the density at each grid point

        Raises:
            ValueError: If points is not 3D
        """
        b = self.backend
        import math

        if points.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got {points.shape[1]}D")

        n_points = points.shape[0]

        # Get bounding box with padding
        min_vals = b.min(points, axis=0)
        max_vals = b.max(points, axis=0)
        b.eval(min_vals, max_vals)

        eps = machine_epsilon(b, points)
        padding = (max_vals - min_vals) * eps
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

        # Compute geodesic distances from each grid point to all data points
        # [grid_size^3, n_points]
        combined = b.concatenate([grid_points, points], axis=0)
        b.eval(combined)
        dist_full = geodesic_distance_matrix(combined, backend=b)
        b.eval(dist_full)
        grid_count = int(grid_points.shape[0])
        dist = dist_full[:grid_count, grid_count:]
        b.eval(dist)

        # Derive k from Berry & Sauer 2016: k >= ceil(log(n))
        k = max(1, min(n_points - 1, int(math.ceil(math.log(n_points)))))
        kth = max(0, min(k, n_points - 1))
        partitioned = b.argpartition(dist, kth, axis=1)
        kth_idx = partitioned[:, kth : kth + 1]
        kth_dist = b.take_along_axis(dist, kth_idx, axis=1)
        kth_dist = b.squeeze(kth_dist, axis=1)
        b.eval(kth_dist)

        # Density = k / r^3
        eps = division_epsilon(b, kth_dist)
        density_flat = k / (kth_dist ** 3 + eps)
        b.eval(density_flat)

        # Reshape to grid
        density = b.reshape(density_flat, (grid_size, grid_size, grid_size))
        b.eval(density)

        return X, Y, Z, density
