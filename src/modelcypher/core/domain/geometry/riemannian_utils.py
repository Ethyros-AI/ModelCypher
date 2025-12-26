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

Neural network activations define points on a manifold. This module computes
exact geometric quantities on that manifold:

1. **Fréchet Mean (Karcher Mean)**: The Riemannian center of mass.
   Minimizes sum of squared geodesic distances: μ = argmin_p Σ d²(p, x_i)

2. **Geodesic Distance**: Shortest path along the manifold surface.
   Computed via k-NN graph - the discrete representation of the manifold.

3. **Exponential/Logarithmic Maps**: Local coordinate systems on manifolds.

4. **Riemannian Covariance**: Covariance computed in tangent space,
   respecting manifold curvature.

Mathematical Background:
    On a Riemannian manifold (M, g), the geodesic distance d(p, q) is the
    length of the shortest path between p and q. The Fréchet mean minimizes:

        μ = argmin_{p ∈ M} Σᵢ d²(p, xᵢ)

    The gradient of this objective is:

        ∇f(p) = -2 Σᵢ Log_p(xᵢ)

    where Log_p is the Riemannian logarithm (inverse of exponential map).

    For discrete point clouds, the manifold is represented by a k-NN graph.
    Geodesic distance = shortest path on this graph. This is exact for the
    discrete manifold structure.

References:
    - Pennec (2006) "Intrinsic Statistics on Riemannian Manifolds"
    - Tenenbaum et al. (2000) "Isomap" - geodesic distance via graph
    - Sra & Hosseini (2015) "Conic Geometric Optimization on the Manifold"
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Session-scoped cache for geodesic distances and Fréchet means
_cache = ComputationCache.shared()


def _set_matrix_element(
    backend: "Backend",
    matrix: "Array",
    i: int,
    j: int,
    value: float,
) -> "Array":
    """Set a single element in a matrix using backend ops.

    This is inefficient for many updates but works on any backend.
    For building sparse adjacency, we accept this cost to stay on GPU.
    """
    # Create a mask: 1 at (i,j), 0 elsewhere
    n = matrix.shape[0]
    m = matrix.shape[1]

    # Create row and column index arrays
    row_idx = backend.arange(n)
    col_idx = backend.arange(m)

    # Broadcast to create masks
    row_mask = row_idx == i  # [n]
    col_mask = col_idx == j  # [m]

    # Outer product for 2D mask
    row_mask_2d = backend.reshape(row_mask, (n, 1))
    col_mask_2d = backend.reshape(col_mask, (1, m))

    # Element-wise AND via multiplication (both are boolean-like 0/1)
    # Convert to float for multiplication
    row_float = backend.astype(row_mask_2d, "float32")
    col_float = backend.astype(col_mask_2d, "float32")
    mask = row_float * col_float  # [n, m], 1.0 at (i,j), 0.0 elsewhere

    # Update: matrix * (1 - mask) + value * mask
    result = matrix * (1.0 - mask) + value * mask
    return result


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
    coverage_uniformity: float  # 0 = highly non-uniform, 1 = perfectly uniform
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


class RiemannianGeometry:
    """
    Riemannian geometry operations for representation spaces.

    This class provides curvature-aware alternatives to Euclidean operations:
    - Fréchet mean instead of arithmetic mean
    - Geodesic distance instead of Euclidean distance
    - Riemannian covariance instead of Euclidean covariance
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def frechet_mean(
        self,
        points: "Array",
        weights: "Array | None" = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        k_neighbors: int | None = None,
        max_k_neighbors: int | None = None,
    ) -> FrechetMeanResult:
        """
        Compute the Fréchet mean (Riemannian center of mass) of a point set.

        The Fréchet mean minimizes the sum of squared geodesic distances:
            μ = argmin_p Σᵢ wᵢ d²(p, xᵢ)

        Uses session-scoped caching to avoid redundant computation when the
        same point set is used multiple times.

        Algorithm:
            1. Initialize at the Euclidean mean (reasonable starting point)
            2. Compute geodesic distances from current estimate to all points
            3. Update estimate using Riemannian gradient descent
            4. Repeat until convergence

        Uses graph-based geodesic distance (Isomap-style): shortest path on k-NN
        graph. This computes exact geodesics on the discrete manifold representation.

        Args:
            points: Point cloud [n, d]
            weights: Optional weights [n] (uniform if None)
            max_iterations: Maximum gradient descent iterations
            tolerance: Convergence threshold for mean position change
            k_neighbors: Optional fixed k for geodesic graph connectivity
            max_k_neighbors: Optional upper bound for adaptive k retries

        Returns:
            FrechetMeanResult with the computed mean
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        d = int(points.shape[1])

        if n == 0:
            return FrechetMeanResult(
                mean=backend.zeros((d,)),
                iterations=0,
                converged=True,
                final_variance=0.0,
            )

        if n == 1:
            return FrechetMeanResult(
                mean=points[0],
                iterations=0,
                converged=True,
                final_variance=0.0,
            )

        k_start = None
        if k_neighbors is not None:
            k_start = max(1, min(int(k_neighbors), n - 1))

        k_max = None
        if max_k_neighbors is not None:
            k_max = max(1, min(int(max_k_neighbors), n - 1))
            if k_start is None:
                k_start = min(10, n - 1)
        else:
            k_max = k_start

        # Initialize weights
        if weights is None:
            weights_arr = backend.ones((n,)) / n
            weights_key = None
        else:
            weights_arr = backend.array(weights)
            # Normalize weights
            weight_sum = backend.sum(weights_arr)
            weights_arr = weights_arr / weight_sum
            weights_key = _cache.make_array_key(weights_arr, backend)

        attempt_k = k_start
        while True:
            cache_key = _cache.make_frechet_key(points, backend, weights_key, attempt_k)
            cached = _cache.get_frechet(cache_key)
            if cached is not None:
                return cached

            start = time.perf_counter()

            # Initialize at weighted Euclidean mean (reasonable starting point for iteration)
            weights_col = backend.reshape(weights_arr, (n, 1))
            mu = backend.sum(points * weights_col, axis=0)

            # Compute geodesic distance matrix once (expensive but reusable, now cached)
            try:
                geo_result = (
                    self.geodesic_distances(points, k_neighbors=attempt_k)
                    if attempt_k is not None
                    else self.geodesic_distances(points)
                )
            except ValueError as exc:
                if self._should_retry_k(exc, attempt_k, k_max):
                    prev_k = attempt_k
                    attempt_k = self._next_k(attempt_k, k_max)
                    logger.warning(
                        "Frechet mean retry after geodesic failure (k=%s -> %s)",
                        prev_k,
                        attempt_k,
                    )
                    continue
                raise

            if attempt_k is not None and k_max is not None and not geo_result.connected:
                if self._should_retry_k(ValueError("disconnected"), attempt_k, k_max):
                    next_k = self._next_k(attempt_k, k_max)
                    if next_k is not None and next_k != attempt_k:
                        logger.warning(
                            "Frechet mean retry after disconnected graph (k=%s -> %s)",
                            attempt_k,
                            next_k,
                        )
                        attempt_k = next_k
                        continue

            # Gradient descent for Fréchet mean
            converged = False
            iterations = 0

            try:
                for it in range(max_iterations):
                    iterations = it + 1

                    # Attach mu to the k-NN graph and compute geodesic distances exactly
                    geo_from_mu = self._geodesic_distances_from_query(
                        points, mu, geo_result=geo_result
                    )

                    # Compute weighted sum of log maps (gradient direction)
                    # On the discrete manifold, log maps are defined by geodesic scaling.
                    new_mu = self._frechet_mean_step(points, mu, geo_from_mu, weights_arr)

                    # Check convergence
                    diff = backend.sqrt(backend.sum((new_mu - mu) ** 2))
                    backend.eval(diff)
                    diff_val = float(backend.to_numpy(diff))

                    if diff_val < tolerance:
                        converged = True
                        mu = new_mu
                        break

                    mu = new_mu
            except ValueError as exc:
                if self._should_retry_k(exc, attempt_k, k_max):
                    next_k = self._next_k(attempt_k, k_max)
                    logger.warning(
                        "Frechet mean retry after log-map failure (k=%s -> %s)",
                        attempt_k,
                        next_k,
                    )
                    attempt_k = next_k
                    continue
                raise

            backend.eval(mu)
            mu_np = backend.to_numpy(mu).flatten()
            non_finite = sum(1 for v in mu_np if not math.isfinite(float(v)))
            if non_finite > 0:
                exc = ValueError(
                    f"Frechet mean contains {non_finite} non-finite values."
                )
                if self._should_retry_k(exc, attempt_k, k_max):
                    next_k = self._next_k(attempt_k, k_max)
                    logger.warning(
                        "Frechet mean retry after non-finite mean (k=%s -> %s)",
                        attempt_k,
                        next_k,
                    )
                    attempt_k = next_k
                    continue
                raise exc

            # Compute final variance (sum of squared geodesic distances)
            final_variance = self._compute_weighted_variance_geodesic(
                points, mu, geo_result, weights_arr
            )

            result = FrechetMeanResult(
                mean=mu,
                iterations=iterations,
                converged=converged,
                final_variance=final_variance,
            )

            # Cache result
            elapsed_ms = (time.perf_counter() - start) * 1000
            _cache.set_frechet(cache_key, result, elapsed_ms)

            return result

    @staticmethod
    def _should_retry_k(
        exc: Exception,
        current_k: int | None,
        max_k: int | None,
    ) -> bool:
        if current_k is None or max_k is None:
            return False
        if current_k >= max_k:
            return False
        message = str(exc)
        return "Log map scale contains" in message or "non-finite" in message or "disconnected" in message

    @staticmethod
    def _next_k(current_k: int | None, max_k: int | None) -> int | None:
        if current_k is None or max_k is None:
            return None
        next_k = max(current_k + 1, current_k * 2)
        return min(next_k, max_k)

    def geodesic_distances(
        self,
        points: "Array",
        k_neighbors: int | None = None,
    ) -> GeodesicDistanceResult:
        """
        Estimate geodesic distances using a k-NN graph and shortest paths.

        This implements the Isomap-style geodesic estimation:
        1. Build a k-NN graph where edge weights are Euclidean distances
        2. Compute shortest paths (geodesics) using Dijkstra's algorithm

        Uses session-scoped caching to avoid redundant computation when the
        same point set is used multiple times (e.g., in frechet_mean,
        riemannian_covariance, and curvature estimation).

        The key insight is that on a curved manifold, the geodesic distance
        follows the manifold surface, while Euclidean distance "cuts through"
        the manifold. For nearby points, geodesic ≈ Euclidean. For distant
        points, geodesic > Euclidean on positive curvature.

        Args:
            points: Point cloud [n, d]
            k_neighbors: Number of neighbors for graph (default: min(10, n-1))

        Returns:
            GeodesicDistanceResult with pairwise geodesic distances
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n <= 1:
            inf_val = float(backend.finfo().max)
            return GeodesicDistanceResult(
                distances=backend.zeros((n, n)),
                adjacency=backend.zeros((n, n)),
                inf_value=inf_val,
                k_neighbors=0,
                connected=True,
            )

        # Default k based on manifold dimension heuristics
        if k_neighbors is None:
            k_neighbors = min(10, n - 1)
        k_neighbors = max(1, min(k_neighbors, n - 1))

        # Check cache first
        cache_key = _cache.make_geodesic_key(points, backend, k_neighbors)
        cached = _cache.get_geodesic(cache_key)
        if cached is not None:
            return cached

        start = time.perf_counter()

        # Compute Euclidean distance matrix
        euclidean_dist = self._euclidean_distance_matrix(points)
        backend.eval(euclidean_dist)
        euclidean_np = backend.to_numpy(euclidean_dist)

        # Build k-NN adjacency and run Floyd-Warshall on backend (no scipy)
        # Use a large finite sentinel derived from dtype to avoid inf arithmetic issues.
        inf_val = float(backend.finfo().max) * 0.25
        adj = backend.full((n, n), inf_val)

        # Set diagonal to zero
        for i in range(n):
            adj = _set_matrix_element(backend, adj, i, i, 0.0)

        # Build symmetric k-NN adjacency
        # Use precision-aware epsilon for edge weight floor
        edge_eps = float(division_epsilon(backend, euclidean_dist))
        for i in range(n):
            # Get distances from point i
            dists = euclidean_np[i, :].tolist()
            # Find k nearest - explicitly exclude self for stability when distances tie
            other_pairs = [(j, dists[j]) for j in range(n) if j != i]
            sorted_pairs = sorted(other_pairs, key=lambda x: x[1])
            nearest_indices = [p[0] for p in sorted_pairs[:k_neighbors]]
            for j in nearest_indices:
                # Symmetric edges - floor at precision-aware epsilon
                edge_weight = max(dists[j], edge_eps)
                adj = _set_matrix_element(backend, adj, i, j, edge_weight)
                adj = _set_matrix_element(backend, adj, j, i, edge_weight)

        backend.eval(adj)

        # Floyd-Warshall on backend: dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        # Vectorized per iteration of k
        geo_dist_arr = adj
        for k in range(n):
            # dist_ik: column k broadcast to all columns
            dist_ik = geo_dist_arr[:, k : k + 1]  # [n, 1]
            # dist_kj: row k broadcast to all rows
            dist_kj = geo_dist_arr[k : k + 1, :]  # [1, n]
            # Path through k
            via_k = dist_ik + dist_kj  # [n, n]
            # Update shortest paths
            geo_dist_arr = backend.minimum(geo_dist_arr, via_k)
            # Periodic eval to avoid graph buildup
            if k % 50 == 0:
                backend.eval(geo_dist_arr)

        backend.eval(geo_dist_arr)

        # Convert to numpy for connectivity check
        geo_np = backend.to_numpy(geo_dist_arr)

        # Mark distances >= inf_val as true infinity (disconnected)
        for i in range(n):
            for j in range(n):
                if geo_np[i, j] >= inf_val * 0.9:  # Near our pseudo-infinity
                    geo_np[i, j] = float("inf")
                elif geo_np[i, j] < 1e-8:  # Near-zero distances are truly zero
                    geo_np[i, j] = 0.0

        # Check connectivity - inf values represent genuinely infinite geodesic distance
        # between disconnected manifold components. No fallback to Euclidean - this is
        # real structural information about the manifold.
        import math

        inf_count = 0
        for i in range(n):
            for j in range(n):
                if math.isinf(geo_np[i, j]):
                    inf_count += 1
        connected = inf_count == 0

        if not connected:
            import logging

            logger = logging.getLogger(__name__)
            n_disconnected = inf_count // 2  # symmetric, so divide by 2
            logger.debug(
                f"k-NN graph has {n_disconnected} disconnected pairs "
                f"(k={k_neighbors}, n={n}). Consider increasing k_neighbors."
            )

        geo_dist = backend.array(geo_np)

        result = GeodesicDistanceResult(
            distances=geo_dist,
            adjacency=adj,
            inf_value=inf_val,
            k_neighbors=k_neighbors,
            connected=connected,
        )

        # Cache result
        elapsed_ms = (time.perf_counter() - start) * 1000
        _cache.set_geodesic(cache_key, result, elapsed_ms)

        return result

    def estimate_local_curvature(
        self,
        points: "Array",
        center_idx: int,
        k_neighbors: int = 10,
    ) -> CurvatureEstimate:
        """
        Estimate local sectional curvature at a point using geodesic defect.

        The geodesic defect compares the ratio of geodesic to Euclidean distances:
        - If geodesic > Euclidean: positive curvature (sphere-like)
        - If geodesic < Euclidean: negative curvature (saddle-like)
        - If geodesic ≈ Euclidean: flat (Euclidean-like)

        This uses the formula from differential geometry relating the geodesic
        excess to sectional curvature via the Jacobi equation.

        Args:
            points: Point cloud [n, d]
            center_idx: Index of the center point
            k_neighbors: Number of neighbors to use for estimation

        Returns:
            CurvatureEstimate with estimated sectional curvature
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n < 3:
            return CurvatureEstimate(
                sectional_curvature=0.0,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        # Get geodesic and Euclidean distances
        geo_result = self.geodesic_distances(points, k_neighbors=k_neighbors)
        euclidean_dist = self._euclidean_distance_matrix(points)

        # Compute precision-aware epsilon before numpy conversion
        eps = division_epsilon(backend, euclidean_dist)

        geo_np = backend.to_numpy(geo_result.distances)
        euc_np = backend.to_numpy(euclidean_dist)

        # Look at the k nearest neighbors of the center point
        center_geo = geo_np[center_idx, :]
        center_euc = euc_np[center_idx, :]

        # Sort by Euclidean distance and take k nearest
        import math

        # argsort manually
        sorted_pairs = sorted(enumerate(center_euc), key=lambda x: x[1])
        sorted_idx = [p[0] for p in sorted_pairs]
        neighbors = sorted_idx[1 : k_neighbors + 1]  # Exclude self

        # Compute geodesic defect: (geodesic - euclidean) / euclidean
        defects = []
        for j in neighbors:
            if center_euc[j] > eps:
                defect = (center_geo[j] - center_euc[j]) / center_euc[j]
                defects.append(defect)

        if len(defects) == 0:
            return CurvatureEstimate(
                sectional_curvature=0.0,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        mean_defect = sum(defects) / len(defects)
        if len(defects) > 1:
            variance = sum((d - mean_defect) ** 2 for d in defects) / len(defects)
            std_defect = math.sqrt(variance)
        else:
            std_defect = 0.0

        # Estimate curvature from defect
        # For a sphere of radius R, geodesic/euclidean ≈ 1 + K*r²/6 for small r
        # where K = 1/R² is the sectional curvature
        # So defect ≈ K*r²/6, giving K ≈ 6*defect/r²

        neighbor_radii = [center_euc[j] for j in neighbors]
        avg_radius = sum(neighbor_radii) / len(neighbor_radii)
        if avg_radius > eps:
            # Rough curvature estimate
            sectional_curvature = 6.0 * mean_defect / (avg_radius * avg_radius)
        else:
            sectional_curvature = 0.0

        # Confidence based on consistency of defects
        confidence = 1.0 / (1.0 + std_defect) if std_defect > 0 else 1.0

        return CurvatureEstimate(
            sectional_curvature=sectional_curvature,
            is_positive=sectional_curvature > 0.01,
            is_negative=sectional_curvature < -0.01,
            confidence=confidence,
        )

    def riemannian_covariance(
        self,
        points: "Array",
        mean: "Array | None" = None,
    ) -> "Array":
        """
        Compute covariance matrix in the tangent space at the Fréchet mean.

        On a Riemannian manifold, covariance is computed by:
        1. Finding the Fréchet mean μ
        2. Mapping all points to the tangent space at μ via Log_μ
        3. Computing Euclidean covariance in the tangent space

        For high-dimensional representations, we compute Log_μ(x) on the
        discrete manifold as the direction from μ to x scaled by the
        geodesic distance.

        Args:
            points: Point cloud [n, d]
            mean: Precomputed Fréchet mean (computed if None)

        Returns:
            Covariance matrix [d, d] in the tangent space
        """
        backend = self._backend
        points = backend.array(points)
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
        geo_result = self.geodesic_distances(points)

        # Map points to tangent space at mean
        # Log_μ(x) = (x - μ) * (geodesic_dist / euclidean_dist)
        tangent_vectors = self._log_map_approximate(points, mean, geo_result)

        # Standard covariance in tangent space
        tangent_mean = backend.mean(tangent_vectors, axis=0, keepdims=True)
        centered = tangent_vectors - tangent_mean

        cov = backend.matmul(backend.transpose(centered), centered) / (n - 1)

        return cov

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
        the geodesic path and interpolates along it. The geodesic is the
        shortest path on the k-NN graph - exact for the discrete manifold.

        Algorithm:
            1. Project p1, p2 onto the discrete manifold (find nearest points)
            2. Reconstruct shortest path from geodesic distance matrix
            3. Compute cumulative arc lengths along path
            4. Interpolate along the path at parameter t

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

        if len(path_indices) == 2:
            # Direct neighbors on the graph - interpolate along this edge
            # This is exact for the discrete manifold (edge IS the geodesic)
            proj1 = points_context[idx1]
            proj2 = points_context[idx2]
            return (1.0 - t) * proj1 + t * proj2

        # 4. Compute cumulative arc lengths along path
        arc_lengths = self._compute_path_arc_lengths(points_context, path_indices)
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

        This is the triangle equality (not inequality) that holds exactly
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
        geo_np = backend.to_numpy(geo_dist)
        n = geo_np.shape[0]

        total_dist = float(geo_np[start_idx, end_idx])

        if math.isinf(total_dist):
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

        while current != end_idx:
            # Find next point: must satisfy triangle equality
            # d(current, next) + d(next, end) ≈ d(current, end)
            dist_to_end = float(geo_np[current, end_idx])

            best_next = end_idx
            best_dist = float(geo_np[current, end_idx])

            for candidate in range(n):
                if candidate == current or candidate in path:
                    continue

                d_to_candidate = float(geo_np[current, candidate])
                d_candidate_to_end = float(geo_np[candidate, end_idx])

                if math.isinf(d_to_candidate) or math.isinf(d_candidate_to_end):
                    continue

                # Check triangle equality (point is on geodesic)
                path_through_candidate = d_to_candidate + d_candidate_to_end

                if abs(path_through_candidate - dist_to_end) <= tolerance:
                    # Candidate is on the geodesic - pick the one closest to current
                    if d_to_candidate < best_dist:
                        best_next = candidate
                        best_dist = d_to_candidate

            path.append(best_next)
            current = best_next

            # Safety: prevent infinite loops
            if len(path) > n:
                break

        return path

    def _compute_path_arc_lengths(
        self,
        points: "Array",
        path_indices: list[int],
    ) -> list[float]:
        """
        Compute cumulative arc lengths along a path.

        Uses Euclidean distance between consecutive points on the path.
        This gives the actual length traveled along the discrete geodesic.

        Args:
            points: Point cloud [n, d]
            path_indices: Indices forming the path

        Returns:
            List of cumulative arc lengths (first element is 0)
        """
        backend = self._backend

        if len(path_indices) <= 1:
            return [0.0]

        arc_lengths = [0.0]
        cumulative = 0.0

        for i in range(len(path_indices) - 1):
            p1 = points[path_indices[i]]
            p2 = points[path_indices[i + 1]]
            diff = p2 - p1
            dist = backend.sqrt(backend.sum(diff * diff))
            backend.eval(dist)
            cumulative += float(backend.to_numpy(dist))
            arc_lengths.append(cumulative)

        return arc_lengths

    def _interpolate_along_path(
        self,
        points: "Array",
        path_indices: list[int],
        arc_lengths: list[float],
        target_length: float,
    ) -> "Array":
        """
        Interpolate along a discrete path at a given arc length.

        Finds the segment containing the target length and performs
        linear interpolation within that segment.

        Args:
            points: Point cloud [n, d]
            path_indices: Indices forming the path
            arc_lengths: Cumulative arc lengths at each path point
            target_length: Target arc length for interpolation

        Returns:
            Interpolated point [d]
        """
        backend = self._backend

        # Use precision-aware threshold for near-zero detection
        eps = division_epsilon(backend, points)

        # Find the segment containing target_length
        for i in range(len(arc_lengths) - 1):
            if arc_lengths[i] <= target_length <= arc_lengths[i + 1]:
                # Interpolate within this segment
                segment_start = arc_lengths[i]
                segment_end = arc_lengths[i + 1]
                segment_length = segment_end - segment_start

                if segment_length < eps:
                    return points[path_indices[i]]

                # Local interpolation parameter within segment
                local_t = (target_length - segment_start) / segment_length

                p1 = points[path_indices[i]]
                p2 = points[path_indices[i + 1]]

                return (1.0 - local_t) * p1 + local_t * p2

        # Fallback: return last point if target exceeds path length
        return points[path_indices[-1]]

    def farthest_point_sampling(
        self,
        points: "Array",
        n_samples: int,
        seed_idx: int = 0,
        k_neighbors: int | None = None,
    ) -> FarthestPointSamplingResult:
        """
        Select points via geodesic farthest point sampling (maximin design).

        FPS iteratively selects the point that maximizes the minimum geodesic
        distance to the already-selected set. This provides optimal coverage
        of the manifold with a given number of samples.

        Algorithm (O(n * n_samples) with precomputed geodesic matrix):
            1. Start with seed point
            2. For each new sample:
               - Compute min geodesic distance from each point to selected set
               - Select the point with maximum min-distance
            3. Return selected indices

        This is the geodesic analog of Euclidean FPS, respecting the
        manifold's intrinsic geometry.

        Args:
            points: Point cloud [n, d]
            n_samples: Number of points to select
            seed_idx: Starting point index (default: 0)
            k_neighbors: k for geodesic graph (default: auto)

        Returns:
            FarthestPointSamplingResult with selected indices and coverage stats
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        if n == 0:
            return FarthestPointSamplingResult(
                selected_indices=[],
                min_distances=backend.zeros((0,)),
                coverage_radius=0.0,
            )

        n_samples = max(1, min(n_samples, n))
        seed_idx = max(0, min(seed_idx, n - 1))

        # Compute geodesic distances (cached)
        geo_result = self.geodesic_distances(points, k_neighbors=k_neighbors)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        # Initialize: select seed
        selected = [seed_idx]

        # Min distance from each point to the selected set
        # Initially, just distance to seed
        min_distances = geo_dist[seed_idx]
        backend.eval(min_distances)

        # Iteratively select farthest point
        for _ in range(n_samples - 1):
            # Find point with maximum min-distance to selected set
            min_dist_np = backend.to_numpy(min_distances).flatten()

            # Exclude already selected points by setting their distance to -inf
            for idx in selected:
                min_dist_np[idx] = float("-inf")

            # Find farthest (handle inf for disconnected points)
            farthest_idx = 0
            farthest_dist = float("-inf")
            for i, d in enumerate(min_dist_np):
                if not math.isinf(d) or d > 0:  # Skip -inf (selected) but allow +inf
                    if d > farthest_dist:
                        farthest_dist = d
                        farthest_idx = i

            selected.append(farthest_idx)

            # Update min distances: element-wise minimum with new point's distances
            new_dists = geo_dist[farthest_idx]
            min_distances = backend.minimum(min_distances, new_dists)
            backend.eval(min_distances)

        # Compute coverage radius (max of final min-distances, excluding selected)
        final_min_np = backend.to_numpy(min_distances).flatten()
        coverage_vals = [
            d for i, d in enumerate(final_min_np)
            if i not in selected and not math.isinf(d)
        ]
        coverage_radius = max(coverage_vals) if coverage_vals else 0.0

        return FarthestPointSamplingResult(
            selected_indices=selected,
            min_distances=min_distances,
            coverage_radius=coverage_radius,
        )

    def directional_coverage(
        self,
        point_idx: int,
        points: "Array",
        k: int = 10,
        n_candidates: int = 100,
    ) -> DirectionalCoverage:
        """
        Analyze directional coverage in tangent space at a point.

        Finds the most under-sampled direction by analyzing the angular
        distribution of neighbors projected onto the tangent sphere.

        Algorithm:
            1. Get k nearest neighbors (by geodesic distance)
            2. Compute tangent vectors to each neighbor
            3. Normalize to unit sphere (tangent sphere S^{d-1})
            4. Find largest angular gap via candidate sampling
            5. Return the sparse direction and coverage metrics

        The sparse direction identifies where to explore for better coverage.

        Args:
            point_idx: Index of the center point
            points: Point cloud [n, d]
            k: Number of neighbors to analyze
            n_candidates: Number of random directions to test for gap finding

        Returns:
            DirectionalCoverage with sparse direction and metrics
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        d = int(points.shape[1])
        point_idx = max(0, min(point_idx, n - 1))
        k = max(1, min(k, n - 1))

        center = points[point_idx]

        # Get geodesic distances for neighbor selection
        geo_result = self.geodesic_distances(points, k_neighbors=k)
        geo_dist_np = backend.to_numpy(geo_result.distances)

        # Find k nearest neighbors by geodesic distance
        center_dists = geo_dist_np[point_idx, :].tolist()
        sorted_pairs = sorted(enumerate(center_dists), key=lambda x: (x[1], x[0]))
        # Exclude self (distance 0)
        neighbors = [idx for idx, dist in sorted_pairs if idx != point_idx][:k]

        if len(neighbors) == 0:
            # Isolated point - any direction is sparse
            sparse_dir = backend.zeros((d,))
            if d > 0:
                sparse_dir = _set_matrix_element(
                    backend, backend.reshape(sparse_dir, (1, d)), 0, 0, 1.0
                )
                sparse_dir = backend.reshape(sparse_dir, (d,))
            return DirectionalCoverage(
                sparse_direction=sparse_dir,
                max_gap_angle=math.pi,  # Full hemisphere is empty
                coverage_uniformity=0.0,
                neighbor_directions=backend.zeros((0, d)),
                point_idx=point_idx,
            )

        # Compute tangent vectors to neighbors
        neighbor_pts = backend.stack([points[i] for i in neighbors], axis=0)
        tangent_vecs = neighbor_pts - backend.reshape(center, (1, d))

        # Normalize to unit tangent sphere
        norms = backend.sqrt(backend.sum(tangent_vecs * tangent_vecs, axis=1, keepdims=True))
        eps = division_epsilon(backend, norms)
        norms_safe = backend.maximum(norms, backend.full(norms.shape, eps))
        tangent_dirs = tangent_vecs / norms_safe
        backend.eval(tangent_dirs)

        # Find sparse direction by sampling candidates on the unit sphere
        # Generate random unit vectors
        backend.random_seed(42)  # Deterministic for reproducibility
        candidates = backend.random_normal((n_candidates, d))
        cand_norms = backend.sqrt(
            backend.sum(candidates * candidates, axis=1, keepdims=True)
        )
        cand_norms_safe = backend.maximum(cand_norms, backend.full(cand_norms.shape, eps))
        candidates = candidates / cand_norms_safe
        backend.eval(candidates)

        # For each candidate, find minimum cosine similarity to any neighbor direction
        # (cosine = 1 means same direction, -1 means opposite)
        # We want the candidate with the smallest maximum similarity (furthest from all)
        # Equivalently: largest angular gap

        # Compute dot products: candidates @ tangent_dirs.T -> [n_candidates, k_actual]
        similarities = backend.matmul(candidates, backend.transpose(tangent_dirs))
        backend.eval(similarities)

        # For each candidate, find the maximum similarity (closest neighbor direction)
        max_sims = backend.max(similarities, axis=1)  # [n_candidates]
        backend.eval(max_sims)

        # The sparse direction is the candidate with minimum max-similarity
        max_sims_np = backend.to_numpy(max_sims).flatten()
        sparse_idx = 0
        min_max_sim = float("inf")
        for i, s in enumerate(max_sims_np):
            if s < min_max_sim:
                min_max_sim = s
                sparse_idx = i

        sparse_direction = candidates[sparse_idx]

        # Convert max similarity to angle: theta = arccos(similarity)
        # The "gap" is the angle to the nearest neighbor direction
        max_gap_angle = math.acos(max(min(-1.0, min_max_sim), 1.0)) if abs(min_max_sim) <= 1.0 else 0.0
        # Fix: arccos domain is [-1, 1]
        clamped_sim = max(-1.0, min(1.0, min_max_sim))
        max_gap_angle = math.acos(clamped_sim)

        # Coverage uniformity: ideal is uniform distribution on sphere
        # Measure as 1 - (variance of similarities)
        # If all neighbors are in one direction, variance is high -> low uniformity
        sim_mean = sum(max_sims_np) / len(max_sims_np)
        sim_var = sum((s - sim_mean) ** 2 for s in max_sims_np) / len(max_sims_np)
        # Normalize variance to [0, 1] range (max variance for similarities is ~1)
        coverage_uniformity = max(0.0, 1.0 - sim_var)

        return DirectionalCoverage(
            sparse_direction=sparse_direction,
            max_gap_angle=max_gap_angle,
            coverage_uniformity=coverage_uniformity,
            neighbor_directions=tangent_dirs,
            point_idx=point_idx,
        )

    def propose_in_sparse_direction(
        self,
        point_idx: int,
        points: "Array",
        step_size: float,
        k: int = 10,
    ) -> "Array":
        """
        Propose a new point by stepping in the sparsest tangent direction.

        This implements tangent space exploration: identify the most
        under-sampled direction at a point and propose a new point
        in that direction via the exponential map.

        For the discrete manifold, we use a first-order approximation:
            x_new = x + step_size * sparse_direction

        This is exact for flat manifolds and a good approximation for
        small step sizes on curved manifolds.

        Args:
            point_idx: Index of the base point
            points: Point cloud [n, d]
            step_size: Distance to step in the sparse direction
            k: Number of neighbors for directional analysis

        Returns:
            Proposed new point [d]
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        point_idx = max(0, min(point_idx, n - 1))

        # Get directional coverage analysis
        coverage = self.directional_coverage(point_idx, points, k=k)

        # Base point
        base = points[point_idx]

        # Exponential map approximation: x_new = x + step_size * v
        # where v is the unit sparse direction
        proposed = base + step_size * coverage.sparse_direction

        return proposed

    # --- Private helper methods ---

    def _euclidean_distance_matrix(self, points: "Array") -> "Array":
        """Compute pairwise Euclidean distances."""
        backend = self._backend
        norms = backend.sum(points * points, axis=1, keepdims=True)
        dots = backend.matmul(points, backend.transpose(points))
        dist_sq = norms + backend.transpose(norms) - 2.0 * dots
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        return backend.sqrt(dist_sq)

    def _geodesic_distances_from_query(
        self,
        points: "Array",
        query: "Array",
        geo_result: GeodesicDistanceResult | None = None,
        k_neighbors: int | None = None,
    ) -> "Array":
        """Compute geodesic distances from an out-of-sample query point.

        The query is attached to the existing k-NN graph by its k closest
        neighbors, then shortest paths are computed exactly on the augmented
        discrete manifold. This is deterministic and preserves manifold geometry.
        """
        backend = self._backend
        points = backend.array(points)
        query = backend.array(query)
        backend.eval(points, query)

        n = int(points.shape[0])
        if n == 0:
            return backend.zeros((0,))

        if geo_result is None:
            geo_result = self.geodesic_distances(points, k_neighbors=k_neighbors)

        if k_neighbors is None:
            k_neighbors = geo_result.k_neighbors
        k_neighbors = max(1, min(int(k_neighbors), n))

        # Euclidean distances from query to all points (graph edge weights)
        diff = points - backend.reshape(query, (1, -1))
        euc_dist = backend.sqrt(backend.sum(diff * diff, axis=1))
        backend.eval(euc_dist)

        # Deterministic neighbor selection with index tie-breaker
        euc_list = backend.to_numpy(euc_dist).tolist()
        sorted_pairs = sorted(enumerate(euc_list), key=lambda x: (x[1], x[0]))
        neighbors = [idx for idx, _ in sorted_pairs[:k_neighbors]]

        neighbors_arr = backend.array(neighbors)
        neighbor_dists = backend.take(euc_dist, neighbors_arr, axis=0)
        geo_rows = backend.take(geo_result.distances, neighbors_arr, axis=0)

        # Exact geodesic distances for the augmented graph:
        # d(q, i) = min_j (d(q, j) + d(j, i))
        weights_col = backend.reshape(neighbor_dists, (len(neighbors), 1))
        candidates = geo_rows + weights_col
        geo_from_query = backend.min(candidates, axis=0)

        return geo_from_query

    def _find_nearest_point(
        self,
        points: "Array",
        query: "Array",
        geo_result: GeodesicDistanceResult,
    ) -> int:
        """Find the geodesic-nearest point to query on the discrete manifold."""
        backend = self._backend

        geo_from_query = self._geodesic_distances_from_query(
            points, query, geo_result=geo_result
        )
        backend.eval(geo_from_query)
        geo_list = backend.to_numpy(geo_from_query).tolist()

        min_val = geo_list[0]
        min_idx = 0
        for i, v in enumerate(geo_list[1:], 1):
            if v < min_val:
                min_val = v
                min_idx = i
        return min_idx

    def _frechet_mean_step(
        self,
        points: "Array",
        mu: "Array",
        geo_from_mu: "Array",
        weights: "Array",
    ) -> "Array":
        """
        Perform one step of Fréchet mean gradient descent.

        The update is: μ_new = μ + η * Σᵢ wᵢ * log_μ(xᵢ)

        Log maps are defined by the discrete manifold's geodesic scaling.
        The geodesic/Euclidean ratio captures the curvature correction:
        - ratio > 1: negative curvature (geodesic longer than Euclidean)
        - ratio < 1: positive curvature (geodesic shorter than Euclidean)
        - ratio = 1: flat space (geodesic equals Euclidean)

        NO CLAMPING: We use the true geodesic/Euclidean ratio. Extreme values
        indicate extreme curvature and should be handled by adjusting k_neighbors
        or using a different algorithm, not by silently corrupting the geometry.

        Raises:
            ValueError: If geodesic/Euclidean scale contains inf or nan values,
                indicating disconnected manifold components or coincident points.
        """
        backend = self._backend

        # Euclidean distances from mu
        diff = points - backend.reshape(mu, (1, -1))
        euc_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        # Compute scaling factor: geodesic / euclidean
        # This corrects the tangent vector length for curvature
        # Use precision-aware floor for safe division
        eps = division_epsilon(backend, euc_dist)
        euc_dist_safe = backend.maximum(euc_dist, backend.full(euc_dist.shape, eps))
        scale = geo_from_mu / euc_dist_safe

        # NO CLAMPING - use true geodesic/Euclidean ratio
        # The ratio IS the curvature signal. Clamping corrupts the geometry.
        #
        # Extreme scales indicate:
        # - ratio >> 1: Strong negative curvature (hyperbolic-like)
        # - ratio << 1: Strong positive curvature (sphere-like)
        # - ratio = inf: Disconnected components (geodesic = inf)
        # - ratio = 0: Points coincide (both distances = 0)
        #
        # Handle inf/nan from disconnected or coincident points
        backend.eval(scale)
        scale_np = backend.to_numpy(scale)

        # Check for numerical issues that indicate manifold problems
        inf_count = sum(1 for s in scale_np.flatten() if math.isinf(float(s)))
        nan_count = sum(1 for s in scale_np.flatten() if math.isnan(float(s)))

        if inf_count > 0 or nan_count > 0:
            raise ValueError(
                f"Geodesic/Euclidean scale contains {inf_count} inf and {nan_count} nan values. "
                f"This indicates disconnected manifold components or coincident points. "
                f"Increase k_neighbors to improve graph connectivity, or check for "
                f"duplicate points in the input."
            )

        # Log extreme curvature for diagnostics (not clamping, just reporting)
        n = len(scale_np)
        extreme_neg_count = sum(1 for s in scale_np.flatten() if float(s) > 5.0)
        extreme_pos_count = sum(1 for s in scale_np.flatten() if float(s) < 0.2)

        if extreme_neg_count > n * 0.1 or extreme_pos_count > n * 0.1:
            logger.warning(
                f"Extreme curvature detected in Fréchet mean: "
                f"{extreme_neg_count}/{n} points with scale > 5.0 (strong negative curvature), "
                f"{extreme_pos_count}/{n} points with scale < 0.2 (strong positive curvature). "
                f"This may cause slow convergence. Consider increasing k_neighbors."
            )

        # Weighted sum of scaled tangent vectors (log maps)
        scale_col = backend.reshape(scale, (-1, 1))
        weights_col = backend.reshape(weights, (-1, 1))
        log_vectors = diff * scale_col

        # Gradient is the weighted mean of log vectors
        gradient = backend.sum(log_vectors * weights_col, axis=0)

        # Step size (could be adaptive, using fixed for simplicity)
        eta = 0.5

        # Update
        new_mu = mu + eta * gradient

        return new_mu

    def _compute_weighted_variance_geodesic(
        self,
        points: "Array",
        mean: "Array",
        geo_result: GeodesicDistanceResult,
        weights: "Array",
    ) -> float:
        """Compute weighted variance using geodesic distance."""
        backend = self._backend

        # Attach mean to the k-NN graph for exact discrete geodesics
        geo_from_mean = self._geodesic_distances_from_query(
            points, mean, geo_result=geo_result
        )

        # Weighted sum of squared geodesic distances
        variance = backend.sum(geo_from_mean * geo_from_mean * weights)
        backend.eval(variance)
        return float(backend.to_numpy(variance))

    def _log_map_approximate(
        self,
        points: "Array",
        mean: "Array",
        geo_result: GeodesicDistanceResult,
    ) -> "Array":
        """
        Compute logarithmic map from mean to all points.

        log_μ(x) = (x - μ) * (geodesic_dist / euclidean_dist)

        This scales the Euclidean tangent vector by the ratio of
        geodesic to Euclidean distance, accounting for curvature.

        NO CLAMPING: Uses true geodesic/Euclidean ratio. See _frechet_mean_step
        for detailed explanation of why clamping corrupts the geometry.

        Raises:
            ValueError: If scale contains inf or nan values.
        """
        backend = self._backend

        # Attach mean to the k-NN graph for exact discrete geodesics
        geo_from_mean = self._geodesic_distances_from_query(
            points, mean, geo_result=geo_result
        )

        # Euclidean vectors from mean
        diff = points - backend.reshape(mean, (1, -1))
        euc_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        # Scale factor
        # Use precision-aware floor for safe division
        eps = division_epsilon(backend, euc_dist)
        euc_safe = backend.maximum(euc_dist, backend.full(euc_dist.shape, eps))
        scale = geo_from_mean / euc_safe

        # NO CLAMPING - use true geodesic/Euclidean ratio
        # See _frechet_mean_step for detailed explanation.
        backend.eval(scale)
        scale_np = backend.to_numpy(scale)

        # Check for numerical issues
        inf_count = sum(1 for s in scale_np.flatten() if math.isinf(float(s)))
        nan_count = sum(1 for s in scale_np.flatten() if math.isnan(float(s)))

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


# Convenience functions


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

    rg = RiemannianGeometry(backend)
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

    rg = RiemannianGeometry(backend)
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

    rg = RiemannianGeometry(backend)
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

    rg = RiemannianGeometry(backend)
    result = rg.directional_coverage(point_idx, points, k=k)
    return result.sparse_direction
