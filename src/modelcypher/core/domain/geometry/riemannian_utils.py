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

import heapq
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Session-scoped cache for geodesic distances and Fréchet means
_cache = ComputationCache.shared()


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
    k_neighbors: int
    connected: bool  # Whether the graph is fully connected


@dataclass(frozen=True)
class CurvatureEstimate:
    """Local curvature estimate at a point."""

    sectional_curvature: float  # Estimated sectional curvature
    is_positive: bool  # Positive curvature (sphere-like)
    is_negative: bool  # Negative curvature (hyperbolic-like)
    confidence: float  # Confidence in the estimate [0, 1]


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
        graph. This IS the geodesic on the discrete manifold representation -
        exact, not an approximation.

        Args:
            points: Point cloud [n, d]
            weights: Optional weights [n] (uniform if None)
            max_iterations: Maximum gradient descent iterations
            tolerance: Convergence threshold for mean position change

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

        # Check cache
        cache_key = _cache.make_frechet_key(points, backend, weights_key)
        cached = _cache.get_frechet(cache_key)
        if cached is not None:
            return cached

        start = time.perf_counter()

        # Initialize at weighted Euclidean mean (reasonable starting point for iteration)
        weights_col = backend.reshape(weights_arr, (n, 1))
        mu = backend.sum(points * weights_col, axis=0)

        # Compute geodesic distance matrix once (expensive but reusable, now cached)
        geo_result = self.geodesic_distances(points)
        geo_dist = geo_result.distances

        # Gradient descent for Fréchet mean
        converged = False
        iterations = 0

        for it in range(max_iterations):
            iterations = it + 1

            # Find the point in our set closest to current mu using Euclidean distance
            # Note: We use Euclidean here because mu is not in the dataset, so we can't
            # look up its geodesic distances directly. The geodesic distances are used
            # for computing the gradient (via log maps) and variance, not for projection.
            mu_idx = self._find_nearest_point(points, mu, geo_dist=None)

            # Compute weighted sum of log maps (gradient direction)
            # In the tangent space at mu, log_mu(x_i) ≈ x_i - mu for small distances
            # For graph geodesics, we use the direction from mu to each point
            # weighted by the geodesic distance ratio

            new_mu = self._frechet_mean_step(
                points, mu, mu_idx, geo_dist, weights_arr
            )

            # Check convergence
            diff = backend.sqrt(backend.sum((new_mu - mu) ** 2))
            backend.eval(diff)
            diff_val = float(backend.to_numpy(diff))

            if diff_val < tolerance:
                converged = True
                mu = new_mu
                break

            mu = new_mu

        # Compute final variance (sum of squared geodesic distances)
        final_variance = self._compute_weighted_variance_geodesic(
            points, mu, geo_dist, weights_arr
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
            return GeodesicDistanceResult(
                distances=backend.zeros((n, n)),
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

        # Build k-NN adjacency for scipy's floyd_warshall
        # NOTE: scipy's C-optimized graph algorithm requires numpy-compatible input
        from scipy.sparse.csgraph import floyd_warshall

        # For each point, find k nearest neighbors
        # Use backend to create infinity-filled matrix, then convert for scipy
        inf_val = float("inf")
        adj_list = [[inf_val] * n for _ in range(n)]
        for i in range(n):
            adj_list[i][i] = 0.0

        for i in range(n):
            # Get distances from point i
            dists = euclidean_np[i, :].tolist()
            # Find k nearest (excluding self) - manual argsort
            sorted_pairs = sorted(enumerate(dists), key=lambda x: x[1])
            nearest_indices = [p[0] for p in sorted_pairs[1 : k_neighbors + 1]]
            for j in nearest_indices:
                # Symmetric edges
                adj_list[i][j] = dists[j]
                adj_list[j][i] = dists[j]

        # Floyd-Warshall for all-pairs shortest paths (scipy's C-optimized version)
        # scipy requires numpy array input - this is the backend-to-scipy interface
        geo_np = floyd_warshall(adj_list, directed=False)

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
            if center_euc[j] > 1e-10:
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
        if avg_radius > 1e-10:
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

        For high-dimensional representations, we approximate Log_μ(x) as
        the direction from μ to x, scaled by the geodesic distance.

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
        geo_dist = geo_result.distances

        # Map points to tangent space at mean
        # Log_μ(x) ≈ (x - μ) * (geodesic_dist / euclidean_dist)
        tangent_vectors = self._log_map_approximate(points, mean, geo_dist)

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
            # No manifold context - linear interpolation is the best we can do
            return (1.0 - t) * p1 + t * p2

        points_context = backend.array(points_context)
        backend.eval(points_context)
        n = int(points_context.shape[0])

        if n < 2:
            return (1.0 - t) * p1 + t * p2

        # 1. Compute geodesic distances
        geo_result = self.geodesic_distances(points_context)

        # 2. Project p1 and p2 onto the discrete manifold
        idx1 = self._find_nearest_point(points_context, p1, geo_dist=geo_result.distances)
        idx2 = self._find_nearest_point(points_context, p2, geo_dist=geo_result.distances)

        if idx1 == idx2:
            # Same projection - linear interpolation in tangent space
            return (1.0 - t) * p1 + t * p2

        # 3. Reconstruct geodesic path
        path_indices = self._reconstruct_geodesic_path(
            geo_result.distances, idx1, idx2
        )

        if len(path_indices) <= 1:
            # Path reconstruction failed - fall back to linear
            return (1.0 - t) * p1 + t * p2

        if len(path_indices) == 2:
            # Direct neighbors - linear interpolation between projections
            proj1 = points_context[idx1]
            proj2 = points_context[idx2]
            return (1.0 - t) * proj1 + t * proj2

        # 4. Compute cumulative arc lengths along path
        arc_lengths = self._compute_path_arc_lengths(points_context, path_indices)
        total_length = arc_lengths[-1]

        if total_length < 1e-10:
            return (1.0 - t) * p1 + t * p2

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

        if total_dist < 1e-10:
            # Same point
            return [start_idx]

        # Greedy path reconstruction: at each step, find the next point on the path
        path = [start_idx]
        current = start_idx
        tolerance = 1e-6 * total_dist  # Tolerance for floating point comparison

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

        # Find the segment containing target_length
        for i in range(len(arc_lengths) - 1):
            if arc_lengths[i] <= target_length <= arc_lengths[i + 1]:
                # Interpolate within this segment
                segment_start = arc_lengths[i]
                segment_end = arc_lengths[i + 1]
                segment_length = segment_end - segment_start

                if segment_length < 1e-10:
                    return points[path_indices[i]]

                # Local interpolation parameter within segment
                local_t = (target_length - segment_start) / segment_length

                p1 = points[path_indices[i]]
                p2 = points[path_indices[i + 1]]

                return (1.0 - local_t) * p1 + local_t * p2

        # Fallback: return last point if target exceeds path length
        return points[path_indices[-1]]

    # --- Private helper methods ---

    def _euclidean_distance_matrix(self, points: "Array") -> "Array":
        """Compute pairwise Euclidean distances."""
        backend = self._backend
        norms = backend.sum(points * points, axis=1, keepdims=True)
        dots = backend.matmul(points, backend.transpose(points))
        dist_sq = norms + backend.transpose(norms) - 2.0 * dots
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        return backend.sqrt(dist_sq)

    def _find_nearest_point(
        self,
        points: "Array",
        query: "Array",
        geo_dist: "Array | None" = None,
    ) -> int:
        """Find the index of the point nearest to query.

        In curved spaces, the Euclidean-nearest point is NOT the geodesic-nearest
        point. When geo_dist is provided, this method uses geodesic distances from
        a reference point to find the true nearest neighbor on the manifold.

        Args:
            points: Data points [n, d]
            query: Query point [d]
            geo_dist: Optional precomputed geodesic distance matrix [n, n].
                      When provided, uses geodesic distance for nearest neighbor.

        Returns:
            Index of the nearest point (geodesic if geo_dist provided, else Euclidean)
        """
        backend = self._backend

        # Always compute Euclidean distances (needed for reference point selection)
        diff = points - backend.reshape(query, (1, -1))
        euc_dists_sq = backend.sum(diff * diff, axis=1)
        backend.eval(euc_dists_sq)
        euc_dists_sq_list = backend.to_numpy(euc_dists_sq).tolist()

        if geo_dist is None:
            # Euclidean fallback - only for initialization before geo_dist computed
            dists_list = euc_dists_sq_list
        else:
            # Use geodesic distances from the Euclidean-nearest reference point
            # Find Euclidean nearest as reference
            ref_idx = min(range(len(euc_dists_sq_list)), key=lambda i: euc_dists_sq_list[i])

            # Get geodesic distances from reference point
            backend.eval(geo_dist)
            geo_from_ref = backend.to_numpy(geo_dist[ref_idx, :])

            # For query not in dataset, approximate geodesic distance using
            # triangle inequality bounds. The geodesic distance from query to
            # point i is bounded by:
            #   |d_geo(ref, i) - d_euc(query, ref)| <= d_geo(query, i) <= d_geo(ref, i) + d_euc(query, ref)
            # We use the geodesic from ref as the primary signal, adjusted by
            # the query's offset from ref.
            query_to_ref_euc = math.sqrt(euc_dists_sq_list[ref_idx])

            dists_list = []
            for i in range(len(euc_dists_sq_list)):
                if i == ref_idx:
                    # Distance to reference is just Euclidean from query
                    dists_list.append(query_to_ref_euc)
                else:
                    # Use geodesic from ref, with Euclidean offset as tiebreaker
                    # This prioritizes geodesic structure while accounting for
                    # query's position relative to the manifold
                    geo_from_ref_i = float(geo_from_ref[i])
                    if math.isinf(geo_from_ref_i):
                        # Disconnected component - use large value
                        dists_list.append(float("inf"))
                    else:
                        # Weighted combination: geodesic dominates, Euclidean breaks ties
                        euc_from_query = math.sqrt(euc_dists_sq_list[i])
                        dists_list.append(geo_from_ref_i + 0.01 * euc_from_query)

        # argmin
        min_val = dists_list[0]
        min_idx = 0
        for i, v in enumerate(dists_list[1:], 1):
            if v < min_val:
                min_val = v
                min_idx = i
        return min_idx

    def _frechet_mean_step(
        self,
        points: "Array",
        mu: "Array",
        mu_idx: int,
        geo_dist: "Array",
        weights: "Array",
    ) -> "Array":
        """
        Perform one step of Fréchet mean gradient descent.

        The update is: μ_new = μ + η * Σᵢ wᵢ * log_μ(xᵢ)

        We approximate log_μ(xᵢ) using the tangent vector direction
        scaled by geodesic distance.

        Scale Clamping:
            The geodesic/Euclidean ratio is clamped to [0.5, 2.0]. This is valid for:
            - Positive curvature: geodesic ≤ Euclidean (ratio ≤ 1.0 for sphere)
            - Negative curvature: geodesic > Euclidean (ratio can exceed 2.0 for
              strongly hyperbolic spaces)

            The [0.5, 2.0] bounds are conservative for most neural network
            representation spaces. If significant clamping occurs, the manifold
            may have extreme curvature requiring more sophisticated treatment.
        """
        backend = self._backend

        # Euclidean distances from mu
        diff = points - backend.reshape(mu, (1, -1))
        euc_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        # Get geodesic distances from the nearest point to mu
        geo_from_mu = geo_dist[mu_idx, :]

        # Compute scaling factor: geodesic / euclidean
        # This corrects the tangent vector length for curvature
        euc_dist_safe = backend.maximum(euc_dist, backend.full(euc_dist.shape, 1e-10))
        scale = geo_from_mu / euc_dist_safe

        # Count clamping events before applying clamps
        backend.eval(scale)
        scale_np = backend.to_numpy(scale)
        n = len(scale_np)
        high_clamp_count = sum(1 for s in scale_np.flatten() if float(s) > 2.0)
        low_clamp_count = sum(1 for s in scale_np.flatten() if float(s) < 0.5)

        if high_clamp_count > 0 or low_clamp_count > 0:
            logger.debug(
                f"Scale clamping in Fréchet mean: {high_clamp_count}/{n} high (>2.0), "
                f"{low_clamp_count}/{n} low (<0.5). High clamping suggests negative "
                f"curvature; low clamping suggests positive curvature."
            )

        # Clamp scale to avoid extreme values
        # See docstring for explanation of bounds
        scale = backend.minimum(scale, backend.full(scale.shape, 2.0))
        scale = backend.maximum(scale, backend.full(scale.shape, 0.5))

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
        geo_dist: "Array",
        weights: "Array",
    ) -> float:
        """Compute weighted variance using geodesic distance."""
        backend = self._backend

        # Find nearest point to mean using geodesic distance
        mean_idx = self._find_nearest_point(points, mean, geo_dist=geo_dist)

        # Get geodesic distances from that point
        geo_from_mean = geo_dist[mean_idx, :]

        # Weighted sum of squared geodesic distances
        variance = backend.sum(geo_from_mean * geo_from_mean * weights)
        backend.eval(variance)
        return float(backend.to_numpy(variance))

    def _log_map_approximate(
        self,
        points: "Array",
        mean: "Array",
        geo_dist: "Array",
    ) -> "Array":
        """
        Approximate logarithmic map from mean to all points.

        log_μ(x) ≈ (x - μ) * (geodesic_dist / euclidean_dist)

        This scales the Euclidean tangent vector by the ratio of
        geodesic to Euclidean distance, accounting for curvature.

        Scale Clamping:
            See _frechet_mean_step for explanation of [0.5, 2.0] bounds.
        """
        backend = self._backend

        # Find nearest point to mean in the dataset using geodesic distance
        mean_idx = self._find_nearest_point(points, mean, geo_dist=geo_dist)

        # Euclidean vectors from mean
        diff = points - backend.reshape(mean, (1, -1))
        euc_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        # Geodesic distances from mean
        geo_from_mean = geo_dist[mean_idx, :]

        # Scale factor
        euc_safe = backend.maximum(euc_dist, backend.full(euc_dist.shape, 1e-10))
        scale = geo_from_mean / euc_safe

        # Count clamping events before applying clamps
        backend.eval(scale)
        scale_np = backend.to_numpy(scale)
        n = len(scale_np)
        high_clamp_count = sum(1 for s in scale_np.flatten() if float(s) > 2.0)
        low_clamp_count = sum(1 for s in scale_np.flatten() if float(s) < 0.5)

        if high_clamp_count > 0 or low_clamp_count > 0:
            logger.debug(
                f"Scale clamping in log map: {high_clamp_count}/{n} high (>2.0), "
                f"{low_clamp_count}/{n} low (<0.5)."
            )

        # Clamp to reasonable range (see _frechet_mean_step for bounds explanation)
        scale = backend.minimum(scale, backend.full(scale.shape, 2.0))
        scale = backend.maximum(scale, backend.full(scale.shape, 0.5))

        # Scaled tangent vectors
        scale_col = backend.reshape(scale, (-1, 1))
        log_vectors = diff * scale_col

        return log_vectors


# Convenience functions


def frechet_mean(
    points: "Array",
    weights: "Array | None" = None,
    backend: "Backend | None" = None,
) -> "Array":
    """
    Compute the Fréchet mean of a point set.

    Convenience function that returns just the mean point.

    Args:
        points: Point cloud [n, d]
        weights: Optional weights [n]
        backend: Backend to use

    Returns:
        Fréchet mean point [d]
    """
    if backend is None:
        backend = get_default_backend()

    rg = RiemannianGeometry(backend)
    result = rg.frechet_mean(points, weights)
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
