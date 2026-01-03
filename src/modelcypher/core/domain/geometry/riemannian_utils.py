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

Research Connections:
    Geodesic distance is the correct metric for neural representations because
    curvature is inherent in high-dimensional spaces. This aligns with the
    Platonic Representation Hypothesis (Huh et al., ICML 2024): if models
    converge to shared representations, they must be compared using the correct
    geometry—not flat Euclidean approximations.

    The k-NN graph IS the discrete manifold. Geodesic = shortest path on graph
    (exact, not approximate). Euclidean distance systematically errs:
    - Positive curvature: Euclidean underestimates true distance
    - Negative curvature: Euclidean overestimates true distance

    See also: docs/RESEARCH-CONNECTIONS.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    is_inf,
    machine_epsilon,
    pi_value,
    regularization_epsilon,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Session-scoped cache for geodesic distances and Fréchet means
_cache = ComputationCache.shared()


# =============================================================================
# Vectorized validation helpers (use backend ops instead of Python loops)
# =============================================================================


def count_nan(arr: "Array", backend: "Backend") -> int:
    """Count NaN values in array using vectorized backend operation.

    Replaces O(n*d) Python loops like:
        sum(1 for row in arr_np for v in row if math.isnan(float(v)))

    With O(1) backend operation:
        int(backend.sum(backend.isnan(arr)))

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        Count of NaN values in the array.
    """
    nan_mask = backend.isnan(arr)
    count = backend.sum(nan_mask)
    backend.eval(count)
    return int(float(backend.to_scalar(count)))


def count_inf(arr: "Array", backend: "Backend") -> int:
    """Count infinite values in array using vectorized backend operation.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        Count of +/- infinity values in the array.
    """
    inf_mask = backend.isinf(arr)
    count = backend.sum(inf_mask)
    backend.eval(count)
    return int(float(backend.to_scalar(count)))


def count_finite(arr: "Array", backend: "Backend") -> int:
    """Count finite values in array using vectorized backend operation.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        Count of finite (not NaN, not Inf) values in the array.
    """
    finite_mask = backend.isfinite(arr)
    count = backend.sum(finite_mask)
    backend.eval(count)
    return int(float(backend.to_scalar(count)))


def count_nonfinite(arr: "Array", backend: "Backend") -> int:
    """Count non-finite values (NaN or Inf) in array.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        Count of NaN or infinite values.
    """
    finite_mask = backend.isfinite(arr)
    # Count non-finite = total - finite
    nonfinite_mask = backend.where(
        finite_mask,
        backend.zeros_like(finite_mask),
        backend.ones_like(finite_mask),
    )
    count = backend.sum(nonfinite_mask)
    backend.eval(count)
    return int(float(backend.to_scalar(count)))


def has_nan(arr: "Array", backend: "Backend") -> bool:
    """Check if array contains any NaN values.

    More efficient than count_nan() > 0 for early exit.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if any NaN values present.
    """
    nan_mask = backend.isnan(arr)
    any_nan = backend.max(nan_mask)  # max of bool array = any True
    backend.eval(any_nan)
    return bool(backend.to_scalar(any_nan))


def has_inf(arr: "Array", backend: "Backend") -> bool:
    """Check if array contains any infinite values.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if any +/- infinity values present.
    """
    inf_mask = backend.isinf(arr)
    any_inf = backend.max(inf_mask)
    backend.eval(any_inf)
    return bool(backend.to_scalar(any_inf))


def all_finite(arr: "Array", backend: "Backend") -> bool:
    """Check if all values in array are finite.

    Args:
        arr: Backend array to check.
        backend: Backend instance.

    Returns:
        True if all values are finite (no NaN, no Inf).
    """
    finite_mask = backend.isfinite(arr)
    all_ok = backend.min(finite_mask)  # min of bool array = all True
    backend.eval(all_ok)
    return bool(backend.to_scalar(all_ok))


# =============================================================================
# Scalar utilities
# =============================================================================


def safe_arithmetic_mean(values: "list[float] | tuple[float, ...]") -> float:
    """Compute arithmetic mean, returning 0.0 for empty sequences.

    This is a simple utility for scalar values. For embeddings on
    curved manifolds, use frechet_mean() instead.

    Args:
        values: Sequence of float values.

    Returns:
        Arithmetic mean, or 0.0 if empty.
    """
    if not values:
        return 0.0
    vals = list(values) if not isinstance(values, list) else values
    return sum(vals) / len(vals)


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
        tolerance: float | None = None,
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

        # Validate input points for NaN (vectorized - O(1) backend op)
        if logger.isEnabledFor(logging.DEBUG):
            nan_count = count_nan(points, backend)
            if nan_count > 0:
                raise ValueError(
                    f"Input points contain {nan_count} NaN values. "
                    f"This indicates corrupted activations from the model."
                )

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

        if n == 2:
            # With two points, the k-NN graph is a single edge and the geodesic
            # is that edge. The Fréchet mean lies at the weighted midpoint.
            if weights is None:
                weights_arr = backend.array([0.5, 0.5])
            else:
                weights_arr = backend.array(weights)
                weight_sum = backend.sum(weights_arr)
                weights_arr = weights_arr / weight_sum

            mean = points[0] * weights_arr[0] + points[1] * weights_arr[1]
            diff0 = points[0] - mean
            diff1 = points[1] - mean
            variance = (
                weights_arr[0] * backend.sum(diff0 * diff0)
                + weights_arr[1] * backend.sum(diff1 * diff1)
            )
            backend.eval(mean, variance)
            return FrechetMeanResult(
                mean=mean,
                iterations=1,
                converged=True,
                final_variance=float(backend.to_scalar(variance)),
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

            # Derive tolerance from dtype if not specified - use sqrt(eps) as standard
            if tolerance is None:
                tol = float(machine_epsilon(backend, mu)) ** 0.5
            else:
                tol = tolerance

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
                    diff_val = float(backend.to_scalar(diff))

                    if diff_val < tol:
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
            # Vectorized count - O(1) vs O(d)
            non_finite = count_nonfinite(mu, backend)
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
        Compute geodesic distances using a k-NN graph and shortest paths.

        This implements the Isomap-style geodesic computation:
        1. Build a k-NN graph where edge weights are Euclidean distances
        2. Compute shortest paths (geodesics) using Floyd-Warshall algorithm

        When k_neighbors is None, the method finds the MINIMUM k that makes
        the graph connected. This is a geometric property of the point cloud -
        the connectivity threshold reveals the manifold's intrinsic structure.

        Uses session-scoped caching to avoid redundant computation when the
        same point set is used multiple times (e.g., in frechet_mean,
        riemannian_covariance, and curvature estimation).

        The key insight is that on a curved manifold, the geodesic distance
        follows the manifold surface, while Euclidean distance "cuts through"
        the manifold. For nearby points, geodesic ≈ Euclidean. For distant
        points, geodesic > Euclidean on positive curvature.

        Args:
            points: Point cloud [n, d]
            k_neighbors: Number of neighbors for graph. If None, automatically
                         finds the minimum k that ensures graph connectivity.
                         This is the geometric answer, not an arbitrary default.

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

        # If k_neighbors specified, use it directly
        if k_neighbors is not None:
            k_neighbors = max(1, min(k_neighbors, n - 1))
            return self._compute_geodesic_for_k(points, k_neighbors)

        # Find minimum k for connectivity - this IS the geometric answer
        # Binary search: start at k=1, double until connected, then binary search
        k_low = 1
        k_high = n - 1

        # First, check if k=1 works (rare but possible for dense clouds)
        result = self._compute_geodesic_for_k(points, k_low)
        if result.connected:
            return result

        # Double k until connected to find upper bound
        k_test = 2
        while k_test < k_high:
            result = self._compute_geodesic_for_k(points, k_test)
            if result.connected:
                k_high = k_test
                break
            k_low = k_test
            k_test = min(k_test * 2, k_high)
        else:
            # Need maximum k
            result = self._compute_geodesic_for_k(points, k_high)
            if result.connected:
                k_low = k_high // 2
                k_high = k_high
            else:
                # Fully connected graph still disconnected - degenerate case
                return result

        # Binary search for minimum k that achieves connectivity
        while k_low < k_high - 1:
            k_mid = (k_low + k_high) // 2
            result = self._compute_geodesic_for_k(points, k_mid)
            if result.connected:
                k_high = k_mid
            else:
                k_low = k_mid

        # Return result for minimum connected k
        return self._compute_geodesic_for_k(points, k_high)

    def _compute_geodesic_for_k(
        self,
        points: "Array",
        k_neighbors: int,
    ) -> GeodesicDistanceResult:
        """Compute geodesic distances for a specific k value."""
        backend = self._backend
        n = int(points.shape[0])
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

        # Diagnostic: check euclidean distance matrix for NaN (vectorized)
        if logger.isEnabledFor(logging.DEBUG):
            euc_nan_count = count_nan(euclidean_dist, backend)
            if euc_nan_count > 0:
                # Also check source points for issues
                points_nan = count_nan(points, backend)
                points_inf = count_inf(points, backend)
                logger.warning(
                    f"Euclidean distance matrix has {euc_nan_count} NaN values! "
                    f"Points shape: {points.shape}, points NaN: {points_nan}, points Inf: {points_inf}"
                )

        # Build k-NN adjacency and run Floyd-Warshall on backend (no scipy)
        # Use a reasonable sentinel value that's:
        # - Large enough to clearly indicate "no direct edge" in the k-NN graph
        # - Small enough to not cause overflow when used in downstream computations
        #   (like scale = geodesic / euclidean in Fréchet mean)
        # 1e20 is a good balance: much larger than any reasonable geodesic distance
        # but small enough that scale = 1e20 / 0.001 = 1e23 won't overflow
        inf_val = 1e20
        eye = backend.eye(n)
        dist_for_sort = euclidean_dist + eye * inf_val

        # Deterministic tie-breaker by column index at machine-epsilon scale.
        tie_eps = machine_epsilon(backend, euclidean_dist)
        col_indices = backend.arange(n)
        col_indices_row = backend.reshape(col_indices, (1, n))
        tie_breaker = backend.astype(col_indices_row, euclidean_dist.dtype) * tie_eps
        dist_for_sort = dist_for_sort + tie_breaker

        sorted_idx = backend.argsort(dist_for_sort, axis=1)
        knn_idx = sorted_idx[:, :k_neighbors]

        adj = backend.full((n, n), inf_val)
        adj = adj * (1.0 - eye)

        # For edge weights, preserve true zeros (identical points) but floor
        # very small non-zero distances to prevent numerical issues.
        # Identical points should have geodesic distance 0.
        edge_eps = float(division_epsilon(backend, euclidean_dist))
        is_effectively_zero = euclidean_dist < edge_eps * 0.1
        dist_floor = backend.where(
            is_effectively_zero,
            backend.zeros_like(euclidean_dist),
            backend.maximum(euclidean_dist, edge_eps),
        )

        for neighbor_rank in range(k_neighbors):
            neighbor_cols = knn_idx[:, neighbor_rank]
            mask = backend.reshape(neighbor_cols, (n, 1)) == col_indices_row
            adj = backend.where(mask, dist_floor, adj)

        adj = backend.minimum(adj, backend.transpose(adj))
        backend.eval(adj)

        # Diagnostic: check adjacency matrix construction (vectorized)
        if logger.isEnabledFor(logging.DEBUG):
            # Count edges (finite and below inf threshold)
            finite_mask = backend.isfinite(adj)
            below_inf = adj < inf_val * 0.9
            edge_mask = finite_mask * below_inf  # element-wise AND
            edge_count = int(float(backend.to_scalar(backend.sum(edge_mask))))
            # Count inf entries (at or above inf threshold)
            inf_mask = adj >= inf_val * 0.9
            inf_count_adj = int(float(backend.to_scalar(backend.sum(inf_mask))))
            # Count NaN entries
            nan_count_adj = count_nan(adj, backend)
            logger.debug(
                f"Adjacency matrix: n={n}, k={k_neighbors}, "
                f"edges={edge_count}, inf_entries={inf_count_adj}, nan_entries={nan_count_adj}"
            )

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

        # Diagnostic: check geodesic matrix after Floyd-Warshall (only when debugging)
        if logger.isEnabledFor(logging.DEBUG):
            # Vectorized counts - O(1) vs O(n²)
            finite_mask = backend.isfinite(geo_dist_arr)
            below_inf = geo_dist_arr < inf_val * 0.9
            fw_finite = int(float(backend.to_scalar(backend.sum(finite_mask * below_inf))))
            fw_inf = int(float(backend.to_scalar(backend.sum(geo_dist_arr >= inf_val * 0.9))))
            fw_nan = count_nan(geo_dist_arr, backend)
            logger.debug(
                f"After Floyd-Warshall: finite={fw_finite}, inf={fw_inf}, nan={fw_nan}"
            )

        # Derive thresholds from dtype for numerical comparisons
        tiny = machine_epsilon(backend, geo_dist_arr)

        # Create indicator for "x >= threshold" using sign arithmetic:
        # sign(x - threshold + tiny) = 1 for x >= threshold, -1 for x < threshold
        # maximum(..., 0) converts -1 to 0, giving 1/0 indicator
        threshold = inf_val * 0.9
        diff_from_threshold = geo_dist_arr - threshold
        near_inf_indicator = backend.maximum(
            backend.sign(diff_from_threshold + tiny),
            backend.zeros_like(geo_dist_arr),
        )

        # Count disconnected pairs (inf values represent genuinely infinite
        # geodesic distance between disconnected manifold components)
        inf_count = int(backend.to_scalar(backend.sum(near_inf_indicator)))
        connected = inf_count == 0

        # Zero out only the diagonal (self-distances should be exactly 0).
        # The diagonal may have accumulated numerical noise from Floyd-Warshall
        # where d(i,k) + d(k,i) is compared against d(i,i)=0.
        # Only apply near_zero cleanup to diagonal entries, not all entries.
        # Zeroing all small distances would corrupt legitimate close point pairs.
        eye = backend.eye(n)
        diag_mask = eye > 0.5  # Boolean mask for diagonal

        # Replace near-inf values with actual infinity, diagonal with 0
        inf_array = backend.full(geo_dist_arr.shape, float("inf"))
        zero_array = backend.zeros_like(geo_dist_arr)

        # where(indicator, replacement, original) - indicator acts as boolean
        geo_dist = backend.where(near_inf_indicator, inf_array, geo_dist_arr)
        geo_dist = backend.where(diag_mask, zero_array, geo_dist)
        backend.eval(geo_dist)

        if not connected:
            n_disconnected = inf_count // 2  # symmetric, so divide by 2
            logger.debug(
                f"k-NN graph has {n_disconnected} disconnected pairs "
                f"(k={k_neighbors}, n={n}). Consider increasing k_neighbors."
            )

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
        k_neighbors: int | None = None,
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
            k_neighbors: Number of neighbors (if None, derived from geometry)

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

        # Get geodesic distances (k=None triggers connectivity-based selection)
        geo_result = self.geodesic_distances(points, k_neighbors=k_neighbors)
        # Use the actual k from the result (may differ from input if None was passed)
        k_neighbors = geo_result.k_neighbors
        euclidean_dist = self._euclidean_distance_matrix(points)

        # Compute precision-aware epsilon before numpy conversion
        eps = division_epsilon(backend, euclidean_dist)

        # Look at the k nearest neighbors of the center point
        center_geo = geo_result.distances[center_idx]
        center_euc = euclidean_dist[center_idx]

        sorted_idx = backend.argsort(center_euc)
        neighbors = sorted_idx[1 : k_neighbors + 1]
        backend.eval(neighbors)

        center_geo_k = backend.take(center_geo, neighbors)
        center_euc_k = backend.take(center_euc, neighbors)
        backend.eval(center_geo_k, center_euc_k)

        # Compute geodesic defect: (geodesic - euclidean) / euclidean
        valid_mask = center_euc_k > eps
        valid_count = backend.sum(backend.astype(valid_mask, "float32"))
        backend.eval(valid_count)
        valid_count_val = int(backend.to_scalar(valid_count))

        if valid_count_val == 0:
            return CurvatureEstimate(
                sectional_curvature=0.0,
                is_positive=False,
                is_negative=False,
                confidence=0.0,
            )

        defects = (center_geo_k - center_euc_k) / center_euc_k
        defects = backend.where(valid_mask, defects, backend.zeros_like(defects))
        sum_defects = backend.sum(defects)
        mean_defect_arr = sum_defects / max(valid_count_val, 1)
        backend.eval(mean_defect_arr)
        mean_defect = float(backend.to_scalar(mean_defect_arr))

        if valid_count_val > 1:
            diff = defects - mean_defect_arr
            diff = backend.where(valid_mask, diff, backend.zeros_like(diff))
            variance = backend.sum(diff * diff) / valid_count_val
            backend.eval(variance)
            std_defect = float(backend.to_scalar(backend.sqrt(variance)))
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
            is_positive=sectional_curvature > 0,  # Any positive curvature
            is_negative=sectional_curvature < 0,  # Any negative curvature
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
        n = int(geo_dist.shape[0])

        total_dist = float(backend.to_scalar(geo_dist[start_idx, end_idx]))

        if is_inf(total_dist, backend):
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
        col_end = backend.take(geo_dist, backend.array([end_idx]), axis=1)
        col_end = backend.squeeze(col_end, axis=1)
        backend.eval(col_end)
        col_end_list = backend.tolist(col_end)

        while current != end_idx:
            # Find next point: must satisfy triangle equality
            # d(current, next) + d(next, end) ≈ d(current, end)
            row = backend.take(geo_dist, backend.array([current]), axis=0)
            row = backend.squeeze(row, axis=0)
            backend.eval(row)
            row_list = backend.tolist(row)
            dist_to_end = float(row_list[end_idx])

            best_next = end_idx
            best_dist = dist_to_end

            for candidate in range(n):
                if candidate == current or candidate in path:
                    continue

                d_to_candidate = float(row_list[candidate])
                d_candidate_to_end = float(col_end_list[candidate])

                if is_inf(d_to_candidate, backend) or is_inf(d_candidate_to_end, backend):
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

        path_idx_arr = backend.array(path_indices)
        path_points = backend.take(points, path_idx_arr, axis=0)
        diffs = path_points[1:] - path_points[:-1]
        segment_lengths = backend.sqrt(backend.sum(diffs * diffs, axis=1))
        backend.eval(segment_lengths)
        segment_list = backend.tolist(segment_lengths)

        arc_lengths = [0.0]
        cumulative = 0.0
        for seg in segment_list:
            cumulative += float(seg)
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
        index_grid = backend.arange(0, n)
        mask = backend.astype(index_grid == seed_idx, "float32")
        backend.eval(mask)

        # Min distance from each point to the selected set
        # Initially, just distance to seed
        min_distances = geo_dist[seed_idx]
        backend.eval(min_distances)

        # Iteratively select farthest point
        for _ in range(n_samples - 1):
            # Find point with maximum min-distance to selected set
            neg_inf = backend.full((n,), float("-inf"))
            masked = backend.where(mask > 0, neg_inf, min_distances)
            masked = backend.where(backend.isfinite(masked), masked, neg_inf)
            backend.eval(masked)

            farthest_idx = int(backend.to_scalar(backend.argmax(masked)))
            selected.append(farthest_idx)

            # Update mask for selected points
            one_hot = backend.astype(index_grid == farthest_idx, "float32")
            mask = backend.minimum(mask + one_hot, backend.ones_like(mask))

            # Update min distances: element-wise minimum with new point's distances
            new_dists = geo_dist[farthest_idx]
            min_distances = backend.minimum(min_distances, new_dists)
            backend.eval(min_distances, mask)

        # Compute coverage radius (max of final min-distances, excluding selected)
        neg_inf = backend.full((n,), float("-inf"))
        masked = backend.where(mask > 0, neg_inf, min_distances)
        masked = backend.where(backend.isfinite(masked), masked, neg_inf)
        backend.eval(masked)
        max_val = float(backend.to_scalar(backend.max(masked)))
        coverage_radius = 0.0 if is_inf(max_val, backend) else max(0.0, max_val)

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
        # k can be 0 when n=1 (single point has no neighbors)
        k = min(k, n - 1)

        # Early exit for isolated point (no neighbors possible)
        if k == 0:
            sparse_dir = backend.zeros((d,))
            if d > 0:
                sparse_dir = _set_matrix_element(
                    backend, backend.reshape(sparse_dir, (1, d)), 0, 0, 1.0
                )
                sparse_dir = backend.reshape(sparse_dir, (d,))
            return DirectionalCoverage(
                sparse_direction=sparse_dir,
                max_gap_angle=pi_value(backend),  # Full hemisphere is empty
                coverage_uniformity=0.0,
                neighbor_directions=backend.zeros((0, d)),
                point_idx=point_idx,
            )

        center = points[point_idx]

        # Get geodesic distances for neighbor selection
        geo_result = self.geodesic_distances(points, k_neighbors=k)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        # Find k nearest neighbors by geodesic distance
        row = backend.take(geo_dist, backend.array([point_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        inf = float("inf")
        row_masked = backend.where(
            backend.arange(0, n) == point_idx,
            backend.full((n,), inf),
            row,
        )
        sorted_indices = backend.argsort(row_masked)
        neighbors = sorted_indices[:k]

        if int(neighbors.shape[0]) == 0:
            # Isolated point - any direction is sparse
            sparse_dir = backend.zeros((d,))
            if d > 0:
                sparse_dir = _set_matrix_element(
                    backend, backend.reshape(sparse_dir, (1, d)), 0, 0, 1.0
                )
                sparse_dir = backend.reshape(sparse_dir, (d,))
            return DirectionalCoverage(
                sparse_direction=sparse_dir,
                max_gap_angle=pi_value(backend),  # Full hemisphere is empty
                coverage_uniformity=0.0,
                neighbor_directions=backend.zeros((0, d)),
                point_idx=point_idx,
            )

        # Compute tangent vectors to neighbors
        neighbor_pts = backend.take(points, neighbors, axis=0)
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
        min_max_sim = float(backend.to_scalar(backend.min(max_sims)))
        sparse_idx = int(backend.to_scalar(backend.argmin(max_sims)))

        sparse_direction = candidates[sparse_idx]

        # Convert max similarity to angle: theta = arccos(similarity)
        # The "gap" is the angle to the nearest neighbor direction
        # Fix: arccos domain is [-1, 1]
        clamped_sim = max(-1.0, min(1.0, min_max_sim))
        max_gap_angle = acos_scalar(clamped_sim, backend)

        # Coverage uniformity: ideal is uniform distribution on sphere
        # Measure as 1 - (variance of similarities)
        # If all neighbors are in one direction, variance is high -> low uniformity
        sim_mean = backend.mean(max_sims)
        sim_var = backend.mean((max_sims - sim_mean) ** 2)
        backend.eval(sim_var)
        # Normalize variance to [0, 1] range (max variance for similarities is ~1)
        coverage_uniformity = max(0.0, 1.0 - float(backend.to_scalar(sim_var)))

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
        """Compute pairwise geodesic-compatible distances.

        Uses the direct difference formula ||a - b||² = sum((a_i - b_i)²)
        which is rotation-invariant by construction and avoids catastrophic
        cancellation in the alternative formula ||a||² + ||b||² - 2*a·b.

        For high-dimensional spaces, these local distances are used as edge
        weights in the k-NN graph. The true geodesic distances are computed
        via shortest paths on this graph.
        """
        backend = self._backend

        # Force float32 to avoid bfloat16 precision issues
        if hasattr(backend, 'astype'):
            points = backend.astype(points, "float32")

        n = int(points.shape[0])
        d = int(points.shape[1]) if len(points.shape) > 1 else 1

        # Direct difference formula: ||a - b||² = sum((a_i - b_i)²)
        # This is rotation-invariant: ||Qa - Qb||² = ||(Q(a-b))||² = ||a-b||²
        # Uses O(n² * d) memory but is the correct approach for manifolds.
        points_i = backend.reshape(points, (n, 1, d))  # [n, 1, d]
        points_j = backend.reshape(points, (1, n, d))  # [1, n, d]
        diffs = points_i - points_j  # [n, n, d] via broadcasting
        dist_sq = backend.sum(diffs * diffs, axis=2)  # [n, n]
        backend.eval(dist_sq)
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

        Adaptively increases k until all points are reachable from the query.
        This is necessary because even though the full graph is connected,
        a subset of k neighbors might not reach all points.
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

        # For query attachment, always use all points to ensure direct paths exist.
        # This is mathematically necessary because excluding any point forces a detour:
        #   d(q, i) via j = d(q, j) + d(j, i) >= d(q, i)  (triangle inequality)
        # Equality holds only when j is on the geodesic from q to i.
        # In flat space, excluding the farthest point causes overestimation.
        # Using all points ensures the minimum always includes the direct path.
        weights_col = backend.reshape(euc_dist, (n, 1))
        candidates = geo_result.distances + weights_col
        geo_from_query = backend.min(candidates, axis=0)

        # Check if all points are reachable (no inf or nan values)
        backend.eval(geo_from_query)
        # Vectorized count - O(1) vs O(n)
        nonfinite_count = count_nonfinite(geo_from_query, backend)

        if nonfinite_count == 0:
            # All points reachable with finite distances
            return geo_from_query

        # If we get here with non-finite values even after using all points,
        # there's a fundamental issue with the geodesic matrix or query position
        backend.eval(geo_from_query)
        # Vectorized counts - O(1) vs O(n) / O(n²)
        final_nonfinite = count_nonfinite(geo_from_query, backend)

        current_k = n
        if final_nonfinite > 0:
            # Check the underlying geodesic matrix for issues
            mat_nonfinite = count_nonfinite(geo_result.distances, backend)

            # Check euclidean distances for issues
            euc_nonfinite = count_nonfinite(euc_dist, backend)

            logger.warning(
                f"_geodesic_distances_from_query: {final_nonfinite}/{n} points unreachable "
                f"even with k={current_k} neighbors. "
                f"Geodesic matrix has {mat_nonfinite} non-finite values. "
                f"Euclidean distances have {euc_nonfinite} non-finite values."
            )

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
        min_idx = int(backend.to_scalar(backend.argmin(geo_from_query)))
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

        # Validate inputs - NaN in mu or geo_from_mu causes downstream NaN propagation
        # Use backend operations to avoid CPU conversion
        backend.eval(mu, geo_from_mu)
        mu_isfinite = backend.isfinite(mu)
        geo_isfinite = backend.isfinite(geo_from_mu)
        backend.eval(mu_isfinite, geo_isfinite)

        # Count non-finite using backend sum (1 - isfinite gives 0/1 mask)
        mu_nonfinite = int(backend.to_scalar(backend.sum(1 - backend.astype(mu_isfinite, "float32"))))
        geo_nonfinite = int(backend.to_scalar(backend.sum(1 - backend.astype(geo_isfinite, "float32"))))

        if mu_nonfinite > 0:
            raise ValueError(
                f"Input mu contains {mu_nonfinite} non-finite values. "
                f"This indicates numerical instability in a previous iteration."
            )
        if geo_nonfinite > 0:
            raise ValueError(
                f"Input geo_from_mu contains {geo_nonfinite} non-finite values. "
                f"This indicates disconnected manifold components."
            )

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
        # Use backend operations to avoid CPU conversion
        backend.eval(scale)
        scale_isfinite = backend.isfinite(scale)
        scale_isinf = backend.isinf(scale)
        scale_isnan = backend.isnan(scale)
        backend.eval(scale_isfinite, scale_isinf, scale_isnan)

        inf_count = int(backend.to_scalar(backend.sum(backend.astype(scale_isinf, "float32"))))
        nan_count = int(backend.to_scalar(backend.sum(backend.astype(scale_isnan, "float32"))))

        if inf_count > 0 or nan_count > 0:
            raise ValueError(
                f"Geodesic/Euclidean scale contains {inf_count} inf and {nan_count} nan values. "
                f"This indicates disconnected manifold components or coincident points. "
                f"Increase k_neighbors to improve graph connectivity, or check for "
                f"duplicate points in the input."
            )

        # Log curvature scale statistics for diagnostics (only when debugging)
        if logger.isEnabledFor(logging.DEBUG):
            n = int(scale.shape[0])
            scale_min = float(backend.to_scalar(backend.min(scale)))
            scale_max = float(backend.to_scalar(backend.max(scale)))
            scale_mean = float(backend.to_scalar(backend.mean(scale)))
            logger.debug(
                f"Curvature scaling in Fréchet mean: n={n}, "
                f"scale range=[{scale_min:.3f}, {scale_max:.3f}], mean={scale_mean:.3f}"
            )

        # Weighted sum of scaled tangent vectors (log maps)
        scale_col = backend.reshape(scale, (-1, 1))
        weights_col = backend.reshape(weights, (-1, 1))
        log_vectors = diff * scale_col

        # Gradient is the weighted mean of log vectors
        gradient = backend.sum(log_vectors * weights_col, axis=0)

        # Adaptive step size based on gradient magnitude
        # This is geometrically valid - we're controlling step size, not the direction
        # Large gradients (from extreme curvature) need smaller steps to avoid overshooting
        backend.eval(gradient)

        # Check for non-finite values in gradient using backend operations
        # IEEE 754: 0 * inf = nan, inf - inf = nan, so we must handle before scaling
        grad_isfinite = backend.isfinite(gradient)
        backend.eval(grad_isfinite)
        has_nonfinite = int(backend.to_scalar(backend.sum(1 - backend.astype(grad_isfinite, "float32")))) > 0

        if has_nonfinite:
            # Gradient contains inf or nan - numerical overflow from extreme curvature
            # Replace non-finite values using backend where operations:
            # - nan -> 0 (no contribution from this component)
            # - inf -> max finite value (preserves direction)
            max_finite = 1e38  # Large but safely below float64 overflow
            grad_isnan = backend.isnan(gradient)
            grad_isinf = backend.isinf(gradient)
            grad_sign = backend.sign(gradient)
            backend.eval(grad_isnan, grad_isinf, grad_sign)

            # First handle nan -> 0
            gradient = backend.where(grad_isnan, backend.zeros_like(gradient), gradient)
            # Then handle inf -> max_finite * sign
            inf_replacement = grad_sign * backend.full(gradient.shape, max_finite)
            gradient = backend.where(grad_isinf, inf_replacement, gradient)
            backend.eval(gradient)

        grad_norm_arr = backend.sqrt(backend.sum(gradient * gradient))
        backend.eval(grad_norm_arr)
        grad_norm = float(backend.to_scalar(grad_norm_arr))

        # Use a step size that limits the maximum movement per iteration
        # This prevents numerical instability from extreme curvature while preserving
        # the gradient direction (which IS the geometric signal)
        base_eta = 0.5
        max_step = 1.0  # Maximum distance to move in one iteration

        if grad_norm > 0 and not is_inf(grad_norm, backend):
            eta = min(base_eta, max_step / grad_norm)
        elif is_inf(grad_norm, backend):
            # Gradient is still inf after clipping - use minimal step
            eta = machine_epsilon(backend, gradient)
        else:
            eta = base_eta

        # Update
        new_mu = mu + eta * gradient

        # Validate output - if update produced NaN, keep old mean
        # Use backend operations to avoid CPU conversion
        backend.eval(new_mu)
        new_mu_isfinite = backend.isfinite(new_mu)
        backend.eval(new_mu_isfinite)
        new_mu_nonfinite = int(backend.to_scalar(backend.sum(1 - backend.astype(new_mu_isfinite, "float32"))))

        if new_mu_nonfinite > 0:
            # Update would produce NaN - skip this iteration
            # This can happen with extreme curvature even after gradient normalization
            logger.warning(
                f"Fréchet mean update would produce {new_mu_nonfinite} non-finite values. "
                f"Skipping update to preserve numerical stability."
            )
            return mu

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
        return float(backend.to_scalar(variance))

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

        # Check for numerical issues - vectorized O(1) vs O(n)
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
