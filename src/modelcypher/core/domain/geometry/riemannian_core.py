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
Riemannian geometry core operations for high-dimensional representation spaces.

Neural network activations define points on a manifold. This module computes
exact geometric quantities on that manifold:

1. **Fréchet Mean (Karcher Mean)**: The Riemannian center of mass.
   Minimizes sum of squared geodesic distances: μ = argmin_p Σ d²(p, x_i)

2. **Geodesic Distance**: Shortest path along the manifold surface.
   Computed via k-NN graph - the discrete representation of the manifold.

3. **Riemannian Covariance**: Covariance computed in tangent space,
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
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    infinity_threshold,
    is_inf,
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_interpolation import (
    RiemannianInterpolationMixin,
)
from modelcypher.core.domain.geometry.riemannian_sampling import (
    RiemannianSamplingMixin,
)
from modelcypher.core.domain.geometry.riemannian_types import (
    CurvatureEstimate,
    FrechetMeanResult,
    GeodesicDistanceResult,
)
from modelcypher.core.domain.geometry.riemannian_validation import (
    count_inf,
    count_nan,
    count_nonfinite,
    validate_array_numerics,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

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


class RiemannianGeometry(RiemannianSamplingMixin, RiemannianInterpolationMixin):
    """
    Riemannian geometry operations for representation spaces.

    This class provides curvature-aware alternatives to chord-based operations:
    - Fréchet mean instead of arithmetic mean
    - Geodesic distance instead of chord distance
    - Riemannian covariance instead of ambient covariance

    Inherits from:
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

    def frechet_mean(
        self,
        points: "Array",
        weights: "Array | None" = None,
        max_iterations: int = 100,
        tolerance: float | None = None,
        k_neighbors: int | None = None,
        max_k_neighbors: int | None = None,
        geo_result: GeodesicDistanceResult | None = None,
    ) -> FrechetMeanResult:
        """
        Compute the Fréchet mean (Riemannian center of mass) of a point set.

        The Fréchet mean minimizes the sum of squared geodesic distances:
            μ = argmin_p Σᵢ wᵢ d²(p, xᵢ)

        Uses session-scoped caching to avoid redundant computation when the
        same point set is used multiple times.

        Algorithm:
            1. Initialize at the geodesic medoid (distance-minimizing point)
            2. Compute geodesic distances from current estimate to all points
            3. Update estimate using Riemannian gradient descent
            4. Repeat until convergence

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

        if geo_result is not None:
            geo_k = max(1, min(int(geo_result.k_neighbors), n - 1))
            if k_start is None or k_start != geo_k:
                k_start = geo_k
            if k_max is None:
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
        if attempt_k is not None and attempt_k >= n - 1:
            # Fully connected graph => geodesic distances reduce to chord distances.
            # The Fréchet mean is the exact weighted mean in this case.
            cache_key = _cache.make_frechet_key(points, backend, weights_key, attempt_k)
            cached = _cache.get_frechet(cache_key)
            if cached is not None:
                return cached

            weights_col = backend.reshape(weights_arr, (n, 1))
            mean = backend.sum(points * weights_col, axis=0)
            diffs = points - mean
            sq_norms = backend.sum(diffs * diffs, axis=1)
            variance = backend.sum(sq_norms * weights_arr)
            backend.eval(mean, variance)

            result = FrechetMeanResult(
                mean=mean,
                iterations=1,
                converged=True,
                final_variance=float(backend.to_scalar(variance)),
            )
            _cache.set_frechet(cache_key, result, 0.0)
            return result

        while True:
            cache_key = _cache.make_frechet_key(points, backend, weights_key, attempt_k)
            cached = _cache.get_frechet(cache_key)
            if cached is not None:
                return cached

            start = time.perf_counter()

            # Compute geodesic distance matrix once (expensive but reusable, now cached)
            try:
                if geo_result is None or (
                    attempt_k is not None and geo_result.k_neighbors != attempt_k
                ):
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

            # Initialize at geodesic medoid (distance-minimizing point).
            geo_dist = geo_result.distances
            backend.eval(geo_dist)
            weights_row = backend.reshape(weights_arr, (1, n))
            weighted = geo_dist * weights_row
            sum_per_point = backend.sum(weighted, axis=1)

            # If multiple medoids tie, initialize at their mean to preserve symmetry.
            min_sum_arr = backend.min(sum_per_point)
            scale = backend.maximum(backend.abs(min_sum_arr), backend.array(1.0))
            tie_eps = machine_epsilon(backend, sum_per_point) * scale
            tie_mask = sum_per_point <= (min_sum_arr + tie_eps)
            tie_weights = backend.astype(tie_mask, points.dtype)
            tie_count_arr = backend.sum(tie_weights)
            backend.eval(sum_per_point, min_sum_arr, tie_count_arr)
            tie_count = float(backend.to_scalar(tie_count_arr))
            if tie_count > 1.0:
                weights_col = backend.reshape(tie_weights, (n, 1))
                mu = backend.sum(points * weights_col, axis=0) / tie_count_arr
            else:
                medoid_idx_arr = backend.argmin(sum_per_point)
                backend.eval(medoid_idx_arr)
                medoid_idx = int(backend.to_scalar(medoid_idx_arr))
                mu = points[medoid_idx]

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

                    # Convergence in tangent space: step size of the update.
                    step_vec = new_mu - mu
                    step_norm_arr = backend.sqrt(backend.sum(step_vec * step_vec))
                    backend.eval(step_norm_arr)
                    step_val = float(backend.to_scalar(step_norm_arr))

                    if step_val < tol:
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

    def frechet_mean_batch(
        self,
        points_batch: "Array",
        weights_batch: "Array | None" = None,
        max_iterations: int = 100,
        tolerance: float | None = None,
        k_neighbors: int | None = None,
        max_k_neighbors: int | None = None,
    ) -> "Array":
        """Compute Fréchet means for a batch of point sets.

        Args:
            points_batch: [B, n, d] batch of point clouds
            weights_batch: Optional [B, n] weights per batch
        Returns:
            [B, d] array of Fréchet means
        """
        backend = self._backend
        points_batch = backend.array(points_batch)
        backend.eval(points_batch)

        if len(points_batch.shape) != 3:
            result = self.frechet_mean(
                points_batch,
                weights=weights_batch,
                max_iterations=max_iterations,
                tolerance=tolerance,
                k_neighbors=k_neighbors,
                max_k_neighbors=max_k_neighbors,
            )
            return backend.reshape(result.mean, (1, -1))

        batch = int(points_batch.shape[0])
        n = int(points_batch.shape[1])
        if k_neighbors is not None and n > 0 and k_neighbors >= n - 1:
            # Fully connected graph => geodesic distances reduce to chord distances.
            # The Fréchet mean is the exact weighted mean in this case.
            if weights_batch is None:
                mean = backend.mean(points_batch, axis=1)
                backend.eval(mean)
                return mean

            weights_batch = backend.array(weights_batch)
            weight_sum = backend.sum(weights_batch, axis=1, keepdims=True)
            weights_norm = weights_batch / weight_sum
            weights_norm = backend.reshape(weights_norm, (batch, n, 1))
            mean = backend.sum(points_batch * weights_norm, axis=1)
            backend.eval(mean)
            return mean

        def _mean_only(points: "Array", weights: "Array | None" = None) -> "Array":
            result = self.frechet_mean(
                points,
                weights=weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                k_neighbors=k_neighbors,
                max_k_neighbors=max_k_neighbors,
            )
            return result.mean

        vmap = getattr(backend, "vmap", None)
        if vmap is not None:
            try:
                if weights_batch is None:
                    mapped = backend.vmap(_mean_only)
                    means = mapped(points_batch)
                else:
                    mapped = backend.vmap(_mean_only, in_axes=(0, 0))
                    means = mapped(points_batch, weights_batch)
                backend.eval(means)
                return means
            except Exception as exc:
                logger.debug("vmap failed for frechet_mean_batch, falling back to sequential: %s", exc)

        means = []
        for i in range(batch):
            weights = weights_batch[i] if weights_batch is not None else None
            result = self.frechet_mean(
                points_batch[i],
                weights=weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                k_neighbors=k_neighbors,
                max_k_neighbors=max_k_neighbors,
            )
            means.append(result.mean)
        stacked = backend.stack(means, axis=0)
        backend.eval(stacked)
        return stacked

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
        1. Build a k-NN graph where edge weights are chord distances (local edges)
        2. Compute shortest paths (geodesics) using Floyd-Warshall or sparse Dijkstra

        When k_neighbors is None, the method finds the MINIMUM k that makes
        the graph connected. This is a geometric property of the point cloud -
        the connectivity threshold reveals the manifold's intrinsic structure.

        Uses session-scoped caching to avoid redundant computation when the
        same point set is used multiple times.

        Args:
            points: Point cloud [n, d]
            k_neighbors: Number of neighbors for graph. If None, automatically
                         finds the minimum k that ensures graph connectivity.

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
        kmin_key = _cache.make_kmin_key(points, backend)
        cached_k = _cache.get_kmin(kmin_key)
        if cached_k is not None:
            cached_geo = _cache.get_geodesic(
                _cache.make_geodesic_key(points, backend, int(cached_k))
            )
            if cached_geo is not None:
                return cached_geo

        k_start_time = time.perf_counter()
        k_min, knn_idx, chord_dist = self._minimum_connected_k(points)
        _cache.set_kmin(kmin_key, k_min, (time.perf_counter() - k_start_time) * 1000)
        return self._compute_geodesic_for_k(
            points, k_min, chord_dist=chord_dist, knn_idx=knn_idx
        )

    def _minimum_connected_k(
        self,
        points: "Array",
        chord_dist: "Array | None" = None,
    ) -> tuple[int, "Array", "Array"]:
        """Find the minimum k that makes the k-NN graph connected."""
        backend = self._backend
        n = int(points.shape[0])
        if n <= 1:
            return 0, backend.zeros((n, 0)), backend.zeros((n, n))

        # Binary search: start at k=1, double until connected, then binary search
        if chord_dist is None:
            chord_dist = self._chord_distance_matrix(points)
            backend.eval(chord_dist)

        k_low = 1
        k_high = n - 1
        knn_high = None

        # First, check if k=1 works (rare but possible for dense clouds)
        knn_low, _ = self._knn_indices_from_chord(chord_dist, k_low)
        if self._is_knn_connected(knn_low):
            return k_low, knn_low, chord_dist

        # Double k until connected to find upper bound
        k_test = 2
        while k_test < k_high:
            knn_test, _ = self._knn_indices_from_chord(chord_dist, k_test)
            if self._is_knn_connected(knn_test):
                k_high = k_test
                knn_high = knn_test
                break
            k_low = k_test
            k_test = min(k_test * 2, k_high)
        else:
            # Need maximum k
            knn_high, _ = self._knn_indices_from_chord(chord_dist, k_high)
            if self._is_knn_connected(knn_high):
                k_low = k_high // 2
            else:
                # Fully connected k-NN still disconnected - degenerate case
                return k_high, knn_high, chord_dist

        # Binary search for minimum k that achieves connectivity
        while k_low < k_high - 1:
            k_mid = (k_low + k_high) // 2
            knn_mid, _ = self._knn_indices_from_chord(chord_dist, k_mid)
            if self._is_knn_connected(knn_mid):
                k_high = k_mid
                knn_high = knn_mid
            else:
                k_low = k_mid

        if knn_high is None:
            knn_high, _ = self._knn_indices_from_chord(chord_dist, k_high)

        return k_high, knn_high, chord_dist

    def _knn_indices_from_chord(
        self,
        chord_dist: "Array",
        k_neighbors: int,
    ) -> tuple["Array", float]:
        """Compute k-NN indices from chord distances (no Floyd-Warshall)."""
        backend = self._backend
        n = int(chord_dist.shape[0])
        k_neighbors = max(1, min(k_neighbors, n - 1))

        max_dist_arr = backend.max(chord_dist)
        backend.eval(max_dist_arr)
        max_dist = float(backend.to_scalar(max_dist_arr))
        eps = machine_epsilon(backend, chord_dist)
        base = max(max_dist, eps)
        inf_val = min(base / eps, backend.finfo(chord_dist.dtype).max)
        eye = backend.eye(n)
        dist_no_self = chord_dist + eye * inf_val
        dist_for_sort = dist_no_self

        # Deterministic tie-breaker by column index at machine-epsilon scale.
        tie_eps = machine_epsilon(backend, chord_dist)
        col_indices = backend.arange(n)
        col_indices_row = backend.reshape(col_indices, (1, n))
        tie_breaker = backend.astype(col_indices_row, chord_dist.dtype) * tie_eps
        dist_for_sort = dist_for_sort + tie_breaker

        kth = max(0, min(k_neighbors - 1, n - 1))
        partitioned_idx = backend.argpartition(dist_for_sort, kth, axis=1)

        # Include all ties at the k-th distance to preserve symmetry.
        kth_idx = partitioned_idx[:, kth : kth + 1]
        kth_vals = backend.take_along_axis(dist_no_self, kth_idx, axis=1)
        tie_mask = dist_no_self <= (kth_vals + tie_eps)
        tie_count = backend.sum(backend.astype(tie_mask, "int32"), axis=1)
        max_ties_arr = backend.max(tie_count)
        backend.eval(max_ties_arr)
        max_ties = int(backend.to_scalar(max_ties_arr))
        if max_ties > k_neighbors:
            k_neighbors = max(1, min(max_ties, n - 1))
            kth = max(0, min(k_neighbors - 1, n - 1))
            partitioned_idx = backend.argpartition(dist_for_sort, kth, axis=1)

        knn_idx = partitioned_idx[:, :k_neighbors]
        backend.eval(knn_idx)
        return knn_idx, inf_val

    def _is_knn_connected(self, knn_idx: "Array") -> bool:
        """Check graph connectivity from k-NN indices without Floyd-Warshall."""
        neighbors = self._backend.tolist(knn_idx)
        n = len(neighbors)
        if n <= 1:
            return True

        undirected: list[set[int]] = [set() for _ in range(n)]
        for i, row in enumerate(neighbors):
            for j in row:
                j_int = int(j)
                if 0 <= j_int < n:
                    undirected[i].add(j_int)
                    undirected[j_int].add(i)

        seen = [False] * n
        stack = [0]
        seen[0] = True
        while stack:
            node = stack.pop()
            for nxt in undirected[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

        return all(seen)

    # NOTE: _should_use_sparse_shortest_paths() and _all_pairs_shortest_paths_sparse()
    # were removed as part of the "No NumPy" optimization. The sparse path used scipy
    # which forced GPU→CPU→GPU roundtrip. Backend Floyd-Warshall is now used exclusively
    # as it's JIT-compiled on GPU and faster for typical n (100-1000 points).

    def _compute_geodesic_for_k(
        self,
        points: "Array",
        k_neighbors: int,
        chord_dist: "Array | None" = None,
        knn_idx: "Array | None" = None,
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

        # Compute chord distance matrix (local edge lengths)
        if chord_dist is None:
            chord_dist = self._chord_distance_matrix(points)
            backend.eval(chord_dist)

        # Fully connected graph: geodesic equals chord distance.
        if k_neighbors >= n - 1:
            max_dist_arr = backend.max(chord_dist)
            backend.eval(max_dist_arr)
            max_dist = float(backend.to_scalar(max_dist_arr))
            eps = machine_epsilon(backend, chord_dist)
            base = max(max_dist, eps)
            inf_val = min(base / eps, backend.finfo(chord_dist.dtype).max)
            return GeodesicDistanceResult(
                distances=chord_dist,
                adjacency=chord_dist,
                inf_value=inf_val,
                k_neighbors=k_neighbors,
                connected=True,
            )

        # Diagnostic: check chord distance matrix for NaN (vectorized)
        if logger.isEnabledFor(logging.DEBUG):
            chord_nan_count = count_nan(chord_dist, backend)
            if chord_nan_count > 0:
                # Also check source points for issues
                points_nan = count_nan(points, backend)
                points_inf = count_inf(points, backend)
                logger.warning(
                    f"Chord distance matrix has {chord_nan_count} NaN values! "
                    f"Points shape: {points.shape}, points NaN: {points_nan}, points Inf: {points_inf}"
                )

        # Build k-NN adjacency and run shortest paths (Floyd-Warshall or sparse Dijkstra).
        # Use a sentinel derived from data scale and dtype precision to avoid overflow.
        max_dist_arr = backend.max(chord_dist)
        backend.eval(max_dist_arr)
        max_dist = float(backend.to_scalar(max_dist_arr))
        eps = machine_epsilon(backend, chord_dist)
        base = max(max_dist, eps)
        inf_val = min(base / eps, backend.finfo(chord_dist.dtype).max)
        eye = backend.eye(n)

        if knn_idx is None:
            knn_idx, _ = self._knn_indices_from_chord(chord_dist, k_neighbors)
        k_neighbors = int(knn_idx.shape[1])

        adj = backend.full((n, n), inf_val)
        adj = adj * (1.0 - eye)

        # For edge weights, preserve true zeros (identical points) but floor
        # very small non-zero distances to prevent numerical issues.
        edge_eps = float(division_epsilon(backend, chord_dist))
        zero_threshold = tiny_value(backend, chord_dist)
        is_effectively_zero = chord_dist < zero_threshold
        dist_floor = backend.where(
            is_effectively_zero,
            backend.zeros_like(chord_dist),
            backend.maximum(chord_dist, edge_eps),
        )

        neighbor_dists = backend.take_along_axis(dist_floor, knn_idx, axis=1)
        adj = backend.put_along_axis(adj, knn_idx, neighbor_dists, axis=1)

        adj = backend.minimum(adj, backend.transpose(adj))
        backend.eval(adj)

        # Compute infinity threshold from dtype precision (not arbitrary 0.9)
        inf_thresh = infinity_threshold(backend, adj)

        # Diagnostic: check adjacency matrix construction (single-pass validation)
        if logger.isEnabledFor(logging.DEBUG):
            # Single-pass validation for NaN/Inf/nonfinite
            nan_count_adj, inf_count_adj, _ = validate_array_numerics(adj, backend)
            # Count edges (finite and below inf threshold) - needed for connectivity info
            finite_mask = backend.isfinite(adj)
            below_inf = adj < inf_thresh
            edge_mask = finite_mask * below_inf
            edge_count_arr = backend.sum(edge_mask)
            backend.eval(edge_count_arr)
            edge_count = int(float(backend.to_scalar(edge_count_arr)))
            logger.debug(
                f"Adjacency matrix: n={n}, k={k_neighbors}, "
                f"edges={edge_count}, inf_entries={inf_count_adj}, nan_entries={nan_count_adj}"
            )

        # Floyd-Warshall: GPU-native all-pairs shortest paths
        # Uses backend.floyd_warshall() which is JIT-compiled on all backends (MLX/JAX/CUDA)
        # For typical n (100-1000), O(n³) on GPU is ~1-2ms - faster than scipy CPU roundtrip
        geo_dist_arr = backend.floyd_warshall(adj)
        backend.eval(geo_dist_arr)

        # Diagnostic: check geodesic matrix after Floyd-Warshall (single-pass validation)
        if logger.isEnabledFor(logging.DEBUG):
            # Single-pass validation for NaN/Inf/nonfinite
            fw_nan, fw_inf, _ = validate_array_numerics(geo_dist_arr, backend)
            # Count finite values below threshold
            finite_mask = backend.isfinite(geo_dist_arr)
            below_inf = geo_dist_arr < inf_thresh
            fw_finite_arr = backend.sum(finite_mask * below_inf)
            backend.eval(fw_finite_arr)
            fw_finite = int(float(backend.to_scalar(fw_finite_arr)))
            logger.debug(
                f"After Floyd-Warshall: finite={fw_finite}, inf={fw_inf}, nan={fw_nan}"
            )

        # Derive thresholds from dtype for numerical comparisons
        tiny = machine_epsilon(backend, geo_dist_arr)

        # Create indicator for "x >= threshold" using sign arithmetic
        diff_from_threshold = geo_dist_arr - inf_thresh
        near_inf_indicator = backend.maximum(
            backend.sign(diff_from_threshold + tiny),
            backend.zeros_like(geo_dist_arr),
        )

        # Count disconnected pairs (inf values represent genuinely infinite
        # geodesic distance between disconnected manifold components)
        inf_count_arr = backend.sum(near_inf_indicator)
        backend.eval(inf_count_arr)
        inf_count = int(backend.to_scalar(inf_count_arr))
        connected = inf_count == 0

        # Zero out only the diagonal (self-distances should be exactly 0).
        eye = backend.eye(n)
        diag_mask = eye > 0

        # Replace near-inf values with actual infinity, diagonal with 0
        inf_array = backend.full(geo_dist_arr.shape, float("inf"))
        zero_array = backend.zeros_like(geo_dist_arr)

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
        Estimate local sectional curvature at a point.

        Uses geodesic defect (geodesic vs chord length) for 2D/3D probes.
        For 4D+, falls back to geodesic-only curvature estimation.

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
        d = int(points.shape[1]) if len(points.shape) > 1 else 1

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

        # Look at the k nearest neighbors of the center point (geodesic order)
        center_geo = geo_result.distances[center_idx]
        inf_val = geo_result.inf_value
        center_geo = backend.where(
            backend.arange(0, n) == center_idx,
            backend.full((n,), inf_val),
            center_geo,
        )
        kth = max(0, min(k_neighbors - 1, n - 1))
        partitioned = backend.argpartition(center_geo, kth)
        neighbors = partitioned[:k_neighbors]
        backend.eval(neighbors)

        if d > 3:
            from modelcypher.core.domain.geometry.manifold_curvature import (
                SectionalCurvatureEstimator,
            )

            center = points[center_idx]
            neighbor_pts = backend.take(points, neighbors, axis=0)
            backend.eval(center, neighbor_pts)
            estimator = SectionalCurvatureEstimator()
            local = estimator.estimate_local_curvature(center, neighbor_pts)
            std_val = sqrt_scalar(local.variance_sectional, backend)
            confidence = 1.0 / (1.0 + std_val) if std_val > 0 else 1.0
            return CurvatureEstimate(
                sectional_curvature=local.mean_sectional,
                is_positive=local.mean_sectional > 0,
                is_negative=local.mean_sectional < 0,
                confidence=confidence,
            )

        # 2D/3D: compute geodesic defect against chord distances.
        center = points[center_idx]
        neighbor_pts = backend.take(points, neighbors, axis=0)
        diffs = neighbor_pts - center
        center_chord_k = backend.sqrt(backend.sum(diffs * diffs, axis=1))
        backend.eval(center_chord_k)

        # Compute precision-aware epsilon
        eps = division_epsilon(backend, center_chord_k)

        center_geo_k = backend.take(center_geo, neighbors)
        backend.eval(center_geo_k)

        # Compute geodesic defect: (geodesic - chord) / chord
        valid_mask = center_chord_k > eps
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

        defects = (center_geo_k - center_chord_k) / center_chord_k
        defects = backend.where(valid_mask, defects, backend.zeros_like(defects))
        sum_defects = backend.sum(defects)
        mean_defect_arr = sum_defects / max(valid_count_val, 1)
        backend.eval(mean_defect_arr)
        mean_defect = float(backend.to_scalar(mean_defect_arr))

        if valid_count_val > 1:
            diff = defects - mean_defect_arr
            diff = backend.where(valid_mask, diff, backend.zeros_like(diff))
            variance = backend.sum(diff * diff) / valid_count_val
            std_defect_arr = backend.sqrt(variance)
            backend.eval(std_defect_arr)
            std_defect = float(backend.to_scalar(std_defect_arr))
        else:
            std_defect = 0.0

        # Estimate curvature from defect
        # For a sphere of radius R, geodesic/chord ≈ 1 + K*r²/6 for small r
        # where K = 1/R² is the sectional curvature
        # So defect ≈ K*r²/6, giving K ≈ 6*defect/r²

        avg_radius_arr = backend.mean(center_chord_k)
        backend.eval(avg_radius_arr)
        avg_radius = float(backend.to_scalar(avg_radius_arr))
        if avg_radius > eps:
            # Rough curvature estimate
            sectional_curvature = 6.0 * mean_defect / (avg_radius * avg_radius)
        else:
            sectional_curvature = 0.0

        # Confidence based on consistency of defects
        confidence = 1.0 / (1.0 + std_defect) if std_defect > 0 else 1.0

        return CurvatureEstimate(
            sectional_curvature=sectional_curvature,
            is_positive=sectional_curvature > 0,
            is_negative=sectional_curvature < 0,
            confidence=confidence,
        )

    def riemannian_covariance(
        self,
        points: "Array",
        mean: "Array | None" = None,
        geo_result: GeodesicDistanceResult | None = None,
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
        if geo_result is None:
            geo_result = self.geodesic_distances(points)

        # Map points to tangent space at mean
        # Log_μ(x) = (x - μ) * (geodesic_dist / chord_dist)
        tangent_vectors = self._log_map_approximate(points, mean, geo_result)

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
    ) -> "Array":
        """Map points to the tangent space at a base point."""
        if geo_result is None:
            geo_result = self.geodesic_distances(points)
        return self._log_map_approximate(points, base, geo_result)

    # --- Private helper methods ---

    def _chord_distance_matrix(
        self, points: "Array", use_cache: bool = True
    ) -> "Array":
        """Compute pairwise geodesic-compatible distances.

        Uses the identity ||a - b||² = ||a||² + ||b||² - 2*a·b to avoid
        O(n² * d) intermediate memory while preserving rotation invariance.

        Args:
            points: Input points [n, d].
            use_cache: If True, check/populate the chord distance cache.
                      Unlike geodesic cache, chord cache is k-independent.

        Returns:
            Chord distance matrix [n, n].
        """
        backend = self._backend

        # Check cache first (chord distance is k-independent, so highly reusable)
        if use_cache:
            cache_key = _cache.make_chord_key(points, backend)
            cached = _cache.get_chord(cache_key)
            if cached is not None:
                return cached

        start_time = time.perf_counter()

        # Force float32 to avoid bfloat16 precision issues
        if hasattr(backend, 'astype'):
            points = backend.astype(points, "float32")

        n = int(points.shape[0])
        points_2d = (
            backend.reshape(points, (n, 1)) if len(points.shape) == 1 else points
        )

        norms_sq = backend.sum(points_2d * points_2d, axis=1)
        norms_col = backend.reshape(norms_sq, (n, 1))
        norms_row = backend.reshape(norms_sq, (1, n))
        cross = backend.matmul(points_2d, backend.transpose(points_2d))
        dist_sq = norms_col + norms_row - 2.0 * cross
        backend.eval(dist_sq)
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        result = backend.sqrt(dist_sq)

        # Cache the result
        if use_cache:
            compute_time_ms = (time.perf_counter() - start_time) * 1000
            _cache.set_chord(cache_key, result, compute_time_ms)

        return result

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
        discrete manifold.
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

        # Chord distances from query to all points (squared for stable top-k)
        diff = points - backend.reshape(query, (1, -1))
        dist_sq = backend.sum(diff * diff, axis=1)
        backend.eval(dist_sq)

        # Attach query to its k nearest neighbors in the manifold graph.
        # This preserves the discrete geodesic structure and avoids O(n^2).
        if k_neighbors >= n:
            chord_dist = backend.sqrt(backend.maximum(dist_sq, backend.zeros_like(dist_sq)))
            backend.eval(chord_dist)
            weights_col = backend.reshape(chord_dist, (n, 1))
            candidates = geo_result.distances + weights_col
            geo_from_query = backend.min(candidates, axis=0)
        else:
            tie_eps = machine_epsilon(backend, dist_sq)
            col_indices = backend.arange(n)
            dist_for_sort = dist_sq + backend.astype(col_indices, dist_sq.dtype) * tie_eps
            kth = max(0, min(k_neighbors - 1, n - 1))
            partitioned_idx = backend.argpartition(dist_for_sort, kth)
            neighbor_idx = partitioned_idx[:k_neighbors]

            # Include all ties at the k-th distance to preserve symmetry.
            kth_idx = partitioned_idx[kth]
            kth_dist_sq = backend.take(dist_sq, kth_idx, axis=0)
            tie_mask = dist_sq <= (kth_dist_sq + tie_eps)
            tie_count_arr = backend.sum(backend.astype(tie_mask, "int32"))
            backend.eval(tie_count_arr)
            tie_count = int(backend.to_scalar(tie_count_arr))
            if tie_count > k_neighbors:
                k_neighbors = tie_count
                kth = max(0, min(k_neighbors - 1, n - 1))
                partitioned_idx = backend.argpartition(dist_for_sort, kth)
                neighbor_idx = partitioned_idx[:k_neighbors]

            neighbor_dist_sq = backend.take(dist_sq, neighbor_idx, axis=0)
            neighbor_dist_sq = backend.maximum(neighbor_dist_sq, backend.zeros_like(neighbor_dist_sq))
            neighbor_dists = backend.sqrt(neighbor_dist_sq)

            neighbor_geo = backend.take(geo_result.distances, neighbor_idx, axis=0)
            weights_col = backend.reshape(neighbor_dists, (k_neighbors, 1))
            candidates = neighbor_geo + weights_col
            geo_from_query = backend.min(candidates, axis=0)

        # Check if all points are reachable (no inf or nan values)
        backend.eval(geo_from_query)
        nonfinite_count = count_nonfinite(geo_from_query, backend)

        if nonfinite_count == 0:
            return geo_from_query

        # Log warning if there are still non-finite values
        backend.eval(geo_from_query)
        final_nonfinite = count_nonfinite(geo_from_query, backend)

        current_k = n
        if final_nonfinite > 0:
            mat_nonfinite = count_nonfinite(geo_result.distances, backend)
            chord_nonfinite = count_nonfinite(chord_dist, backend)

            logger.warning(
                f"_geodesic_distances_from_query: {final_nonfinite}/{n} points unreachable "
                f"even with k={current_k} neighbors. "
                f"Geodesic matrix has {mat_nonfinite} non-finite values. "
                f"Chord distances have {chord_nonfinite} non-finite values."
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
        min_idx_arr = backend.argmin(geo_from_query)
        backend.eval(min_idx_arr)
        min_idx = int(backend.to_scalar(min_idx_arr))
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
        The geodesic/chord ratio captures the curvature correction.

        Raises:
            ValueError: If geodesic/chord scale contains inf or nan values.
        """
        backend = self._backend

        # Validate inputs
        mu_isfinite = backend.isfinite(mu)
        geo_isfinite = backend.isfinite(geo_from_mu)
        mu_nonfinite_arr = backend.sum(1 - backend.astype(mu_isfinite, "float32"))
        geo_nonfinite_arr = backend.sum(1 - backend.astype(geo_isfinite, "float32"))
        backend.eval(mu_nonfinite_arr, geo_nonfinite_arr)
        mu_nonfinite = int(backend.to_scalar(mu_nonfinite_arr))
        geo_nonfinite = int(backend.to_scalar(geo_nonfinite_arr))

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

        # Compute geodesic/chord ratio for curvature correction
        diff = points - backend.reshape(mu, (1, -1))
        chord_dist = backend.sqrt(backend.sum(diff * diff, axis=1))

        eps = division_epsilon(backend, chord_dist)
        chord_dist_safe = backend.maximum(chord_dist, backend.full(chord_dist.shape, eps))
        scale = geo_from_mu / chord_dist_safe

        # Check for numerical issues
        scale_isinf = backend.isinf(scale)
        scale_isnan = backend.isnan(scale)
        inf_count_arr = backend.sum(backend.astype(scale_isinf, "float32"))
        nan_count_arr = backend.sum(backend.astype(scale_isnan, "float32"))
        backend.eval(inf_count_arr, nan_count_arr)
        inf_count = int(backend.to_scalar(inf_count_arr))
        nan_count = int(backend.to_scalar(nan_count_arr))

        if inf_count > 0 or nan_count > 0:
            raise ValueError(
                f"Geodesic/chord scale contains {inf_count} inf and {nan_count} nan values. "
                f"This indicates disconnected manifold components or coincident points. "
                f"Increase k_neighbors to improve graph connectivity."
            )

        # Weighted sum of scaled tangent vectors (log maps)
        scale_col = backend.reshape(scale, (-1, 1))
        weights_col = backend.reshape(weights, (-1, 1))
        log_vectors = diff * scale_col

        # Gradient is the weighted mean of log vectors
        gradient = backend.sum(log_vectors * weights_col, axis=0)

        # Check for non-finite values in gradient
        grad_isfinite = backend.isfinite(gradient)
        grad_nonfinite_arr = backend.sum(1 - backend.astype(grad_isfinite, "float32"))
        backend.eval(grad_nonfinite_arr)
        has_nonfinite = int(backend.to_scalar(grad_nonfinite_arr)) > 0

        if has_nonfinite:
            max_finite = 1e38
            grad_isnan = backend.isnan(gradient)
            grad_isinf = backend.isinf(gradient)
            grad_sign = backend.sign(gradient)
            backend.eval(grad_isnan, grad_isinf, grad_sign)

            gradient = backend.where(grad_isnan, backend.zeros_like(gradient), gradient)
            inf_replacement = grad_sign * backend.full(gradient.shape, max_finite)
            gradient = backend.where(grad_isinf, inf_replacement, gradient)
            backend.eval(gradient)

        # Compute gradient norm for step size
        grad_norm_arr = backend.sqrt(backend.sum(gradient * gradient))
        backend.eval(grad_norm_arr)
        grad_norm = float(backend.to_scalar(grad_norm_arr))

        # Adaptive step size based on data scale
        mean_geo_arr = backend.mean(geo_from_mu)
        backend.eval(mean_geo_arr)
        mean_geo = float(backend.to_scalar(mean_geo_arr))
        n_points = int(geo_from_mu.shape[0])
        eps_float = float(machine_epsilon(backend, gradient))

        max_step = max(mean_geo, eps_float)
        base_eta = 1.0 / max(1, n_points)

        if grad_norm > 0 and not is_inf(grad_norm, backend):
            eta = min(base_eta, max_step / grad_norm)
        elif is_inf(grad_norm, backend):
            eta = machine_epsilon(backend, gradient)
        else:
            eta = base_eta

        # Update
        new_mu = mu + eta * gradient

        # Validate output
        backend.eval(new_mu)
        new_mu_isfinite = backend.isfinite(new_mu)
        backend.eval(new_mu_isfinite)
        new_mu_nonfinite_arr = backend.sum(1 - backend.astype(new_mu_isfinite, "float32"))
        backend.eval(new_mu_nonfinite_arr)
        new_mu_nonfinite = int(backend.to_scalar(new_mu_nonfinite_arr))

        if new_mu_nonfinite > 0:
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

        geo_from_mean = self._geodesic_distances_from_query(
            points, mean, geo_result=geo_result
        )

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

        log_μ(x) = (x - μ) * (geodesic_dist / chord_dist)

        Raises:
            ValueError: If scale contains inf or nan values.
        """
        backend = self._backend

        # Attach mean to the k-NN graph for exact discrete geodesics
        geo_from_mean = self._geodesic_distances_from_query(
            points, mean, geo_result=geo_result
        )

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


__all__ = [
    "RiemannianGeometry",
    "_get_riemannian_geometry",
    "_RG_CACHE",
    "_cache",
]
