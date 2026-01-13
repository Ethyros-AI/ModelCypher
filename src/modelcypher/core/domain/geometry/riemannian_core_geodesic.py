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

"""Geodesic distance utilities for RiemannianGeometry."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    infinity_threshold,
    machine_epsilon,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_types import GeodesicDistanceResult
from modelcypher.core.domain.geometry.riemannian_validation import (
    count_inf,
    count_nan,
    count_nonfinite,
    validate_array_numerics,
)

from .riemannian_core_utils import _promote_precision

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()


class RiemannianGeodesicMixin:
    """Geodesic distance and graph construction utilities."""

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
        use_cache: bool = True,
    ) -> GeodesicDistanceResult:
        """
        Compute geodesic distances using a k-NN graph and shortest paths.

        This implements the Isomap-style geodesic computation:
        1. Build a k-NN graph where edge weights are Euclidean (chord) distances
        2. Compute shortest paths (geodesics) using Floyd-Warshall

        CHORD BOOTSTRAP: Local edge weights use Euclidean distances, which
        approximate geodesic distances when k is small (local flatness).
        For highly curved manifolds, alternatives like heat kernel method
        (Varadhan's formula) would provide more accurate local distances.
        See _chord_distance_matrix() docstring for details.

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
        points = _promote_precision(backend.array(points), backend)
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
            return self._compute_geodesic_for_k(points, k_neighbors, use_cache=use_cache)

        # Find minimum k for connectivity - this IS the geometric answer
        if use_cache:
            kmin_key = _cache.make_kmin_key(points, backend)
            cached_k = _cache.get_kmin(kmin_key)
            if cached_k is not None:
                cached_geo = _cache.get_geodesic(
                    _cache.make_geodesic_key(points, backend, int(cached_k))
                )
                if cached_geo is not None:
                    return cached_geo

        k_start_time = time.perf_counter()
        k_min, knn_idx, chord_dist = self._minimum_connected_k(points, use_cache=use_cache)
        if use_cache:
            _cache.set_kmin(kmin_key, k_min, (time.perf_counter() - k_start_time) * 1000)
        return self._compute_geodesic_for_k(
            points, k_min, chord_dist=chord_dist, knn_idx=knn_idx, use_cache=use_cache
        )

    def _minimum_connected_k(
        self,
        points: "Array",
        chord_dist: "Array | None" = None,
        use_cache: bool = True,
    ) -> tuple[int, "Array", "Array"]:
        """Find the minimum k that makes the k-NN graph connected."""
        backend = self._backend
        n = int(points.shape[0])
        if n <= 1:
            return 0, backend.zeros((n, 0)), backend.zeros((n, n))

        # Binary search: start at k=1, double until connected, then binary search
        if chord_dist is None:
            chord_dist = self._chord_distance_matrix(points, use_cache=use_cache)
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
        use_cache: bool = True,
    ) -> GeodesicDistanceResult:
        """Compute geodesic distances for a specific k value."""
        backend = self._backend
        n = int(points.shape[0])
        k_neighbors = max(1, min(k_neighbors, n - 1))

        # Check cache first
        if use_cache:
            cache_key = _cache.make_geodesic_key(points, backend, k_neighbors)
            cached = _cache.get_geodesic(cache_key)
            if cached is not None:
                return cached

        start = time.perf_counter()

        # Compute chord distance matrix (local edge lengths)
        if chord_dist is None:
            chord_dist = self._chord_distance_matrix(points, use_cache=use_cache)
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
        if use_cache:
            _cache.set_geodesic(cache_key, result, elapsed_ms)

        return result

    def _chord_distance_matrix(
        self, points: "Array", use_cache: bool = True
    ) -> "Array":
        """Compute pairwise Euclidean (chord) distances for k-NN graph construction.

        Uses the identity ||a - b||² = ||a||² + ||b||² - 2*a·b to avoid
        O(n² * d) intermediate memory while preserving rotation invariance.

        NOTE ON CHORD BOOTSTRAP:
        This method computes Euclidean (chord) distances, which are used as
        edge weights in the k-NN graph. This is the standard Isomap approach.
        For small k, local Euclidean distances approximate local geodesic
        distances (the manifold is approximately flat at small scales).

        For curved manifolds with high curvature, chords underestimate true
        geodesic arc lengths. Alternative approaches that avoid chord bootstrap:
        - Heat kernel method: d²(x,y) ≈ -4t log(k_t(x,y)) via Varadhan's formula
        - Intrinsic Laplacian eigenvectors
        - Local PCA tangent estimation

        The current Isomap approach is accurate when:
        - k is small relative to manifold curvature (local flatness)
        - The manifold is well-sampled (no gaps)

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

        points = _promote_precision(points, backend)
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
        points = _promote_precision(backend.array(points), backend)
        query = _promote_precision(backend.array(query), backend)
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
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        backend.eval(dist_sq)
        chord_dist = backend.sqrt(dist_sq)

        # Attach query to its k nearest neighbors in the manifold graph.
        # This preserves the discrete geodesic structure and avoids O(n^2).
        if k_neighbors >= n:
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
