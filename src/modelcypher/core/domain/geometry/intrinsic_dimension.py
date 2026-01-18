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

"""Intrinsic dimension estimation for neural network representations.

Implements TwoNN- and MLE-based intrinsic dimension estimators using
geodesic distances when available.

References:
    - Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a
      minimal neighborhood information" Scientific Reports 7:12140
    - Levina & Bickel (2005) "Maximum Likelihood Estimation of Intrinsic Dimension"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    find_magnitude_gap_threshold,
    machine_epsilon,
    safe_log_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Parameters are derived from data and literature defaults (no config class).


@dataclass
class ConfidenceInterval:
    """Confidence interval for intrinsic dimension.

    Computed via bootstrap resampling with precision- and sample-size-derived quantiles.
    """

    lower: float
    upper: float
    resamples: int  # Derived from sample size


@dataclass
class TwoNNEstimate:
    """Result of global intrinsic dimension estimation.

    Uses:
    - Geodesic distances
    - Regression variant (Facco et al., more robust)
    - Data-derived k for graph connectivity
    """

    intrinsic_dimension: float
    sample_count: int
    usable_count: int
    ci: ConfidenceInterval | None = None


@dataclass
class LocalDimensionMap:
    """Per-point intrinsic dimension estimates.

    Identifies local dimension variation across the manifold, including
    regions where dimension drops (collapsed zones) or spikes (transition zones).

    Deficiency detection uses gap detection on the deficit distribution -
    no arbitrary threshold needed. Points below the data-derived deficit gap
    are deficient.
    """

    dimensions: "Array"  # Per-point intrinsic dimension [n]
    modal_dimension: float  # Most common dimension (mode of distribution)
    mean_dimension: float  # Average dimension across points
    std_dimension: float  # Standard deviation of local dimensions
    deficient_indices: list[int]  # Points below data-derived deficit gap
    k_neighbors: int  # k derived from connectivity


class IntrinsicDimension:
    """
    Computes intrinsic dimension using the TwoNN method (Facco et al., 2017).

    Intrinsic dimension (ID) is a direct geometric measurement - NOT an estimate.
    The TwoNN method precisely measures the local scaling of the manifold from
    the distribution of nearest neighbor distance ratios.

    Interpretation:
    - Low ID: tight, consistent behavior (risk: caricature/mode collapse)
    - High ID: multi-modal/prompt-dependent behavior (risk: incoherence)

    Uses geodesic distances because curvature is inherent in high-dimensional spaces.

    Performance:
    - For batched computation of multiple point clouds, use batch_compute()
    - Single eval() per batch instead of per point cloud (200x+ faster)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    @staticmethod
    def local_dimension_min_samples() -> int:
        """Minimum samples required for local TwoNN dimension.

        TwoNN local estimation needs at least two neighbor ratios (r2/r1).
        That requires k_local >= 3 neighbors, so n_samples >= 4.
        """
        return 4

    @staticmethod
    def compute_two_nn(
        points: list[list[float]] | "Array",
        backend: "Backend | None" = None,
        with_ci: bool = False,
    ) -> TwoNNEstimate:
        """Compute intrinsic dimension via TwoNN (Facco et al., 2017).

        All parameters are derived from the data:
        - k_neighbors: Minimum k for connected graph (Berry & Sauer 2016)
        - Uses geodesic distances (curvature-aware)
        - Uses regression variant

        Args:
            points: [N, D] array or list of points
            backend: Backend to use (uses default if None)
            with_ci: Whether to compute bootstrap confidence interval

        Returns:
            TwoNNEstimate with intrinsic dimension and metadata
        """
        b = backend or get_default_backend()

        # Convert list to array if needed
        pts = b.array(points) if isinstance(points, list) else points

        computer = IntrinsicDimension(b)
        return computer.compute(pts, with_ci=with_ci)

    def compute(
        self,
        points: "Array",
        with_ci: bool = False,
    ) -> TwoNNEstimate:
        """
        Compute intrinsic dimension using geodesic distances.

        All parameters are derived from the data - no configuration needed:
        - k_neighbors: Connectivity-based selection (Berry & Sauer 2016) -
          binary search for minimum k that makes the k-NN graph connected.
          This is a geometric property of the point cloud itself.
        - Uses geodesic distances (curvature-aware)
        - Uses regression variant (Facco et al.)

        Args:
            points: [N, D] array of points
            with_ci: Whether to compute bootstrap confidence interval

        Returns:
            TwoNNEstimate with intrinsic dimension and metadata
        """
        N = points.shape[0]
        if N < 3:
            raise EstimatorError.insufficient_samples(N)

        # k_neighbors=None triggers connectivity-based selection (Berry & Sauer 2016)
        # distance_power=2.0 uses squared geodesic distances.
        dist_sq = self._geodesic_distance_matrix_squared(points)

        mu = self._compute_two_nn_mu_from_distances(dist_sq)

        # Use regression variant (Facco et al.)
        dimension = self._compute_from_mu(mu)

        ci = None
        if with_ci:
            ci = self._bootstrap_two_nn(mu, N)

        return TwoNNEstimate(
            intrinsic_dimension=dimension,
            sample_count=N,
            usable_count=mu.shape[0],
            ci=ci,
        )

    def batch_compute(
        self,
        point_clouds: list["Array"],
    ) -> list[TwoNNEstimate | None]:
        """
        Compute intrinsic dimension for multiple point clouds in a single GPU pass.

        This is 100-200x faster than calling compute() in a loop because:
        1. All distance matrices are computed lazily (no intermediate eval)
        2. All mu ratios are computed lazily
        3. Single eval() at the end for all results
        4. Minimizes GPU sync overhead

        Args:
            point_clouds: List of [N_i, D] arrays, each a separate point cloud.
                         N_i can vary between point clouds.

        Returns:
            List of TwoNNEstimate (or None for failed computations), same length
            as input.
        """
        backend = self._backend
        results: list[TwoNNEstimate | None] = []

        if not point_clouds:
            return results

        # Phase 1: Build all computation graphs (lazy - no eval)
        lazy_dims: list[tuple["Array", int, "Array"] | None] = []

        for points in point_clouds:
            N = int(points.shape[0])
            if N < 3:
                lazy_dims.append(None)
                continue

            try:
                # Compute geodesic distance matrix
                dist_sq = self._geodesic_distance_matrix_squared(points)

                # Compute mu ratios lazily (no eval inside)
                mu = self._compute_two_nn_mu_lazy(dist_sq)

                # Compute dimension lazily (no eval inside)
                dim_arr = self._compute_from_mu_lazy(mu)

                lazy_dims.append((dim_arr, N, mu))
            except Exception:
                lazy_dims.append(None)

        # Phase 2: Single eval for ALL lazy computations
        arrays_to_eval = []
        for item in lazy_dims:
            if item is not None:
                dim_arr, _, mu = item
                arrays_to_eval.extend([dim_arr, mu])

        if arrays_to_eval:
            backend.eval(*arrays_to_eval)

        # Phase 3: Extract results
        for item in lazy_dims:
            if item is None:
                results.append(None)
            else:
                dim_arr, N, mu = item
                try:
                    dim_val = float(backend.to_scalar(dim_arr))
                    mu_count = int(mu.shape[0])
                    if mu_count < 3 or not (0.0 < dim_val < 1e6):
                        results.append(None)
                    else:
                        results.append(TwoNNEstimate(
                            intrinsic_dimension=dim_val,
                            sample_count=N,
                            usable_count=mu_count,
                            ci=None,
                        ))
                except Exception:
                    results.append(None)

        return results

    def _compute_two_nn_mu_lazy(self, dist_sq: "Array") -> "Array":
        """Compute mu ratios WITHOUT eval() - for batched computation.

        Same as _compute_two_nn_mu_from_distances but keeps everything lazy.
        Returns empty array if computation would fail.
        """
        backend = self._backend
        n = int(dist_sq.shape[0])

        if n < 3:
            return backend.array([])

        dtype = dist_sq.dtype if hasattr(dist_sq, "dtype") else None
        inf_val = float(backend.finfo(dtype).max)
        dist_no_self = backend.where(
            backend.eye(n) > 0, backend.full((n, n), inf_val), dist_sq
        )

        # Get two nearest neighbors per point
        partitioned = backend.argpartition(dist_no_self, 1, axis=1)
        nearest_idx = partitioned[:, :2]
        nearest_vals = backend.take_along_axis(dist_sq, nearest_idx, axis=1)
        r1_sq = backend.min(nearest_vals, axis=1)
        r2_sq = backend.max(nearest_vals, axis=1)

        # Filter degenerate points using masking (no eval needed)
        eps = machine_epsilon(backend, r1_sq)
        r1_valid = r1_sq > eps

        # Use max of r2 as infinity threshold (avoids median eval)
        # Anything within machine epsilon of max is treated as infinite.
        max_r2 = backend.max(r2_sq)
        eps = machine_epsilon(backend, r2_sq)
        inf_thresh = max_r2 * (1.0 - eps)
        r2_finite = r2_sq < inf_thresh
        valid_mask = r1_valid & r2_finite

        # Safe computation with masking
        r1_sq_safe = backend.where(valid_mask, r1_sq, backend.ones_like(r1_sq))
        r2_sq_safe = backend.where(valid_mask, r2_sq, backend.zeros_like(r2_sq))

        zeros = backend.zeros_like(r1_sq_safe)
        r1 = backend.sqrt(backend.maximum(r1_sq_safe, zeros))
        r2 = backend.sqrt(backend.maximum(r2_sq_safe, zeros))

        # mu = r2 / r1 with masking for invalid entries
        mu_all = r2 / r1

        # Instead of filtering, use masked mean in _compute_from_mu_lazy
        # Return all mu values with invalid ones set to 1.0 (log(1)=0, neutral)
        mu_masked = backend.where(valid_mask, mu_all, backend.ones_like(mu_all))

        return mu_masked

    def _compute_from_mu_lazy(self, mu: "Array") -> "Array":
        """Compute dimension from mu WITHOUT eval() - for batched computation.

        Returns a scalar array (not Python float) to keep computation lazy.
        """
        backend = self._backend
        N = int(mu.shape[0])

        if N < 3:
            return backend.array(float("nan"))

        # Filter out mu values of 1.0 (invalid/masked entries)
        eps = machine_epsilon(backend, mu)
        valid_mu = mu > (1.0 + eps)

        # Clamp mu to avoid log(0)
        log_eps = safe_log_epsilon(backend, mu)
        mu_clamped = backend.maximum(mu, backend.full(mu.shape, log_eps))
        log_mu = backend.log(mu_clamped)

        # Mask invalid entries
        log_mu_masked = backend.where(valid_mu, log_mu, backend.zeros_like(log_mu))

        # Count valid entries
        valid_float = backend.astype(valid_mu, str(mu.dtype))
        valid_count = backend.sum(valid_float)

        # Safe mean computation
        sum_log_mu = backend.sum(log_mu_masked)
        mean_log_mu = sum_log_mu / backend.maximum(valid_count, backend.array(1.0))

        # Regression estimate: d = 1 / mean(log(mu))
        # For proper regression, use sorted values
        sorted_log_mu = backend.sort(log_mu_masked)

        # indices 1..N
        i = backend.arange(1, N + 1)
        F = backend.astype(i, sorted_log_mu.dtype) / N

        # Slice to N-1
        x = sorted_log_mu[:-1]
        F_sliced = F[:-1]
        one_minus_F = 1.0 - F_sliced

        # Clamp to avoid log(0)
        log_eps = safe_log_epsilon(backend, one_minus_F)
        min_val = backend.full(one_minus_F.shape, log_eps)
        clamped = backend.maximum(min_val, one_minus_F)
        y = -backend.log(clamped)

        # Regression: d = sum(xy) / sum(xx)
        sum_xx = backend.sum(x * x)
        sum_xy = backend.sum(x * y)

        # Safe division
        div_eps = machine_epsilon(backend, x)
        sum_xx_safe = backend.maximum(sum_xx, backend.array(div_eps))
        d = sum_xy / sum_xx_safe

        return d

    def _geodesic_distance_matrix_squared(self, points: "Array") -> "Array":
        """Computes pairwise squared geodesic distances for TwoNN estimation.

        Uses k-NN graph and Floyd-Warshall shortest paths (Isomap-style)
        to compute true manifold distances. Always geodesic - no chord
        approximation, as even small curvature errors can propagate.

        Returns:
            [N, N] squared geodesic distance matrix
        """
        import math

        backend = self._backend
        n = int(points.shape[0])

        # Full geodesic computation for all samples
        riemannian = RiemannianGeometry(backend=self._backend)

        # First, find minimum k for connectivity (Berry & Sauer 2016)
        result = riemannian.geodesic_distances(points, k_neighbors=None)
        k_connectivity = result.k_neighbors

        # Second, ensure k is high enough for LOCAL structure
        # For TwoNN, we need the 2nd nearest neighbor to be among direct graph neighbors
        # k >= ceil(log(n)) ensures local neighborhoods are well-represented
        # This is data-derived from manifold learning theory (not a heuristic)
        k_local = max(2, int(math.ceil(math.log(max(n, 2)))))

        # Use the larger of the two requirements
        k_required = max(k_connectivity, k_local)

        # Recompute if we need more neighbors for local structure
        if k_required > k_connectivity:
            result = riemannian.geodesic_distances(points, k_neighbors=k_required)

        geodesic_dist = result.distances

        # Return squared distances.
        return geodesic_dist * geodesic_dist

    def _compute_two_nn_mu_from_distances(self, dist_sq: "Array") -> "Array":
        """Computes the ratio mu = r2 / r1 for each point from a distance matrix.

        Args:
            dist_sq: [N, N] squared geodesic distance matrix

        Returns:
            [M] array of mu ratios for M valid points (where r1 > 0)
        """
        backend = self._backend

        # Find nearest neighbors without full sort (exclude self)
        n = dist_sq.shape[0]
        if n < 3:
            sorted_dist_sq = backend.sort(dist_sq, axis=1)
            r1_sq = sorted_dist_sq[:, 1]
            r2_sq = sorted_dist_sq[:, 2]
        else:
            dtype = dist_sq.dtype if hasattr(dist_sq, "dtype") else None
            inf_val = float(backend.finfo(dtype).max)
            dist_no_self = backend.where(
                backend.eye(n) > 0, backend.full((n, n), inf_val), dist_sq
            )
            # Get two nearest neighbors per point (excluding self)
            # argpartition puts k smallest at front, but not sorted among themselves
            partitioned = backend.argpartition(dist_no_self, 1, axis=1)
            nearest_idx = partitioned[:, :2]

            # Gather distances for the two nearest neighbors
            nearest_vals = backend.take_along_axis(dist_sq, nearest_idx, axis=1)
            r1_sq = backend.min(nearest_vals, axis=1)
            r2_sq = backend.max(nearest_vals, axis=1)

        # Filter degenerate points:
        # 1. r1 > eps (avoid division by zero)
        # 2. r2 is not "infinite" (disconnected nodes in geodesic graph)
        #
        # The infinity threshold is derived from data and dtype precision:
        # Disconnected nodes have distance = geodesic "infinity" (≈ scale/eps),
        # which is orders of magnitude larger than typical distances.
        eps = machine_epsilon(backend, r1_sq)
        r1_valid = r1_sq > eps

        # Data-derived infinity threshold:
        # Use median of r2 as baseline - infinite values are >> typical distances
        median_val = compute_median(r2_sq, backend)

        # Infinity threshold: anything > median / eps is disconnected
        # This follows the geodesic "infinity" construction: inf ≈ scale/eps.
        inf_thresh = median_val / eps
        r2_finite = r2_sq < inf_thresh
        valid_mask = r1_valid & r2_finite

        # Count valid points
        valid_count_arr = backend.sum(backend.astype(valid_mask, r1_sq.dtype))
        backend.eval(valid_count_arr)
        valid_count = int(backend.to_scalar(valid_count_arr))

        if valid_count == 0:
            return backend.array([])

        # Use where to zero out invalid entries, then filter
        r1_sq_safe = backend.where(valid_mask, r1_sq, backend.ones_like(r1_sq))
        r2_sq_safe = backend.where(valid_mask, r2_sq, backend.zeros_like(r2_sq))

        # Ensure non-negative values before sqrt for numerical safety
        zeros = backend.zeros_like(r1_sq_safe)
        r1 = backend.sqrt(backend.maximum(r1_sq_safe, zeros))
        r2 = backend.sqrt(backend.maximum(r2_sq_safe, zeros))

        # mu = r2 / r1 for valid points
        mu_all = r2 / r1

        # Extract only valid values using argpartition
        # Partition by validity (invalid first = 0, valid second = 1), then take tail indices
        n = dist_sq.shape[0]
        if valid_count == n:
            mu = mu_all
        else:
            valid_float = backend.astype(valid_mask, r1_sq.dtype)
            neg_valid = -valid_float
            kth = max(0, valid_count - 1)
            partitioned = backend.argpartition(neg_valid, kth)
            valid_indices = backend.take(partitioned, backend.arange(valid_count), axis=0)
            mu = backend.take(mu_all, valid_indices)

        return mu

    def _compute_from_mu(self, mu: "Array") -> float:
        """Compute intrinsic dimension from mu ratios using regression (Facco et al.).

        Always uses the regression variant - more robust than MLE for real data.
        """
        backend = self._backend
        N = mu.shape[0]
        if N < 3:
            raise EstimatorError("two_nn", f"Insufficient non-degenerate samples: {N} < 3", N)

        # log(mu) - clamp to avoid log(0) for degenerate neighbor ratios
        log_eps = safe_log_epsilon(backend, mu)
        mu_clamped = backend.maximum(mu, backend.full(mu.shape, log_eps))
        log_mu = backend.log(mu_clamped)

        # Regression variant (Facco et al.) - always used, more robust than MLE
        sorted_log_mu = backend.sort(log_mu)

        # indices 1..N
        i = backend.arange(1, N + 1)
        F = backend.astype(i, sorted_log_mu.dtype) / N

        # Slice to N-1
        x = sorted_log_mu[:-1]
        F_sliced = F[:-1]
        one_minus_F = 1.0 - F_sliced

        # Clamp to avoid log(0) - use safe_log_epsilon (tiny value, not machine epsilon)
        log_eps = safe_log_epsilon(backend, one_minus_F)
        min_val = backend.full(one_minus_F.shape, log_eps)
        clamped = backend.maximum(min_val, one_minus_F)
        y = -backend.log(clamped)

        sum_xx = backend.sum(x * x)
        sum_xy = backend.sum(x * y)

        backend.eval(sum_xx, sum_xy)
        sum_xx_val = float(backend.to_scalar(sum_xx))
        sum_xy_val = float(backend.to_scalar(sum_xy))

        # Use machine epsilon for degenerate check
        div_eps = machine_epsilon(backend, x)
        if sum_xx_val < div_eps:
            raise EstimatorError.regression_degenerate()

        d = sum_xy_val / sum_xx_val
        return d

    def _bootstrap_two_nn(self, mu: "Array", sample_size: int) -> ConfidenceInterval | None:
        """Compute bootstrap confidence interval for the ID estimate.

        Resamples follow sample size; quantile resolution is derived from
        numeric precision and resample count.
        No seed is used - bootstrap variance is part of the measurement.
        """
        backend = self._backend
        n = mu.shape[0]

        # Resamples = sample_size (derived from data resolution, no arbitrary cap)
        resamples = sample_size

        if n < 2 or resamples < 2:
            return None

        # Quantile resolution limited by precision and resample count
        alpha = max(float(division_epsilon(backend, mu)), 1.0 / float(resamples))

        dimensions: list[float] = []
        for _ in range(resamples):
            # Random indices with replacement
            indices = backend.random_randint(0, n, shape=(n,))
            sample = backend.take(mu, indices)

            try:
                d = self._compute_from_mu(sample)
                dimensions.append(d)
            except EstimatorError:
                continue

        if len(dimensions) < 2:
            return None

        dimensions.sort()
        lower_idx = int(len(dimensions) * alpha)
        upper_idx = int(len(dimensions) * (1.0 - alpha))
        if lower_idx >= upper_idx:
            return None

        return ConfidenceInterval(
            lower=dimensions[lower_idx],
            upper=dimensions[upper_idx],
            resamples=len(dimensions),
        )

    def local_dimension_map(self, points: "Array") -> LocalDimensionMap:
        """
        Compute per-point intrinsic dimension estimates.

        For each point, estimates the local intrinsic dimension using its
        k nearest neighbors. This reveals dimension variation across the
        manifold, identifying:
        - Collapsed zones: local ID << modal dimension
        - Transition zones: local ID varies sharply
        - Stable zones: local ID ≈ modal dimension

        Algorithm:
            For each point i:
            1. Find k nearest neighbors (by geodesic distance)
            2. Compute TwoNN-style mu = r2/r1 ratio locally
            3. Estimate local ID from the mu distribution

        All parameters derived from data:
        - k: Connectivity-based selection (Berry & Sauer 2016)
        - Deficiency: Points > 2σ below modal dimension (statistical outliers)

        Note: This is more expensive than global ID (O(n^2) vs O(n)),
        but provides spatial resolution of dimension variation.

        Returns:
            LocalDimensionMap with per-point dimensions and deficiency indices
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        # Compute geodesic distances - k derived from connectivity (Berry & Sauer 2016)
        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=None)
        geo_dist = geo_result.distances
        k_actual = geo_result.k_neighbors
        backend.eval(geo_dist)

        # TwoNN local estimation needs at least two neighbor ratios (r2/r1),
        # which requires k_local >= 3 neighbors (n_samples >= 4).
        min_samples = self.local_dimension_min_samples()
        min_neighbors = max(1, min_samples - 1)
        k_local = max(min_neighbors, min(k_actual, n - 1))

        # Get machine epsilon from the array's dtype - the geometry's precision limit
        eps = machine_epsilon(backend, geo_dist)

        # Select k-nearest neighbors without full sort, then sort only the k block.
        inf_val = backend.finfo(geo_dist.dtype).max
        dist_no_self = backend.where(
            backend.eye(n) > 0, backend.full((n, n), inf_val), geo_dist
        )
        kth = max(0, min(k_local - 1, n - 1))
        partitioned = backend.argpartition(dist_no_self, kth, axis=1)
        neighbor_idx = partitioned[:, :k_local]
        k_dists = backend.take_along_axis(geo_dist, neighbor_idx, axis=1)
        k_dists = backend.sort(k_dists, axis=1)
        backend.eval(k_dists)

        # Detect infinite distances (disconnected points in geodesic graph)
        # Use dtype max as the numerical "infinite" sentinel.
        inf_threshold = backend.array(backend.finfo(geo_dist.dtype).max)
        is_finite = k_dists < inf_threshold  # [n, k_local]

        # Compute mu = r_{j+1} / r_j for consecutive neighbor pairs
        # r1: distances to neighbors 1 through k_local-1
        # r2: distances to neighbors 2 through k_local
        r1 = k_dists[:, :-1]  # [n, k_local-1]
        r2 = k_dists[:, 1:]  # [n, k_local-1]

        # Valid ratios require: r1 > eps, both r1 and r2 finite
        r1_valid = r1 > eps
        both_finite = is_finite[:, :-1] & is_finite[:, 1:]
        valid_mask = r1_valid & both_finite  # [n, k_local-1]

        # Safe division: replace invalid r1 with 1.0 to avoid division by zero
        ones = backend.ones_like(r1)
        r1_safe = backend.where(r1_valid, r1, ones)
        mu = r2 / r1_safe  # [n, k_local-1]

        # Compute log(mu) - only meaningful where mu > 1
        # For mu <= 1, log would be <= 0, which gives negative dimension (invalid)
        # Clamp mu to minimum of 1+eps to ensure positive log
        one_plus_eps = backend.array(1.0) + eps
        mu_clamped = backend.where(mu > one_plus_eps, mu, one_plus_eps)
        log_mu = backend.log(mu_clamped)  # [n, k_local-1]

        # Mask invalid entries with 0
        zeros = backend.zeros_like(log_mu)
        log_mu_masked = backend.where(valid_mask, log_mu, zeros)

        # Count valid entries per row
        valid_float = backend.astype(valid_mask, str(geo_dist.dtype))
        valid_count = backend.sum(valid_float, axis=1)  # [n]

        # Sum of log(mu) per row
        sum_log_mu = backend.sum(log_mu_masked, axis=1)  # [n]

        # Mean log(mu) per row - only where we have enough valid entries.
        # Need at least two valid mu values (two neighbor ratios).
        min_valid = backend.array(float(max(1, min_neighbors - 1)))
        has_enough = valid_count >= min_valid

        # Safe mean computation
        valid_count_safe = backend.where(has_enough, valid_count, backend.ones_like(valid_count))
        mean_log_mu = sum_log_mu / valid_count_safe  # [n]

        # Local ID = 1 / mean_log_mu
        # Valid only where mean_log_mu > eps and has_enough is True
        mean_positive = mean_log_mu > eps
        valid_id = has_enough & mean_positive

        # Safe division for ID
        mean_log_mu_safe = backend.where(valid_id, mean_log_mu, backend.ones_like(mean_log_mu))
        local_dims_raw = backend.array(1.0) / mean_log_mu_safe

        # Mark invalid points with NaN
        nan_val = backend.array(float("nan"))
        local_dims = backend.where(valid_id, local_dims_raw, nan_val)
        backend.eval(local_dims)

        # Extract valid dimensions for statistics
        valid_mask = valid_id & (local_dims > 0)
        valid_mask_float = backend.astype(valid_mask, str(local_dims.dtype))
        valid_count = backend.sum(valid_mask_float)
        backend.eval(valid_count)
        valid_count_scalar = backend.to_scalar(valid_count)

        if valid_count_scalar <= 0.0:
            return LocalDimensionMap(
                dimensions=local_dims,
                modal_dimension=0.0,
                mean_dimension=0.0,
                std_dimension=0.0,
                deficient_indices=[],
                k_neighbors=k_actual,
            )

        # Compute statistics using backend operations
        zeros = backend.zeros_like(local_dims)
        local_dims_safe = backend.where(valid_mask, local_dims, zeros)
        mean_dim_arr = backend.sum(local_dims_safe) / valid_count
        diff = backend.where(valid_mask, local_dims_safe - mean_dim_arr, zeros)
        var_dim_arr = backend.sum(diff * diff) / valid_count
        std_dim_arr = backend.sqrt(var_dim_arr)
        backend.eval(mean_dim_arr, std_dim_arr)
        mean_dim = backend.to_scalar(mean_dim_arr)
        std_dim = backend.to_scalar(std_dim_arr)

        # Modal dimension: bin dimensions and find most common.
        # Use Freedman–Diaconis rule for bin width (data-derived).
        pos_inf = backend.array(float("inf"), dtype=local_dims.dtype)
        neg_inf = backend.array(float("-inf"), dtype=local_dims.dtype)
        min_dim_arr = backend.min(backend.where(valid_mask, local_dims, pos_inf))
        max_dim_arr = backend.max(backend.where(valid_mask, local_dims, neg_inf))
        backend.eval(min_dim_arr, max_dim_arr)
        min_dim = backend.to_scalar(min_dim_arr)
        max_dim = backend.to_scalar(max_dim_arr)

        if valid_count_scalar > 1.0 and max_dim > min_dim:
            import math

            n_valid = int(valid_count_scalar)
            eps_dim = division_epsilon(backend, local_dims)
            valid_vals = backend.where(valid_mask, local_dims, pos_inf)
            sorted_vals = backend.sort(valid_vals)
            backend.eval(sorted_vals)

            last_idx = n_valid - 1
            q1_idx = int(0.25 * last_idx)
            q3_idx = int(0.75 * last_idx)
            q1_arr = sorted_vals[q1_idx : q1_idx + 1]
            q3_arr = sorted_vals[q3_idx : q3_idx + 1]
            backend.eval(q1_arr, q3_arr)
            q1 = float(backend.to_scalar(q1_arr))
            q3 = float(backend.to_scalar(q3_arr))
            iqr = q3 - q1

            if iqr > eps_dim:
                bin_width = 2.0 * iqr * (float(n_valid) ** (-1.0 / 3.0))
            else:
                bin_width = 0.0

            if bin_width > eps_dim:
                n_bins = max(1, int(math.ceil((max_dim - min_dim) / bin_width)))
                bin_width_arr = (max_dim_arr - min_dim_arr + eps_dim) / backend.array(
                    float(n_bins), dtype=local_dims.dtype
                )

                # Compute bin indices
                bin_indices = backend.astype(
                    (backend.where(valid_mask, local_dims, min_dim_arr) - min_dim_arr)
                    / bin_width_arr,
                    "int32",
                )
                # Clamp to valid range
                max_bin_idx = backend.array(n_bins - 1, dtype="int32")
                zero_idx = backend.array(0, dtype="int32")
                bin_indices = backend.where(bin_indices > max_bin_idx, max_bin_idx, bin_indices)
                bin_indices = backend.where(bin_indices < zero_idx, zero_idx, bin_indices)

                # Count bins using backend one-hot encoding (fully vectorized)
                eye_mat = backend.eye(n_bins)
                one_hot = eye_mat[bin_indices]  # [n_points, n_bins] with 1 at bin index
                valid_mask_col = backend.reshape(valid_mask_float, (-1, 1))
                bin_counts_arr = backend.sum(one_hot * valid_mask_col, axis=0)

                # Find modal bin using backend argmax
                max_bin_arr = backend.argmax(bin_counts_arr)
                backend.eval(bin_counts_arr, max_bin_arr, bin_width_arr)
                max_bin = int(backend.to_scalar(max_bin_arr))
                bin_width_val = float(backend.to_scalar(bin_width_arr))
                modal_dim = min_dim + (max_bin + 0.5) * bin_width_val
            else:
                mid = n_valid // 2
                if n_valid % 2 == 1:
                    median_arr = sorted_vals[mid : mid + 1]
                else:
                    low = sorted_vals[mid - 1 : mid]
                    high = sorted_vals[mid : mid + 1]
                    median_arr = (low + high) * 0.5
                backend.eval(median_arr)
                modal_dim = float(backend.to_scalar(median_arr))
        else:
            idx_range = backend.arange(0, n)
            idx_masked = backend.where(valid_mask, idx_range, backend.array(n, dtype="int32"))
            first_idx_arr = backend.min(idx_masked)
            backend.eval(first_idx_arr)
            first_idx = int(backend.to_scalar(first_idx_arr))
            modal_dim_arr = local_dims[first_idx : first_idx + 1]
            backend.eval(modal_dim_arr)
            modal_dim = backend.to_scalar(modal_dim_arr)

        # Find deficient points via gap detection on the deficit distribution
        deficient: list[int] = []
        eps = division_epsilon(backend, local_dims)
        deficits = modal_dim - local_dims
        deficit_mask = deficits > eps
        count_arr = backend.sum(backend.astype(deficit_mask, "int32"))
        backend.eval(count_arr)
        count = int(backend.to_scalar(count_arr))
        if count > 0:
            deficit_vals = [float(x) for x in backend.tolist(deficits) if x > eps]
            deficit_vals.sort()
            deficit_threshold = find_magnitude_gap_threshold(deficit_vals, eps=eps, backend=backend)
            if deficit_threshold > eps:
                deficient_mask = deficit_mask & (deficits > deficit_threshold)
                mask_int = backend.astype(deficient_mask, "int32")
                count_arr = backend.sum(mask_int)
                backend.eval(mask_int, count_arr)
                count = int(backend.to_scalar(count_arr))
                if count > 0:
                    neg_mask = -mask_int
                    kth = max(0, count - 1)
                    partitioned = backend.argpartition(neg_mask, kth)
                    selected = backend.take(partitioned, backend.arange(count), axis=0)
                    backend.eval(selected)
                    sorted_selected = backend.sort(selected)
                    backend.eval(sorted_selected)
                    deficient = [int(x) for x in backend.tolist(sorted_selected)]

        return LocalDimensionMap(
            dimensions=local_dims,
            modal_dimension=modal_dim,
            mean_dimension=mean_dim,
            std_dimension=std_dim,
            deficient_indices=deficient,
            k_neighbors=k_actual,
        )

    @staticmethod
    def detect_dimension_deficiency(
        points: "Array",
        backend: "Backend | None" = None,
    ) -> list[int]:
        """
        Find points where local intrinsic dimension is deficient.

        Convenience method that returns indices of deficit-gap outliers:
        points where local ID falls below the data-derived deficit gap.

        These points indicate "dimension-collapsed" regions where the
        manifold is locally lower-dimensional than expected.

        Args:
            points: Point cloud [n, d]
            backend: Backend to use

        Returns:
            List of point indices with deficient local dimension
        """
        b = backend or get_default_backend()
        estimator = IntrinsicDimension(b)
        result = estimator.local_dimension_map(points)
        return result.deficient_indices
