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

Measures the true dimensionality of the manifold on which activations lie,
independent of the ambient (embedding) dimension. A 4096-dimensional embedding
may have intrinsic dimension of only 50-200.

Mathematical Foundation:
    TwoNN Estimator (Facco et al., 2017): Uses ratio of distances to first and
    second nearest neighbors. For a d-dimensional manifold, the ratio r = d₂/d₁
    follows a specific distribution that depends only on d.

    The estimator uses geodesic distances (not Euclidean) because curvature is
    inherent in high-dimensional spaces. Euclidean distance systematically
    under/overestimates true manifold distance depending on local curvature.

References:
    - Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a
      minimal neighborhood information" Scientific Reports 7:12140
    - Levina & Bickel (2005) "Maximum Likelihood Estimation of Intrinsic Dimension"

Research Connections:
    Intrinsic dimension tracking during training reveals geometric evolution of
    representations. Shen et al. (arXiv 2507.01966) found that brain-AI alignment
    precedes performance improvements—suggesting models develop brain-like
    representational structure as a stepping stone to capability.

    Our hypothesis: Models may show dimension expansion (learning) followed by
    compression (abstraction), analogous to the Blue Brain Project's "build then
    raze" pattern in biological circuits.

    See also: docs/RESEARCH-CONNECTIONS.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class GeodesicConfiguration:
    """Configuration for geodesic distance estimation.

    In high-dimensional spaces, curvature is inherent. Geodesic distance is
    the correct metric. Geodesic distances are computed via k-NN graph
    shortest paths (Isomap-style).

    k_neighbors: When None, uses connectivity-based selection (Berry & Sauer 2016).
                 This binary searches for the minimum k that makes the graph connected,
                 which is a geometric property of the point cloud itself.
    """

    k_neighbors: int | None = None
    distance_power: float = 2.0


@dataclass
class TwoNNConfiguration:
    """Two-nearest-neighbors estimator configuration.

    Attributes:
        use_regression: Use regression variant (Facco et al.) vs MLE.
        geodesic: Geodesic distance configuration. Always uses geodesic
            distances since curvature is inherent in high-dimensional spaces.
    """

    use_regression: bool = True
    geodesic: GeodesicConfiguration = field(default_factory=GeodesicConfiguration)


@dataclass
class BootstrapConfiguration:
    """Bootstrap configuration for confidence intervals."""

    resamples: int = 200
    confidence_level: float = 0.95
    seed: int = 42


@dataclass
class ConfidenceInterval:
    """Confidence interval for intrinsic dimension."""

    level: float
    lower: float
    upper: float
    resamples: int
    seed: int


@dataclass
class TwoNNEstimate:
    """Result of global intrinsic dimension estimation."""

    intrinsic_dimension: float
    sample_count: int
    usable_count: int
    uses_regression: bool
    uses_geodesic: bool = False
    ci: ConfidenceInterval | None = None


@dataclass
class LocalDimensionMap:
    """Per-point intrinsic dimension estimates.

    Identifies local dimension variation across the manifold, including
    regions where dimension drops (collapsed zones) or spikes (transition zones).
    """

    dimensions: "Array"  # Per-point intrinsic dimension [n]
    modal_dimension: float  # Most common dimension (mode of distribution)
    mean_dimension: float  # Average dimension across points
    std_dimension: float  # Standard deviation of local dimensions
    deficient_indices: list[int]  # Points where local ID < threshold * modal_dimension
    deficiency_threshold: float  # Threshold used (e.g., 0.8)
    k_neighbors: int  # k used for local estimation


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
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    @staticmethod
    def compute_two_nn(
        points: list[list[float]] | "Array",
        configuration: "TwoNNConfiguration | None" = None,
        backend: "Backend | None" = None,
    ) -> TwoNNEstimate:
        """Static convenience method for computing intrinsic dimension.

        Args:
            points: [N, D] array or list of points
            configuration: Computation config (uses defaults if None)
            backend: Backend to use (uses default if None)

        Returns:
            TwoNNEstimate with intrinsic dimension and metadata
        """
        b = backend or get_default_backend()
        config = configuration or TwoNNConfiguration()

        # Convert list to array if needed
        pts = b.array(points) if isinstance(points, list) else points

        computer = IntrinsicDimension(b)
        return computer.compute(pts, config)

    def compute(
        self,
        points: "Array",
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
        bootstrap: BootstrapConfiguration | None = None,
    ) -> TwoNNEstimate:
        """
        Compute intrinsic dimension using geodesic distances.

        When k_neighbors is None in configuration, uses connectivity-based k
        selection (Berry & Sauer 2016): binary search for the minimum k that
        makes the k-NN graph connected. This is a geometric property of the
        point cloud itself, not a heuristic.

        Args:
            points: [N, D] array of points
            configuration: Computation config
            bootstrap: Optional bootstrap configuration for confidence intervals

        Note:
            Geodesic distances are computed via k-NN graph shortest paths.
            This is the correct metric for curved manifolds.
        """
        N = points.shape[0]
        if N < 3:
            raise EstimatorError.insufficient_samples(N)

        # k_neighbors is either specified, or None for connectivity-based selection
        k_neighbors = configuration.geodesic.k_neighbors

        # Compute with geodesic distances (k=None triggers connectivity-based selection)
        dist_sq = self._geodesic_distance_matrix_squared(
            points,
            k_neighbors=k_neighbors,
            distance_power=configuration.geodesic.distance_power,
        )

        mu = self._compute_two_nn_mu_from_distances(dist_sq)

        dimension = self._compute_from_mu(mu, use_regression=configuration.use_regression)

        ci = None
        if bootstrap:
            ci = self._bootstrap_two_nn(
                mu,
                use_regression=configuration.use_regression,
                config=bootstrap,
            )

        return TwoNNEstimate(
            intrinsic_dimension=dimension,
            sample_count=N,
            usable_count=mu.shape[0],
            uses_regression=configuration.use_regression,
            uses_geodesic=True,
            ci=ci,
        )

    def _geodesic_distance_matrix_squared(
        self,
        points: "Array",
        k_neighbors: int | None,
        distance_power: float = 2.0,
    ) -> "Array":
        """Computes pairwise squared geodesic distances via k-NN graph.

        Uses the Isomap-style approach:
        1. Build k-nearest-neighbor graph with Euclidean edge weights
        2. Compute shortest paths = geodesics on the discrete manifold

        The k-NN graph represents the discrete manifold. Geodesic distance on
        this graph is exact (not an approximation). This corrects for curvature
        effects where Euclidean distance is incorrect:
        - Positive curvature: Euclidean underestimates true distance
        - Negative curvature: Euclidean overestimates true distance

        Args:
            points: [N, D] array of points
            k_neighbors: Number of neighbors for graph. When None, uses connectivity-based
                         selection (Berry & Sauer 2016) - the geometric answer.
            distance_power: Power for distance weighting (2.0 = squared distances)

        Returns:
            [N, N] squared geodesic distance matrix
        """
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        riemannian = RiemannianGeometry(backend=self._backend)

        # Get geodesic distances (k=None triggers connectivity-based selection)
        result = riemannian.geodesic_distances(points, k_neighbors=k_neighbors)
        geodesic_dist = result.distances

        # Return squared distances
        return geodesic_dist * geodesic_dist

    def _compute_two_nn_mu_from_distances(self, dist_sq: "Array") -> "Array":
        """Computes the ratio mu = r2 / r1 for each point from a distance matrix.

        Args:
            dist_sq: [N, N] squared distance matrix (Euclidean or geodesic)

        Returns:
            [M] array of mu ratios for M valid points (where r1 > 0)
        """
        backend = self._backend

        # Find nearest neighbors (excluding self at index 0 in sorted order)
        sorted_dist_sq = backend.sort(dist_sq, axis=1)

        # index 0 is self (dist 0), index 1 is NN1, index 2 is NN2
        r1_sq = sorted_dist_sq[:, 1]
        r2_sq = sorted_dist_sq[:, 2]

        # Filter degenerate points (r1 > 0 to avoid division by zero)
        # Use machine epsilon for precision-aware threshold
        eps = machine_epsilon(backend, r1_sq)
        valid_mask = r1_sq > eps

        # Count valid points
        valid_count_arr = backend.sum(backend.astype(valid_mask, r1_sq.dtype))
        backend.eval(valid_count_arr)
        valid_count = int(backend.to_scalar(valid_count_arr))

        if valid_count == 0:
            return backend.array([])

        # Use where to zero out invalid entries, then filter
        r1_sq_safe = backend.where(valid_mask, r1_sq, backend.ones_like(r1_sq))
        r2_sq_safe = backend.where(valid_mask, r2_sq, backend.zeros_like(r2_sq))

        r1 = backend.sqrt(r1_sq_safe)
        r2 = backend.sqrt(r2_sq_safe)

        # mu = r2 / r1 for valid points
        mu_all = r2 / r1

        # Extract only valid values using argsort trick
        # Get indices where valid_mask is True
        # Backend doesn't have boolean indexing, so we use a different approach
        # Sort by validity (invalid first = 0, valid second = 1), then take last N
        valid_float = backend.astype(valid_mask, r1_sq.dtype)
        sort_keys = valid_float * 1e10 + mu_all  # Valid entries have large keys
        sorted_mu = backend.sort(sort_keys)

        # Take the last valid_count entries (the valid ones)
        n = dist_sq.shape[0]
        if valid_count == n:
            mu = mu_all
        else:
            # Use the sorted approach - valid entries are at the end
            mu = sorted_mu[n - valid_count:]

        return mu

    def _compute_from_mu(self, mu: "Array", use_regression: bool) -> float:
        backend = self._backend
        N = mu.shape[0]
        if N < 3:
            raise EstimatorError("two_nn", f"Insufficient non-degenerate samples: {N} < 3", N)

        # log(mu)
        log_mu = backend.log(mu)

        if not use_regression:
            # MLE form: d = 1 / mean(log(mu))
            mean_log_mu_arr = backend.mean(log_mu)
            backend.eval(mean_log_mu_arr)
            mean_log_mu = float(backend.to_scalar(mean_log_mu_arr))
            # Use machine epsilon for precision-aware threshold
            eps = machine_epsilon(backend, log_mu)
            if mean_log_mu < eps:
                raise EstimatorError.regression_degenerate()
            return 1.0 / mean_log_mu

        # Regression variant (Facco et al.)
        sorted_log_mu = backend.sort(log_mu)

        # indices 1..N
        i = backend.arange(1, N + 1)
        F = backend.astype(i, sorted_log_mu.dtype) / N

        # Slice to N-1
        x = sorted_log_mu[:-1]
        F_sliced = F[:-1]
        one_minus_F = 1.0 - F_sliced

        # Clamp to avoid log(0) - use machine epsilon
        eps = machine_epsilon(backend, one_minus_F)
        min_val = backend.full(one_minus_F.shape, eps)
        clamped = backend.maximum(min_val, one_minus_F)
        y = -backend.log(clamped)

        sum_xx = backend.sum(x * x)
        sum_xy = backend.sum(x * y)

        backend.eval(sum_xx, sum_xy)
        sum_xx_val = float(backend.to_scalar(sum_xx))
        sum_xy_val = float(backend.to_scalar(sum_xy))

        # Use machine epsilon for degenerate check
        if sum_xx_val < eps:
            raise EstimatorError.regression_degenerate()

        d = sum_xy_val / sum_xx_val
        return d

    def _bootstrap_two_nn(
        self,
        mu: "Array",
        use_regression: bool,
        config: BootstrapConfiguration,
    ) -> ConfidenceInterval | None:
        """Compute bootstrap confidence interval for the ID estimate."""
        backend = self._backend
        n = mu.shape[0]
        if n < 3:
            return None

        resamples = config.resamples
        if resamples <= 0:
            return None

        alpha = (1.0 - config.confidence_level) / 2.0

        # Use backend random with seed
        backend.random_seed(config.seed)

        dimensions: list[float] = []
        for _ in range(resamples):
            # Random indices with replacement
            indices = backend.random_randint(0, n, shape=(n,))
            sample = backend.take(mu, indices)

            try:
                d = self._compute_from_mu(sample, use_regression)
                dimensions.append(d)
            except EstimatorError:
                continue

        if len(dimensions) < 10:  # Require a minimum number of successful computations
            return None

        dimensions.sort()
        lower_idx = int(len(dimensions) * alpha)
        upper_idx = int(len(dimensions) * (1.0 - alpha))

        return ConfidenceInterval(
            level=config.confidence_level,
            lower=dimensions[lower_idx],
            upper=dimensions[upper_idx],
            resamples=resamples,
            seed=config.seed,
        )

    def local_dimension_map(
        self,
        points: "Array",
        k: int | None = None,
        deficiency_threshold: float | None = None,
    ) -> LocalDimensionMap:
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

        Note: This is more expensive than global ID (O(n^2) vs O(n)),
        but provides spatial resolution of dimension variation.

        Args:
            points: Point cloud [n, d]
            k: Number of neighbors for local estimation. If None, uses
                connectivity-based selection (minimum k for connected graph).
            deficiency_threshold: Optional threshold for deficiency detection.
                If provided, points with local ID < threshold * modal_dimension
                are flagged. Common values: 0.5 (50% of modal), 0.8 (80% of modal).
                If None, deficiency detection is skipped but dimensions are still computed.

        Returns:
            LocalDimensionMap with per-point dimensions and deficiency indices
        """
        # deficiency_threshold is optional - if not provided, we skip deficiency detection
        # but still compute per-point dimensions and statistics
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])

        # Compute geodesic distances (k=None triggers connectivity-based selection)
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        geo_dist = geo_result.distances
        k_actual = geo_result.k_neighbors
        backend.eval(geo_dist)

        # Need at least 3 neighbors for meaningful local ID
        k_local = max(3, min(k_actual, n - 1))

        # Get machine epsilon from the array's dtype - the geometry's precision limit
        eps = machine_epsilon(backend, geo_dist)

        # Sort distances per row to get k-nearest neighbors
        # Shape: [n, n] -> [n, n] sorted per row
        sorted_dists = backend.sort(geo_dist, axis=1)
        backend.eval(sorted_dists)

        # Skip self (column 0, distance 0), take k_local neighbors
        # k_dists[:, 0] is distance to self (0), k_dists[:, 1:k_local+1] are k nearest
        k_dists = sorted_dists[:, 1 : k_local + 1]  # [n, k_local]

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

        # Mean log(mu) per row - only where we have enough valid entries
        # Need at least 2 valid mu values for meaningful estimate
        min_valid = backend.array(2.0)
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
                deficiency_threshold=deficiency_threshold,
                k_neighbors=k_actual,
            )

        # Compute statistics using backend operations
        zeros = backend.zeros_like(local_dims)
        local_dims_safe = backend.where(valid_mask, local_dims, zeros)
        mean_dim_arr = backend.sum(local_dims_safe) / valid_count
        backend.eval(mean_dim_arr)
        mean_dim = backend.to_scalar(mean_dim_arr)
        diff = backend.where(valid_mask, local_dims_safe - mean_dim_arr, zeros)
        var_dim_arr = backend.sum(diff * diff) / valid_count
        std_dim_arr = backend.sqrt(var_dim_arr)
        backend.eval(std_dim_arr)
        std_dim = backend.to_scalar(std_dim_arr)

        # Modal dimension: bin dimensions and find most common
        # Use histogram with bins of width 0.5
        pos_inf = backend.array(float("inf"), dtype=local_dims.dtype)
        neg_inf = backend.array(float("-inf"), dtype=local_dims.dtype)
        min_dim_arr = backend.min(backend.where(valid_mask, local_dims, pos_inf))
        max_dim_arr = backend.max(backend.where(valid_mask, local_dims, neg_inf))
        backend.eval(min_dim_arr, max_dim_arr)
        min_dim = backend.to_scalar(min_dim_arr)
        max_dim = backend.to_scalar(max_dim_arr)

        if valid_count_scalar > 1.0 and max_dim > min_dim:
            n_bins = max(1, int((max_dim - min_dim) / 0.5) + 1)
            bin_width_arr = (max_dim_arr - min_dim_arr + eps) / backend.array(
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
            backend.eval(bin_indices)

            # Count bins using backend one-hot encoding (fully vectorized)
            # Create one-hot via eye indexing: one_hot[i] = eye[bin_idx[i]]
            eye_mat = backend.eye(n_bins)
            one_hot = eye_mat[bin_indices]  # [n_points, n_bins] with 1 at bin index
            backend.eval(one_hot)
            # Sum columns to get bin counts
            valid_mask_col = backend.reshape(valid_mask_float, (-1, 1))
            bin_counts_arr = backend.sum(one_hot * valid_mask_col, axis=0)
            backend.eval(bin_counts_arr)

            # Find modal bin using backend argmax
            max_bin_arr = backend.argmax(bin_counts_arr)
            backend.eval(max_bin_arr)
            max_bin = int(backend.to_scalar(max_bin_arr))

            backend.eval(bin_width_arr)
            bin_width_val = float(backend.to_scalar(bin_width_arr))
            modal_dim = min_dim + (max_bin + 0.5) * bin_width_val
        else:
            idx_range = backend.arange(0, n)
            idx_masked = backend.where(valid_mask, idx_range, backend.array(n, dtype="int32"))
            first_idx_arr = backend.min(idx_masked)
            backend.eval(first_idx_arr)
            first_idx = int(backend.to_scalar(first_idx_arr))
            modal_dim_arr = local_dims[first_idx : first_idx + 1]
            backend.eval(modal_dim_arr)
            modal_dim = backend.to_scalar(modal_dim_arr)

        # Find deficient points (only if threshold provided)
        deficient: list[int] = []
        if deficiency_threshold is not None:
            threshold = deficiency_threshold * modal_dim
            # Use backend operations to find deficient points
            deficient_mask = valid_id & (local_dims < backend.array(threshold))
            mask_int = backend.astype(deficient_mask, "int32")
            count_arr = backend.sum(mask_int)
            backend.eval(mask_int, count_arr)
            count = int(backend.to_scalar(count_arr))
            if count > 0:
                kth = max(0, n - count)
                partitioned = backend.argpartition(mask_int, kth)
                selected = backend.take(
                    partitioned, backend.arange(kth, n), axis=0
                )
                backend.eval(selected)
                deficient = sorted(int(x) for x in backend.tolist(selected))

        return LocalDimensionMap(
            dimensions=local_dims,
            modal_dimension=modal_dim,
            mean_dimension=mean_dim,
            std_dimension=std_dim,
            deficient_indices=deficient,
            deficiency_threshold=deficiency_threshold or 0.0,
            k_neighbors=k_actual,
        )

    @staticmethod
    def detect_dimension_deficiency(
        points: "Array",
        threshold: float,
        k: int | None = None,
        backend: "Backend | None" = None,
    ) -> list[int]:
        """
        Find points where local intrinsic dimension is deficient.

        Convenience method that returns just the indices of points where
        local ID < threshold * modal_dimension.

        These points indicate "dimension-collapsed" regions where the
        manifold is locally lower-dimensional than expected.

        Args:
            points: Point cloud [n, d]
            threshold: Deficiency threshold. This is a ratio (0.0-1.0) that
                determines when local ID is considered deficient relative to
                modal dimension. Must be explicitly provided.
            k: Number of neighbors for local estimation. If None, uses
                connectivity-based selection (minimum k for connected graph).
            backend: Backend to use

        Returns:
            List of point indices with deficient local dimension
        """
        b = backend or get_default_backend()
        estimator = IntrinsicDimension(b)
        result = estimator.local_dimension_map(points, k=k, deficiency_threshold=threshold)
        return result.deficient_indices
