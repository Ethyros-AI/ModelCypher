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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class GeodesicConfiguration:
    """Configuration for geodesic distance estimation.

    In high-dimensional spaces, curvature is inherent. Geodesic distance is
    the correct metric. Geodesic distances are estimated via k-NN graph
    shortest paths (Isomap-style).
    """

    k_neighbors: int = 10
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

    # Backward compatibility alias
    estimate_two_nn = compute_two_nn

    def compute(
        self,
        points: "Array",
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
        bootstrap: BootstrapConfiguration | None = None,
    ) -> TwoNNEstimate:
        """
        Compute intrinsic dimension using geodesic distances.

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

        # Compute geodesic distance matrix (curvature is inherent in HD space)
        dist_sq = self._geodesic_distance_matrix_squared(
            points,
            k_neighbors=configuration.geodesic.k_neighbors,
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
        k_neighbors: int = 10,
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
            k_neighbors: Number of neighbors for graph construction
            distance_power: Power for distance weighting (2.0 = squared distances)

        Returns:
            [N, N] squared geodesic distance matrix
        """
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        riemannian = RiemannianGeometry(backend=self._backend)

        # Get geodesic distances (not squared)
        result = riemannian.geodesic_distances(points, k_neighbors=k_neighbors)
        geodesic_dist = result.distances

        # Return squared distances for consistency with Euclidean version
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
        # Use a mask-based approach on GPU
        threshold = 1e-9
        valid_mask = r1_sq > threshold

        # Count valid points
        backend.eval(valid_mask)
        valid_count = int(backend.to_numpy(backend.sum(backend.astype(valid_mask, r1_sq.dtype))))

        if valid_count == 0:
            return backend.array([])

        # Use where to zero out invalid entries, then filter
        # For simplicity, convert to numpy for filtering then back
        # This is a minor numpy usage for filtering - use backend operations instead
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
            mean_log_mu = float(backend.to_numpy(mean_log_mu_arr))
            if mean_log_mu < 1e-9:
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

        # Clamp to avoid log(0)
        min_val = backend.full(one_minus_F.shape, 1e-12)
        clamped = backend.maximum(min_val, one_minus_F)
        y = -backend.log(clamped)

        sum_xx = backend.sum(x * x)
        sum_xy = backend.sum(x * y)

        backend.eval(sum_xx, sum_xy)
        sum_xx_val = float(backend.to_numpy(sum_xx))
        sum_xy_val = float(backend.to_numpy(sum_xy))

        if sum_xx_val < 1e-9:
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
        k: int = 10,
        deficiency_threshold: float = 0.8,
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
            k: Number of neighbors for local estimation (must be >= 3)
            deficiency_threshold: Threshold for deficiency detection (default 0.8)
                Points with local ID < threshold * modal_dimension are flagged

        Returns:
            LocalDimensionMap with per-point dimensions and deficiency indices
        """
        backend = self._backend
        points = backend.array(points)
        backend.eval(points)

        n = int(points.shape[0])
        k = max(3, min(k, n - 1))  # Need at least 3 neighbors for TwoNN

        # Compute geodesic distances once
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)
        geo_np = backend.to_numpy(geo_dist)

        # Compute local ID for each point
        local_dims: list[float] = []
        import math

        for i in range(n):
            # Get distances from point i to all others
            dists = geo_np[i, :].tolist()

            # Sort to get k+1 nearest (including self at distance 0)
            sorted_pairs = sorted(enumerate(dists), key=lambda x: (x[1], x[0]))

            # Skip self (index 0), take neighbors 1 to k
            # We need at least 2 neighbors for TwoNN (r1 and r2)
            neighbors_with_dist = [
                (idx, d) for idx, d in sorted_pairs
                if idx != i and not math.isinf(d)
            ][:k]

            if len(neighbors_with_dist) < 2:
                # Can't compute local ID - mark as NaN or use global
                local_dims.append(float("nan"))
                continue

            # Compute mu values for local neighborhood
            # For TwoNN, we need the ratio r2/r1 for multiple point pairs
            # Simplest approach: use the first few neighbor distance ratios
            mu_vals: list[float] = []
            for j in range(1, len(neighbors_with_dist)):
                r1 = neighbors_with_dist[j - 1][1]  # Distance to (j-1)-th neighbor
                r2 = neighbors_with_dist[j][1]  # Distance to j-th neighbor
                if r1 > 1e-12:
                    mu_vals.append(r2 / r1)

            if len(mu_vals) < 2:
                local_dims.append(float("nan"))
                continue

            # Local ID estimate: MLE form d = 1 / mean(log(mu))
            log_mu_vals = [math.log(mu) for mu in mu_vals if mu > 0]
            if len(log_mu_vals) == 0 or sum(log_mu_vals) < 1e-12:
                local_dims.append(float("nan"))
                continue

            mean_log_mu = sum(log_mu_vals) / len(log_mu_vals)
            if mean_log_mu < 1e-12:
                local_dims.append(float("nan"))
                continue

            local_id = 1.0 / mean_log_mu
            local_dims.append(local_id)

        # Convert to backend array
        # Replace nan with 0 for statistics, but keep track for filtering
        valid_dims = [d for d in local_dims if not math.isnan(d) and d > 0]

        if len(valid_dims) == 0:
            return LocalDimensionMap(
                dimensions=backend.array(local_dims),
                modal_dimension=0.0,
                mean_dimension=0.0,
                std_dimension=0.0,
                deficient_indices=[],
                deficiency_threshold=deficiency_threshold,
                k_neighbors=k,
            )

        # Compute statistics
        mean_dim = sum(valid_dims) / len(valid_dims)
        var_dim = sum((d - mean_dim) ** 2 for d in valid_dims) / len(valid_dims)
        std_dim = math.sqrt(var_dim)

        # Modal dimension: bin dimensions and find most common
        # Use histogram with bins of width 0.5
        if len(valid_dims) > 1:
            min_dim = min(valid_dims)
            max_dim = max(valid_dims)
            n_bins = max(1, int((max_dim - min_dim) / 0.5) + 1)
            bin_width = (max_dim - min_dim + 1e-9) / n_bins

            bin_counts: list[int] = [0] * n_bins
            for d in valid_dims:
                bin_idx = min(n_bins - 1, int((d - min_dim) / bin_width))
                bin_counts[bin_idx] += 1

            max_bin = 0
            max_count = bin_counts[0]
            for i, c in enumerate(bin_counts):
                if c > max_count:
                    max_count = c
                    max_bin = i

            modal_dim = min_dim + (max_bin + 0.5) * bin_width
        else:
            modal_dim = valid_dims[0]

        # Find deficient points
        threshold = deficiency_threshold * modal_dim
        deficient: list[int] = []
        for i, d in enumerate(local_dims):
            if not math.isnan(d) and d < threshold:
                deficient.append(i)

        return LocalDimensionMap(
            dimensions=backend.array(local_dims),
            modal_dimension=modal_dim,
            mean_dimension=mean_dim,
            std_dimension=std_dim,
            deficient_indices=deficient,
            deficiency_threshold=deficiency_threshold,
            k_neighbors=k,
        )

    @staticmethod
    def detect_dimension_deficiency(
        points: "Array",
        threshold: float = 0.8,
        k: int = 10,
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
            threshold: Deficiency threshold (default 0.8)
            k: Number of neighbors for local estimation
            backend: Backend to use

        Returns:
            List of point indices with deficient local dimension
        """
        b = backend or get_default_backend()
        estimator = IntrinsicDimension(b)
        result = estimator.local_dimension_map(points, k=k, deficiency_threshold=threshold)
        return result.deficient_indices


# Backward compatibility alias - IntrinsicDimension is the correct name
# (this is a measurement, not an estimate)
IntrinsicDimensionEstimator = IntrinsicDimension
