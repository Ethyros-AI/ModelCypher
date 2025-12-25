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
    the correct metric; Euclidean is the approximation that ignores curvature.
    Geodesic distances are estimated via k-NN graph shortest paths (Isomap-style).
    """

    enabled: bool = True
    k_neighbors: int = 10
    distance_power: float = 2.0


@dataclass
class TwoNNConfiguration:
    """Two-nearest-neighbors estimator configuration.

    Attributes:
        use_regression: Use regression variant (Facco et al.) vs MLE.
        geodesic: Geodesic distance configuration. Uses geodesic distances by
            default since curvature is inherent in high-dimensional spaces.
            Set geodesic.enabled=False only for low-dimensional data.
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


class IntrinsicDimensionEstimator:
    """
    Estimates intrinsic dimension using the TwoNN method (Facco et al., 2017).

    The reference implementation uses intrinsic dimension (ID) as a geometry-first quality metric:
    - Low ID: tight, consistent behavior (risk: caricature/mode collapse)
    - High ID: multi-modal/prompt-dependent behavior (risk: incoherence)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def estimate_two_nn(
        self,
        points: "Array",
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
        bootstrap: BootstrapConfiguration | None = None,
    ) -> TwoNNEstimate:
        """
        Estimates intrinsic dimension.

        Args:
            points: [N, D] array of points
            configuration: Estimation config
            bootstrap: Optional bootstrap configuration for confidence intervals

        Note:
            When geodesic.enabled=True, uses manifold-aware distances via k-NN graph
            shortest paths. This corrects for curvature bias in the TwoNN estimator:
            - Positive curvature: Euclidean underestimates distances → inflated ID
            - Negative curvature: Euclidean overestimates distances → deflated ID
        """
        N = points.shape[0]
        if N < 3:
            raise EstimatorError(f"Insufficient samples: {N} < 3")

        # Compute distance matrix (Euclidean or geodesic)
        uses_geodesic = configuration.geodesic.enabled
        if uses_geodesic:
            dist_sq = self._geodesic_distance_matrix_squared(
                points,
                k_neighbors=configuration.geodesic.k_neighbors,
                distance_power=configuration.geodesic.distance_power,
            )
        else:
            dist_sq = self._squared_euclidean_distance_matrix(points)

        mu = self._compute_two_nn_mu_from_distances(dist_sq)

        estimate = self._estimate_from_mu(mu, use_regression=configuration.use_regression)

        ci = None
        if bootstrap:
            ci = self._bootstrap_two_nn(
                mu,
                use_regression=configuration.use_regression,
                config=bootstrap,
            )

        return TwoNNEstimate(
            intrinsic_dimension=estimate,
            sample_count=N,
            usable_count=mu.shape[0],
            uses_regression=configuration.use_regression,
            uses_geodesic=uses_geodesic,
            ci=ci,
        )

    def _squared_euclidean_distance_matrix(self, points: "Array") -> "Array":
        """Computes pairwise squared euclidean distances efficiently."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        # points: [N, D]
        dots = self._backend.matmul(points, self._backend.transpose(points))  # [N, N]
        norms = self._backend.sum(points * points, axis=1)  # [N]
        # broadcasting norms: [N, 1] + [1, N]
        dist_sq = norms[:, None] + norms[None, :] - 2 * dots
        return self._backend.abs(dist_sq)  # ensure non-negative due to float errors

    def _geodesic_distance_matrix_squared(
        self,
        points: "Array",
        k_neighbors: int = 10,
        distance_power: float = 2.0,
    ) -> "Array":
        """Computes pairwise squared geodesic distances via k-NN graph.

        Uses the Isomap-style approach:
        1. Build k-nearest-neighbor graph with Euclidean edge weights
        2. Compute shortest paths (approximates geodesics on manifold)

        This corrects for curvature:
        - On positively curved manifolds, Euclidean < geodesic
        - On negatively curved manifolds, Euclidean > geodesic

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
        geodesic_dist = riemannian.geodesic_distance_matrix(
            points, k_neighbors=k_neighbors
        )

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

    def _estimate_from_mu(self, mu: "Array", use_regression: bool) -> float:
        backend = self._backend
        N = mu.shape[0]
        if N < 3:
            raise EstimatorError(f"Insufficient non-degenerate samples: {N} < 3")

        # log(mu)
        log_mu = backend.log(mu)

        if not use_regression:
            # MLE form: d = 1 / mean(log(mu))
            mean_log_mu_arr = backend.mean(log_mu)
            backend.eval(mean_log_mu_arr)
            mean_log_mu = float(backend.to_numpy(mean_log_mu_arr))
            if mean_log_mu < 1e-9:
                raise EstimatorError("Regression degenerate: mean(log(mu)) ~ 0")
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
            raise EstimatorError("Regression degenerate: sum(xx) ~ 0")

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

        estimates: list[float] = []
        for _ in range(resamples):
            # Random indices with replacement
            indices = backend.random_randint(0, n, shape=(n,))
            sample = backend.take(mu, indices)

            try:
                d = self._estimate_from_mu(sample, use_regression)
                estimates.append(d)
            except EstimatorError:
                continue

        if len(estimates) < 10:  # Require a minimum number of successful estimates
            return None

        estimates.sort()
        lower_idx = int(len(estimates) * alpha)
        upper_idx = int(len(estimates) * (1.0 - alpha))

        return ConfidenceInterval(
            level=config.confidence_level,
            lower=estimates[lower_idx],
            upper=estimates[upper_idx],
            resamples=resamples,
            seed=config.seed,
        )
