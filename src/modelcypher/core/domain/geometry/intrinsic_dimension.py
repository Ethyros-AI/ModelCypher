from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.ports.backend import Array, Backend

@dataclass
class TwoNNConfiguration:
    """Two-nearest-neighbors estimator configuration."""
    use_regression: bool = True
    
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
    ci: Optional[ConfidenceInterval] = None

class IntrinsicDimensionEstimator:
    """
    Estimates intrinsic dimension using the TwoNN method (Facco et al., 2017).

    The reference implementation uses intrinsic dimension (ID) as a geometry-first quality metric:
    - Low ID: tight, consistent behavior (risk: caricature/mode collapse)
    - High ID: multi-modal/prompt-dependent behavior (risk: incoherence)
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()

    def estimate_two_nn(
        self,
        points: Array,
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
        bootstrap: Optional[BootstrapConfiguration] = None,
    ) -> TwoNNEstimate:
        """
        Estimates intrinsic dimension.

        Args:
            points: [N, D] array of points
            configuration: Estimation config
            bootstrap: Optional bootstrap configuration for confidence intervals
        """
        N = points.shape[0]
        if N < 3:
            raise EstimatorError(f"Insufficient samples: {N} < 3")

        mu = self._compute_two_nn_mu(points)

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
            ci=ci,
        )

    def _squared_euclidean_distance_matrix(self, points: Array) -> Array:
        """Computes pairwise squared euclidean distances efficiently."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        # points: [N, D]
        dots = self._backend.matmul(points, self._backend.transpose(points))  # [N, N]
        norms = self._backend.sum(points * points, axis=1)  # [N]
        # broadcasting norms: [N, 1] + [1, N]
        dist_sq = norms[:, None] + norms[None, :] - 2 * dots
        return self._backend.abs(dist_sq)  # ensure non-negative due to float errors

    def _compute_two_nn_mu(self, points: Array) -> Array:
        """
        Computes the ratio mu = r2 / r1 for each point.
        """
        # 1. Compute pairwise distances
        dist_sq = self._squared_euclidean_distance_matrix(points)

        # 2. Find nearest neighbors (excluding self at index 0 in sorted order)
        sorted_dist_sq = self._backend.sort(dist_sq, axis=1)

        # index 0 is self (dist 0), index 1 is NN1, index 2 is NN2
        r1_sq = sorted_dist_sq[:, 1]
        r2_sq = sorted_dist_sq[:, 2]

        # Filter degenerate points (r1 > 0 to avoid division by zero)
        r1_np = self._backend.to_numpy(r1_sq)
        r2_np = self._backend.to_numpy(r2_sq)
        mask_np = r1_np > 1e-9

        r1_valid = r1_np[mask_np]
        r2_valid = r2_np[mask_np]

        if r1_valid.size == 0:
            return self._backend.array([])

        r1 = np.sqrt(r1_valid)
        r2 = np.sqrt(r2_valid)

        mu = r2 / r1

        return self._backend.array(mu)

    def _estimate_from_mu(self, mu: Array, use_regression: bool) -> float:
        N = mu.shape[0]
        if N < 3:
            raise EstimatorError(f"Insufficient non-degenerate samples: {N} < 3")

        # log(mu)
        log_mu = self._backend.log(mu)

        if not use_regression:
            # MLE form: d = 1 / mean(log(mu))
            mean_log_mu = float(self._backend.to_numpy(self._backend.mean(log_mu)))
            if mean_log_mu < 1e-9:
                raise EstimatorError("Regression degenerate: mean(log(mu)) ~ 0")
            return 1.0 / mean_log_mu

        # Regression variant (Facco et al.)
        sorted_log_mu = self._backend.sort(log_mu)

        # indices 1..N
        i = self._backend.arange(1, N + 1)
        F = i / N

        # Slice to N-1
        x = sorted_log_mu[:-1]
        F_sliced = F[:-1]
        one_minus_F = 1.0 - F_sliced
        clamped = self._backend.maximum(self._backend.array([1e-12]), one_minus_F)
        y = -self._backend.log(clamped)

        sum_xx = float(self._backend.to_numpy(self._backend.sum(x * x)))
        sum_xy = float(self._backend.to_numpy(self._backend.sum(x * y)))

        if sum_xx < 1e-9:
            raise EstimatorError("Regression degenerate: sum(xx) ~ 0")

        d = sum_xy / sum_xx
        return d

    def _bootstrap_two_nn(
        self,
        mu: Array,
        use_regression: bool,
        config: BootstrapConfiguration,
    ) -> Optional[ConfidenceInterval]:
        """Compute bootstrap confidence interval for the ID estimate."""
        n = mu.shape[0]
        if n < 3:
            return None

        resamples = config.resamples
        if resamples <= 0:
            return None

        alpha = (1.0 - config.confidence_level) / 2.0

        # Use numpy for sampling
        np.random.seed(config.seed)
        mu_np = self._backend.to_numpy(mu)

        estimates = []
        for _ in range(resamples):
            indices = np.random.choice(n, size=n, replace=True)
            sample = self._backend.array(mu_np[indices])

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
