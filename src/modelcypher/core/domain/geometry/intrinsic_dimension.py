
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import math
import mlx.core as mx

class EstimatorError(Exception):
    pass

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
    
    @staticmethod
    def estimate_two_nn(
        points: mx.array, 
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
        bootstrap: Optional[BootstrapConfiguration] = None
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
            
        mu = IntrinsicDimensionEstimator._compute_two_nn_mu(points)
        
        estimate = IntrinsicDimensionEstimator._estimate_from_mu(
            mu, 
            use_regression=configuration.use_regression
        )

        ci = None
        if bootstrap:
            ci = IntrinsicDimensionEstimator._bootstrap_two_nn(
                mu, 
                use_regression=configuration.use_regression,
                config=bootstrap
            )
        
        return TwoNNEstimate(
            intrinsic_dimension=estimate,
            sample_count=N,
            usable_count=mu.shape[0],
            uses_regression=configuration.use_regression,
            ci=ci
        )
        
    @staticmethod
    def _squared_euclidean_distance_matrix(points: mx.array) -> mx.array:
        """Computes pairwise squared euclidean distances efficiently."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        # points: [N, D]
        dots = points @ points.T # [N, N]
        norms = mx.sum(points * points, axis=1) # [N]
        # broadcasing norms: [N, 1] + [1, N]
        dist_sq = norms[:, None] + norms[None, :] - 2 * dots
        return mx.abs(dist_sq) # ensure non-negative due to float errors

    @staticmethod
    def _compute_two_nn_mu(points: mx.array) -> mx.array:
        """
        Computes the ratio mu = r2 / r1 for each point.
        """
        # 1. Compute pairwise distances
        dist_sq = IntrinsicDimensionEstimator._squared_euclidean_distance_matrix(points)
        
        # 2. Find nearest neighbors (excluding self at index 0 in sorted order)
        # We need 1st and 2nd nearest neighbors (indices 1 and 2 in sorted list of distances)
        # Using sort since partition isn't fully exposed/stable in all backends yet for small k, or topk (which gives max) 
        # For distances we want smallest. sort is O(N log N) but usually fine for typical N < 10k in this context.
        # Alternatively, use -dist_sq with topk(k=3).
        
        # NOTE: Using full sort for robustness if N is small, or topk if large.
        # Given 'points' implies manifold samples, N is often hundreds to thousands.
        # dist_sq is [N, N]. topk is fast.
        
        sorted_dist_sq = mx.sort(dist_sq, axis=1)
        
        # index 0 is self (dist 0), index 1 is NN1, index 2 is NN2
        r1_sq = sorted_dist_sq[:, 1]
        r2_sq = sorted_dist_sq[:, 2]
        
        # Filter degenerate points
        # r1 > 0 to avoid division by zero (duplicates)
        mask = r1_sq > 1e-9
        
        # MLX boolean indexing workaround
        import numpy as np
        r1_np = np.array(r1_sq)
        r2_np = np.array(r2_sq)
        mask_np = np.array(mask)
        
        r1_valid = r1_np[mask_np]
        r2_valid = r2_np[mask_np]
        
        # Back to MLX
        r1 = mx.array(r1_valid)
        r2 = mx.array(r2_valid)
        
        if r1.size == 0:
             return mx.array([])
             
        r1 = mx.sqrt(r1)
        r2 = mx.sqrt(r2)
        
        mu = r2 / r1
        
        # mu must be >= 1.0 (by definition r2 >= r1). Float errors might make it 0.999...
        # We simply filter valid ones.
        return mu

    @staticmethod
    def _estimate_from_mu(mu: mx.array, use_regression: bool) -> float:
        N = mu.shape[0]
        if N < 3:
             raise EstimatorError(f"Insufficient non-degenerate samples: {N} < 3")
             
        # log(mu)
        log_mu = mx.log(mu)
        
        if not use_regression:
            # MLE form: d = 1 / mean(log(mu))
            mean_log_mu = mx.mean(log_mu).item()
            if mean_log_mu < 1e-9:
                 raise EstimatorError("Regression degenerate: mean(log(mu)) ~ 0")
            return 1.0 / mean_log_mu
            
        # Regression variant (Facco et al.)
        sorted_log_mu = mx.sort(log_mu)
        
        # indices 1..N
        i = mx.arange(1, N + 1, dtype=mx.float32)
        F = i / N
        
        # Slice to N-1
        x = sorted_log_mu[:-1]
        F_sliced = F[:-1]
        y = -mx.log(mx.maximum(mx.array(1e-12), 1.0 - F_sliced))
        
        sum_xx = mx.sum(x * x).item()
        sum_xy = mx.sum(x * y).item()
        
        if sum_xx < 1e-9:
             raise EstimatorError("Regression degenerate: sum(xx) ~ 0")
        
        d = sum_xy / sum_xx
        return d

    @staticmethod
    def _bootstrap_two_nn(
        mu: mx.array,
        use_regression: bool,
        config: BootstrapConfiguration
    ) -> Optional[ConfidenceInterval]:
        """Compute bootstrap confidence interval for the ID estimate."""
        import numpy as np
        
        n = mu.shape[0]
        if n < 3:
            return None
            
        resamples = config.resamples
        if resamples <= 0:
            return None
            
        alpha = (1.0 - config.confidence_level) / 2.0
        
        # MLX doesn't have choice replacement yet, use numpy for sampling
        np.random.seed(config.seed)
        mu_np = np.array(mu)
        
        estimates = []
        for _ in range(resamples):
            indices = np.random.choice(n, size=n, replace=True)
            sample = mx.array(mu_np[indices])
            
            try:
                d = IntrinsicDimensionEstimator._estimate_from_mu(sample, use_regression)
                estimates.append(d)
            except EstimatorError:
                continue
                
        if len(estimates) < 10: # Require a minimum number of successful estimates
            return None
            
        estimates.sort()
        lower_idx = int(len(estimates) * alpha)
        upper_idx = int(len(estimates) * (1.0 - alpha))
        
        return ConfidenceInterval(
            level=config.confidence_level,
            lower=estimates[lower_idx],
            upper=estimates[upper_idx],
            resamples=resamples,
            seed=config.seed
        )
