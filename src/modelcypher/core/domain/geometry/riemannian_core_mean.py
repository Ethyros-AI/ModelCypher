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

"""Fréchet mean utilities for RiemannianGeometry."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_inf,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_types import (
    FrechetMeanResult,
    GeodesicDistanceResult,
)
from modelcypher.core.domain.geometry.riemannian_validation import (
    count_nan,
    count_nonfinite,
)

from .riemannian_core_utils import _count_mask, _count_not_mask, _promote_precision

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)
_cache = ComputationCache.shared()


class RiemannianMeanMixin:
    """Fréchet mean computation for RiemannianGeometry."""

    def frechet_mean(
        self,
        points: "Array",
        weights: "Array | None" = None,
        max_iterations: int | None = None,
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
            max_iterations: Maximum gradient descent iterations (derived from n if None)
            tolerance: Convergence threshold for mean position change
            k_neighbors: Optional fixed k for geodesic graph connectivity
            max_k_neighbors: Optional upper bound for adaptive k retries

        Returns:
            FrechetMeanResult with the computed mean
        """
        backend = self._backend
        points = _promote_precision(backend.array(points), backend)
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

        # Derive max iterations from problem size if not specified
        # Uses logarithmic scaling: max_iter = max(50, 10 * ceil(log2(n + 1)))
        if max_iterations is None:
            max_iterations = max(50, 10 * int(math.ceil(math.log2(max(2, n) + 1))))

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
                    from modelcypher.core.domain.geometry.riemannian_utils import (
                        geodesic_norms,
                    )

                    step_norm_arr = geodesic_norms(
                        backend.reshape(step_vec, (1, -1)), backend, use_cache=False
                    )
                    backend.eval(step_norm_arr)
                    step_val = float(backend.to_scalar(step_norm_arr[0]))

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
        max_iterations: int | None = None,
        tolerance: float | None = None,
        k_neighbors: int | None = None,
        max_k_neighbors: int | None = None,
    ) -> "Array":
        """Compute Fréchet means for a batch of point sets.

        Args:
            points_batch: [B, n, d] batch of point clouds
            weights_batch: Optional [B, n] weights per batch
            max_iterations: Maximum iterations per mean (derived from n if None)
        Returns:
            [B, d] array of Fréchet means
        """
        backend = self._backend
        points_batch = _promote_precision(backend.array(points_batch), backend)
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
        mu_nonfinite_arr = _count_not_mask(mu_isfinite, backend, dtype_source=mu)
        geo_nonfinite_arr = _count_not_mask(geo_isfinite, backend, dtype_source=geo_from_mu)
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
        inf_count_arr = _count_mask(scale_isinf, backend, dtype_source=scale)
        nan_count_arr = _count_mask(scale_isnan, backend, dtype_source=scale)
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
        grad_nonfinite_arr = _count_not_mask(grad_isfinite, backend, dtype_source=gradient)
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
        from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

        grad_norm_arr = geodesic_norms(
            backend.reshape(gradient, (1, -1)), backend, use_cache=False
        )
        backend.eval(grad_norm_arr)
        grad_norm = float(backend.to_scalar(grad_norm_arr[0]))

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
        new_mu_nonfinite_arr = _count_not_mask(new_mu_isfinite, backend, dtype_source=new_mu)
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
