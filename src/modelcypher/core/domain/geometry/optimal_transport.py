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

"""Optimal transport solvers for representation alignment.

Provides a GPU-accelerated Sinkhorn algorithm for entropy-regularized optimal
transport plans.

References:
    - Cuturi (2013) "Sinkhorn Distances" NeurIPS
    - Schmitzer (2019) "Stabilized Sparse Scaling Algorithms" SIAM J. Sci. Comput.
    - Peyré & Cuturi (2019) "Computational Optimal Transport"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    is_finite,
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
    safe_log_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_cosine_between_sets,
    geodesic_norms,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class SinkhornResult:
    """Result from Sinkhorn optimal transport solver."""

    plan: "Array"
    converged: bool
    iterations: int
    marginal_error: float
    cost: float


class SinkhornSolver:
    """GPU-accelerated Sinkhorn optimal transport solver.

    Computes entropy-regularized optimal transport plans between probability
    distributions. All parameters are derived from the data - no configuration.

    Example:
        >>> solver = SinkhornSolver(backend)
        >>> result = solver.solve(cost_matrix)
        >>> transport_plan = result.plan
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    @property
    def backend(self) -> "Backend":
        return self._backend

    def solve(
        self,
        cost_matrix: "Array",
        source_marginal: "Array | None" = None,
        target_marginal: "Array | None" = None,
    ) -> SinkhornResult:
        """Solve optimal transport between source and target distributions.

        All parameters (epsilon, convergence threshold, etc.) are derived from
        the cost matrix dtype and scale. No configuration needed.

        Args:
            cost_matrix: Cost matrix [n, m] where cost[i,j] is cost of transporting
                mass from source[i] to target[j]
            source_marginal: Source distribution [n]. Defaults to uniform.
            target_marginal: Target distribution [m]. Defaults to uniform.

        Returns:
            SinkhornResult with transport plan, convergence info, and cost.
        """
        backend = self._backend

        # Convert inputs to backend arrays
        cost_matrix = backend.array(cost_matrix)
        if source_marginal is not None:
            source_marginal = backend.array(source_marginal)
        if target_marginal is not None:
            target_marginal = backend.array(target_marginal)

        # Derive all tolerances from data - no configuration
        epsilon = self._derive_epsilon(cost_matrix)
        convergence_threshold = regularization_epsilon(backend, cost_matrix)

        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])

        # Default to uniform marginals
        weights_dtype = precision_dtype(backend, reference=cost_matrix)
        mu = (
            source_marginal
            if source_marginal is not None
            else backend.ones((n,), dtype=weights_dtype) / float(n)
        )
        nu = (
            target_marginal
            if target_marginal is not None
            else backend.ones((m,), dtype=weights_dtype) / float(m)
        )
        backend.eval(mu, nu)

        # Always use log-domain for numerical stability
        return self._solve_log_domain(
            cost_matrix,
            mu,
            nu,
            epsilon,
            convergence_threshold,
        )

    def _derive_epsilon(self, cost: "Array") -> float:
        """Derive Sinkhorn regularization epsilon from cost matrix scale.

        Sinkhorn convergence requires epsilon proportional to cost scale
        (Peyre & Cuturi 2019, Sec. 4.2). Using median(cost) * sqrt(machine_epsilon)
        keeps regularization error below float32 resolution.
        """
        backend = self._backend
        median_val = compute_median(cost, backend)
        if median_val == 0.0:
            return float(division_epsilon(backend, cost))

        eps = float(machine_epsilon(backend, cost))
        return max(median_val * (eps ** 0.5), eps)

    def _precision_update_floor(self, reference: "Array", *values: "Array") -> float:
        """Compute the smallest representable update scale for the given values."""
        backend = self._backend
        scale = backend.array([1.0])
        for value in values:
            value_abs = backend.abs(value)
            value_max = backend.max(value_abs)
            scale = backend.maximum(scale, value_max)
        backend.eval(scale)
        eps = machine_epsilon(backend, reference)
        return eps * float(self._to_scalar(scale))

    def solve_linear_ot(
        self,
        cost: "Array",
        p: "Array",
        q: "Array",
        epsilon: float,
        threshold: float | None = None,
        max_iterations: int | None = None,
    ) -> "Array":
        """Solve linear optimal transport in log-domain — fast version for inner loops.

        Log-domain Sinkhorn (Schmitzer 2019, Algorithm 1) is numerically stable
        for all epsilon > 0. No epsilon floor or iteration cap needed. Convergence
        is guaranteed by Banach fixed-point contraction on dual potentials
        (Franklin & Lorenz 1989).

        Args:
            cost: Cost matrix [n, m]
            p: Source marginal [n]
            q: Target marginal [m]
            epsilon: Entropy regularization strength
            threshold: Optional convergence threshold. When None, derives
                from dtype precision as sqrt(machine_epsilon).
            max_iterations: Optional iteration cap for callers that need
                deterministic upper bounds.

        Returns:
            Transport plan [n, m]

        References:
            Schmitzer (2019) "Stabilized Sparse Scaling Algorithms" SIAM J. Sci. Comput.
            Franklin & Lorenz (1989) "On the scaling of multidimensional matrices" LAA
        """
        backend = self._backend
        n = int(cost.shape[0])
        m = int(cost.shape[1])

        if n == 0 or m == 0:
            return backend.zeros((n, m))

        convergence_threshold = (
            float(threshold)
            if threshold is not None and threshold > 0.0
            else regularization_epsilon(backend, cost)
        )

        # Log-domain: work with dual potentials (f, g) instead of scaling
        # vectors (u, v). Transport plan: P_ij = exp(f_i + logK_ij + g_j).
        # logsumexp subtracts max before exp(), so exp() argument is always <= 0
        # and output is always in [0, 1]. No underflow for any epsilon > 0.
        log_eps = safe_log_epsilon(backend, p)
        log_p = backend.log(backend.maximum(p, backend.full(p.shape, log_eps)))
        log_q = backend.log(backend.maximum(q, backend.full(q.shape, log_eps)))

        # Row-wise centering improves logsumexp numerics (shift doesn't change solution)
        cost_min = backend.min(cost, axis=1, keepdims=True)
        cost_centered = cost - cost_min
        logK = -cost_centered / epsilon
        backend.eval(log_p, log_q, logK)

        f = backend.zeros((n,))
        g = backend.zeros((m,))
        logK_T = backend.transpose(logK)

        prev_error: float | None = None
        iterations = 0

        while True:
            iterations += 1
            # Sinkhorn update in log-domain (Schmitzer 2019, Eq. 3.1)
            logK_plus_g = logK + g.reshape((1, m))
            f_new = log_p - self._logsumexp(logK_plus_g, axis=1)

            logKT_plus_f = logK_T + f_new.reshape((1, n))
            g_new = log_q - self._logsumexp(logKT_plus_f, axis=1)

            # Check marginal error (row marginals suffice for convergence detection)
            logK_plus_g_new = logK + g_new.reshape((1, m))
            row_log_sum = self._logsumexp(logK_plus_g_new, axis=1)
            row_sums = backend.exp(f_new + row_log_sum)
            row_error = backend.max(backend.abs(row_sums - p))
            backend.eval(row_error)
            marginal_error = float(self._to_scalar(row_error))

            f = f_new
            g = g_new

            if not is_finite(marginal_error, backend):
                break
            if marginal_error <= convergence_threshold:
                break
            if max_iterations is not None and iterations >= max_iterations:
                break
            # Stall detection: if error stopped improving, we've reached precision limit
            if prev_error is not None:
                progress = abs(prev_error - marginal_error)
                eps = machine_epsilon(backend, cost)
                if progress <= eps * max(abs(prev_error), 1.0):
                    break
            prev_error = marginal_error

        # Recover transport plan: P = exp(f + logK + g)
        logP = f.reshape((n, 1)) + logK + g.reshape((1, m))
        plan = backend.exp(logP)
        backend.eval(plan)
        return plan

    def _solve_log_domain(
        self,
        cost_matrix: "Array",
        mu: "Array",
        nu: "Array",
        epsilon: float,
        convergence_threshold: float,
    ) -> SinkhornResult:
        """Log-domain Sinkhorn (Schmitzer 2019). Numerically stable for all epsilon > 0."""
        backend = self._backend
        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])

        # Use safe_log_epsilon for proper log domain clamping
        log_eps = safe_log_epsilon(backend, mu)
        log_mu = backend.log(
            backend.maximum(mu, backend.full(mu.shape, log_eps))
        )
        log_nu = backend.log(
            backend.maximum(nu, backend.full(nu.shape, log_eps))
        )
        cost_min = backend.min(cost_matrix, axis=1, keepdims=True)
        cost_centered = cost_matrix - cost_min
        logK = -cost_centered / epsilon
        backend.eval(log_mu, log_nu, logK)

        log_dtype = precision_dtype(backend, reference=cost_matrix)
        f = backend.zeros((n,), dtype=log_dtype)
        g = backend.zeros((m,), dtype=log_dtype)
        backend.eval(f, g)

        converged = False
        iterations = 0
        marginal_error = float("inf")
        prev_error: float | None = None
        prev_diff: float | None = None

        logK_T = backend.transpose(logK)
        while True:
            iterations += 1
            logK_plus_g = logK + g.reshape((1, m))
            f_new = log_mu - self._logsumexp(logK_plus_g, axis=1)
            logKT_plus_f = logK_T + f_new.reshape((1, n))
            col_log_sum = self._logsumexp(logKT_plus_f, axis=1)
            g_new = log_nu - col_log_sum

            f_diff = backend.max(backend.abs(f_new - f))
            g_diff = backend.max(backend.abs(g_new - g))
            backend.eval(f_diff, g_diff)
            max_diff = max(
                float(self._to_scalar(f_diff)), float(self._to_scalar(g_diff))
            )

            logK_plus_g_new = logK + g_new.reshape((1, m))
            row_log_sum = self._logsumexp(logK_plus_g_new, axis=1)
            row_sums = backend.exp(f_new + row_log_sum)
            col_sums = backend.exp(g_new + col_log_sum)
            row_error = backend.max(backend.abs(row_sums - mu))
            col_error = backend.max(backend.abs(col_sums - nu))
            backend.eval(row_error, col_error)
            marginal_error = max(
                float(self._to_scalar(row_error)), float(self._to_scalar(col_error))
            )

            update_floor = self._precision_update_floor(cost_matrix, f_new, g_new)
            effective_threshold = max(convergence_threshold, update_floor)

            f = f_new
            g = g_new
            if not is_finite(max_diff, backend) or not is_finite(marginal_error, backend):
                break
            if marginal_error <= effective_threshold and max_diff <= effective_threshold:
                converged = True
                break
            if prev_error is not None and prev_diff is not None:
                error_delta = abs(prev_error - marginal_error)
                diff_delta = abs(prev_diff - max_diff)
                progress_floor = machine_epsilon(backend, cost_matrix) * max(
                    abs(prev_error), abs(marginal_error), abs(prev_diff), abs(max_diff), 1.0
                )
                if error_delta <= progress_floor and diff_delta <= progress_floor and max_diff <= update_floor:
                    break
            prev_error = marginal_error
            prev_diff = max_diff

        logP = f.reshape((n, 1)) + logK + g.reshape((1, m))
        plan = backend.exp(logP)
        backend.eval(plan)

        row_sums = backend.sum(plan, axis=1)
        col_sums = backend.sum(plan, axis=0)
        row_error = backend.max(backend.abs(row_sums - mu))
        col_error = backend.max(backend.abs(col_sums - nu))
        backend.eval(row_error, col_error)
        marginal_error = max(
            float(self._to_scalar(row_error)), float(self._to_scalar(col_error))
        )

        cost = backend.sum(cost_matrix * plan)
        backend.eval(cost)

        return SinkhornResult(
            plan=plan,
            converged=converged,
            iterations=iterations,
            marginal_error=marginal_error,
            cost=float(self._to_scalar(cost)),
        )

    def squared_chord_cost(
        self, source: "Array", target: "Array", normalize: bool = True
    ) -> "Array":
        """Compute squared geodesic cost matrix (preferred name)."""
        backend = self._backend
        s = backend.array(source)
        t = backend.array(target)

        if normalize:
            div_eps = division_epsilon(backend, s)
            s_norms = geodesic_norms(s, backend)
            t_norms = geodesic_norms(t, backend)
            backend.eval(s_norms, t_norms)
            s_norm = backend.reshape(s_norms, (-1, 1)) + div_eps
            t_norm = backend.reshape(t_norms, (-1, 1)) + div_eps
            s = s / s_norm
            t = t / t_norm
            backend.eval(s, t)
        return self.squared_geodesic_cost(s, t, k_neighbors=None)


    def squared_geodesic_cost(
        self,
        source: "Array",
        target: "Array",
        k_neighbors: int | None = None,
    ) -> "Array":
        """Compute squared geodesic cost matrix on the manifold."""
        backend = self._backend
        s = backend.array(source)
        t = backend.array(target)
        backend.eval(s, t)

        n = int(s.shape[0])
        m = int(t.shape[0])

        combined = backend.concatenate([s, t], axis=0)
        backend.eval(combined)

        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        dist = geodesic_distance_matrix(combined, k_neighbors=k_neighbors, backend=backend)
        backend.eval(dist)

        row_idx = backend.arange(n)
        col_idx = backend.arange(n, n + m)
        sub = backend.take(dist, row_idx, axis=0)
        sub = backend.take(sub, col_idx, axis=1)
        return sub * sub

    def cosine_cost(self, source: "Array", target: "Array") -> "Array":
        """Compute cosine distance cost matrix.

        Args:
            source: Source points [n, d]
            target: Target points [m, d]

        Returns:
            Cost matrix [n, m] where cost = 1 - cosine_similarity
        """
        backend = self._backend
        s = backend.array(source)
        t = backend.array(target)
        similarity = geodesic_cosine_between_sets(s, t, backend)
        cost = 1 - similarity
        clamped = backend.minimum(
            backend.maximum(cost, backend.array(0.0)), backend.array(2.0)
        )
        backend.eval(clamped)
        return clamped

    def _logsumexp(self, array: "Array", axis: int) -> "Array":
        """Numerically stable log-sum-exp with additional guards."""
        backend = self._backend
        max_val = backend.max(array, axis=axis, keepdims=True)
        shifted = array - max_val
        sum_exp = backend.sum(backend.exp(shifted), axis=axis)
        # Guard against log(0) when sum_exp is very small
        log_eps = safe_log_epsilon(backend, sum_exp)
        safe_sum = backend.maximum(sum_exp, backend.full(sum_exp.shape, log_eps))
        return backend.squeeze(max_val, axis=axis) + backend.log(safe_sum)

    def _to_scalar(self, array: "Array") -> float:
        """Extract scalar value from array."""
        backend = self._backend
        if hasattr(array, "item"):
            return array.item()
        arr = backend.array(array)
        backend.eval(arr)
        return backend.to_scalar(arr)
