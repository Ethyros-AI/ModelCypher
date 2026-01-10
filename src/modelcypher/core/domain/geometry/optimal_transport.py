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
Optimal Transport solvers for representation alignment.

Provides GPU-accelerated Sinkhorn algorithm for computing optimal transport
plans between probability distributions. Used by:
- Gromov-Wasserstein distance (as Frank-Wolfe inner loop)
- Soft Procrustes alignment (for non-corresponding anchors)
- Representation matching across models

Mathematical Foundation:
    The Sinkhorn algorithm solves the entropy-regularized optimal transport:

        min_P <C, P> + epsilon * H(P)
        subject to: P @ 1 = mu, P.T @ 1 = nu

    where C is the cost matrix, P is the transport plan, and H(P) is entropy.

    As epsilon -> 0, the solution approaches exact optimal transport.

References:
    - Cuturi (2013) "Sinkhorn Distances" NeurIPS
    - Peyré & Cuturi (2019) "Computational Optimal Transport"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    regularization_epsilon,
    safe_log_epsilon,
    tiny_value,
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


def _derive_max_iterations(n: int, m: int) -> int:
    """Derive max Sinkhorn iterations from problem size.

    Uses logarithmic scaling: max_iter = max(50, 10 * ceil(log2(max(n, m) + 1)))
    This gives reasonable upper bounds that scale with problem size:
    - n=m=10: 50 iterations
    - n=m=100: 70 iterations
    - n=m=1000: 100 iterations

    Convergence-based stopping typically triggers well before this bound.
    """
    import math
    return max(50, 10 * int(math.ceil(math.log2(max(n, m) + 1))))


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
        stability_epsilon = division_epsilon(backend, cost_matrix)

        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])

        # Default to uniform marginals
        mu = (
            source_marginal
            if source_marginal is not None
            else backend.ones((n,), dtype="float32") / float(n)
        )
        nu = (
            target_marginal
            if target_marginal is not None
            else backend.ones((m,), dtype="float32") / float(m)
        )
        backend.eval(mu, nu)

        # Always use log-domain for numerical stability
        max_iterations = _derive_max_iterations(n, m)
        return self._solve_log_domain(
            cost_matrix,
            mu,
            nu,
            epsilon,
            convergence_threshold,
            stability_epsilon,
            max_iterations,
        )

    def _derive_epsilon(self, cost: "Array") -> float:
        """Derive Sinkhorn regularization epsilon from cost matrix scale.

        Standard practice in OT: epsilon proportional to cost scale.
        Using median(cost) * sqrt(machine_epsilon) provides principled
        balance between accuracy and numerical stability.
        """
        backend = self._backend
        median_val = compute_median(cost, backend)
        if median_val == 0.0:
            return float(division_epsilon(backend, cost))

        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
        eps = float(machine_epsilon(backend, cost))
        return max(median_val * (eps ** 0.5), eps)

    def solve_linear_ot(
        self,
        cost: "Array",
        p: "Array",
        q: "Array",
        epsilon: float,
        max_iterations: int | None = None,
        threshold: float = 0.0,
    ) -> "Array":
        """Solve linear optimal transport - fast version for inner loops.

        This is a simplified interface for use in algorithms like Gromov-Wasserstein
        where the Sinkhorn is called repeatedly as an inner loop and only the
        transport plan is needed (not convergence diagnostics).

        Args:
            cost: Cost matrix [n, m]
            p: Source marginal [n]
            q: Target marginal [m]
            epsilon: Entropy regularization strength
            max_iterations: Maximum Sinkhorn iterations (derived from problem size if None)
            threshold: Convergence threshold (0 = run all iterations)

        Returns:
            Transport plan [n, m]
        """
        backend = self._backend
        n = int(cost.shape[0])
        m = int(cost.shape[1])

        # Derive max iterations from problem size if not specified
        if max_iterations is None:
            max_iterations = _derive_max_iterations(n, m)

        if n == 0 or m == 0:
            return backend.zeros((n, m))

        # Use precision-aware epsilon and tiny value
        eps = division_epsilon(backend, cost)
        floor = tiny_value(backend, cost)
        floor_vec_n = backend.full((n,), floor)
        floor_vec_m = backend.full((m,), floor)
        floor_mat = backend.full((n, m), floor)

        # Stabilized Sinkhorn with row-wise centering
        cost_min = backend.min(cost, axis=1, keepdims=True)
        cost_centered = cost - cost_min
        log_K = -cost_centered / max(epsilon, eps)

        # Clamp to avoid underflow
        log_K = backend.maximum(log_K, backend.full((n, m), -80.0))
        K = backend.exp(log_K)
        K = backend.maximum(K, floor_mat)

        # Initialize scaling vectors
        u = backend.ones((n,))
        v = backend.ones((m,))

        K_T = backend.transpose(K)
        for _ in range(max_iterations):
            # Row scaling: u = p / (K @ v)
            Kv = backend.matmul(K, v)
            Kv = backend.maximum(Kv, floor_vec_n)
            u_new = p / Kv

            # Column scaling: v = q / (K.T @ u)
            Ktu = backend.matmul(K_T, u_new)
            Ktu = backend.maximum(Ktu, floor_vec_m)
            v_new = q / Ktu

            # Check convergence if threshold > 0
            if threshold > 0:
                u_diff = backend.max(backend.abs(u_new - u))
                v_diff = backend.max(backend.abs(v_new - v))
                backend.eval(u_diff, v_diff)
                if max(
                    float(backend.to_scalar(u_diff)),
                    float(backend.to_scalar(v_diff)),
                ) < threshold:
                    u = u_new
                    v = v_new
                    break

            u = u_new
            v = v_new

        # Recover transport plan: G = diag(u) @ K @ diag(v)
        G = K * backend.reshape(u, (n, 1)) * backend.reshape(v, (1, m))
        return G

    def _solve_log_domain(
        self,
        cost_matrix: "Array",
        mu: "Array",
        nu: "Array",
        epsilon: float,
        convergence_threshold: float,
        stability_epsilon: float,
        max_iterations: int,
    ) -> SinkhornResult:
        """Log-domain Sinkhorn for improved numerical stability."""
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
        logK = -cost_matrix / epsilon
        backend.eval(log_mu, log_nu, logK)

        f = backend.zeros((n,), dtype="float32")
        g = backend.zeros((m,), dtype="float32")
        backend.eval(f, g)

        converged = False
        iterations = 0
        marginal_error = float("inf")

        logK_T = backend.transpose(logK)
        for i in range(max_iterations):
            iterations = i + 1
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

            f = f_new
            g = g_new
            if marginal_error < convergence_threshold or max_diff < convergence_threshold:
                converged = True
                break

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
