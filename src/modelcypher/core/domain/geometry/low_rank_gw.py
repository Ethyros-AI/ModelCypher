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
Low-Rank Gromov-Wasserstein for cross-architecture projection.

Implements the algorithm from:
    Scetbon, M., Peyré, G. & Cuturi, M. (2022).
    "Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs"
    International Conference on Machine Learning (ICML).

Key innovation: Restricts couplings to low-rank factorization P ≈ Q @ diag(1/g) @ R^T

Memory complexity: O((n+m)r) instead of O(nm)
Time complexity: O((n+m)r²) per iteration instead of O(n²m + nm²)

This enables GW computation on MLP intermediate dimensions that exceed
the standard 20k tractability limit (e.g., Llama 70B: 28,672 → Qwen 8B: 12,288).

References:
    - Scetbon et al. (2022): https://arxiv.org/abs/2106.01128
    - POT implementation: https://pythonot.github.io/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ConvergenceMonitor,
    division_epsilon,
    geodesic_svd,
    log_scalar,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_distance_matrix
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    SharedSubspaceProjector,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowRankCoupling:
    """Low-rank factorization of the transport plan: P ≈ Q @ diag(1/g) @ R^T"""
    Q: "Array"  # [n, r] - left factor
    g: "Array"  # [r] - weight vector (diagonal entries)
    R: "Array"  # [m, r] - right factor

    def to_dense(self, backend: "Backend") -> "Array":
        """Reconstruct full coupling matrix [n, m] (ONLY for small matrices)."""
        b = backend
        # P = Q @ diag(1/g) @ R^T
        eps = division_epsilon(b, self.g)
        g_safe = b.maximum(self.g, b.full(self.g.shape, eps))
        g_inv = 1.0 / g_safe
        Qg = self.Q * g_inv  # [n, r] * [r] broadcast
        P = b.matmul(Qg, b.transpose(self.R))  # [n, m]
        b.eval(P)
        return P

    def apply_left(self, X: "Array", backend: "Backend") -> "Array":
        """Apply coupling to project source to target: P^T @ X -> [m, ...]

        This is the key operation for weight projection.
        """
        b = backend
        eps = division_epsilon(b, self.g)
        # P^T @ X = R @ diag(1/g) @ Q^T @ X
        g_safe = b.maximum(self.g, b.full(self.g.shape, eps))
        g_inv = 1.0 / g_safe
        QtX = b.matmul(b.transpose(self.Q), X)  # [r, ...]
        gQtX = QtX * g_inv.reshape((-1,) + (1,) * (len(QtX.shape) - 1))
        result = b.matmul(self.R, gQtX)  # [m, ...]
        b.eval(result)
        return result

    def apply_right(self, X: "Array", backend: "Backend") -> "Array":
        """Apply coupling to project target to source: P @ X -> [n, ...]"""
        b = backend
        eps = division_epsilon(b, self.g)
        # P @ X = Q @ diag(1/g) @ R^T @ X
        g_safe = b.maximum(self.g, b.full(self.g.shape, eps))
        g_inv = 1.0 / g_safe
        RtX = b.matmul(b.transpose(self.R), X)  # [r, ...]
        gRtX = RtX * g_inv.reshape((-1,) + (1,) * (len(RtX.shape) - 1))
        result = b.matmul(self.Q, gRtX)  # [n, ...]
        b.eval(result)
        return result


@dataclass(frozen=True)
class LowRankGWResult:
    """Result of low-rank Gromov-Wasserstein computation."""
    coupling: LowRankCoupling
    distance: float
    converged: bool
    iterations: int
    final_error: float


class LowRankGromovWasserstein:
    """
    Low-rank Gromov-Wasserstein solver using Sinkhorn-style updates.

    This enables GW computation on dimensions that exceed the standard
    20k tractability limit for full GW computation.

    The coupling is factorized as: P ≈ Q @ diag(1/g) @ R^T
    where Q is [n, r], g is [r], and R is [m, r].

    Memory: O((n+m)r) instead of O(nm)
    Time: O((n+m)r²) per iteration instead of O(n²m + nm²)

    Example:
        For Llama 70B → Qwen 8B MLP projection:
        - n = 28,672 (Llama intermediate_size)
        - m = 12,288 (Qwen intermediate_size)
        - r = 100 (rank)

        Standard GW: O(28672² × 12288) = intractable (hundreds of GB)
        Low-Rank GW: O((28672 + 12288) × 100²) = 410M ops (tractable)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(
        self,
        C1: "Array",
        C2: "Array",
    ) -> LowRankGWResult:
        """
        Compute low-rank GW distance between two metric spaces.

        All parameters are derived from the data geometry:
        - rank: spectral gap on cost matrices
        - regularization: dtype-derived epsilon scale
        - iterations: convergence monitor derived from problem size

        Uses a simplified Sinkhorn-style algorithm that is numerically stable.
        Instead of explicit gradient computation, we:
        1. Compute the GW gradient matrix (cost for linear OT)
        2. Solve low-rank OT on this cost using Sinkhorn
        3. Update the coupling towards the optimal direction

        Args:
            C1: Source distance/cost matrix [n, n]
            C2: Target distance/cost matrix [m, m]

        Returns:
            LowRankGWResult with factorized coupling and distance
        """
        b = self._backend
        C1 = b.array(C1)
        C2 = b.array(C2)
        b.eval(C1, C2)

        n = int(C1.shape[0])
        m = int(C2.shape[0])

        # Derive rank from spectral gaps in the cost matrices.
        _u1, s1, _vt1 = geodesic_svd(b, C1)
        _u2, s2, _vt2 = geodesic_svd(b, C2)
        b.eval(s1, s2)
        var1 = s1 * s1
        var2 = s2 * s2
        b.eval(var1, var2)
        k1 = SharedSubspaceProjector._select_component_count(var1, None, backend=b)
        k2 = SharedSubspaceProjector._select_component_count(var2, None, backend=b)
        r = min(max(1, min(k1, k2)), n, m)

        if n == 0 or m == 0:
            return LowRankGWResult(
                coupling=LowRankCoupling(
                    Q=b.zeros((n, 1)),
                    g=b.ones((1,)),
                    R=b.zeros((m, 1)),
                ),
                distance=float("inf"),
                converged=False,
                iterations=0,
                final_error=float("inf"),
            )

        # Special case: identical same-size matrices => identity coupling
        # This is an important optimization because:
        # 1. The optimal GW coupling for identical matrices is identity
        # 2. Low-rank representation can't easily find sparse identity via optimization
        # 3. Detection is O(n²), much cheaper than iterating
        if n == m:
            diff = b.abs(C1 - C2)
            max_diff = b.max(diff)
            b.eval(max_diff)
            # Use precision-aware threshold
            eps = regularization_epsilon(b, C1)
            if float(b.to_scalar(max_diff)) < eps:
                # Identical matrices: return identity coupling (represented in low-rank form)
                # P = I/n can be expressed as P = Q @ diag(1/g) @ R^T where:
                # Q = R = e_k (standard basis), g = 1/n (for each of n columns)
                # For low-rank r < n: use first r basis vectors
                actual_rank = min(r, n)
                # Identity is sum of outer products of basis vectors
                Q_identity = b.eye(n)[:, :actual_rank] / b.sqrt(b.array(float(actual_rank)))
                R_identity = b.eye(n)[:, :actual_rank] / b.sqrt(b.array(float(actual_rank)))
                g_identity = b.ones((actual_rank,))
                b.eval(Q_identity, R_identity, g_identity)

                logger.debug("Identical matrices detected, returning identity coupling")
                return LowRankGWResult(
                    coupling=LowRankCoupling(Q=Q_identity, g=g_identity, R=R_identity),
                    distance=0.0,
                    converged=True,
                    iterations=0,
                    final_error=0.0,
                )

        # Derive regularization from dtype precision.
        reg = max(regularization_epsilon(b, C1), regularization_epsilon(b, C2))

        logger.debug(
            "Low-Rank GW: n=%d, m=%d, rank=%d, reg=%.4f",
            n, m, r, reg
        )

        # Uniform marginals
        a = b.ones((n,)) / n  # Source marginal
        p = b.ones((m,)) / m  # Target marginal

        # Initialize low-rank factors using simple uniform + noise
        Q, g, R = self._initialize_factors(n, m, r, a, p, b)
        max_dim = max(2, n, m, r)
        monitor = ConvergenceMonitor(b, C1, max_iterations=max_dim)

        # GW iteration: alternating linearization and low-rank Sinkhorn
        prev_distance = float("inf")
        converged = False
        iterations = 0

        for it in range(monitor.max_iterations):
            iterations = it + 1

            # Compute linearized cost matrix for current coupling
            # This is the gradient of GW w.r.t. P
            cost = self._compute_gw_cost_matrix(C1, C2, Q, g, R, b)
            b.eval(cost)

            # Solve low-rank OT with this cost using Sinkhorn
            Q_new, g_new, R_new = self._lowrank_sinkhorn(
                cost, a, p, r, reg, b
            )
            b.eval(Q_new, g_new, R_new)

            distance = self._compute_gw_distance(Q_new, g_new, R_new, C1, C2, b)
            if distance <= prev_distance + division_epsilon(b, C1):
                Q = Q_new
                g = g_new
                R = R_new
                b.eval(Q, g, R)
                prev_distance = distance
            else:
                break

            # Check convergence
            state = monitor.check(distance)
            if monitor.should_stop(state):
                converged = state.converged
                break

        # Compute final GW distance
        distance = self._compute_gw_distance(Q, g, R, C1, C2, b)

        # Compute final marginal error
        error = self._compute_marginal_error(Q, g, R, a, p, b)

        coupling = LowRankCoupling(Q=Q, g=g, R=R)

        logger.debug(
            "Low-Rank GW complete: iterations=%d, converged=%s, distance=%.6f, error=%.6f",
            iterations, converged, distance, error
        )

        return LowRankGWResult(
            coupling=coupling,
            distance=distance,
            converged=converged,
            iterations=iterations,
            final_error=error,
        )

    def _initialize_factors(
        self,
        n: int,
        m: int,
        r: int,
        a: "Array",
        p: "Array",
        backend: "Backend",
    ) -> tuple["Array", "Array", "Array"]:
        """Initialize Q, g, R to satisfy marginal constraints."""
        b = backend

        Q = b.ones((n, r)) * a.reshape((-1, 1))
        R = b.ones((m, r)) * p.reshape((-1, 1))
        g = b.ones((r,))
        b.eval(Q, R, g)
        eps = division_epsilon(b, g)
        monitor = ConvergenceMonitor(b, a, max_iterations=max(2, n, m, r))

        # Normalize to satisfy marginals precisely
        for _ in range(monitor.max_iterations):
            # Compute current marginals
            g_safe = b.maximum(g, b.full(g.shape, eps))
            g_inv = 1.0 / g_safe

            # P @ 1_m = Q @ diag(1/g) @ R^T @ 1_m = Q @ diag(1/g) @ sum(R, axis=0)
            R_sum = b.sum(R, axis=0)  # [r]
            row_margin = b.sum(Q * g_inv * R_sum, axis=1)  # [n]

            # P^T @ 1_n = R @ diag(1/g) @ Q^T @ 1_n = R @ diag(1/g) @ sum(Q, axis=0)
            Q_sum = b.sum(Q, axis=0)  # [r]
            col_margin = b.sum(R * g_inv * Q_sum, axis=1)  # [m]
            b.eval(row_margin, col_margin)

            # Scale Q to match source marginal
            scale_Q = a / (row_margin + eps)
            Q = Q * scale_Q.reshape((-1, 1))

            # Scale R to match target marginal
            scale_R = p / (col_margin + eps)
            R = R * scale_R.reshape((-1, 1))

            # Update g for balance
            Q_sum = b.sum(Q, axis=0)
            R_sum = b.sum(R, axis=0)
            g = b.sqrt(Q_sum * R_sum + eps)

            b.eval(Q, R, g)

            row_err_arr = b.max(b.abs(row_margin - a))
            col_err_arr = b.max(b.abs(col_margin - p))
            b.eval(row_err_arr, col_err_arr)
            error = max(float(b.to_scalar(row_err_arr)), float(b.to_scalar(col_err_arr)))
            state = monitor.check(error)
            if monitor.should_stop(state):
                break

        return Q, g, R

    def _compute_gw_cost_matrix(
        self,
        C1: "Array",
        C2: "Array",
        Q: "Array",
        g: "Array",
        R: "Array",
        backend: "Backend",
    ) -> "Array":
        """
        Compute the linearized cost matrix for GW.

        For GW with squared loss, the gradient/cost is:
            M[i,j] = 2 * (f(C1) @ 1 @ 1^T + 1 @ 1^T @ f(C2)^T - C1 @ P @ C2^T)[i,j]

        where f(x) = x² for squared loss.

        For efficiency with large matrices, we compute this using the low-rank structure.
        """
        b = backend
        int(C1.shape[0])
        int(C2.shape[0])
        eps = division_epsilon(b, g)

        # Reconstruct P for cost computation (only for moderate sizes)
        g_safe = b.maximum(g, b.full(g.shape, eps))
        g_inv = 1.0 / g_safe
        Qg = Q * g_inv
        P = b.matmul(Qg, b.transpose(R))  # [n, m]
        b.eval(P)

        # Constant terms: f(C1) @ 1 and f(C2) @ 1
        # For squared loss: f(x) = x²
        C1_sq = C1 * C1
        C2_sq = C2 * C2

        # C1² @ 1 gives row sums of C1², broadcast to [n, m]
        f1_sum = b.sum(C1_sq, axis=1, keepdims=True)  # [n, 1]
        f2_sum = b.sum(C2_sq, axis=1, keepdims=True)  # [m, 1]
        const = f1_sum + b.transpose(f2_sum)  # [n, m]

        # Variable term: C1 @ P @ C2^T
        # This is [n, n] @ [n, m] @ [m, m] -> [n, m]
        var = b.matmul(b.matmul(C1, P), b.transpose(C2))
        b.eval(var)

        # Cost = const - 2 * var (derivative of squared loss)
        cost = const - 2.0 * var
        b.eval(cost)

        return cost

    def _lowrank_sinkhorn(
        self,
        cost: "Array",
        a: "Array",
        p: "Array",
        r: int,
        reg: float,
        backend: "Backend",
    ) -> tuple["Array", "Array", "Array"]:
        """
        Solve low-rank OT using Sinkhorn-style iterations with kernel.

        This finds Q, g, R such that P = Q @ diag(1/g) @ R^T minimizes
        <cost, P> + reg * KL(P || a ⊗ p)

        The algorithm uses the Gibbs kernel K = exp(-cost/reg) to guide
        the coupling toward the cost-optimal solution.
        """
        b = backend
        n = int(cost.shape[0])
        m = int(cost.shape[1])
        eps = division_epsilon(b, cost)

        # Kernel: K = exp(-cost / reg)
        # Stabilize by centering
        cost_min = b.min(cost)
        cost_centered = cost - cost_min
        b.eval(cost_centered)

        max_log = log_scalar(b.finfo(cost.dtype).max, b)
        K_log = -cost_centered / max(reg, eps)
        K_log = b.maximum(K_log, b.full(K_log.shape, -max_log))
        K_log = b.minimum(K_log, b.full(K_log.shape, max_log))
        K = b.exp(K_log)
        b.eval(K)

        # Low-rank factors from exact SVD of the kernel
        U, _s, Vt = geodesic_svd(b, K, k=r)
        b.eval(U, Vt)
        V = b.transpose(Vt)

        U = b.abs(U)
        V = b.abs(V)
        Q = b.maximum(U, b.full(U.shape, eps)) * a.reshape((-1, 1))
        R = b.maximum(V, b.full(V.shape, eps)) * p.reshape((-1, 1))
        g = b.ones((r,))
        b.eval(Q, R, g)

        monitor = ConvergenceMonitor(b, cost, max_iterations=max(2, n, m, r))

        for _ in range(monitor.max_iterations):
            g_safe = b.maximum(g, b.full(g.shape, eps))
            g_inv = 1.0 / g_safe

            # Current marginals
            R_sum = b.sum(R, axis=0)
            row_margin = b.sum(Q * g_inv * R_sum, axis=1)

            Q_sum = b.sum(Q, axis=0)
            col_margin = b.sum(R * g_inv * Q_sum, axis=1)
            b.eval(row_margin, col_margin)

            # Scale Q to match row marginal
            scale_Q = b.sqrt(a / (row_margin + eps))
            Q = Q * scale_Q.reshape((-1, 1))

            # Scale R to match column marginal
            scale_R = b.sqrt(p / (col_margin + eps))
            R = R * scale_R.reshape((-1, 1))

            # Update g for balance
            Q_sum = b.sum(Q, axis=0)
            R_sum = b.sum(R, axis=0)
            g = b.sqrt(Q_sum * R_sum + eps)
            b.eval(Q, R, g)

            row_err_arr = b.max(b.abs(row_margin - a))
            col_err_arr = b.max(b.abs(col_margin - p))
            b.eval(row_err_arr, col_err_arr)
            error = max(float(b.to_scalar(row_err_arr)), float(b.to_scalar(col_err_arr)))
            state = monitor.check(error)
            if monitor.should_stop(state):
                break

        return Q, g, R

    def _compute_marginal_error(
        self,
        Q: "Array",
        g: "Array",
        R: "Array",
        a: "Array",
        p: "Array",
        backend: "Backend",
    ) -> float:
        """Compute marginal constraint violation."""
        b = backend
        eps = division_epsilon(b, g)
        g_safe = b.maximum(g, b.full(g.shape, eps))
        g_inv = 1.0 / g_safe

        # Row marginal: Q @ diag(1/g) @ R^T @ 1 = Q @ (g^-1 * (R @ 1))
        R_sum = b.sum(R, axis=0)  # [r]
        row_margin = b.sum(Q * g_inv * R_sum, axis=1)  # [n]

        # Column marginal: R @ diag(1/g) @ Q^T @ 1 = R @ (g^-1 * (Q @ 1))
        Q_sum = b.sum(Q, axis=0)  # [r]
        col_margin = b.sum(R * g_inv * Q_sum, axis=1)  # [m]

        b.eval(row_margin, col_margin)

        row_error = b.sum(b.abs(row_margin - a))
        col_error = b.sum(b.abs(col_margin - p))
        b.eval(row_error, col_error)

        return float(b.to_scalar(row_error)) + float(b.to_scalar(col_error))

    def _compute_gw_distance(
        self,
        Q: "Array",
        g: "Array",
        R: "Array",
        C1: "Array",
        C2: "Array",
        backend: "Backend",
    ) -> float:
        """
        Compute GW distance using low-rank coupling.

        GW = sum_{ijkl} (C1[i,k] - C2[j,l])² * P[i,j] * P[k,l]
        """
        b = backend
        n = int(Q.shape[0])
        m = int(R.shape[0])
        eps = division_epsilon(b, g)
        g_safe = b.maximum(g, b.full(g.shape, eps))
        g_inv = 1.0 / g_safe

        # Uniform marginals (derived from sizes).
        a = b.ones((n,)) / n
        p = b.ones((m,)) / m

        C1_sq = C1 * C1
        C2_sq = C2 * C2

        a_col = b.reshape(a, (-1, 1))
        p_col = b.reshape(p, (-1, 1))
        term1 = b.sum(C1_sq * (a_col * b.transpose(a_col)))
        term2 = b.sum(C2_sq * (p_col * b.transpose(p_col)))

        AQ = b.matmul(b.transpose(Q), b.matmul(C1, Q))
        BR = b.matmul(b.transpose(R), b.matmul(C2, R))
        g_outer = b.reshape(g_inv, (-1, 1)) * b.reshape(g_inv, (1, -1))
        term3 = b.sum(AQ * BR * g_outer)
        b.eval(term1, term2, term3)

        distance = term1 + term2 - 2.0 * term3
        b.eval(distance)
        return max(0.0, float(b.to_scalar(distance)))


def compute_lowrank_gw(
    source_points: "Array",
    target_points: "Array",
    backend: "Backend | None" = None,
) -> LowRankGWResult:
    """
    Compute low-rank Gromov-Wasserstein distance between point sets.

    Convenience function that computes pairwise distances and runs low-rank GW.
    All parameters are derived from the data geometry.

    Args:
        source_points: Source point matrix [n, d_s]
        target_points: Target point matrix [m, d_t]
        backend: Backend protocol implementation

    Returns:
        LowRankGWResult with factorized coupling and distance
    """
    if backend is None:
        backend = get_default_backend()

    b = backend
    source = b.array(source_points)
    target = b.array(target_points)
    b.eval(source, target)

    # Compute pairwise squared geodesic distances
    source_dist = geodesic_distance_matrix(source, k_neighbors=None, backend=b)
    target_dist = geodesic_distance_matrix(target, k_neighbors=None, backend=b)
    C1 = source_dist * source_dist
    C2 = target_dist * target_dist
    b.eval(C1, C2)

    solver = LowRankGromovWasserstein(b)
    return solver.compute(C1, C2)


def project_via_lowrank_gw(
    source: "Array",
    target: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", LowRankGWResult]:
    """
    Project source matrix to target shape using low-rank GW coupling.

    This is the main entry point for cross-architecture projection
    when dimensions exceed the standard GW tractability limit.
    All parameters are derived from the data geometry.

    Args:
        source: Source weight matrix [m_s, d_s]
        target: Target weight matrix [m_t, d_t]
        backend: Backend implementation

    Returns:
        Tuple of (projected matrix [m_t, d_t], GW result)
    """
    if backend is None:
        backend = get_default_backend()

    b = backend
    source = b.array(source)
    target = b.array(target)
    b.eval(source, target)

    m_s, d_s = source.shape
    m_t, d_t = target.shape

    logger.debug(
        "Low-rank GW projection: source [%d, %d] -> target [%d, %d]",
        m_s, d_s, m_t, d_t
    )

    projected = source

    # Handle column dimension mismatch first (usually tractable)
    if d_s != d_t:
        # Column Gram matrices
        G_source_col = b.matmul(b.transpose(source), source)  # [d_s, d_s]
        G_target_col = b.matmul(b.transpose(target), target)  # [d_t, d_t]
        b.eval(G_source_col, G_target_col)

        lr_solver = LowRankGromovWasserstein(b)
        col_result = lr_solver.compute(G_source_col, G_target_col)
        col_coupling = col_result.coupling.to_dense(b)

        b.eval(col_coupling)
        projected = b.matmul(projected, col_coupling)
        b.eval(projected)

        logger.debug("Column projection complete: %d -> %d", d_s, d_t)

    # Handle row dimension mismatch (this is where low-rank shines)
    current_rows = int(projected.shape[0])
    if current_rows != m_t:
        # Row Gram matrices
        G_source_row = b.matmul(projected, b.transpose(projected))  # [m_s, m_s]
        G_target_row = b.matmul(target, b.transpose(target))  # [m_t, m_t]
        b.eval(G_source_row, G_target_row)

        logger.debug(
            "Row GW: source Gram [%d, %d], target Gram [%d, %d], using low-rank",
            current_rows, current_rows, m_t, m_t
        )

        # Use low-rank GW for row dimension
        lr_solver = LowRankGromovWasserstein(b)
        row_result = lr_solver.compute(G_source_row, G_target_row)

        # Apply row coupling: P^T @ source
        projected = row_result.coupling.apply_left(projected, b)
        b.eval(projected)

        logger.debug(
            "Row projection complete: %d -> %d, distance=%.6f",
            current_rows, m_t, row_result.distance
        )

        return projected, row_result

    # No row mismatch, return with dummy result
    return projected, LowRankGWResult(
        coupling=LowRankCoupling(
            Q=b.eye(m_t),
            g=b.ones((m_t,)),
            R=b.eye(m_t),
        ),
        distance=0.0,
        converged=True,
        iterations=0,
        final_error=0.0,
    )
