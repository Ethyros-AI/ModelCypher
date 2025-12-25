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
Gromov-Wasserstein distance computation for representation space comparison.

GPU-accelerated implementation using the Backend protocol (MLX/JAX/CUDA).

Mathematical Foundation:
    The Gromov-Wasserstein distance measures structural similarity between
    metric spaces without requiring point-to-point correspondence. Given
    source (X, dX) and target (Y, dY) metric spaces with probability measures
    μ and ν, the GW objective minimizes:

        GW(μ, ν) = min_γ ∑_{i,j,k,l} L(dX(xi, xk), dY(yj, yl)) · γij · γkl

    where γ is a coupling matrix with marginals μ and ν.

Key Concepts:
    - Coupling Matrix: Soft matching between source and target points
    - Entropic Regularization: Adds entropy term εH(γ) for tractability
    - Sinkhorn Algorithm: Iteratively projects onto transport polytope
    - Normalized Distance: 1 - exp(-GW) maps to [0, 1] for interpretability

Complexity:
    O(n²m²) per outer iteration due to cost matrix computation.
    Sinkhorn inner loop is O(nm) per iteration.

References:
    - Peyré & Cuturi (2019) "Computational Optimal Transport"
    - Mémoli (2011) "Gromov-Wasserstein distances and the metric approach"

See also: docs/geometry/gromov_wasserstein.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class Result:
    distance: float
    coupling: "Array"
    converged: bool
    iterations: int

    @property
    def normalized_distance(self) -> float:
        return 1.0 - math.exp(-self.distance) if math.isfinite(self.distance) else 1.0

    @property
    def compatibility_score(self) -> float:
        return math.exp(-self.distance) if math.isfinite(self.distance) else 0.0


@dataclass(frozen=True)
class Config:
    epsilon: float = 0.05
    epsilon_min: float = 0.005
    epsilon_decay: float = 1.0
    max_outer_iterations: int = 50
    min_outer_iterations: int = 3
    max_inner_iterations: int = 100
    convergence_threshold: float = 1e-5
    relative_objective_threshold: float = 1e-5
    use_squared_loss: bool = True


class GromovWassersteinDistance:
    """GPU-accelerated Gromov-Wasserstein distance computation."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(
        self,
        source_distances: "Array",
        target_distances: "Array",
        config: Config = Config(),
    ) -> Result:
        """
        Compute Gromov-Wasserstein distance between two metric spaces.

        Args:
            source_distances: Pairwise distance matrix for source [n, n]
            target_distances: Pairwise distance matrix for target [m, m]
            config: Algorithm configuration

        Returns:
            Result with distance, coupling matrix, and convergence info
        """
        backend = self._backend

        # Convert inputs to backend arrays if needed
        source_distances = backend.array(source_distances)
        target_distances = backend.array(target_distances)
        backend.eval(source_distances, target_distances)

        n = int(source_distances.shape[0])
        m = int(target_distances.shape[0])

        if n == 0 or m == 0:
            return Result(
                distance=float("inf"),
                coupling=backend.zeros((0, 0)),
                converged=False,
                iterations=0,
            )

        # Check for identical matrices
        if n == m:
            diff = backend.abs(source_distances - target_distances)
            max_diff = backend.max(diff)
            backend.eval(max_diff)
            if float(backend.to_numpy(max_diff)) < 1e-6:
                # Identical - return identity coupling
                coupling = backend.eye(n) / n
                return Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        # Initialize uniform coupling
        coupling = backend.ones((n, m)) / (n * m)

        converged = False
        iterations = 0
        prev_distance = float("inf")

        min_outer = max(1, min(config.min_outer_iterations, config.max_outer_iterations))
        decay = min(max(config.epsilon_decay, 0.0), 1.0)
        min_epsilon = max(1e-6, min(config.epsilon_min, config.epsilon))
        epsilon = max(config.epsilon, min_epsilon)

        for outer in range(config.max_outer_iterations):
            iterations = outer + 1

            # Compute cost matrix using GPU-accelerated tensor operations
            cost = self._compute_cost_vectorized(
                source_distances, target_distances, coupling, config.use_squared_loss
            )

            # Sinkhorn step
            new_coupling = self._sinkhorn_step(
                cost,
                epsilon=epsilon,
                max_iterations=config.max_inner_iterations,
                convergence_threshold=config.convergence_threshold,
            )

            # Check coupling convergence
            diff = backend.abs(new_coupling - coupling)
            max_change_arr = backend.max(diff)
            backend.eval(max_change_arr)
            max_change = float(backend.to_numpy(max_change_arr))

            coupling = new_coupling

            # Compute objective
            distance = self._compute_objective_vectorized(
                source_distances, target_distances, coupling, config.use_squared_loss
            )

            objective_change = abs(distance - prev_distance) if math.isfinite(prev_distance) else float("inf")
            relative_change = (
                objective_change / max(abs(prev_distance), 1e-8)
                if math.isfinite(prev_distance)
                else float("inf")
            )

            meets_coupling = max_change < config.convergence_threshold
            meets_objective = (
                objective_change < config.convergence_threshold
                or relative_change < config.relative_objective_threshold
            )

            if iterations >= min_outer and (meets_coupling or meets_objective):
                converged = True
                break

            prev_distance = distance
            epsilon = max(min_epsilon, epsilon * decay)

        final_distance = self._compute_objective_vectorized(
            source_distances, target_distances, coupling, config.use_squared_loss
        )

        return Result(
            distance=final_distance,
            coupling=coupling,
            converged=converged,
            iterations=iterations,
        )

    def compute_pairwise_distances(self, points: "Array") -> "Array":
        """
        Compute pairwise Euclidean distances using GPU-accelerated operations.

        Args:
            points: Point matrix [n, d]

        Returns:
            Distance matrix [n, n]
        """
        backend = self._backend
        # Convert to backend array if needed (e.g., from Python list)
        points = backend.array(points)
        backend.eval(points)
        n = int(points.shape[0])

        if n == 0:
            return backend.zeros((0, 0))

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        norms = backend.sum(points * points, axis=1, keepdims=True)
        dots = backend.matmul(points, backend.transpose(points))
        dist_sq = norms + backend.transpose(norms) - 2.0 * dots

        # Ensure non-negative
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))
        dist = backend.sqrt(dist_sq)

        return dist

    def _compute_cost_vectorized(
        self,
        source_distances: "Array",
        target_distances: "Array",
        coupling: "Array",
        use_squared_loss: bool,
    ) -> "Array":
        """
        Compute cost matrix using vectorized GPU operations.

        The cost at (i,j) is:
            C[i,j] = sum_{i',j'} L(dX[i,i'], dY[j,j']) * coupling[i',j']

        This can be computed efficiently using einsum-like operations.
        For squared loss: L(a,b) = (a-b)^2 = a^2 - 2ab + b^2

        With squared loss we can expand:
            C[i,j] = sum_{i',j'} (dX[i,i'] - dY[j,j'])^2 * coupling[i',j']
                   = sum_{i',j'} dX[i,i']^2 * coupling[i',j']
                   - 2 * sum_{i',j'} dX[i,i'] * dY[j,j'] * coupling[i',j']
                   + sum_{i',j'} dY[j,j']^2 * coupling[i',j']
        """
        backend = self._backend
        n = source_distances.shape[0]
        m = target_distances.shape[0]

        if use_squared_loss:
            # Squared source distances
            src_sq = source_distances * source_distances  # [n, n]
            tgt_sq = target_distances * target_distances  # [m, m]

            # Term 1: sum_{i',j'} dX[i,i']^2 * coupling[i',j']
            # = sum_{i'} dX[i,i']^2 * (sum_{j'} coupling[i',j'])
            # = dX^2 @ mu where mu[i'] = sum_{j'} coupling[i',j']
            mu = backend.sum(coupling, axis=1, keepdims=True)  # [n, 1] - row marginal
            nu = backend.sum(coupling, axis=0, keepdims=True)  # [1, m] - col marginal

            term1 = backend.matmul(src_sq, mu)  # [n, n] @ [n, 1] = [n, 1]
            term1 = backend.broadcast_to(term1, (n, m))

            # Term 3: sum_{i',j'} dY[j,j']^2 * coupling[i',j']
            # = sum_{j'} dY[j,j']^2 * (sum_{i'} coupling[i',j'])
            # = dY^2 @ nu^T where nu[j'] = sum_{i'} coupling[i',j']
            term3 = backend.matmul(tgt_sq, backend.transpose(nu))  # [m, m] @ [m, 1] = [m, 1]
            term3 = backend.transpose(term3)  # [1, m]
            term3 = backend.broadcast_to(term3, (n, m))

            # Term 2: -2 * sum_{i',j'} dX[i,i'] * coupling[i',j'] * dY[j,j']
            # = -2 * (dX @ coupling @ dY^T)
            term2 = -2.0 * backend.matmul(backend.matmul(source_distances, coupling), backend.transpose(target_distances))

            cost = term1 + term2 + term3
        else:
            # Absolute loss - need full loop (expensive but correct)
            # Fall back to simpler vectorized version
            # This is O(n*m) per (i,j) pair = O(n^2 * m^2) total
            # We vectorize over i,j: cost[i,j] = sum_{i',j'} |dX[i,i'] - dY[j,j']| * coupling[i',j']

            # Expand to 4D: [n, 1, n, 1] - [1, m, 1, m] -> [n, m, n, m]
            src_exp = backend.reshape(source_distances, (n, 1, n, 1))
            tgt_exp = backend.reshape(target_distances, (1, m, 1, m))

            # This creates a large tensor - may OOM for large n,m
            # diff[i,j,i',j'] = |dX[i,i'] - dY[j,j']|
            diff = backend.abs(src_exp - tgt_exp)  # [n, m, n, m]

            # Reshape coupling for broadcasting
            coupling_exp = backend.reshape(coupling, (1, 1, n, m))

            # cost[i,j] = sum_{i',j'} diff[i,j,i',j'] * coupling[i',j']
            weighted = diff * coupling_exp
            cost = backend.sum(weighted, axis=(2, 3))  # [n, m]

        return cost

    def _compute_objective_vectorized(
        self,
        source_distances: "Array",
        target_distances: "Array",
        coupling: "Array",
        use_squared_loss: bool,
    ) -> float:
        """
        Compute GW objective using vectorized operations.

        Objective = sum_{i,j,i',j'} L(dX[i,i'], dY[j,j']) * coupling[i,j] * coupling[i',j']
        """
        backend = self._backend
        n = source_distances.shape[0]
        m = target_distances.shape[0]

        if use_squared_loss:
            # For squared loss, use efficient formulation
            # = <coupling, C @ coupling^T> where C is cost-like
            # Actually: sum_{i,j} coupling[i,j] * C[i,j] where C uses coupling

            # More efficient: use the fact that
            # obj = ||dX @ coupling - coupling @ dY^T||_F^2 approximately
            # But exact is: sum of (dX[i,i'] - dY[j,j'])^2 * coupling[i,j] * coupling[i',j']

            # Compute: dX @ coupling @ dY^T
            cross = backend.matmul(backend.matmul(source_distances, coupling), target_distances)

            # Term: sum (dX @ coupling_marginal)^2 + sum (coupling_marginal^T @ dY)^2 - 2 * trace(cross @ coupling^T)
            # This is a simplification. Let's use direct computation.

            # Full computation (still vectorized):
            # = sum_{i,j,i',j'} (dX[i,i']^2 - 2*dX[i,i']*dY[j,j'] + dY[j,j']^2) * coupling[i,j] * coupling[i',j']

            # Term 1: sum_{i,i'} dX[i,i']^2 * coupling_row[i] * coupling_row[i']
            # where coupling_row[i] = sum_j coupling[i,j]
            row_sums = backend.sum(coupling, axis=1)  # [n]
            src_sq = source_distances * source_distances
            weighted_src = backend.sum(src_sq * backend.reshape(row_sums, (n, 1)) * backend.reshape(row_sums, (1, n)))

            # Term 3: sum_{j,j'} dY[j,j']^2 * coupling_col[j] * coupling_col[j']
            col_sums = backend.sum(coupling, axis=0)  # [m]
            tgt_sq = target_distances * target_distances
            weighted_tgt = backend.sum(tgt_sq * backend.reshape(col_sums, (m, 1)) * backend.reshape(col_sums, (1, m)))

            # Term 2: -2 * sum dX[i,i'] * dY[j,j'] * coupling[i,j] * coupling[i',j']
            # = -2 * sum_{i,i'} dX[i,i'] * (sum_j coupling[i,j] * sum_{j'} dY[j,j'] * coupling[i',j'])
            # = -2 * sum_{i,i'} dX[i,i'] * coupling[i,:] @ dY @ coupling[i',:]^T
            # = -2 * trace(dX^T @ (coupling @ dY @ coupling^T))
            # = -2 * trace((coupling @ dY @ coupling^T)^T @ dX)
            # = -2 * <coupling @ dY @ coupling^T, dX>
            cross_term = backend.matmul(backend.matmul(coupling, target_distances), backend.transpose(coupling))
            term2 = -2.0 * backend.sum(cross_term * source_distances)

            backend.eval(weighted_src, weighted_tgt, term2)
            obj = float(backend.to_numpy(weighted_src)) + float(backend.to_numpy(weighted_tgt)) + float(backend.to_numpy(term2))
        else:
            # Absolute loss - expensive 4D computation
            n = source_distances.shape[0]
            m = target_distances.shape[0]

            src_exp = backend.reshape(source_distances, (n, 1, n, 1))
            tgt_exp = backend.reshape(target_distances, (1, m, 1, m))
            diff = backend.abs(src_exp - tgt_exp)

            coupling_exp1 = backend.reshape(coupling, (n, m, 1, 1))
            coupling_exp2 = backend.reshape(coupling, (1, 1, n, m))

            obj_arr = backend.sum(diff * coupling_exp1 * coupling_exp2)
            backend.eval(obj_arr)
            obj = float(backend.to_numpy(obj_arr))

        return obj

    def _sinkhorn_step(
        self,
        cost: "Array",
        epsilon: float,
        max_iterations: int,
        convergence_threshold: float | None = None,
    ) -> "Array":
        """
        Sinkhorn-Knopp algorithm for entropic optimal transport.

        GPU-accelerated implementation using vectorized operations.

        Args:
            cost: Cost matrix C of shape (n, m)
            epsilon: Entropic regularization (higher = more diffuse coupling)
            max_iterations: Maximum Sinkhorn iterations
            convergence_threshold: Stop when max(|u_new - u|, |v_new - v|) < threshold

        Returns:
            Transport plan γ of shape (n, m) with uniform marginals
        """
        backend = self._backend
        n = cost.shape[0]
        m = cost.shape[1]

        if n == 0 or m == 0:
            return backend.zeros((n, m))

        safe_epsilon = max(epsilon, 1e-6)

        # Compute Gibbs kernel with row-wise stabilization
        # K[i,j] = exp(-(C[i,j] - row_min[i]) / epsilon)
        row_min = backend.min(cost, axis=1, keepdims=True)
        centered_cost = cost - row_min
        exponent = -centered_cost / safe_epsilon
        # Clamp to avoid underflow
        exponent = backend.maximum(exponent, backend.full(exponent.shape, -80.0))
        kernel = backend.exp(exponent)
        # Floor to avoid numerical issues
        kernel = backend.maximum(kernel, backend.full(kernel.shape, 1e-20))

        # Initialize dual variables
        u = backend.ones((n,))
        v = backend.ones((m,))

        # Uniform marginals
        mu = 1.0 / n
        nu = 1.0 / m

        for _ in range(max_iterations):
            # Row scaling: u = mu / (K @ v)
            kv = backend.matmul(kernel, v)
            kv = backend.maximum(kv, backend.full(kv.shape, 1e-10))
            u_new = mu / kv

            # Column scaling: v = nu / (K^T @ u)
            ktu = backend.matmul(backend.transpose(kernel), u_new)
            ktu = backend.maximum(ktu, backend.full(ktu.shape, 1e-10))
            v_new = nu / ktu

            if convergence_threshold is not None and convergence_threshold > 0:
                u_diff = backend.max(backend.abs(u_new - u))
                v_diff = backend.max(backend.abs(v_new - v))
                backend.eval(u_diff, v_diff)
                max_delta = max(float(backend.to_numpy(u_diff)), float(backend.to_numpy(v_diff)))
                if max_delta < convergence_threshold:
                    u = u_new
                    v = v_new
                    break

            u = u_new
            v = v_new

        # Recover transport plan: gamma = diag(u) @ K @ diag(v)
        # Efficiently: gamma[i,j] = u[i] * K[i,j] * v[j]
        plan = kernel * backend.reshape(u, (n, 1)) * backend.reshape(v, (1, m))

        return plan


# Convenience function for backward compatibility
def compute_gromov_wasserstein(
    source_points: "Array",
    target_points: "Array",
    config: Config = Config(),
    backend: "Backend | None" = None,
) -> Result:
    """
    Compute Gromov-Wasserstein distance between point sets.

    Convenience function that computes pairwise distances and then GW distance.

    Args:
        source_points: Source point matrix [n, d]
        target_points: Target point matrix [m, d]
        config: Algorithm configuration
        backend: Backend protocol implementation. If None, uses default.

    Returns:
        Result with distance, coupling, and convergence info
    """
    if backend is None:
        backend = get_default_backend()

    gw = GromovWassersteinDistance(backend)
    source_dist = gw.compute_pairwise_distances(source_points)
    target_dist = gw.compute_pairwise_distances(target_points)

    return gw.compute(source_dist, target_dist, config)
