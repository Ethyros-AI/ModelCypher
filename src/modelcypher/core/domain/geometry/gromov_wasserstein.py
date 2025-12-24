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

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Result:
    distance: float
    coupling: list[list[float]]
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
    @staticmethod
    def compute(
        source_distances: list[list[float]],
        target_distances: list[list[float]],
        config: Config = Config(),
    ) -> Result:
        n = len(source_distances)
        m = len(target_distances)
        if n == 0 or m == 0:
            return Result(distance=float("inf"), coupling=[], converged=False, iterations=0)

        if n == m and GromovWassersteinDistance._equivalent_matrices(
            source_distances,
            target_distances,
            1e-6,
        ):
            coupling = [[0.0 for _ in range(m)] for _ in range(n)]
            weight = 1.0 / float(n)
            for i in range(n):
                coupling[i][i] = weight
            return Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        coupling = [[1.0 / float(n * m) for _ in range(m)] for _ in range(n)]

        converged = False
        iterations = 0
        prev_distance = float("inf")

        min_outer = max(1, min(config.min_outer_iterations, config.max_outer_iterations))
        decay = min(max(config.epsilon_decay, 0.0), 1.0)
        min_epsilon = max(1e-6, min(config.epsilon_min, config.epsilon))
        epsilon = max(config.epsilon, min_epsilon)

        for outer in range(config.max_outer_iterations):
            iterations = outer + 1
            cost = GromovWassersteinDistance._compute_cost(
                source_distances,
                target_distances,
                coupling,
                config.use_squared_loss,
            )
            new_coupling = GromovWassersteinDistance._sinkhorn_step(
                cost,
                epsilon=epsilon,
                max_iterations=config.max_inner_iterations,
                convergence_threshold=config.convergence_threshold,
            )

            max_change = 0.0
            for i in range(n):
                for j in range(m):
                    delta = abs(new_coupling[i][j] - coupling[i][j])
                    if delta > max_change:
                        max_change = delta
            coupling = new_coupling

            distance = GromovWassersteinDistance._compute_objective(
                source_distances,
                target_distances,
                coupling,
                config.use_squared_loss,
            )

            objective_change = abs(distance - prev_distance) if math.isfinite(prev_distance) else float("inf")
            relative_change = (
                objective_change / max(abs(prev_distance), 1e-8) if math.isfinite(prev_distance) else float("inf")
            )

            meets_coupling = max_change < config.convergence_threshold
            meets_objective = objective_change < config.convergence_threshold or relative_change < config.relative_objective_threshold

            if iterations >= min_outer and (meets_coupling or meets_objective):
                converged = True
                break
            prev_distance = distance
            epsilon = max(min_epsilon, epsilon * decay)

        final_distance = GromovWassersteinDistance._compute_objective(
            source_distances,
            target_distances,
            coupling,
            config.use_squared_loss,
        )

        return Result(
            distance=float(final_distance),
            coupling=coupling,
            converged=converged,
            iterations=iterations,
        )

    @staticmethod
    def compute_pairwise_distances(points: list[list[float]]) -> list[list[float]]:
        n = len(points)
        if n == 0:
            return []
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2:
            distances = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    sum_sq = 0.0
                    min_len = min(len(points[i]), len(points[j]))
                    for k in range(min_len):
                        diff = points[i][k] - points[j][k]
                        sum_sq += diff * diff
                    if len(points[i]) > min_len:
                        for k in range(min_len, len(points[i])):
                            diff = points[i][k]
                            sum_sq += diff * diff
                    if len(points[j]) > min_len:
                        for k in range(min_len, len(points[j])):
                            diff = points[j][k]
                            sum_sq += diff * diff
                    dist = math.sqrt(sum_sq)
                    distances[i][j] = dist
                    distances[j][i] = dist
            return distances
        norms = np.sum(array * array, axis=1, keepdims=True)
        dist_sq = norms + norms.T - 2.0 * (array @ array.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        return dist.tolist()

    @staticmethod
    def _equivalent_matrices(lhs: list[list[float]], rhs: list[list[float]], tolerance: float) -> bool:
        lhs_array = np.asarray(lhs, dtype=np.float64)
        rhs_array = np.asarray(rhs, dtype=np.float64)
        if lhs_array.shape != rhs_array.shape:
            return False
        return bool(np.all(np.abs(lhs_array - rhs_array) <= tolerance))

    @staticmethod
    def _compute_cost(
        source_distances: list[list[float]],
        target_distances: list[list[float]],
        coupling: list[list[float]],
        use_squared_loss: bool,
    ) -> list[list[float]]:
        n = len(source_distances)
        m = len(target_distances)
        cost = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                total = 0.0
                for ip in range(n):
                    for jp in range(m):
                        diff = source_distances[i][ip] - target_distances[j][jp]
                        loss = diff * diff if use_squared_loss else abs(diff)
                        total += loss * coupling[ip][jp]
                cost[i][j] = total
        return cost

    @staticmethod
    def _compute_objective(
        source_distances: list[list[float]],
        target_distances: list[list[float]],
        coupling: list[list[float]],
        use_squared_loss: bool,
    ) -> float:
        n = len(source_distances)
        m = len(target_distances)
        total = 0.0
        for i in range(n):
            for j in range(m):
                for ip in range(n):
                    for jp in range(m):
                        diff = source_distances[i][ip] - target_distances[j][jp]
                        loss = diff * diff if use_squared_loss else abs(diff)
                        total += loss * coupling[i][j] * coupling[ip][jp]
        return total

    @staticmethod
    def _sinkhorn_step(
        cost: list[list[float]] | np.ndarray,
        epsilon: float,
        max_iterations: int,
        convergence_threshold: float | None = None,
    ) -> list[list[float]]:
        """
        Sinkhorn-Knopp algorithm for entropic optimal transport.

        Computes the optimal coupling given a cost matrix and entropic
        regularization parameter epsilon. The algorithm alternates between
        scaling rows and columns to satisfy marginal constraints.

        Algorithm:
            1. Compute Gibbs kernel: K[i,j] = exp(-C[i,j] / ε)
            2. Initialize dual variables: u = 1, v = 1
            3. Iterate until convergence:
               - Row scaling: u = μ / (K @ v)
               - Column scaling: v = ν / (K.T @ u)
            4. Recover coupling: γ = diag(u) @ K @ diag(v)

        Numerical Stability:
            - Row-wise centering of cost matrix before exponentiation
            - Clamped exponents to avoid underflow (min -80)
            - Floor on kernel values (1e-20) and denominators (1e-10)

        Args:
            cost: Cost matrix C of shape (n, m)
            epsilon: Entropic regularization (higher = more diffuse coupling)
            max_iterations: Maximum Sinkhorn iterations
            convergence_threshold: Stop when max(|u_new - u|, |v_new - v|) < threshold

        Returns:
            Transport plan γ of shape (n, m) with uniform marginals

        Complexity:
            O(nm) per iteration, typically 50-100 iterations for convergence.
        """
        n = len(cost)
        m = len(cost[0]) if cost else 0
        if n == 0 or m == 0:
            return []

        safe_epsilon = max(epsilon, 1e-6)
        kernel = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            row_min = min(cost[i]) if cost[i] else 0.0
            if not math.isfinite(row_min):
                row_min = 0.0
            for j in range(m):
                exponent = -(cost[i][j] - row_min) / safe_epsilon
                clamped = max(exponent, -80.0)
                value = math.exp(clamped)
                kernel[i][j] = max(value, 1e-20)

        u = [1.0 for _ in range(n)]
        v = [1.0 for _ in range(m)]
        mu = 1.0 / float(n)
        nu = 1.0 / float(m)

        for _ in range(max_iterations):
            max_delta = 0.0
            for i in range(n):
                denom = 0.0
                for j in range(m):
                    denom += kernel[i][j] * v[j]
                new_u = mu / max(denom, 1e-10)
                max_delta = max(max_delta, abs(new_u - u[i]))
                u[i] = new_u

            for j in range(m):
                denom = 0.0
                for i in range(n):
                    denom += kernel[i][j] * u[i]
                new_v = nu / max(denom, 1e-10)
                max_delta = max(max_delta, abs(new_v - v[j]))
                v[j] = new_v

            if convergence_threshold is not None and convergence_threshold > 0 and max_delta < convergence_threshold:
                break

        plan = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                plan[i][j] = u[i] * kernel[i][j] * v[j]
        return plan
