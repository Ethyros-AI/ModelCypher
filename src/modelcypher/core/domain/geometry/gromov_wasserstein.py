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
        source_array = np.asarray(source_distances, dtype=np.float64)
        target_array = np.asarray(target_distances, dtype=np.float64)
        n = int(source_array.shape[0])
        m = int(target_array.shape[0])
        if n == 0 or m == 0:
            return Result(distance=float("inf"), coupling=[], converged=False, iterations=0)

        if n == m and GromovWassersteinDistance._equivalent_matrices(
            source_array,
            target_array,
            1e-6,
        ):
            coupling = [[0.0 for _ in range(m)] for _ in range(n)]
            weight = 1.0 / float(n)
            for i in range(n):
                coupling[i][i] = weight
            return Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        coupling = np.full((n, m), 1.0 / float(n * m), dtype=np.float64)

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
                source_array,
                target_array,
                coupling,
                config.use_squared_loss,
            )
            new_coupling = GromovWassersteinDistance._sinkhorn_step(
                cost,
                epsilon=epsilon,
                max_iterations=config.max_inner_iterations,
                convergence_threshold=config.convergence_threshold,
            )

            diff = np.abs(new_coupling - coupling)
            max_change = float(np.max(diff)) if diff.size else 0.0
            coupling = new_coupling

            distance = GromovWassersteinDistance._compute_objective(
                source_array,
                target_array,
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
            source_array,
            target_array,
            coupling,
            config.use_squared_loss,
        )

        return Result(
            distance=float(final_distance),
            coupling=coupling.tolist(),
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
        if use_squared_loss:
            source = np.asarray(source_distances, dtype=np.float64)
            target = np.asarray(target_distances, dtype=np.float64)
            plan = np.asarray(coupling, dtype=np.float64)
            row_mass = plan.sum(axis=1)
            col_mass = plan.sum(axis=0)
            source_sq = source * source
            target_sq = target * target
            const_source = source_sq @ row_mass
            const_target = target_sq @ col_mass
            interaction = source @ plan @ target.T
            cost = const_source[:, None] + const_target[None, :] - 2.0 * interaction
            return cost

        n = len(source_distances)
        m = len(target_distances)
        cost = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                total = 0.0
                for ip in range(n):
                    for jp in range(m):
                        diff = source_distances[i][ip] - target_distances[j][jp]
                        total += abs(diff) * coupling[ip][jp]
                cost[i][j] = total
        return cost

    @staticmethod
    def _compute_objective(
        source_distances: list[list[float]],
        target_distances: list[list[float]],
        coupling: list[list[float]],
        use_squared_loss: bool,
    ) -> float:
        if use_squared_loss:
            cost = GromovWassersteinDistance._compute_cost(
                source_distances,
                target_distances,
                coupling,
                use_squared_loss,
            )
            plan = np.asarray(coupling, dtype=np.float64)
            return float(np.sum(cost * plan))

        n = len(source_distances)
        m = len(target_distances)
        total = 0.0
        for i in range(n):
            for j in range(m):
                for ip in range(n):
                    for jp in range(m):
                        diff = source_distances[i][ip] - target_distances[j][jp]
                        total += abs(diff) * coupling[i][j] * coupling[ip][jp]
        return total

    @staticmethod
    def _sinkhorn_step(
        cost: list[list[float]] | np.ndarray,
        epsilon: float,
        max_iterations: int,
        convergence_threshold: float | None = None,
    ) -> np.ndarray:
        cost_array = np.asarray(cost, dtype=np.float64)
        n = int(cost_array.shape[0])
        m = int(cost_array.shape[1]) if cost_array.ndim == 2 else 0
        if n == 0 or m == 0:
            return []

        safe_epsilon = max(epsilon, 1e-6)
        row_min = np.min(cost_array, axis=1, keepdims=True)
        row_min = np.where(np.isfinite(row_min), row_min, 0.0)
        exponent = -(cost_array - row_min) / safe_epsilon
        exponent = np.maximum(exponent, -80.0)
        kernel = np.exp(exponent)
        kernel = np.maximum(kernel, 1e-20)

        u = np.ones((n,), dtype=np.float64)
        v = np.ones((m,), dtype=np.float64)
        mu = 1.0 / float(n)
        nu = 1.0 / float(m)

        for _ in range(max_iterations):
            denom_u = kernel @ v
            denom_u = np.maximum(denom_u, 1e-10)
            new_u = mu / denom_u
            max_delta = float(np.max(np.abs(new_u - u)))
            u = new_u

            denom_v = kernel.T @ u
            denom_v = np.maximum(denom_v, 1e-10)
            new_v = nu / denom_v
            max_delta = max(max_delta, float(np.max(np.abs(new_v - v))))
            v = new_v

            if convergence_threshold is not None and convergence_threshold > 0 and max_delta < convergence_threshold:
                break

        plan = (u[:, None] * kernel) * v[None, :]
        return plan
