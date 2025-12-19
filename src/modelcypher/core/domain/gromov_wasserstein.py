from __future__ import annotations

from dataclasses import dataclass
import math


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

        if n == m and GromovWassersteinDistance._equivalent_matrices(source_distances, target_distances, 1e-6):
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

    @staticmethod
    def _equivalent_matrices(lhs: list[list[float]], rhs: list[list[float]], tolerance: float) -> bool:
        if len(lhs) != len(rhs):
            return False
        for row_idx, row in enumerate(lhs):
            if len(row) != len(rhs[row_idx]):
                return False
            for col_idx, value in enumerate(row):
                if abs(value - rhs[row_idx][col_idx]) > tolerance:
                    return False
        return True

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
        cost: list[list[float]],
        epsilon: float,
        max_iterations: int,
        convergence_threshold: float | None = None,
    ) -> list[list[float]]:
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
