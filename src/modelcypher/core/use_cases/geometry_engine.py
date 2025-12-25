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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modelcypher.core.domain.geometry import DoRADecomposition
from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class LoRAAdapterGeometryMetrics:
    trainable_scalar_count: int
    parameter_l2: float
    step_l2: float | None
    weight_update_fro_norm: float | None


@dataclass(frozen=True)
class ProcrustesResult:
    omega: Array
    error: float


@dataclass(frozen=True)
class SinkhornResult:
    plan: Array
    converged: bool
    iterations: int
    marginal_error: float
    cost: float


class GeometryEngine:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def compute_lora_geometry(
        self,
        trainable_parameters: dict[str, Array],
        previous_trainable_parameters: dict[str, Array] | None,
        scale: float,
    ) -> LoRAAdapterGeometryMetrics:
        trainable_scalar_count = sum(int(param.size) for param in trainable_parameters.values())
        if not trainable_parameters:
            return LoRAAdapterGeometryMetrics(
                trainable_scalar_count=trainable_scalar_count,
                parameter_l2=0.0,
                step_l2=None,
                weight_update_fro_norm=None,
            )

        parameter_squared_sum = self.backend.zeros((), dtype=np.float32)
        step_squared_sum = self.backend.zeros((), dtype=np.float32)
        has_step_delta = False

        for key, parameter in trainable_parameters.items():
            fp32 = self.backend.astype(parameter, np.float32)
            parameter_squared_sum = parameter_squared_sum + self.backend.sum(fp32 * fp32)
            if previous_trainable_parameters and key in previous_trainable_parameters:
                prev = self.backend.astype(previous_trainable_parameters[key], np.float32)
                delta = fp32 - prev
                step_squared_sum = step_squared_sum + self.backend.sum(delta * delta)
                has_step_delta = True

        parameter_l2_tensor = self.backend.sqrt(parameter_squared_sum)
        step_l2_tensor = self.backend.sqrt(step_squared_sum) if has_step_delta else None

        weight_update_fro_tensor = None
        if scale and np.isfinite(scale) and scale > 0:
            weight_update_fro_tensor = self._weight_update_fro_norm(trainable_parameters, scale)

        eval_targets = [parameter_l2_tensor]
        if step_l2_tensor is not None:
            eval_targets.append(step_l2_tensor)
        if weight_update_fro_tensor is not None:
            eval_targets.append(weight_update_fro_tensor)
        self.backend.eval(*eval_targets)

        parameter_l2 = float(self._item(parameter_l2_tensor))
        step_l2 = float(self._item(step_l2_tensor)) if step_l2_tensor is not None else None
        weight_update_fro = (
            float(self._item(weight_update_fro_tensor))
            if weight_update_fro_tensor is not None
            else None
        )

        return LoRAAdapterGeometryMetrics(
            trainable_scalar_count=trainable_scalar_count,
            parameter_l2=parameter_l2,
            step_l2=step_l2,
            weight_update_fro_norm=weight_update_fro,
        )

    def orthogonal_procrustes(
        self,
        source_anchors: Array,
        target_anchors: Array,
        source_basis: Array,
        target_basis: Array,
        anchor_weights: list[float] | None = None,
    ) -> ProcrustesResult:
        z_source = self.backend.matmul(source_anchors, source_basis)
        z_target = self.backend.matmul(target_anchors, target_basis)
        self.backend.eval(z_source, z_target)

        if anchor_weights and len(anchor_weights) == int(z_source.shape[0]):
            sqrt_weights = self.backend.array(
                np.sqrt(np.array(anchor_weights, dtype=np.float32))
            ).reshape((len(anchor_weights), 1))
            z_source = z_source * sqrt_weights
            z_target = z_target * sqrt_weights
            self.backend.eval(z_source, z_target)

        m = self.backend.matmul(self.backend.transpose(z_source), z_target)
        self.backend.eval(m)

        m_cpu = self._to_numpy(m).astype(np.float32)
        u, _, vt = np.linalg.svd(m_cpu, full_matrices=False)
        omega_pre = u @ vt
        det = self._determinant_sign(omega_pre)
        if det < 0:
            u[:, -1] *= -1.0
        omega_cpu = u @ vt
        omega = self.backend.array(omega_cpu, dtype=np.float32)

        diff = self.backend.matmul(z_source, omega) - z_target
        rss = self.backend.sqrt(self.backend.sum(diff * diff))
        denom = self.backend.sqrt(self.backend.sum(z_target * z_target))
        self.backend.eval(omega, rss, denom)

        rss_value = float(self._item(rss))
        denom_value = float(self._item(denom))
        if not np.isfinite(rss_value) or not np.isfinite(denom_value) or denom_value <= 0:
            raise ValueError("Non-finite Procrustes residuals")

        error = rss_value / denom_value
        if not np.isfinite(error):
            raise ValueError("Non-finite Procrustes relative error")

        return ProcrustesResult(omega=omega, error=error)

    def soft_procrustes_alignment(
        self,
        source_anchors: Array,
        target_anchors: Array,
        source_basis: Array,
        target_basis: Array,
        config: SinkhornSolverConfig | None = None,
    ) -> tuple[Array, Array, float, SinkhornResult]:
        solver = SinkhornSolver(self.backend)
        if config is None:
            config = SinkhornSolverConfig()

        z_source = self.backend.matmul(source_anchors, source_basis)
        z_target = self.backend.matmul(target_anchors, target_basis)
        self.backend.eval(z_source, z_target)

        cost_matrix = solver.squared_euclidean_cost(z_source, z_target, normalize=True)
        sinkhorn_result = solver.solve(cost_matrix, config=config)

        transported_mass = self.backend.matmul(sinkhorn_result.plan, z_target)
        row_sums = self.backend.sum(sinkhorn_result.plan, axis=1, keepdims=True)
        stabilized = self.backend.maximum(row_sums, self.backend.array(config.stability_epsilon))
        transported_target = transported_mass / stabilized
        self.backend.eval(transported_target)

        m = self.backend.matmul(self.backend.transpose(z_source), transported_target)
        self.backend.eval(m)
        m_cpu = self._to_numpy(m).astype(np.float32)
        u, _, vt = np.linalg.svd(m_cpu, full_matrices=False)
        omega_pre = u @ vt
        det = self._determinant_sign(omega_pre)
        if det < 0:
            u[:, -1] *= -1.0
        omega_cpu = u @ vt
        omega = self.backend.array(omega_cpu, dtype=np.float32)
        self.backend.eval(omega)

        aligned = self.backend.matmul(z_source, omega)
        diff = aligned - transported_target
        rss = self.backend.sqrt(self.backend.sum(diff * diff))
        denom = self.backend.sqrt(self.backend.sum(transported_target * transported_target))
        self.backend.eval(rss, denom)
        rss_value = float(self._item(rss))
        denom_value = float(self._item(denom))
        if not np.isfinite(rss_value) or not np.isfinite(denom_value) or denom_value <= 0:
            raise ValueError("Non-finite soft Procrustes residuals")
        error = rss_value / denom_value

        return omega, sinkhorn_result.plan, error, sinkhorn_result

    @staticmethod
    def compute_dora(
        base_weights: dict[str, list[float]],
        current_weights: dict[str, list[float]],
    ):
        import mlx.core as mx

        base_mx = {k: mx.array(v) for k, v in base_weights.items()}
        current_mx = {k: mx.array(v) for k, v in current_weights.items()}
        decomposer = DoRADecomposition()
        return decomposer.analyze_adapter(base_mx, current_mx)

    def _weight_update_fro_norm(
        self, trainable_parameters: dict[str, Array], scale: float
    ) -> Array | None:
        lora_a_by_prefix: dict[str, Array] = {}
        lora_b_by_prefix: dict[str, Array] = {}

        for key, value in trainable_parameters.items():
            prefix, kind = self._lora_key_parts(key)
            if prefix is None:
                continue
            if kind == "a":
                lora_a_by_prefix[prefix] = value
            else:
                lora_b_by_prefix[prefix] = value

        if not lora_a_by_prefix or not lora_b_by_prefix:
            return None

        squared_sum = self.backend.zeros((), dtype=np.float32)
        had_pairs = False

        for prefix, lora_a in lora_a_by_prefix.items():
            lora_b = lora_b_by_prefix.get(prefix)
            if lora_b is None:
                continue
            had_pairs = True
            a = self.backend.astype(lora_a, np.float32)
            b = self.backend.astype(lora_b, np.float32)
            a_gram = self.backend.matmul(self.backend.transpose(a), a)
            b_gram = self.backend.matmul(b, self.backend.transpose(b))
            pair_squared = self.backend.sum(a_gram * b_gram)
            squared_sum = squared_sum + pair_squared

        if not had_pairs:
            return None

        return self.backend.array(scale, dtype=np.float32) * self.backend.sqrt(squared_sum)

    @staticmethod
    def _lora_key_parts(key: str) -> tuple[str | None, str | None]:
        if key.endswith(".lora_a"):
            return key[: -len(".lora_a")], "a"
        if key.endswith(".lora_b"):
            return key[: -len(".lora_b")], "b"
        if key.endswith("lora_a"):
            return key[: -len("lora_a")], "a"
        if key.endswith("lora_b"):
            return key[: -len("lora_b")], "b"
        return None, None

    def _item(self, array: Any) -> Any:
        if array is None:
            return None
        if hasattr(array, "item"):
            return array.item()
        return np.asarray(array).item()

    def _to_numpy(self, array: Any) -> np.ndarray:
        if hasattr(self.backend, "to_numpy"):
            return np.asarray(self.backend.to_numpy(array))
        return np.asarray(array)

    @staticmethod
    def _determinant_sign(matrix: np.ndarray) -> float:
        k = matrix.shape[0]
        if k == 0 or k != matrix.shape[1]:
            return 1.0
        if k == 1:
            return 1.0 if matrix[0, 0] >= 0 else -1.0

        a = matrix.astype(float).copy()
        sign = 1.0
        for i in range(k):
            max_row = i
            max_val = abs(a[i, i])
            for row in range(i + 1, k):
                val = abs(a[row, i])
                if val > max_val:
                    max_val = val
                    max_row = row
            if max_row != i:
                a[[i, max_row], :] = a[[max_row, i], :]
                sign = -sign
            pivot = a[i, i]
            if abs(pivot) < 1e-10:
                return 1.0
            for row in range(i + 1, k):
                factor = a[row, i] / pivot
                a[row, i:k] -= factor * a[i, i:k]

        diag_product = float(np.prod(np.diag(a)))
        return 1.0 if diag_product * sign >= 0 else -1.0


@dataclass(frozen=True)
class SinkhornSolverConfig:
    epsilon: float = 0.1
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_log_domain: bool = True
    stability_epsilon: float = 1e-9


class SinkhornSolver:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def solve(
        self,
        cost_matrix: Array,
        source_marginal: Array | None = None,
        target_marginal: Array | None = None,
        config: SinkhornSolverConfig = SinkhornSolverConfig(),
    ) -> SinkhornResult:
        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])
        mu = (
            source_marginal
            if source_marginal is not None
            else self.backend.ones((n,), dtype=np.float32) / float(n)
        )
        nu = (
            target_marginal
            if target_marginal is not None
            else self.backend.ones((m,), dtype=np.float32) / float(m)
        )
        self.backend.eval(mu, nu)

        if config.use_log_domain:
            return self._solve_log_domain(cost_matrix, mu, nu, config)
        return self._solve_standard(cost_matrix, mu, nu, config)

    def _solve_standard(
        self,
        cost_matrix: Array,
        mu: Array,
        nu: Array,
        config: SinkhornSolverConfig,
    ) -> SinkhornResult:
        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])
        K = self.backend.exp(-cost_matrix / config.epsilon)
        self.backend.eval(K)

        u = self.backend.ones((n,), dtype=np.float32)
        v = self.backend.ones((m,), dtype=np.float32)
        self.backend.eval(u, v)

        converged = False
        iterations = 0
        marginal_error = float("inf")

        for i in range(config.max_iterations):
            iterations = i + 1
            Kv = self.backend.matmul(K, v.reshape((m, 1))).reshape((n,))
            u_new = mu / self.backend.maximum(Kv, self.backend.array(config.stability_epsilon))
            KTu = self.backend.matmul(self.backend.transpose(K), u_new.reshape((n, 1))).reshape(
                (m,)
            )
            v_new = nu / self.backend.maximum(KTu, self.backend.array(config.stability_epsilon))
            Kv_new = self.backend.matmul(K, v_new.reshape((m, 1))).reshape((n,))
            row_marginal = u_new * Kv_new
            col_marginal = v_new * KTu
            row_error = self.backend.max(self.backend.abs(row_marginal - mu))
            col_error = self.backend.max(self.backend.abs(col_marginal - nu))
            self.backend.eval(row_error, col_error)
            marginal_error = max(float(self._item(row_error)), float(self._item(col_error)))
            u = u_new
            v = v_new
            if marginal_error < config.convergence_threshold:
                converged = True
                break

        plan = u.reshape((n, 1)) * K * v.reshape((1, m))
        self.backend.eval(plan)
        cost = self.backend.sum(cost_matrix * plan)
        self.backend.eval(cost)
        return SinkhornResult(
            plan=plan,
            converged=converged,
            iterations=iterations,
            marginal_error=marginal_error,
            cost=float(self._item(cost)),
        )

    def _solve_log_domain(
        self,
        cost_matrix: Array,
        mu: Array,
        nu: Array,
        config: SinkhornSolverConfig,
    ) -> SinkhornResult:
        n = int(cost_matrix.shape[0])
        m = int(cost_matrix.shape[1])
        log_mu = self.backend.log(
            self.backend.maximum(mu, self.backend.array(config.stability_epsilon))
        )
        log_nu = self.backend.log(
            self.backend.maximum(nu, self.backend.array(config.stability_epsilon))
        )
        logK = -cost_matrix / config.epsilon
        self.backend.eval(log_mu, log_nu, logK)

        f = self.backend.zeros((n,), dtype=np.float32)
        g = self.backend.zeros((m,), dtype=np.float32)
        self.backend.eval(f, g)

        converged = False
        iterations = 0
        marginal_error = float("inf")

        for i in range(config.max_iterations):
            iterations = i + 1
            logK_plus_g = logK + g.reshape((1, m))
            f_new = log_mu - self._logsumexp(logK_plus_g, axis=1)
            logKT_plus_f = self.backend.transpose(logK) + f_new.reshape((1, n))
            col_log_sum = self._logsumexp(logKT_plus_f, axis=1)
            g_new = log_nu - col_log_sum

            f_diff = self.backend.max(self.backend.abs(f_new - f))
            g_diff = self.backend.max(self.backend.abs(g_new - g))
            self.backend.eval(f_diff, g_diff)
            max_diff = max(float(self._item(f_diff)), float(self._item(g_diff)))

            logK_plus_g_new = logK + g_new.reshape((1, m))
            row_log_sum = self._logsumexp(logK_plus_g_new, axis=1)
            row_sums = self.backend.exp(f_new + row_log_sum)
            col_sums = self.backend.exp(g_new + col_log_sum)
            row_error = self.backend.max(self.backend.abs(row_sums - mu))
            col_error = self.backend.max(self.backend.abs(col_sums - nu))
            self.backend.eval(row_error, col_error)
            marginal_error = max(float(self._item(row_error)), float(self._item(col_error)))

            f = f_new
            g = g_new
            if (
                marginal_error < config.convergence_threshold
                or max_diff < config.convergence_threshold
            ):
                converged = True
                break

        logP = f.reshape((n, 1)) + logK + g.reshape((1, m))
        plan = self.backend.exp(logP)
        self.backend.eval(plan)

        row_sums = self.backend.sum(plan, axis=1)
        col_sums = self.backend.sum(plan, axis=0)
        row_error = self.backend.max(self.backend.abs(row_sums - mu))
        col_error = self.backend.max(self.backend.abs(col_sums - nu))
        self.backend.eval(row_error, col_error)
        marginal_error = max(float(self._item(row_error)), float(self._item(col_error)))

        cost = self.backend.sum(cost_matrix * plan)
        self.backend.eval(cost)

        return SinkhornResult(
            plan=plan,
            converged=converged,
            iterations=iterations,
            marginal_error=marginal_error,
            cost=float(self._item(cost)),
        )

    def squared_euclidean_cost(self, source: Array, target: Array, normalize: bool = True) -> Array:
        s = source
        t = target
        if normalize:
            s_norm = self.backend.sqrt(self.backend.sum(s * s, axis=1, keepdims=True) + 1e-8)
            t_norm = self.backend.sqrt(self.backend.sum(t * t, axis=1, keepdims=True) + 1e-8)
            s = s / s_norm
            t = t / t_norm
            self.backend.eval(s, t)
        s_norm_sq = self.backend.sum(s * s, axis=1, keepdims=True)
        t_norm_sq = self.backend.sum(t * t, axis=1, keepdims=True)
        inner = self.backend.matmul(s, self.backend.transpose(t))
        cost = s_norm_sq + self.backend.transpose(t_norm_sq) - 2 * inner
        clamped = self.backend.maximum(cost, self.backend.array(0.0))
        self.backend.eval(clamped)
        return clamped

    def cosine_cost(self, source: Array, target: Array) -> Array:
        s_norm = self.backend.sqrt(self.backend.sum(source * source, axis=1, keepdims=True) + 1e-8)
        t_norm = self.backend.sqrt(self.backend.sum(target * target, axis=1, keepdims=True) + 1e-8)
        s = source / s_norm
        t = target / t_norm
        similarity = self.backend.matmul(s, self.backend.transpose(t))
        cost = 1 - similarity
        clamped = self.backend.minimum(
            self.backend.maximum(cost, self.backend.array(0.0)), self.backend.array(2.0)
        )
        self.backend.eval(clamped)
        return clamped

    def _logsumexp(self, array: Array, axis: int) -> Array:
        max_val = self.backend.max(array, axis=axis, keepdims=True)
        shifted = array - max_val
        sum_exp = self.backend.sum(self.backend.exp(shifted), axis=axis)
        return self.backend.squeeze(max_val, axis=axis) + self.backend.log(sum_exp)

    def _item(self, array: Any) -> Any:
        if hasattr(array, "item"):
            return array.item()
        return np.asarray(array).item()
