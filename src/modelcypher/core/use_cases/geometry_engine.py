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

from modelcypher.core.domain.geometry import DoRADecomposition
from modelcypher.core.domain.geometry.backend_matrix_utils import BackendMatrixUtils
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_finite,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.optimal_transport import (
    SinkhornResult,
    SinkhornSolver,
)
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


class GeometryEngine:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._matrix_utils = BackendMatrixUtils(backend)

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

        parameter_squared_sum = self.backend.zeros((), dtype="float32")
        step_squared_sum = self.backend.zeros((), dtype="float32")
        has_step_delta = False

        for key, parameter in trainable_parameters.items():
            # Convert to backend array if needed
            parameter = self.backend.array(parameter)
            fp32 = self.backend.astype(parameter, "float32")
            parameter_squared_sum = parameter_squared_sum + self.backend.sum(fp32 * fp32)
            if previous_trainable_parameters and key in previous_trainable_parameters:
                prev_param = self.backend.array(previous_trainable_parameters[key])
                prev = self.backend.astype(prev_param, "float32")
                delta = fp32 - prev
                step_squared_sum = step_squared_sum + self.backend.sum(delta * delta)
                has_step_delta = True

        parameter_l2_tensor = self.backend.sqrt(parameter_squared_sum)
        step_l2_tensor = self.backend.sqrt(step_squared_sum) if has_step_delta else None

        weight_update_fro_tensor = None
        if scale and is_finite(scale, self.backend) and scale > 0:
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
        # Convert inputs to backend arrays if needed
        source_anchors = self.backend.array(source_anchors)
        target_anchors = self.backend.array(target_anchors)
        source_basis = self.backend.array(source_basis)
        target_basis = self.backend.array(target_basis)

        z_source = self.backend.matmul(source_anchors, source_basis)
        z_target = self.backend.matmul(target_anchors, target_basis)
        self.backend.eval(z_source, z_target)

        if anchor_weights and len(anchor_weights) == int(z_source.shape[0]):
            weights_arr = self.backend.array(anchor_weights, dtype="float32")
            sqrt_weights = self.backend.sqrt(weights_arr)
            sqrt_weights = self.backend.reshape(sqrt_weights, (len(anchor_weights), 1))
            z_source = z_source * sqrt_weights
            z_target = z_target * sqrt_weights
            self.backend.eval(z_source, z_target)

        procrustes = self._matrix_utils.procrustes_rotation(z_source, z_target)
        omega = procrustes.rotation
        self.backend.eval(omega)

        diff = self.backend.matmul(z_source, omega) - z_target
        rss = self.backend.sqrt(self.backend.sum(diff * diff))
        denom = self.backend.sqrt(self.backend.sum(z_target * z_target))
        self.backend.eval(omega, rss, denom)

        rss_value = float(self._item(rss))
        denom_value = float(self._item(denom))
        if not is_finite(rss_value, self.backend) or not is_finite(denom_value, self.backend) or denom_value <= 0:
            raise ValueError("Non-finite Procrustes residuals")

        error = rss_value / denom_value
        if not is_finite(error, self.backend):
            raise ValueError("Non-finite Procrustes relative error")

        return ProcrustesResult(omega=omega, error=error)

    def soft_procrustes_alignment(
        self,
        source_anchors: Array,
        target_anchors: Array,
        source_basis: Array,
        target_basis: Array,
    ) -> tuple[Array, Array, float, SinkhornResult]:
        """Compute soft Procrustes alignment using optimal transport.

        Projects anchors into basis space, computes geodesic cost matrix,
        solves optimal transport, and finds the orthogonal rotation.
        All parameters are derived from the data - no configuration needed.

        Args:
            source_anchors: Source anchor points
            target_anchors: Target anchor points
            source_basis: Source projection basis
            target_basis: Target projection basis

        Returns:
            Tuple of (rotation_matrix, transport_plan, alignment_error, sinkhorn_result)
        """
        solver = SinkhornSolver(self.backend)

        z_source = self.backend.matmul(source_anchors, source_basis)
        z_target = self.backend.matmul(target_anchors, target_basis)
        self.backend.eval(z_source, z_target)

        cost_matrix = self._geodesic_cost_matrix(z_source, z_target)
        sinkhorn_result = solver.solve(cost_matrix)

        transported_mass = self.backend.matmul(sinkhorn_result.plan, z_target)
        row_sums = self.backend.sum(sinkhorn_result.plan, axis=1, keepdims=True)
        stability_eps = division_epsilon(self.backend, row_sums)
        stabilized = self.backend.maximum(row_sums, self.backend.array(stability_eps))
        transported_target = transported_mass / stabilized
        self.backend.eval(transported_target)

        procrustes = self._matrix_utils.procrustes_rotation(z_source, transported_target)
        omega = procrustes.rotation
        self.backend.eval(omega)

        aligned = self.backend.matmul(z_source, omega)
        diff = aligned - transported_target
        rss = self.backend.sqrt(self.backend.sum(diff * diff))
        denom = self.backend.sqrt(self.backend.sum(transported_target * transported_target))
        self.backend.eval(rss, denom)
        rss_value = float(self._item(rss))
        denom_value = float(self._item(denom))
        if not is_finite(rss_value, self.backend) or not is_finite(denom_value, self.backend) or denom_value <= 0:
            raise ValueError("Non-finite soft Procrustes residuals")
        error = rss_value / denom_value

        return omega, sinkhorn_result.plan, error, sinkhorn_result

    def _geodesic_cost_matrix(self, source: Array, target: Array) -> Array:
        """Compute geodesic cost matrix for Sinkhorn alignment.

        Geodesic distance is the correct metric on curved manifolds; Euclidean
        costs are invalid for high-dimensional geometry.

        k-neighbors for the geodesic graph is derived from sqrt(n), which is
        the standard rule-of-thumb for k-NN on manifolds.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )
        combined = self.backend.concatenate([source, target], axis=0)
        combined = self.backend.astype(combined, "float32")

        # Derive k from data size: sqrt(n) is the standard rule for k-NN on manifolds
        total_points = int(combined.shape[0])
        k_neighbors = max(3, int(sqrt_scalar(float(total_points), self.backend)))

        geo = geodesic_distance_matrix(combined, k_neighbors=k_neighbors, backend=self.backend)
        self.backend.eval(geo)

        n = int(source.shape[0])
        m = int(target.shape[0])
        cost = geo[:n, n : n + m]

        # Squared cost for Wasserstein-2
        cost = cost * cost

        # Normalize for numerical stability
        max_val = self.backend.max(cost)
        stability_eps = division_epsilon(self.backend, cost)
        denom = self.backend.maximum(max_val, self.backend.array(stability_eps))
        cost = cost / denom

        self.backend.eval(cost)
        return cost

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

        squared_sum = self.backend.zeros((), dtype="float32")
        had_pairs = False

        for prefix, lora_a in lora_a_by_prefix.items():
            lora_b = lora_b_by_prefix.get(prefix)
            if lora_b is None:
                continue
            had_pairs = True
            # Convert to backend arrays if needed
            lora_a = self.backend.array(lora_a)
            lora_b = self.backend.array(lora_b)
            a = self.backend.astype(lora_a, "float32")
            b = self.backend.astype(lora_b, "float32")
            a_gram = self.backend.matmul(self.backend.transpose(a), a)
            b_gram = self.backend.matmul(b, self.backend.transpose(b))
            pair_squared = self.backend.sum(a_gram * b_gram)
            squared_sum = squared_sum + pair_squared

        if not had_pairs:
            return None

        return self.backend.array(scale, dtype="float32") * self.backend.sqrt(squared_sum)

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
        arr = self.backend.array(array)
        self.backend.eval(arr)
        return self.backend.to_scalar(arr)
