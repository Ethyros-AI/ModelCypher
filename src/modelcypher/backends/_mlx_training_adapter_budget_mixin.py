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

"""Budget and gradient measurement helpers for MLX training."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from modelcypher.core.domain.training.mass_step_size import ControllerLayerMeasurement
from modelcypher.core.domain.training.spectral_budget import compute_pissa_budget_ratios

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_optimizer import OptimizerGeometryConfig

logger = logging.getLogger(__name__)


class _MLXTrainingAdapterBudgetMixin:
    def _structural_sigma_budget_is_enforceable(
        self,
        *,
        use_pissa_lora: bool,
    ) -> bool:
        """Return whether structural ``sigma_k`` can gate control decisions.

        PiSSA trains a low-rank displacement on top of an exact-reconstruction
        initialization. The stored ``sigma_k`` is measured at the Shannon
        structural boundary, which can sit far outside the adapter-rank block.
        In that regime ``||ΔW||₂ / sigma_k`` is still a useful structural
        diagnostic, but not an enforceable behavioral or controller budget.
        """
        return not use_pissa_lora

    def _seed_remaining_budget(
        self,
        *,
        use_pissa_lora: bool,
        use_mass_step_sizing: bool,
        sigma_k_min: float,
    ) -> float | None:
        """Seed conformal remaining budget when a live margin exists.

        For PiSSA exact-reconstruction adapters we do not seed a running margin:
        the structural ``sigma_k`` boundary is diagnostic-only when the adapter
        rank is much smaller than the Shannon structural rank. NB-LoRA enforces
        its scale bound by construction, so there is no separate seeded margin.
        """
        if not use_mass_step_sizing or sigma_k_min <= 0.0:
            return None
        if use_pissa_lora:
            return None
        return None

    def _reanchor_pissa_budget(
        self,
        model: Any,
        sigma_k_min: float,
    ) -> float | None:
        """Measure PiSSA structural displacement against the stored ``sigma_k``.

        Runs power iteration on the implicit displacement operator
        ``D = a_curr @ b_curr - a_init @ b_init`` for every PiSSA layer to get
        the true spectral norm of cumulative displacement.  Returns
        ``sigma_k_min * (1 - max_ratio)`` clamped to ``[0, sigma_k_min]``.

        This remains useful for structural postmortems even when the resulting
        ratio is not enforced as a controller or stopping budget.
        """
        init_factors = getattr(self, '_pissa_init_factors', {})
        per_layer_sk = getattr(self, '_pissa_per_layer_sigma_k', {})
        if not init_factors:
            return None

        pissa_products: list[tuple[float, Any, Any, Any, Any, float]] = []
        pissa_module_names: list[str] = []
        for name, lora in self._iter_pissa_lora_modules(model):
            if name not in init_factors or name not in per_layer_sk:
                continue
            a_init, b_init = init_factors[name]
            pissa_products.append((
                float(lora.scale),
                lora.lora_a, lora.lora_b,
                a_init, b_init,
                per_layer_sk[name],
            ))
            pissa_module_names.append(name)
            mx.eval(lora.lora_a, lora.lora_b)

        if not pissa_products:
            return None

        ratios = compute_pissa_budget_ratios(pissa_products, self._backend)
        if not ratios:
            return None

        max_ratio = max(ratios)
        if len(ratios) == len(pissa_module_names):
            worst_idx = ratios.index(max_ratio)
            worst_name = pissa_module_names[worst_idx]
            worst_sk = pissa_products[worst_idx][5]
            rank = pissa_products[worst_idx][1].shape[-1]  # lora_a shape
            logger.info(
                "PiSSA budget bottleneck: %s ratio=%.4f sigma_k=%.4f rank=%d",
                worst_name, max_ratio, worst_sk, rank,
            )
            if not getattr(self, '_pissa_full_dump_done', False):
                self._pissa_full_dump_done = True
                for i, (name, ratio) in enumerate(
                    zip(pissa_module_names, ratios)
                ):
                    sk = pissa_products[i][5]
                    r = pissa_products[i][1].shape[-1]
                    logger.info(
                        "  [budget] %s: ratio=%.4f sigma_k=%.4f rank=%d remaining=%.4f",
                        name, ratio, sk, r, max(0.0, sk * (1.0 - ratio)),
                    )
        return max(0.0, sigma_k_min * (1.0 - max_ratio))

    def _measure_pissa_effective_update_norm(
        self,
        model,
        update_direction: dict[str, Any],
    ) -> float | None:
        """Measure first-order PiSSA update norm in weight space.

        Thin wrapper over ``_measure_pissa_effective_update_norm_sq_expression``
        for call sites that need a realized Python float.
        """
        total_norm_sq = self._measure_pissa_effective_update_norm_sq_expression(
            model,
            update_direction,
        )
        if total_norm_sq is None:
            return None

        mx.eval(total_norm_sq)
        total_norm_sq_val = (
            float(total_norm_sq.item())
            if hasattr(total_norm_sq, "item")
            else float(total_norm_sq)
        )
        total_norm = math.sqrt(max(total_norm_sq_val, 0.0))
        if not math.isfinite(total_norm) or total_norm <= 0.0:
            return None
        return total_norm

    def _measure_pissa_effective_update_norm_sq_expression(
        self,
        model,
        update_direction: dict[str, Any],
    ) -> Any | None:
        """Build the induced PiSSA weight-step Frobenius norm squared lazily.

        MASS uses ``eta_weyl = sigma_k / ||D||`` to bound spectral displacement.
        For PiSSA, the trainable coordinates are factor matrices ``(A, B)``, but
        Weyl applies to the induced weight update:

            D_W = scale * (dA @ B + A @ dB)

        The factor-space norm can dramatically understate the real displacement
        when ``A`` and ``B`` already encode the model's principal directions.
        This helper returns the unevaluated Frobenius norm squared of the
        induced update across all active PiSSA layers so callers can batch the
        realization with other scalar diagnostics.
        """
        total_norm_sq_terms: list[Any] = []

        for layer_key, lora_module in self._iter_pissa_lora_modules(model):
            prefix = layer_key.removesuffix(".weight")
            d_a = update_direction.get(prefix + ".lora_a")
            d_b = update_direction.get(prefix + ".lora_b")
            if d_a is None and d_b is None:
                continue

            induced = None
            if d_a is not None:
                induced = mx.matmul(
                    d_a.astype(mx.float32),
                    lora_module.lora_b.astype(mx.float32),
                )
            if d_b is not None:
                a_db = mx.matmul(
                    lora_module.lora_a.astype(mx.float32),
                    d_b.astype(mx.float32),
                )
                induced = a_db if induced is None else induced + a_db
            if induced is None:
                continue

            induced = float(lora_module.scale) * induced
            total_norm_sq_terms.append(mx.sum(induced * induced))

        if not total_norm_sq_terms:
            return None

        total_norm_sq = total_norm_sq_terms[0]
        for term in total_norm_sq_terms[1:]:
            total_norm_sq = total_norm_sq + term
        return total_norm_sq

    def _batched_layer_measurements_from_gradient(
        self,
        *,
        model: Any,
        grad_map: dict[str, Any],
        step_learning_rate: float,
        use_pissa_lora: bool,
        opt_config: "OptimizerGeometryConfig | None",
    ) -> dict[str, ControllerLayerMeasurement] | None:
        """Build per-layer controller measurements with one scalar sync."""
        if not grad_map:
            return None

        layer_inputs: list[tuple[str, Any, float | None, float | None]] = []
        if use_pissa_lora:
            lora_iter = self._iter_pissa_lora_modules(model)
            param_suffixes = (".lora_a", ".lora_b")
        else:
            lora_iter = self._iter_nb_lora_modules(model)
            param_suffixes = (".A_tilde", ".B_tilde", ".S_raw")

        for layer_key, lora_module in lora_iter:
            prefix = layer_key.removesuffix(".weight")
            grad_norm_sq_terms: list[Any] = []
            for suffix in param_suffixes:
                grad_array = grad_map.get(prefix + suffix)
                if grad_array is None or grad_array.size == 0:
                    continue
                grad_norm_sq_terms.append(mx.sum(grad_array * grad_array))
            if not grad_norm_sq_terms:
                continue

            grad_norm_sq = grad_norm_sq_terms[0]
            for term in grad_norm_sq_terms[1:]:
                grad_norm_sq = grad_norm_sq + term

            decay_scale = None
            if opt_config is not None and layer_key in opt_config.layer_configs:
                decay_scale = float(opt_config.layer_configs[layer_key].decay_scale)
            scale_bound_val = None if use_pissa_lora else float(lora_module._scale_bound)
            layer_inputs.append((layer_key, grad_norm_sq, decay_scale, scale_bound_val))

        if not layer_inputs:
            return None

        mx.eval(*[grad_norm_sq for _, grad_norm_sq, _, _ in layer_inputs])

        per_layer: dict[str, ControllerLayerMeasurement] = {}
        layer_norms: dict[str, float] = {}
        total_norm_sq = 0.0

        for layer_key, grad_norm_sq, decay_scale, scale_bound_val in layer_inputs:
            grad_norm_sq_val = (
                float(grad_norm_sq.item())
                if hasattr(grad_norm_sq, "item")
                else float(grad_norm_sq)
            )
            grad_norm_sq_val = max(grad_norm_sq_val, 0.0)
            grad_norm = math.sqrt(grad_norm_sq_val)
            layer_norms[layer_key] = grad_norm
            total_norm_sq += grad_norm_sq_val
            per_layer[layer_key] = ControllerLayerMeasurement(
                parameter_update_norm=step_learning_rate * grad_norm,
                total_step_fraction=None,
                decay_scale=decay_scale,
                scale_bound=scale_bound_val,
                step_learning_rate=step_learning_rate,
            )

        total_norm = math.sqrt(total_norm_sq)
        if total_norm > 0.0:
            for layer_key, layer_norm in layer_norms.items():
                current = per_layer[layer_key]
                per_layer[layer_key] = ControllerLayerMeasurement(
                    parameter_update_norm=current.parameter_update_norm,
                    behavioral_transport_norm=current.behavioral_transport_norm,
                    spectral_budget_ratio=current.spectral_budget_ratio,
                    remaining_budget=current.remaining_budget,
                    total_step_fraction=layer_norm / total_norm,
                    decay_scale=current.decay_scale,
                    scale_bound=current.scale_bound,
                    step_learning_rate=current.step_learning_rate,
                )
        return per_layer


