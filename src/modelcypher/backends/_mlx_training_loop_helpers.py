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

"""Local helper computations for the MLX training loop."""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten as mlx_flatten
from mlx.utils import tree_unflatten as mlx_unflatten

from modelcypher.backends.mlx_training_adapter_core import (
    iterate_bilm_margin_batches,
    iterate_masked_batches,
    iterate_structured_batches,
    iterate_vl_batches,
)
from modelcypher.backends.mlx_training_adapter_metrics import EpochMetrics
from modelcypher.core.domain.training.mass_step_size import (
    _SQRT_EPS_F32,
    CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    CONTROLLER_MODE_BEHAVIORAL_PROBE,
    BehavioralStateMeasurement,
)

logger = logging.getLogger(__name__)


def build_objective_components(
    *,
    use_answer_mask: bool,
    use_constrained: bool,
    geometric_reshape: bool,
    entropy_regularization: bool,
    use_vl: bool,
    use_bilm_margin: bool,
) -> list[str]:
    components = ["ce_answer_masked" if use_answer_mask else "ce"]
    if use_constrained:
        components.append("constraint")
    if geometric_reshape:
        components.append("geometric_reshape")
    if entropy_regularization:
        components.append("entropy_regularization")
    if use_vl:
        components.append("vision_language")
    if use_bilm_margin:
        components.append("bilm_margin")
    return components


def _freeze_param_names_for_layer(
    layer_key: str,
    *,
    use_pissa_lora: bool,
) -> tuple[str, ...]:
    prefix = layer_key.removesuffix(".weight")
    if use_pissa_lora:
        return (prefix + ".lora_a", prefix + ".lora_b")
    return (prefix + ".A_tilde", prefix + ".B_tilde", prefix + ".S_raw")


def mask_frozen_gradients(
    grad_tree: Any,
    *,
    frozen_layers: set[str],
    use_pissa_lora: bool,
) -> tuple[Any, dict[str, Any], tuple[str, ...]]:
    grad_flat_local = dict(mlx_flatten(grad_tree))
    if not frozen_layers:
        return grad_tree, grad_flat_local, ()
    masked_param_names: list[str] = []
    for layer_key in sorted(frozen_layers):
        for param_name in _freeze_param_names_for_layer(
            layer_key,
            use_pissa_lora=use_pissa_lora,
        ):
            grad_value = grad_flat_local.get(param_name)
            if grad_value is None:
                continue
            grad_flat_local[param_name] = mx.zeros_like(grad_value)
            masked_param_names.append(param_name)
    if not masked_param_names:
        return grad_tree, grad_flat_local, ()
    masked_tree = mlx_unflatten(list(grad_flat_local.items()))
    return masked_tree, grad_flat_local, tuple(sorted(masked_param_names))


def summarize_fisher_state(
    fisher_state: Any,
) -> tuple[float | None, float | None, float | None]:
    """Return (second_moment_norm, preconditioner_scale, first_moment_norm)."""
    if fisher_state is None:
        return None, None, None
    fisher_terms = [
        arr.reshape(-1)
        for arr in fisher_state.v.values()
        if getattr(arr, "size", 0) > 0
    ]
    if not fisher_terms:
        return None, None, None
    fisher_norm_sq = sum(mx.sum(arr * arr) for arr in fisher_terms)
    preconditioner_terms = [mx.sqrt(arr) for arr in fisher_terms]
    preconditioner_scale = sum(mx.mean(arr) for arr in preconditioner_terms) / len(
        preconditioner_terms,
    )
    first_moment_norm = None
    if fisher_state.beta1 > 0.0 and fisher_state.m:
        m_terms = [
            arr.reshape(-1)
            for arr in fisher_state.m.values()
            if getattr(arr, "size", 0) > 0
        ]
        if m_terms:
            m_norm_sq = sum(mx.sum(arr * arr) for arr in m_terms)
            mx.eval(fisher_norm_sq, preconditioner_scale, m_norm_sq)
            first_moment_norm = float(mx.sqrt(m_norm_sq).item())
            return (
                float(mx.sqrt(fisher_norm_sq).item()),
                float(preconditioner_scale.item()),
                first_moment_norm,
            )
    mx.eval(fisher_norm_sq, preconditioner_scale)
    return (
        float(mx.sqrt(fisher_norm_sq).item()),
        float(preconditioner_scale.item()),
        None,
    )


def summarize_adamw_state(
    optimizer: Any,
) -> tuple[float | None, float | None, float | None]:
    if optimizer is None:
        return None, None, None
    state_flat = dict(mlx_flatten(optimizer.state))
    m_terms = [
        value.reshape(-1)
        for key, value in state_flat.items()
        if key.endswith(".m") and getattr(value, "size", 0) > 0
    ]
    v_terms = [
        value.reshape(-1)
        for key, value in state_flat.items()
        if key.endswith(".v") and getattr(value, "size", 0) > 0
    ]
    if not m_terms and not v_terms:
        return None, None, None
    first_moment = None
    second_moment = None
    preconditioner_scale = None
    m_norm_sq = None
    v_norm_sq = None
    denom_scale = None
    if m_terms:
        m_norm_sq = sum(mx.sum(arr * arr) for arr in m_terms)
    if v_terms:
        v_norm_sq = sum(mx.sum(arr * arr) for arr in v_terms)
        denom_scale = sum(mx.mean(mx.sqrt(arr)) for arr in v_terms) / len(v_terms)
    realized = []
    if m_norm_sq is not None:
        realized.append(m_norm_sq)
    if v_norm_sq is not None and denom_scale is not None:
        realized.extend([v_norm_sq, denom_scale])
    if realized:
        mx.eval(*realized)
    if m_norm_sq is not None:
        m_norm_sq_val = (
            float(m_norm_sq.item())
            if hasattr(m_norm_sq, "item")
            else float(m_norm_sq)
        )
        first_moment = math.sqrt(max(m_norm_sq_val, 0.0))
    if v_norm_sq is not None and denom_scale is not None:
        v_norm_sq_val = (
            float(v_norm_sq.item())
            if hasattr(v_norm_sq, "item")
            else float(v_norm_sq)
        )
        second_moment = math.sqrt(max(v_norm_sq_val, 0.0))
        preconditioner_scale = float(denom_scale.item())
    return first_moment, second_moment, preconditioner_scale


def build_behavioral_probe_state(
    *,
    adapter: Any,
    model: Any,
    controller_mode: str,
    use_pissa_lora: bool,
    base_activations: dict[int, list[Any]] | None,
    online_eval_problems: list[Any] | None,
    tokenizer: Any,
    baseline_margins: dict[str, float] | None,
    baseline_accuracy: float | None,
    per_layer_budget_ratio: dict[str, float] | None,
    per_layer_remaining_budget: dict[str, float] | None,
    online_eval_accuracy: float | None,
    online_eval_n_lost: int | None,
    online_eval_n_gained: int | None,
) -> BehavioralStateMeasurement | None:
    if controller_mode not in {
        CONTROLLER_MODE_BEHAVIORAL_PROBE,
        CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    }:
        return None

    per_layer_transport: dict[str, float] = {}
    if base_activations:
        if use_pissa_lora:
            transport_iter = adapter._iter_pissa_lora_modules(model)
        else:
            transport_iter = adapter._iter_nb_lora_modules(model)
        for layer_key, lora_module in transport_iter:
            try:
                layer_idx = int(layer_key.split(".")[2])
            except (IndexError, ValueError):
                continue
            layer_acts = base_activations.get(layer_idx)
            if not layer_acts:
                continue
            layer_stack = mx.stack(layer_acts)
            if use_pissa_lora:
                in_features = int(lora_module.lora_a.shape[0])
                if int(layer_stack.shape[-1]) != in_features:
                    continue
                delta_w_t = lora_module.scale * mx.matmul(
                    lora_module.lora_a,
                    lora_module.lora_b,
                )
            else:
                if int(layer_stack.shape[-1]) != int(lora_module._in_features):
                    continue
                a_factor, b_factor = lora_module._cayley_transform()
                scales = mx.clip(
                    lora_module.S_raw,
                    0.0,
                    lora_module._scale_bound,
                )
                delta_w_t = 2.0 * mx.matmul((scales[:, None] * a_factor).T, b_factor)
            transport = mx.matmul(
                layer_stack.astype(mx.float32),
                delta_w_t.astype(mx.float32),
            )
            transport_norm = mx.sqrt(mx.sum(transport * transport))
            mx.eval(transport_norm)
            per_layer_transport[layer_key] = float(transport_norm.item())

    margin_mean_delta = None
    margin_n_flipped_sign = None
    margin_n_near_zero_baseline = None
    margin_n_near_zero_current = None
    if online_eval_problems and tokenizer is not None and baseline_margins is not None:
        from modelcypher.core.domain.training.online_eval import compute_answer_margin

        def _collect_logits(prompt: str):
            return adapter._backend.collect_logits(model, tokenizer, prompt)

        current_margins = compute_answer_margin(
            online_eval_problems,
            _collect_logits,
            adapter._backend,
        )
        baseline_values = list(baseline_margins.values())
        current_values = list(current_margins.values())
        if baseline_values and current_values:
            margin_mean_delta = (
                sum(current_values) / len(current_values)
                - sum(baseline_values) / len(baseline_values)
            )
            margin_n_near_zero_baseline = sum(
                1 for value in baseline_values if abs(value) < _SQRT_EPS_F32
            )
            margin_n_near_zero_current = sum(
                1 for value in current_values if abs(value) < _SQRT_EPS_F32
            )
            shared_ids = set(baseline_margins.keys()) & set(current_margins.keys())
            margin_n_flipped_sign = sum(
                1
                for problem_id in shared_ids
                if (
                    baseline_margins[problem_id] > 0.0
                    and current_margins[problem_id] <= 0.0
                )
                or (
                    baseline_margins[problem_id] <= 0.0
                    and current_margins[problem_id] > 0.0
                )
            )

    accuracy_delta = None
    if online_eval_accuracy is not None and baseline_accuracy is not None:
        accuracy_delta = float(online_eval_accuracy) - float(baseline_accuracy)

    if (
        not per_layer_transport
        and per_layer_budget_ratio is None
        and margin_mean_delta is None
        and accuracy_delta is None
    ):
        return None

    return BehavioralStateMeasurement(
        per_layer_behavioral_transport_norm=per_layer_transport or None,
        per_layer_spectral_budget_ratio=per_layer_budget_ratio,
        per_layer_remaining_budget=per_layer_remaining_budget,
        margin_mean_delta=margin_mean_delta,
        margin_n_flipped_sign=margin_n_flipped_sign,
        margin_n_near_zero_baseline=margin_n_near_zero_baseline,
        margin_n_near_zero_current=margin_n_near_zero_current,
        online_eval_accuracy_delta=accuracy_delta,
        online_eval_n_lost=online_eval_n_lost,
        online_eval_n_gained=online_eval_n_gained,
    )


def build_batch_iterator_plan(
    *,
    geometric_reshape: bool,
    use_constrained: bool,
    use_answer_mask: bool,
    use_vl: bool,
    use_bilm_margin: bool,
    paired_dataset: list[dict[str, Any]] | None,
    answer_masked_dataset: list[tuple[Any, Any, int]] | None,
    train_dataset: list[Any],
    batch_size: int,
    seq_length: int,
    logic_groups: dict[str, list[int]] | None,
    template_groups: dict[str, list[int]] | None,
    seed: int,
    grad_accum_steps: int,
) -> tuple[Any, int, int]:
    if geometric_reshape and paired_dataset is not None:
        batch_iter = iterate_structured_batches(
            paired_dataset,
            batch_size,
            seq_length,
            logic_groups=logic_groups or {},
            template_groups=template_groups or {},
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_structured_batches(
            paired_dataset,
            batch_size,
            seq_length,
            logic_groups=logic_groups or {},
            template_groups=template_groups or {},
            loop=False,
            seed=seed,
        )))
    elif use_constrained and paired_dataset is not None:
        batch_iter = iterate_structured_batches(
            paired_dataset,
            batch_size,
            seq_length,
            logic_groups=logic_groups or {},
            template_groups=template_groups or {},
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_structured_batches(
            paired_dataset,
            batch_size,
            seq_length,
            logic_groups=logic_groups or {},
            template_groups=template_groups or {},
            loop=False,
            seed=seed,
        )))
    elif use_answer_mask:
        batch_iter = iterate_masked_batches(
            answer_masked_dataset,
            batch_size,
            seq_length,
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_masked_batches(
            answer_masked_dataset,
            batch_size,
            seq_length,
            loop=False,
            seed=seed,
        )))
    elif use_vl:
        batch_iter = iterate_vl_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_vl_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=False,
            seed=seed,
        )))
    elif use_bilm_margin:
        batch_iter = iterate_bilm_margin_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_bilm_margin_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=False,
            seed=seed,
        )))
    else:
        from mlx_lm.tuner.trainer import iterate_batches

        micro_bs = math.ceil(batch_size / grad_accum_steps) if grad_accum_steps > 1 else batch_size
        if grad_accum_steps > 1:
            logger.info(
                "Gradient accumulation: logical_batch=%d, micro_batch=%d, accum_steps=%d",
                batch_size,
                micro_bs,
                grad_accum_steps,
            )
        batch_iter = iterate_batches(
            train_dataset,
            micro_bs,
            seq_length,
            loop=True,
            seed=seed,
        )
        n_batches_per_epoch = len(list(iterate_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=False,
            seed=seed,
        )))
    return batch_iter, n_batches_per_epoch, grad_accum_steps


def build_epoch_metrics(local_state: dict[str, Any]) -> EpochMetrics:
    mass_metrics = local_state["mass_metrics"]
    adaptive_lr = local_state["adaptive_lr"]
    max_disp_to_remaining = local_state["max_disp_to_remaining_this_epoch"]
    max_effective_gain = local_state["max_effective_gain_this_epoch"]
    behavioral_state = local_state["behavioral_state"]
    return EpochMetrics(
        epoch=local_state["epoch_num"],
        train_loss=local_state["loss_val"],
        val_loss=local_state["v_loss"],
        eta=local_state["current_eta"],
        update_norm=local_state["update_norm"],
        max_spectral_ratio=local_state["max_ratio"],
        spectral_ratio_growth_per_iter=local_state["spectral_ratio_growth_per_iter"],
        mean_token_entropy=local_state["mean_entropy"],
        repetition_rate=local_state["rep_rate"],
        elapsed_seconds=local_state["epoch_elapsed"],
        eta_ceiling=local_state["eta_ceiling"] if adaptive_lr else None,
        adapter_saturation_median_ratio=local_state["median_budget_ratio"],
        expert_saturation_map=local_state["expert_saturation_map"],
        n_saturated_experts=local_state["n_saturated_experts"],
        n_total_target_experts=local_state["n_total_target_experts"],
        displacement=mass_metrics.get("displacement"),
        eta_sps=mass_metrics.get("eta_sps"),
        eta_weyl=mass_metrics.get("eta_weyl"),
        eta_step=mass_metrics.get("eta_step"),
        d_norm=mass_metrics.get("d_norm"),
        eta_margin=mass_metrics.get("eta_margin"),
        remaining_budget=local_state["remaining_budget"],
        max_displacement_to_remaining=(
            max_disp_to_remaining if max_disp_to_remaining > 0 else None
        ),
        max_effective_gain_ratio=(
            max_effective_gain if max_effective_gain > 0 else None
        ),
        online_eval_accuracy=local_state["online_eval_acc"],
        online_eval_n_correct=local_state["online_eval_n_correct"],
        online_eval_n_total=local_state["online_eval_n_total"],
        online_eval_degraded=local_state["online_eval_degraded"],
        online_eval_degraded_raw=local_state["online_eval_degraded_raw"],
        online_eval_degraded_significant=local_state["online_eval_degraded_significant"],
        online_eval_alpha=local_state["online_eval_alpha"],
        online_eval_current_ci_lower=local_state["online_eval_current_ci_lower"],
        online_eval_current_ci_upper=local_state["online_eval_current_ci_upper"],
        online_eval_baseline_ci_lower=local_state["online_eval_baseline_ci_lower"],
        online_eval_baseline_ci_upper=local_state["online_eval_baseline_ci_upper"],
        online_eval_pre_accuracy=local_state["online_eval_acc"],
        online_eval_pre_n_correct=local_state["online_eval_n_correct"],
        online_eval_pre_n_total=local_state["online_eval_n_total"],
        online_eval_pre_degraded=local_state["online_eval_degraded"],
        online_eval_pre_degraded_raw=local_state["online_eval_degraded_raw"],
        online_eval_pre_degraded_significant=local_state["online_eval_degraded_significant"],
        online_eval_pre_alpha=local_state["online_eval_alpha"],
        online_eval_pre_current_ci_lower=local_state["online_eval_current_ci_lower"],
        online_eval_pre_current_ci_upper=local_state["online_eval_current_ci_upper"],
        online_eval_pre_baseline_ci_lower=local_state["online_eval_baseline_ci_lower"],
        online_eval_pre_baseline_ci_upper=local_state["online_eval_baseline_ci_upper"],
        online_eval_pre_n_lost=local_state["online_eval_n_lost"],
        online_eval_pre_n_gained=local_state["online_eval_n_gained"],
        online_eval_pre_per_type_correct=local_state["online_eval_per_type_correct"],
        online_eval_pre_per_type_total=local_state["online_eval_per_type_total"],
        online_eval_stop_basis_accuracy=local_state["online_eval_stop_basis_acc"],
        online_eval_stop_basis_n_correct=local_state["online_eval_stop_basis_n_correct"],
        online_eval_stop_basis_n_total=local_state["online_eval_stop_basis_n_total"],
        online_eval_stop_basis_degraded=local_state["online_eval_stop_basis_degraded"],
        online_eval_stop_basis_degraded_raw=local_state["online_eval_stop_basis_degraded_raw"],
        online_eval_stop_basis_degraded_significant=local_state[
            "online_eval_stop_basis_degraded_significant"
        ],
        online_eval_stop_basis_alpha=local_state["online_eval_stop_basis_alpha"],
        online_eval_stop_basis_current_ci_lower=local_state[
            "online_eval_stop_basis_current_ci_lower"
        ],
        online_eval_stop_basis_current_ci_upper=local_state[
            "online_eval_stop_basis_current_ci_upper"
        ],
        online_eval_stop_basis_baseline_ci_lower=local_state[
            "online_eval_stop_basis_baseline_ci_lower"
        ],
        online_eval_stop_basis_baseline_ci_upper=local_state[
            "online_eval_stop_basis_baseline_ci_upper"
        ],
        online_eval_stop_basis_stage=local_state["online_eval_stop_basis_stage"],
        gate_confound_event=local_state["gate_confound_event"],
        projected_residual_max=local_state["projected_residual_max"],
        controller_mode=local_state["controller_mode"],
        optimizer_research_mode=local_state["optimizer_research_mode"],
        controller_trace={
            "step_traces": list(local_state["epoch_step_traces"]),
            "behavioral_state": (
                behavioral_state.to_dict() if behavioral_state is not None else None
            ),
        },
    )
