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

# ruff: noqa: F403,F405

"""Training loop methods for :class:`MLXTrainingAdapter`."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.backends.mlx_training_adapter_core import *  # noqa: F403
from modelcypher.core.domain.training.geometric_early_stopping import (  # noqa: F401
    check_loss_stable,
    check_val_loss_converged,
    should_certificate_stop,
)
from modelcypher.core.domain.training.spectral_budget import (  # noqa: F401
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    compute_projected_residuals,
    is_budget_exhausted,
)
from modelcypher.core.domain.training.mass_step_size import (
    CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP,
    CONTROLLER_MODE_BEHAVIORAL_PROBE,
    CONTROLLER_MODE_STRUCTURAL_OBSERVE,
    OPTIMIZER_MODE_ADAMW_MATCHED_TRACE,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    BehavioralStateMeasurement,
    ControllerLayerMeasurement,
    ControllerStepTrace,
    validate_controller_mode,
    validate_optimizer_research_mode,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_optimizer import OptimizerGeometryConfig


class _MLXTrainingAdapterTrainMixin:
    def train_loop(
        self,
        model,
        train_dataset,
        batch_size: int,
        seq_length: int,
        max_iters: int,
        seed: int,
        sigma_max: float,
        eval_dataset: list | None = None,
        eval_batches: int | None = None,
        adaptive_lr: bool = True,
        lr_monotonic: bool = False,
        sigma_k_min: float = 0.0,
        tokenizer=None,
        opt_config: "OptimizerGeometryConfig | None" = None,
        topo_monitor: bool = False,
        topo_probe_texts: list[str] | None = None,
        dim_monitor: bool = False,
        dim_probe_texts: list[str] | None = None,
        # Constrained geometric training (paired data) — EXPERIMENTAL
        constraint_config: Any = None,  # ConstraintConfig or None
        constraint_state: Any = None,  # ConstraintState or None
        paired_dataset: list[dict[str, Any]] | None = None,
        logic_groups: dict[str, list[int]] | None = None,
        template_groups: dict[str, list[int]] | None = None,
        # Geometric reshaping (constructive loss — expand + contrastive)
        geometric_reshape: bool = False,
        # Optional gradient hook: applied to gradient before optimizer step
        gradient_hook: "Callable | None" = None,
        # Anti-degeneration: entropy floor regularization
        entropy_regularization: bool = False,
        # Online correctness evaluation at epoch boundaries
        online_eval_problems: list | None = None,
        online_eval_baseline_ids: "frozenset | None" = None,
        # Answer-span masked CE (train only on answer tokens + EOS)
        answer_masked_dataset: list[tuple[Any, Any, int]] | None = None,
        answer_masked_eval: list[tuple[Any, Any, int]] | None = None,
        # Envelope caps: hard limits to prevent stop-signal erosion
        max_epochs: int | None = None,
        budget_cap: float | None = None,
        # Sub-epoch evaluation: override epoch-based check interval
        eval_interval: int | None = None,
        # Validation-loss convergence window. None derives minimum valid window.
        loss_stability_window_epochs: int | None = None,
        # Global EOS exclusion: exclude EOS token from CE in all paths
        eos_exclude: bool = False,
        # Outer similarity monitoring (Kucukahmetler et al. 2026)
        rss_monitor: bool = False,
        base_activations: dict | None = None,
        controller_mode: str = CONTROLLER_MODE_STRUCTURAL_OBSERVE,
        optimizer_research_mode: str = OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
        baseline_margins: dict[str, float] | None = None,
        # Ablation experiment params (research only, not CLI-exposed)
        entropy_floor_fraction: float | None = None,
        # Per-epoch degeneration gate: few-shot prompts + baseline max n-gram.
        # When provided, generates responses at epoch boundaries and stops
        # if max n-gram repetition exceeds baseline + sqrt(eps_f32).
        # n-gram order must be derived from readout effective rank (birthday
        # paradox); gate is disabled when degen_ngram_order is None.
        degen_prompts: list[str] | None = None,
        degen_baseline_max: float | None = None,
        degen_ngram_order: int | None = None,
        # Readout effective rank for diagnostics probe n-gram derivation.
        readout_erank: float | None = None,
        # Gradient accumulation: when > 1, batch_size is the logical batch
        # and micro_batch_size = ceil(batch_size / grad_accum_steps) is used
        # for forward/backward passes. Mathematically equivalent.
        grad_accum_steps: int = 1,
        # AdamW-decoupled weight decay: θ -= lr * λ * θ per step.
        # 0.0 = no decay (default). Research variable — isolated for testing.
        weight_decay: float = 0.0,
    ) -> tuple[list[tuple[int, float, float]], str, list[EpochMetrics]]:
        """Train with geometric stopping and MASS step sizing.

        Supports two adapter modes:
        - PiSSA-LoRA: PiSSA-initialized standard LoRA on geometry-derived surface.
          No Cayley retraction or scale clamping. MASS bounds displacement.
        - NB-LoRA (legacy): Cayley-parameterized retraction on the Stiefel manifold.
          NB-LoRA factors are Cayley-transformed at each step.

        MASS (Measured-Adaptive Step Size) — three-layer system:
        1. Spectral ceiling: eta_ceiling = sigma_k_min / sigma_max (Weyl 1912, static)
        2. Per-step SPS: eta_sps = f(x_t) / ||d_t||^2 (Loizou et al. 2020)
        3. Per-step Weyl: eta_weyl = sigma_k_min / ||d_t|| (displacement bound)
        Combined: eta_step = min(eta_sps, eta_weyl, eta_ceiling)

        No Armijo backtracking — every constant in Armijo (c, β, max_backtracks)
        was heuristic.  MASS derives the step bound per iteration from measurement:
        eta_step = min(eta_sps, eta_weyl, eta_ceiling).  See mass_step_size.py.

        Stopping (any one triggers):
        1. Validation loss convergence or degradation (overfitting)
        2. Weyl adapter-saturation exhaustion (per-layer spectral crossing)
        3. Training loss stability (fallback if no eval_dataset)
        4. Safety cap (max_iters)

        After each step: clamp S_raw (enforce bound).

        Returns: (losses, stop_reason, epoch_metrics)
        """
        if eval_batches is None:
            if eval_dataset is not None and len(eval_dataset) > 0:
                eval_batches = len(eval_dataset)
            elif answer_masked_eval is not None and len(answer_masked_eval) > 0:
                eval_batches = len(answer_masked_eval)
            else:
                eval_batches = max(1, len(train_dataset))

        controller_mode = validate_controller_mode(controller_mode)
        optimizer_research_mode = validate_optimizer_research_mode(
            optimizer_research_mode,
        )
        if controller_mode == CONTROLLER_MODE_BEHAVIORAL_CLOSED_LOOP:
            raise ValueError(
                "mass_behavioral_closed_loop is reserved until an offline-derived "
                "control law has been validated."
            )

        import mlx.optimizers as opt
        from mlx.utils import tree_flatten as mlx_flatten, tree_map as mlx_tree_map
        from mlx.utils import tree_unflatten as mlx_unflatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        # Resolve base CE loss: exclude EOS globally if requested.
        # The base model already has EOS behaviour from pre-training; training CE
        # on EOS produces gradients that erode the adapter's stopping ability.
        if eos_exclude and tokenizer is not None:
            _eos_id = getattr(tokenizer, "eos_token_id", None)
            if _eos_id is not None:
                base_ce_loss = make_eos_excluded_loss(_eos_id)
                logger.info("EOS exclusion: target_id=%d excluded from CE globally", _eos_id)
            else:
                logger.warning("eos_exclude requested but tokenizer has no eos_token_id")
                base_ce_loss = default_loss
        else:
            base_ce_loss = default_loss

        # Constrained training mode: use paired loss + paired batch iterator
        use_constrained = (
            constraint_config is not None
            and constraint_state is not None
            and paired_dataset is not None
            and not geometric_reshape  # geometric reshape supersedes constraints
        )

        use_answer_mask = False  # Set True in answer_masked_dataset branch
        use_vl = (
            isinstance(train_dataset, list)
            and len(train_dataset) > 0
            and isinstance(train_dataset[0], dict)
            and "tokens" in train_dataset[0]
            and "pixel_values" in train_dataset[0]
        )
        if use_vl and grad_accum_steps > 1:
            logger.info(
                "VL path disables gradient accumulation (variable-size visual tensors). "
                "Using grad_accum_steps=1."
            )
            grad_accum_steps = 1

        if geometric_reshape and paired_dataset is not None:
            # Determine target layers for geometric reshaping.
            # Use middle-to-late layers where reasoning processing happens.
            base = getattr(model, "model", model)
            n_layers = len(base.layers)
            # All transformer blocks participate.  Embedding and output head
            # are outside model.layers; no geometric basis to exclude any block.
            reshape_target_layers = list(range(n_layers))
            loss_fn = make_geometric_reshaping_loss(reshape_target_layers)
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)
            logger.info(
                "Geometric reshaping: target_layers=%s (expand erank + contrastive)",
                reshape_target_layers,
            )
            # Calibrate gradient weights: measure per-component gradient norms
            # on a single batch and set weights so all three components contribute
            # equally to the parameter update. Data-derived, no magic numbers.
            calib_iter = iterate_structured_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=False, seed=seed,
            )
            calib_batch = next(calib_iter)
            cb, cl, cam, cinv, ccf = calib_batch
            # Trigger init_values by running one forward pass first
            (init_loss, _), _ = loss_value_and_grad(
                model, cb, cl, cam, cinv, ccf,
            )
            mx.eval(init_loss)
            calib_info = calibrate_geometric_weights(
                model, loss_fn, cb, cl, cam, cinv, ccf,
            )
            # Rebuild loss_value_and_grad with calibrated weights
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)
            logger.info(
                "Gradient calibration: ||∇ce||=%.4e ||∇expand||=%.4e "
                "||∇contrast||=%.4e → w_expand=%.1f w_contrast=%.1f",
                calib_info["ce_gnorm"],
                calib_info["expand_gnorm"],
                calib_info["contrast_gnorm"],
                calib_info["w_expand"],
                calib_info["w_contrast"],
            )
        elif use_constrained:
            loss_fn = make_constrained_loss(constraint_state, constraint_config)
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)
            logger.info(
                "Constrained training: ε_inv=%.4f, m_sep=%.4f, ε_tail=%.4f, "
                "target_layers=%s",
                constraint_config.epsilon_inv,
                constraint_config.margin_sep,
                constraint_config.epsilon_tail,
                constraint_config.target_layers,
            )
        elif answer_masked_dataset is not None:
            # Answer-span masking: CE only on answer tokens + EOS
            if entropy_regularization:
                # Combined: answer-masked CE + entropy floor on all tokens
                baseline_ent = measure_baseline_entropy(
                    model, train_dataset, batch_size, seq_length,
                    n_batches=eval_batches,
                )
                if entropy_floor_fraction is not None:
                    ent_floor = baseline_ent * entropy_floor_fraction
                    logger.info(
                        "Entropy floor override (answer-masked): fraction=%.4f, baseline=%.4f, floor=%.4f",
                        entropy_floor_fraction, baseline_ent, ent_floor,
                    )
                else:
                    ent_floor = self._derive_entropy_floor_or_fail(
                        baseline_entropy=baseline_ent,
                        dataset_samples=len(train_dataset),
                        scope="answer_masked_training",
                    )
                am_loss_fn = make_entropy_regularized_answer_masked_loss(ent_floor)
                logger.info(
                    "Answer-masked CE + entropy reg: baseline=%.4f, floor=%.4f",
                    baseline_ent, ent_floor,
                )
            else:
                def am_loss_fn(model, inputs, targets, masks):
                    logits = model(inputs)
                    logits = logits.astype(mx.float32)
                    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
                    masked_ce = ce * masks
                    ntoks = masks.sum()
                    return masked_ce.sum() / mx.maximum(ntoks, mx.array(1.0)), ntoks

            loss_value_and_grad = nn.value_and_grad(model, am_loss_fn)
            logger.info("Answer-masked CE: training on answer tokens + EOS only")
            use_answer_mask = True
        elif use_vl:
            image_token_id = train_dataset[0].get("image_token_id")
            video_token_id = train_dataset[0].get("video_token_id")
            loss_fn = make_vl_loss(
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)
            logger.info(
                "VL training: image-conditioned CE with visual embedding injection "
                "(image_token_id=%s, video_token_id=%s)",
                image_token_id,
                video_token_id,
            )
        else:
            if entropy_regularization:
                # Measure baseline entropy to derive the floor
                baseline_ent = measure_baseline_entropy(
                    model, train_dataset, batch_size, seq_length,
                    n_batches=eval_batches,
                )
                if entropy_floor_fraction is not None:
                    # Ablation override: use explicit fraction instead of sqrt(eps)
                    ent_floor = baseline_ent * entropy_floor_fraction
                    logger.info(
                        "Entropy floor override: fraction=%.4f, baseline=%.4f, floor=%.4f",
                        entropy_floor_fraction, baseline_ent, ent_floor,
                    )
                else:
                    ent_floor = self._derive_entropy_floor_or_fail(
                        baseline_entropy=baseline_ent,
                        dataset_samples=len(train_dataset),
                        scope="full_sequence_training",
                    )
                loss_fn = make_entropy_regularized_loss(ent_floor)
                logger.info(
                    "Entropy regularization: baseline=%.4f, floor=%.4f",
                    baseline_ent, ent_floor,
                )
            else:
                loss_fn = base_ce_loss
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)

        # Learning rate: MASS (Measured-Adaptive Step Size)
        # Layer 1: Spectral ceiling from Weyl 1912 (static, from SVD geometry)
        # Layer 2: Per-step SPS (Loizou et al. 2020) + per-step Weyl displacement
        # Layer 3: Validation-guided backoff (existing, measured)
        eta_ceiling = self._derive_spectral_ceiling(
            sigma_k_min=sigma_k_min,
            sigma_max_global=sigma_max,
        )
        current_eta = eta_ceiling

        # Curvature-aware Cayley-Stiefel: diagonal Fisher preconditioning
        # with direction smoothing (first moment). Both moments operate in
        # unconstrained (A_tilde, B_tilde) space BEFORE Cayley retraction,
        # which is valid because the retraction maps unconstrained parameters
        # to orthonormal factors regardless.
        # m_t → direction smoothing, v_t → diag(F_empirical) (Hwang et al. 2024, FAdam).
        from modelcypher.core.domain.training.diagonal_fisher_preconditioner import (
            init_fisher_state,
            precondition_gradient as fisher_precondition,
            update_fisher_state,
        )

        # fisher_state and optimizer are initialized after n_batches_per_epoch
        # is known (~line 710+), so both paths use the same derived β₁/β₂.
        # The closures below (_summarize_fisher_state, _summarize_adamw_state)
        # capture these by name and are only called inside the training loop.
        fisher_state = None  # type: ignore[assignment]
        optimizer = None

        baseline_accuracy = None
        if online_eval_problems:
            baseline_accuracy = len(online_eval_baseline_ids or frozenset()) / max(
                len(online_eval_problems), 1,
            )

        def _objective_components() -> list[str]:
            components = ["ce"]
            if use_answer_mask:
                components[0] = "ce_answer_masked"
            if use_constrained:
                components.append("constraint")
            if geometric_reshape:
                components.append("geometric_reshape")
            if entropy_regularization:
                components.append("entropy_regularization")
            if use_vl:
                components.append("vision_language")
            return components

        def _layer_measurements_from_gradient(
            *,
            grad_map: dict[str, Any],
            step_learning_rate: float,
        ) -> dict[str, ControllerLayerMeasurement] | None:
            if not grad_map:
                return None
            per_layer: dict[str, ControllerLayerMeasurement] = {}
            total_norm_sq = 0.0
            layer_norms: dict[str, float] = {}

            if use_pissa_lora:
                lora_iter = self._iter_pissa_lora_modules(model)
                param_suffixes = (".lora_a", ".lora_b")
            else:
                lora_iter = self._iter_nb_lora_modules(model)
                param_suffixes = (".A_tilde", ".B_tilde", ".S_raw")

            for layer_key, lora_module in lora_iter:
                prefix = layer_key.removesuffix(".weight")
                grad_norm_sq_terms = []
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
                mx.eval(grad_norm_sq)
                grad_norm = float(mx.sqrt(grad_norm_sq).item())
                layer_norms[layer_key] = grad_norm
                total_norm_sq += grad_norm * grad_norm
                decay_scale = None
                if opt_config is not None and layer_key in opt_config.layer_configs:
                    decay_scale = float(opt_config.layer_configs[layer_key].decay_scale)
                scale_bound_val = (
                    None if use_pissa_lora
                    else float(lora_module._scale_bound)
                )
                per_layer[layer_key] = ControllerLayerMeasurement(
                    parameter_update_norm=step_learning_rate * grad_norm,
                    total_step_fraction=None,
                    decay_scale=decay_scale,
                    scale_bound=scale_bound_val,
                    step_learning_rate=step_learning_rate,
                )

            if not per_layer:
                return None

            total_norm = math.sqrt(total_norm_sq)
            if total_norm > 0.0:
                for layer_key, layer_norm in layer_norms.items():
                    fraction = layer_norm / total_norm
                    current = per_layer[layer_key]
                    per_layer[layer_key] = ControllerLayerMeasurement(
                        parameter_update_norm=current.parameter_update_norm,
                        behavioral_transport_norm=current.behavioral_transport_norm,
                        spectral_budget_ratio=current.spectral_budget_ratio,
                        remaining_budget=current.remaining_budget,
                        total_step_fraction=fraction,
                        decay_scale=current.decay_scale,
                        scale_bound=current.scale_bound,
                        step_learning_rate=current.step_learning_rate,
                    )
            return per_layer

        def _summarize_fisher_state() -> tuple[float | None, float | None, float | None]:
            """Return (second_moment_norm, preconditioner_scale, first_moment_norm)."""
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
            # First moment norm (if β₁ > 0)
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
            return float(mx.sqrt(fisher_norm_sq).item()), float(preconditioner_scale.item()), None

        def _summarize_adamw_state() -> tuple[float | None, float | None, float | None]:
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
            if m_terms:
                m_norm_sq = sum(mx.sum(arr * arr) for arr in m_terms)
                mx.eval(m_norm_sq)
                first_moment = float(mx.sqrt(m_norm_sq).item())
            if v_terms:
                v_norm_sq = sum(mx.sum(arr * arr) for arr in v_terms)
                denom_scale = sum(mx.mean(mx.sqrt(arr)) for arr in v_terms) / len(v_terms)
                mx.eval(v_norm_sq, denom_scale)
                second_moment = float(mx.sqrt(v_norm_sq).item())
                preconditioner_scale = float(denom_scale.item())
            return first_moment, second_moment, preconditioner_scale

        def _behavioral_probe_state(
            *,
            per_layer_budget_ratio: dict[str, float] | None,
            per_layer_remaining_budget: dict[str, float] | None,
            online_eval_accuracy: float | None,
            online_eval_n_lost: int | None,
            online_eval_n_gained: int | None,
        ) -> BehavioralStateMeasurement | None:
            if controller_mode != CONTROLLER_MODE_BEHAVIORAL_PROBE:
                return None

            per_layer_transport: dict[str, float] = {}
            if base_activations:
                if use_pissa_lora:
                    transport_iter = self._iter_pissa_lora_modules(model)
                else:
                    transport_iter = self._iter_nb_lora_modules(model)
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
                        # DeltaW^T = scale * lora_a @ lora_b  [in, out]
                        delta_w_t = lora_module.scale * mx.matmul(
                            lora_module.lora_a, lora_module.lora_b,
                        )
                    else:
                        if int(layer_stack.shape[-1]) != int(lora_module._in_features):
                            continue
                        A, B = lora_module._cayley_transform()
                        S = mx.clip(lora_module.S_raw, 0.0, lora_module._scale_bound)
                        delta_w_t = 2.0 * mx.matmul((S[:, None] * A).T, B)
                    transport = mx.matmul(layer_stack.astype(mx.float32), delta_w_t.astype(mx.float32))
                    transport_norm = mx.norm(transport)
                    mx.eval(transport_norm)
                    per_layer_transport[layer_key] = float(transport_norm.item())

            margin_mean_delta = None
            margin_n_flipped_sign = None
            margin_n_near_zero_baseline = None
            margin_n_near_zero_current = None
            if online_eval_problems and tokenizer is not None and baseline_margins is not None:
                from modelcypher.core.domain.training.online_eval import compute_answer_margin

                def _collect_logits(prompt: str):
                    return self._backend.collect_logits(model, tokenizer, prompt)

                current_margins = compute_answer_margin(
                    online_eval_problems,
                    _collect_logits,
                    self._backend,
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

        # Cayley constraint preserved in NBLoRALinear. Pullback metric P
        # removed (falsification 2026-02-23: P ≈ I throughout training,
        # median ||P-I||/√r = 0.001, cos(Pg,g) > 0.999, 3 seeds × 2
        # families). The Stiefel constraint drives the validated benefit
        # (val_loss 1.27 vs 1.38), not the pullback metric.

        losses: list[tuple[int, float, float]] = []
        val_losses: list[float] = []
        epoch_metrics_list: list[EpochMetrics] = []
        epoch_step_traces: list[dict[str, Any]] = []
        last_max_spectral_ratio: float | None = None
        dim_snapshots: list = []  # DimensionalSnapshot history for trend analysis
        stop_reason: str | None = None
        use_pissa_lora = self._has_pissa_lora(model)
        best_val_loss = float("inf")
        best_weights: dict[str, Any] | None = None
        val_loss_baseline: float | None = None  # First epoch's val loss for certificate condition 5

        if geometric_reshape and paired_dataset is not None:
            batch_iter = iterate_structured_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=True, seed=seed,
            )
            n_batches_per_epoch = len(list(iterate_structured_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=False, seed=seed,
            )))
        elif use_constrained and paired_dataset is not None:
            # Constrained training requires both invariance and counterfactual
            # pairs in every batch. Template-first structured sampling guarantees
            # non-zero counterfactual coverage; pair-only sampling can produce
            # cf_pairs == 0 for entire epochs on sparse template overlap.
            batch_iter = iterate_structured_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=True, seed=seed,
            )
            n_batches_per_epoch = len(list(iterate_structured_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=False, seed=seed,
            )))
        elif use_answer_mask:
            batch_iter = iterate_masked_batches(
                answer_masked_dataset, batch_size, seq_length,
                loop=True, seed=seed,
            )
            n_batches_per_epoch = len(list(iterate_masked_batches(
                answer_masked_dataset, batch_size, seq_length,
                loop=False, seed=seed,
            )))
        elif use_vl:
            # VL path currently uses direct batches (no grad accumulation) because
            # each sample carries variable-size visual tensors.
            batch_iter = iterate_vl_batches(
                train_dataset, batch_size, seq_length, loop=True, seed=seed,
            )
            n_batches_per_epoch = len(
                list(iterate_vl_batches(
                    train_dataset, batch_size, seq_length, loop=False, seed=seed,
                ))
            )
        else:
            # Gradient accumulation: use smaller micro-batches for forward/backward
            # to avoid OOM, accumulate grad_accum_steps micro-batches per optimizer step.
            micro_bs = math.ceil(batch_size / grad_accum_steps) if grad_accum_steps > 1 else batch_size
            if grad_accum_steps > 1:
                logger.info(
                    "Gradient accumulation: logical_batch=%d, micro_batch=%d, accum_steps=%d",
                    batch_size, micro_bs, grad_accum_steps,
                )
            batch_iter = iterate_batches(
                train_dataset, micro_bs, seq_length, loop=True, seed=seed,
            )
            # Epoch structure uses logical batch_size (optimizer steps per epoch)
            n_batches_per_epoch = len(
                list(iterate_batches(train_dataset, batch_size, seq_length, loop=False, seed=seed))
            )
        if n_batches_per_epoch <= 0:
            raise ValueError("Training dataset produced zero batches")

        # Initialize fisher_state and optimizer now that n_batches_per_epoch
        # is known. Both paths use the same derived β₁ and β₂.
        trainable_flat = dict(mlx_flatten(model.trainable_parameters()))
        fisher_state = init_fisher_state(
            trainable_flat, self._backend,
            n_batches_per_epoch=n_batches_per_epoch,
        )
        del trainable_flat  # free reference
        if fisher_state.beta1 > 0.0:
            logger.info(
                "First moment enabled: β₁=%.4f (n_batches_per_epoch=%d)",
                fisher_state.beta1, n_batches_per_epoch,
            )
        if optimizer_research_mode == OPTIMIZER_MODE_ADAMW_MATCHED_TRACE:
            optimizer = opt.AdamW(
                learning_rate=current_eta,
                betas=[fisher_state.beta1, fisher_state.beta2],
                eps=_SQRT_EPS_F32,
                weight_decay=weight_decay,
                bias_correction=True,
            )
            optimizer.init(model.trainable_parameters())

        # MASS √N epoch budget correction (Brownian scaling).
        from modelcypher.core.domain.training.mass_step_size import (
            apply_sqrt_n_epoch_correction,
        )

        eta_ceiling_before = eta_ceiling
        eta_ceiling = apply_sqrt_n_epoch_correction(
            eta_ceiling, n_batches_per_epoch,
        )
        if eta_ceiling != eta_ceiling_before:
            current_eta = eta_ceiling
            logger.info(
                "MASS √N budget: ceiling %.4e / √%d = %.4e",
                eta_ceiling_before, n_batches_per_epoch, eta_ceiling,
            )

        use_val_stopping = (
            (eval_dataset is not None and len(eval_dataset) > 0)
            or (use_answer_mask and answer_masked_eval is not None and len(answer_masked_eval) > 0)
        )
        # Sample variance requires n>=2 observations per window.
        # Two windows are then compared: previous vs recent.
        if loss_stability_window_epochs is None:
            loss_stability_window_epochs = 2
        if loss_stability_window_epochs < 2:
            raise ValueError("loss_stability_window_epochs must be >= 2")
        min_val_windows_for_stop = 2 * loss_stability_window_epochs
        # Eval batch size: data-derived (dataset size / eval_batches)
        eval_batch_size = min(
            batch_size,
            max(1, len(eval_dataset) // max(1, eval_batches)) if eval_dataset else 2,
        )

        check_interval = max(1, n_batches_per_epoch)
        if eval_interval is not None and eval_interval > 0:
            check_interval = eval_interval
            logger.info(
                "Sub-epoch eval: check every %d iters (epoch=%d)",
                check_interval, n_batches_per_epoch,
            )

        lr_mode = "constant"
        if adaptive_lr:
            lr_mode = "adaptive-monotonic" if lr_monotonic else "adaptive"
        optimizer_name = (
            "AdamW-matched-trace"
            if optimizer_research_mode == OPTIMIZER_MODE_ADAMW_MATCHED_TRACE
            else ("PiSSA-Fisher" if use_pissa_lora else "Cayley-Fisher")
        )
        logger.info(
            "Training: optimizer=%s, stop=%s, cap=%d, epoch=%d batches, lr=%.2e, mode=%s",
            optimizer_name,
            "certificate" if use_val_stopping else "training loss",
            max_iters, n_batches_per_epoch, current_eta, lr_mode,
        )
        # Track params at epoch start for update_norm
        epoch_start_params: dict[str, Any] | None = None
        epoch_start_time = time.time()

        # Last-step gradient for stopping certificate
        grad_last: Any = None
        # Gradient norm history for stochastic stationarity
        grad_norm_history: list[float] = []
        # Conformal margin tracking (Sahraee-Ardakan et al. 2026)
        remaining_budget: float | None = None
        max_effective_gain_this_epoch: float = 0.0
        max_disp_to_remaining_this_epoch: float = 0.0

        for it in range(max_iters):
            # Snapshot params at epoch start
            if it % n_batches_per_epoch == 0:
                trainable = dict(mlx_flatten(model.trainable_parameters()))
                epoch_start_params = {k: mx.array(v) for k, v in trainable.items()}
                mx.eval(*epoch_start_params.values())
                epoch_start_time = time.time()
                max_effective_gain_this_epoch = 0.0
                max_disp_to_remaining_this_epoch = 0.0
                epoch_step_traces = []

            t_step = time.time()

            if use_constrained or geometric_reshape:
                batch, lengths, answer_masks, inv_pairs, cf_pairs = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(
                    model, batch, lengths, answer_masks, inv_pairs, cf_pairs,
                )
            elif use_answer_mask:
                inputs, targets, masks = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(
                    model, inputs, targets, masks,
                )
            elif use_vl:
                batch, lengths, pixel_values_batch, position_ids_batch = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(
                    model, batch, lengths, pixel_values_batch, position_ids_batch,
                )
            else:
                if grad_accum_steps <= 1:
                    batch, lengths = next(batch_iter)
                    (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)
                else:
                    # Gradient accumulation: sum gradients over micro-batches,
                    # then divide by number of steps. Mathematically equivalent
                    # to a single large-batch forward/backward.
                    accum_grad = None
                    accum_loss = 0.0
                    accum_ntoks = 0
                    for _accum_i in range(grad_accum_steps):
                        batch, lengths = next(batch_iter)
                        (mb_loss, mb_ntoks), mb_grad = loss_value_and_grad(
                            model, batch, lengths,
                        )
                        mx.eval(mb_loss)
                        accum_loss += float(mb_loss) * int(mb_ntoks)
                        accum_ntoks += int(mb_ntoks)
                        if accum_grad is None:
                            accum_grad = mb_grad
                        else:
                            accum_grad = mlx_tree_map(
                                lambda a, b: a + b, accum_grad, mb_grad,
                            )
                        mx.eval(*[v for _, v in mlx_flatten(accum_grad)])
                    # Average accumulated gradient
                    grad = mlx_tree_map(
                        lambda g: g / grad_accum_steps, accum_grad,
                    )
                    loss = mx.array(accum_loss / max(accum_ntoks, 1))
                    ntoks = accum_ntoks

            # Save gradient for stopping certificate (overwritten each step;
            # at epoch boundary, holds the last step's gradient).
            grad_last = grad

            # MASS Layer 2: Per-step measured rates on the active update
            # direction. Cayley-Fisher uses the diagonal-Fisher preconditioned
            # direction; AdamW-matched-trace uses the raw gradient because the
            # optimizer applies its own stateful preconditioner internally.
            from modelcypher.core.domain.training.mass_step_size import (
                compute_per_step_rates,
            )

            grad_flat = dict(mlx_flatten(grad))
            ce_grad_norm = None
            objective_components = _objective_components()
            if len(objective_components) == 1 and objective_components[0] in {
                "ce",
                "ce_answer_masked",
            }:
                raw_flat = [p.reshape(-1) for p in grad_flat.values() if p.size > 0]
                if raw_flat:
                    ce_norm_sq = sum(mx.sum(p * p) for p in raw_flat)
                    mx.eval(ce_norm_sq)
                    ce_grad_norm = float(mx.sqrt(ce_norm_sq).item())

            fisher_second_moment_norm = None
            fisher_preconditioner_scale = None
            fisher_first_moment_norm = None
            if optimizer_research_mode == OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS:
                fisher_state = update_fisher_state(fisher_state, grad_flat, self._backend)
                update_direction = fisher_precondition(
                    grad_flat, fisher_state, self._backend,
                )
                (
                    fisher_second_moment_norm,
                    fisher_preconditioner_scale,
                    fisher_first_moment_norm,
                ) = _summarize_fisher_state()
            else:
                update_direction = grad_flat

            mass_metrics: dict[str, float] = {}
            d_flat = [p.reshape(-1) for p in update_direction.values() if p.size > 0]
            d_norm_sq = sum(mx.sum(p * p) for p in d_flat)
            mx.eval(d_norm_sq, loss)
            d_norm_val = float(mx.sqrt(d_norm_sq).item())
            loss_float = float(loss)

            eta_step, eta_sps_val, eta_weyl_val, displacement_val, eta_margin_val = (
                compute_per_step_rates(
                    loss_float, d_norm_val, sigma_k_min, eta_ceiling,
                    remaining_budget=remaining_budget,
                )
            )

            mass_metrics["eta_step"] = eta_step
            mass_metrics["eta_sps"] = eta_sps_val
            mass_metrics["eta_weyl"] = eta_weyl_val
            mass_metrics["displacement"] = displacement_val
            mass_metrics["d_norm"] = d_norm_val
            if eta_margin_val is not None:
                mass_metrics["eta_margin"] = eta_margin_val

            # Track effective gain ratio for stability certificate
            if eta_ceiling > 0:
                gain_ratio = eta_step / eta_ceiling
                max_effective_gain_this_epoch = max(
                    max_effective_gain_this_epoch, gain_ratio,
                )

            # Track displacement-to-remaining ratio (epoch-level diagnostic).
            # NOTE: remaining_budget is NOT decremented per step. Weyl per-step
            # displacement is an upper bound that overestimates cumulative
            # spectral impact by ~50× (updates spread across singular
            # directions and partially cancel). The epoch-boundary spectral
            # measurement is the sole source of truth for remaining budget.
            if remaining_budget is not None and remaining_budget > 0:
                disp_ratio = displacement_val / remaining_budget
                max_disp_to_remaining_this_epoch = max(
                    max_disp_to_remaining_this_epoch, disp_ratio,
                )

            # Optional gradient hook (e.g. format bias projection).
            # Applied to raw gradient, then the active optimizer path rebuilds
            # its measured direction from the hooked gradient.
            if gradient_hook is not None:
                grad = gradient_hook(grad)
                grad_flat = dict(mlx_flatten(grad))
                if optimizer_research_mode == OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS:
                    update_direction = fisher_precondition(
                        grad_flat, fisher_state, self._backend,
                    )
                else:
                    update_direction = grad_flat

            step_layer_measurements = _layer_measurements_from_gradient(
                grad_map=grad_flat,
                step_learning_rate=eta_step,
            )

            # Apply the optimizer-specific update with the same MASS-derived
            # global step size.
            if optimizer_research_mode == OPTIMIZER_MODE_ADAMW_MATCHED_TRACE:
                optimizer.learning_rate = mx.array(eta_step)
                optimizer.update(model, grad)
                mx.eval(model.trainable_parameters(), optimizer.state)
                (
                    optimizer_first_moment_norm,
                    optimizer_second_moment_norm,
                    optimizer_preconditioner_scale,
                ) = _summarize_adamw_state()
            else:
                eta_arr = mx.array(eta_step)
                wd_factor = mx.array(1.0 - eta_step * weight_decay) if weight_decay > 0 else None
                current_params = dict(mlx_flatten(model.trainable_parameters()))
                if wd_factor is not None:
                    # AdamW-decoupled weight decay: θ = (1 - lr*λ)*θ - lr*d
                    updated_params = [
                        (k, wd_factor * current_params[k] - eta_arr * update_direction[k])
                        for k in current_params
                        if k in update_direction
                    ]
                else:
                    updated_params = [
                        (k, current_params[k] - eta_arr * update_direction[k])
                        for k in current_params
                        if k in update_direction
                    ]
                model.load_weights(updated_params, strict=False)
                mx.eval(*[v for _, v in mlx_flatten(model.trainable_parameters())])
                optimizer_first_moment_norm = fisher_first_moment_norm
                optimizer_second_moment_norm = fisher_second_moment_norm
                optimizer_preconditioner_scale = fisher_preconditioner_scale

            # Dual variable update for constrained training (outside gradient tape)
            if use_constrained and constraint_state is not None:
                # Materialize pending constraint values
                if hasattr(constraint_state, '_pending_c_inv'):
                    mx.eval(
                        constraint_state._pending_ce,
                        constraint_state._pending_c_inv,
                        constraint_state._pending_c_sep,
                        constraint_state._pending_c_geo,
                    )
                    c_inv_val = float(constraint_state._pending_c_inv.item())
                    c_sep_val = float(constraint_state._pending_c_sep.item())
                    c_geo_val = float(constraint_state._pending_c_geo.item())
                    constraint_state.last_ce_loss = float(
                        constraint_state._pending_ce.item()
                    )
                    # Use effective step size from MASS.
                    alpha_dual = mass_metrics.get("eta_step", current_eta)
                    constraint_state.dual_update(
                        c_inv_val, c_sep_val, c_geo_val,
                        constraint_config, alpha_dual,
                    )

            # NB-LoRA: clamp S_raw after every step (Cayley bound enforcement).
            # PiSSA LoRA: no clamping — MASS step sizing bounds displacement.
            if not use_pissa_lora:
                self._clamp_all_scales(model)

            epoch_step_traces.append(
                ControllerStepTrace(
                    step=it + 1,
                    controller_mode=controller_mode,
                    optimizer_research_mode=optimizer_research_mode,
                    objective_components=objective_components,
                    ce_grad_norm=ce_grad_norm,
                    auxiliary_grad_norms=None,
                    total_effective_step_norm=displacement_val,
                    eta_ceiling=eta_ceiling,
                    eta_sps=eta_sps_val,
                    eta_weyl=eta_weyl_val,
                    eta_margin=eta_margin_val,
                    eta_step=eta_step,
                    effective_gain_ratio=(eta_step / eta_ceiling) if eta_ceiling > 0 else None,
                    optimizer_first_moment_norm=optimizer_first_moment_norm,
                    optimizer_second_moment_norm=optimizer_second_moment_norm,
                    optimizer_preconditioner_scale=optimizer_preconditioner_scale,
                    per_layer_measurements=step_layer_measurements,
                ).to_dict()
            )

            loss_val = float(loss)
            ntoks_val = float(ntoks)
            elapsed = time.time() - t_step
            tps = float("inf") if elapsed <= 0 else ntoks_val / elapsed

            losses.append((it, loss_val, tps))

            # Log at first iter
            if it == 0:
                logger.info(
                    "Iter 1 (epoch 0.0) | train_loss=%.4f | tokens/sec=%.1f",
                    loss_val, tps,
                )

            # ── Epoch boundary: eval, adapt, measure, check ──
            if (it + 1) % check_interval == 0:
                epoch_num = (it + 1) // n_batches_per_epoch
                epoch_elapsed = time.time() - epoch_start_time

                # 1. Validation loss
                v_loss = None
                if use_answer_mask and answer_masked_eval is not None:
                    # Evaluate with masked loss on eval set
                    v_loss = self._evaluate_masked_loss(
                        model, answer_masked_eval, batch_size, seq_length,
                        eval_batches,
                    )
                elif use_val_stopping:
                    v_loss, _ = self.evaluate_loss(
                        model=model,
                        dataset=eval_dataset,
                        tokenizer=None,
                        batch_size=eval_batch_size,
                        seq_length=seq_length,
                        n_batches=eval_batches,
                    )
                    val_losses.append(v_loss)
                    # Record baseline val loss from first evaluation
                    if val_loss_baseline is None:
                        val_loss_baseline = v_loss
                    # Track best checkpoint for restoration.
                    # MLX arrays are immutable — load_weights creates new arrays,
                    # so storing references is safe (no in-place mutation).
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_weights = dict(mlx_flatten(model.trainable_parameters()))

                # 2. Update norm (||θ_end - θ_start||)
                update_norm = None
                if epoch_start_params is not None:
                    current_params = dict(mlx_flatten(model.trainable_parameters()))
                    update_norm = math.sqrt(sum(
                        float(mx.sum((current_params[k] - epoch_start_params[k]) ** 2))
                        for k in epoch_start_params if k in current_params
                    ))

                # 3. Adaptive LR: validation-guided ceiling backoff
                # Per-step rates (SPS, Weyl) adapt within the ceiling automatically.
                # The ceiling only decreases when validation loss degrades.
                from modelcypher.core.domain.training.mass_step_size import (
                    apply_validation_backoff,
                )

                prev_ceiling = eta_ceiling
                eta_ceiling = apply_validation_backoff(
                    eta_ceiling, val_losses,
                    adaptive_lr=adaptive_lr,
                )
                if eta_ceiling != prev_ceiling:
                    current_eta = eta_ceiling
                    backoff = eta_ceiling / prev_ceiling
                    logger.info(
                        "Val loss increased (%.4f → %.4f): ceiling backoff=%.3f, "
                        "eta_ceiling %.2e -> %.2e",
                        val_losses[-2], val_losses[-1], backoff,
                        prev_ceiling, eta_ceiling,
                    )

                # 3b. Val loss convergence/overfitting check
                if use_val_stopping and len(val_losses) >= min_val_windows_for_stop:
                    should_stop_val, val_reason, val_threshold = check_val_loss_converged(
                        val_losses, window=loss_stability_window_epochs,
                    )
                    if should_stop_val:
                        stop_reason = (
                            f"{val_reason} (threshold={val_threshold:.4e}, epoch={epoch_num})"
                        )
                        logger.info("Val loss stop at iter %d: %s", it + 1, stop_reason)
                        break

                # 4. Weyl adapter-saturation monitoring
                max_ratio = None
                budget_exhausted_flag = False
                median_budget_ratio = None
                projected_residual_max = None
                expert_saturation_map: dict[str, float] | None = None
                per_layer_budget_ratio: dict[str, float] | None = None
                per_layer_remaining_budget: dict[str, float] | None = None
                n_saturated_experts: int | None = None
                n_total_target_experts: int | None = None
                all_target_experts_saturated = False

                if use_pissa_lora:
                    # PiSSA LoRA: no hard spectral bound. MASS step sizing
                    # bounds per-step displacement; val_loss convergence stops.
                    # Budget monitoring for PiSSA requires tracking displacement
                    # from initialization — deferred to follow-up.
                    pass
                else:
                    # NB-LoRA: bounded by construction (||BA||₂ ≤ σ_k via Cayley).
                    # Monitor capacity usage: ||BA||₂/σ_k → 1.0.
                    try:
                        lora_products = []
                        lora_module_names: list[str] = []
                        for name, nb_lora in self._iter_nb_lora_modules(model):
                            A, B = nb_lora._cayley_transform()
                            S = mx.clip(nb_lora.S_raw, 0.0, nb_lora._scale_bound)
                            lora_products.append((
                                2.0,
                                (S[:, None] * A).T,  # [in, r]
                                B,                    # [r, out]
                                nb_lora._scale_bound,
                            ))
                            lora_module_names.append(name)
                            mx.eval(A, B, S)

                        ratios = compute_budget_ratios(
                            lora_products, self._backend,
                        )
                        if ratios:
                            budget_exhausted_flag, median_budget_ratio = is_budget_exhausted(
                                ratios,
                                threshold=DTYPE_THRESHOLD_F32,
                            )
                            max_ratio = max(ratios)
                            if len(ratios) == len(lora_module_names):
                                per_layer_budget_ratio = {
                                    module_name: float(ratio)
                                    for module_name, ratio in zip(lora_module_names, ratios)
                                }
                                per_layer_remaining_budget = {}
                                for module_name, ratio in zip(lora_module_names, ratios):
                                    module_ref = next(
                                        (
                                            nb
                                            for name, nb in self._iter_nb_lora_modules(model)
                                            if name == module_name
                                        ),
                                        None,
                                    )
                                    if module_ref is not None:
                                        per_layer_remaining_budget[module_name] = max(
                                            0.0,
                                            float(module_ref._scale_bound) * (1.0 - float(ratio)),
                                        )

                            if len(ratios) == len(lora_module_names):
                                per_expert: dict[str, float] = {}
                                for module_name, ratio in zip(lora_module_names, ratios):
                                    expert_key = self._expert_key_from_layer_key(module_name)
                                    if expert_key is None:
                                        continue
                                    existing = per_expert.get(expert_key)
                                    if existing is None or ratio > existing:
                                        per_expert[expert_key] = float(ratio)
                                if per_expert:
                                    expert_saturation_map = dict(sorted(per_expert.items()))
                                    n_total_target_experts = len(expert_saturation_map)
                                    n_saturated_experts = sum(
                                        1
                                        for ratio in expert_saturation_map.values()
                                        if ratio >= DTYPE_THRESHOLD_F32
                                    )
                                    all_target_experts_saturated = (
                                        n_total_target_experts > 0
                                        and n_saturated_experts == n_total_target_experts
                                    )
                                    logger.info(
                                        "Expert capacity: saturated=%d/%d, headroom=%d",
                                        n_saturated_experts,
                                        n_total_target_experts,
                                        n_total_target_experts - n_saturated_experts,
                                    )

                        # Update conformal margin from fresh spectral measurement
                        if max_ratio is not None:
                            remaining_budget = max(
                                0.0, sigma_k_min * (1.0 - max_ratio),
                            )

                        # Projected residual diagnostic (tighter than spectral norm)
                        base_u_ks = []
                        base_v_ks = []
                        for _name, nb in self._iter_nb_lora_modules(model):
                            if nb.base_u_k is not None and nb.base_v_k is not None:
                                base_u_ks.append(nb.base_u_k)
                                base_v_ks.append(nb.base_v_k)
                        if base_u_ks and len(base_u_ks) == len(lora_products):
                            proj_residuals = compute_projected_residuals(
                                lora_products, base_u_ks, base_v_ks,
                                self._backend,
                            )
                            if proj_residuals:
                                projected_residual_max = max(proj_residuals)
                    except Exception:
                        raise RuntimeError(
                            "Adapter spectral-budget monitoring failed. "
                            "Weyl bound exists to prevent spectral damage — "
                            "cannot continue training without budget verification."
                        )

                # 5. Entropy and repetition probe
                mean_entropy, rep_rate = self._probe_entropy_and_repetition(
                    model, tokenizer, readout_erank=readout_erank,
                )

                # 5a. Spectral-ratio growth rate (per-iter perturbation slope)
                spectral_ratio_growth_per_iter = None
                if (
                    max_ratio is not None
                    and last_max_spectral_ratio is not None
                    and check_interval > 0
                ):
                    spectral_ratio_growth_per_iter = (
                        max_ratio - last_max_spectral_ratio
                    ) / float(check_interval)
                if max_ratio is not None:
                    last_max_spectral_ratio = max_ratio

                # 5b. Online correctness evaluation (optional)
                online_eval_acc = None
                online_eval_n_correct = None
                online_eval_n_total = None
                online_eval_degraded = None
                online_eval_degraded_raw = None
                online_eval_degraded_significant = None
                online_eval_alpha = None
                online_eval_current_ci_lower = None
                online_eval_current_ci_upper = None
                online_eval_baseline_ci_lower = None
                online_eval_baseline_ci_upper = None
                online_eval_n_lost = None
                online_eval_n_gained = None
                online_eval_per_type_correct = None
                online_eval_per_type_total = None
                online_eval_stop_basis_acc = None
                online_eval_stop_basis_n_correct = None
                online_eval_stop_basis_n_total = None
                online_eval_stop_basis_degraded = None
                online_eval_stop_basis_degraded_raw = None
                online_eval_stop_basis_degraded_significant = None
                online_eval_stop_basis_alpha = None
                online_eval_stop_basis_current_ci_lower = None
                online_eval_stop_basis_current_ci_upper = None
                online_eval_stop_basis_baseline_ci_lower = None
                online_eval_stop_basis_baseline_ci_upper = None
                gate_confound_event = False
                eval_result = None
                if online_eval_problems and tokenizer is not None:
                    from modelcypher.core.domain.training.online_eval import (
                        evaluate_correctness,
                    )

                    def _generate_fn(prompt: str, max_toks: int) -> str:
                        return self._backend.generate(
                            model, tokenizer, prompt, max_toks,
                        )

                    eval_result = evaluate_correctness(
                        problems=online_eval_problems,
                        generate_fn=_generate_fn,
                        epoch=epoch_num,
                        baseline_correct_ids=online_eval_baseline_ids,
                        max_tokens=seq_length,
                    )
                    online_eval_acc = eval_result.accuracy
                    online_eval_n_correct = eval_result.n_correct
                    online_eval_n_total = eval_result.n_total
                    online_eval_degraded = eval_result.degraded
                    online_eval_degraded_raw = eval_result.degraded_raw
                    online_eval_degraded_significant = eval_result.degraded_significant
                    online_eval_alpha = eval_result.alpha
                    online_eval_current_ci_lower = eval_result.current_ci_lower
                    online_eval_current_ci_upper = eval_result.current_ci_upper
                    online_eval_baseline_ci_lower = eval_result.baseline_ci_lower
                    online_eval_baseline_ci_upper = eval_result.baseline_ci_upper
                    online_eval_n_lost = eval_result.n_lost
                    online_eval_n_gained = eval_result.n_gained
                    online_eval_per_type_correct = dict(eval_result.per_type_correct)
                    online_eval_per_type_total = dict(eval_result.per_type_total)
                    online_eval_stop_basis_acc = online_eval_acc
                    online_eval_stop_basis_n_correct = online_eval_n_correct
                    online_eval_stop_basis_n_total = online_eval_n_total
                    online_eval_stop_basis_degraded = online_eval_degraded
                    online_eval_stop_basis_degraded_raw = online_eval_degraded_raw
                    online_eval_stop_basis_degraded_significant = (
                        online_eval_degraded_significant
                    )
                    online_eval_stop_basis_alpha = online_eval_alpha
                    online_eval_stop_basis_current_ci_lower = online_eval_current_ci_lower
                    online_eval_stop_basis_current_ci_upper = online_eval_current_ci_upper
                    online_eval_stop_basis_baseline_ci_lower = online_eval_baseline_ci_lower
                    online_eval_stop_basis_baseline_ci_upper = online_eval_baseline_ci_upper

                gate_confound_event = False
                online_eval_stop_basis_stage = "pre_outcome"
                behavioral_state = _behavioral_probe_state(
                    per_layer_budget_ratio=per_layer_budget_ratio,
                    per_layer_remaining_budget=per_layer_remaining_budget,
                    online_eval_accuracy=online_eval_acc,
                    online_eval_n_lost=online_eval_n_lost,
                    online_eval_n_gained=online_eval_n_gained,
                )

                # 6. Collect epoch metrics
                epoch_metrics_list.append(EpochMetrics(
                    epoch=epoch_num,
                    train_loss=loss_val,
                    val_loss=v_loss,
                    eta=current_eta,
                    update_norm=update_norm,
                    max_spectral_ratio=max_ratio,
                    spectral_ratio_growth_per_iter=spectral_ratio_growth_per_iter,
                    mean_token_entropy=mean_entropy,
                    repetition_rate=rep_rate,
                    elapsed_seconds=epoch_elapsed,
                    eta_ceiling=eta_ceiling if adaptive_lr else None,
                    adapter_saturation_median_ratio=median_budget_ratio,
                    expert_saturation_map=expert_saturation_map,
                    n_saturated_experts=n_saturated_experts,
                    n_total_target_experts=n_total_target_experts,
                    displacement=mass_metrics.get("displacement"),
                    eta_sps=mass_metrics.get("eta_sps"),
                    eta_weyl=mass_metrics.get("eta_weyl"),
                    eta_step=mass_metrics.get("eta_step"),
                    d_norm=mass_metrics.get("d_norm"),
                    eta_margin=mass_metrics.get("eta_margin"),
                    remaining_budget=remaining_budget,
                    max_displacement_to_remaining=(
                        max_disp_to_remaining_this_epoch
                        if max_disp_to_remaining_this_epoch > 0 else None
                    ),
                    max_effective_gain_ratio=(
                        max_effective_gain_this_epoch
                        if max_effective_gain_this_epoch > 0 else None
                    ),
                    online_eval_accuracy=online_eval_acc,
                    online_eval_n_correct=online_eval_n_correct,
                    online_eval_n_total=online_eval_n_total,
                    online_eval_degraded=online_eval_degraded,
                    online_eval_degraded_raw=online_eval_degraded_raw,
                    online_eval_degraded_significant=online_eval_degraded_significant,
                    online_eval_alpha=online_eval_alpha,
                    online_eval_current_ci_lower=online_eval_current_ci_lower,
                    online_eval_current_ci_upper=online_eval_current_ci_upper,
                    online_eval_baseline_ci_lower=online_eval_baseline_ci_lower,
                    online_eval_baseline_ci_upper=online_eval_baseline_ci_upper,
                    online_eval_pre_accuracy=online_eval_acc,
                    online_eval_pre_n_correct=online_eval_n_correct,
                    online_eval_pre_n_total=online_eval_n_total,
                    online_eval_pre_degraded=online_eval_degraded,
                    online_eval_pre_degraded_raw=online_eval_degraded_raw,
                    online_eval_pre_degraded_significant=online_eval_degraded_significant,
                    online_eval_pre_alpha=online_eval_alpha,
                    online_eval_pre_current_ci_lower=online_eval_current_ci_lower,
                    online_eval_pre_current_ci_upper=online_eval_current_ci_upper,
                    online_eval_pre_baseline_ci_lower=online_eval_baseline_ci_lower,
                    online_eval_pre_baseline_ci_upper=online_eval_baseline_ci_upper,
                    online_eval_pre_n_lost=online_eval_n_lost,
                    online_eval_pre_n_gained=online_eval_n_gained,
                    online_eval_pre_per_type_correct=online_eval_per_type_correct,
                    online_eval_pre_per_type_total=online_eval_per_type_total,
                    online_eval_stop_basis_accuracy=online_eval_stop_basis_acc,
                    online_eval_stop_basis_n_correct=online_eval_stop_basis_n_correct,
                    online_eval_stop_basis_n_total=online_eval_stop_basis_n_total,
                    online_eval_stop_basis_degraded=online_eval_stop_basis_degraded,
                    online_eval_stop_basis_degraded_raw=online_eval_stop_basis_degraded_raw,
                    online_eval_stop_basis_degraded_significant=(
                        online_eval_stop_basis_degraded_significant
                    ),
                    online_eval_stop_basis_alpha=online_eval_stop_basis_alpha,
                    online_eval_stop_basis_current_ci_lower=(
                        online_eval_stop_basis_current_ci_lower
                    ),
                    online_eval_stop_basis_current_ci_upper=(
                        online_eval_stop_basis_current_ci_upper
                    ),
                    online_eval_stop_basis_baseline_ci_lower=(
                        online_eval_stop_basis_baseline_ci_lower
                    ),
                    online_eval_stop_basis_baseline_ci_upper=(
                        online_eval_stop_basis_baseline_ci_upper
                    ),
                    online_eval_stop_basis_stage=online_eval_stop_basis_stage,
                    gate_confound_event=gate_confound_event,
                    projected_residual_max=projected_residual_max,
                    controller_mode=controller_mode,
                    optimizer_research_mode=optimizer_research_mode,
                    controller_trace={
                        "step_traces": list(epoch_step_traces),
                        "behavioral_state": (
                            behavioral_state.to_dict()
                            if behavioral_state is not None
                            else None
                        ),
                    },
                ))

                # 6b. Topological phase metrics (optional)
                if topo_monitor and tokenizer is not None:
                    _topo_probes = topo_probe_texts or [
                        "The", "Once upon a time", "In the beginning",
                        "What is", "The answer is",
                    ]
                    topo_m = self._compute_topological_metrics(
                        model, tokenizer, _topo_probes,
                    )
                    if topo_m:
                        em = epoch_metrics_list[-1]
                        em.topo_betti_0 = topo_m.get("topo_betti_0")
                        em.topo_betti_1 = topo_m.get("topo_betti_1")
                        em.topo_persistence_entropy = topo_m.get(
                            "topo_persistence_entropy",
                        )
                        em.topo_mean_ricci_curvature = topo_m.get(
                            "topo_mean_ricci_curvature",
                        )
                        em.topo_ricci_curvature_std = topo_m.get(
                            "topo_ricci_curvature_std",
                        )
                        logger.info(
                            "Topo: B0=%s B1=%s PE=%.4f Ricci=%.4f±%.4f",
                            em.topo_betti_0,
                            em.topo_betti_1,
                            em.topo_persistence_entropy or 0.0,
                            em.topo_mean_ricci_curvature or 0.0,
                            em.topo_ricci_curvature_std or 0.0,
                        )

                # 6c. Dimensional expansion monitoring (optional)
                if dim_monitor and tokenizer is not None:
                    dim_snapshot = self._compute_dimensional_snapshot(
                        model, tokenizer,
                        dim_probe_texts or ["The", "Once upon a time", "In the beginning",
                                           "What is", "The answer is"],
                        epoch_num,
                    )
                    if dim_snapshot is not None:
                        from modelcypher.core.domain.training.dimensional_monitor import (
                            compute_null_space_recruitment,
                        )

                        em = epoch_metrics_list[-1]
                        em.dim_expansion_ratio = dim_snapshot.expansion_ratio
                        em.dim_peak_dim = dim_snapshot.peak_dim
                        em.dim_final_dim = dim_snapshot.final_dim
                        used_fraction = dim_snapshot.final_used_fraction
                        null_fraction = dim_snapshot.final_null_fraction
                        if used_fraction == used_fraction:
                            em.dim_final_used_fraction = used_fraction
                        if null_fraction == null_fraction:
                            em.dim_final_null_fraction = null_fraction
                        dim_snapshots.append(dim_snapshot)
                        baseline_snapshot = dim_snapshots[0]
                        recruitment = compute_null_space_recruitment(
                            baseline_snapshot, dim_snapshot,
                        )
                        if recruitment == recruitment:
                            em.dim_null_recruitment_from_baseline = recruitment
                        if len(dim_snapshots) >= 2:
                            from modelcypher.core.domain.training.dimensional_monitor import (
                                assess_trend,
                            )
                            trend = assess_trend(dim_snapshots)
                            em.dim_delta_from_baseline = trend.delta
                            em.dim_is_contracting = trend.is_contracting
                            if trend.is_contracting:
                                logger.warning(
                                    "DIMENSIONAL CONTRACTION: expansion_ratio %.3f → %.3f (Δ=%.3f)",
                                    trend.baseline_expansion_ratio,
                                    trend.current_expansion_ratio,
                                    trend.delta,
                                )
                        logger.info(
                            "Dim: exp_ratio=%.3f peak=%.1f final=%.1f",
                            dim_snapshot.expansion_ratio,
                            dim_snapshot.peak_dim,
                            dim_snapshot.final_dim,
                        )

                # 6d. Constraint diagnostics (constrained training mode)
                if use_constrained and constraint_state is not None:
                    em = epoch_metrics_list[-1]
                    em.constraint_mu_inv = constraint_state.mu_inv
                    em.constraint_mu_sep = constraint_state.mu_sep
                    em.constraint_mu_geo = constraint_state.mu_geo
                    em.constraint_C_inv = constraint_state.last_C_inv
                    em.constraint_C_sep = constraint_state.last_C_sep
                    em.constraint_C_geo = constraint_state.last_C_geo
                    logger.info(
                        "Constraints: μ_inv=%.3f μ_sep=%.3f μ_geo=%.3f "
                        "C_inv=%.4f C_sep=%.4f C_geo=%.4f",
                        constraint_state.mu_inv,
                        constraint_state.mu_sep,
                        constraint_state.mu_geo,
                        constraint_state.last_C_inv,
                        constraint_state.last_C_sep,
                        constraint_state.last_C_geo,
                    )

                # 6e. Geometric reshaping diagnostics
                if geometric_reshape and hasattr(loss_fn, "component_metrics"):
                    cm = loss_fn.component_metrics
                    cw = getattr(loss_fn, "component_weights", {})
                    em = epoch_metrics_list[-1]
                    em.reshape_ce_norm = float(cm.get("ce_norm", 0))
                    em.reshape_expand_norm = float(cm.get("expand_norm", 0))
                    em.reshape_contrast_norm = float(cm.get("contrast_norm", 0))
                    em.reshape_n_cf_pairs = int(cm.get("n_cf_pairs", 0))
                    em.reshape_n_inv_pairs = int(cm.get("n_inv_pairs", 0))
                    alpha_val = float(cm.get("alpha", 0))
                    logger.info(
                        "Reshape: α=%.3f ce=%.3f expand=%.3f(w=%.1f) "
                        "contrast=%.3f(w=%.1f) cf=%d inv=%d",
                        alpha_val,
                        em.reshape_ce_norm,
                        em.reshape_expand_norm,
                        cw.get("expand", 1.0),
                        em.reshape_contrast_norm,
                        cw.get("contrast", 1.0),
                        em.reshape_n_cf_pairs,
                        em.reshape_n_inv_pairs,
                    )

                # 6f. Outer similarity monitoring (RSS — Kucukahmetler et al. 2026)
                if rss_monitor and tokenizer is not None and base_activations is not None:
                    rss_result = self._compute_rss_metrics(
                        model, tokenizer, base_activations, eval_dataset,
                    )
                    if rss_result is not None:
                        em = epoch_metrics_list[-1]
                        em.rss_cosine = rss_result.cosine_rss
                        em.rss_spearman = rss_result.spearman_rank
                        em.rss_top1_agreement = rss_result.top1_agreement
                        logger.info(
                            "RSS: cos=%.4f spearman=%.4f top1=%.4f",
                            rss_result.cosine_rss,
                            rss_result.spearman_rank,
                            rss_result.top1_agreement,
                        )

                # Log
                log_parts = [
                    f"Epoch {epoch_num} | train_loss={loss_val:.4f}",
                ]
                if v_loss is not None:
                    log_parts.append(f"val_loss={v_loss:.4f}")
                log_parts.append(f"eta={current_eta:.2e}")
                log_parts.append(f"eta_ceiling={eta_ceiling:.2e}")
                if update_norm is not None:
                    log_parts.append(f"‖Δθ‖={update_norm:.4f}")
                if max_ratio is not None:
                    log_parts.append(f"spectral={max_ratio:.4f}")
                if median_budget_ratio is not None:
                    log_parts.append(f"adapter_sat={median_budget_ratio:.4f}")
                if remaining_budget is not None:
                    log_parts.append(f"remaining={remaining_budget:.4e}")
                if mass_metrics.get("eta_margin") is not None:
                    log_parts.append(f"η_margin={mass_metrics['eta_margin']:.2e}")
                if mean_entropy is not None:
                    log_parts.append(f"entropy={mean_entropy:.2f}")
                if rep_rate is not None:
                    log_parts.append(f"rep={rep_rate:.3f}")
                if mass_metrics:
                    es = mass_metrics.get("eta_step", 0)
                    disp = mass_metrics.get("displacement", 0)
                    dn = mass_metrics.get("d_norm", 0)
                    e_sps = mass_metrics.get("eta_sps", 0)
                    e_weyl = mass_metrics.get("eta_weyl", 0)
                    log_parts.append(f"η_eff={es:.2e}")
                    log_parts.append(f"η_sps={e_sps:.2e}")
                    log_parts.append(f"η_weyl={e_weyl:.2e}")
                    log_parts.append(f"‖g‖={dn:.4f}")
                    log_parts.append(f"disp={disp:.4e}")
                logger.info(" | ".join(log_parts))

                # 7a. Weyl adapter-saturation exhaustion check (any layer crossing)
                if budget_exhausted_flag:
                    stop_reason = (
                        f"adapter_saturation_exhausted (Weyl crossing, "
                        f"median_ratio={median_budget_ratio:.4f}, epoch={epoch_num})"
                    )
                    logger.info(
                        "Adapter saturation stop at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                if all_target_experts_saturated:
                    stop_reason = (
                        "moe_expert_saturation_exhausted "
                        f"(saturated={n_saturated_experts}/{n_total_target_experts}, "
                        f"epoch={epoch_num})"
                    )
                    logger.info(
                        "MoE expert saturation stop at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                # 7a'. Budget cap: stop when median ratio exceeds user ceiling
                if (
                    budget_cap is not None
                    and median_budget_ratio is not None
                    and median_budget_ratio >= budget_cap
                ):
                    stop_reason = (
                        f"adapter_saturation_cap (median_ratio={median_budget_ratio:.4f} "
                        f">= cap={budget_cap:.4f}, epoch={epoch_num})"
                    )
                    logger.info(
                        "Adapter saturation cap at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                # 7a''. Max epochs: hard cap to prevent stop-signal erosion
                if max_epochs is not None and epoch_num >= max_epochs:
                    stop_reason = (
                        f"max_epochs (epoch={epoch_num} >= cap={max_epochs})"
                    )
                    logger.info(
                        "Epoch cap at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                # 7b. Geometric stopping certificate
                if (
                    use_val_stopping
                    and epoch_num >= 2
                    and grad_last is not None
                ):
                    certificate = self._compute_certificate_quantities(
                        model=model,
                        grad=grad_last,
                        eval_dataset=eval_dataset,
                        batch_size=eval_batch_size,
                        seq_length=seq_length,
                        n_batches=eval_batches,
                        mean_token_entropy=mean_entropy,
                        repetition_rate=rep_rate,
                        grad_norm_history=grad_norm_history,
                        seed=seed,
                        val_loss_baseline=val_loss_baseline,
                        val_loss_current=val_losses[-1] if val_losses else None,
                    )
                    # Append this epoch's gradient norm to history
                    grad_norm_history.append(certificate.grad_norm)
                    # Update epoch metrics with certificate fields
                    epoch_metrics_list[-1] = EpochMetrics(
                        **{
                            **epoch_metrics_list[-1].to_dict(),
                            "cert_grad_norm": certificate.grad_norm,
                            "cert_alignment": certificate.alignment,
                            "cert_curvature": certificate.curvature,
                            "cert_delta_max_val": certificate.delta_max_val,
                            "cert_val_ci_half_width": certificate.val_ci_half_width,
                            "cert_delta_max_worst": certificate.delta_max_worst,
                            "cert_task_improvement_met": certificate.task_improvement_met,
                            "cert_all_met": certificate.all_conditions_met,
                        }
                    )
                    logger.info(
                        "Certificate: ‖g‖=%.2e SE=%.2e stat=%s | "
                        "a=%.2e b=%.2e Δmax=%.2e CI=%.2e | "
                        "worst=%.2e | drift=%s | task_imp=%s | met=%s",
                        certificate.grad_norm,
                        certificate.stationarity_floor,
                        certificate.stationarity_met,
                        certificate.alignment,
                        certificate.curvature,
                        certificate.delta_max_val,
                        certificate.val_ci_half_width,
                        certificate.delta_max_worst,
                        "none" if certificate.no_drift else "DETECTED",
                        certificate.task_improvement_met,
                        certificate.all_conditions_met,
                    )
                    if should_certificate_stop(
                        certificate.all_conditions_met, val_losses,
                    ):
                        stop_reason = (
                            f"certificate (‖g‖={certificate.grad_norm:.2e}, "
                            f"Δmax={certificate.delta_max_val:.2e}"
                            f"<CI={certificate.val_ci_half_width:.2e}, "
                            f"epoch={epoch_num})"
                        )
                        logger.info(
                            "Certificate stop at iter %d: %s",
                            it + 1, stop_reason,
                        )
                        break
                    elif certificate.all_conditions_met:
                        logger.info(
                            "Certificate met at epoch %d but val_loss "
                            "improved (%.4f → %.4f) — continuing",
                            epoch_num,
                            val_losses[-2],
                            val_losses[-1],
                        )
                # 7c. Online eval degradation stop
                if online_eval_stop_basis_degraded:
                    stop_reason = (
                        f"online_eval_degraded_significant ("
                        f"stage={online_eval_stop_basis_stage}, "
                        f"{online_eval_stop_basis_n_correct}/{online_eval_stop_basis_n_total} correct, "
                        f"epoch={epoch_num})"
                    )
                    logger.info(
                        "Online eval stop at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                # 7d. Degeneration gate: n-gram repetition on few-shot prompts
                if (degen_prompts
                        and degen_baseline_max is not None
                        and degen_ngram_order is not None
                        and tokenizer is not None):
                    from modelcypher.core.domain.training.degeneration import (
                        ngram_repetition_rate,
                    )
                    _sqrt_eps = math.sqrt(
                        float(self._backend.finfo().eps)
                    )
                    _degen_rates: list[float] = []
                    for _dp in degen_prompts:
                        try:
                            _resp = self._backend.generate(
                                model, tokenizer, _dp, max_tokens=512,
                            )
                            _degen_rates.append(
                                ngram_repetition_rate(_resp, degen_ngram_order)
                            )
                        except Exception:
                            pass
                    if _degen_rates:
                        _degen_max = max(_degen_rates)
                        _degen_mean = sum(_degen_rates) / len(_degen_rates)
                        logger.info(
                            "Degeneration check (epoch %d): max_ngram(%d)=%.3f, "
                            "mean=%.3f, baseline_max=%.3f (%d prompts)",
                            epoch_num, degen_ngram_order, _degen_max,
                            _degen_mean,
                            degen_baseline_max, len(_degen_rates),
                        )
                        if _degen_max > degen_baseline_max + _sqrt_eps:
                            stop_reason = (
                                f"degeneration_exceeded ("
                                f"max_ngram({degen_ngram_order})={_degen_max:.3f} > "
                                f"baseline={degen_baseline_max:.3f}+eps, "
                                f"epoch={epoch_num})"
                            )
                            logger.info(
                                "Degeneration stop at iter %d: %s",
                                it + 1, stop_reason,
                            )
                            break

                # 7e. Loss stability stop (fallback when no val dataset)
                if not use_val_stopping and it >= (
                    2 * loss_stability_window_epochs * n_batches_per_epoch
                ):
                    stable, threshold = check_loss_stable(
                        losses, window=loss_stability_window_epochs * n_batches_per_epoch,
                    )
                    if stable:
                        stop_reason = f"loss_stable (|Δ_epoch| < SE = {threshold:.4e})"
                        logger.info(
                            "Training stop at iter %d: %s", it + 1, stop_reason,
                        )
                        break
        else:
            stop_reason = f"safety_cap ({max_iters} iters)"
            logger.error(
                "Hit safety cap at %d iters — geometric stopping certificate "
                "failed to fire. This indicates a convergence failure.",
                max_iters,
            )

        if val_losses:
            logger.info(
                "Validation trajectory: %s",
                " → ".join(f"{v:.4f}" for v in val_losses),
            )
        if epoch_metrics_list:
            logger.info(
                "LR trajectory: %s",
                " → ".join(f"{m.eta:.2e}" for m in epoch_metrics_list),
            )

        # Restore best checkpoint if final val loss regressed
        if best_weights is not None and val_losses:
            last_val = val_losses[-1]
            # Restore only if the regression is numerically distinguishable.
            # fl(a - b) has absolute error eps * max(|a|, |b|) (Higham 2002, §1.2).
            _eps_f32 = math.ldexp(1.0, -23)
            numeric_floor = _eps_f32 * max(abs(last_val), abs(best_val_loss))
            if last_val - best_val_loss > numeric_floor:
                logger.info(
                    "Restoring best checkpoint (val_loss %.4f vs final %.4f)",
                    best_val_loss, last_val,
                )
                # Restore only trainables; avoids missing-parameter failures on
                # hybrid architectures where load_weights expects full tensors.
                model.update(mlx_unflatten(best_weights))
                # Eval only trainable params, not the entire 8B+ base model.
                mx.eval(model.trainable_parameters())
                logger.info("Best checkpoint restored successfully")

        logger.info("train_loop returning (stop_reason=%s)", stop_reason)
        return losses, stop_reason, epoch_metrics_list
