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

"""Training loop methods for :class:`MLXTrainingAdapter`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from modelcypher.backends.mlx_training_adapter_core import *  # noqa: F403

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_lora import LayerGeometry
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
        lr_override: float | None = None,
        eval_dataset: list | None = None,
        eval_batches: int = 10,
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
        # Optional gradient hook: applied after Cayley preconditioner, before optimizer
        gradient_hook: "Callable | None" = None,
        # Anti-degeneration: entropy floor regularization
        entropy_regularization: bool = False,
        # Online correctness evaluation at epoch boundaries
        online_eval_problems: list | None = None,
        online_eval_baseline_ids: "frozenset | None" = None,
        # REINFORCE outcome training (Layer 3)
        outcome_training: bool = False,
        outcome_problems: list | None = None,
        # Answer-span masked CE (train only on answer tokens + EOS)
        answer_masked_dataset: list[tuple[Any, Any, int]] | None = None,
        answer_masked_eval: list[tuple[Any, Any, int]] | None = None,
        # Envelope caps: hard limits to prevent stop-signal erosion
        max_epochs: int | None = None,
        budget_cap: float | None = None,
        # Sub-epoch evaluation: override epoch-based check interval
        eval_interval: int | None = None,
        # Global EOS exclusion: exclude EOS token from CE in all paths
        eos_exclude: bool = False,
        # Outer similarity monitoring (Kucukahmetler et al. 2026)
        rss_monitor: bool = False,
        base_activations: dict | None = None,
        # Ablation experiment params (research only, not CLI-exposed)
        entropy_floor_fraction: float | None = None,
        kl_reference_penalty: bool = False,
        outcome_signal_density_gate: float = 0.0,
    ) -> tuple[list[tuple[int, float, float]], str, list[EpochMetrics]]:
        """Train with ScaledGD, Weyl adapter-saturation monitoring, and geometric stopping.

        Optimizer: ScaledGD preconditioning (Tong et al. JMLR 2021) when opt_config
        is provided. Falls back to plain SGD otherwise. ScaledGD preconditions each
        LoRA factor's gradient through the pseudoinverse of the other factor, giving
        condition-number-free convergence on the rank-r manifold.

        MASS (Measured-Adaptive Step Size) — three-layer system:
        1. Spectral ceiling: eta_ceiling = sigma_k_min / sigma_max (Weyl 1912, static)
        2. Per-step SPS: eta_sps = f(x_t) / ||d_t||^2 (Loizou et al. 2020)
        3. Per-step Weyl: eta_weyl = sigma_k_min / ||d_t|| (displacement bound)
        Combined: eta_step = min(eta_sps, eta_weyl, eta_ceiling)
        Override: lr_override bypasses everything.

        Armijo backtracking (when eta_ceiling is the binding constraint):
        Verifies sufficient decrease along the Cayley retraction curve.
        f(R_X(α·d)) ≤ f(X) - c·α·⟨∇f, d⟩  (c=1e-4, Nocedal & Wright 2006)
        Backtrack factor β=0.5, max 3 attempts. Only triggered when the
        static ceiling is binding — SPS/Weyl already incorporate current
        loss/geometry and rarely overshoot.
        Ref: Absil et al. (2008) Optimization on Manifolds, Th. 4.3.1.

        Stopping (any one triggers):
        1. Validation loss convergence or degradation (overfitting)
        2. Weyl adapter-saturation exhaustion (per-layer spectral crossing)
        3. Training loss stability (fallback if no eval_dataset)
        4. Safety cap (max_iters)

        After each step: clamp S_raw (enforce bound).

        Returns: (losses, stop_reason, epoch_metrics)
        """
        import mlx.optimizers as opt
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten as mlx_unflatten
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
            lr_override=lr_override,
        )
        current_eta = eta_ceiling
        # momentum=0.0 required: Cayley-Riemannian natural gradient assumes
        # vanilla SGD (Amari 1998). Momentum would compound with the pullback
        # metric correction P_left = (I+Z)(I+Z)^T and violate the MASS step
        # size bound (which assumes single-step displacement ≤ eta × ||d||).
        optimizer = opt.SGD(learning_rate=current_eta, momentum=0.0)

        # Cayley-aware Riemannian preconditioning (pullback metric correction).
        # The Cayley transform maps free (A_tilde, B_tilde) to the Stiefel
        # manifold. The pullback metric G = J^T J distorts the Euclidean gradient.
        # We precondition by the dominant left metric factor P_left ≈ M M^T where
        # M = I + Z, correcting for the Cayley transform's curvature in the rank-r
        # rotation subspace. This is a one-sided approximation in free
        # (A_tilde, B_tilde) coordinates (the exact pullback is a block operator).
        # Refs: Amari (1998), Wen & Yin (2013), Li et al. (ICLR 2020).
        use_cayley_precond = True

        losses: list[tuple[int, float, float]] = []
        val_losses: list[float] = []
        epoch_metrics_list: list[EpochMetrics] = []
        last_max_spectral_ratio: float | None = None
        dim_snapshots: list = []  # DimensionalSnapshot history for trend analysis
        stop_reason: str | None = None
        best_val_loss = float("inf")
        best_weights: dict[str, Any] | None = None

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
        else:
            batch_iter = iterate_batches(
                train_dataset, batch_size, seq_length, loop=True, seed=seed,
            )
            n_batches_per_epoch = len(
                list(iterate_batches(train_dataset, batch_size, seq_length, loop=False, seed=seed))
            )
        if n_batches_per_epoch <= 0:
            raise ValueError("Training dataset produced zero batches")

        # MASS √N epoch budget correction (Brownian scaling).
        # The per-step Weyl bound (eta * ||d|| ≤ sigma_k_min) prevents any
        # single step from crossing an eigenvalue boundary. But over N steps
        # per epoch, the accumulated displacement scales as √N * eta * ||d||
        # (random walk). To keep accumulated displacement within sigma_k_min:
        #   √N * eta_ceiling * ||d_typical|| ≤ sigma_k_min
        # This is equivalent to dividing the per-step ceiling by √N:
        #   eta_ceiling_epoch = eta_ceiling_step / √N
        if lr_override is None and n_batches_per_epoch > 1:
            sqrt_n = math.sqrt(n_batches_per_epoch)
            eta_ceiling_before = eta_ceiling
            eta_ceiling = eta_ceiling / sqrt_n
            current_eta = eta_ceiling
            optimizer.learning_rate = mx.array(current_eta)
            logger.info(
                "MASS √N budget: ceiling %.4e / √%d = %.4e",
                eta_ceiling_before, n_batches_per_epoch, eta_ceiling,
            )

        use_val_stopping = (
            (eval_dataset is not None and len(eval_dataset) > 0)
            or (use_answer_mask and answer_masked_eval is not None and len(answer_masked_eval) > 0)
        )
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
        optimizer_name = "Cayley-Riemann" if use_cayley_precond else "SGD"
        logger.info(
            "Training: optimizer=%s, stop=%s, cap=%d, epoch=%d batches, lr=%.2e, mode=%s",
            optimizer_name,
            "certificate" if use_val_stopping else "training loss",
            max_iters, n_batches_per_epoch, current_eta, lr_mode,
        )

        # Track params at epoch start for update_norm
        epoch_start_params: dict[str, Any] | None = None
        epoch_start_time = time.time()

        # Last-step gradients for stopping certificate
        grad_raw_last: Any = None
        grad_precond_last: Any = None
        # Gradient norm history for stochastic stationarity
        grad_norm_history: list[float] = []

        for it in range(max_iters):
            # Snapshot params at epoch start
            if it % n_batches_per_epoch == 0:
                trainable = dict(mlx_flatten(model.trainable_parameters()))
                epoch_start_params = {k: mx.array(v) for k, v in trainable.items()}
                mx.eval(*epoch_start_params.values())
                epoch_start_time = time.time()

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
            else:
                batch, lengths = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)

            # Save raw gradient for stopping certificate (overwritten each step;
            # at epoch boundary, holds the last step's gradient).
            grad_raw_last = grad

            # Cayley-Riemannian preconditioning: correct for pullback metric
            # distortion of the Cayley parameterization (natural gradient).
            # d_t = P_t @ g_t, then η_t = min(SPS, Weyl, ceiling).
            precond_metrics: dict[str, float] = {}
            if use_cayley_precond:
                grad, precond_metrics = self._apply_cayley_preconditioner(
                    model, grad,
                )
                # MASS Layer 2: Per-step measured rates
                # Compute ||d_t|| (preconditioned gradient direction norm)
                d_flat = [p.reshape(-1) for _, p in mlx_flatten(grad) if p.size > 0]
                d_norm_sq = sum(mx.sum(p * p) for p in d_flat)
                mx.eval(d_norm_sq, loss)
                d_norm_val = float(mx.sqrt(d_norm_sq).item())
                loss_float = float(loss)

                if d_norm_val > 0:
                    # SPS (Loizou et al. 2020): eta = f(x) / ||d||^2, f* = 0
                    eta_sps_val = loss_float / (d_norm_val ** 2)
                    # Weyl displacement bound: eta * ||d|| <= sigma_k_min
                    eta_weyl_val = sigma_k_min / d_norm_val
                else:
                    eta_sps_val = eta_ceiling
                    eta_weyl_val = eta_ceiling

                eta_step = min(eta_sps_val, eta_weyl_val, eta_ceiling)
                displacement_val = eta_step * d_norm_val

                precond_metrics["eta_step"] = eta_step
                precond_metrics["eta_sps"] = eta_sps_val
                precond_metrics["eta_weyl"] = eta_weyl_val
                precond_metrics["displacement"] = displacement_val
                precond_metrics["d_norm"] = d_norm_val
                optimizer.learning_rate = mx.array(eta_step)
            else:
                optimizer.learning_rate = mx.array(current_eta)

            # Optional gradient hook (e.g. format bias projection)
            if gradient_hook is not None:
                grad = gradient_hook(grad)

            # Save actual update direction for stopping certificate
            grad_precond_last = grad

            # MASS step size (SPS + Weyl + ceiling) already bounds the step.
            # No Armijo backtracking — every constant in Armijo (c=1e-4,
            # beta=0.5, max_backtracks=3) was a guess.  MASS derives the bound
            # per step from measurement.
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)

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
                    # Use effective step size (after preconditioner bounds),
                    # not the scheduled LR. When Cayley curvature grows,
                    # eta_step < current_eta, so dual updates should slow too.
                    alpha_dual = precond_metrics.get("eta_step", current_eta)
                    constraint_state.dual_update(
                        c_inv_val, c_sep_val, c_geo_val,
                        constraint_config, alpha_dual,
                    )

            # THE constraint: clamp S_raw after every step
            self._clamp_all_scales(model)

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
                    # Track best checkpoint for restoration.
                    # MLX arrays are immutable — optimizer creates new arrays,
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
                if adaptive_lr and lr_override is None:
                    if (v_loss is not None and len(val_losses) >= 2
                            and val_losses[-1] > val_losses[-2]
                            and val_losses[-1] > 0):
                        _BACKOFF_FLOOR = math.ldexp(1.0, -23) ** 0.5  # sqrt(eps_f32)
                        backoff = max(val_losses[-2] / val_losses[-1], _BACKOFF_FLOOR)
                        prev_ceiling = eta_ceiling
                        eta_ceiling = eta_ceiling * backoff
                        current_eta = eta_ceiling
                        logger.info(
                            "Val loss increased (%.4f → %.4f): ceiling backoff=%.3f, "
                            "eta_ceiling %.2e -> %.2e",
                            val_losses[-2], val_losses[-1], backoff,
                            prev_ceiling, eta_ceiling,
                        )

                # 3b. Val loss convergence/overfitting check
                if use_val_stopping and len(val_losses) >= 6:
                    should_stop_val, val_reason, val_threshold = check_val_loss_converged(
                        val_losses, window=3,
                    )
                    if should_stop_val:
                        stop_reason = (
                            f"{val_reason} (threshold={val_threshold:.4e}, epoch={epoch_num})"
                        )
                        logger.info("Val loss stop at iter %d: %s", it + 1, stop_reason)
                        break

                # 4. Weyl adapter-saturation monitoring
                # NB-LoRA is bounded by construction (||BA||₂ ≤ σ_k via Cayley).
                # Per-layer Weyl crossing thresholds (gap/(2σ_k)) apply to unbounded
                # LoRA. For NB-LoRA, we monitor capacity usage: ||BA||₂/σ_k → 1.0.
                # Budget exhaustion means the adapter has consumed its available
                # spectral capacity — further training cannot improve without
                # violating bounds.
                max_ratio = None
                budget_exhausted_flag = False
                median_budget_ratio = None
                projected_residual_max = None
                try:
                    lora_products = []
                    for name, nb_lora in self._iter_nb_lora_modules(model):
                        A, B = nb_lora._cayley_transform()
                        S = mx.clip(nb_lora.S_raw, 0.0, nb_lora._scale_bound)
                        # Product = 2 * A^T @ diag(S) @ B → [in, out]
                        # compute_budget_ratios: product = scale * lora_a @ lora_b
                        lora_products.append((
                            2.0,
                            (S[:, None] * A).T,  # [in, r]
                            B,                    # [r, out]
                            nb_lora._scale_bound,
                        ))
                        mx.eval(A, B, S)

                    ratios = compute_budget_ratios(
                        lora_products, self._backend,
                    )
                    if ratios:
                        # Scalar threshold: capacity exhaustion (ratio → 1.0)
                        budget_exhausted_flag, median_budget_ratio = is_budget_exhausted(
                            ratios,
                            threshold=DTYPE_THRESHOLD_F32,
                        )
                        max_ratio = max(ratios)

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
                    logger.warning(
                        "Adapter spectral-budget monitoring failed; "
                        "falling back to verify_bounds for this epoch.",
                        exc_info=True,
                    )
                    # Fallback: simple verify_bounds
                    try:
                        _, max_ratio, _ = self.verify_bounds(model)
                    except Exception:
                        logger.warning(
                            "Fallback verify_bounds also failed; "
                            "continuing epoch without budget telemetry.",
                            exc_info=True,
                        )

                # 5. Entropy and repetition probe
                mean_entropy, rep_rate = self._probe_entropy_and_repetition(
                    model, tokenizer,
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

                # 5c. REINFORCE outcome training (optional)
                outcome_n_problems_epoch = None
                outcome_n_active_epoch = None
                outcome_signal_density_epoch = None
                outcome_n_steps_epoch = None
                outcome_target_step_norm_epoch = None
                outcome_target_step_source_epoch = None
                outcome_o_eta_epoch = None
                outcome_o_grad_norm_epoch = None
                if outcome_training and outcome_problems and tokenizer is not None:
                    from modelcypher.core.domain.star.prompting import (
                        default_few_shot_examples,
                    )
                    from modelcypher.core.domain.training.outcome_objective import (
                        collect_outcomes,
                    )

                    def _outcome_gen_fn(prompt: str, max_toks: int) -> str:
                        return self._backend.generate(
                            model, tokenizer, prompt, max_toks,
                        )

                    def _outcome_tok_fn(text: str) -> list[int]:
                        return tokenizer.encode(text)

                    # n_variants derived from the number of unique demonstrations
                    # available in the prompting module (currently 3).
                    _n_variants = len(default_few_shot_examples())

                    # Phase A: collect outcomes (eval mode, no gradients)
                    outcome_result = collect_outcomes(
                        problems=outcome_problems,
                        generate_fn=_outcome_gen_fn,
                        tokenize_fn=_outcome_tok_fn,
                        n_variants=_n_variants,
                        max_tokens=seq_length,
                    )

                    # Phase B: REINFORCE gradient steps on nonzero-advantage completions
                    active_completions = [
                        (c.tokens, c.advantage, c.response_start)
                        for c in outcome_result.completions
                        if c.advantage != 0.0
                    ]

                    # Signal density gate: skip REINFORCE when too few problems
                    # contribute gradient (high-noise regime).
                    if (outcome_signal_density_gate > 0.0
                            and outcome_result.signal_density < outcome_signal_density_gate):
                        logger.info(
                            "REINFORCE skipped: signal_density=%.1f%% < gate %.1f%%",
                            outcome_result.signal_density * 100,
                            outcome_signal_density_gate * 100,
                        )
                        active_completions = []

                    n_outcome_steps = 0
                    if active_completions:
                        outcome_batches = prepare_outcome_batches(
                            active_completions, batch_size, seq_length,
                        )

                        # KL reference penalty: snapshot base logits before training
                        if kl_reference_penalty:
                            ref_log_probs_dict = {}
                            for bi, (ob_b, ob_l, _ob_a, _ob_rs) in enumerate(outcome_batches):
                                ref_inputs = ob_b[:, :-1]
                                ref_targets = ob_b[:, 1:]
                                ref_logits = model(ref_inputs)
                                ref_logits = ref_logits.astype(mx.float32)
                                ref_ce = nn.losses.cross_entropy(ref_logits, ref_targets)
                                ref_lp = mx.stop_gradient(-ref_ce)
                                mx.eval(ref_lp)
                                ref_log_probs_dict[bi] = ref_lp

                            # Derive beta geometrically: equal weighting at init.
                            # Run one forward pass to measure initial REINFORCE loss magnitude.
                            _init_loss_fn = make_outcome_loss()
                            _init_loss, _ = _init_loss_fn(
                                model,
                                outcome_batches[0][0],
                                outcome_batches[0][1],
                                outcome_batches[0][2],
                                outcome_batches[0][3],
                            )
                            mx.eval(_init_loss)
                            init_reinforce_mag = abs(float(_init_loss.item()))
                            # beta = |L_reinforce| so that when KL reaches 1 nat,
                            # the penalty equals the REINFORCE loss magnitude.
                            # When magnitude is zero, all outcomes got the same reward —
                            # no REINFORCE signal exists, so KL penalty is zero.
                            kl_beta = init_reinforce_mag
                            logger.info(
                                "KL reference penalty: beta=%.4e (from |L_reinforce|=%.4e)",
                                kl_beta, init_reinforce_mag,
                            )

                            outcome_loss_fn = make_outcome_loss_with_kl(ref_log_probs_dict, kl_beta)

                            # Wrap to match expected signature (inject batch_idx)
                            _batch_counter = [0]

                            def _kl_loss_wrapper(model, batch, lengths, advantages, response_starts):
                                idx = _batch_counter[0]
                                result = outcome_loss_fn(model, batch, lengths, advantages, response_starts, idx)
                                _batch_counter[0] += 1
                                return result

                            outcome_vg = nn.value_and_grad(model, _kl_loss_wrapper)
                        else:
                            outcome_loss_fn = make_outcome_loss()
                            outcome_vg = nn.value_and_grad(model, outcome_loss_fn)

                        # REINFORCE displacement budget: Weyl remainder after CE.
                        # Total epoch displacement must stay within sigma_k_min.
                        # CE consumed update_norm of the budget. REINFORCE gets
                        # the remainder, distributed across sqrt(N_re) steps
                        # (Brownian scaling — same argument as CE's sqrt(N_ce)).
                        n_re = len(outcome_batches)
                        sqrt_n_re = math.sqrt(max(1, n_re))
                        budget_remaining: float | None = None

                        if update_norm is not None and update_norm > 0:
                            budget_remaining = max(0.0, sigma_k_min - update_norm)
                            if budget_remaining <= 0.0:
                                logger.info(
                                    "REINFORCE skipped: CE displacement %.4e >= "
                                    "sigma_k_min %.4e (Weyl budget exhausted)",
                                    update_norm, sigma_k_min,
                                )
                                active_completions = []
                                target_step_norm = 0.0
                                target_step_source = "budget_exhausted"
                            else:
                                target_step_norm = budget_remaining / sqrt_n_re
                                target_step_source = "weyl_remainder"
                        else:
                            # Fallback: no CE displacement measured.
                            # Share sigma_k_min across total steps (CE + RE).
                            n_total = max(1, check_interval) + max(1, n_re)
                            if sigma_k_min > 0:
                                target_step_norm = sigma_k_min / math.sqrt(n_total)
                                target_step_source = "sigma_k_min_shared"
                            else:
                                # sigma_k_min = 0 → adapter rank is saturated,
                                # no spectral headroom for REINFORCE steps.
                                target_step_norm = 0.0
                                target_step_source = "budget_exhausted"

                        logger.info(
                            "REINFORCE budget: sigma_k_min=%.4e, "
                            "ce_displacement=%.4e, remaining=%.4e, "
                            "n_re=%d, target_step=%.4e (%s)",
                            sigma_k_min,
                            update_norm if update_norm is not None else 0.0,
                            budget_remaining if budget_remaining is not None else sigma_k_min,
                            n_re,
                            target_step_norm,
                            target_step_source,
                        )

                        from mlx.utils import tree_flatten as _rf_flatten

                        for ob_batch, ob_lengths, ob_advantages, ob_rs in outcome_batches:
                            (o_loss, o_ntoks), o_grad = outcome_vg(
                                model, ob_batch, ob_lengths, ob_advantages, ob_rs,
                            )
                            # Cayley preconditioner: corrects for parameterization
                            # curvature (manifold-aware, loss-independent).
                            # Also reconstructs gradient tree for optimizer.
                            if use_cayley_precond:
                                o_grad, _ = self._apply_cayley_preconditioner(
                                    model, o_grad,
                                )

                            # Measure preconditioned gradient norm
                            o_flat = [
                                p.reshape(-1)
                                for _, p in _rf_flatten(o_grad)
                                if p.size > 0
                            ]
                            if o_flat:
                                o_grad_norm = mx.sqrt(
                                    sum(mx.sum(p * p) for p in o_flat)
                                ).item()
                            else:
                                o_grad_norm = 1.0

                            # Scale LR so ‖η · g‖ ≤ target_step_norm
                            if o_grad_norm > 0:
                                o_eta = min(
                                    current_eta,
                                    target_step_norm / o_grad_norm,
                                )
                            else:
                                o_eta = current_eta

                            optimizer.learning_rate = mx.array(o_eta)
                            optimizer.update(model, o_grad)
                            mx.eval(model.parameters(), optimizer.state)
                            self._clamp_all_scales(model)
                            n_outcome_steps += 1

                    outcome_n_problems_epoch = outcome_result.n_problems
                    outcome_n_active_epoch = len(active_completions)
                    outcome_target_step_norm_epoch = target_step_norm if active_completions else None
                    outcome_target_step_source_epoch = target_step_source if active_completions else None
                    # Capture last-batch values (most representative of final state)
                    if n_outcome_steps > 0:
                        outcome_o_eta_epoch = o_eta
                        outcome_o_grad_norm_epoch = o_grad_norm
                    outcome_signal_density_epoch = outcome_result.signal_density
                    outcome_n_steps_epoch = n_outcome_steps

                    logger.info(
                        "REINFORCE: %d problems, %d completions, "
                        "%d correct, %d incorrect, %d mixed, "
                        "%d active, %d steps, signal=%.1f%%, "
                        "target_step=%.2e (%s)",
                        outcome_result.n_problems,
                        len(outcome_result.completions),
                        outcome_result.n_correct,
                        outcome_result.n_incorrect,
                        outcome_result.n_mixed_problems,
                        len(active_completions),
                        n_outcome_steps,
                        outcome_result.signal_density * 100,
                        target_step_norm,
                        target_step_source,
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
                    precond_lambda_max=precond_metrics.get("precond_lambda_max"),
                    precond_lambda_max_raw=precond_metrics.get("precond_lambda_max_raw"),
                    precond_cond_max=precond_metrics.get("precond_cond_max"),
                    precond_ipz_kappa_upper_max=precond_metrics.get("precond_ipz_kappa_upper_max"),
                    precond_ipz_rel_error_upper_max=precond_metrics.get(
                        "precond_ipz_rel_error_upper_max",
                    ),
                    precond_ipz_warn_fraction=precond_metrics.get("precond_ipz_warn_fraction"),
                    precond_gain_mean=precond_metrics.get("precond_gain_mean"),
                    precond_eta_step=precond_metrics.get("eta_step"),
                    displacement=precond_metrics.get("displacement"),
                    eta_sps=precond_metrics.get("eta_sps"),
                    eta_weyl=precond_metrics.get("eta_weyl"),
                    d_norm=precond_metrics.get("d_norm"),
                    online_eval_accuracy=online_eval_acc,
                    online_eval_n_correct=online_eval_n_correct,
                    online_eval_n_total=online_eval_n_total,
                    online_eval_degraded=online_eval_degraded,
                    outcome_n_problems=outcome_n_problems_epoch,
                    outcome_n_active=outcome_n_active_epoch,
                    outcome_signal_density=outcome_signal_density_epoch,
                    outcome_n_steps=outcome_n_steps_epoch,
                    outcome_target_step_norm=outcome_target_step_norm_epoch,
                    outcome_target_step_source=outcome_target_step_source_epoch,
                    outcome_o_eta=outcome_o_eta_epoch,
                    outcome_o_grad_norm=outcome_o_grad_norm_epoch,
                    outcome_budget_remaining=(
                        max(0.0, sigma_k_min - update_norm)
                        if outcome_n_steps_epoch and outcome_n_steps_epoch > 0
                        and update_norm is not None
                        else None
                    ),
                    projected_residual_max=projected_residual_max,
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
                if mean_entropy is not None:
                    log_parts.append(f"entropy={mean_entropy:.2f}")
                if rep_rate is not None:
                    log_parts.append(f"rep={rep_rate:.3f}")
                if precond_metrics:
                    lm = precond_metrics.get("precond_lambda_max", 0)
                    lmr = precond_metrics.get("precond_lambda_max_raw", 0)
                    cm = precond_metrics.get("precond_cond_max", 0)
                    ipz_kappa = precond_metrics.get("precond_ipz_kappa_upper_max", 0)
                    ipz_rel_err = precond_metrics.get("precond_ipz_rel_error_upper_max", 0)
                    ipz_warn = precond_metrics.get("precond_ipz_warn_fraction", 0)
                    es = precond_metrics.get("eta_step", 0)
                    disp = precond_metrics.get("displacement", 0)
                    dn = precond_metrics.get("d_norm", 0)
                    e_sps = precond_metrics.get("eta_sps", 0)
                    e_weyl = precond_metrics.get("eta_weyl", 0)
                    log_parts.append(f"P:λ={lm:.2f}")
                    if lmr > 0:
                        log_parts.append(f"P:λ_raw={lmr:.2f}")
                    log_parts.append(f"P:κ={cm:.1f}")
                    if ipz_kappa > 0:
                        log_parts.append(f"I+Z:κ≤{ipz_kappa:.1f}")
                    if ipz_rel_err > 0:
                        log_parts.append(f"I+Z:κε≤{ipz_rel_err:.2e}")
                    if ipz_warn > 0:
                        log_parts.append(f"I+Z:warn={ipz_warn:.2f}")
                    log_parts.append(f"η_eff={es:.2e}")
                    log_parts.append(f"η_sps={e_sps:.2e}")
                    log_parts.append(f"η_weyl={e_weyl:.2e}")
                    log_parts.append(f"‖d‖={dn:.4f}")
                    log_parts.append(f"disp={disp:.4e}")
                logger.info(" | ".join(log_parts))
                if precond_metrics.get("precond_ipz_warn_any", 0.0) > 0.0:
                    logger.warning(
                        "Cayley conditioning alert: κ(I+Z)*eps >= sqrt(eps) in %.1f%% of "
                        "preconditioned layers (κ_upper_max=%.2e, κeps_upper_max=%.2e)",
                        100.0 * precond_metrics.get("precond_ipz_warn_fraction", 0.0),
                        precond_metrics.get("precond_ipz_kappa_upper_max", 0.0),
                        precond_metrics.get("precond_ipz_rel_error_upper_max", 0.0),
                    )

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
                    and grad_raw_last is not None
                    and grad_precond_last is not None
                ):
                    certificate = self._compute_certificate_quantities(
                        model=model,
                        grad_raw=grad_raw_last,
                        grad_precond=grad_precond_last,
                        eval_dataset=eval_dataset,
                        batch_size=eval_batch_size,
                        seq_length=seq_length,
                        n_batches=eval_batches,
                        mean_token_entropy=mean_entropy,
                        repetition_rate=rep_rate,
                        grad_norm_history=grad_norm_history,
                        seed=seed,
                    )
                    # Append this epoch's gradient norm to history
                    grad_norm_history.append(certificate.precond_grad_norm)
                    # Update epoch metrics with certificate fields
                    epoch_metrics_list[-1] = EpochMetrics(
                        **{
                            **epoch_metrics_list[-1].to_dict(),
                            "cert_precond_grad_norm": certificate.precond_grad_norm,
                            "cert_alignment": certificate.alignment,
                            "cert_curvature": certificate.curvature,
                            "cert_delta_max_val": certificate.delta_max_val,
                            "cert_val_ci_half_width": certificate.val_ci_half_width,
                            "cert_delta_max_worst": certificate.delta_max_worst,
                            "cert_all_met": certificate.all_conditions_met,
                        }
                    )
                    logger.info(
                        "Certificate: ‖Pg‖=%.2e SE=%.2e stat=%s | "
                        "a=%.2e b=%.2e Δmax=%.2e CI=%.2e | "
                        "worst=%.2e | drift=%s | met=%s",
                        certificate.precond_grad_norm,
                        certificate.stationarity_floor,
                        certificate.stationarity_met,
                        certificate.alignment,
                        certificate.curvature,
                        certificate.delta_max_val,
                        certificate.val_ci_half_width,
                        certificate.delta_max_worst,
                        "none" if certificate.no_drift else "DETECTED",
                        certificate.all_conditions_met,
                    )
                    if certificate.all_conditions_met:
                        stop_reason = (
                            f"certificate (‖Pg‖={certificate.precond_grad_norm:.2e}, "
                            f"Δmax={certificate.delta_max_val:.2e}"
                            f"<CI={certificate.val_ci_half_width:.2e}, "
                            f"epoch={epoch_num})"
                        )
                        logger.info(
                            "Certificate stop at iter %d: %s",
                            it + 1, stop_reason,
                        )
                        break
                # 7c. Online eval degradation stop
                if online_eval_degraded:
                    stop_reason = (
                        f"online_eval_degraded ("
                        f"{online_eval_n_correct}/{online_eval_n_total} correct, "
                        f"epoch={epoch_num})"
                    )
                    logger.info(
                        "Online eval stop at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                elif not use_val_stopping and it >= 6 * n_batches_per_epoch:
                    stable, threshold = check_loss_stable(
                        losses, window=3 * n_batches_per_epoch,
                    )
                    if stable:
                        stop_reason = f"loss_stable (|Δ_epoch| < SE = {threshold:.4e})"
                        logger.info(
                            "Training stop at iter %d: %s", it + 1, stop_reason,
                        )
                        break
        else:
            stop_reason = f"safety_cap ({max_iters} iters)"
            logger.warning("Hit safety cap at %d iters — loss did not stabilize", max_iters)

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

    def verify_bounds(self, model) -> tuple[bool, float, list[dict[str, Any]]]:
        """Verify spectral bounds post-training.

        Should ALWAYS pass. If it doesn't, there's a mathematical bug.

        Returns (all_ok, max_ratio, details).
        """
        details: list[dict[str, Any]] = []
        max_ratio = 0.0

        for name, module in self._iter_nb_lora_modules(model):
            spectral_norm = module.get_spectral_norm()
            theoretical_max = 2.0 * module.scale_bound
            ratio = spectral_norm / theoretical_max if theoretical_max > 0 else float("inf")

            # SVD error bound (Demmel & Kahan 1990): relative error in
            # computed singular values ≤ sqrt(max(m,n)) * eps.
            _eps_f32 = math.ldexp(1.0, -23)
            _max_dim = max(int(module.A_tilde.shape[1]),
                           int(module.B_tilde.shape[1]))
            _svd_tol = math.sqrt(_max_dim) * _eps_f32
            details.append({
                "layer": name,
                "spectral_norm": spectral_norm,
                "theoretical_max": theoretical_max,
                "ratio": ratio,
                "ok": ratio <= 1.0 + _svd_tol,
            })
            max_ratio = max(max_ratio, ratio)

        all_ok = all(d["ok"] for d in details)

        if not all_ok:
            logger.error(
                "SPECTRAL BOUND VIOLATION: max_ratio=%.4f (should be <= 1.0). "
                "This is a mathematical bug in the Cayley transform.",
                max_ratio,
            )
        else:
            logger.info(
                "Spectral bounds verified: %d layers, max_ratio=%.4f (by construction)",
                len(details), max_ratio,
            )

        return all_ok, max_ratio, details

    def save_adapter(
        self,
        model,
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save NB-LoRA adapter in standard LoRA format for compatibility.

        Converts Cayley-parameterized (A_tilde, B_tilde, S_raw) to standard
        (lora_a, lora_b) pairs with scale=1.0. The conversion is exact.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_weights: dict[str, Any] = {}
        target_modules: set[str] = set()
        discovered_ranks: list[int] = []
        per_layer_rank_map: dict[str, int] = {}

        for name, module in self._iter_nb_lora_modules(model):
            lora_a, lora_b = module.to_standard_lora()
            rank = int(lora_a.shape[1])
            discovered_ranks.append(rank)

            key_base = name.replace(".weight", "")
            adapter_weights[f"{key_base}.lora_a"] = lora_a
            adapter_weights[f"{key_base}.lora_b"] = lora_b
            target_modules.add(self._module_name_from_layer_key(name))
            per_layer_rank_map[name] = rank

        if not adapter_weights:
            raise ValueError("No NB-LoRA layers found to export")

        # Pad to global rank for compatibility
        global_rank = max(discovered_ranks)
        for key in list(adapter_weights.keys()):
            arr = adapter_weights[key]
            if key.endswith(".lora_a"):
                # lora_a is [in, r] — pad columns
                if int(arr.shape[1]) < global_rank:
                    pad = mx.zeros((int(arr.shape[0]), global_rank - int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=1)
            elif key.endswith(".lora_b"):
                # lora_b is [r, out] — pad rows
                if int(arr.shape[0]) < global_rank:
                    pad = mx.zeros((global_rank - int(arr.shape[0]), int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=0)

        mx.eval(*adapter_weights.values())

        metadata_str: dict[str, str] | None = None
        if metadata:
            metadata_str = {str(k): str(v) for k, v in metadata.items()}

        weights_path = output_dir / "adapters.safetensors"
        self._backend.save_safetensors(str(weights_path), adapter_weights, metadata=metadata_str)

        config = {
            "fine_tune_type": "lora",
            "num_layers": int(self._backend.get_num_layers(model)),
            "lora_parameters": {
                "rank": int(global_rank),
                "scale": 1.0,
                "dropout": 0.0,
                "keys": sorted(target_modules),
            },
            "target_modules": sorted(target_modules),
            "rank": int(global_rank),
            "per_layer_ranks": per_layer_rank_map,
            "method": "nb_lora_cayley",
        }
        if metadata:
            config["metadata"] = metadata

        config_path = output_dir / "adapter_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        logger.info(
            "Saved NB-LoRA adapter: %d layers, rank=%d, path=%s",
            len(discovered_ranks), global_rank, output_dir,
        )
        return output_dir

    def apply_standard_lora_adapter(self, model, adapter_path: str | Path) -> int:
        """Merge a saved standard LoRA adapter into model weights.

        This applies delta_W = lora_b^T @ lora_a^T to each target layer weight.
        Used for cumulative STaR rounds that continue from prior adapter state.
        """
        adapter_dir = Path(adapter_path).expanduser().resolve()
        weights_path = adapter_dir / "adapters.safetensors"
        if not weights_path.exists():
            weights_path = adapter_dir / "adapter.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"No adapter weights found at {adapter_dir}")

        adapter_weights = self._backend.load_safetensors(str(weights_path))
        merged_layers = 0

        for key in sorted(adapter_weights.keys()):
            if not key.endswith(".lora_a"):
                continue
            key_base = key[:-7]
            key_b = f"{key_base}.lora_b"
            if key_b not in adapter_weights:
                continue

            layer_key = f"{key_base}.weight"
            try:
                parent, attr_name = self._resolve_parent_and_attr(model, layer_key)
                linear = getattr(parent, attr_name)
            except Exception:
                logger.warning("Skipping adapter merge for unresolved layer %s", layer_key)
                continue

            if not hasattr(linear, "weight"):
                logger.warning("Skipping adapter merge for non-linear layer %s", layer_key)
                continue

            lora_a = adapter_weights[key]
            lora_b = adapter_weights[key_b]

            # LoRA forward: x @ lora_a @ lora_b
            # Weight delta for [out, in] weight layout: lora_b^T @ lora_a^T
            delta = mx.matmul(mx.transpose(lora_b), mx.transpose(lora_a))
            delta = mx.astype(delta, linear.weight.dtype)
            linear.weight = linear.weight + delta
            mx.eval(linear.weight)
            merged_layers += 1

        logger.info(
            "Applied prior adapter: %d layers merged from %s",
            merged_layers,
            adapter_dir,
        )
        return merged_layers

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _clamp_all_scales(self, model) -> None:
        """Clamp S_raw in all NBLoRALinear modules after optimizer step."""
        for _, module in self._iter_nb_lora_modules(model):
            module.clamp_scale()
            mx.eval(module.S_raw)

    def _apply_cayley_preconditioner(
        self, model, grad,
    ) -> tuple[Any, dict[str, float]]:
        """Cayley-aware Riemannian preconditioning for NB-LoRA gradients.

        The Cayley transform maps free (A_tilde, B_tilde) to semi-orthogonal
        (A, B) via W = (I + Z)^{-1}. The exact pullback metric in
        (A_tilde, B_tilde) coordinates is a block operator; this routine uses
        the dominant one-sided rank-r factor approximation:

            P_left = M M^T, where M = I + Z,

        and applies d = P_left @ g to A_tilde/B_tilde gradients.

        The preconditioner P = M M^T is normalized by its spectral radius:

            P_hat = P / λ_max(P)

        This preserves anisotropy (eigenvalue ratios) while fixing global
        scale. For a scalar c>0, using cP with step η is equivalent to P with
        step cη, so spectral normalization does not change directions, only
        step units.

        Active step-size control is handled outside this helper by MASS:
            η_step = min(η_ceiling, η_sps, η_weyl)

        Historical Lipschitz invariants (for example m = η * L * λ_max(P_hat))
        are retained only as deprecated telemetry compatibility context.
        Raw λ_max(P) is still logged for diagnostics.

        Properties:
        - No mx.linalg.inv needed (M M^T is a product, not an inverse)
        - Always positive definite: M M^T = I + 2 Y^T Y + Z Z^T
        - r×r cost (same as the Cayley transform's own matrix ops)
        - NOT in autograd path (applied to gradients post-backward)
        - λ_max from power iteration on r×r matrix (dynamic convergence)

        Returns:
            (preconditioned_grad, metrics) where metrics["precond_lambda_max"]
            is the max λ_max of normalized preconditioners (1.0 by construction)
            and ``precond_lambda_max_raw`` tracks the unnormalized value.

        References:
            Amari (1998). Natural gradient: G^{-1} @ grad.
            Wen & Yin (2013). Cayley retraction on Stiefel manifold.
            Li et al. (ICLR 2020). Cayley SGD with convergence proof.
            Nesterov (2004). Stability bound: η ≤ 2/(L * λ_max(P)).
        """
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

        grad_flat = dict(mlx_flatten(grad))

        # Per-layer metrics (aggregated at the end)
        all_lambda_max: list[float] = []
        all_lambda_max_raw: list[float] = []
        all_cond: list[float] = []
        all_gain: list[float] = []  # ||Pg|| / ||g||
        all_ipz_kappa_upper: list[float] = []
        all_ipz_rel_error_upper: list[float] = []
        ipz_warn_count = 0

        from modelcypher.core.domain.geometry.numerical_stability import division_epsilon, machine_epsilon

        div_eps_val = float(division_epsilon(self._backend, mx.array([1.0])))
        eps_val = float(machine_epsilon(self._backend, mx.array([1.0])))
        tol = math.sqrt(eps_val)

        for name, nb_lora in self._iter_nb_lora_modules(model):
            prefix = name.replace(".weight", "")
            a_key, b_key = None, None
            for k in grad_flat:
                if k.endswith("A_tilde") and prefix.replace("model.", "") in k:
                    a_key = k
                elif k.endswith("B_tilde") and prefix.replace("model.", "") in k:
                    b_key = k

            if a_key is None or b_key is None:
                continue

            r = nb_lora._rank

            # Compute Z from current free parameters (same math as _cayley_transform)
            stacked = mx.concatenate([nb_lora.A_tilde.T, nb_lora.B_tilde.T], axis=0)
            X = stacked[:r, :]
            Y = stacked[r:, :]
            Z = (X - X.T) + Y.T @ Y  # [r, r]

            # M = I + Z
            M = mx.eye(r) + Z  # [r, r]

            # P = M M^T (one-sided inverse-metric factor, NO normalization)
            P = M @ M.T  # [r, r]

            # λ_max(P) via power iteration on the r×r SPD matrix (dynamic convergence)
            v = mx.ones((r, 1)) / math.sqrt(r)
            mx.eval(v)
            lam = 1.0
            
            # Use dynamic numerical bounds instead of hardcoded iterations and 1e-30.
            
            lam_prev = -1.0
            while True:
                u = P @ v
                mx.eval(u)
                lam = float(mx.sum(v * u))  # Rayleigh quotient
                norm_u = float(mx.sqrt(mx.sum(u * u)))
                if norm_u < div_eps_val:
                    break
                    
                if lam_prev >= 0:
                    diff = abs(lam - lam_prev)
                    if diff < tol * max(1.0, lam):
                        break
                lam_prev = lam
                
                v = u * (1.0 / norm_u)
                mx.eval(v)
            lambda_max_raw = max(lam, 1.0)  # Floor at 1 (P = I at init)

            # Condition number: λ_max / λ_min
            # P = M M^T = I + 2 Y^T Y + Z Z^T, so eigenvalues ≥ 1 always.
            # For tighter bound: λ_min ≥ trace(P) - (r-1)*λ_max
            tr = float(mx.trace(P))
            lambda_min = max(tr - (r - 1) * lambda_max_raw, 1.0)
            cond = lambda_max_raw / lambda_min
            # Since P = M M^T, κ_2(M) = sqrt(κ_2(P)). Here cond is a conservative
            # upper bound from λ_min lower bound, so κ(I+Z) below is also an upper
            # bound. This keeps the diagnostic cheap and safety-oriented.
            kappa_ipz_upper = math.sqrt(cond)
            # First-order inverse sensitivity bound: relative inverse error
            # scales as κ(M) * eps.
            rel_error_upper = kappa_ipz_upper * eps_val
            warn_ipz = rel_error_upper >= tol

            # Normalize by spectral radius: preserves anisotropy, fixes global scale.
            P = P * (1.0 / lambda_max_raw)
            lambda_max = 1.0

            # Measure gain: ||Pg|| / ||g|| for A_tilde gradient
            g_a = grad_flat[a_key]
            g_norm = float(mx.sqrt(mx.sum(g_a * g_a)))
            Pg_a = P @ g_a  # [r,r] @ [r,in]
            Pg_norm = float(mx.sqrt(mx.sum(Pg_a * Pg_a)))
            gain = Pg_norm / max(g_norm, div_eps_val)

            # Apply normalized one-sided preconditioner factor
            grad_flat[a_key] = Pg_a
            grad_flat[b_key] = P @ grad_flat[b_key]  # [r,r] @ [r,out]
            # S_raw lives in R^r (Euclidean) — no preconditioning

            mx.eval(grad_flat[a_key], grad_flat[b_key])

            all_lambda_max.append(lambda_max)
            all_lambda_max_raw.append(lambda_max_raw)
            all_cond.append(cond)
            all_gain.append(gain)
            all_ipz_kappa_upper.append(kappa_ipz_upper)
            all_ipz_rel_error_upper.append(rel_error_upper)
            ipz_warn_count += 1 if warn_ipz else 0

        metrics: dict[str, float] = {}
        if all_lambda_max:
            metrics["precond_lambda_max"] = max(all_lambda_max)
            metrics["precond_lambda_max_mean"] = sum(all_lambda_max) / len(all_lambda_max)
            metrics["precond_lambda_max_raw"] = max(all_lambda_max_raw)
            metrics["precond_lambda_max_raw_mean"] = (
                sum(all_lambda_max_raw) / len(all_lambda_max_raw)
            )
            metrics["precond_cond_max"] = max(all_cond)
            metrics["precond_cond_mean"] = sum(all_cond) / len(all_cond)
            metrics["precond_gain_mean"] = sum(all_gain) / len(all_gain)
            metrics["precond_gain_max"] = max(all_gain)
            metrics["precond_ipz_kappa_upper_max"] = max(all_ipz_kappa_upper)
            metrics["precond_ipz_kappa_upper_mean"] = (
                sum(all_ipz_kappa_upper) / len(all_ipz_kappa_upper)
            )
            metrics["precond_ipz_rel_error_upper_max"] = max(all_ipz_rel_error_upper)
            metrics["precond_ipz_rel_error_upper_mean"] = (
                sum(all_ipz_rel_error_upper) / len(all_ipz_rel_error_upper)
            )
            metrics["precond_ipz_warn_fraction"] = ipz_warn_count / len(all_ipz_kappa_upper)
            metrics["precond_ipz_warn_any"] = 1.0 if ipz_warn_count > 0 else 0.0

        return tree_unflatten(list(grad_flat.items())), metrics

    # ── Certificate computation methods ─────────────────────────────

    def _compute_val_gradient(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> dict[str, Any] | None:
        """Compute flat gradient of validation loss at current params.

        Averages gradients across ``n_batches`` validation batches.

        Returns:
            Flat dict {param_key: gradient_array}, or None on failure.
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_vg = nn.value_and_grad(model, default_loss)
        accum: dict[str, Any] | None = None
        count = 0

        try:
            for batch, lengths in iterate_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            ):
                if count >= n_batches:
                    break
                (loss, _), grads = loss_vg(model, batch, lengths)
                mx.eval(loss)
                flat = dict(mlx_flatten(grads))
                if accum is None:
                    accum = {k: mx.zeros_like(v) for k, v in flat.items()}
                    mx.eval(*accum.values())
                for k in accum:
                    if k in flat:
                        accum[k] = accum[k] + flat[k]
                mx.eval(*accum.values())
                count += 1
        except Exception:
            logger.debug("Val gradient computation failed", exc_info=True)
            return None

        if accum is None or count == 0:
            return None

        for k in accum:
            accum[k] = accum[k] * (1.0 / count)
        mx.eval(*accum.values())
        return accum

    def _compute_val_hvp(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
        direction: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compute H_val @ d via central-difference HVP on validation data.

        H_val @ d ≈ (∇L_val(θ+εd) - ∇L_val(θ-εd)) / 2ε

        ε = (3·ε_mach)^(1/3) × max(||params||, 1.0) (Nocedal & Wright 2006, §8.1).

        Cost: 2 × n_batches backward passes.

        Returns:
            Flat dict {param_key: hvp_array}, or None on failure.
        """
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

        trainable = dict(mlx_flatten(model.trainable_parameters()))
        original = {k: mx.array(v) for k, v in trainable.items()}
        mx.eval(*original.values())

        # Central-difference optimal perturbation (Nocedal & Wright 2006, §8.1):
        # Minimizing truncation (h²) + roundoff (eps_f/h) gives h = (3*eps_f)^(1/3).
        # Scale by ||θ|| to make relative to parameter magnitude.
        param_norm = math.sqrt(
            sum(float(mx.sum(v * v)) for v in trainable.values())
        )
        _eps_f32 = math.ldexp(1.0, -23)
        eps = (3.0 * _eps_f32) ** (1.0 / 3.0) * max(param_norm, 1.0)

        try:
            # θ + ε d
            plus_p = {k: trainable[k] + eps * direction[k]
                       for k in trainable if k in direction}
            model.update(tree_unflatten(plus_p))
            mx.eval(model.parameters())
            g_plus = self._compute_val_gradient(
                model, eval_dataset, batch_size, seq_length, n_batches,
            )

            # θ - ε d
            minus_p = {k: trainable[k] - eps * direction[k]
                        for k in trainable if k in direction}
            model.update(tree_unflatten(minus_p))
            mx.eval(model.parameters())
            g_minus = self._compute_val_gradient(
                model, eval_dataset, batch_size, seq_length, n_batches,
            )

            if g_plus is None or g_minus is None:
                return None

            hvp = {
                k: (g_plus[k] - g_minus[k]) * (1.0 / (2.0 * eps))
                for k in g_plus if k in g_minus
            }
            mx.eval(*hvp.values())
            return hvp

        except Exception:
            logger.debug("Val HVP computation failed", exc_info=True)
            return None
        finally:
            model.update(tree_unflatten(original))
            mx.eval(model.parameters())

    def _compute_per_batch_val_losses(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> list[float]:
        """Compute per-batch validation losses (forward-only, no grad).

        Returns:
            List of per-batch average loss values.
        """
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        per_batch: list[float] = []
        for batch, lengths in iterate_batches(
            eval_dataset, batch_size, seq_length, loop=False,
        ):
            if len(per_batch) >= n_batches:
                break
            loss, ntoks = default_loss(model, batch, lengths)
            mx.eval(loss, ntoks)
            n = float(ntoks)
            if n > 0:
                per_batch.append(float(loss))
        return per_batch
