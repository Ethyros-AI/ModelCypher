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

"""MLX adapter for geometric LoRA training via NB-LoRA.

One training method. Cayley-parameterized LoRA with spectral bounds by construction.
ScaledGD preconditioning for condition-number-free convergence on the rank-r manifold.
Weyl budget monitoring for per-layer spectral crossing detection.
Measured Lipschitz LR (1/λ_max(Hessian)) for optimal step size.

The Cayley transform maps unconstrained (A_tilde, B_tilde) to semi-orthogonal
(A, B), guaranteeing ||2 * B^T @ S @ A||_2 <= 2 * max(S) <= sigma_k.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from modelcypher.core.domain.training.geometric_early_stopping import (
    check_loss_stable,
    check_val_loss_converged,
)
from modelcypher.core.domain.training.gradient_smoothness_estimator import (
    GradientSmoothnessEstimator,
)
from modelcypher.core.domain.training.spectral_budget import (
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    is_budget_exhausted,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_lora import LayerGeometry
    from modelcypher.core.domain.training.geometric_optimizer import OptimizerGeometryConfig
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Per-epoch diagnostic metrics for mechanism analysis."""

    epoch: int
    train_loss: float
    val_loss: float | None
    lipschitz_L: float | None
    eta: float
    update_norm: float | None
    max_spectral_ratio: float | None
    mean_token_entropy: float | None
    repetition_rate: float | None
    elapsed_seconds: float
    eta_ceiling: float | None = None
    budget_median_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Custom VJP for matrix inverse (MLX doesn't implement Inverse VJP)
# =============================================================================


@mx.custom_function
def _inv_with_grad(A):
    """Matrix inverse with custom VJP for autograd through Cayley transform."""
    return mx.linalg.inv(A, stream=mx.cpu)


@_inv_with_grad.vjp
def _inv_with_grad_vjp(primals, cotangent, output):
    """VJP: if Y = inv(A), then dL/dA = -Y^T @ (dL/dY) @ Y^T."""
    Y = output
    return (-Y.T @ cotangent @ Y.T,)


# =============================================================================
# NBLoRALinear: The one correct way to do LoRA
# =============================================================================


class NBLoRALinear(nn.Module):
    """Linear layer with NB-LoRA: norm-bounded adaptation via Cayley transform.

    Forward: base_linear(x) + 2 * (x @ A^T * S) @ B

    Where A, B are semi-orthogonal factors from the Cayley transform of
    unconstrained free parameters A_tilde, B_tilde. The spectral norm of the
    LoRA contribution is bounded by 2 * max(S) <= sigma_k by construction.

    Subclasses nn.Module so MLX autograd discovers trainable parameters.

    Trainable parameters: A_tilde [r, in], B_tilde [r, out], S_raw [r]
    Frozen: everything in self.linear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scale_bound: float,
        init_std: float = 0.01,
    ):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._rank = rank
        self._scale_bound = scale_bound

        # Unconstrained free parameters — Cayley transform handles the rest
        self.A_tilde = mx.random.normal((rank, in_features)) * init_std
        self.B_tilde = mx.random.normal((rank, out_features)) * init_std
        # S_raw clamped to [0, scale_bound] at every forward and after every step
        self.S_raw = mx.ones((rank,)) * (0.5 * scale_bound)

        mx.eval(self.A_tilde, self.B_tilde, self.S_raw)

    @classmethod
    def from_base(
        cls,
        linear,
        rank: int,
        scale_bound: float,
        init_std: float = 0.01,
    ) -> "NBLoRALinear":
        """Create from existing nn.Linear or nn.QuantizedLinear."""
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits

        obj = cls(input_dims, output_dims, rank, scale_bound, init_std)
        obj.linear = linear
        return obj

    def __call__(self, x):
        """Forward: base(x) + Cayley LoRA contribution."""
        base_out = self.linear(x)
        lora_out = self._cayley_forward(x)
        return base_out + lora_out

    def _cayley_transform(self):
        """Cayley transform: (A_tilde, B_tilde) → semi-orthogonal (A, B).

        [A^T; B^T] has orthonormal columns, guaranteeing:
            ||2 * B^T @ diag(S) @ A||_2 <= 2 * max(S)
        """
        r = self._rank

        # Stack [A_tilde^T; B_tilde^T] → [(n_in + n_out), r]
        stacked = mx.concatenate([self.A_tilde.T, self.B_tilde.T], axis=0)

        # Split: X [r, r], Y [(n_in + n_out - r), r]
        X = stacked[:r, :]
        Y = stacked[r:, :]

        # Z = (X - X^T) + Y^T @ Y  (skew-symmetric + PSD → I+Z always invertible)
        Z = (X - X.T) + Y.T @ Y

        I = mx.eye(r)
        IpZ_inv = _inv_with_grad(I + Z)

        # Semi-orthogonal factors
        A_core = (I - Z) @ IpZ_inv
        B_core = -2.0 * (Y @ IpZ_inv)

        # Reconstruct and split
        output = mx.concatenate([A_core, B_core], axis=0)
        n_in = self._in_features
        A = output[:n_in, :].T  # [r, n_in]
        B = output[n_in:, :].T  # [r, n_out]

        return A, B

    def _cayley_forward(self, x):
        """Compute 2 * (x @ A^T * S) @ B via Cayley transform."""
        A, B = self._cayley_transform()
        S = mx.clip(self.S_raw, 0.0, self._scale_bound)

        # Efficient rank-r forward: 3 matmuls
        z = x @ A.T         # [..., r]
        z = z * S            # [..., r] element-wise broadcast
        out = 2.0 * (z @ B)  # [..., out_features]

        return out.astype(x.dtype)

    def clamp_scale(self):
        """Clamp S_raw to [0, scale_bound]. Call after every optimizer step."""
        self.S_raw = mx.clip(self.S_raw, 0.0, self._scale_bound)

    def get_effective_delta(self):
        """Get weight delta: 2 * B^T @ diag(S) @ A  [out, in]."""
        A, B = self._cayley_transform()
        S = mx.clip(self.S_raw, 0.0, self._scale_bound)
        # delta = 2 * B^T @ diag(S) @ A = [out, r] @ [r, r] @ [r, in]
        S_A = A * S[:, None]  # Scale rows of A by S
        delta = 2.0 * (B.T @ S_A)
        mx.eval(delta)
        return delta

    def get_spectral_norm(self) -> float:
        """Actual spectral norm (should be <= 2 * scale_bound by construction)."""
        delta = self.get_effective_delta().astype(mx.float32)
        mx.eval(delta)
        _, S, _ = mx.linalg.svd(delta, stream=mx.cpu)
        mx.eval(S)
        return float(S[0])

    def to_standard_lora(self):
        """Convert to standard (lora_a, lora_b) format for saving.

        Returns (lora_a [in, r], lora_b [r, out]) with scale=1.0 such that
        scale * lora_b^T @ lora_a^T = 2 * B^T @ diag(S) @ A = delta.
        """
        A, B = self._cayley_transform()
        S = mx.clip(self.S_raw, 0.0, self._scale_bound)

        lora_a = A.T                    # [in, r]
        lora_b = 2.0 * (S[:, None] * B)  # [r, out] — diag(S) @ B, scaled by 2
        mx.eval(lora_a, lora_b)
        return lora_a, lora_b

    @property
    def scale_bound(self) -> float:
        return self._scale_bound


# =============================================================================
# MLX Training Adapter
# =============================================================================


class MLXTrainingAdapter:
    """MLX-specific training operations using NB-LoRA.

    One method. Cayley-parameterized. Geometry-derived. Bounds by construction.
    """

    def __init__(self, backend: "Backend"):
        self._backend = backend

    def prepare_dataset(self, samples: list[dict[str, Any]], tokenizer) -> list[tuple[Any, int]]:
        """Tokenize samples into mlx-lm iterate_batches format."""
        dataset: list[tuple[Any, int]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            dataset.append((mx.array(tokens, dtype=mx.int32), 0))
        return dataset

    def extract_weight_matrices(self, model) -> dict[str, Any]:
        """Extract 2D projection weights from the model."""
        weights: dict[str, Any] = {}
        base = getattr(model, "model", model)

        if not hasattr(base, "layers"):
            raise ValueError("Model has no .layers attribute — unsupported architecture")

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj = getattr(attn, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
                        weights[key] = proj.weight
                        mx.eval(proj.weight)

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for proj_name in ("up_proj", "down_proj", "gate_proj"):
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                        weights[key] = proj.weight
                        mx.eval(proj.weight)

        logger.info(
            "Extracted %d weight matrices from %d layers",
            len(weights),
            len(base.layers),
        )
        return weights

    def inject_nb_lora(
        self,
        model,
        geometries: dict[str, "LayerGeometry"],
        target_modules: list[str],
        safety_margin: float = 0.9,
    ) -> int:
        """Replace target linear layers with NBLoRALinear.

        Scale bound per layer: (sigma_k / 2) * safety_margin
        Rank per layer: tail_dims from geometry (null-space capacity)

        Returns number of layers injected.
        """
        injected = 0

        for key in target_modules:
            geom = geometries.get(key)
            if geom is None or geom.tail_dims <= 0:
                continue

            rank = geom.tail_dims
            # Geometry-derived scale bound: 2 * max(S) <= sigma_k
            scale_bound = (geom.sigma_k / 2.0) * safety_margin

            if scale_bound <= 0:
                logger.warning("Skipping %s: sigma_k=%.6f produces zero bound", key, geom.sigma_k)
                continue

            try:
                parent, attr_name = self._resolve_parent_and_attr(model, key)
                linear = getattr(parent, attr_name)

                nb_lora = NBLoRALinear.from_base(
                    linear,
                    rank=rank,
                    scale_bound=scale_bound,
                )
                mx.eval(nb_lora.A_tilde, nb_lora.B_tilde, nb_lora.S_raw)

                setattr(parent, attr_name, nb_lora)
                injected += 1

                logger.debug(
                    "Injected NB-LoRA at %s: rank=%d, σ_k=%.6f, bound=%.6f",
                    key, rank, geom.sigma_k, scale_bound,
                )
            except Exception as exc:
                logger.warning("Failed to inject NB-LoRA at %s: %s", key, exc)

        logger.info("Injected %d NB-LoRA layers (bounds by construction)", injected)
        return injected

    def freeze_and_apply_lora(self, model) -> None:
        """Freeze entire model, then unfreeze only NB-LoRA parameters."""
        model.freeze()
        # Walk model tree and unfreeze A_tilde, B_tilde, S_raw in each NBLoRALinear
        for _, nb_lora in self._iter_nb_lora_modules(model):
            nb_lora.unfreeze(keys=["A_tilde", "B_tilde", "S_raw"])

    def evaluate_loss(
        self,
        model,
        dataset,
        tokenizer,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> tuple[float, float]:
        """Compute average loss and perplexity over a dataset."""
        del tokenizer
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        total_loss = 0.0
        total_tokens = 0.0
        n_evaluated = 0

        for batch, lengths in iterate_batches(dataset, batch_size, seq_length, loop=False):
            loss, ntoks = default_loss(model, batch, lengths)
            mx.eval(loss, ntoks)
            total_loss += float(loss) * float(ntoks)
            total_tokens += float(ntoks)
            n_evaluated += 1
            if n_evaluated >= n_batches:
                break

        if total_tokens == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100.0))
        return avg_loss, perplexity

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
        lipschitz_batches: int = 3,
        tokenizer=None,
        opt_config: "OptimizerGeometryConfig | None" = None,
    ) -> tuple[list[tuple[int, float, float]], str, list[EpochMetrics]]:
        """Train with ScaledGD, Weyl budget monitoring, and geometric stopping.

        Optimizer: ScaledGD preconditioning (Tong et al. JMLR 2021) when opt_config
        is provided. Falls back to plain SGD otherwise. ScaledGD preconditions each
        LoRA factor's gradient through the pseudoinverse of the other factor, giving
        condition-number-free convergence on the rank-r manifold.

        LR derivation (in priority order):
        1. lr_override — user knows best (disables adaptive LR)
        2. 1/L where L = median(λ_max(Hessian)) — robust multi-batch estimate
        3. 1/σ_max — conservative spectral fallback

        Stopping (any one triggers):
        1. Validation loss convergence or degradation (overfitting)
        2. Weyl budget exhaustion (per-layer spectral crossing)
        3. Training loss stability (fallback if no eval_dataset)
        4. Safety cap (max_iters)

        After each step: clamp S_raw (enforce bound).

        Returns: (losses, stop_reason, epoch_metrics)
        """
        import mlx.optimizers as opt
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_value_and_grad = nn.value_and_grad(model, default_loss)

        # Learning rate: override > measured Lipschitz > spectral proxy
        can_remeasure = False

        if lr_override is not None:
            eta = float(lr_override)
            logger.info("LR from override: %.2e", eta)
        else:
            # Robust Lipschitz: median across multiple batches
            L = self._measure_lipschitz_robust(
                model, train_dataset, batch_size, seq_length,
                default_loss, n_batches=lipschitz_batches, n_iters=10, seed=seed,
            )

            if L is not None and L > 0:
                eta = 1.0 / L
                can_remeasure = True
                logger.info(
                    "LR from Hessian: 1/L = 1/%.4f = %.2e (robust, %d batches)",
                    L, eta, lipschitz_batches,
                )
            else:
                # Spectral fallback: 1/σ_max (no magic numbers)
                if sigma_max > 0:
                    eta = 1.0 / sigma_max
                else:
                    # Machine-precision fallback — should never happen with valid weights
                    eta = math.sqrt(math.ldexp(1.0, -23))  # sqrt(eps_f32)
                logger.info(
                    "LR from spectral fallback: 1/σ_max = 1/%.4f = %.2e",
                    sigma_max, eta,
                )

        current_eta = eta
        eta_ceiling = eta  # Spectral ceiling: LR can recover up to initial estimate
        optimizer = opt.SGD(learning_rate=current_eta, momentum=0.0)

        # ScaledGD preconditioning: use effective A, B from Cayley transform
        use_scaled_gd = opt_config is not None
        # Default epsilon from machine precision (sqrt(eps_f32))
        sqrt_eps_default = math.sqrt(math.ldexp(1.0, -23))

        losses: list[tuple[int, float, float]] = []
        val_losses: list[float] = []
        epoch_metrics_list: list[EpochMetrics] = []
        stop_reason: str | None = None

        batch_iter = iterate_batches(
            train_dataset, batch_size, seq_length, loop=True, seed=seed,
        )

        n_batches_per_epoch = len(
            list(iterate_batches(train_dataset, batch_size, seq_length, loop=False, seed=seed))
        )
        if n_batches_per_epoch <= 0:
            raise ValueError("Training dataset produced zero batches")

        use_val_stopping = eval_dataset is not None and len(eval_dataset) > 0
        # Eval batch size: data-derived (dataset size / eval_batches)
        eval_batch_size = min(
            batch_size,
            max(1, len(eval_dataset) // max(1, eval_batches)) if eval_dataset else 2,
        )

        check_interval = max(1, n_batches_per_epoch)

        lr_mode = "constant"
        if adaptive_lr:
            lr_mode = "adaptive-monotonic" if lr_monotonic else "adaptive"
        optimizer_name = "ScaledGD" if use_scaled_gd else "SGD"
        logger.info(
            "Training: optimizer=%s, stop=%s, cap=%d, epoch=%d batches, lr=%.2e, mode=%s",
            optimizer_name,
            "validation loss" if use_val_stopping else "training loss",
            max_iters, n_batches_per_epoch, current_eta, lr_mode,
        )

        # Track params at epoch start for update_norm
        epoch_start_params: dict[str, Any] | None = None
        epoch_start_time = time.time()

        for it in range(max_iters):
            # Snapshot params at epoch start
            if it % n_batches_per_epoch == 0:
                trainable = dict(mlx_flatten(model.trainable_parameters()))
                epoch_start_params = {k: mx.array(v) for k, v in trainable.items()}
                mx.eval(*epoch_start_params.values())
                epoch_start_time = time.time()

            t_step = time.time()
            batch, lengths = next(batch_iter)

            (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)

            # ScaledGD preconditioning: precondition A_tilde/B_tilde gradients
            # using the effective Cayley factors (A, B) for scale normalization
            if use_scaled_gd:
                grad = self._apply_scaled_gd(model, grad, opt_config, sqrt_eps_default)

            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)

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
                if use_val_stopping:
                    v_loss, _ = self.evaluate_loss(
                        model=model,
                        dataset=eval_dataset,
                        tokenizer=None,
                        batch_size=eval_batch_size,
                        seq_length=seq_length,
                        n_batches=eval_batches,
                    )
                    val_losses.append(v_loss)

                # 2. Update norm (||θ_end - θ_start||)
                update_norm = None
                if epoch_start_params is not None:
                    current_params = dict(mlx_flatten(model.trainable_parameters()))
                    update_norm = math.sqrt(sum(
                        float(mx.sum((current_params[k] - epoch_start_params[k]) ** 2))
                        for k in epoch_start_params if k in current_params
                    ))

                # 3. Adaptive LR: re-measure curvature + backoff
                measured_L = None
                if adaptive_lr and can_remeasure and lr_override is None:
                    measured_L = self._measure_lipschitz_robust(
                        model, train_dataset, batch_size, seq_length,
                        default_loss, n_batches=lipschitz_batches, n_iters=5,
                        seed=seed + epoch_num,  # Vary batches across epochs
                    )

                    if measured_L is not None and measured_L > 0:
                        eta_spectral = 1.0 / measured_L
                    else:
                        eta_spectral = current_eta

                    # Validation-guided backoff: proportional to loss ratio
                    if (v_loss is not None and len(val_losses) >= 2
                            and val_losses[-1] > val_losses[-2]
                            and val_losses[-1] > 0):
                        # Scale LR by inverse of loss increase ratio
                        backoff = val_losses[-2] / val_losses[-1]
                        current_eta *= max(backoff, 0.1)  # Floor at 10× reduction
                        logger.info(
                            "Val loss increased (%.4f → %.4f): LR backoff=%.3f to %.2e",
                            val_losses[-2], val_losses[-1], backoff, current_eta,
                        )

                    if lr_monotonic:
                        # Legacy: eta can only decrease
                        current_eta = min(eta_spectral, current_eta)
                    else:
                        # Non-monotonic: eta can recover up to initial ceiling
                        current_eta = min(eta_spectral, eta_ceiling)

                    # Update optimizer (SGD w/ momentum=0 has no state to preserve)
                    optimizer = opt.SGD(learning_rate=current_eta, momentum=0.0)

                # 4. Weyl budget monitoring (replaces simple verify_bounds)
                max_ratio = None
                budget_exhausted_flag = False
                median_budget_ratio = None
                try:
                    if opt_config is not None:
                        # Weyl-derived per-layer crossing thresholds
                        lora_products = []
                        spectral_gaps = []
                        sigma_ks_list = []
                        for name, nb_lora in self._iter_nb_lora_modules(model):
                            A, B = nb_lora._cayley_transform()
                            S = mx.clip(nb_lora.S_raw, 0.0, nb_lora._scale_bound)
                            lora_products.append((
                                2.0,
                                (S[:, None] * A),
                                B,
                                nb_lora._scale_bound,
                            ))
                            layer_opt = opt_config.layer_configs.get(name)
                            if layer_opt:
                                spectral_gaps.append(layer_opt.spectral_gap)
                                sigma_ks_list.append(layer_opt.sigma_k)
                            mx.eval(A, B, S)

                        ratios = compute_budget_ratios(
                            lora_products, self._backend,
                        )
                        if ratios:
                            budget_exhausted_flag, median_budget_ratio = is_budget_exhausted(
                                ratios,
                                threshold=DTYPE_THRESHOLD_F32,
                                spectral_gaps=spectral_gaps if spectral_gaps else None,
                                sigma_ks=sigma_ks_list if sigma_ks_list else None,
                            )
                            max_ratio = max(ratios) if ratios else None
                    else:
                        # Fallback: simple verify_bounds
                        _, max_ratio, _ = self.verify_bounds(model)
                except Exception:
                    pass

                # 5. Entropy and repetition probe
                mean_entropy, rep_rate = self._probe_entropy_and_repetition(
                    model, tokenizer,
                )

                # 6. Collect epoch metrics
                epoch_metrics_list.append(EpochMetrics(
                    epoch=epoch_num,
                    train_loss=loss_val,
                    val_loss=v_loss,
                    lipschitz_L=measured_L,
                    eta=current_eta,
                    update_norm=update_norm,
                    max_spectral_ratio=max_ratio,
                    mean_token_entropy=mean_entropy,
                    repetition_rate=rep_rate,
                    elapsed_seconds=epoch_elapsed,
                    eta_ceiling=eta_ceiling if adaptive_lr else None,
                    budget_median_ratio=median_budget_ratio,
                ))

                # Log
                log_parts = [
                    f"Epoch {epoch_num} | train_loss={loss_val:.4f}",
                ]
                if v_loss is not None:
                    log_parts.append(f"val_loss={v_loss:.4f}")
                log_parts.append(f"eta={current_eta:.2e}")
                if measured_L is not None:
                    log_parts.append(f"L={measured_L:.4f}")
                if update_norm is not None:
                    log_parts.append(f"‖Δθ‖={update_norm:.4f}")
                if max_ratio is not None:
                    log_parts.append(f"spectral={max_ratio:.4f}")
                if median_budget_ratio is not None:
                    log_parts.append(f"budget={median_budget_ratio:.4f}")
                if mean_entropy is not None:
                    log_parts.append(f"entropy={mean_entropy:.2f}")
                if rep_rate is not None:
                    log_parts.append(f"rep={rep_rate:.3f}")
                logger.info(" | ".join(log_parts))

                # 7a. Weyl budget exhaustion check (any layer crossing)
                if budget_exhausted_flag:
                    stop_reason = (
                        f"budget_exhausted (Weyl crossing, "
                        f"median_ratio={median_budget_ratio:.4f}, epoch={epoch_num})"
                    )
                    logger.info(
                        "Budget stop at iter %d: %s", it + 1, stop_reason,
                    )
                    break

                # 7b. Convergence check
                if use_val_stopping and len(val_losses) >= 6:
                    should_stop, reason, threshold = check_val_loss_converged(
                        val_losses, window=3,
                    )
                    if should_stop:
                        stop_reason = f"{reason} (SE={threshold:.4e}, epochs={len(val_losses)})"
                        logger.info(
                            "Validation stop at iter %d: %s", it + 1, stop_reason,
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

            details.append({
                "layer": name,
                "spectral_norm": spectral_norm,
                "theoretical_max": theoretical_max,
                "ratio": ratio,
                "ok": ratio <= 1.0 + math.sqrt(math.ldexp(1.0, -23)),  # sqrt(eps_f32) tolerance
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

        for name, module in self._iter_nb_lora_modules(model):
            lora_a, lora_b = module.to_standard_lora()
            rank = int(lora_a.shape[1])
            discovered_ranks.append(rank)

            key_base = name.replace(".weight", "")
            adapter_weights[f"{key_base}.lora_a"] = lora_a
            adapter_weights[f"{key_base}.lora_b"] = lora_b
            target_modules.add(self._module_name_from_layer_key(name))

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

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _clamp_all_scales(self, model) -> None:
        """Clamp S_raw in all NBLoRALinear modules after optimizer step."""
        for _, module in self._iter_nb_lora_modules(model):
            module.clamp_scale()
            mx.eval(module.S_raw)

    def _apply_scaled_gd(
        self,
        model,
        grad,
        opt_config: "OptimizerGeometryConfig",
        sqrt_eps_default: float,
    ):
        """Apply ScaledGD preconditioning to NB-LoRA gradients.

        For each NB-LoRA module, uses the effective Cayley factors (A, B)
        to precondition the unconstrained parameter gradients:
            grad_A_tilde := grad_A_tilde @ (B @ B^T + εI)^{-1}
            grad_B_tilde := (A^T @ A + εI)^{-1} @ grad_B_tilde

        This normalizes out the scale of each factor, giving condition-number-free
        convergence on the rank-r manifold (Tong et al. JMLR 2021).

        Also applies condition-aware weight decay: σ_k/σ_max per layer.

        The inversions are NOT in the autograd path (applied to gradients post-backward),
        so they use plain mx.linalg.inv (not _inv_with_grad).
        """
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

        # Flatten gradient tree for mutation
        grad_flat = dict(mlx_flatten(grad))

        for name, nb_lora in self._iter_nb_lora_modules(model):
            # Find gradient keys for this module's trainable params
            # The key prefix in the flat tree matches the model structure
            prefix = name.replace(".weight", "")
            a_key = None
            b_key = None
            s_key = None
            for k in grad_flat:
                if k.endswith("A_tilde") and prefix.replace("model.", "") in k:
                    a_key = k
                elif k.endswith("B_tilde") and prefix.replace("model.", "") in k:
                    b_key = k
                elif k.endswith("S_raw") and prefix.replace("model.", "") in k:
                    s_key = k

            if a_key is None or b_key is None:
                continue

            # Get effective Cayley factors for preconditioning
            A, B = nb_lora._cayley_transform()
            r = nb_lora._rank

            # Per-layer epsilon from geometric analysis
            layer_opt = opt_config.layer_configs.get(name)
            eps = layer_opt.epsilon if layer_opt else sqrt_eps_default

            # Precondition grad_A_tilde: @ (B B^T + εI)^{-1}
            BBt = B @ B.T + eps * mx.eye(r)
            BBt_inv = mx.linalg.inv(BBt, stream=mx.cpu)
            grad_flat[a_key] = grad_flat[a_key] @ BBt_inv

            # Precondition grad_B_tilde: (A^T A + εI)^{-1} @
            AtA = A.T @ A + eps * mx.eye(r)
            AtA_inv = mx.linalg.inv(AtA, stream=mx.cpu)
            grad_flat[b_key] = AtA_inv @ grad_flat[b_key]

            # Condition-aware weight decay: σ_k / σ_max
            decay = layer_opt.decay_scale if layer_opt else 0.0
            if decay > 0:
                grad_flat[a_key] = grad_flat[a_key] + decay * nb_lora.A_tilde
                grad_flat[b_key] = grad_flat[b_key] + decay * nb_lora.B_tilde

            mx.eval(grad_flat[a_key], grad_flat[b_key])

        # Reconstruct gradient tree
        return tree_unflatten(list(grad_flat.items()))

    def _iter_nb_lora_modules(self, model):
        """Yield (layer_key, NBLoRALinear) pairs from model tree."""
        base = getattr(model, "model", model)
        if not hasattr(base, "layers"):
            return

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, NBLoRALinear):
                        key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
                        yield key, proj

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for proj_name in ("up_proj", "down_proj", "gate_proj"):
                    proj = getattr(mlp, proj_name, None)
                    if isinstance(proj, NBLoRALinear):
                        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                        yield key, proj

    def _measure_lipschitz(
        self,
        model,
        batch,
        lengths,
        loss_fn,
        n_iters: int = 10,
    ) -> float | None:
        """Measure λ_max(Hessian) via power iteration on a single batch.

        Uses central-difference HVP: H@v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / 2ε
        Power iteration converges to the top eigenvalue at rate |λ₂/λ₁|^k.

        Same math as event-buffer Lipschitz measurement, pure MLX implementation.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        trainable_tree = model.trainable_parameters()
        flat_pairs = tree_flatten(trainable_tree)
        if not flat_pairs:
            return None

        params = dict(flat_pairs)

        # Save original params for restoration
        original = {k: mx.array(v) for k, v in params.items()}
        mx.eval(*original.values())

        loss_vg = nn.value_and_grad(model, loss_fn)
        # HVP epsilon: sqrt(ε_mach) × ||params|| (optimal for central differences)
        param_norm = math.sqrt(sum(float(mx.sum(v * v)) for v in params.values()))
        sqrt_eps_mach = math.sqrt(math.ldexp(1.0, -23))  # sqrt(eps_f32)
        eps = sqrt_eps_mach * max(param_norm, 1.0)

        def grad_at(p):
            """Compute flat gradients at given params."""
            model.update(tree_unflatten(p))
            mx.eval(model.parameters())
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)
            return dict(tree_flatten(grads))

        def norm(d):
            s = sum(float(mx.sum(v * v)) for v in d.values())
            return math.sqrt(s)

        try:
            # Random direction, normalized (use hash of params for reproducibility)
            mx.random.seed(hash(tuple(params.keys())) % (2**31))
            v = {k: mx.random.normal(p.shape) for k, p in params.items()}
            v_norm = norm(v)
            v = {k: val / v_norm for k, val in v.items()}

            eigenvalue = 0.0
            prev = float("inf")

            for it in range(n_iters):
                # Central-difference HVP: H@v = (∇L(θ+εv) - ∇L(θ-εv)) / 2ε
                plus_p = {k: params[k] + eps * v[k] for k in params}
                minus_p = {k: params[k] - eps * v[k] for k in params}

                g_plus = grad_at(plus_p)
                g_minus = grad_at(minus_p)

                hv = {k: (g_plus[k] - g_minus[k]) / (2.0 * eps) for k in g_plus if k in g_minus}
                if not hv:
                    return None

                # Rayleigh quotient: v^T @ Hv
                eigenvalue = sum(float(mx.sum(v[k] * hv[k])) for k in v if k in hv)

                # Relative convergence: |Δλ| < ε_mach × |λ|
                if abs(eigenvalue) > 0 and abs(eigenvalue - prev) < sqrt_eps_mach * abs(eigenvalue):
                    break
                prev = eigenvalue

                # Normalize Hv for next iteration
                hv_norm = norm(hv)
                if hv_norm <= sqrt_eps_mach * sqrt_eps_mach:  # eps_mach (double sqrt = eps)
                    return None
                v = {k: val / hv_norm for k, val in hv.items()}

            L = abs(float(eigenvalue))
            if math.isfinite(L) and L > 0:
                logger.info(
                    "Lipschitz measured: L=%.4f (%d power iterations)",
                    L, min(it + 1, n_iters),
                )
                return L
            return None

        except Exception:
            logger.debug("Lipschitz measurement failed", exc_info=True)
            return None
        finally:
            # Restore original params
            model.update(tree_unflatten(original))
            mx.eval(model.parameters())

    def _measure_lipschitz_robust(
        self,
        model,
        dataset,
        batch_size: int,
        seq_length: int,
        loss_fn,
        n_batches: int = 3,
        n_iters: int = 10,
        seed: int = 42,
    ) -> float | None:
        """Median of L estimates across multiple batches (robust to outliers).

        Draws n_batches distinct batches from the training set, runs power
        iteration on each, and returns the median L.  This reduces the variance
        that caused seed 456 to get trapped at eta=3.7e-4 in the first
        adaptive-lr experiment.
        """
        from mlx_lm.tuner.trainer import iterate_batches

        estimates: list[float] = []
        batch_iter = iterate_batches(
            dataset, batch_size, seq_length, loop=False, seed=seed,
        )

        for i, (batch, lengths) in enumerate(batch_iter):
            if i >= n_batches:
                break
            L = self._measure_lipschitz(model, batch, lengths, loss_fn, n_iters=n_iters)
            if L is not None and L > 0:
                estimates.append(L)

        if not estimates:
            return None

        median_L = statistics.median(estimates)
        logger.info(
            "Robust Lipschitz: median=%.4f from %d/%d valid batches (values: %s)",
            median_L, len(estimates), n_batches,
            ", ".join(f"{v:.4f}" for v in estimates),
        )
        return median_L

    def _probe_entropy_and_repetition(
        self,
        model,
        tokenizer,
        n_sequences: int = 3,
        max_tokens: int = 64,
    ) -> tuple[float | None, float | None]:
        """Generate short sequences and measure entropy + repetition.

        Returns (mean_token_entropy, repetition_rate) or (None, None) on failure.
        Entropy: mean Shannon entropy per token across all generated sequences.
        Repetition: fraction of 4-grams that are repeated.
        """
        if tokenizer is None:
            return None, None

        prompts = ["The", "Once upon a time", "In the beginning"]

        all_entropies: list[float] = []
        all_tokens: list[int] = []

        try:
            for prompt in prompts[:n_sequences]:
                input_ids = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

                generated_tokens: list[int] = []
                for _ in range(max_tokens):
                    logits = model(input_ids)  # (1, seq_len, vocab_size)
                    next_logits = logits[:, -1, :]  # (1, vocab_size)

                    # Entropy from softmax distribution
                    probs = mx.softmax(next_logits, axis=-1)
                    log_probs = mx.log(probs + 1e-10)
                    entropy = -mx.sum(probs * log_probs, axis=-1)
                    all_entropies.append(float(entropy[0]))

                    # Greedy next token
                    next_token = int(mx.argmax(next_logits, axis=-1)[0])
                    generated_tokens.append(next_token)

                    # EOS check
                    if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
                        break

                    input_ids = mx.concatenate(
                        [input_ids, mx.array([[next_token]])], axis=1,
                    )

                all_tokens.extend(generated_tokens)

        except Exception:
            logger.debug("Entropy/repetition probe failed", exc_info=True)
            return None, None

        mean_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else None

        # 4-gram repetition rate
        n = 4
        if len(all_tokens) >= n:
            ngrams = [tuple(all_tokens[i : i + n]) for i in range(len(all_tokens) - n + 1)]
            repetition_rate = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        else:
            repetition_rate = 0.0

        return mean_entropy, repetition_rate

    def derive_critical_batch_size(
        self,
        model,
        train_dataset,
        seq_length: int,
        n_samples: int = 20,
    ) -> int:
        """Derive critical batch size from gradient noise scale.

        B_crit = Var(g) / ||E[g]||^2 = 1 / SNR

        Collects per-sample gradients (batch_size=1), then uses
        GradientSmoothnessEstimator to compute the noise scale.
        Same math as event-buffer path in lora_memory_store.derive_critical_batch_size().
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_vg = nn.value_and_grad(model, default_loss)

        per_sample_gradients: list[dict[str, Any]] = []

        for batch, lengths in iterate_batches(
            train_dataset, 1, seq_length, loop=False, seed=0,
        ):
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat_grads = dict(mlx_flatten(grads))
            mx.eval(*flat_grads.values())
            per_sample_gradients.append(flat_grads)

            if len(per_sample_gradients) >= n_samples:
                break

        if len(per_sample_gradients) < 2:
            logger.info("Too few samples for B_crit estimation, defaulting to 1")
            return 1

        quality = GradientSmoothnessEstimator.gradient_quality(
            per_sample_gradients=per_sample_gradients,
            backend=self._backend,
        )

        if quality is None or quality.snr <= 0:
            logger.info("Could not estimate gradient noise, defaulting to 1")
            return 1

        b_crit = max(1, math.ceil(1.0 / quality.snr))
        b_crit = min(b_crit, len(train_dataset))

        logger.info(
            "B_crit = %d (SNR=%.6f, variance=%.6f, %d samples)",
            b_crit, quality.snr, quality.variance, quality.sample_count,
        )
        return b_crit

    def _resolve_parent_and_attr(self, model, layer_key: str) -> tuple[Any, str]:
        path_parts = layer_key.replace(".weight", "").split(".")
        obj = model
        for part in path_parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj, path_parts[-1]

    def _module_name_from_layer_key(self, layer_key: str) -> str:
        parts = layer_key.replace(".weight", "").split(".")
        if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
            return ".".join(parts[3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return layer_key.replace(".weight", "")
