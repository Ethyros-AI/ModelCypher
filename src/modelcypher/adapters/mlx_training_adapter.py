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
    # Cayley-Riemannian preconditioner diagnostics
    precond_lambda_max: float | None = None
    precond_cond_max: float | None = None
    precond_gain_mean: float | None = None
    precond_m_invariant: float | None = None  # η * L * λ_max(P) ≤ 2
    precond_eta_step: float | None = None     # actual per-step η
    # Geometric stopping certificate
    cert_precond_grad_norm: float | None = None
    cert_alignment: float | None = None
    cert_curvature: float | None = None
    cert_delta_max_val: float | None = None
    cert_val_ci_half_width: float | None = None
    cert_delta_max_worst: float | None = None
    cert_all_met: bool | None = None
    # Topological phase diagnostics (optional, computed when topo_monitor=True)
    topo_betti_0: int | None = None
    topo_betti_1: int | None = None
    topo_persistence_entropy: float | None = None
    topo_mean_ricci_curvature: float | None = None
    topo_ricci_curvature_std: float | None = None
    # Dimensional expansion monitoring (optional, computed when dim_monitor=True)
    dim_expansion_ratio: float | None = None
    dim_peak_dim: float | None = None
    dim_final_dim: float | None = None
    dim_delta_from_baseline: float | None = None
    dim_is_contracting: bool | None = None
    # Constrained training diagnostics (optional, when constraint_config provided)
    constraint_mu_inv: float | None = None
    constraint_mu_sep: float | None = None
    constraint_mu_geo: float | None = None
    constraint_C_inv: float | None = None
    constraint_C_sep: float | None = None
    constraint_C_geo: float | None = None

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
# Paired Batch Iterator (for constrained geometric training)
# =============================================================================


def iterate_paired_batches(
    dataset: list[dict[str, Any]],
    batch_size: int,
    max_seq_length: int,
    logic_groups: dict[str, list[int]],
    template_groups: dict[str, list[int]],
    loop: bool = False,
    seed: int | None = None,
):
    """Iterate batches that contain paired samples for constraint computation.

    Each batch includes:
    - batch: [batch_size, seq_length] token array
    - lengths: [batch_size, 2] offset/length array
    - answer_masks: [batch_size, seq_length] answer token mask
    - inv_pairs: list of (i, j) indices within the batch that share logic_id
    - cf_pairs: list of (i, j) indices within the batch that share template_id

    Strategy: build batches that maximize pair coverage.
    1. Seed with a logic_id group (invariance pairs).
    2. Preferentially add samples sharing template_id with seed members
       but having different logic_id (counterfactual pairs).
    3. Fill remaining slots from other samples.
    """
    import numpy as np

    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Paired dataset must have at least batch_size={batch_size} "
            f"examples but only has {n}."
        )

    # Build sample pool with indices
    if seed is not None:
        np.random.seed(seed)

    # Group indices by logic_id for pair-aware batching
    logic_id_list = list(logic_groups.keys())

    while True:
        np.random.shuffle(logic_id_list)
        used: set[int] = set()

        for lid in logic_id_list:
            members = [i for i in logic_groups[lid] if i not in used]
            if not members:
                continue

            # Start batch with this logic group (invariance set)
            batch_indices: list[int] = list(members[:batch_size])
            used.update(batch_indices)

            # Preferentially add counterfactual partners: samples that share
            # a template_id with seed members but have a different logic_id
            if len(batch_indices) < batch_size:
                seed_templates = {dataset[i]["template_id"] for i in batch_indices}
                cf_candidates = []
                for tid in seed_templates:
                    for idx in template_groups.get(tid, []):
                        if idx not in used and dataset[idx]["logic_id"] != lid:
                            cf_candidates.append(idx)
                np.random.shuffle(cf_candidates)
                for idx in cf_candidates:
                    if len(batch_indices) >= batch_size:
                        break
                    batch_indices.append(idx)
                    used.add(idx)

            # Fill remaining slots from other samples
            if len(batch_indices) < batch_size:
                remaining = [i for i in range(n) if i not in used]
                np.random.shuffle(remaining)
                for idx in remaining:
                    if len(batch_indices) >= batch_size:
                        break
                    batch_indices.append(idx)
                    used.add(idx)

            if len(batch_indices) < batch_size:
                continue  # not enough samples

            batch_indices = batch_indices[:batch_size]

            # Build tensors
            batch_samples = [dataset[i] for i in batch_indices]
            lengths_list = [s["n_tokens"] for s in batch_samples]

            pad_to = 32
            max_len = 1 + pad_to * ((max(lengths_list) + pad_to - 1) // pad_to)
            max_len = min(max_len, max_seq_length)

            batch_arr = np.zeros((batch_size, max_len), dtype=np.int32)
            mask_arr = np.zeros((batch_size, max_len), dtype=np.float32)

            for j, s in enumerate(batch_samples):
                tlen = min(s["n_tokens"], max_seq_length)
                tokens_np = np.array(s["tokens"].tolist()[:tlen], dtype=np.int32)
                batch_arr[j, :tlen] = tokens_np
                amask_np = np.array(s["answer_mask"].tolist()[:tlen], dtype=np.float32)
                mask_arr[j, :tlen] = amask_np
                lengths_list[j] = tlen

            batch_tensor = mx.array(batch_arr)
            lengths_tensor = mx.array(
                [[0, l] for l in lengths_list], dtype=mx.int32,
            )
            answer_masks_tensor = mx.array(mask_arr)

            # Find pairs within this batch
            # Map batch position -> sample metadata
            batch_logic_ids = [batch_samples[j]["logic_id"] for j in range(batch_size)]
            batch_template_ids = [batch_samples[j]["template_id"] for j in range(batch_size)]

            # Invariance pairs: same logic_id within batch
            inv_pairs: list[tuple[int, int]] = []
            logic_to_pos: dict[str, list[int]] = {}
            for pos, lid_val in enumerate(batch_logic_ids):
                logic_to_pos.setdefault(lid_val, []).append(pos)
            for positions in logic_to_pos.values():
                for a in range(len(positions)):
                    for b in range(a + 1, len(positions)):
                        # Only pair if different template (true invariance)
                        if batch_template_ids[positions[a]] != batch_template_ids[positions[b]]:
                            inv_pairs.append((positions[a], positions[b]))

            # Counterfactual pairs: same template_id, different logic_id within batch
            cf_pairs: list[tuple[int, int]] = []
            tmpl_to_pos: dict[str, list[int]] = {}
            for pos, tid_val in enumerate(batch_template_ids):
                tmpl_to_pos.setdefault(tid_val, []).append(pos)
            for positions in tmpl_to_pos.values():
                for a in range(len(positions)):
                    for b in range(a + 1, len(positions)):
                        if batch_logic_ids[positions[a]] != batch_logic_ids[positions[b]]:
                            cf_pairs.append((positions[a], positions[b]))

            yield batch_tensor, lengths_tensor, answer_masks_tensor, inv_pairs, cf_pairs

        if not loop:
            break


# =============================================================================
# Constrained Loss Factory (for constrained geometric training)
# =============================================================================


def make_constrained_loss(
    constraint_state: "ConstraintState",
    config: "ConstraintConfig",
):
    """Create a loss function with answer-only CE + constraint penalties.

    The returned function has signature:
        loss_fn(model, batch, lengths, answer_masks, inv_pairs, cf_pairs) -> (loss, ntoks)

    It manually iterates through model layers (MLX has no forward hooks)
    to collect hidden states at target layers, computes answer-only CE,
    and adds constraint penalty terms weighted by Lagrange multipliers.

    Args:
        constraint_state: Mutable state holding Lagrange multipliers (μ_inv, μ_sep, μ_geo).
            Updated outside the gradient tape after each step.
        config: Constraint thresholds and target layer configuration.
    """
    target_layers_set = set(config.target_layers)
    baseline_entropy = config.baseline_entropy

    def constrained_loss(model, batch, lengths, answer_masks, inv_pairs, cf_pairs):
        """Answer-only CE + invariance + separation + geodesic tail constraints."""
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Answer mask for shifted target sequence (drop first token)
        amask = answer_masks[:, 1:]

        # --- Manual forward pass with hidden state collection ---
        base = getattr(model, "model", model)
        h = base.embed_tokens(inputs)

        # Route masks per layer type (LFM2 hybrid: attention + convolution layers)
        # Attention layers expect "causal" string, conv layers expect None
        layer_hiddens: dict[int, Any] = {}
        for idx, layer in enumerate(base.layers):
            if getattr(layer, "is_attention_layer", True):
                mask = "causal"
            else:
                mask = None
            h = layer(h, mask=mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            if idx in target_layers_set:
                layer_hiddens[idx] = h

        # Final norm + logits (handle different model architectures)
        if hasattr(base, "norm"):
            h = base.norm(h)
        elif hasattr(base, "embedding_norm"):
            h = base.embedding_norm(h)
        if hasattr(model, "lm_head"):
            logits = model.lm_head(h)
        else:
            logits = base.embed_tokens.as_linear(h)

        # --- L_answer: CE on answer tokens only ---
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(
            steps >= lengths[:, 0:1], steps <= lengths[:, 1:],
        ).astype(mx.float32)
        # Combine length mask with answer mask
        combined_mask = length_mask * amask

        ce = nn.losses.cross_entropy(logits, targets) * combined_mask
        ntoks = mx.maximum(combined_mask.sum(), mx.array(1.0))
        ce_loss = ce.astype(mx.float32).sum() / ntoks

        # --- C_inv: Invariance constraint ---
        # Mean hidden-state L2 distance for same-logic pairs across target layers
        c_inv = mx.array(0.0)
        n_inv = 0
        if inv_pairs:
            for layer_idx, hidden in layer_hiddens.items():
                # hidden: [batch, seq, hidden_dim]
                # Mean-pool over sequence dimension
                h_mean = mx.mean(hidden, axis=1)  # [batch, hidden_dim]
                for i, j in inv_pairs:
                    diff = h_mean[i] - h_mean[j]
                    dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
                    c_inv = c_inv + dist
                    n_inv += 1
        if n_inv > 0:
            c_inv = c_inv / n_inv

        # --- C_sep: Separation constraint ---
        # Mean hidden-state L2 distance for different-logic pairs across target layers
        c_sep = mx.array(0.0)
        n_sep = 0
        if cf_pairs:
            for layer_idx, hidden in layer_hiddens.items():
                h_mean = mx.mean(hidden, axis=1)
                for i, j in cf_pairs:
                    diff = h_mean[i] - h_mean[j]
                    dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
                    c_sep = c_sep + dist
                    n_sep += 1
        if n_sep > 0:
            c_sep = c_sep / n_sep

        # --- C_geo: Geodesic tail guardrail ---
        # Effective rank preservation at target layers.
        # Uses trace²/||G||_F² (Roy & Vetterli 2007) as a differentiable
        # proxy for spectral entropy. SVD has no VJP in MLX, so we use
        # the Gram matrix which is fully differentiable.
        c_geo = mx.array(0.0)
        for layer_idx, hidden in layer_hiddens.items():
            if layer_idx not in baseline_entropy:
                continue
            base_erank = baseline_entropy[layer_idx]

            # Flatten [batch, seq, hidden] -> [n, hidden]
            flat = hidden.reshape(-1, hidden.shape[-1]).astype(mx.float32)
            # Gram matrix G = X^T X ([hidden, hidden])
            G = flat.T @ flat
            trace_G = mx.sum(mx.diag(G))
            frobenius_sq = mx.sum(G * G)
            current_erank = (trace_G * trace_G) / (frobenius_sq + 1e-10)

            # Penalty: max(0, baseline_erank - current_erank) = rank drop
            gap = mx.array(base_erank) - current_erank
            c_geo = c_geo + mx.maximum(gap, mx.array(0.0))

        # --- Primal-dual combination ---
        # Multipliers are plain floats read from constraint_state (not in tape)
        inv_penalty = mx.maximum(c_inv - mx.array(config.epsilon_inv), mx.array(0.0))
        sep_penalty = mx.maximum(mx.array(config.margin_sep) - c_sep, mx.array(0.0))
        geo_penalty = mx.maximum(c_geo - mx.array(config.epsilon_tail), mx.array(0.0))

        total_loss = (
            ce_loss
            + constraint_state.mu_inv * inv_penalty
            + constraint_state.mu_sep * sep_penalty
            + constraint_state.mu_geo * geo_penalty
        )

        # Store constraint values for dual update (outside tape)
        # Use mx.eval to materialize before storing as Python floats
        _store_constraint_values(constraint_state, ce_loss, c_inv, c_sep, c_geo)

        return total_loss, ntoks

    return constrained_loss


def _store_constraint_values(state, ce_loss, c_inv, c_sep, c_geo):
    """Store constraint values on state for dual update.

    Called inside the loss function but these are just for logging/dual update,
    not for gradient computation.
    """
    # These will be evaluated after mx.eval in the training loop
    state._pending_ce = ce_loss
    state._pending_c_inv = c_inv
    state._pending_c_sep = c_sep
    state._pending_c_geo = c_geo


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

    def prepare_paired_dataset(
        self,
        samples: list[dict[str, Any]],
        tokenizer,
    ) -> list[dict[str, Any]]:
        """Tokenize paired samples with answer span masks and pair metadata.

        Returns list of dicts with keys:
            tokens: mx.array of token IDs
            answer_mask: mx.array of 0/1 mask (1 = answer token)
            logic_id: str
            template_id: str
            n_tokens: int
        """
        dataset: list[dict[str, Any]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue

            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            # Compute answer token mask
            answer_start_str = sample.get("answer_start", "")
            if answer_start_str and answer_start_str in text:
                # Find character offset of answer_start in text
                char_offset = text.index(answer_start_str)
                # Tokenize the prefix to find the token boundary
                prefix = text[:char_offset]
                prefix_tokens = tokenizer.encode(prefix)
                answer_token_start = len(prefix_tokens)
            else:
                # No answer_start or not found — mask everything (full sequence CE)
                answer_token_start = 0

            # answer_mask: 1 for answer tokens, 0 for scaffold tokens
            # Applied to the shifted target sequence (tokens[1:])
            mask = [0] * len(tokens)
            for i in range(answer_token_start, len(tokens)):
                mask[i] = 1

            dataset.append({
                "tokens": mx.array(tokens, dtype=mx.int32),
                "answer_mask": mx.array(mask, dtype=mx.float32),
                "logic_id": sample.get("logic_id", ""),
                "template_id": sample.get("template_id", ""),
                "n_tokens": len(tokens),
            })

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

    def measure_baseline_constraints(
        self,
        model,
        tokenizer,
        paired_dataset: list[dict[str, Any]],
        logic_groups: dict[str, list[int]],
        template_groups: dict[str, list[int]],
        target_layers: list[int],
        max_seq_length: int = 256,
    ) -> tuple[list[float], list[float], dict[int, float]]:
        """Measure baseline invariance/separation distances and spectral entropy.

        Runs on the BASE model (before NB-LoRA injection) to derive constraint
        thresholds. All thresholds come from geometry, not heuristics.

        Returns:
            (inv_distances, sep_distances, layer_entropies) where:
            - inv_distances: L2 distances between invariance pairs (same logic)
            - sep_distances: L2 distances between counterfactual pairs (same template)
            - layer_entropies: spectral entropy per target layer
        """
        target_layers_set = set(target_layers)

        # Collect hidden states at target layers for a subset of samples
        n_samples = min(len(paired_dataset), 50)  # limit for speed
        sample_indices = list(range(n_samples))

        # Forward pass each sample, collect hidden states per layer.
        # Store BOTH mean-pooled (for C_inv/C_sep distances) and full token-level
        # (for C_geo spectral entropy) to match training loss computation.
        hidden_states_mean: list[dict[int, Any]] = []  # per sample: {layer: [hidden]}
        hidden_states_full: list[dict[int, Any]] = []  # per sample: {layer: [seq, hidden]}

        base = getattr(model, "model", model)
        for idx in sample_indices:
            s = paired_dataset[idx]
            tokens = s["tokens"][:max_seq_length].reshape(1, -1)

            h = base.embed_tokens(tokens)

            layer_h_mean: dict[int, Any] = {}
            layer_h_full: dict[int, Any] = {}
            for layer_idx, layer in enumerate(base.layers):
                # Route masks per layer type (LFM2 hybrid architecture)
                if getattr(layer, "is_attention_layer", True):
                    layer_mask = "causal"
                else:
                    layer_mask = None
                h = layer(h, mask=layer_mask, cache=None)
                if isinstance(h, tuple):
                    h = h[0]
                if layer_idx in target_layers_set:
                    # Mean pool for C_inv/C_sep distance computation
                    mean_h = mx.mean(h, axis=(0, 1))
                    mx.eval(mean_h)
                    layer_h_mean[layer_idx] = mean_h
                    # Full token states for C_geo spectral entropy
                    # h is [1, seq, hidden] -> squeeze to [seq, hidden]
                    full_h = h.reshape(-1, h.shape[-1])
                    mx.eval(full_h)
                    layer_h_full[layer_idx] = full_h

            hidden_states_mean.append(layer_h_mean)
            hidden_states_full.append(layer_h_full)

        # Compute pairwise distances (using mean-pooled hidden states)
        inv_distances: list[float] = []
        sep_distances: list[float] = []

        # Invariance: same logic_id, different template_id
        for lid, members in logic_groups.items():
            active = [i for i in members if i < n_samples]
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ia, ib = active[a], active[b]
                    if paired_dataset[ia]["template_id"] == paired_dataset[ib]["template_id"]:
                        continue  # skip same template (not a true invariance pair)
                    for layer_idx in target_layers:
                        if layer_idx in hidden_states_mean[ia] and layer_idx in hidden_states_mean[ib]:
                            diff = hidden_states_mean[ia][layer_idx] - hidden_states_mean[ib][layer_idx]
                            dist = float(mx.sqrt(mx.sum(diff * diff)).item())
                            inv_distances.append(dist)

        # Separation: same template_id, different logic_id
        for tid, members in template_groups.items():
            active = [i for i in members if i < n_samples]
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ia, ib = active[a], active[b]
                    if paired_dataset[ia]["logic_id"] == paired_dataset[ib]["logic_id"]:
                        continue
                    for layer_idx in target_layers:
                        if layer_idx in hidden_states_mean[ia] and layer_idx in hidden_states_mean[ib]:
                            diff = hidden_states_mean[ia][layer_idx] - hidden_states_mean[ib][layer_idx]
                            dist = float(mx.sqrt(mx.sum(diff * diff)).item())
                            sep_distances.append(dist)

        # Effective rank per target layer (differentiable proxy for spectral entropy).
        # Uses trace(G)²/||G||_F² (Roy & Vetterli 2007) to match the training loss
        # computation which also uses this formula (SVD has no VJP in MLX).
        # Concatenates FULL token-level states across samples.
        layer_entropies: dict[int, float] = {}
        for layer_idx in target_layers:
            all_h = []
            for hs in hidden_states_full:
                if layer_idx in hs:
                    all_h.append(hs[layer_idx])
            if len(all_h) < 2:
                continue
            # Concatenate along token dimension: [total_tokens, hidden_dim]
            stacked = mx.concatenate(all_h, axis=0)
            flat = stacked.astype(mx.float32)
            # Gram matrix G = X^T X
            G = flat.T @ flat
            trace_G = float(mx.sum(mx.diag(G)).item())
            frobenius_sq = float(mx.sum(G * G).item())
            erank = (trace_G * trace_G) / (frobenius_sq + 1e-10)
            layer_entropies[layer_idx] = erank

        logger.info(
            "Baseline constraints: %d inv_distances (mean=%.4f), "
            "%d sep_distances (mean=%.4f), %d layer_entropies",
            len(inv_distances),
            sum(inv_distances) / max(1, len(inv_distances)),
            len(sep_distances),
            sum(sep_distances) / max(1, len(sep_distances)),
            len(layer_entropies),
        )

        return inv_distances, sep_distances, layer_entropies

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
        topo_monitor: bool = False,
        topo_probe_texts: list[str] | None = None,
        dim_monitor: bool = False,
        dim_probe_texts: list[str] | None = None,
        # Constrained geometric training (paired data)
        constraint_config: Any = None,  # ConstraintConfig or None
        constraint_state: Any = None,  # ConstraintState or None
        paired_dataset: list[dict[str, Any]] | None = None,
        logic_groups: dict[str, list[int]] | None = None,
        template_groups: dict[str, list[int]] | None = None,
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

        # Constrained training mode: use paired loss + paired batch iterator
        use_constrained = (
            constraint_config is not None
            and constraint_state is not None
            and paired_dataset is not None
        )

        if use_constrained:
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
            # For Lipschitz, use default_loss (without constraints) for cleaner estimate
            lipschitz_loss_fn = default_loss
        else:
            loss_fn = default_loss
            loss_value_and_grad = nn.value_and_grad(model, loss_fn)
            lipschitz_loss_fn = default_loss

        # Learning rate: override > measured Lipschitz > spectral proxy
        can_remeasure = False

        if lr_override is not None:
            eta = float(lr_override)
            logger.info("LR from override: %.2e", eta)
        else:
            # Robust Lipschitz: median across multiple batches
            L = self._measure_lipschitz_robust(
                model, train_dataset, batch_size, seq_length,
                lipschitz_loss_fn, n_batches=lipschitz_batches, n_iters=10, seed=seed,
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
        # Track measured Lipschitz constant for preconditioner-aware step bound
        L_current = L if (L is not None and L > 0) else (1.0 / eta)
        optimizer = opt.SGD(learning_rate=current_eta, momentum=0.0)

        # Cayley-aware Riemannian preconditioning (pullback metric correction).
        # The Cayley transform maps free (A_tilde, B_tilde) to the Stiefel
        # manifold. The pullback metric G = J^T J distorts the Euclidean gradient.
        # We precondition by G^{-1} ≈ M M^T where M = I + Z, correcting for the
        # Cayley transform's curvature. This gives the natural gradient (Amari 1998)
        # in unconstrained coordinates, equivalent to canonical Riemannian GD on
        # the Stiefel manifold. Refs: Wen & Yin (2013), Li et al. (ICLR 2020).
        #
        # Cayley-Riemannian natural gradient with preconditioner-aware step
        # bound: η ≤ 2/(L * λ_max(P)). Full anisotropy preserved. The caller
        # enforces the stability invariant m = η * L * λ_max(P) ≤ 2 per step.
        use_cayley_precond = True

        losses: list[tuple[int, float, float]] = []
        val_losses: list[float] = []
        epoch_metrics_list: list[EpochMetrics] = []
        dim_snapshots: list = []  # DimensionalSnapshot history for trend analysis
        stop_reason: str | None = None

        if use_constrained and paired_dataset is not None:
            batch_iter = iterate_paired_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
                loop=True, seed=seed,
            )
            n_batches_per_epoch = len(list(iterate_paired_batches(
                paired_dataset, batch_size, seq_length,
                logic_groups=logic_groups or {},
                template_groups=template_groups or {},
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

            if use_constrained:
                batch, lengths, answer_masks, inv_pairs, cf_pairs = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(
                    model, batch, lengths, answer_masks, inv_pairs, cf_pairs,
                )
            else:
                batch, lengths = next(batch_iter)
                (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)

            # Save raw gradient for stopping certificate (overwritten each step;
            # at epoch boundary, holds the last step's gradient).
            grad_raw_last = grad

            # Cayley-Riemannian preconditioning: correct for pullback metric
            # distortion of the Cayley parameterization (natural gradient).
            # d_t = P_t @ g_t, then θ -= η_t * d_t where η_t respects
            # the stability bound η ≤ 2/(L * λ_max(P)).
            precond_metrics: dict[str, float] = {}
            if use_cayley_precond:
                grad, precond_metrics = self._apply_cayley_preconditioner(
                    model, grad,
                )
                # Enforce stability bound: η ≤ 2/(L * λ_max(P))
                lambda_max_P = precond_metrics.get("precond_lambda_max", 1.0)
                eps_mach = math.ldexp(1.0, -23)  # float32 machine epsilon
                eta_max_precond = 2.0 / (L_current * lambda_max_P + eps_mach)
                eta_step = min(current_eta, eta_max_precond)
                # Invariant: m = η * L * λ_max(P) ≤ 2
                m_invariant = eta_step * L_current * lambda_max_P
                precond_metrics["eta_step"] = eta_step
                precond_metrics["eta_max_precond"] = eta_max_precond
                precond_metrics["m_invariant"] = m_invariant
                optimizer.learning_rate = mx.array(eta_step)
            else:
                optimizer.learning_rate = mx.array(current_eta)

            # Save preconditioned gradient for stopping certificate
            grad_precond_last = grad

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
                        lipschitz_loss_fn, n_batches=lipschitz_batches, n_iters=5,
                        seed=seed + epoch_num,  # Vary batches across epochs
                    )

                    if measured_L is not None and measured_L > 0:
                        eta_spectral = 1.0 / measured_L
                        L_current = measured_L  # Update for precond bound
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
                        current_eta = min(eta_spectral, current_eta)
                    else:
                        current_eta = min(eta_spectral, eta_ceiling)

                # 4. Weyl budget monitoring
                # NB-LoRA is bounded by construction (||BA||₂ ≤ σ_k via Cayley).
                # Per-layer Weyl crossing thresholds (gap/(2σ_k)) apply to unbounded
                # LoRA. For NB-LoRA, we monitor capacity usage: ||BA||₂/σ_k → 1.0.
                # Budget exhaustion means the adapter has consumed its available
                # spectral capacity — further training cannot improve without
                # violating bounds.
                max_ratio = None
                budget_exhausted_flag = False
                median_budget_ratio = None
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
                except Exception:
                    # Fallback: simple verify_bounds
                    try:
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
                    precond_lambda_max=precond_metrics.get("precond_lambda_max"),
                    precond_cond_max=precond_metrics.get("precond_cond_max"),
                    precond_gain_mean=precond_metrics.get("precond_gain_mean"),
                    precond_m_invariant=precond_metrics.get("m_invariant"),
                    precond_eta_step=precond_metrics.get("eta_step"),
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
                        em = epoch_metrics_list[-1]
                        em.dim_expansion_ratio = dim_snapshot.expansion_ratio
                        em.dim_peak_dim = dim_snapshot.peak_dim
                        em.dim_final_dim = dim_snapshot.final_dim
                        dim_snapshots.append(dim_snapshot)
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
                if precond_metrics:
                    lm = precond_metrics.get("precond_lambda_max", 0)
                    cm = precond_metrics.get("precond_cond_max", 0)
                    mi = precond_metrics.get("m_invariant", 0)
                    es = precond_metrics.get("eta_step", 0)
                    log_parts.append(f"P:λ={lm:.2f}")
                    log_parts.append(f"P:κ={cm:.1f}")
                    log_parts.append(f"η_eff={es:.2e}")
                    log_parts.append(f"m={mi:.3f}")
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

    def _apply_cayley_preconditioner(
        self, model, grad,
    ) -> tuple[Any, dict[str, float]]:
        """Cayley-aware Riemannian preconditioning for NB-LoRA gradients.

        The Cayley transform maps free (A_tilde, B_tilde) to semi-orthogonal
        (A, B) via W = (I + Z)^{-1}. The pullback metric tensor G = J^T J
        scales as W^T W. The natural gradient (Amari 1998) preconditions by
        G^{-1} = M M^T where M = I + Z.

        The preconditioner P = M M^T is applied WITHOUT normalization. The
        caller is responsible for enforcing the stability bound:

            η ≤ 2 / (L * λ_max(P))

        where L is the measured Lipschitz constant and λ_max(P) is the
        preconditioner's spectral radius (returned in metrics). This
        preserves the full anisotropy of the pullback metric — the
        preconditioner redistributes gradient across eigenspaces according
        to the actual Cayley curvature.

        The invariant m = η * L * λ_max(P) ≤ 2 must hold at every step.

        Properties:
        - No mx.linalg.inv needed (M M^T is a product, not an inverse)
        - Always positive definite: M M^T = I + 2 Y^T Y + Z Z^T
        - r×r cost (same as the Cayley transform's own matrix ops)
        - NOT in autograd path (applied to gradients post-backward)
        - λ_max from power iteration on r×r matrix (5 iters, negligible cost)

        Returns:
            (preconditioned_grad, metrics) where metrics["precond_lambda_max"]
            is the max λ_max across all layers (needed for step size bound).

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
        all_cond: list[float] = []
        all_gain: list[float] = []  # ||Pg|| / ||g||

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

            # P = M M^T (full pullback metric inverse, NO normalization)
            P = M @ M.T  # [r, r]

            # λ_max(P) via power iteration on the r×r SPD matrix (5 iters)
            v = mx.ones((r, 1)) / math.sqrt(r)
            mx.eval(v)
            lam = 1.0
            for _ in range(5):
                u = P @ v
                mx.eval(u)
                lam = float(mx.sum(v * u))  # Rayleigh quotient
                norm_u = float(mx.sqrt(mx.sum(u * u)))
                if norm_u < 1e-30:
                    break
                v = u * (1.0 / norm_u)
                mx.eval(v)
            lambda_max = max(lam, 1.0)  # Floor at 1 (P = I at init)

            # Condition number: λ_max / λ_min
            # P = M M^T = I + 2 Y^T Y + Z Z^T, so eigenvalues ≥ 1 always.
            # For tighter bound: λ_min ≥ trace(P) - (r-1)*λ_max
            tr = float(mx.trace(P))
            lambda_min = max(tr - (r - 1) * lambda_max, 1.0)
            cond = lambda_max / lambda_min

            # Measure gain: ||Pg|| / ||g|| for A_tilde gradient
            g_a = grad_flat[a_key]
            g_norm = float(mx.sqrt(mx.sum(g_a * g_a)))
            Pg_a = P @ g_a  # [r,r] @ [r,in]
            Pg_norm = float(mx.sqrt(mx.sum(Pg_a * Pg_a)))
            gain = Pg_norm / max(g_norm, 1e-30)

            # Apply full unnormalized preconditioner
            grad_flat[a_key] = Pg_a
            grad_flat[b_key] = P @ grad_flat[b_key]  # [r,r] @ [r,out]
            # S_raw lives in R^r (Euclidean) — no preconditioning

            mx.eval(grad_flat[a_key], grad_flat[b_key])

            all_lambda_max.append(lambda_max)
            all_cond.append(cond)
            all_gain.append(gain)

        metrics: dict[str, float] = {}
        if all_lambda_max:
            metrics["precond_lambda_max"] = max(all_lambda_max)
            metrics["precond_lambda_max_mean"] = sum(all_lambda_max) / len(all_lambda_max)
            metrics["precond_cond_max"] = max(all_cond)
            metrics["precond_cond_mean"] = sum(all_cond) / len(all_cond)
            metrics["precond_gain_mean"] = sum(all_gain) / len(all_gain)
            metrics["precond_gain_max"] = max(all_gain)

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

        Uses the same epsilon derivation as ``_measure_lipschitz()``:
        ε = sqrt(ε_mach) × max(||params||, 1.0).

        Cost: 2 × n_batches backward passes.

        Returns:
            Flat dict {param_key: hvp_array}, or None on failure.
        """
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

        trainable = dict(mlx_flatten(model.trainable_parameters()))
        original = {k: mx.array(v) for k, v in trainable.items()}
        mx.eval(*original.values())

        # Epsilon: sqrt(ε_mach) × ||params|| (optimal for central differences)
        param_norm = math.sqrt(
            sum(float(mx.sum(v * v)) for v in trainable.values())
        )
        sqrt_eps_mach = math.sqrt(math.ldexp(1.0, -23))
        eps = sqrt_eps_mach * max(param_norm, 1.0)

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

    def _compute_certificate_quantities(
        self,
        model,
        grad_raw: Any,
        grad_precond: Any,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
        mean_token_entropy: float | None,
        repetition_rate: float | None,
        grad_norm_history: list[float] | None = None,
    ):
        """Compute all quantities for the geometric stopping certificate.

        Orchestrates: preconditioned gradient norm, validation gradient,
        Hessian-vector product, bootstrap CI, per-batch worst-group bounds.

        Returns:
            ``StoppingCertificate`` from ``check_stopping_certificate()``.
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from modelcypher.core.domain.training.geometric_early_stopping import (
            check_stopping_certificate,
        )
        from modelcypher.core.support.statistics import bootstrap_ci

        # 1. Preconditioned gradient norm: ||P^{1/2} g|| = sqrt(g^T P g) = sqrt(g^T d)
        raw_flat = dict(mlx_flatten(grad_raw))
        precond_flat = dict(mlx_flatten(grad_precond))
        dot = sum(
            float(mx.sum(raw_flat[k] * precond_flat[k]))
            for k in raw_flat if k in precond_flat
        )
        precond_grad_norm = math.sqrt(max(dot, 0.0))

        # 2. Direction d_t (preconditioned gradient, already flat)
        d_t = precond_flat

        # 3. Validation gradient
        grad_val = self._compute_val_gradient(
            model, eval_dataset, batch_size, seq_length, n_batches,
        )

        alignment = 0.0
        curvature = 0.0
        per_batch_alignments: list[float] = []
        per_batch_curvatures: list[float] = []
        per_batch_ci_half_widths: list[float] = []

        if grad_val is not None:
            # 4. Alignment: a_t = grad_val^T @ d_t
            alignment = sum(
                float(mx.sum(grad_val[k] * d_t[k]))
                for k in grad_val if k in d_t
            )

            # 5. Val HVP: H_val @ d_t
            hvp = self._compute_val_hvp(
                model, eval_dataset, batch_size, seq_length, n_batches, d_t,
            )
            if hvp is not None:
                # 6. Curvature: b_t = d_t^T @ H_val @ d_t
                curvature = sum(
                    float(mx.sum(d_t[k] * hvp[k]))
                    for k in d_t if k in hvp
                )

            # 7. Per-batch worst-group: per-batch alignment with aggregate curvature
            from mlx_lm.tuner.trainer import default_loss, iterate_batches

            loss_vg = nn.value_and_grad(model, default_loss)
            batch_count = 0
            for batch, lengths in iterate_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            ):
                if batch_count >= n_batches:
                    break
                try:
                    (loss_i, _), grads_i = loss_vg(model, batch, lengths)
                    mx.eval(loss_i)
                    flat_i = dict(mlx_flatten(grads_i))
                    a_i = sum(
                        float(mx.sum(flat_i[k] * d_t[k]))
                        for k in flat_i if k in d_t
                    )
                    per_batch_alignments.append(a_i)
                    # Use aggregate curvature (avoids per-batch HVP)
                    per_batch_curvatures.append(curvature)
                except Exception:
                    pass
                batch_count += 1

        # 8. Per-batch losses for bootstrap CI
        per_batch_losses = self._compute_per_batch_val_losses(
            model, eval_dataset, batch_size, seq_length, n_batches,
        )

        val_ci_half_width = 0.0
        if len(per_batch_losses) >= 2:
            lower, upper = bootstrap_ci(
                per_batch_losses, confidence=0.95, n_bootstrap=200, seed=42,
            )
            val_ci_half_width = (upper - lower) / 2.0

        # Per-batch CI half-widths (each batch gets the aggregate CI as proxy)
        per_batch_ci_half_widths = [val_ci_half_width] * len(per_batch_alignments)

        return check_stopping_certificate(
            precond_grad_norm=precond_grad_norm,
            grad_norm_history=grad_norm_history,
            alignment=alignment,
            curvature=curvature,
            val_ci_half_width=val_ci_half_width,
            per_batch_alignments=per_batch_alignments if per_batch_alignments else None,
            per_batch_curvatures=per_batch_curvatures if per_batch_curvatures else None,
            per_batch_ci_half_widths=per_batch_ci_half_widths if per_batch_ci_half_widths else None,
            mean_token_entropy=mean_token_entropy,
            repetition_rate=repetition_rate,
        )

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

    def _compute_topological_metrics(
        self,
        model,
        tokenizer,
        probe_texts: list[str],
    ) -> dict[str, Any]:
        """Compute topological fingerprint and Ricci curvature from activation cloud.

        Experimental. Used for grokking phase detection: tracks Betti numbers,
        persistence entropy, and Ollivier-Ricci curvature per epoch.

        Args:
            model: Current model state.
            tokenizer: Tokenizer for the model.
            probe_texts: Short texts to collect activations from.

        Returns:
            Dict with topo_betti_0, topo_betti_1, topo_persistence_entropy,
            topo_mean_ricci_curvature, topo_ricci_curvature_std.
        """
        try:
            # 1. Collect activations for probe texts
            acts = self._backend.collect_hidden_activations(
                model, tokenizer, probe_texts,
            )
            if not acts:
                return {}

            # 2. Pool to [n_probes, hidden_dim] for middle layer
            mid_layer = max(acts.keys()) // 2
            layer_act = acts[mid_layer]  # [batch, seq, hidden]
            # Mean-pool over seq dimension → [batch, hidden]
            pooled = self._backend.mean(layer_act, axis=1)
            self._backend.eval(pooled)

            # Convert to list[list[float]] for TopologicalFingerprint
            points = pooled.tolist()

            # 3. Topological fingerprint (Betti numbers + persistence entropy)
            from modelcypher.core.domain.geometry.topological_fingerprint import (
                BackendTopologicalFingerprint,
            )

            topo = BackendTopologicalFingerprint(self._backend)
            fingerprint = topo.compute(points)

            # 4. Ollivier-Ricci curvature
            from modelcypher.core.domain.geometry.ollivier_ricci import (
                OllivierRicciCurvature,
            )

            orc = OllivierRicciCurvature(self._backend)
            ricci = orc.compute(pooled)

            return {
                "topo_betti_0": fingerprint.betti_numbers.get(0, 0),
                "topo_betti_1": fingerprint.betti_numbers.get(1, 0),
                "topo_persistence_entropy": fingerprint.summary.persistence_entropy,
                "topo_mean_ricci_curvature": ricci.mean_edge_curvature,
                "topo_ricci_curvature_std": ricci.std_edge_curvature,
            }
        except Exception:
            logger.debug("Topological metrics computation failed", exc_info=True)
            return {}

    def _compute_dimensional_snapshot(
        self,
        model,
        tokenizer,
        probe_texts: list[str],
        epoch: int,
    ):
        """Collect activations on probe texts, compute expansion ratio.

        Processes each text separately to handle variable sequence lengths,
        then merges token-level activations per layer before computing ID.

        Returns a DimensionalSnapshot or None on failure.
        """
        try:
            from modelcypher.core.domain.training.dimensional_monitor import (
                compute_expansion_from_activations,
            )

            # Process each text individually to avoid seq-length mismatch
            merged: dict[int, list] = {}
            for text in probe_texts:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                if not acts:
                    continue
                for layer_idx, act in acts.items():
                    # act shape: [1, seq, hidden] → reshape to [seq, hidden]
                    shape = act.shape
                    if len(shape) == 3:
                        reshaped = self._backend.reshape(act, (shape[1], shape[2]))
                    else:
                        reshaped = act
                    if layer_idx not in merged:
                        merged[layer_idx] = []
                    merged[layer_idx].append(reshaped)

            if not merged:
                return None

            # Concatenate all tokens per layer: list of [seq_i, hidden] → [total_tokens, hidden]
            combined = {}
            for layer_idx, arrays in merged.items():
                combined[layer_idx] = self._backend.concatenate(arrays, axis=0)
                self._backend.eval(combined[layer_idx])

            return compute_expansion_from_activations(combined, self._backend, epoch)
        except Exception:
            logger.debug("Dimensional snapshot computation failed", exc_info=True)
            return None

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
