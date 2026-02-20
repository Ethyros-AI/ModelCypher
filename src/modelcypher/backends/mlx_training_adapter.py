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
Weyl adapter-saturation monitoring for per-layer spectral crossing detection.
Measured Lipschitz LR (1/λ_max(Hessian)) for optimal step size.

The Cayley transform maps unconstrained (A_tilde, B_tilde) to semi-orthogonal
(A, B), guaranteeing ||2 * B^T @ S @ A||_2 <= 2 * max(S) <= sigma_k.
"""

from __future__ import annotations

import json
import hashlib
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
from modelcypher.core.domain.training.spectral_budget import (
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    compute_projected_residuals,
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
    spectral_ratio_growth_per_iter: float | None = None
    eta_ceiling: float | None = None
    adapter_saturation_median_ratio: float | None = None
    # Cayley-Riemannian preconditioner diagnostics
    precond_lambda_max: float | None = None
    precond_lambda_max_raw: float | None = None
    precond_cond_max: float | None = None
    precond_ipz_kappa_upper_max: float | None = None
    precond_ipz_rel_error_upper_max: float | None = None
    precond_ipz_warn_fraction: float | None = None
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
    dim_final_used_fraction: float | None = None
    dim_final_null_fraction: float | None = None
    dim_delta_from_baseline: float | None = None
    dim_null_recruitment_from_baseline: float | None = None
    dim_is_contracting: bool | None = None
    # Constrained training diagnostics (optional, when constraint_config provided)
    constraint_mu_inv: float | None = None
    constraint_mu_sep: float | None = None
    constraint_mu_geo: float | None = None
    constraint_C_inv: float | None = None
    constraint_C_sep: float | None = None
    constraint_C_geo: float | None = None
    # Geometric reshaping diagnostics (optional, when geometric_reshape=True)
    reshape_ce_norm: float | None = None
    reshape_expand_norm: float | None = None
    reshape_contrast_norm: float | None = None
    reshape_n_cf_pairs: int | None = None
    reshape_n_inv_pairs: int | None = None
    # Online correctness evaluation (optional, when eval_problems provided)
    online_eval_accuracy: float | None = None
    online_eval_n_correct: int | None = None
    online_eval_n_total: int | None = None
    online_eval_degraded: bool | None = None
    # REINFORCE outcome training (optional, when outcome_training=True)
    outcome_n_problems: int | None = None
    outcome_n_active: int | None = None
    outcome_signal_density: float | None = None
    outcome_n_steps: int | None = None
    # Outer similarity monitoring (optional, when rss_monitor=True)
    # Kucukahmetler et al. (2026) TMLR — base vs adapted relative representations
    rss_cosine: float | None = None
    rss_spearman: float | None = None
    rss_top1_agreement: float | None = None
    # Projected residual diagnostic (tighter than spectral norm ratio)
    projected_residual_max: float | None = None

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

    def set_initialization_vectors(self, u_k, v_k):
        """Store k-th singular vectors for projected residual monitoring.

        These are frozen (not trainable) and used only for diagnostics.
        Call once after creating the layer from a base weight.

        Args:
            u_k: k-th left singular vector of base weight [out, 1].
            v_k: k-th right singular vector of base weight [in, 1].
        """
        self._base_u_k = u_k
        self._base_v_k = v_k
        self.freeze(keys=["_base_u_k", "_base_v_k"])

    @property
    def base_u_k(self):
        """k-th left singular vector of base weight, or None."""
        return getattr(self, "_base_u_k", None)

    @property
    def base_v_k(self):
        """k-th right singular vector of base weight, or None."""
        return getattr(self, "_base_v_k", None)

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
    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Paired dataset must have at least batch_size={batch_size} "
            f"examples but only has {n}."
        )

    # Build sample pool with indices
    if seed is not None:
        import random
        random.seed(seed)

    # Group indices by logic_id for pair-aware batching
    logic_id_list = list(logic_groups.keys())

    while True:
        random.shuffle(logic_id_list)
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
                import random
                random.shuffle(cf_candidates)
                for idx in cf_candidates:
                    if len(batch_indices) >= batch_size:
                        break
                    batch_indices.append(idx)
                    used.add(idx)

            # Fill remaining slots from other samples
            if len(batch_indices) < batch_size:
                remaining = [i for i in range(n) if i not in used]
                import random
                random.shuffle(remaining)
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

            batch_list: list[list[int]] = []
            mask_list: list[list[float]] = []

            for j, s in enumerate(batch_samples):
                tlen = min(s["n_tokens"], max_seq_length)
                
                # Build padded token sequences
                toks = s["tokens"].tolist()[:tlen] if hasattr(s["tokens"], "tolist") else list(s["tokens"])[:tlen]
                toks = [int(t) for t in toks] + [0] * (max_len - tlen)
                batch_list.append(toks)
                
                # Build padded mask sequences
                amask = s["answer_mask"].tolist()[:tlen] if hasattr(s["answer_mask"], "tolist") else list(s["answer_mask"])[:tlen]
                amask = [float(a) for a in amask] + [0.0] * (max_len - tlen)
                mask_list.append(amask)
                
                lengths_list[j] = tlen

            batch_tensor = mx.array(batch_list, dtype=mx.int32)
            lengths_tensor = mx.array(
                [[0, l] for l in lengths_list], dtype=mx.int32,
            )
            answer_masks_tensor = mx.array(mask_list, dtype=mx.float32)

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
# Answer-Masked Batch Iterator (for answer-span CE training)
# =============================================================================


def iterate_masked_batches(
    dataset: list[tuple[Any, Any, int]],
    batch_size: int,
    max_seq_length: int,
    *,
    train: bool = True,
    seed: int = 0,
    loop: bool = True,
):
    """Yield (inputs, targets, masks) batches for answer-masked CE training.

    Each dataset element is (tokens, mask, 0) from ``prepare_masked_dataset``.
    Pads sequences and masks to max length in batch. Mask is shifted to align
    with shifted targets (mask[1:]).

    Args:
        dataset: List of (tokens_array, mask_array, placeholder) tuples.
        batch_size: Samples per batch.
        max_seq_length: Truncation length.
        train: Shuffle if True.
        seed: Random seed for shuffling.
        loop: If True, loop forever; if False, single pass.
    """
    import random as _rng

    indices = list(range(len(dataset)))

    while True:
        if train:
            _rng.seed(seed)
            _rng.shuffle(indices)
            seed += 1

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            # Collect and truncate
            batch_tokens = []
            batch_masks = []
            batch_lengths = []

            for idx in batch_indices:
                tokens_arr, mask_arr, _ = dataset[idx]
                # Convert to lists for truncation + padding
                toks = tokens_arr.tolist()[:max_seq_length]
                msk = mask_arr.tolist()[:max_seq_length]
                batch_tokens.append(toks)
                batch_masks.append(msk)
                batch_lengths.append(len(toks))

            # Pad to max length in batch
            max_len = max(batch_lengths)
            for i in range(len(batch_tokens)):
                pad_len = max_len - len(batch_tokens[i])
                batch_tokens[i] = batch_tokens[i] + [0] * pad_len
                batch_masks[i] = batch_masks[i] + [0.0] * pad_len

            all_tokens = mx.array(batch_tokens, dtype=mx.int32)
            all_masks = mx.array(batch_masks, dtype=mx.float32)

            # Shift: inputs = tokens[:-1], targets = tokens[1:], masks = mask[1:]
            inputs = all_tokens[:, :-1]
            targets = all_tokens[:, 1:]
            shifted_masks = all_masks[:, 1:]

            # Also zero out padding positions in mask (already 0 from pad_val=0.0)
            # and positions beyond each sequence's actual length
            lengths = mx.array(batch_lengths)
            steps = mx.arange(1, targets.shape[1] + 1)
            length_mask = (steps[None] < lengths[:, None]).astype(mx.float32)
            shifted_masks = shifted_masks * length_mask

            yield inputs, targets, shifted_masks

        if not loop:
            break


# =============================================================================
# Structured Batch Sampler (template-first for guaranteed contrastive pairs)
# =============================================================================


def iterate_structured_batches(
    dataset: list[dict[str, Any]],
    batch_size: int,
    max_seq_length: int,
    logic_groups: dict[str, list[int]],
    template_groups: dict[str, list[int]],
    loop: bool = False,
    seed: int | None = None,
):
    """Build batches template-first to guarantee contrastive pairs.

    Strategy:
    1. Pick K logic forms at random (K = batch_size // templates_per_batch).
    2. For each batch, pick ceil(batch_size / K) templates.
    3. Include one sample per (logic, template) combination.

    This guarantees every batch has:
    - Contrastive pairs (same template, different logic) in every batch.
    - Invariance pairs (same logic, different template) in every batch.

    With 10 logic forms × 44 templates, typical batch of 8:
    - 2 logic forms × 4 templates = 8 samples
    - 4 contrastive pairs (one per template)
    - 6 invariance pairs per logic × 2 logics = 12 invariance pairs
    """
    import numpy as np

    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {n}."
        )

    if seed is not None:
        import random
        random.seed(seed)

    # Build (template_id, logic_id) -> sample index mapping
    tl_to_idx: dict[tuple[str, str], list[int]] = {}
    for idx, s in enumerate(dataset):
        key = (s["template_id"], s["logic_id"])
        tl_to_idx.setdefault(key, []).append(idx)

    all_templates = sorted(set(s["template_id"] for s in dataset))
    all_logics = sorted(set(s["logic_id"] for s in dataset))

    # Decide how many logic forms and templates per batch
    # We want at least 2 logic forms (for contrastive) and at least 2 templates (for invariance)
    n_logics_per_batch = max(2, min(len(all_logics), batch_size // 2))
    n_templates_per_batch = max(2, batch_size // n_logics_per_batch)
    # Adjust so product doesn't exceed batch_size
    while n_logics_per_batch * n_templates_per_batch > batch_size and n_logics_per_batch > 2:
        n_logics_per_batch -= 1

    while True:
        np.random.shuffle(all_templates)
        np.random.shuffle(all_logics)

        # Walk through templates in chunks
        for t_start in range(0, len(all_templates), n_templates_per_batch):
            batch_templates = all_templates[t_start : t_start + n_templates_per_batch]
            if len(batch_templates) < 2:
                continue

            # Pick logic forms for this batch
            batch_logics = all_logics[:n_logics_per_batch]
            np.random.shuffle(all_logics)  # rotate for next batch

            # Select one sample per (template, logic) pair
            batch_indices: list[int] = []
            for tid in batch_templates:
                for lid in batch_logics:
                    candidates = tl_to_idx.get((tid, lid), [])
                    if candidates:
                        batch_indices.append(candidates[np.random.randint(len(candidates))])
                    if len(batch_indices) >= batch_size:
                        break
                if len(batch_indices) >= batch_size:
                    break

            if len(batch_indices) < 4:  # need at least 4 for meaningful pairs
                continue

            actual_bs = min(len(batch_indices), batch_size)
            batch_indices = batch_indices[:actual_bs]

            # Build tensors (same format as iterate_paired_batches)
            batch_samples = [dataset[i] for i in batch_indices]
            lengths_list = [s["n_tokens"] for s in batch_samples]

            pad_to = 32
            max_len = 1 + pad_to * ((max(lengths_list) + pad_to - 1) // pad_to)
            max_len = min(max_len, max_seq_length)

            batch_arr = np.zeros((actual_bs, max_len), dtype=np.int32)
            mask_arr = np.zeros((actual_bs, max_len), dtype=np.float32)

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

            # Discover pairs within this batch
            batch_logic_ids = [batch_samples[j]["logic_id"] for j in range(actual_bs)]
            batch_template_ids = [batch_samples[j]["template_id"] for j in range(actual_bs)]

            # Invariance pairs: same logic_id, different template_id
            inv_pairs: list[tuple[int, int]] = []
            logic_to_pos: dict[str, list[int]] = {}
            for pos, lid_val in enumerate(batch_logic_ids):
                logic_to_pos.setdefault(lid_val, []).append(pos)
            for positions in logic_to_pos.values():
                for a in range(len(positions)):
                    for b in range(a + 1, len(positions)):
                        if batch_template_ids[positions[a]] != batch_template_ids[positions[b]]:
                            inv_pairs.append((positions[a], positions[b]))

            # Contrastive pairs: same template_id, different logic_id
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
# Geometry-Reshaping Loss Factory
# =============================================================================


def make_geometric_reshaping_loss(
    target_layers: list[int],
):
    """Create a loss that actively reshapes the model's internal geometry.

    Three components, gradient-balanced:

    1. L_ce: Answer-only cross-entropy (model must still predict correct tokens).
    2. L_expand: -log(erank/d) at target layers. MAXIMIZES effective rank,
       forcing the model to use more of its available dimensions.
       Uses trace²/||G||_F² as differentiable erank proxy (Roy & Vetterli 2007).
    3. L_contrast: Contrastive loss on hidden states. SEPARATES different-logic
       trajectories and ALIGNS same-logic trajectories across templates.
       Vectorized cosine similarity computation.

    Gradient balancing: At step 0, each component's initial value is recorded
    for value normalization (so each starts at 1.0). Then calibrate_weights()
    measures the gradient norm of each component separately and sets
    component_weights so all three contribute equally to parameter updates.
    Weights are derived from the model's own gradient structure — no magic numbers.

    Returns function with signature:
        loss_fn(model, batch, lengths, answer_masks, inv_pairs, cf_pairs) -> (loss, ntoks)
    """
    target_layers_set = set(target_layers)

    # Self-normalization state (captured in closure)
    init_values: dict[str, Any] = {}
    # Gradient-balanced weights (set by calibrate_weights, default equal)
    component_weights: dict[str, float] = {"ce": 1.0, "expand": 1.0, "contrast": 1.0}
    # Calibration mode flag: when True, uses explicit weights for all components
    # (no alpha coupling) so gradient norms can be measured independently.
    calibrating: dict[str, bool] = {"active": False}
    # Latest component values for logging (populated each call, read externally)
    component_metrics: dict[str, Any] = {}

    def geometric_loss(model, batch, lengths, answer_masks, inv_pairs, cf_pairs):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        amask = answer_masks[:, 1:]

        # --- Manual forward pass with hidden state collection ---
        base = getattr(model, "model", model)
        h = base.embed_tokens(inputs)

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

        # Final norm + logits
        if hasattr(base, "norm"):
            h = base.norm(h)
        elif hasattr(base, "embedding_norm"):
            h = base.embedding_norm(h)
        if hasattr(model, "lm_head"):
            logits = model.lm_head(h)
        else:
            logits = base.embed_tokens.as_linear(h)

        # --- L_ce: answer-only cross-entropy ---
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(
            steps >= lengths[:, 0:1], steps <= lengths[:, 1:],
        ).astype(mx.float32)
        combined_mask = length_mask * amask
        ce = nn.losses.cross_entropy(logits, targets) * combined_mask
        ntoks = mx.maximum(combined_mask.sum(), mx.array(1.0))
        ce_loss = ce.astype(mx.float32).sum() / ntoks

        # --- L_expand: maximize effective rank at target layers ---
        # erank = trace(G)^2 / ||G||_F^2  (Roy & Vetterli 2007)
        # Loss = -log(erank / d), always positive, 0 when fully utilizing all dimensions.
        expand_loss = mx.array(0.0)
        n_expand = 0
        for _layer_idx, hidden in layer_hiddens.items():
            d = hidden.shape[-1]
            flat = hidden.reshape(-1, d).astype(mx.float32)
            G = flat.T @ flat
            trace_G = mx.sum(mx.diag(G))
            frob_sq = mx.sum(G * G)
            erank = (trace_G * trace_G) / (frob_sq + 1e-10)
            # -log(erank/d): penalizes low effective rank
            expand_loss = expand_loss + (-mx.log(erank / d + 1e-10))
            n_expand += 1
        if n_expand > 0:
            expand_loss = expand_loss / n_expand

        # --- L_contrast: contrastive trajectory separation ---
        # Vectorized: build index arrays for all pairs, compute in one shot.
        contrast_loss = mx.array(0.0)
        n_contrast_terms = 0

        has_cf = len(cf_pairs) > 0
        has_inv = len(inv_pairs) > 0

        if has_cf or has_inv:
            for _layer_idx, hidden in layer_hiddens.items():
                # Mean-pool over sequence → [batch, hidden_dim]
                h_mean = mx.mean(hidden, axis=1).astype(mx.float32)
                h_norms = mx.sqrt(mx.sum(h_mean * h_mean, axis=-1, keepdims=True) + 1e-8)
                h_unit = h_mean / h_norms  # [batch, d], unit vectors

                # Contrastive: same template, different logic → push APART
                if has_cf:
                    cf_arr = mx.array(cf_pairs, dtype=mx.int32)  # [N_cf, 2]
                    h_i = h_unit[cf_arr[:, 0]]  # [N_cf, d]
                    h_j = h_unit[cf_arr[:, 1]]  # [N_cf, d]
                    cf_sim = mx.sum(h_i * h_j, axis=-1)  # [N_cf]
                    # Map [-1, 1] → [0, 1]: 0 when orthogonal, 1 when identical
                    contrast_loss = contrast_loss + mx.mean((cf_sim + 1) / 2)
                    n_contrast_terms += 1

                # Invariance: same logic, different template → pull TOGETHER
                if has_inv:
                    inv_arr = mx.array(inv_pairs, dtype=mx.int32)  # [N_inv, 2]
                    h_i = h_unit[inv_arr[:, 0]]
                    h_j = h_unit[inv_arr[:, 1]]
                    inv_sim = mx.sum(h_i * h_j, axis=-1)  # [N_inv]
                    # Map [-1, 1] → [0, 1]: 0 when identical, 1 when orthogonal
                    contrast_loss = contrast_loss + mx.mean((1 - inv_sim) / 2)
                    n_contrast_terms += 1

        if n_contrast_terms > 0:
            contrast_loss = contrast_loss / n_contrast_terms

        # --- Self-normalization ---
        # At step 0, record initial values. Divide each component by its
        # initial value so all start at 1.0. No arbitrary lambda weights.
        if "ce" not in init_values:
            init_values["ce"] = mx.stop_gradient(ce_loss) + 1e-10
            init_values["expand"] = mx.stop_gradient(expand_loss) + 1e-10
            init_values["contrast"] = (
                mx.stop_gradient(contrast_loss) + 1e-10
                if n_contrast_terms > 0
                else mx.array(1.0)
            )

        ce_norm = ce_loss / init_values["ce"]
        expand_norm = expand_loss / init_values["expand"]
        contrast_norm = contrast_loss / init_values["contrast"]

        if calibrating["active"]:
            # Calibration mode: explicit weights for all 3 components,
            # no alpha coupling. Used to measure per-component gradient norms.
            total = (
                component_weights["ce"] * ce_norm
                + component_weights["expand"] * expand_norm
                + component_weights["contrast"] * contrast_norm
            )
        else:
            # Training mode: CE always on, geometric terms scale with
            # CE convergence progress. alpha=0 when CE at initial value,
            # alpha→1 as CE→0. stop_gradient: constant for backprop.
            alpha = mx.stop_gradient(mx.clip(1.0 - ce_norm, 0.0, 1.0))
            total = (
                ce_norm
                + alpha * component_weights["expand"] * expand_norm
                + alpha * component_weights["contrast"] * contrast_norm
            )

        # Store raw component values for logging (outside gradient tape)
        component_metrics["ce_raw"] = ce_loss
        component_metrics["expand_raw"] = expand_loss
        component_metrics["contrast_raw"] = contrast_loss
        component_metrics["ce_norm"] = ce_norm
        component_metrics["expand_norm"] = expand_norm
        component_metrics["contrast_norm"] = contrast_norm
        component_metrics["n_cf_pairs"] = len(cf_pairs)
        component_metrics["n_inv_pairs"] = len(inv_pairs)
        component_metrics["alpha"] = alpha if not calibrating["active"] else mx.array(1.0)

        return total, ntoks

    geometric_loss.component_metrics = component_metrics  # type: ignore[attr-defined]
    geometric_loss.component_weights = component_weights  # type: ignore[attr-defined]
    geometric_loss.calibrating = calibrating  # type: ignore[attr-defined]
    return geometric_loss


def calibrate_geometric_weights(
    model,
    loss_fn,
    batch,
    lengths,
    answer_masks,
    inv_pairs: list,
    cf_pairs: list,
) -> dict[str, float]:
    """Measure per-component gradient norms and set balanced weights.

    Runs 3 backward passes (one per component) on a single calibration batch.
    Sets loss_fn.component_weights so all three components contribute equally
    to the parameter update norm. All weights derived from the model's own
    gradient structure.

    Returns:
        Dict with keys: ce_gnorm, expand_gnorm, contrast_gnorm, w_ce, w_expand,
        w_contrast, ratio_expand, ratio_contrast.
    """
    from mlx.utils import tree_flatten as _flatten

    # Use calibration mode: explicit weights for all components, no alpha.
    weights_ref = loss_fn.component_weights
    calib_ref = loss_fn.calibrating
    calib_ref["active"] = True

    def _grad_norm(w_ce: float, w_expand: float, w_contrast: float) -> float:
        """Set weights, compute grad, return L2 norm of all trainable params."""
        old = dict(weights_ref)
        weights_ref["ce"] = w_ce
        weights_ref["expand"] = w_expand
        weights_ref["contrast"] = w_contrast

        vag = nn.value_and_grad(model, loss_fn)
        (loss_val, _), grads = vag(model, batch, lengths, answer_masks, inv_pairs, cf_pairs)
        mx.eval(loss_val)

        flat = dict(_flatten(grads))
        sq_sum = 0.0
        for _, g in flat.items():
            sq_sum += float(mx.sum(g * g))
        mx.eval()

        # Restore weights
        weights_ref.update(old)
        return math.sqrt(sq_sum)

    ce_gnorm = _grad_norm(1.0, 0.0, 0.0)
    expand_gnorm = _grad_norm(0.0, 1.0, 0.0)
    contrast_gnorm = _grad_norm(0.0, 0.0, 1.0)

    # Set weights to equalize gradient contributions:
    # w_i = ce_gnorm / component_gnorm (so ||w_i * ∇L_i|| ≈ ||∇L_ce||)
    # CE weight stays at 1.0 as reference.
    w_expand = ce_gnorm / max(expand_gnorm, 1e-12)
    w_contrast = ce_gnorm / max(contrast_gnorm, 1e-12)

    weights_ref["ce"] = 1.0
    weights_ref["expand"] = w_expand
    weights_ref["contrast"] = w_contrast
    # Switch back to training mode (adaptive CE coupling)
    calib_ref["active"] = False

    return {
        "ce_gnorm": ce_gnorm,
        "expand_gnorm": expand_gnorm,
        "contrast_gnorm": contrast_gnorm,
        "w_ce": 1.0,
        "w_expand": w_expand,
        "w_contrast": w_contrast,
        "ratio_expand": ce_gnorm / max(expand_gnorm, 1e-12),
        "ratio_contrast": ce_gnorm / max(contrast_gnorm, 1e-12),
    }


# =============================================================================
# EOS-Excluded Loss (Global EOS Mask)
# =============================================================================


def make_eos_excluded_loss(eos_token_id: int):
    """Like ``default_loss`` but excludes EOS tokens from CE.

    The base model already has EOS behaviour from pre-training.  Training CE
    on EOS in full-sequence data produces gradients that erode the adapter's
    ability to stop generating, leading to degenerate (non-terminating) outputs.
    """

    def _loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)

        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        # Exclude positions where the target is the EOS token
        eos_mask = targets != eos_token_id
        mask = mx.logical_and(mask, eos_mask)

        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        ce = ce.astype(mx.float32).sum() / mx.maximum(ntoks, mx.array(1.0))

        return ce, ntoks

    return _loss


# =============================================================================
# Entropy-Regularized Loss Wrapper
# =============================================================================


def make_entropy_regularized_loss(entropy_floor: float):
    """Wrap ``default_loss`` with a logit-entropy floor regularizer.

    When mean per-token entropy drops below *entropy_floor*, a penalty is
    added: ``L_entropy = max(0, entropy_floor - mean_entropy)``.

    No artificial weight parameter. Both CE and entropy are in nats
    (same units), so their gradients are in the same scale. The penalty
    gradient has natural magnitude determined by the softmax Jacobian.
    Any multiplicative weight would be a magic number.

    The entropy floor should be derived from a baseline measurement:
    ``floor = baseline_entropy * (1 - sqrt(eps_f32))``.

    The returned function has the same signature as ``default_loss``:
        loss_fn(model, batch, lengths) -> (loss, ntoks)
    """
    # IEEE 754: smallest positive normal float32 = 2^(-126) ≈ 1.175e-38.
    # log(tiny) ≈ -87.3, well within float32 range.
    _LOG_EPS = math.ldexp(1.0, -126)

    # Mutable state exposed for diagnostics
    entropy_metrics: dict[str, float] = {}

    def _entropy_loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # --- Standard CE (replicate default_loss) ---
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(
            steps >= lengths[:, 0:1], steps <= lengths[:, 1:],
        ).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, targets) * length_mask
        ntoks = mx.maximum(length_mask.sum(), mx.array(1.0))
        ce_loss = ce.sum() / ntoks

        # --- Entropy floor regularization ---
        # Shannon entropy: H = -sum(p * log(p))
        # log stability: finfo(float32).tiny (IEEE 754 derived)
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(probs + _LOG_EPS)
        token_entropy = -mx.sum(probs * log_probs, axis=-1)  # [batch, seq]
        # Mask to only count valid tokens
        masked_entropy = token_entropy * length_mask
        mean_entropy = masked_entropy.sum() / ntoks

        # ReLU penalty: only active when entropy drops below floor
        entropy_penalty = mx.maximum(
            mx.array(0.0),
            mx.array(entropy_floor) - mean_entropy,
        )

        total_loss = ce_loss + entropy_penalty

        # Store metrics for external logging
        entropy_metrics["mean_entropy"] = float(mean_entropy)
        entropy_metrics["entropy_penalty"] = float(entropy_penalty)
        entropy_metrics["entropy_floor"] = entropy_floor
        entropy_metrics["ce_loss"] = float(ce_loss)

        return total_loss, ntoks

    _entropy_loss.entropy_metrics = entropy_metrics
    _entropy_loss.entropy_floor = entropy_floor

    return _entropy_loss


def make_entropy_regularized_answer_masked_loss(entropy_floor: float):
    """Answer-masked CE with logit-entropy floor regularizer.

    CE is computed only on answer tokens (via mask). Entropy penalty is
    computed on ALL valid tokens — collapse can happen anywhere, not just
    in answer spans.

    Same signature as ``_answer_masked_loss``:
        loss_fn(model, inputs, targets, masks) -> (loss, ntoks)
    """
    _LOG_EPS = math.ldexp(1.0, -126)  # IEEE 754: smallest positive normal float32
    entropy_metrics: dict[str, float] = {}

    def _ent_masked_loss(model, inputs, targets, masks):
        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # --- Answer-masked CE ---
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        masked_ce = ce * masks
        ntoks = masks.sum()
        ce_loss = masked_ce.sum() / mx.maximum(ntoks, mx.array(1.0))

        # --- Entropy floor on ALL valid tokens ---
        # A token is valid if it has any mask > 0 OR if it precedes
        # a masked token.  For simplicity, use (targets != pad) which
        # for packed sequences equals "all positions".  Since answer-masked
        # data is not length-padded (each sample is pre-truncated), every
        # position is valid.  Use masks.sum(axis=0) > 0 would exclude
        # non-answer tokens.  We want ALL tokens for entropy health.
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(probs + _LOG_EPS)
        token_entropy = -mx.sum(probs * log_probs, axis=-1)  # [batch, seq]
        # All positions valid in answer-masked data (no length padding)
        mean_entropy = token_entropy.mean()

        entropy_penalty = mx.maximum(
            mx.array(0.0),
            mx.array(entropy_floor) - mean_entropy,
        )

        total_loss = ce_loss + entropy_penalty

        entropy_metrics["mean_entropy"] = float(mean_entropy)
        entropy_metrics["entropy_penalty"] = float(entropy_penalty)
        entropy_metrics["entropy_floor"] = entropy_floor
        entropy_metrics["ce_loss"] = float(ce_loss)

        return total_loss, ntoks

    _ent_masked_loss.entropy_metrics = entropy_metrics
    _ent_masked_loss.entropy_floor = entropy_floor

    return _ent_masked_loss


def measure_baseline_entropy(model, dataset, batch_size, seq_length, *, n_batches: int):
    """Measure baseline mean token entropy.

    Returns the mean token entropy, which is used to derive the entropy floor:
    ``floor = baseline_entropy * (1 - sqrt(eps_f32))``.

    Parameters
    ----------
    n_batches : int
        Number of batches to average over. Should match ``eval_batches``
        (the same batch count used for validation loss), so the entropy
        estimate has comparable coverage to the loss estimate.
    """
    from mlx_lm.tuner.trainer import iterate_batches

    _LOG_EPS = math.ldexp(1.0, -126)  # IEEE 754: smallest positive normal float32

    all_entropies: list[float] = []
    count = 0

    for batch, lengths in iterate_batches(
        dataset, batch_size, seq_length,
    ):
        if count >= n_batches:
            break
        inputs = batch[:, :-1]
        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # Mask: match default_loss format (lengths has start/end columns)
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(
            steps >= lengths[:, 0:1], steps <= lengths[:, 1:],
        ).astype(mx.float32)

        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(probs + _LOG_EPS)
        token_entropy = -mx.sum(probs * log_probs, axis=-1)
        masked_entropy = token_entropy * length_mask
        ntoks = mx.maximum(length_mask.sum(), mx.array(1.0))
        mean_ent = float(masked_entropy.sum() / ntoks)
        mx.eval(mean_ent)

        all_entropies.append(mean_ent)
        count += 1

    if not all_entropies:
        return None

    return sum(all_entropies) / len(all_entropies)


# =============================================================================
# REINFORCE Outcome Loss (Layer 3 of outcome-based training)
# =============================================================================


def make_outcome_loss():
    """Create a REINFORCE outcome loss function.

    Computes teacher-forced log-probabilities weighted by per-completion
    advantages:

        L_outcome = -mean(A_i * sum_t log π(y_{i,t} | y_{i,<t}))

    Positive advantage (correct completion) → increase its log-prob.
    Negative advantage (incorrect completion) → decrease its log-prob.
    CE can only increase probability. REINFORCE can also decrease it.

    The returned function has signature:
        loss_fn(model, batch, lengths, advantages) -> (loss, ntoks)

    Where:
        batch: [B, S] token IDs (padded)
        lengths: [B] actual sequence lengths (for masking)
        advantages: [B] per-completion advantage values (constants, not differentiated)

    NB-LoRA replaces KL regularization. Standard GRPO uses β * KL(π || π_ref)
    to prevent drift. NB-LoRA's spectral budget (||BA||₂/σ_k < 1) bounds drift
    by construction. No reference model needed. No β to tune.

    Reference: Williams (1992), "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning", Machine Learning 8(3-4)
    """

    def _outcome_loss(model, batch, lengths, advantages):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # Length mask: valid tokens only
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = (steps[None] < lengths[:, None]).astype(mx.float32)

        # Per-token log probs (teacher-forced)
        # nn.losses.cross_entropy returns -log P(target), always positive.
        # Negate to get log P(target), always negative.
        ce_per_token = nn.losses.cross_entropy(logits, targets)  # [B, S]
        log_probs = -ce_per_token  # [B, S]

        # Sum log probs per sequence (within mask)
        seq_log_probs = (log_probs * length_mask).sum(axis=-1)  # [B]

        # REINFORCE: L = -mean(A_i * seq_log_prob_i)
        # A > 0 (correct): seq_log_prob < 0, product < 0, -product > 0
        #   → minimizing loss increases log prob of correct completions
        # A < 0 (incorrect): seq_log_prob < 0, product > 0, -product < 0
        #   → minimizing loss decreases log prob of incorrect completions
        weighted = advantages * seq_log_probs  # [B]
        loss = -weighted.mean()

        ntoks = length_mask.sum()
        return loss, ntoks

    return _outcome_loss


def prepare_outcome_batches(completions, batch_size, seq_length):
    """Convert (tokens, advantage) pairs into padded MLX batches.

    Parameters
    ----------
    completions : list[tuple[list[int], float]]
        (token_ids, advantage) pairs. Only nonzero-advantage completions.
    batch_size : int
        Batch size for REINFORCE gradient steps.
    seq_length : int
        Maximum sequence length (truncate + pad).

    Returns
    -------
    list[tuple[mx.array, mx.array, mx.array]]
        (batch [B, S], lengths [B], advantages [B]) tuples.
    """
    if not completions:
        return []

    batches = []
    for i in range(0, len(completions), batch_size):
        group = completions[i:i + batch_size]
        tokens_list = []
        advs = []
        lens = []
        for tokens, advantage in group:
            t = tokens[:seq_length]
            lens.append(len(t))
            # Pad to seq_length with zeros
            padded = t + [0] * (seq_length - len(t))
            tokens_list.append(padded)
            advs.append(advantage)

        batch = mx.array(tokens_list)
        lengths = mx.array(lens)
        advantages = mx.array(advs)
        batches.append((batch, lengths, advantages))

    return batches


# =============================================================================
# Constrained Loss Factory (for constrained geometric training — EXPERIMENTAL)
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
    target_entropy = (
        config.target_entropy if getattr(config, "target_entropy", None)
        else config.baseline_entropy
    )

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

        # --- C_geo: Geometric expansion objective ---
        # Effective-rank shortfall at target layers.
        # Uses trace²/||G||_F² (Roy & Vetterli 2007) as a differentiable
        # proxy for spectral entropy. SVD has no VJP in MLX, so we use
        # the Gram matrix which is fully differentiable.
        c_geo = mx.array(0.0)
        n_geo = 0
        for layer_idx, hidden in layer_hiddens.items():
            if layer_idx not in target_entropy:
                continue
            target_erank = target_entropy[layer_idx]

            # Flatten [batch, seq, hidden] -> [n, hidden]
            flat = hidden.reshape(-1, hidden.shape[-1]).astype(mx.float32)
            # Gram matrix G = X^T X ([hidden, hidden])
            G = flat.T @ flat
            trace_G = mx.sum(mx.diag(G))
            frobenius_sq = mx.sum(G * G)
            current_erank = (trace_G * trace_G) / (frobenius_sq + 1e-10)

            # Penalty: max(0, target_erank - current_erank) = shortfall
            gap = mx.array(target_erank) - current_erank
            c_geo = c_geo + mx.maximum(gap, mx.array(0.0))
            n_geo += 1
        if n_geo > 0:
            c_geo = c_geo / n_geo

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
        """Tokenize samples into mlx-lm iterate_batches format.

        Appends EOS token to each sequence so the model learns when to stop.
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        dataset: list[tuple[Any, int]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            tokens = tokenizer.encode(text)
            if eos_id is not None and (not tokens or tokens[-1] != eos_id):
                tokens.append(eos_id)
            if len(tokens) < 2:
                continue
            dataset.append((mx.array(tokens, dtype=mx.int32), 0))
        return dataset

    def prepare_masked_dataset(
        self, samples: list[dict[str, Any]], tokenizer
    ) -> list[tuple[Any, Any, int]]:
        """Tokenize with answer-span masks for answer-only CE training.

        Returns list of (tokens, mask, 0) tuples where:
        - tokens: mx.array of token IDs (with EOS appended)
        - mask: mx.array of floats, 1.0 for answer tokens, 0.0 for prompt and EOS
        - 0: placeholder for compatibility

        The mask is aligned with the full token sequence. When computing loss,
        the caller shifts mask[1:] to align with shifted targets.

        EOS is excluded from the mask (mask=0.0) because the base model already
        has EOS behaviour from pre-training. Training CE on EOS in every answer
        span creates an outsized gradient that biases the adapter toward
        premature termination.

        Samples missing ``answer_start`` get mask=1.0 for all content tokens
        (full-sequence CE) with EOS still excluded.
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        dataset: list[tuple[Any, Any, int]] = []

        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue

            tokens = tokenizer.encode(text)
            if eos_id is not None and (not tokens or tokens[-1] != eos_id):
                tokens.append(eos_id)
            if len(tokens) < 2:
                continue

            answer_start_char = sample.get("answer_start")
            if answer_start_char is not None and isinstance(answer_start_char, int):
                # Tokenize prefix to find answer token boundary
                prefix_tokens = tokenizer.encode(text[:answer_start_char])
                answer_token_idx = len(prefix_tokens)
            else:
                # No answer_start — full-sequence CE (e.g. retention samples)
                answer_token_idx = 0

            # Clamp to valid range
            answer_token_idx = min(answer_token_idx, len(tokens))

            mask = [0.0] * answer_token_idx + [1.0] * (len(tokens) - answer_token_idx)

            # Exclude EOS from the answer mask.  The base model already
            # has EOS behaviour from pre-training; training CE on EOS in
            # every answer span produces an outsized gradient that biases
            # the adapter toward premature termination (EOS p ≈ 0.65 after
            # 1-2 answer tokens instead of baseline p ≈ 5e-6).
            if eos_id is not None and tokens[-1] == eos_id:
                mask[-1] = 0.0

            dataset.append((
                mx.array(tokens, dtype=mx.int32),
                mx.array(mask, dtype=mx.float32),
                0,
            ))

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
        rank_overrides: dict[str, int] | None = None,
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

            rank = rank_overrides[key] if rank_overrides and key in rank_overrides else geom.tail_dims
            # Validate: rank must be in [1, tail_dims]
            if rank <= 0:
                logger.warning("Skipping %s: rank_override=%d is non-positive", key, rank)
                continue
            if rank > geom.tail_dims:
                logger.warning(
                    "Clamping %s: rank_override=%d exceeds tail_dims=%d",
                    key, rank, geom.tail_dims,
                )
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

    def _evaluate_masked_loss(
        self,
        model,
        masked_dataset: list[tuple[Any, Any, int]],
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> float:
        """Compute average answer-masked loss over a dataset."""
        total_loss = 0.0
        total_answer_tokens = 0.0
        n_evaluated = 0

        for inputs, targets, masks in iterate_masked_batches(
            masked_dataset, batch_size, seq_length, train=False, loop=False,
        ):
            logits = model(inputs)
            logits = logits.astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, targets, reduction="none")
            masked_ce = ce * masks
            ntoks = masks.sum()
            batch_loss = masked_ce.sum()
            mx.eval(batch_loss, ntoks)
            total_loss += float(batch_loss)
            total_answer_tokens += float(ntoks)
            n_evaluated += 1
            if n_evaluated >= n_batches:
                break

        if total_answer_tokens == 0:
            return float("inf")

        return total_loss / total_answer_tokens

    def measure_baseline_constraints(
        self,
        model,
        tokenizer,
        paired_dataset: list[dict[str, Any]],
        logic_groups: dict[str, list[int]],
        template_groups: dict[str, list[int]],
        target_layers: list[int],
        max_seq_length: int = 256,
    ) -> tuple[list[float], list[float], dict[int, float], dict[int, float]]:
        """Measure baseline invariance/separation distances and spectral entropy.

        Runs on the BASE model (before NB-LoRA injection) to derive constraint
        thresholds. All thresholds come from geometry, not heuristics.

        Returns:
            (inv_distances, sep_distances, layer_entropies, layer_entropy_stds) where:
            - inv_distances: L2 distances between invariance pairs (same logic)
            - sep_distances: L2 distances between counterfactual pairs (same template)
            - layer_entropies: mean effective rank per target layer
            - layer_entropy_stds: per-layer effective-rank spread across samples
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
        # We compute per-sample effective rank to estimate local layer variability
        # (mean + std), then derive layer-specific targets from that variability.
        layer_entropies: dict[int, float] = {}
        layer_entropy_stds: dict[int, float] = {}
        for layer_idx in target_layers:
            erank_vals: list[float] = []
            for hs in hidden_states_full:
                if layer_idx in hs:
                    flat = hs[layer_idx].astype(mx.float32)
                    # Gram matrix G = X^T X
                    G = flat.T @ flat
                    trace_G = float(mx.sum(mx.diag(G)).item())
                    frobenius_sq = float(mx.sum(G * G).item())
                    erank = (trace_G * trace_G) / (frobenius_sq + 1e-10)
                    erank_vals.append(erank)
            if not erank_vals:
                continue
            mean_erank = sum(erank_vals) / len(erank_vals)
            if len(erank_vals) >= 2:
                variance = sum((v - mean_erank) ** 2 for v in erank_vals) / (len(erank_vals) - 1)
                std_erank = math.sqrt(variance)
            else:
                std_erank = 0.0
            layer_entropies[layer_idx] = mean_erank
            layer_entropy_stds[layer_idx] = std_erank

        logger.info(
            "Baseline constraints: %d inv_distances (mean=%.4f), "
            "%d sep_distances (mean=%.4f), %d layer_entropies "
            "(mean erank=%.4f, mean std=%.4f)",
            len(inv_distances),
            sum(inv_distances) / max(1, len(inv_distances)),
            len(sep_distances),
            sum(sep_distances) / max(1, len(sep_distances)),
            len(layer_entropies),
            sum(layer_entropies.values()) / max(1, len(layer_entropies)),
            sum(layer_entropy_stds.values()) / max(1, len(layer_entropy_stds)),
        )

        return inv_distances, sep_distances, layer_entropies, layer_entropy_stds

    def compute_mean_gradient(
        self,
        model,
        tokenizer,
        samples: list[dict],
        n_samples: int | None = None,
    ) -> "Array":
        """Compute mean gradient direction over samples. Returns float32 MLX array.

        Used for format bias decomposition: μ = (1/N) Σ ∇L(x_i).
        Only includes LoRA parameter gradients (A_tilde, B_tilde, lora_a, lora_b).
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss

        loss_vg = nn.value_and_grad(model, default_loss)

        if n_samples is not None:
            samples = samples[:n_samples]

        # Tokenize
        dataset = []
        for s in samples:
            text = s.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            dataset.append(mx.array(tokens, dtype=mx.int32))

        sum_g = None
        count = 0

        for tokens in dataset:
            batch = tokens.reshape(1, -1)
            lengths = mx.array([[0, batch.shape[1]]])
            (loss, ntoks), grad = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat = []
            for name, arr in mlx_flatten(grad):
                if any(k in name for k in ('A_tilde', 'B_tilde', 'lora_a', 'lora_b')):
                    flat.append(arr.reshape(-1).astype(mx.float32))
            if flat:
                g = mx.concatenate(flat)
                mx.eval(g)
                if sum_g is None:
                    sum_g = g
                else:
                    sum_g += g
                count += 1

        if count == 0:
            raise RuntimeError("No valid gradients computed for format bias")
        return (sum_g / count).astype(mx.float32)

    def build_projection_hook(self, v_format: "Array"):
        """Build a gradient hook that projects out the format bias direction.

        Args:
            v_format: [d] float32 abstract Array (MLX array) — unit format bias direction

        Returns:
            Callable that takes a gradient pytree and returns a decontaminated pytree.
        """
        from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

        mx.eval(v_format)

        def hook(grad):
            flat = dict(mlx_flatten(grad))
            pieces = []
            lora_keys = []
            for key in flat:
                if any(k in key for k in ('A_tilde', 'B_tilde', 'lora_a', 'lora_b')):
                    lora_keys.append(key)
                    pieces.append(flat[key].reshape(-1).astype(mx.float32))
            if not pieces:
                return grad

            g_vec = mx.concatenate(pieces)
            mx.eval(g_vec)

            # Project out: g_clean = g - (v^T g) v
            coeff = mx.sum(v_format * g_vec)
            g_clean = g_vec - coeff * v_format
            mx.eval(g_clean)

            # Unflatten back
            offset = 0
            for key in lora_keys:
                size = flat[key].size
                shape = flat[key].shape
                flat[key] = g_clean[offset:offset + size].reshape(shape)
                offset += size

            return tree_unflatten(flat)

        return hook

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
    ) -> tuple[list[tuple[int, float, float]], str, list[EpochMetrics]]:
        """Train with ScaledGD, Weyl adapter-saturation monitoring, and geometric stopping.

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
            # Target: layers in the middle 60% of the network (skip early embedding
            # layers and final output-formatting layers).
            start = max(1, n_layers // 5)       # skip first 20%
            end = max(start + 2, 4 * n_layers // 5)  # up to 80%
            reshape_target_layers = list(range(start, end))
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
            # For Lipschitz, use default_loss for cleaner estimate
            lipschitz_loss_fn = default_loss
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
            # For Lipschitz, use default_loss (without constraints) for cleaner estimate
            lipschitz_loss_fn = default_loss
        elif answer_masked_dataset is not None:
            # Answer-span masking: CE only on answer tokens + EOS
            if entropy_regularization:
                # Combined: answer-masked CE + entropy floor on all tokens
                _EPS_F32 = float(mx.finfo(mx.float32).eps)
                baseline_ent = measure_baseline_entropy(
                    model, train_dataset, batch_size, seq_length,
                    n_batches=eval_batches,
                )
                if baseline_ent is not None and baseline_ent > 0:
                    ent_floor = baseline_ent * (1.0 - _EPS_F32 ** 0.5)
                    am_loss_fn = make_entropy_regularized_answer_masked_loss(ent_floor)
                    logger.info(
                        "Answer-masked CE + entropy reg: baseline=%.4f, floor=%.4f",
                        baseline_ent, ent_floor,
                    )
                else:
                    logger.warning(
                        "Could not measure baseline entropy, falling back to plain answer-mask",
                    )
                    def am_loss_fn(model, inputs, targets, masks):
                        logits = model(inputs)
                        logits = logits.astype(mx.float32)
                        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
                        masked_ce = ce * masks
                        ntoks = masks.sum()
                        return masked_ce.sum() / mx.maximum(ntoks, mx.array(1.0)), ntoks
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
            lipschitz_loss_fn = default_loss
            use_answer_mask = True
        else:
            if entropy_regularization:
                # Measure baseline entropy to derive the floor
                _EPS_F32 = float(mx.finfo(mx.float32).eps)
                baseline_ent = measure_baseline_entropy(
                    model, train_dataset, batch_size, seq_length,
                    n_batches=eval_batches,
                )
                if baseline_ent is not None and baseline_ent > 0:
                    ent_floor = baseline_ent * (1.0 - _EPS_F32 ** 0.5)
                    loss_fn = make_entropy_regularized_loss(ent_floor)
                    logger.info(
                        "Entropy regularization: baseline=%.4f, floor=%.4f",
                        baseline_ent, ent_floor,
                    )
                else:
                    logger.warning(
                        "Could not measure baseline entropy, falling back to base_ce_loss",
                    )
                    loss_fn = base_ce_loss
            else:
                loss_fn = base_ce_loss
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
        # We precondition by the dominant left metric factor P_left ≈ M M^T where
        # M = I + Z, correcting for the Cayley transform's curvature in the rank-r
        # rotation subspace. This is a one-sided approximation in free
        # (A_tilde, B_tilde) coordinates (the exact pullback is a block operator).
        # Refs: Amari (1998), Wen & Yin (2013), Li et al. (ICLR 2020).
        #
        # Cayley-Riemannian natural gradient with preconditioner-aware step
        # bound: η ≤ 2/(L * λ_max(P)). Left-factor anisotropy preserved. The caller
        # enforces the stability invariant m = η * L * λ_max(P) ≤ 2 per step.
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

            # Optional gradient hook (e.g. format bias projection)
            if gradient_hook is not None:
                grad = gradient_hook(grad)

            # Save actual update direction for stopping certificate
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
                        # Scale LR by inverse of loss increase ratio.
                        # Floor: sqrt(eps_f32) — below this, the multiplicative
                        # update is indistinguishable from zero in float32.
                        _BACKOFF_FLOOR = math.ldexp(1.0, -23) ** 0.5  # sqrt(eps_f32)
                        backoff = val_losses[-2] / val_losses[-1]
                        current_eta *= max(backoff, _BACKOFF_FLOOR)
                        logger.info(
                            "Val loss increased (%.4f → %.4f): LR backoff=%.3f to %.2e",
                            val_losses[-2], val_losses[-1], backoff, current_eta,
                        )

                    if lr_monotonic:
                        current_eta = min(eta_spectral, current_eta)
                    else:
                        current_eta = min(eta_spectral, eta_ceiling)

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
                    # Fallback: simple verify_bounds
                    try:
                        _, max_ratio, _ = self.verify_bounds(model)
                    except Exception:
                        pass

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
                        (c.tokens, c.advantage)
                        for c in outcome_result.completions
                        if c.advantage != 0.0
                    ]

                    n_outcome_steps = 0
                    if active_completions:
                        outcome_batches = prepare_outcome_batches(
                            active_completions, batch_size, seq_length,
                        )
                        outcome_loss_fn = make_outcome_loss()
                        outcome_vg = nn.value_and_grad(model, outcome_loss_fn)

                        # Calibrate REINFORCE step size to match CE step magnitude.
                        # CE moved ‖Δθ‖ = update_norm over check_interval steps.
                        # Each REINFORCE step should move ≤ CE average per step.
                        # Fallback uses machine-precision relative displacement:
                        # sqrt(eps) * ||θ|| / steps (dtype + measured parameter norm).
                        ce_steps_done = max(1, check_interval)
                        if update_norm is not None and update_norm > 0:
                            target_step_norm = update_norm / ce_steps_done
                            target_step_source = "ce_update_norm"
                        else:
                            sqrt_eps = math.sqrt(float(mx.finfo(mx.float32).eps))
                            trainable_params = dict(mlx_flatten(model.trainable_parameters()))
                            theta_sq = mx.array(0.0, dtype=mx.float32)
                            for p in trainable_params.values():
                                if p.size > 0:
                                    theta_sq = theta_sq + mx.sum(p * p)
                            mx.eval(theta_sq)
                            theta_norm = float(mx.sqrt(theta_sq).item())
                            if theta_norm > 0:
                                target_step_norm = (sqrt_eps * theta_norm) / ce_steps_done
                            else:
                                target_step_norm = sqrt_eps / ce_steps_done
                            target_step_source = "sqrt_eps_param_norm"

                        from mlx.utils import tree_flatten as _rf_flatten

                        for ob_batch, ob_lengths, ob_advantages in outcome_batches:
                            (o_loss, o_ntoks), o_grad = outcome_vg(
                                model, ob_batch, ob_lengths, ob_advantages,
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
                    lipschitz_L=measured_L,
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
                    precond_m_invariant=precond_metrics.get("m_invariant"),
                    precond_eta_step=precond_metrics.get("eta_step"),
                    online_eval_accuracy=online_eval_acc,
                    online_eval_n_correct=online_eval_n_correct,
                    online_eval_n_total=online_eval_n_total,
                    online_eval_degraded=online_eval_degraded,
                    outcome_n_problems=outcome_n_problems_epoch,
                    outcome_n_active=outcome_n_active_epoch,
                    outcome_signal_density=outcome_signal_density_epoch,
                    outcome_n_steps=outcome_n_steps_epoch,
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
                if measured_L is not None:
                    log_parts.append(f"L={measured_L:.4f}")
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
                    mi = precond_metrics.get("m_invariant", 0)
                    es = precond_metrics.get("eta_step", 0)
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
                    log_parts.append(f"m={mi:.3f}")
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
            numeric_floor = math.sqrt(math.ldexp(1.0, -23))
            if last_val - best_val_loss > numeric_floor:
                logger.info(
                    "Restoring best checkpoint (val_loss %.4f vs final %.4f)",
                    best_val_loss, last_val,
                )
                # Restore only trainables; avoids missing-parameter failures on
                # hybrid architectures where load_weights expects full tensors.
                model.update(mlx_unflatten(best_weights))
                mx.eval(model.parameters())

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
        step units. The caller enforces the stability bound with λ_max(P_hat)=1:

            η ≤ 2 / L

        where L is the measured Lipschitz constant. This preserves anisotropy
        in the approximated rank-r left factor and redistributes gradient
        across eigenspaces according to Cayley-induced curvature.

        The invariant m = η * L * λ_max(P_hat) ≤ 2 must hold at every step.
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
        seed: int = 0,
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
            # Data-derived bootstrap parameters (G1 compliance):
            # n_bootstrap = n^2 — quadratic in sample count ensures
            # bootstrap SE ∝ 1/n, matching the CLT convergence rate.
            # confidence = 1 - 1/n_bootstrap — tail probability scales
            # inversely with resample count (tighter CI with more data).
            n = len(per_batch_losses)
            _n_bootstrap = n * n
            _confidence = 1.0 - 1.0 / _n_bootstrap
            lower, upper = bootstrap_ci(
                per_batch_losses,
                confidence=_confidence,
                n_bootstrap=_n_bootstrap,
                seed=seed,
            )
            ci_half = (upper - lower) / 2.0
            # Keep CI distinguishable from numerical noise. A zero-width CI with
            # real validation data makes the improvement test vacuous.
            numeric_floor = math.sqrt(math.ldexp(1.0, -23))
            val_ci_half_width = max(ci_half, numeric_floor)

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
            # Random direction, normalized.
            # Use a stable (process-independent) seed from parameter keys.
            key_material = "|".join(sorted(params.keys())).encode("utf-8")
            digest = hashlib.sha256(key_material).digest()
            stable_seed = int.from_bytes(digest[:4], "little")
            mx.random.seed(stable_seed)
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

    def _compute_rss_metrics(
        self,
        model,
        tokenizer,
        base_activations: dict[int, list],
        eval_samples: list | None,
        n_probes: int = 20,
    ):
        """Compute outer similarity between base and adapted model representations.

        Uses anchor-relative representations at the middle layer.
        Returns an OuterSimilarityResult or None on failure.

        Reference: Kucukahmetler et al. (2026), TMLR.

        Note: RSS alignment != accuracy. These metrics track geometric drift
        of the adapted model relative to the base, not task performance.
        See online_eval for correctness measurement (Kucukahmetler et al. 2026).
        """
        try:
            from modelcypher.core.domain.geometry.relative_representation import (
                compute_anchor_embeddings,
                compute_outer_similarity,
                compute_relative_representation,
            )

            if not eval_samples or not base_activations:
                return None

            # Pick middle layer
            layer_indices = sorted(base_activations.keys())
            if not layer_indices:
                return None
            mid_layer = layer_indices[len(layer_indices) // 2]

            base_list = base_activations.get(mid_layer, [])
            if len(base_list) < 2:
                return None

            # Get anchor embeddings from the (frozen) embedding matrix
            embed_matrix = None
            base_model = model
            # Navigate model structure to find embedding weights
            for attr in ("base", "model"):
                if hasattr(base_model, attr):
                    base_model = getattr(base_model, attr)
            if hasattr(base_model, "embed_tokens"):
                embed_matrix = base_model.embed_tokens.weight
            if embed_matrix is None:
                return None

            anchors, anchor_ids = compute_anchor_embeddings(
                embed_matrix, tokenizer,
            )
            if len(anchor_ids) < 3:
                return None

            base_stack = mx.stack(base_list[:n_probes])
            mx.eval(base_stack)

            # Collect adapted model activations for same probes
            probe_texts = [s["text"][:200] for s in eval_samples[:n_probes]]
            adapted_list = []
            for text in probe_texts[:len(base_list)]:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                if mid_layer in acts:
                    act = acts[mid_layer]
                    shape = act.shape
                    if len(shape) == 3:
                        pooled = mx.mean(act, axis=1)
                        pooled = mx.reshape(pooled, (-1,))
                    else:
                        pooled = mx.reshape(act, (-1,))
                    mx.eval(pooled)
                    adapted_list.append(pooled)

            if len(adapted_list) != len(base_list[:n_probes]):
                return None

            adapted_stack = mx.stack(adapted_list)
            mx.eval(adapted_stack)

            # Compute relative representations and outer similarity
            rel_base = compute_relative_representation(base_stack, anchors)
            rel_adapted = compute_relative_representation(adapted_stack, anchors)
            return compute_outer_similarity(rel_base, rel_adapted)
        except Exception:
            logger.debug("RSS metric computation failed", exc_info=True)
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

        Uses two streaming passes over per-sample gradients (batch_size=1):
        1) mean gradient + mean norm
        2) variance around the mean gradient

        This avoids storing all per-sample gradient trees in memory.
        Same math as event-buffer path in lora_memory_store.derive_critical_batch_size().
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_vg = nn.value_and_grad(model, default_loss)

        def _norm_sq(flat_grads: dict[str, Any]) -> float:
            total = 0.0
            for grad in flat_grads.values():
                g = grad.astype(mx.float32)
                total += float(mx.sum(g * g))
            return total

        count = 0
        total_norm_sum = 0.0
        mean_grad: dict[str, Any] = {}

        # Pass 1: accumulate mean gradient and mean norm.
        for batch, lengths in iterate_batches(
            train_dataset, 1, seq_length, loop=False, seed=0,
        ):
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat_grads = dict(mlx_flatten(grads))
            if flat_grads:
                mx.eval(*flat_grads.values())

            norm_sq = _norm_sq(flat_grads)
            total_norm_sum += math.sqrt(norm_sq)

            if count == 0:
                mean_grad = {k: v.astype(mx.float32) for k, v in flat_grads.items()}
            else:
                for k, v in flat_grads.items():
                    if k in mean_grad:
                        mean_grad[k] = mean_grad[k] + v.astype(mx.float32)
                    else:
                        mean_grad[k] = v.astype(mx.float32)

            count += 1
            if count >= n_samples:
                break

        if count < 2:
            logger.info("Too few samples for B_crit estimation, defaulting to 1")
            return 1

        inv_count = 1.0 / float(count)
        for key, grad_sum in list(mean_grad.items()):
            mean_grad[key] = grad_sum * inv_count
        if mean_grad:
            mx.eval(*mean_grad.values())

        mean_norm = total_norm_sum / float(count)

        # Pass 2: accumulate variance around mean gradient.
        variance_sum = 0.0
        second_count = 0
        for batch, lengths in iterate_batches(
            train_dataset, 1, seq_length, loop=False, seed=0,
        ):
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat_grads = dict(mlx_flatten(grads))
            if flat_grads:
                mx.eval(*flat_grads.values())

            diff_sq = 0.0
            for key, grad in flat_grads.items():
                if key not in mean_grad:
                    continue
                diff = grad.astype(mx.float32) - mean_grad[key]
                diff_sq += float(mx.sum(diff * diff))
            variance_sum += diff_sq

            second_count += 1
            if second_count >= count:
                break

        variance = variance_sum / float(count - 1)
        mean_grad_norm_sq = _norm_sq(mean_grad)
        eps = math.ldexp(1.0, -23)  # IEEE 754 float32 epsilon
        snr = mean_grad_norm_sq / (variance + eps)

        if not math.isfinite(snr) or snr <= 0:
            logger.info("Could not estimate gradient noise, defaulting to 1")
            return 1

        b_crit = max(1, math.ceil(1.0 / snr))
        b_crit = min(b_crit, len(train_dataset))

        logger.info(
            "B_crit = %d (SNR=%.6f, variance=%.6f, %d samples)",
            b_crit, snr, variance, count,
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
