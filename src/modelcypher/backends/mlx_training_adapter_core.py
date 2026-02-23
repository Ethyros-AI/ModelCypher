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
MASS (Measured-Adaptive Step Size): spectral ceiling (Weyl 1912) + per-step SPS
(Loizou et al. 2020) + per-step Weyl displacement bound. Every number from SVD,
IEEE 754, or per-step measurement.

The Cayley transform maps unconstrained (A_tilde, B_tilde) to semi-orthogonal
(A, B), guaranteeing ||2 * B^T @ S @ A||_2 <= 2 * max(S) <= sigma_k.
"""

from __future__ import annotations

import json
import logging
import math
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
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.spectral_budget import (
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    compute_initialization_vectors,
    compute_projected_residuals,
    is_budget_exhausted,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_lora import LayerGeometry
    from modelcypher.core.domain.training.geometric_optimizer import OptimizerGeometryConfig
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

# IEEE 754 float32 derived stability constants — the ONLY epsilons allowed.
_EPS_F32 = float(mx.finfo(mx.float32).eps)       # ~1.19e-7
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)               # ~3.45e-4


@dataclass
class EpochMetrics:
    """Per-epoch diagnostic metrics for mechanism analysis."""

    epoch: int
    train_loss: float
    val_loss: float | None
    eta: float
    update_norm: float | None
    max_spectral_ratio: float | None
    mean_token_entropy: float | None
    repetition_rate: float | None
    elapsed_seconds: float
    spectral_ratio_growth_per_iter: float | None = None
    eta_ceiling: float | None = None
    adapter_saturation_median_ratio: float | None = None
    # Cayley-Stiefel preconditioner diagnostics
    precond_lambda_max: float | None = None
    precond_lambda_max_raw: float | None = None
    precond_cond_max: float | None = None
    precond_ipz_kappa_upper_max: float | None = None
    precond_ipz_rel_error_upper_max: float | None = None
    precond_ipz_warn_fraction: float | None = None
    precond_gain_mean: float | None = None
    precond_eta_step: float | None = None     # actual per-step η
    # MASS (Measured-Adaptive Step Size) diagnostics
    displacement: float | None = None  # eta_step * ||d||
    eta_sps: float | None = None       # Stochastic Polyak step-size (Loizou et al. 2020)
    eta_weyl: float | None = None      # Per-step Weyl displacement bound
    d_norm: float | None = None        # Preconditioned gradient direction norm
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
    outcome_target_step_norm: float | None = None
    outcome_target_step_source: str | None = None
    outcome_o_eta: float | None = None
    outcome_o_grad_norm: float | None = None
    outcome_budget_remaining: float | None = None
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
    ):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._rank = rank
        self._scale_bound = scale_bound

        # Unconstrained free parameters. Start at exact zero-perturbation:
        # Cayley then yields zero effective delta, preserving base behavior
        # before any data-driven updates.
        self.A_tilde = mx.zeros((rank, in_features))
        self.B_tilde = mx.zeros((rank, out_features))
        # S_raw clamped to [0, scale_bound] at every forward and after every step.
        # Initialized at geometric midpoint of [0, σ_k]: minimizes max distance
        # to either Cayley endpoint (minimax point of the interval).
        self.S_raw = mx.ones((rank,)) * (scale_bound / 2.0)

    @classmethod
    def from_base(
        cls,
        linear,
        rank: int,
        scale_bound: float,
    ) -> "NBLoRALinear":
        """Create from existing nn.Linear or nn.QuantizedLinear."""
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits

        obj = cls(input_dims, output_dims, rank, scale_bound)
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
            erank = (trace_G * trace_G) / (frob_sq + _EPS_F32)
            # -log(erank/d): penalizes low effective rank
            expand_loss = expand_loss + (-mx.log(erank / d + _EPS_F32))
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
                h_norms = mx.sqrt(mx.sum(h_mean * h_mean, axis=-1, keepdims=True) + _EPS_F32)
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
            init_values["ce"] = mx.stop_gradient(ce_loss) + _EPS_F32
            init_values["expand"] = mx.stop_gradient(expand_loss) + _EPS_F32
            init_values["contrast"] = (
                mx.stop_gradient(contrast_loss) + _EPS_F32
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
    w_expand = ce_gnorm / max(expand_gnorm, _EPS_F32)
    w_contrast = ce_gnorm / max(contrast_gnorm, _EPS_F32)

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
        "ratio_expand": ce_gnorm / max(expand_gnorm, _EPS_F32),
        "ratio_contrast": ce_gnorm / max(contrast_gnorm, _EPS_F32),
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
    """Create a REINFORCE outcome loss function with response-only masking.

    Computes teacher-forced log-probabilities weighted by per-completion
    advantages, masked to response tokens only:

        L_outcome = -mean(A_i * (1/T_i) Σ_{t∈response} log π(y_{i,t} | y_{i,<t}))

    Williams (1992) REINFORCE: ∇_θ log π(action|state) × A.
    The prompt is the STATE (externally given context).
    The response is the ACTION (policy's chosen output).
    Only response tokens contribute to the policy gradient.

    Length normalization: dividing by response token count T_i makes the
    loss the average per-token log-prob, independent of response length.
    Without this, longer responses dominate the gradient regardless of
    correctness (scale bias).

    The returned function has signature:
        loss_fn(model, batch, lengths, advantages, response_starts) -> (loss, ntoks)

    Where:
        batch: [B, S] token IDs (padded)
        lengths: [B] actual sequence lengths (for masking)
        advantages: [B] per-completion advantage values (constants, not differentiated)
        response_starts: [B] token index where response begins in each sequence

    Reference: Williams (1992), "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning", Machine Learning 8(3-4)
    """

    def _outcome_loss(model, batch, lengths, advantages, response_starts):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # Length mask: valid tokens only
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = (steps[None] < lengths[:, None]).astype(mx.float32)

        # Response-only mask: exclude prompt tokens from policy gradient.
        # Williams (1992): gradient is over actions (response), not states (prompt).
        # After teacher-forcing shift, position t in targets predicts token t+1
        # in the original sequence. Response tokens start at response_start in
        # the original sequence, mapping to index (response_start - 1) in the
        # shifted targets array.
        steps_0 = mx.arange(targets.shape[1])[None, :]             # [1, S-1]
        rs_shifted = mx.maximum(response_starts[:, None] - 1, 0)   # [B, 1]
        response_mask = (steps_0 >= rs_shifted).astype(mx.float32)  # [B, S-1]

        # Combined mask: response tokens within valid length
        mask = response_mask * length_mask  # [B, S-1]

        # Per-token log probs (teacher-forced)
        # nn.losses.cross_entropy returns -log P(target), always positive.
        # Negate to get log P(target), always negative.
        ce_per_token = nn.losses.cross_entropy(logits, targets)  # [B, S-1]
        log_probs = -ce_per_token  # [B, S-1]

        # Response-only, length-normalized log probs
        response_lengths = mx.maximum(mask.sum(axis=-1), 1.0)  # [B]
        seq_log_probs = (log_probs * mask).sum(axis=-1) / response_lengths  # [B]

        # REINFORCE: L = -mean(A_i * avg_response_log_prob_i)
        # A > 0 (correct): seq_log_prob < 0, product < 0, -product > 0
        #   → minimizing loss increases log prob of correct completions
        # A < 0 (incorrect): seq_log_prob < 0, product > 0, -product < 0
        #   → minimizing loss decreases log prob of incorrect completions
        weighted = advantages * seq_log_probs  # [B]
        loss = -weighted.mean()

        ntoks = mask.sum()
        return loss, ntoks

    return _outcome_loss


def make_outcome_loss_with_kl(ref_log_probs_dict: dict, beta: float):
    """Create a REINFORCE outcome loss with KL penalty from frozen reference.

    L = L_reinforce + beta * KL(current || reference)

    Both REINFORCE and KL terms are masked to response tokens only
    (Williams 1992) and length-normalized per sequence.

    Parameters
    ----------
    ref_log_probs_dict : dict
        Maps batch index to [B, S] reference log-prob arrays (frozen, no grad).
    beta : float
        KL penalty coefficient. Derived geometrically: |E[L_reinforce]| / E[KL]
        at initialization so both terms start equally weighted.
    """

    def _outcome_loss_kl(model, batch, lengths, advantages, response_starts, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        logits = logits.astype(mx.float32)

        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = (steps[None] < lengths[:, None]).astype(mx.float32)

        # Response-only mask (Williams 1992)
        steps_0 = mx.arange(targets.shape[1])[None, :]
        rs_shifted = mx.maximum(response_starts[:, None] - 1, 0)
        response_mask = (steps_0 >= rs_shifted).astype(mx.float32)
        mask = response_mask * length_mask

        ce_per_token = nn.losses.cross_entropy(logits, targets)
        log_probs = -ce_per_token

        # Response-only, length-normalized REINFORCE
        response_lengths = mx.maximum(mask.sum(axis=-1), 1.0)
        seq_log_probs = (log_probs * mask).sum(axis=-1) / response_lengths
        weighted = advantages * seq_log_probs
        reinforce_loss = -weighted.mean()

        # KL penalty: response-only divergence from reference
        ref_lp = ref_log_probs_dict[batch_idx]
        kl_per_token = (log_probs - ref_lp) * mask
        kl_penalty = (kl_per_token.sum(axis=-1) / response_lengths).mean()

        loss = reinforce_loss + beta * kl_penalty
        ntoks = mask.sum()
        return loss, ntoks

    return _outcome_loss_kl


def prepare_outcome_batches(completions, batch_size, seq_length):
    """Convert (tokens, advantage, response_start) triples into padded MLX batches.

    Parameters
    ----------
    completions : list[tuple[list[int], float, int]]
        (token_ids, advantage, response_start) triples. Only nonzero-advantage completions.
    batch_size : int
        Batch size for REINFORCE gradient steps.
    seq_length : int
        Maximum sequence length (truncate + pad).

    Returns
    -------
    list[tuple[mx.array, mx.array, mx.array, mx.array]]
        (batch [B, S], lengths [B], advantages [B], response_starts [B]) tuples.
    """
    if not completions:
        return []

    batches = []
    for i in range(0, len(completions), batch_size):
        group = completions[i:i + batch_size]
        tokens_list = []
        advs = []
        lens = []
        rs_list = []
        for tokens, advantage, response_start in group:
            t = tokens[:seq_length]
            lens.append(len(t))
            # Pad to seq_length with zeros
            padded = t + [0] * (seq_length - len(t))
            tokens_list.append(padded)
            advs.append(advantage)
            # Clamp response_start to truncated length (not seq_length - 1).
            # When response_start >= len(t), the entire window is prompt-only.
            # Setting rs = len(t) produces an all-zero response mask in the
            # loss function → zero gradient contribution (correct by Williams 1992).
            rs_list.append(min(response_start, len(t)))

        batch = mx.array(tokens_list)
        lengths = mx.array(lens)
        advantages = mx.array(advs)
        response_starts = mx.array(rs_list)
        batches.append((batch, lengths, advantages, response_starts))

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
                    dist = mx.sqrt(mx.sum(diff * diff) + _EPS_F32)
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
                    dist = mx.sqrt(mx.sum(diff * diff) + _EPS_F32)
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
            current_erank = (trace_G * trace_G) / (frobenius_sq + _EPS_F32)

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


