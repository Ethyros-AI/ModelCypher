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

"""
Stage 2: PERMUTE - Permutation alignment for MLP neurons.

Uses PermutationAligner to solve the permutation symmetry problem.
Neural networks have N! permutation symmetries per MLP layer.
We find P, S such that W_aligned = S @ P @ W @ P^T @ S^T

Reference: Ainsworth et al. (2022) "Git Re-Basin"

This stage runs BEFORE transplant to reduce the delta magnitude between
source and target weights. By aligning neuron orderings first, the
null-space projection in transplant has less work to do.

Pipeline: VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import power_iteration_eigh
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Permutation alignment is ALWAYS enabled.
# MLP layers have N! permutation symmetries that must be resolved.
# Skipping permutation alignment increases merge error - there is no valid reason to disable it.


@dataclass
class PermuteResult:
    """Result of Stage 2 permutation."""

    weights: dict[str, Any]  # Array (backend-agnostic)
    metrics: dict[str, Any]


def stage_permute(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    infer_hidden_dim_fn: Callable[[dict[str, Any]], int],
    backend: "Backend | None" = None,
) -> PermuteResult:
    """
    Stage 2: PURE GEOMETRIC PERMUTATION ALIGNMENT.

    MLP layers have N! permutation symmetries. Two networks with identical
    function can have completely different weight orderings. This stage
    solves for the optimal permutation P that minimizes:

        ||W_target - P @ W_source||_F

    via the Hungarian algorithm (linear assignment problem).

    No configuration - permutation alignment is always run.
    No arbitrary thresholds - the optimal permutation is computed exactly.
    """
    b = backend or get_default_backend()

    from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
    from modelcypher.core.domain.geometry.numerical_stability import (
        regularization_epsilon,
    )
    from modelcypher.core.domain.geometry.permutation_aligner import (
        PermutationAligner,
    )

    # Convert weights to backend arrays (with dequantization for quantized models)
    source_arr: dict[str, "Array"] = {}
    target_arr: dict[str, "Array"] = {}

    for key, val in source_weights.items():
        # Skip quantization metadata keys
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        # Dequantize if quantized, then convert to float32
        dequant = dequantize_if_needed(val, key, source_weights, b)
        arr = b.astype(b.array(dequant), "float32")
        b.eval(arr)
        source_arr[key] = arr
    for key, val in target_weights.items():
        # Skip quantization metadata keys
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        # Dequantize if quantized, then convert to float32
        dequant = dequantize_if_needed(val, key, target_weights, b)
        arr = b.astype(b.array(dequant), "float32")
        b.eval(arr)
        target_arr[key] = arr

    # Build anchor embeddings from BOTH models' embedding layers
    # Each model needs its own anchors to compute meaningful signatures
    # Must find .weight specifically, not .scales or .biases
    source_anchor_key = None
    target_anchor_key = None

    for key in source_weights:
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if "embed_tokens" in key or "wte" in key or "embedding" in key.lower():
            source_anchor_key = key
            break

    for key in target_weights:
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if "embed_tokens" in key or "wte" in key or "embedding" in key.lower():
            target_anchor_key = key
            break

    source_anchors = None
    target_anchors = None
    num_anchors: int | None = None

    if source_anchor_key is not None:
        embed = source_weights[source_anchor_key]
        embed = dequantize_if_needed(embed, source_anchor_key, source_weights, b)
        num_anchors = min(embed.shape[0], embed.shape[1])
        source_anchors = b.astype(b.array(embed[:num_anchors]), "float32")
        b.eval(source_anchors)
        logger.info("PERMUTE: Source anchors from %s (%d tokens)", source_anchor_key, num_anchors)

    if target_anchor_key is not None:
        embed = target_weights[target_anchor_key]
        embed = dequantize_if_needed(embed, target_anchor_key, target_weights, b)
        target_count = min(embed.shape[0], embed.shape[1])
        num_anchors = target_count if num_anchors is None else min(num_anchors, target_count)
        target_anchors = b.astype(b.array(embed[:num_anchors]), "float32")
        b.eval(target_anchors)
        logger.info("PERMUTE: Target anchors from %s (%d tokens)", target_anchor_key, num_anchors)

    if source_anchors is None or target_anchors is None:
        raise RuntimeError(
            "PERMUTE: Embedding anchors missing; cannot reach exact kernel alignment for permutation."
        )

    if source_anchors.shape[1] != target_anchors.shape[1]:
        raise RuntimeError(
            "PERMUTE: Hidden dimension mismatch (source=%d, target=%d)."
            % (source_anchors.shape[1], target_anchors.shape[1])
        )

    # Dtype-derived precision threshold (sqrt(machine_epsilon))
    precision_tol = regularization_epsilon(b, source_anchors)
    # Use feature_bias_correction=True and .best to avoid false negatives from underestimation
    embed_cka = compute_cka(
        source_anchors,
        target_anchors,
        backend=b,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best
    if embed_cka < 1.0 - precision_tol:
        # Polar decomposition via GPU-only power iteration eigendecomposition (no CPU linear algebra).
        M = b.matmul(b.transpose(source_anchors), target_anchors)
        mtm = b.matmul(b.transpose(M), M)
        # Cast to float32 for numerical stability
        mtm_f32 = b.astype(mtm, "float32")
        b.eval(mtm_f32)

        n = int(mtm_f32.shape[0])
        eigvals, eigvecs = power_iteration_eigh(b, mtm_f32, k=n)
        b.eval(eigvals, eigvecs)

        # Use backend operations to find min/max eigenvalues (avoid CPU conversion)
        max_eig_arr = b.max(eigvals)
        min_eig_arr = b.min(eigvals)
        b.eval(max_eig_arr, min_eig_arr)
        max_eig = float(b.to_scalar(max_eig_arr))
        min_eig = float(b.to_scalar(min_eig_arr))
        threshold = max_eig * precision_tol
        if min_eig <= threshold:
            raise RuntimeError(
                "PERMUTE: Anchor covariance is rank-deficient; "
                "expand anchor selection before permutation."
            )

        inv_sqrt_vals = 1.0 / b.sqrt(eigvals)
        inv_sqrt = b.matmul(
            eigvecs * b.reshape(inv_sqrt_vals, (1, -1)),
            b.transpose(eigvecs),
        )
        embedding_rotation = b.matmul(M, inv_sqrt)
        b.eval(embedding_rotation)

        # Ensure det(R) = 1 to avoid reflection.
        det_R = b.det(embedding_rotation)
        b.eval(det_R)
        det_val = float(b.to_scalar(det_R))
        if det_val < 0:
            n = embedding_rotation.shape[1]
            rot_cols = [embedding_rotation[:, i : i + 1] for i in range(n - 1)]
            rot_cols.append(embedding_rotation[:, -1:] * -1.0)
            embedding_rotation = b.concatenate(rot_cols, axis=1)
            b.eval(embedding_rotation)

        # Apply rotation to all source weights that operate on hidden dimension.
        hidden_dim = source_anchors.shape[1]
        for key in list(source_arr.keys()):
            w = source_arr[key]
            if w.ndim != 2:
                continue
            out_dim, in_dim = w.shape
            if in_dim == hidden_dim:
                source_arr[key] = b.matmul(w, embedding_rotation)
                b.eval(source_arr[key])
            elif out_dim == hidden_dim:
                source_arr[key] = b.matmul(b.transpose(embedding_rotation), w)
                b.eval(source_arr[key])

        source_rotated = b.matmul(source_anchors, embedding_rotation)
        b.eval(source_rotated)
        # Use feature_bias_correction=True and .best to avoid false negatives from underestimation
        embed_cka = compute_cka(
            source_rotated,
            target_anchors,
            backend=b,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        ).best
        if embed_cka < 1.0 - precision_tol:
            raise RuntimeError(
                "PERMUTE: Embedding rotation failed to reach exact kernel alignment (CKA=%.8f)."
                % embed_cka
            )
        source_anchors = source_rotated
        logger.info(
            "PERMUTE: Embedding rotation exact kernel alignment achieved (CKA=%.8f).",
            embed_cka,
        )
    else:
        logger.info(
            "PERMUTE: Embedding anchors exact kernel alignment achieved (CKA=%.8f). "
            "Skipping rotation.",
            embed_cka,
        )

    # Run MLP re-basin alignment with separate source/target anchors
    # This is critical: each model needs its own embeddings to compute meaningful signatures
    aligned, mean_quality, blocks_aligned = PermutationAligner.rebasin_mlp_with_activations(
        source_arr,
        target_arr,
        source_anchors,
        target_anchors,
    )
    # Eval all aligned weights
    for val in aligned.values():
        b.eval(val)

    logger.info(
        "PERMUTE: Aligned %d MLP blocks, mean quality=%.3f",
        blocks_aligned,
        mean_quality,
    )

    metrics = {
        "layers_permuted": blocks_aligned,
        "mean_quality": float(mean_quality),
    }

    return PermuteResult(aligned, metrics)


def infer_hidden_dim(weights: dict[str, Any]) -> int:
    """Infer hidden dimension from weight shapes.

    Must find a valid projection layer - no arbitrary defaults.
    """
    # Prefer norm weights: 1D and directly encode hidden size even for quantized models.
    for key, val in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if not hasattr(val, "shape") or len(val.shape) != 1:
            continue
        if key.endswith(("norm.weight", "layernorm.weight", "rms_norm.weight")):
            return int(val.shape[0])

    # Fall back to projection matrices (ignore quantization metadata like *.scales).
    for key, val in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if not hasattr(val, "shape") or len(val.shape) != 2:
            continue
        if not key.endswith(".weight"):
            continue
        if "q_proj" in key or "o_proj" in key:
            return int(max(val.shape))
        if "k_proj" in key or "v_proj" in key:
            return int(max(val.shape))
        if "up_proj" in key or "gate_proj" in key or "down_proj" in key:
            return int(min(val.shape))
    raise ValueError(
        "Cannot infer hidden dimension: no q_proj, k_proj, up_proj, or gate_proj "
        "found in weights. Available keys: " + ", ".join(list(weights.keys())[:10])
    )
