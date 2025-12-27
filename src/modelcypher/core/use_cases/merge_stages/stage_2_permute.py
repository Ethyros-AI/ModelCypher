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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class PermuteConfig:
    """Configuration for Stage 2 permutation.

    PURE GEOMETRY: MLP layers have N! permutation symmetries.
    We solve for the optimal permutation P that minimizes ||W_target - P @ W_source||_F
    via the Hungarian algorithm (optimal transport on discrete space).

    No arbitrary thresholds - the permutation that minimizes error is computed exactly.
    """

    # Whether to run permutation alignment (disabling skips this stage)
    enable_permutation: bool = True


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
    config: PermuteConfig,
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

    No arbitrary thresholds. The optimal permutation is computed exactly.
    """
    b = backend or get_default_backend()

    from modelcypher.core.domain.geometry.permutation_aligner import (
        Config as PAConfig,
    )
    from modelcypher.core.domain.geometry.permutation_aligner import (
        PermutationAligner,
    )
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

    if not config.enable_permutation:
        logger.info("PERMUTE: Disabled")
        return PermuteResult(source_weights, {"skipped": True})

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
        raise RuntimeError("PERMUTE: Embedding anchors missing; cannot phase-lock permutation.")

    if source_anchors.shape[1] != target_anchors.shape[1]:
        logger.info(
            "PERMUTE: Hidden dimension mismatch (source=%d, target=%d); skipping permutation stage.",
            source_anchors.shape[1],
            target_anchors.shape[1],
        )
        return PermuteResult(source_weights, {"skipped": True, "reason": "hidden_dim_mismatch"})

    precision_tol = max(machine_epsilon(b, source_anchors), 1e-12)
    embed_cka = compute_cka(source_anchors, target_anchors, backend=b).cka
    if embed_cka < 1.0 - precision_tol:
        raise RuntimeError(
            "PERMUTE: Embedding anchors not phase-locked (CKA=%.8f). "
            "Complete 1D/2D alignment before permutation."
            % embed_cka
        )

    logger.info(
        "PERMUTE: Embedding anchors phase-locked (CKA=%.8f). Skipping Procrustes rotation.",
        embed_cka,
    )

    # Configure aligner - no arbitrary thresholds
    pa_config = PAConfig(use_anchor_grounding=True)

    # Run MLP re-basin alignment with separate source/target anchors
    # This is critical: each model needs its own embeddings to compute meaningful signatures
    try:
        aligned, mean_quality, blocks_aligned = PermutationAligner.rebasin_mlp_with_activations(
            source_arr,
            target_arr,
            anchors=None,  # Use separate anchors instead
            anchor_activations=None,
            config=pa_config,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
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

    except Exception as e:
        logger.warning("PERMUTE: Alignment failed (%s), returning original weights", e)
        return PermuteResult(
            source_weights,
            {
                "skipped": True,
                "reason": "alignment_failed",
                "error": str(e),
            },
        )


def infer_hidden_dim(weights: dict[str, Any]) -> int:
    """Infer hidden dimension from weight shapes."""
    for key, val in weights.items():
        if "q_proj" in key or "k_proj" in key:
            return val.shape[1]
        if "up_proj" in key or "gate_proj" in key:
            return val.shape[1]
    return 4096
