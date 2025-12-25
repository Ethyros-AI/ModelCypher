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

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class PermuteConfig:
    """Configuration for Stage 2 permutation."""

    enable_permutation: bool = True
    permutation_confidence_threshold: float = 0.6


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
    Stage 2: Permutation alignment for MLP neurons.

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        intersection_map_obj: IntersectionMap object (dimension-level correlations)
        layer_confidences: Per-layer confidence scores from probing
        config: Permutation configuration
        infer_hidden_dim_fn: Function to infer hidden dimension from weights
        backend: Backend protocol instance

    Returns:
        PermuteResult with aligned weights and metrics
    """
    b = backend or get_default_backend()

    from modelcypher.core.domain.geometry.permutation_aligner import (
        Config as PAConfig,
    )
    from modelcypher.core.domain.geometry.permutation_aligner import (
        PermutationAligner,
    )

    if not config.enable_permutation:
        logger.info("PERMUTE: Disabled")
        return PermuteResult(source_weights, {"skipped": True})

    # Use IntersectionMap dimension correlations for targeted permutation if available
    if intersection_map_obj is not None:
        pass

    # Convert weights to backend arrays
    source_arr: dict[str, "Array"] = {}
    target_arr: dict[str, "Array"] = {}

    for key, val in source_weights.items():
        arr = b.astype(b.array(val), "float32")
        b.eval(arr)
        source_arr[key] = arr
    for key, val in target_weights.items():
        arr = b.astype(b.array(val), "float32")
        b.eval(arr)
        target_arr[key] = arr

    # Build anchor embeddings from model's embedding layer
    anchor_key = None
    for key in target_weights:
        if "embed_tokens" in key or "wte" in key or "embedding" in key.lower():
            anchor_key = key
            break

    if anchor_key is not None:
        embed = target_weights[anchor_key]
        num_anchors = min(128, embed.shape[0])
        embed_slice = embed[:num_anchors]
        anchors = b.astype(b.array(embed_slice), "float32")
        b.eval(anchors)
        logger.info("PERMUTE: Using %d embedding anchors from %s", num_anchors, anchor_key)
    else:
        hidden_dim = infer_hidden_dim_fn(target_weights)
        b.random_seed(42)
        anchors = b.random_normal((64, hidden_dim)) * 0.1
        b.eval(anchors)
        logger.warning("PERMUTE: No embedding found, using random anchors (dim=%d)", hidden_dim)

    # Check mean confidence
    conf_vals = list(layer_confidences.values())
    mean_confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
    if mean_confidence < config.permutation_confidence_threshold:
        logger.info(
            "PERMUTE: Skipped (mean confidence %.3f < threshold %.3f)",
            mean_confidence,
            config.permutation_confidence_threshold,
        )
        return PermuteResult(
            source_weights,
            {
                "skipped": True,
                "reason": "low_confidence",
                "mean_confidence": float(mean_confidence),
            },
        )

    # Configure aligner
    pa_config = PAConfig(
        min_match_threshold=0.1,
        use_anchor_grounding=True,
    )

    # Run MLP re-basin alignment
    try:
        aligned, mean_quality, blocks_aligned = PermutationAligner.rebasin_mlp_with_activations(
            source_arr,
            target_arr,
            anchors,
            anchor_activations=None,
            config=pa_config,
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
            "threshold": config.permutation_confidence_threshold,
            "mean_confidence": float(mean_confidence),
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
