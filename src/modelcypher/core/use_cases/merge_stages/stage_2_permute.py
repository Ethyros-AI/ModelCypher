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
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def _is_mlx_array(arr: Any) -> bool:
    """Check if array is an MLX array."""
    try:
        import mlx.core as mx

        return isinstance(arr, mx.array)
    except ImportError:
        return False


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to numpy."""
    if _is_mlx_array(arr):
        import mlx.core as mx

        mx.eval(arr)
        return np.array(arr)
    return np.asarray(arr)


@dataclass
class PermuteConfig:
    """Configuration for Stage 2 permutation."""

    enable_permutation: bool = True
    permutation_confidence_threshold: float = 0.6


@dataclass
class PermuteResult:
    """Result of Stage 2 permutation."""

    weights: dict[str, Any]  # np.ndarray or mx.array
    metrics: dict[str, Any]


def stage_permute(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    config: PermuteConfig,
    infer_hidden_dim_fn: Callable[[dict[str, Any]], int],
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

    Returns:
        PermuteResult with aligned weights and metrics
    """
    import mlx.core as mx

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
    dimension_correlations = {}
    if intersection_map_obj is not None:
        dimension_correlations = intersection_map_obj.dimension_correlations

    # Convert weights to MLX arrays (handles both numpy and MLX input)
    source_mx: dict[str, mx.array] = {}
    target_mx: dict[str, mx.array] = {}

    for key, val in source_weights.items():
        if _is_mlx_array(val):
            source_mx[key] = val.astype(mx.float32)
        else:
            source_mx[key] = mx.array(np.asarray(val, dtype=np.float32))
    for key, val in target_weights.items():
        if _is_mlx_array(val):
            target_mx[key] = val.astype(mx.float32)
        else:
            target_mx[key] = mx.array(np.asarray(val, dtype=np.float32))

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
        if _is_mlx_array(embed_slice):
            anchors = embed_slice.astype(mx.float32)
        else:
            anchors = mx.array(np.asarray(embed_slice, dtype=np.float32))
        logger.info("PERMUTE: Using %d embedding anchors from %s", num_anchors, anchor_key)
    else:
        hidden_dim = infer_hidden_dim_fn(target_weights)
        anchors = mx.random.normal((64, hidden_dim)) * 0.1
        logger.warning("PERMUTE: No embedding found, using random anchors (dim=%d)", hidden_dim)

    # Check mean confidence
    mean_confidence = np.mean(list(layer_confidences.values())) if layer_confidences else 0.0
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
        aligned_mx, mean_quality, blocks_aligned = PermutationAligner.rebasin_mlp_with_activations(
            source_mx,
            target_mx,
            anchors,
            anchor_activations=None,
            config=pa_config,
        )
        mx.eval(aligned_mx)

        # Convert back to numpy
        permuted: dict[str, np.ndarray] = {}
        for key, val in aligned_mx.items():
            permuted[key] = np.asarray(val)

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

        return PermuteResult(permuted, metrics)

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


def infer_hidden_dim(weights: dict[str, np.ndarray]) -> int:
    """Infer hidden dimension from weight shapes."""
    for key, val in weights.items():
        if "q_proj" in key or "k_proj" in key:
            return val.shape[1]
        if "up_proj" in key or "gate_proj" in key:
            return val.shape[1]
    return 4096
