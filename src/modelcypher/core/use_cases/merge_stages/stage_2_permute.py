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
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PermuteConfig:
    """Configuration for Stage 2 permutation."""

    enable_permutation: bool = True
    permutation_confidence_threshold: float = 0.6


@dataclass
class PermuteResult:
    """Result of Stage 2 permutation."""

    weights: dict[str, np.ndarray]
    metrics: dict[str, Any]


def stage_permute(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    intersection_map_obj: Optional[Any],
    layer_confidences: dict[int, float],
    config: PermuteConfig,
    infer_hidden_dim_fn: Callable[[dict[str, np.ndarray]], int],
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
        PermutationAligner,
        Config as PAConfig,
    )

    if not config.enable_permutation:
        logger.info("PERMUTE: Disabled")
        return PermuteResult(source_weights, {"skipped": True})

    # Use IntersectionMap dimension correlations for targeted permutation if available
    dimension_correlations = {}
    if intersection_map_obj is not None:
        dimension_correlations = intersection_map_obj.dimension_correlations

    # Convert numpy weights to MLX arrays
    source_mx: dict[str, mx.array] = {}
    target_mx: dict[str, mx.array] = {}

    for key, val in source_weights.items():
        source_mx[key] = mx.array(val.astype(np.float32))
    for key, val in target_weights.items():
        target_mx[key] = mx.array(val.astype(np.float32))

    # Build anchor embeddings from model's embedding layer
    anchor_key = None
    for key in target_weights:
        if "embed_tokens" in key or "wte" in key or "embedding" in key.lower():
            anchor_key = key
            break

    if anchor_key is not None:
        embed = target_weights[anchor_key]
        num_anchors = min(128, embed.shape[0])
        anchors = mx.array(embed[:num_anchors].astype(np.float32))
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
