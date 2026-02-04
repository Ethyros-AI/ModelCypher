# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Experiment 7: Cross-Model Alignment Transfer.

Tests whether alignment can transfer across models after Procrustes alignment.

Key Hypothesis:
    - Models encode invariant relationships in different coordinate systems
    - Procrustes alignment finds the rotation between coordinate systems
    - Refusal direction from Model A can transfer to Model B via the aligned transform
    - This tests the universality of alignment geometry

Method:
    1. Collect activations from both models on identical prompts
    2. Use Procrustes to align representations to consensus space
    3. Extract refusal direction from Model A
    4. Transform direction to Model B's space using Procrustes rotations
    5. Test classification accuracy with transferred direction
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.generalized_procrustes import GeneralizedProcrustes
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirectionDetector,
)
from modelcypher.experiments.refusal_direction import collect_activations_by_layer
from modelcypher.experiments.utils import (
    load_harmful_prompts,
    load_harmless_prompts,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class CrossModelAlignmentResult:
    """Results from cross-model alignment transfer experiment."""

    source_model: str
    target_model: str
    alignment_layer: int

    # Procrustes alignment metrics
    alignment_error: float
    consensus_variance_ratio: float
    converged: bool

    # Source model metrics (native refusal direction)
    source_accuracy: float
    source_separation: float

    # Target model metrics (native refusal direction)
    target_native_accuracy: float
    target_native_separation: float

    # Cross-model transfer metrics
    target_transferred_accuracy: float
    target_transferred_separation: float

    # Transfer effectiveness
    transfer_ratio: float  # transferred_accuracy / native_accuracy

    # Sample counts
    num_harmful: int
    num_harmless: int


def run_cross_model_alignment(
    source_model_path: str,
    target_model_path: str,
    alignment_layer: int | None = None,
    output_path: Path | None = None,
    backend: "Backend | None" = None,
) -> CrossModelAlignmentResult:
    """Run cross-model alignment transfer experiment.

    Tests whether refusal direction transfers across models after Procrustes alignment.

    Args:
        source_model_path: Path to source model (e.g., LFM)
        target_model_path: Path to target model (e.g., Qwen)
        alignment_layer: Layer to use for alignment (auto-detect if None)
        output_path: Where to save results
        backend: Backend for tensor operations

    Returns:
        CrossModelAlignmentResult with transfer metrics
    """
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.adapters.model_loader import ModelLoader

    b = backend or get_default_backend()

    logger.info("Starting cross-model alignment transfer experiment")
    logger.info("Source model: %s", source_model_path)
    logger.info("Target model: %s", target_model_path)

    # Load prompts
    harmful_prompts = load_harmful_prompts()
    harmless_prompts = load_harmless_prompts()

    # Use same prompts for both models
    n_prompts = min(30, len(harmful_prompts), len(harmless_prompts))
    harmful_prompts = harmful_prompts[:n_prompts]
    harmless_prompts = harmless_prompts[:n_prompts]

    logger.info("Using %d harmful, %d harmless prompts", len(harmful_prompts), len(harmless_prompts))

    # Load both models
    model_loader = ModelLoader()
    activation_provider = MLXActivationProvider()

    logger.info("Loading source model...")
    source_model, source_tokenizer = model_loader.load_model_for_training(str(source_model_path))

    logger.info("Loading target model...")
    target_model, target_tokenizer = model_loader.load_model_for_training(str(target_model_path))

    # Verify hidden dimensions match (required for Procrustes)
    # Get hidden dim from model config or embedding layer
    source_hidden = getattr(source_model, 'args', None)
    source_hidden = source_hidden.hidden_size if source_hidden else source_model.model.embed_tokens.weight.shape[1]
    target_hidden = getattr(target_model, 'args', None)
    target_hidden = target_hidden.hidden_size if target_hidden else target_model.model.embed_tokens.weight.shape[1]

    if source_hidden != target_hidden:
        raise ValueError(
            f"Hidden dimensions must match for Procrustes alignment. "
            f"Source: {source_hidden}, Target: {target_hidden}"
        )

    logger.info("Hidden dimension: %d", source_hidden)

    # Collect activations from source model
    logger.info("Collecting source model activations...")
    source_harmful_by_layer = collect_activations_by_layer(
        source_model, source_tokenizer, harmful_prompts, activation_provider
    )
    source_harmless_by_layer = collect_activations_by_layer(
        source_model, source_tokenizer, harmless_prompts, activation_provider
    )

    # Collect activations from target model on SAME prompts
    logger.info("Collecting target model activations...")
    target_harmful_by_layer = collect_activations_by_layer(
        target_model, target_tokenizer, harmful_prompts, activation_provider
    )
    target_harmless_by_layer = collect_activations_by_layer(
        target_model, target_tokenizer, harmless_prompts, activation_provider
    )

    # Find common layers (use min of both)
    source_layers = sorted(source_harmful_by_layer.keys())
    target_layers = sorted(target_harmful_by_layer.keys())
    common_layers = sorted(set(source_layers) & set(target_layers))

    logger.info("Source layers: %d, Target layers: %d, Common: %d",
                len(source_layers), len(target_layers), len(common_layers))

    # Auto-select alignment layer by max classification accuracy on source
    if alignment_layer is None:
        best_layer = common_layers[len(common_layers) // 2]
        best_accuracy = 0.0

        for layer_idx in common_layers:
            h_acts = b.stack(source_harmless_by_layer[layer_idx], axis=0)
            f_acts = b.stack(source_harmful_by_layer[layer_idx], axis=0)
            b.eval(h_acts, f_acts)

            # Compute direction
            h_mean = b.mean(h_acts, axis=0)
            f_mean = b.mean(f_acts, axis=0)
            direction = f_mean - h_mean
            norm = b.sqrt(b.sum(direction * direction))
            direction = direction / norm
            b.eval(direction)

            # Project and classify
            n_harmless = h_acts.shape[0]
            n_harmful = f_acts.shape[0]

            h_projs = []
            for i in range(n_harmless):
                act = b.take(h_acts, b.array([i]), axis=0)
                act = b.reshape(act, (act.shape[1],))
                proj = float(b.to_scalar(b.sum(act * direction)))
                h_projs.append(proj)

            f_projs = []
            for i in range(n_harmful):
                act = b.take(f_acts, b.array([i]), axis=0)
                act = b.reshape(act, (act.shape[1],))
                proj = float(b.to_scalar(b.sum(act * direction)))
                f_projs.append(proj)

            # Threshold at 95th percentile
            sorted_harmless = sorted(h_projs)
            threshold_idx = int(0.95 * len(sorted_harmless))
            threshold = sorted_harmless[min(threshold_idx, len(sorted_harmless) - 1)]

            tp = sum(1 for p in f_projs if p > threshold)
            tn = sum(1 for p in h_projs if p <= threshold)
            accuracy = (tp + tn) / (n_harmful + n_harmless)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_idx

        alignment_layer = best_layer
        logger.info("Selected layer %d with %.1f%% accuracy", alignment_layer, best_accuracy * 100)

    if alignment_layer not in common_layers:
        alignment_layer = common_layers[len(common_layers) // 2]
        logger.warning("Requested layer not in common layers, using %d", alignment_layer)

    logger.info("Using alignment layer: %d", alignment_layer)

    # Stack activations for the selected layer
    source_harmful_acts = b.stack(source_harmful_by_layer[alignment_layer], axis=0)
    source_harmless_acts = b.stack(source_harmless_by_layer[alignment_layer], axis=0)
    target_harmful_acts = b.stack(target_harmful_by_layer[alignment_layer], axis=0)
    target_harmless_acts = b.stack(target_harmless_by_layer[alignment_layer], axis=0)
    b.eval(source_harmful_acts, source_harmless_acts, target_harmful_acts, target_harmless_acts)

    # Combine all activations for Procrustes alignment
    # Same prompts = same rows, different models = different coordinate systems
    source_all = b.concatenate([source_harmless_acts, source_harmful_acts], axis=0)
    target_all = b.concatenate([target_harmless_acts, target_harmful_acts], axis=0)
    b.eval(source_all, target_all)

    # Run Procrustes alignment
    logger.info("Running Procrustes alignment...")
    procrustes = GeneralizedProcrustes(backend=b)

    # Convert to list format for Procrustes
    source_list = b.tolist(source_all)
    target_list = b.tolist(target_all)

    result = procrustes.align([source_list, target_list])

    if result is None:
        raise RuntimeError("Procrustes alignment failed")

    logger.info("Procrustes converged: %s, error: %.6f, variance ratio: %.4f",
                result.converged, result.alignment_error, result.consensus_variance_ratio)

    # Extract rotation matrices
    # rotations[0] is source rotation, rotations[1] is target rotation
    source_rotation = b.array(result.rotations[0])  # [d, d]
    target_rotation = b.array(result.rotations[1])  # [d, d]
    b.eval(source_rotation, target_rotation)

    # Compute source model's native refusal direction
    logger.info("Computing source model's refusal direction...")
    source_h_mean = b.mean(source_harmless_acts, axis=0)
    source_f_mean = b.mean(source_harmful_acts, axis=0)
    source_direction = source_f_mean - source_h_mean
    source_norm = b.sqrt(b.sum(source_direction * source_direction))
    source_direction = source_direction / source_norm
    b.eval(source_direction)

    # Compute target model's native refusal direction
    logger.info("Computing target model's refusal direction...")
    target_h_mean = b.mean(target_harmless_acts, axis=0)
    target_f_mean = b.mean(target_harmful_acts, axis=0)
    target_native_direction = target_f_mean - target_h_mean
    target_native_norm = b.sqrt(b.sum(target_native_direction * target_native_direction))
    target_native_direction = target_native_direction / target_native_norm
    b.eval(target_native_direction)

    # Transform source direction to target's coordinate system via Procrustes
    # Source direction in consensus space: d_source @ R_source
    # Then transform to target space: (d_source @ R_source) @ R_target.T
    logger.info("Transforming direction via Procrustes...")
    direction_in_consensus = b.matmul(
        b.reshape(source_direction, (1, -1)),
        source_rotation
    )
    # R_target maps target -> consensus, so R_target.T maps consensus -> target
    transferred_direction = b.matmul(
        direction_in_consensus,
        b.transpose(target_rotation)
    )
    transferred_direction = b.reshape(transferred_direction, (-1,))
    transferred_norm = b.sqrt(b.sum(transferred_direction * transferred_direction))
    transferred_direction = transferred_direction / transferred_norm
    b.eval(transferred_direction)

    # Function to compute classification metrics
    def compute_metrics(direction: "Array", harmless_acts: "Array", harmful_acts: "Array") -> tuple[float, float]:
        n_harmless = harmless_acts.shape[0]
        n_harmful = harmful_acts.shape[0]

        h_projs = []
        for i in range(n_harmless):
            act = b.take(harmless_acts, b.array([i]), axis=0)
            act = b.reshape(act, (act.shape[1],))
            proj = float(b.to_scalar(b.sum(act * direction)))
            h_projs.append(proj)

        f_projs = []
        for i in range(n_harmful):
            act = b.take(harmful_acts, b.array([i]), axis=0)
            act = b.reshape(act, (act.shape[1],))
            proj = float(b.to_scalar(b.sum(act * direction)))
            f_projs.append(proj)

        # Separation
        h_mean = sum(h_projs) / len(h_projs)
        f_mean = sum(f_projs) / len(f_projs)
        separation = abs(f_mean - h_mean)

        # Accuracy at 95th percentile threshold
        sorted_harmless = sorted(h_projs)
        threshold_idx = int(0.95 * len(sorted_harmless))
        threshold = sorted_harmless[min(threshold_idx, len(sorted_harmless) - 1)]

        tp = sum(1 for p in f_projs if p > threshold)
        tn = sum(1 for p in h_projs if p <= threshold)
        accuracy = (tp + tn) / (n_harmful + n_harmless)

        return accuracy, separation

    # Compute metrics for all three cases
    logger.info("Computing source model metrics...")
    source_accuracy, source_separation = compute_metrics(
        source_direction, source_harmless_acts, source_harmful_acts
    )

    logger.info("Computing target native metrics...")
    target_native_accuracy, target_native_separation = compute_metrics(
        target_native_direction, target_harmless_acts, target_harmful_acts
    )

    logger.info("Computing target transferred metrics...")
    target_transferred_accuracy, target_transferred_separation = compute_metrics(
        transferred_direction, target_harmless_acts, target_harmful_acts
    )

    # Transfer ratio: how well does transferred direction compare to native
    transfer_ratio = target_transferred_accuracy / max(target_native_accuracy, 1e-8)

    logger.info("Source accuracy: %.1f%%", source_accuracy * 100)
    logger.info("Target native accuracy: %.1f%%", target_native_accuracy * 100)
    logger.info("Target transferred accuracy: %.1f%%", target_transferred_accuracy * 100)
    logger.info("Transfer ratio: %.2f", transfer_ratio)

    result_data = CrossModelAlignmentResult(
        source_model=source_model_path,
        target_model=target_model_path,
        alignment_layer=alignment_layer,
        alignment_error=result.alignment_error,
        consensus_variance_ratio=result.consensus_variance_ratio,
        converged=result.converged,
        source_accuracy=source_accuracy,
        source_separation=source_separation,
        target_native_accuracy=target_native_accuracy,
        target_native_separation=target_native_separation,
        target_transferred_accuracy=target_transferred_accuracy,
        target_transferred_separation=target_transferred_separation,
        transfer_ratio=transfer_ratio,
        num_harmful=len(harmful_prompts),
        num_harmless=len(harmless_prompts),
    )

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(result_data), f, indent=2)
        logger.info("Results saved to %s", output_path)

    return result_data


__all__ = [
    "CrossModelAlignmentResult",
    "run_cross_model_alignment",
]
