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

"""Experiment 1: Detect Alignment Geometrically.

Proves that base vs instruct models occupy measurably different manifold positions.

Method:
    1. Load base model and instruct model (same architecture)
    2. Run identical prompts through both, extract activations at each layer
    3. Compute:
       - Raw CKA (before alignment) - expect ~0.6-0.8
       - Aligned CKA (after Procrustes) - expect ~0.95-1.0
       - Subspace overlap via principal angles
       - Intrinsic dimension at each layer
    4. Identify the "alignment delta" = geometric difference

Hypothesis:
    Instruct models have additional structure in sparse directions (low variance
    in base, high variance in instruct). This is the "alignment manifold."

Usage:
    from modelcypher.experiments.alignment_detection import run_alignment_detection

    result = run_alignment_detection(
        base_model_path="/path/to/base",
        instruct_model_path="/path/to/instruct",
        output_path="results/alignment_detection.json",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka, compute_cka_split
from modelcypher.core.domain.geometry.direction_novelty import (
    compute_per_direction_novelty,
    diagnose_variance_distribution,
)
from modelcypher.core.domain.geometry.generalized_procrustes import (
    GeneralizedProcrustes,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)
from modelcypher.core.domain.geometry.subspace import compute_subspace_overlap

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerAlignmentMetrics:
    """Alignment metrics for a single layer."""

    layer_index: int
    raw_cka: float  # CKA before Procrustes alignment
    aligned_cka: float  # CKA after Procrustes alignment
    subspace_overlap: float  # Principal angle-based overlap
    base_intrinsic_dim: float  # Intrinsic dimension of base activations
    instruct_intrinsic_dim: float  # Intrinsic dimension of instruct activations
    novel_count: int  # Directions novel to instruct
    shared_count: int  # Directions shared between models
    mean_novelty: float  # Mean novelty ratio


@dataclass
class AlignmentDetectionResult:
    """Complete result of alignment detection experiment."""

    base_model_path: str
    instruct_model_path: str
    num_prompts: int
    num_layers: int
    layer_metrics: list[LayerAlignmentMetrics]
    aggregate_metrics: dict[str, float]
    variance_diagnostics: dict[str, dict[str, float]]


def collect_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    activation_provider: "ActivationProvider",
) -> dict[int, "Array"]:
    """Collect activations for all prompts, stacked by layer.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompts: List of text prompts
        activation_provider: Provider for activation extraction

    Returns:
        Dict mapping layer_idx -> stacked activations [n_prompts, hidden_dim]
    """
    backend = get_default_backend()

    # Collect activations for each prompt
    all_activations: list[dict[int, Any]] = []
    for prompt in prompts:
        acts = activation_provider.collect_hidden_activations(model, tokenizer, prompt)
        all_activations.append(acts)

    if not all_activations:
        return {}

    # Stack by layer
    layer_indices = sorted(all_activations[0].keys())
    stacked: dict[int, Any] = {}

    for layer_idx in layer_indices:
        layer_acts = [acts[layer_idx] for acts in all_activations]
        # Stack into [n_prompts, hidden_dim]
        stacked_arr = backend.stack(layer_acts, axis=0)
        backend.eval(stacked_arr)
        stacked[layer_idx] = stacked_arr

    return stacked


def compute_layer_metrics(
    base_activations: "Array",
    instruct_activations: "Array",
    layer_index: int,
    backend: "Backend",
) -> LayerAlignmentMetrics:
    """Compute alignment metrics for a single layer.

    Args:
        base_activations: Base model activations [n_prompts, hidden_dim]
        instruct_activations: Instruct model activations [n_prompts, hidden_dim]
        layer_index: Index of this layer
        backend: Backend for tensor operations

    Returns:
        LayerAlignmentMetrics with all computed metrics
    """
    # 1. Raw CKA (before alignment)
    raw_cka_result = compute_cka(base_activations, instruct_activations, backend=backend)
    raw_cka = raw_cka_result.cka

    # 2. Procrustes alignment
    procrustes = GeneralizedProcrustes(backend=backend)
    alignment_result = procrustes.align([base_activations, instruct_activations])

    # Apply rotation to base activations to align with instruct
    # rotations[0] is the rotation for the first matrix (base)
    rotation_base = backend.array(alignment_result.rotations[0])
    backend.eval(rotation_base)
    base_aligned = backend.matmul(base_activations, rotation_base)
    backend.eval(base_aligned)

    # 3. Aligned CKA (after alignment)
    aligned_cka_result = compute_cka(base_aligned, instruct_activations, backend=backend)
    aligned_cka = aligned_cka_result.cka

    # 4. Subspace overlap via principal angles
    subspace_result = compute_subspace_overlap(base_aligned, instruct_activations, backend=backend)
    subspace_overlap = subspace_result.overlap_fraction

    # 5. Intrinsic dimension
    base_id = IntrinsicDimension.compute_two_nn(base_activations, backend=backend)
    instruct_id = IntrinsicDimension.compute_two_nn(instruct_activations, backend=backend)

    # 6. Direction novelty (what's active in instruct but dormant in base)
    # Note: We swap order because we want instruct-novel directions
    novelty_result = compute_per_direction_novelty(
        instruct_activations,  # "source" = instruct
        base_activations,  # "target" = base
        backend=backend,
    )

    return LayerAlignmentMetrics(
        layer_index=layer_index,
        raw_cka=float(raw_cka),
        aligned_cka=float(aligned_cka),
        subspace_overlap=float(subspace_overlap),
        base_intrinsic_dim=float(base_id.intrinsic_dimension),
        instruct_intrinsic_dim=float(instruct_id.intrinsic_dimension),
        novel_count=novelty_result.novel_count,
        shared_count=novelty_result.shared_count,
        mean_novelty=novelty_result.mean_novelty,
    )


def run_alignment_detection(
    base_model_path: str | Path,
    instruct_model_path: str | Path,
    prompts: list[str] | None = None,
    output_path: str | Path | None = None,
    layers_to_analyze: list[int] | None = None,
) -> AlignmentDetectionResult:
    """Run the alignment detection experiment.

    Args:
        base_model_path: Path to base model
        instruct_model_path: Path to instruct model
        prompts: List of prompts to use (uses default if None)
        output_path: Path to save results JSON (optional)
        layers_to_analyze: Specific layers to analyze (all if None)

    Returns:
        AlignmentDetectionResult with full analysis
    """
    from modelcypher.adapters.activation_provider import ActivationProvider
    from modelcypher.adapters.model_loader import ModelLoader

    backend = get_default_backend()

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "What is the capital of France?",
            "Explain quantum entanglement in simple terms.",
            "Write a haiku about autumn leaves.",
            "How do neural networks learn?",
            "The sun sets over the mountains, casting long shadows.",
            "Calculate the sum of 1 to 100.",
            "Describe the water cycle.",
            "Why is the sky blue?",
            "List the planets in our solar system.",
        ]

    # Load models using MLX model loader
    model_loader = ModelLoader()

    logger.info("Loading base model from %s", base_model_path)
    base_model, base_tokenizer = model_loader.load_model_for_training(str(base_model_path))

    logger.info("Loading instruct model from %s", instruct_model_path)
    instruct_model, instruct_tokenizer = model_loader.load_model_for_training(str(instruct_model_path))

    # Get activation provider
    activation_provider = ActivationProvider()

    logger.info("Collecting activations for %d prompts", len(prompts))

    # Collect activations
    base_acts = collect_activations(base_model, base_tokenizer, prompts, activation_provider)
    instruct_acts = collect_activations(
        instruct_model, instruct_tokenizer, prompts, activation_provider
    )

    # Determine layers to analyze
    all_layers = sorted(base_acts.keys())
    if layers_to_analyze is not None:
        layers = [l for l in layers_to_analyze if l in all_layers]
    else:
        layers = all_layers

    logger.info("Analyzing %d layers: %s", len(layers), layers)

    # Compute metrics for each layer
    layer_metrics: list[LayerAlignmentMetrics] = []
    variance_diagnostics: dict[str, dict[str, float]] = {}

    for layer_idx in layers:
        logger.info("Analyzing layer %d...", layer_idx)

        base_layer_acts = base_acts[layer_idx]
        instruct_layer_acts = instruct_acts[layer_idx]

        metrics = compute_layer_metrics(
            base_layer_acts, instruct_layer_acts, layer_idx, backend
        )
        layer_metrics.append(metrics)

        # Compute variance diagnostics for this layer
        novelty_result = compute_per_direction_novelty(
            instruct_layer_acts, base_layer_acts, backend=backend
        )
        var_diag = diagnose_variance_distribution(novelty_result, backend=backend)
        variance_diagnostics[f"layer_{layer_idx}"] = var_diag

    # Compute aggregate metrics
    if layer_metrics:
        aggregate_metrics = {
            "mean_raw_cka": sum(m.raw_cka for m in layer_metrics) / len(layer_metrics),
            "mean_aligned_cka": sum(m.aligned_cka for m in layer_metrics) / len(layer_metrics),
            "mean_subspace_overlap": sum(m.subspace_overlap for m in layer_metrics)
            / len(layer_metrics),
            "mean_base_id": sum(m.base_intrinsic_dim for m in layer_metrics) / len(layer_metrics),
            "mean_instruct_id": sum(m.instruct_intrinsic_dim for m in layer_metrics)
            / len(layer_metrics),
            "total_novel_directions": sum(m.novel_count for m in layer_metrics),
            "total_shared_directions": sum(m.shared_count for m in layer_metrics),
            "cka_improvement": (
                sum(m.aligned_cka for m in layer_metrics) / len(layer_metrics)
                - sum(m.raw_cka for m in layer_metrics) / len(layer_metrics)
            ),
        }
    else:
        aggregate_metrics = {}

    result = AlignmentDetectionResult(
        base_model_path=str(base_model_path),
        instruct_model_path=str(instruct_model_path),
        num_prompts=len(prompts),
        num_layers=len(layers),
        layer_metrics=layer_metrics,
        aggregate_metrics=aggregate_metrics,
        variance_diagnostics=variance_diagnostics,
    )

    # Save results if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        result_dict = {
            "base_model_path": result.base_model_path,
            "instruct_model_path": result.instruct_model_path,
            "num_prompts": result.num_prompts,
            "num_layers": result.num_layers,
            "layer_metrics": [asdict(m) for m in result.layer_metrics],
            "aggregate_metrics": result.aggregate_metrics,
            "variance_diagnostics": result.variance_diagnostics,
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("ALIGNMENT DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Base model: %s", base_model_path)
    logger.info("Instruct model: %s", instruct_model_path)
    logger.info("Prompts analyzed: %d", len(prompts))
    logger.info("Layers analyzed: %d", len(layers))
    logger.info("-" * 60)
    logger.info("Mean Raw CKA: %.4f", aggregate_metrics.get("mean_raw_cka", 0))
    logger.info("Mean Aligned CKA: %.4f", aggregate_metrics.get("mean_aligned_cka", 0))
    logger.info("CKA Improvement: +%.4f", aggregate_metrics.get("cka_improvement", 0))
    logger.info("Mean Subspace Overlap: %.4f", aggregate_metrics.get("mean_subspace_overlap", 0))
    logger.info("Mean Base ID: %.2f", aggregate_metrics.get("mean_base_id", 0))
    logger.info("Mean Instruct ID: %.2f", aggregate_metrics.get("mean_instruct_id", 0))
    logger.info("Total Novel Directions: %d", aggregate_metrics.get("total_novel_directions", 0))
    logger.info("=" * 60)

    return result


__all__ = [
    "AlignmentDetectionResult",
    "LayerAlignmentMetrics",
    "collect_activations",
    "compute_layer_metrics",
    "run_alignment_detection",
]
