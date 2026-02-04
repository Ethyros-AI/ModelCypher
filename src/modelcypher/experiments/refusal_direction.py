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

"""Experiment 2: Extract Refusal Direction.

Reproduces Arditi et al.'s refusal direction finding using our infrastructure.

Method:
    1. Prepare prompt pairs (harmful, harmless)
    2. Run through instruct model, extract activations at each layer
    3. Compute refusal direction: r = mean(harmful) - mean(harmless)
    4. Validate by checking projections

Hypothesis:
    Refusal is mediated by a single low-dimensional direction in activation space.
    Adding this direction to harmless prompts should induce refusal-like geometry.

Usage:
    from modelcypher.experiments.refusal_direction import run_refusal_direction_experiment

    result = run_refusal_direction_experiment(
        model_path="/path/to/instruct/model",
        output_path="results/refusal_direction.json",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirectionDetector,
    RefusalDirection,
)
from modelcypher.experiments.utils import load_harmful_prompts, load_harmless_prompts

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerRefusalMetrics:
    """Refusal direction metrics for a single layer."""

    layer_index: int
    strength: float  # ||r|| - magnitude of refusal direction
    explained_variance: float  # How much variance direction explains
    mean_harmful_projection: float  # Mean projection of harmful prompts onto r
    mean_harmless_projection: float  # Mean projection of harmless prompts onto r
    separation: float  # Difference between harmful and harmless projections
    classification_accuracy: float  # How well r separates harmful from harmless


@dataclass
class RefusalDirectionResult:
    """Complete result of refusal direction extraction experiment."""

    model_path: str
    num_harmful_prompts: int
    num_harmless_prompts: int
    num_layers: int
    layer_metrics: list[LayerRefusalMetrics]
    best_layer: int  # Layer with highest separation
    best_layer_accuracy: float
    aggregate_metrics: dict[str, float]


def collect_activations_by_layer(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    activation_provider: "ActivationProvider",
) -> dict[int, list["Array"]]:
    """Collect activations for all prompts, organized by layer.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompts: List of text prompts
        activation_provider: Provider for activation extraction

    Returns:
        Dict mapping layer_idx -> list of activation vectors [hidden_dim]
    """
    backend = get_default_backend()

    # Collect activations for each prompt
    by_layer: dict[int, list[Any]] = {}

    for prompt in prompts:
        acts = activation_provider.collect_hidden_activations(model, tokenizer, prompt)
        for layer_idx, activation in acts.items():
            if layer_idx not in by_layer:
                by_layer[layer_idx] = []
            by_layer[layer_idx].append(activation)

    return by_layer


def compute_projections(
    activations: "Array",
    direction: "Array",
    backend: "Backend",
) -> "Array":
    """Compute projections of activations onto direction.

    Args:
        activations: [n_samples, hidden_dim] activation matrix
        direction: [hidden_dim] direction vector (normalized)
        backend: Backend for tensor operations

    Returns:
        [n_samples] array of projection values
    """
    # projection = dot(activation, direction)
    projections = backend.sum(activations * direction, axis=1)
    backend.eval(projections)
    return projections


def compute_layer_refusal_metrics(
    harmful_activations: "Array",
    harmless_activations: "Array",
    layer_index: int,
    model_id: str,
    backend: "Backend",
) -> tuple[LayerRefusalMetrics, RefusalDirection | None]:
    """Compute refusal direction metrics for a single layer.

    Args:
        harmful_activations: [n_harmful, hidden_dim] harmful prompt activations
        harmless_activations: [n_harmless, hidden_dim] harmless prompt activations
        layer_index: Index of this layer
        model_id: Model identifier
        backend: Backend for tensor operations

    Returns:
        Tuple of (LayerRefusalMetrics, RefusalDirection or None)
    """
    # Compute refusal direction using the detector
    refusal_dir = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful_activations,
        harmless_activations=harmless_activations,
        layer_index=layer_index,
        model_id=model_id,
    )

    if refusal_dir is None:
        # Return zero metrics if direction couldn't be computed
        return LayerRefusalMetrics(
            layer_index=layer_index,
            strength=0.0,
            explained_variance=0.0,
            mean_harmful_projection=0.0,
            mean_harmless_projection=0.0,
            separation=0.0,
            classification_accuracy=0.0,
        ), None

    # Get the direction vector
    direction = refusal_dir.direction
    backend.eval(direction)

    # Compute projections
    harmful_proj = compute_projections(harmful_activations, direction, backend)
    harmless_proj = compute_projections(harmless_activations, direction, backend)

    # Compute means
    mean_harmful = float(backend.to_scalar(backend.mean(harmful_proj)))
    mean_harmless = float(backend.to_scalar(backend.mean(harmless_proj)))
    separation = mean_harmful - mean_harmless

    # Compute classification accuracy using threshold at midpoint
    threshold = (mean_harmful + mean_harmless) / 2.0

    # Count correct classifications
    # Harmful should have projection > threshold
    # Harmless should have projection < threshold
    harmful_correct = backend.sum(
        backend.astype(harmful_proj > threshold, backend.dtype(harmful_proj))
    )
    harmless_correct = backend.sum(
        backend.astype(harmless_proj < threshold, backend.dtype(harmless_proj))
    )
    backend.eval(harmful_correct, harmless_correct)

    n_harmful = int(backend.shape(harmful_proj)[0])
    n_harmless = int(backend.shape(harmless_proj)[0])
    total = n_harmful + n_harmless

    accuracy = (
        float(backend.to_scalar(harmful_correct))
        + float(backend.to_scalar(harmless_correct))
    ) / max(total, 1)

    return LayerRefusalMetrics(
        layer_index=layer_index,
        strength=refusal_dir.strength,
        explained_variance=refusal_dir.explained_variance,
        mean_harmful_projection=mean_harmful,
        mean_harmless_projection=mean_harmless,
        separation=separation,
        classification_accuracy=accuracy,
    ), refusal_dir


def run_refusal_direction_experiment(
    model_path: str | Path,
    harmful_prompts: list[str] | None = None,
    harmless_prompts: list[str] | None = None,
    output_path: str | Path | None = None,
    layers_to_analyze: list[int] | None = None,
) -> RefusalDirectionResult:
    """Run the refusal direction extraction experiment.

    Args:
        model_path: Path to instruct model
        harmful_prompts: List of harmful prompts (uses default if None)
        harmless_prompts: List of harmless prompts (uses default if None)
        output_path: Path to save results JSON (optional)
        layers_to_analyze: Specific layers to analyze (all if None)

    Returns:
        RefusalDirectionResult with full analysis
    """
    from modelcypher.adapters.activation_provider import ActivationProvider
    from modelcypher.adapters.model_loader import ModelLoader

    backend = get_default_backend()

    # Load prompts
    if harmful_prompts is None:
        harmful_prompts = load_harmful_prompts()
    if harmless_prompts is None:
        harmless_prompts = load_harmless_prompts()

    if not harmful_prompts or not harmless_prompts:
        raise ValueError("No prompts available. Check datasets directory.")

    logger.info("Loading model from %s", model_path)
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model_for_training(str(model_path))

    # Get model ID from path
    model_id = Path(model_path).name

    # Get activation provider
    activation_provider = ActivationProvider()

    logger.info(
        "Collecting activations for %d harmful and %d harmless prompts",
        len(harmful_prompts),
        len(harmless_prompts),
    )

    # Collect activations
    harmful_by_layer = collect_activations_by_layer(
        model, tokenizer, harmful_prompts, activation_provider
    )
    harmless_by_layer = collect_activations_by_layer(
        model, tokenizer, harmless_prompts, activation_provider
    )

    # Determine layers to analyze
    all_layers = sorted(harmful_by_layer.keys())
    if layers_to_analyze is not None:
        layers = [l for l in layers_to_analyze if l in all_layers]
    else:
        layers = all_layers

    logger.info("Analyzing %d layers: %s", len(layers), layers)

    # Compute metrics for each layer
    layer_metrics: list[LayerRefusalMetrics] = []
    refusal_directions: dict[int, RefusalDirection] = {}

    for layer_idx in layers:
        logger.info("Analyzing layer %d...", layer_idx)

        # Stack activations for this layer
        harmful_acts = backend.stack(harmful_by_layer[layer_idx], axis=0)
        harmless_acts = backend.stack(harmless_by_layer[layer_idx], axis=0)
        backend.eval(harmful_acts, harmless_acts)

        metrics, refusal_dir = compute_layer_refusal_metrics(
            harmful_acts, harmless_acts, layer_idx, model_id, backend
        )
        layer_metrics.append(metrics)

        if refusal_dir is not None:
            refusal_directions[layer_idx] = refusal_dir

        logger.info(
            "  Layer %d: strength=%.4f, separation=%.4f, accuracy=%.2f%%",
            layer_idx,
            metrics.strength,
            metrics.separation,
            metrics.classification_accuracy * 100,
        )

    # Find best layer (highest separation)
    if layer_metrics:
        best_layer_metrics = max(layer_metrics, key=lambda m: abs(m.separation))
        best_layer = best_layer_metrics.layer_index
        best_layer_accuracy = best_layer_metrics.classification_accuracy
    else:
        best_layer = 0
        best_layer_accuracy = 0.0

    # Compute aggregate metrics
    if layer_metrics:
        aggregate_metrics = {
            "mean_strength": sum(m.strength for m in layer_metrics) / len(layer_metrics),
            "mean_explained_variance": sum(m.explained_variance for m in layer_metrics)
            / len(layer_metrics),
            "mean_separation": sum(abs(m.separation) for m in layer_metrics)
            / len(layer_metrics),
            "mean_accuracy": sum(m.classification_accuracy for m in layer_metrics)
            / len(layer_metrics),
            "max_separation": max(abs(m.separation) for m in layer_metrics),
            "max_accuracy": max(m.classification_accuracy for m in layer_metrics),
            "best_layer": best_layer,
        }
    else:
        aggregate_metrics = {}

    result = RefusalDirectionResult(
        model_path=str(model_path),
        num_harmful_prompts=len(harmful_prompts),
        num_harmless_prompts=len(harmless_prompts),
        num_layers=len(layers),
        layer_metrics=layer_metrics,
        best_layer=best_layer,
        best_layer_accuracy=best_layer_accuracy,
        aggregate_metrics=aggregate_metrics,
    )

    # Save results if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        result_dict = {
            "model_path": result.model_path,
            "num_harmful_prompts": result.num_harmful_prompts,
            "num_harmless_prompts": result.num_harmless_prompts,
            "num_layers": result.num_layers,
            "layer_metrics": [asdict(m) for m in result.layer_metrics],
            "best_layer": result.best_layer,
            "best_layer_accuracy": result.best_layer_accuracy,
            "aggregate_metrics": result.aggregate_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("REFUSAL DIRECTION EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Model: %s", model_path)
    logger.info("Harmful prompts: %d", len(harmful_prompts))
    logger.info("Harmless prompts: %d", len(harmless_prompts))
    logger.info("Layers analyzed: %d", len(layers))
    logger.info("-" * 60)
    logger.info("Best layer: %d", best_layer)
    logger.info("Best layer accuracy: %.2f%%", best_layer_accuracy * 100)
    logger.info("Mean separation: %.4f", aggregate_metrics.get("mean_separation", 0))
    logger.info("Max separation: %.4f", aggregate_metrics.get("max_separation", 0))
    logger.info("Mean accuracy: %.2f%%", aggregate_metrics.get("mean_accuracy", 0) * 100)
    logger.info("=" * 60)

    return result


__all__ = [
    "LayerRefusalMetrics",
    "RefusalDirectionResult",
    "collect_activations_by_layer",
    "compute_layer_refusal_metrics",
    "run_refusal_direction_experiment",
]
