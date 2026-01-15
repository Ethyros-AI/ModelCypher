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

"""Experiment 3: Cross-Model Universality of Refusal Direction.

Tests whether alignment geometry is universal across model architectures.

Method:
    1. Extract refusal direction from multiple instruct models
    2. Compare layer-wise patterns of refusal strength and accuracy
    3. For models with same hidden dim, compute direct cosine similarity
    4. Test cross-model transfer where possible

Hypothesis:
    Aligned refusal directions have similar patterns across architectures,
    supporting the Platonic Representation Hypothesis.

Usage:
    from modelcypher.experiments.cross_model_universality import (
        run_cross_model_experiment
    )

    result = run_cross_model_experiment(
        model_paths=["/path/to/model1", "/path/to/model2"],
        output_path="results/cross_model.json",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirectionDetector,
)
from modelcypher.experiments.refusal_direction import (
    collect_activations_by_layer,
    compute_projections,
)
from modelcypher.experiments.utils import load_harmful_prompts, load_harmless_prompts

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ModelRefusalProfile:
    """Refusal direction profile for a single model."""

    model_path: str
    model_name: str
    hidden_size: int
    num_layers: int
    best_layer: int
    best_accuracy: float
    best_separation: float
    layer_accuracies: list[float]
    layer_separations: list[float]
    layer_strengths: list[float]


@dataclass
class CrossModelComparison:
    """Comparison between two models' refusal profiles."""

    model_a: str
    model_b: str
    accuracy_correlation: float  # Correlation of layer-wise accuracies
    separation_correlation: float  # Correlation of layer-wise separations
    strength_correlation: float  # Correlation of layer-wise strengths
    cosine_similarity: float | None  # Direct cosine if same hidden_dim


@dataclass
class CrossModelResult:
    """Complete result of cross-model universality experiment."""

    model_profiles: list[ModelRefusalProfile]
    pairwise_comparisons: list[CrossModelComparison]
    aggregate_metrics: dict[str, float]


def compute_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation between two lists.

    Handles lists of different lengths by truncating to minimum length.
    """
    backend = get_default_backend()

    # Truncate to minimum length
    min_len = min(len(x), len(y))
    if min_len < 2:
        return 0.0

    x_arr = backend.array(x[:min_len])
    y_arr = backend.array(y[:min_len])

    # Compute means
    x_mean = backend.mean(x_arr)
    y_mean = backend.mean(y_arr)
    backend.eval(x_mean, y_mean)

    # Compute deviations
    x_dev = x_arr - x_mean
    y_dev = y_arr - y_mean

    # Compute correlation
    numerator = backend.sum(x_dev * y_dev)
    x_std = backend.sqrt(backend.sum(x_dev * x_dev))
    y_std = backend.sqrt(backend.sum(y_dev * y_dev))

    backend.eval(numerator, x_std, y_std)

    eps = 1e-8
    denom = float(backend.to_scalar(x_std)) * float(backend.to_scalar(y_std))
    if denom < eps:
        return 0.0

    return float(backend.to_scalar(numerator)) / denom


def compute_cosine_similarity_directions(
    dir_a: "Array",
    dir_b: "Array",
    backend: "Backend",
) -> float:
    """Compute cosine similarity between two direction vectors."""
    dot = backend.sum(dir_a * dir_b)
    norm_a = backend.sqrt(backend.sum(dir_a * dir_a))
    norm_b = backend.sqrt(backend.sum(dir_b * dir_b))
    backend.eval(dot, norm_a, norm_b)

    eps = 1e-8
    denom = float(backend.to_scalar(norm_a)) * float(backend.to_scalar(norm_b))
    if denom < eps:
        return 0.0

    return float(backend.to_scalar(dot)) / denom


def extract_model_profile(
    model_path: str,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    backend: "Backend",
) -> tuple[ModelRefusalProfile, dict[int, Any]]:
    """Extract refusal profile for a single model.

    Returns:
        Tuple of (ModelRefusalProfile, dict of layer_idx -> refusal_direction)
    """
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    model_name = Path(model_path).name
    logger.info("Extracting refusal profile for %s", model_name)

    # Load model
    model_loader = MLXModelLoader()
    model, tokenizer = model_loader.load_model_for_training(str(model_path))

    # Get activation provider
    activation_provider = MLXActivationProvider()

    # Collect activations
    harmful_by_layer = collect_activations_by_layer(
        model, tokenizer, harmful_prompts, activation_provider
    )
    harmless_by_layer = collect_activations_by_layer(
        model, tokenizer, harmless_prompts, activation_provider
    )

    layers = sorted(harmful_by_layer.keys())
    num_layers = len(layers)

    # Get hidden size from first activation
    first_layer = layers[0]
    hidden_size = int(backend.shape(harmful_by_layer[first_layer][0])[0])

    # Compute refusal direction at each layer
    layer_accuracies = []
    layer_separations = []
    layer_strengths = []
    refusal_directions: dict[int, Any] = {}

    for layer_idx in layers:
        # Stack activations
        harmful_acts = backend.stack(harmful_by_layer[layer_idx], axis=0)
        harmless_acts = backend.stack(harmless_by_layer[layer_idx], axis=0)
        backend.eval(harmful_acts, harmless_acts)

        # Compute refusal direction
        refusal_dir = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_acts,
            harmless_activations=harmless_acts,
            layer_index=layer_idx,
            model_id=model_name,
        )

        if refusal_dir is None:
            layer_accuracies.append(0.5)
            layer_separations.append(0.0)
            layer_strengths.append(0.0)
            continue

        refusal_directions[layer_idx] = refusal_dir.direction

        # Compute projections and accuracy
        direction = refusal_dir.direction
        backend.eval(direction)

        harmful_proj = compute_projections(harmful_acts, direction, backend)
        harmless_proj = compute_projections(harmless_acts, direction, backend)

        mean_harmful = float(backend.to_scalar(backend.mean(harmful_proj)))
        mean_harmless = float(backend.to_scalar(backend.mean(harmless_proj)))
        separation = mean_harmful - mean_harmless

        # Classification accuracy
        threshold = (mean_harmful + mean_harmless) / 2.0
        harmful_correct = backend.sum(
            backend.astype(harmful_proj > threshold, backend.dtype(harmful_proj))
        )
        harmless_correct = backend.sum(
            backend.astype(harmless_proj < threshold, backend.dtype(harmless_proj))
        )
        backend.eval(harmful_correct, harmless_correct)

        n_total = int(backend.shape(harmful_proj)[0]) + int(backend.shape(harmless_proj)[0])
        accuracy = (
            float(backend.to_scalar(harmful_correct))
            + float(backend.to_scalar(harmless_correct))
        ) / max(n_total, 1)

        layer_accuracies.append(accuracy)
        layer_separations.append(abs(separation))
        layer_strengths.append(refusal_dir.strength)

    # Find best layer
    best_idx = max(range(len(layer_accuracies)), key=lambda i: layer_accuracies[i])
    best_layer = layers[best_idx]
    best_accuracy = layer_accuracies[best_idx]
    best_separation = layer_separations[best_idx]

    profile = ModelRefusalProfile(
        model_path=model_path,
        model_name=model_name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        best_layer=best_layer,
        best_accuracy=best_accuracy,
        best_separation=best_separation,
        layer_accuracies=layer_accuracies,
        layer_separations=layer_separations,
        layer_strengths=layer_strengths,
    )

    return profile, refusal_directions


def run_cross_model_experiment(
    model_paths: list[str | Path],
    harmful_prompts: list[str] | None = None,
    harmless_prompts: list[str] | None = None,
    output_path: str | Path | None = None,
) -> CrossModelResult:
    """Run the cross-model universality experiment.

    Args:
        model_paths: List of paths to instruct models
        harmful_prompts: List of harmful prompts (uses default if None)
        harmless_prompts: List of harmless prompts (uses default if None)
        output_path: Path to save results JSON (optional)

    Returns:
        CrossModelResult with full analysis
    """
    backend = get_default_backend()

    # Load prompts
    if harmful_prompts is None:
        harmful_prompts = load_harmful_prompts()
    if harmless_prompts is None:
        harmless_prompts = load_harmless_prompts()

    if not harmful_prompts or not harmless_prompts:
        raise ValueError("No prompts available. Check datasets directory.")

    logger.info("Running cross-model experiment with %d models", len(model_paths))

    # Extract profiles for each model
    profiles: list[ModelRefusalProfile] = []
    all_directions: dict[str, dict[int, Any]] = {}

    for model_path in model_paths:
        try:
            profile, directions = extract_model_profile(
                str(model_path), harmful_prompts, harmless_prompts, backend
            )
            profiles.append(profile)
            all_directions[profile.model_name] = directions

            logger.info(
                "  %s: hidden=%d, layers=%d, best_acc=%.1f%% (layer %d)",
                profile.model_name,
                profile.hidden_size,
                profile.num_layers,
                profile.best_accuracy * 100,
                profile.best_layer,
            )
        except Exception as e:
            logger.warning("Failed to process %s: %s", model_path, e)
            continue

    # Compute pairwise comparisons
    comparisons: list[CrossModelComparison] = []

    for i, profile_a in enumerate(profiles):
        for j, profile_b in enumerate(profiles):
            if j <= i:
                continue

            # Compute correlations of layer-wise metrics
            acc_corr = compute_correlation(
                profile_a.layer_accuracies, profile_b.layer_accuracies
            )
            sep_corr = compute_correlation(
                profile_a.layer_separations, profile_b.layer_separations
            )
            str_corr = compute_correlation(
                profile_a.layer_strengths, profile_b.layer_strengths
            )

            # Compute direct cosine if same hidden dimension
            cosine_sim = None
            if profile_a.hidden_size == profile_b.hidden_size:
                # Compare best layer directions
                dirs_a = all_directions.get(profile_a.model_name, {})
                dirs_b = all_directions.get(profile_b.model_name, {})

                # Find common layer (use best from each, or first common)
                common_layers = set(dirs_a.keys()) & set(dirs_b.keys())
                if common_layers:
                    # Use the layer that's best for model_a
                    best_common = min(
                        common_layers,
                        key=lambda l: abs(l - profile_a.best_layer)
                    )
                    if best_common in dirs_a and best_common in dirs_b:
                        cosine_sim = compute_cosine_similarity_directions(
                            dirs_a[best_common], dirs_b[best_common], backend
                        )

            comparison = CrossModelComparison(
                model_a=profile_a.model_name,
                model_b=profile_b.model_name,
                accuracy_correlation=acc_corr,
                separation_correlation=sep_corr,
                strength_correlation=str_corr,
                cosine_similarity=cosine_sim,
            )
            comparisons.append(comparison)

            logger.info(
                "  %s vs %s: acc_corr=%.3f, sep_corr=%.3f, cos_sim=%s",
                profile_a.model_name[:20],
                profile_b.model_name[:20],
                acc_corr,
                sep_corr,
                f"{cosine_sim:.3f}" if cosine_sim is not None else "N/A",
            )

    # Compute aggregate metrics
    if comparisons:
        aggregate_metrics = {
            "num_models": len(profiles),
            "mean_accuracy_correlation": sum(c.accuracy_correlation for c in comparisons)
            / len(comparisons),
            "mean_separation_correlation": sum(c.separation_correlation for c in comparisons)
            / len(comparisons),
            "mean_strength_correlation": sum(c.strength_correlation for c in comparisons)
            / len(comparisons),
            "mean_best_accuracy": sum(p.best_accuracy for p in profiles) / len(profiles),
        }

        # Add mean cosine if available
        cosine_values = [c.cosine_similarity for c in comparisons if c.cosine_similarity is not None]
        if cosine_values:
            aggregate_metrics["mean_cosine_similarity"] = sum(cosine_values) / len(cosine_values)
    else:
        aggregate_metrics = {"num_models": len(profiles)}

    result = CrossModelResult(
        model_profiles=profiles,
        pairwise_comparisons=comparisons,
        aggregate_metrics=aggregate_metrics,
    )

    # Save results
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "model_profiles": [
                {
                    "model_path": p.model_path,
                    "model_name": p.model_name,
                    "hidden_size": p.hidden_size,
                    "num_layers": p.num_layers,
                    "best_layer": p.best_layer,
                    "best_accuracy": p.best_accuracy,
                    "best_separation": p.best_separation,
                    "layer_accuracies": p.layer_accuracies,
                    "layer_separations": p.layer_separations,
                    "layer_strengths": p.layer_strengths,
                }
                for p in profiles
            ],
            "pairwise_comparisons": [
                {
                    "model_a": c.model_a,
                    "model_b": c.model_b,
                    "accuracy_correlation": c.accuracy_correlation,
                    "separation_correlation": c.separation_correlation,
                    "strength_correlation": c.strength_correlation,
                    "cosine_similarity": c.cosine_similarity,
                }
                for c in comparisons
            ],
            "aggregate_metrics": aggregate_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("CROSS-MODEL UNIVERSALITY SUMMARY")
    logger.info("=" * 60)
    logger.info("Models analyzed: %d", len(profiles))
    for p in profiles:
        logger.info("  %s: best_acc=%.1f%% (layer %d)", p.model_name, p.best_accuracy * 100, p.best_layer)
    logger.info("-" * 60)
    logger.info("Mean accuracy correlation: %.3f", aggregate_metrics.get("mean_accuracy_correlation", 0))
    logger.info("Mean separation correlation: %.3f", aggregate_metrics.get("mean_separation_correlation", 0))
    logger.info("Mean strength correlation: %.3f", aggregate_metrics.get("mean_strength_correlation", 0))
    if "mean_cosine_similarity" in aggregate_metrics:
        logger.info("Mean cosine similarity: %.3f", aggregate_metrics["mean_cosine_similarity"])
    logger.info("=" * 60)

    return result


__all__ = [
    "CrossModelComparison",
    "CrossModelResult",
    "ModelRefusalProfile",
    "run_cross_model_experiment",
]
