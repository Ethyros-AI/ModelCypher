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

"""Experiment 4: Geometric Jailbreak Detection.

Detects jailbreak attempts from activation geometry alone, before output is generated.

Method:
    1. Collect activations for normal, harmful, and jailbreak prompts
    2. Compute geometric features:
       - Projection onto refusal direction
       - Distance from safe centroid
    3. Train simple classifier on geometric features
    4. Evaluate detection accuracy

Hypothesis:
    Jailbreaks geometrically suppress the refusal direction (low projection onto r).
    We can detect this before the model generates output.

Usage:
    from modelcypher.experiments.jailbreak_detection import run_jailbreak_detection

    result = run_jailbreak_detection(
        model_path="/path/to/instruct/model",
        output_path="results/jailbreak_detection.json",
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
from modelcypher.experiments.refusal_direction import collect_activations_by_layer
from modelcypher.experiments.utils import (
    DATASETS_DIR,
    load_harmful_prompts,
    load_harmless_prompts,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def load_jailbreak_prompts() -> list[str]:
    """Load jailbreak prompts dataset."""
    path = DATASETS_DIR / "jailbreak_prompts.json"
    if not path.exists():
        logger.warning("Jailbreak prompts dataset not found at %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("prompts", [])


@dataclass
class PromptGeometry:
    """Geometric features for a single prompt."""

    prompt: str
    prompt_type: str  # "harmless", "harmful", "jailbreak"
    refusal_projection: float
    distance_to_centroid: float
    layer_index: int


@dataclass
class DetectionMetrics:
    """Detection metrics for a classifier."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass
class JailbreakDetectionResult:
    """Complete result of jailbreak detection experiment."""

    model_path: str
    num_harmless: int
    num_harmful: int
    num_jailbreak: int
    detection_layer: int
    prompt_geometries: list[PromptGeometry]
    mean_harmless_projection: float
    mean_harmful_projection: float
    mean_jailbreak_projection: float
    detection_metrics: DetectionMetrics
    aggregate_metrics: dict[str, float]


def compute_prompt_geometry(
    activation: "Array",
    refusal_direction: "Array",
    safe_centroid: "Array",
    prompt: str,
    prompt_type: str,
    layer_index: int,
    backend: "Backend",
) -> PromptGeometry:
    """Compute geometric features for a single prompt."""
    # Projection onto refusal direction
    projection = float(backend.to_scalar(backend.sum(activation * refusal_direction)))

    # Distance from safe centroid
    diff = activation - safe_centroid
    distance = float(backend.to_scalar(backend.sqrt(backend.sum(diff * diff))))

    return PromptGeometry(
        prompt=prompt[:100],  # Truncate for storage
        prompt_type=prompt_type,
        refusal_projection=projection,
        distance_to_centroid=distance,
        layer_index=layer_index,
    )


def compute_detection_metrics(
    predictions: list[bool],
    labels: list[bool],
) -> DetectionMetrics:
    """Compute classification metrics.

    Args:
        predictions: Predicted labels (True = jailbreak detected)
        labels: True labels (True = is jailbreak)
    """
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

    total = len(predictions)
    accuracy = (tp + tn) / max(total, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return DetectionMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def run_jailbreak_detection(
    model_path: str | Path,
    harmless_prompts: list[str] | None = None,
    harmful_prompts: list[str] | None = None,
    jailbreak_prompts: list[str] | None = None,
    detection_layer: int | None = None,
    output_path: str | Path | None = None,
) -> JailbreakDetectionResult:
    """Run the jailbreak detection experiment.

    Args:
        model_path: Path to instruct model
        harmless_prompts: List of harmless prompts (uses default if None)
        harmful_prompts: List of harmful prompts (uses default if None)
        jailbreak_prompts: List of jailbreak prompts (uses default if None)
        detection_layer: Layer to use for detection (auto-select if None)
        output_path: Path to save results JSON (optional)

    Returns:
        JailbreakDetectionResult with full analysis
    """
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    backend = get_default_backend()

    # Load prompts
    if harmless_prompts is None:
        harmless_prompts = load_harmless_prompts()
    if harmful_prompts is None:
        harmful_prompts = load_harmful_prompts()
    if jailbreak_prompts is None:
        jailbreak_prompts = load_jailbreak_prompts()

    if not harmless_prompts or not harmful_prompts or not jailbreak_prompts:
        raise ValueError("Missing prompts. Check datasets directory.")

    logger.info("Loading model from %s", model_path)
    model_loader = MLXModelLoader()
    model, tokenizer = model_loader.load_model_for_training(str(model_path))

    model_id = Path(model_path).name
    activation_provider = MLXActivationProvider()

    logger.info(
        "Collecting activations: %d harmless, %d harmful, %d jailbreak",
        len(harmless_prompts),
        len(harmful_prompts),
        len(jailbreak_prompts),
    )

    # Collect activations for each category
    harmless_by_layer = collect_activations_by_layer(
        model, tokenizer, harmless_prompts, activation_provider
    )
    harmful_by_layer = collect_activations_by_layer(
        model, tokenizer, harmful_prompts, activation_provider
    )
    jailbreak_by_layer = collect_activations_by_layer(
        model, tokenizer, jailbreak_prompts, activation_provider
    )

    layers = sorted(harmless_by_layer.keys())

    # Auto-select detection layer by finding layer with maximum separation
    if detection_layer is None:
        best_layer = layers[0]
        best_accuracy = 0.0

        for layer_idx in layers:
            h_acts = backend.stack(harmless_by_layer[layer_idx], axis=0)
            f_acts = backend.stack(harmful_by_layer[layer_idx], axis=0)
            backend.eval(h_acts, f_acts)

            # Compute refusal direction for this layer
            h_mean = backend.mean(h_acts, axis=0)
            f_mean = backend.mean(f_acts, axis=0)
            direction = f_mean - h_mean
            norm = backend.sqrt(backend.sum(direction * direction))
            direction = direction / norm
            backend.eval(direction)

            # Compute projections for each prompt
            h_projs = backend.sum(h_acts * direction, axis=1)
            f_projs = backend.sum(f_acts * direction, axis=1)
            backend.eval(h_projs, f_projs)

            # Convert to lists for threshold computation
            h_projs_list = [
                float(backend.to_scalar(backend.take(h_projs, backend.array([i]), axis=0)))
                for i in range(len(harmless_prompts))
            ]
            f_projs_list = [
                float(backend.to_scalar(backend.take(f_projs, backend.array([i]), axis=0)))
                for i in range(len(harmful_prompts))
            ]

            # Compute threshold (95th percentile of harmless)
            sorted_harmless = sorted(h_projs_list)
            threshold_idx = int(0.95 * len(sorted_harmless))
            threshold_idx = min(threshold_idx, len(sorted_harmless) - 1)
            threshold = sorted_harmless[threshold_idx]

            # Compute classification accuracy
            # Positive = harmful (above threshold), Negative = harmless (below threshold)
            tp = sum(1 for p in f_projs_list if p > threshold)
            tn = sum(1 for p in h_projs_list if p <= threshold)
            accuracy = (tp + tn) / (len(f_projs_list) + len(h_projs_list))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_idx

        detection_layer = best_layer
        logger.info("Layer selection: max accuracy %.4f at layer %d", best_accuracy, detection_layer)

    logger.info("Using layer %d for detection", detection_layer)

    # Stack activations for detection layer
    harmless_acts = backend.stack(harmless_by_layer[detection_layer], axis=0)
    harmful_acts = backend.stack(harmful_by_layer[detection_layer], axis=0)
    jailbreak_acts = backend.stack(jailbreak_by_layer[detection_layer], axis=0)
    backend.eval(harmless_acts, harmful_acts, jailbreak_acts)

    # Compute refusal direction using harmful vs harmless
    logger.info("Computing refusal direction...")
    refusal_dir_result = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful_acts,
        harmless_activations=harmless_acts,
        layer_index=detection_layer,
        model_id=model_id,
    )

    if refusal_dir_result is None:
        raise RuntimeError("Failed to compute refusal direction")

    refusal_direction = refusal_dir_result.direction
    backend.eval(refusal_direction)

    # Compute safe centroid (from harmless prompts)
    safe_centroid = backend.mean(harmless_acts, axis=0)
    backend.eval(safe_centroid)

    # Compute geometry for all prompts
    logger.info("Computing prompt geometries...")
    prompt_geometries: list[PromptGeometry] = []

    # Process harmless prompts
    for i, prompt in enumerate(harmless_prompts):
        act = harmless_by_layer[detection_layer][i]
        geom = compute_prompt_geometry(
            act, refusal_direction, safe_centroid, prompt, "harmless", detection_layer, backend
        )
        prompt_geometries.append(geom)

    # Process harmful prompts
    for i, prompt in enumerate(harmful_prompts):
        act = harmful_by_layer[detection_layer][i]
        geom = compute_prompt_geometry(
            act, refusal_direction, safe_centroid, prompt, "harmful", detection_layer, backend
        )
        prompt_geometries.append(geom)

    # Process jailbreak prompts
    for i, prompt in enumerate(jailbreak_prompts):
        act = jailbreak_by_layer[detection_layer][i]
        geom = compute_prompt_geometry(
            act, refusal_direction, safe_centroid, prompt, "jailbreak", detection_layer, backend
        )
        prompt_geometries.append(geom)

    # Compute mean projections by category
    harmless_projs = [g.refusal_projection for g in prompt_geometries if g.prompt_type == "harmless"]
    harmful_projs = [g.refusal_projection for g in prompt_geometries if g.prompt_type == "harmful"]
    jailbreak_projs = [g.refusal_projection for g in prompt_geometries if g.prompt_type == "jailbreak"]

    mean_harmless = sum(harmless_projs) / max(len(harmless_projs), 1)
    mean_harmful = sum(harmful_projs) / max(len(harmful_projs), 1)
    mean_jailbreak = sum(jailbreak_projs) / max(len(jailbreak_projs), 1)

    logger.info("Mean projections - Harmless: %.4f, Harmful: %.4f, Jailbreak: %.4f",
                mean_harmless, mean_harmful, mean_jailbreak)

    # Threshold derived from harmless distribution
    # Use the 95th percentile of harmless projections as threshold
    # This gives us a controlled 5% false positive rate by construction
    sorted_harmless = sorted(harmless_projs)
    threshold_idx = int(0.95 * len(sorted_harmless))
    threshold_idx = min(threshold_idx, len(sorted_harmless) - 1)
    threshold_jailbreak = sorted_harmless[threshold_idx]

    # Also compute threshold for harmful (95th percentile)
    threshold_harmful = threshold_jailbreak  # Same threshold for consistency

    logger.info("Detection threshold (95th percentile of harmless): %.4f", threshold_jailbreak)

    # Classify: projection > threshold => potential harmful/jailbreak
    predictions: list[bool] = []
    labels: list[bool] = []

    for geom in prompt_geometries:
        # Predict: HIGH projection suggests harmful content
        # Use a threshold that captures both harmful AND jailbreaks (which are in-between)
        pred = geom.refusal_projection > threshold_jailbreak
        predictions.append(pred)

        # True label: is this something we should flag? (harmful or jailbreak)
        is_flaggable = geom.prompt_type in ("harmful", "jailbreak")
        labels.append(is_flaggable)

    detection_metrics = compute_detection_metrics(predictions, labels)

    logger.info("Detection Results:")
    logger.info("  Accuracy: %.1f%%", detection_metrics.accuracy * 100)
    logger.info("  Precision: %.1f%%", detection_metrics.precision * 100)
    logger.info("  Recall: %.1f%%", detection_metrics.recall * 100)
    logger.info("  F1 Score: %.3f", detection_metrics.f1_score)

    # Also compute jailbreak-specific detection (jailbreaks vs harmless only)
    jailbreak_only_preds = []
    jailbreak_only_labels = []
    for geom in prompt_geometries:
        if geom.prompt_type in ("harmless", "jailbreak"):
            pred = geom.refusal_projection > threshold_jailbreak
            jailbreak_only_preds.append(pred)
            jailbreak_only_labels.append(geom.prompt_type == "jailbreak")

    jailbreak_metrics = compute_detection_metrics(jailbreak_only_preds, jailbreak_only_labels)

    # Aggregate metrics
    aggregate_metrics = {
        "mean_harmless_projection": mean_harmless,
        "mean_harmful_projection": mean_harmful,
        "mean_jailbreak_projection": mean_jailbreak,
        "detection_threshold_harmful": threshold_harmful,
        "detection_threshold_jailbreak": threshold_jailbreak,
        "overall_accuracy": detection_metrics.accuracy,
        "overall_precision": detection_metrics.precision,
        "overall_recall": detection_metrics.recall,
        "overall_f1": detection_metrics.f1_score,
        "jailbreak_only_accuracy": jailbreak_metrics.accuracy,
        "jailbreak_only_precision": jailbreak_metrics.precision,
        "jailbreak_only_recall": jailbreak_metrics.recall,
        "jailbreak_only_f1": jailbreak_metrics.f1_score,
        "jailbreak_suppression": mean_harmless - mean_jailbreak,  # How much jailbreaks suppress refusal
        "harmful_vs_harmless_separation": mean_harmful - mean_harmless,
    }

    result = JailbreakDetectionResult(
        model_path=str(model_path),
        num_harmless=len(harmless_prompts),
        num_harmful=len(harmful_prompts),
        num_jailbreak=len(jailbreak_prompts),
        detection_layer=detection_layer,
        prompt_geometries=prompt_geometries,
        mean_harmless_projection=mean_harmless,
        mean_harmful_projection=mean_harmful,
        mean_jailbreak_projection=mean_jailbreak,
        detection_metrics=detection_metrics,
        aggregate_metrics=aggregate_metrics,
    )

    # Save results
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "model_path": result.model_path,
            "num_harmless": result.num_harmless,
            "num_harmful": result.num_harmful,
            "num_jailbreak": result.num_jailbreak,
            "detection_layer": result.detection_layer,
            "mean_harmless_projection": result.mean_harmless_projection,
            "mean_harmful_projection": result.mean_harmful_projection,
            "mean_jailbreak_projection": result.mean_jailbreak_projection,
            "detection_metrics": {
                "accuracy": detection_metrics.accuracy,
                "precision": detection_metrics.precision,
                "recall": detection_metrics.recall,
                "f1_score": detection_metrics.f1_score,
                "true_positives": detection_metrics.true_positives,
                "false_positives": detection_metrics.false_positives,
                "true_negatives": detection_metrics.true_negatives,
                "false_negatives": detection_metrics.false_negatives,
            },
            "aggregate_metrics": aggregate_metrics,
            "prompt_geometries": [
                {
                    "prompt": g.prompt,
                    "prompt_type": g.prompt_type,
                    "refusal_projection": g.refusal_projection,
                    "distance_to_centroid": g.distance_to_centroid,
                }
                for g in prompt_geometries
            ],
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("JAILBREAK DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Model: %s", model_path)
    logger.info("Detection layer: %d", detection_layer)
    logger.info("Prompts: %d harmless, %d harmful, %d jailbreak",
                len(harmless_prompts), len(harmful_prompts), len(jailbreak_prompts))
    logger.info("-" * 60)
    logger.info("Mean Projections:")
    logger.info("  Harmless: %.4f", mean_harmless)
    logger.info("  Harmful: %.4f", mean_harmful)
    logger.info("  Jailbreak: %.4f", mean_jailbreak)
    logger.info("-" * 60)
    logger.info("Overall Detection (harmful + jailbreak vs harmless):")
    logger.info("  Accuracy: %.1f%%", detection_metrics.accuracy * 100)
    logger.info("  Precision: %.1f%%", detection_metrics.precision * 100)
    logger.info("  Recall: %.1f%%", detection_metrics.recall * 100)
    logger.info("  F1: %.3f", detection_metrics.f1_score)
    logger.info("-" * 60)
    logger.info("Jailbreak-Only Detection (jailbreak vs harmless):")
    logger.info("  Accuracy: %.1f%%", jailbreak_metrics.accuracy * 100)
    logger.info("  Precision: %.1f%%", jailbreak_metrics.precision * 100)
    logger.info("  Recall: %.1f%%", jailbreak_metrics.recall * 100)
    logger.info("  F1: %.3f", jailbreak_metrics.f1_score)
    logger.info("=" * 60)

    return result


__all__ = [
    "DetectionMetrics",
    "JailbreakDetectionResult",
    "PromptGeometry",
    "load_jailbreak_prompts",
    "run_jailbreak_detection",
]
