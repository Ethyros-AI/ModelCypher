# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Experiment 6: Geometric Guardrails.

Tests mathematical guardrails that detect and steer activations
when they leave the alignment boundary during inference.

Key Hypothesis:
    - Harmful and jailbreak prompts will trigger boundary violations
    - Harmless prompts will stay within boundary (low false positive rate)
    - Steering can recover safe behavior for violating activations
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_boundary import (
    AlignmentBoundary,
    BoundaryCheckResult,
    BoundaryViolationType,
    check_boundary,
    compute_alignment_boundary,
    steer_to_boundary,
)
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirectionDetector,
)
from modelcypher.experiments.refusal_direction import collect_activations_by_layer
from modelcypher.experiments.utils import (
    load_harmful_prompts,
    load_harmless_prompts,
    load_jailbreak_prompts,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GuardrailMetrics:
    """Detection metrics for geometric guardrails."""

    # Detection rates
    harmful_violation_rate: float  # True positives on harmful
    harmless_violation_rate: float  # False positives on harmless
    jailbreak_violation_rate: float  # True positives on jailbreaks

    # Precision/recall for harmful detection
    precision: float
    recall: float
    f1_score: float

    # Steering effectiveness
    steering_recovery_rate: float  # % of violations fixed by steering


@dataclass
class GuardrailResult:
    """Full results from geometric guardrails experiment."""

    model_path: str
    detection_layer: int

    # Boundary parameters
    refusal_threshold: float
    safe_radius: float

    # Sample counts
    num_harmless: int
    num_harmful: int
    num_jailbreak: int

    # Metrics
    metrics: GuardrailMetrics

    # Detailed results per category
    harmless_results: list[dict]
    harmful_results: list[dict]
    jailbreak_results: list[dict]

    # Aggregate metrics for JSON serialization
    aggregate_metrics: dict


def run_geometric_guardrails(
    model_path: str,
    detection_layer: int | None = None,
    output_path: Path | None = None,
    backend: "Backend | None" = None,
) -> GuardrailResult:
    """Run geometric guardrails experiment.

    Args:
        model_path: Path to instruct model
        detection_layer: Layer to use for detection (auto-detect if None)
        output_path: Where to save results
        backend: Backend for tensor operations

    Returns:
        GuardrailResult with detection metrics
    """
    from modelcypher.adapters.activation_provider import ActivationProvider
    from modelcypher.adapters.model_loader import ModelLoader

    b = backend or get_default_backend()

    logger.info("Starting geometric guardrails experiment")
    logger.info("Model: %s", model_path)

    # Load prompts
    harmful_prompts = load_harmful_prompts()
    harmless_prompts = load_harmless_prompts()
    jailbreak_prompts = load_jailbreak_prompts()

    logger.info(
        "Loaded prompts: %d harmful, %d harmless, %d jailbreak",
        len(harmful_prompts),
        len(harmless_prompts),
        len(jailbreak_prompts),
    )

    if not harmful_prompts or not harmless_prompts:
        raise ValueError("Missing harmful or harmless prompts. Check datasets directory.")

    # Load model
    logger.info("Loading model from %s", model_path)
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model_for_training(str(model_path))

    model_id = Path(model_path).name
    activation_provider = ActivationProvider()

    # Split data into training (for boundary) and test sets
    n_train = min(20, len(harmful_prompts) // 2, len(harmless_prompts) // 2)
    train_harmful = harmful_prompts[:n_train]
    train_harmless = harmless_prompts[:n_train]
    test_harmful = harmful_prompts[n_train:]
    test_harmless = harmless_prompts[n_train:]

    logger.info("Training set: %d harmful, %d harmless", len(train_harmful), len(train_harmless))
    logger.info("Test set: %d harmful, %d harmless, %d jailbreak",
                len(test_harmful), len(test_harmless), len(jailbreak_prompts))

    # Collect activations for training set
    logger.info("Collecting training activations...")
    train_harmful_by_layer = collect_activations_by_layer(
        model, tokenizer, train_harmful, activation_provider
    )
    train_harmless_by_layer = collect_activations_by_layer(
        model, tokenizer, train_harmless, activation_provider
    )

    layers = sorted(train_harmful_by_layer.keys())

    # Auto-select detection layer by maximizing separation margins
    if detection_layer is None:
        best_layer = layers[0]
        best_score = float("-inf")

        for layer_idx in layers:
            h_acts = b.stack(train_harmless_by_layer[layer_idx], axis=0)
            f_acts = b.stack(train_harmful_by_layer[layer_idx], axis=0)
            b.eval(h_acts, f_acts)

            # Compute direction
            h_mean = b.mean(h_acts, axis=0)
            f_mean = b.mean(f_acts, axis=0)
            direction = f_mean - h_mean
            norm = b.sqrt(b.sum(direction * direction))
            direction = direction / norm
            b.eval(direction)

            # Compute projections and distances
            h_projs = b.sum(h_acts * direction, axis=1)
            f_projs = b.sum(f_acts * direction, axis=1)
            centered_h = h_acts - h_mean
            centered_f = f_acts - h_mean
            h_dists = b.sqrt(b.sum(centered_h * centered_h, axis=1))
            f_dists = b.sqrt(b.sum(centered_f * centered_f, axis=1))
            b.eval(h_projs, f_projs, h_dists, f_dists)

            # Separation margins (positive = separable)
            min_h_proj = float(b.to_scalar(b.min(h_projs)))
            max_f_proj = float(b.to_scalar(b.max(f_projs)))
            max_h_dist = float(b.to_scalar(b.max(h_dists)))
            min_f_dist = float(b.to_scalar(b.min(f_dists)))

            proj_margin = min_h_proj - max_f_proj
            dist_margin = min_f_dist - max_h_dist
            score = min(proj_margin, dist_margin)

            if score > best_score:
                best_score = score
                best_layer = layer_idx

        detection_layer = best_layer
        logger.info(
            "Layer selection: max separation score %.4f at layer %d",
            best_score, detection_layer
        )

    logger.info("Using detection layer: %d / %d", detection_layer, len(layers))

    # Stack training activations
    train_harmful_acts = b.stack(train_harmful_by_layer[detection_layer], axis=0)
    train_harmless_acts = b.stack(train_harmless_by_layer[detection_layer], axis=0)
    b.eval(train_harmful_acts, train_harmless_acts)

    # Compute refusal direction
    logger.info("Computing refusal direction...")
    refusal_result = RefusalDirectionDetector.compute_direction(
        harmful_activations=train_harmful_acts,
        harmless_activations=train_harmless_acts,
        layer_index=detection_layer,
        model_id=model_id,
    )

    if refusal_result is None:
        raise RuntimeError("Failed to compute refusal direction")

    refusal_direction = refusal_result.direction
    b.eval(refusal_direction)

    # Compute tight alignment boundary from harmless activations
    logger.info("Computing alignment boundary from safe activations...")
    boundary = compute_alignment_boundary(
        refusal_direction=refusal_direction,
        safe_activations=train_harmless_acts,
        layer_index=detection_layer,
        backend=b,
    )

    logger.info(
        "Boundary: refusal_threshold=%.4f, safe_radius=%.4f",
        boundary.refusal_threshold,
        boundary.safe_radius,
    )

    # Collect test activations
    logger.info("Collecting test activations...")
    test_harmful_by_layer = collect_activations_by_layer(
        model, tokenizer, test_harmful, activation_provider
    ) if test_harmful else {}
    test_harmless_by_layer = collect_activations_by_layer(
        model, tokenizer, test_harmless, activation_provider
    ) if test_harmless else {}
    jailbreak_by_layer = collect_activations_by_layer(
        model, tokenizer, jailbreak_prompts, activation_provider
    ) if jailbreak_prompts else {}

    # Function to check a category
    def check_category(prompts: list[str], acts_by_layer: dict, category: str) -> tuple[list[dict], float]:
        if not prompts or detection_layer not in acts_by_layer:
            return [], 0.0

        results = []
        violations = 0

        for i, prompt in enumerate(prompts):
            act = acts_by_layer[detection_layer][i]
            check_result = check_boundary(act, boundary, backend=b)

            # Test steering effectiveness
            if not check_result.is_within_boundary:
                violations += 1
                steered = steer_to_boundary(act, boundary, backend=b)
                steered_result = check_boundary(steered, boundary, backend=b)
                steering_fixed = steered_result.is_within_boundary
            else:
                steering_fixed = None

            results.append({
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "is_violation": not check_result.is_within_boundary,
                "violation_type": check_result.violation_type.value,
                "refusal_projection": check_result.refusal_projection,
                "distance_to_centroid": check_result.distance_to_centroid,
                "refusal_margin": check_result.refusal_margin,
                "distance_margin": check_result.distance_margin,
                "steering_fixed": steering_fixed,
            })

        violation_rate = violations / max(len(prompts), 1)
        return results, violation_rate

    logger.info("Checking harmful prompts...")
    harmful_results, harmful_violation_rate = check_category(
        test_harmful, test_harmful_by_layer, "harmful"
    )

    logger.info("Checking harmless prompts...")
    harmless_results, harmless_violation_rate = check_category(
        test_harmless, test_harmless_by_layer, "harmless"
    )

    logger.info("Checking jailbreak prompts...")
    jailbreak_results, jailbreak_violation_rate = check_category(
        jailbreak_prompts, jailbreak_by_layer, "jailbreak"
    )

    # Compute metrics
    harmful_violations = sum(1 for r in harmful_results if r["is_violation"])
    jailbreak_violations = sum(1 for r in jailbreak_results if r["is_violation"])
    harmless_violations = sum(1 for r in harmless_results if r["is_violation"])

    true_positives = harmful_violations + jailbreak_violations
    false_positives = harmless_violations
    true_negatives = len(harmless_results) - harmless_violations
    false_negatives = (len(harmful_results) - harmful_violations) + (len(jailbreak_results) - jailbreak_violations)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

    # Steering effectiveness
    all_violations = [r for r in harmful_results + jailbreak_results if r["is_violation"]]
    steering_fixed = sum(1 for r in all_violations if r["steering_fixed"])
    steering_recovery_rate = steering_fixed / max(len(all_violations), 1)

    metrics = GuardrailMetrics(
        harmful_violation_rate=harmful_violation_rate,
        harmless_violation_rate=harmless_violation_rate,
        jailbreak_violation_rate=jailbreak_violation_rate,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        steering_recovery_rate=steering_recovery_rate,
    )

    aggregate_metrics = {
        "harmful_violation_rate": harmful_violation_rate,
        "harmless_violation_rate": harmless_violation_rate,
        "jailbreak_violation_rate": jailbreak_violation_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "steering_recovery_rate": steering_recovery_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }

    result = GuardrailResult(
        model_path=model_path,
        detection_layer=detection_layer,
        refusal_threshold=boundary.refusal_threshold,
        safe_radius=boundary.safe_radius,
        num_harmless=len(harmless_results),
        num_harmful=len(harmful_results),
        num_jailbreak=len(jailbreak_results),
        metrics=metrics,
        harmless_results=harmless_results,
        harmful_results=harmful_results,
        jailbreak_results=jailbreak_results,
        aggregate_metrics=aggregate_metrics,
    )

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model_path": result.model_path,
                    "detection_layer": result.detection_layer,
                    "refusal_threshold": result.refusal_threshold,
                    "safe_radius": result.safe_radius,
                    "num_harmless": result.num_harmless,
                    "num_harmful": result.num_harmful,
                    "num_jailbreak": result.num_jailbreak,
                    "metrics": asdict(result.metrics),
                    "aggregate_metrics": result.aggregate_metrics,
                    "harmless_results": result.harmless_results,
                    "harmful_results": result.harmful_results,
                    "jailbreak_results": result.jailbreak_results,
                },
                f,
                indent=2,
            )
        logger.info("Results saved to %s", output_path)

    return result


__all__ = [
    "GuardrailMetrics",
    "GuardrailResult",
    "run_geometric_guardrails",
]
