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

"""Experiment 5: Surgical Alignment Transfer.

Transfers alignment from instruct model to base model using geometric steering.

Method:
    1. Extract refusal direction and threshold from instruct model
    2. Compute threshold as 95th percentile of instruct's harmless projections
    3. On base model: count harmful prompts above threshold (baseline refusal rate)
    4. Steer base model's harmful activations by adding refusal direction
    5. Count harmful prompts above threshold after steering (transfer effectiveness)

Steering is applied only to harmful prompts. Transfer effectiveness is the
increase in harmful prompts crossing the refusal threshold.

Usage:
    from modelcypher.experiments.alignment_transfer import run_alignment_transfer

    result = run_alignment_transfer(
        instruct_model_path="/path/to/instruct",
        base_model_path="/path/to/base",
        output_path="results/alignment_transfer.json",
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
from modelcypher.experiments.utils import load_harmful_prompts, load_harmless_prompts

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    """Result of steering a single prompt."""

    prompt: str
    original_projection: float
    steered_projection: float
    projection_increase: float
    steering_strength: float


@dataclass
class AlignmentTransferResult:
    """Complete result of alignment transfer experiment."""

    instruct_model_path: str
    base_model_path: str
    steering_layer: int
    steering_strength: float
    num_harmful_prompts: int
    num_harmless_prompts: int
    # Refusal direction metrics
    refusal_direction_strength: float
    refusal_direction_explained_var: float
    # Threshold from instruct model
    refusal_threshold: float  # 95th percentile of instruct harmless projections
    # Instruct model rates (baseline for comparison)
    instruct_harmful_refusal_rate: float  # % harmful above threshold
    instruct_harmless_refusal_rate: float  # % harmless above threshold (false positives)
    # Base model before steering
    base_harmful_refusal_rate: float  # % harmful above threshold
    base_harmless_refusal_rate: float  # % harmless above threshold
    # Base model after steering (only harmful prompts steered)
    steered_harmful_refusal_rate: float  # % harmful above threshold after steering
    # Transfer effectiveness
    refusal_rate_increase: float  # steered - base harmful refusal rate
    transfer_effectiveness: float  # How close steered base is to instruct
    steering_results: list[SteeringResult]
    aggregate_metrics: dict[str, float]


def compute_steered_projection(
    activation: "Array",
    refusal_direction: "Array",
    steering_strength: float,
    backend: "Backend",
) -> tuple[float, float, "Array"]:
    """Apply steering and compute projections.

    Args:
        activation: Original activation [hidden_dim]
        refusal_direction: Refusal direction [hidden_dim]
        steering_strength: How much to add (multiplier on direction)
        backend: Backend

    Returns:
        Tuple of (original_projection, steered_projection, steered_activation)
    """
    # Original projection
    orig_proj = float(backend.to_scalar(backend.sum(activation * refusal_direction)))

    # Apply steering: add refusal direction scaled by strength
    steered = activation + steering_strength * refusal_direction
    backend.eval(steered)

    # Steered projection
    steered_proj = float(backend.to_scalar(backend.sum(steered * refusal_direction)))

    return orig_proj, steered_proj, steered


def run_alignment_transfer(
    instruct_model_path: str | Path,
    base_model_path: str | Path,
    harmful_prompts: list[str] | None = None,
    harmless_prompts: list[str] | None = None,
    steering_layer: int | None = None,
    steering_strength: float = 1.0,
    output_path: str | Path | None = None,
) -> AlignmentTransferResult:
    """Run the alignment transfer experiment.

    Args:
        instruct_model_path: Path to source instruct model (alignment donor)
        base_model_path: Path to target base model (alignment recipient)
        harmful_prompts: List of harmful prompts (uses default if None)
        harmless_prompts: List of harmless prompts (uses default if None)
        steering_layer: Layer to apply steering (auto-select if None)
        steering_strength: Multiplier for refusal direction (default 1.0)
        output_path: Path to save results JSON (optional)

    Returns:
        AlignmentTransferResult with full analysis
    """
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    backend = get_default_backend()

    # Load prompts
    if harmful_prompts is None:
        harmful_prompts = load_harmful_prompts()
    if harmless_prompts is None:
        harmless_prompts = load_harmless_prompts()

    # Use subset for faster experimentation
    harmful_prompts = harmful_prompts[:30]
    harmless_prompts = harmless_prompts[:30]

    if not harmful_prompts or not harmless_prompts:
        raise ValueError("Missing prompts. Check datasets directory.")

    model_loader = MLXModelLoader()
    activation_provider = MLXActivationProvider()

    # =========================================================================
    # STEP 1: Extract refusal direction from instruct model
    # =========================================================================
    logger.info("Loading instruct model from %s", instruct_model_path)
    instruct_model, instruct_tokenizer = model_loader.load_model_for_training(
        str(instruct_model_path)
    )

    logger.info("Collecting instruct model activations...")
    instruct_harmful_by_layer = collect_activations_by_layer(
        instruct_model, instruct_tokenizer, harmful_prompts, activation_provider
    )
    instruct_harmless_by_layer = collect_activations_by_layer(
        instruct_model, instruct_tokenizer, harmless_prompts, activation_provider
    )

    layers = sorted(instruct_harmful_by_layer.keys())

    # Auto-select steering layer (use layer with best separation)
    if steering_layer is None:
        best_accuracy = 0.0
        best_layer = layers[len(layers) // 2]
        for layer_idx in layers:
            harmful_acts = backend.stack(instruct_harmful_by_layer[layer_idx], axis=0)
            harmless_acts = backend.stack(instruct_harmless_by_layer[layer_idx], axis=0)
            backend.eval(harmful_acts, harmless_acts)

            refusal_dir = RefusalDirectionDetector.compute_direction(
                harmful_activations=harmful_acts,
                harmless_activations=harmless_acts,
                layer_index=layer_idx,
                model_id="instruct",
            )
            if refusal_dir is not None:
                harmful_projs = backend.sum(harmful_acts * refusal_dir.direction, axis=1)
                harmless_projs = backend.sum(harmless_acts * refusal_dir.direction, axis=1)
                backend.eval(harmful_projs, harmless_projs)

                # Convert to lists for threshold computation
                n_harmful = harmful_acts.shape[0]
                n_harmless = harmless_acts.shape[0]
                h_projs_list = [
                    float(backend.to_scalar(backend.take(harmless_projs, backend.array([i]), axis=0)))
                    for i in range(n_harmless)
                ]
                f_projs_list = [
                    float(backend.to_scalar(backend.take(harmful_projs, backend.array([i]), axis=0)))
                    for i in range(n_harmful)
                ]

                # Compute threshold (95th percentile of harmless)
                sorted_harmless = sorted(h_projs_list)
                threshold_idx = int(0.95 * len(sorted_harmless))
                threshold_idx = min(threshold_idx, len(sorted_harmless) - 1)
                threshold = sorted_harmless[threshold_idx]

                # Compute classification accuracy
                tp = sum(1 for p in f_projs_list if p > threshold)
                tn = sum(1 for p in h_projs_list if p <= threshold)
                accuracy = (tp + tn) / (n_harmful + n_harmless)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_layer = layer_idx

        steering_layer = best_layer
        logger.info("Layer selection: max accuracy %.4f at layer %d", best_accuracy, steering_layer)

    logger.info("Using steering layer: %d", steering_layer)

    # Compute refusal direction at steering layer
    instruct_harmful_acts = backend.stack(instruct_harmful_by_layer[steering_layer], axis=0)
    instruct_harmless_acts = backend.stack(instruct_harmless_by_layer[steering_layer], axis=0)
    backend.eval(instruct_harmful_acts, instruct_harmless_acts)

    refusal_direction_result = RefusalDirectionDetector.compute_direction(
        harmful_activations=instruct_harmful_acts,
        harmless_activations=instruct_harmless_acts,
        layer_index=steering_layer,
        model_id=Path(instruct_model_path).name,
    )

    if refusal_direction_result is None:
        raise RuntimeError("Failed to compute refusal direction from instruct model")

    refusal_direction = refusal_direction_result.direction
    backend.eval(refusal_direction)

    logger.info(
        "Refusal direction: strength=%.4f, explained_var=%.4f",
        refusal_direction_result.strength,
        refusal_direction_result.explained_variance,
    )

    # Compute instruct model projections
    instruct_harmful_projs = backend.sum(instruct_harmful_acts * refusal_direction, axis=1)
    instruct_harmless_projs = backend.sum(instruct_harmless_acts * refusal_direction, axis=1)
    backend.eval(instruct_harmful_projs, instruct_harmless_projs)

    # Convert to lists for threshold computation
    instruct_harmful_projs_list = [
        float(backend.to_scalar(backend.take(instruct_harmful_projs, backend.array([i]), axis=0)))
        for i in range(len(harmful_prompts))
    ]
    instruct_harmless_projs_list = [
        float(backend.to_scalar(backend.take(instruct_harmless_projs, backend.array([i]), axis=0)))
        for i in range(len(harmless_prompts))
    ]

    # Compute refusal threshold: 95th percentile of instruct harmless projections
    # Prompts above this threshold are in the "refusal zone"
    sorted_harmless = sorted(instruct_harmless_projs_list)
    threshold_idx = int(0.95 * len(sorted_harmless))
    threshold_idx = min(threshold_idx, len(sorted_harmless) - 1)
    refusal_threshold = sorted_harmless[threshold_idx]

    logger.info("Refusal threshold (95th percentile of instruct harmless): %.4f", refusal_threshold)

    # Compute instruct model refusal rates
    instruct_harmful_refusal_rate = sum(
        1 for p in instruct_harmful_projs_list if p > refusal_threshold
    ) / len(instruct_harmful_projs_list)
    instruct_harmless_refusal_rate = sum(
        1 for p in instruct_harmless_projs_list if p > refusal_threshold
    ) / len(instruct_harmless_projs_list)

    logger.info(
        "Instruct model: harmful refusal rate=%.2f%%, harmless refusal rate=%.2f%%",
        instruct_harmful_refusal_rate * 100,
        instruct_harmless_refusal_rate * 100,
    )

    # Free instruct model memory
    del instruct_model, instruct_tokenizer
    del instruct_harmful_by_layer, instruct_harmless_by_layer

    # =========================================================================
    # STEP 2: Load base model and compute baseline projections
    # =========================================================================
    logger.info("Loading base model from %s", base_model_path)
    base_model, base_tokenizer = model_loader.load_model_for_training(
        str(base_model_path)
    )

    logger.info("Collecting base model activations...")
    base_harmful_by_layer = collect_activations_by_layer(
        base_model, base_tokenizer, harmful_prompts, activation_provider
    )
    base_harmless_by_layer = collect_activations_by_layer(
        base_model, base_tokenizer, harmless_prompts, activation_provider
    )

    # Get base model activations at steering layer
    base_harmful_acts = backend.stack(base_harmful_by_layer[steering_layer], axis=0)
    base_harmless_acts = backend.stack(base_harmless_by_layer[steering_layer], axis=0)
    backend.eval(base_harmful_acts, base_harmless_acts)

    # Compute base model projections onto instruct's refusal direction
    base_harmful_projs = backend.sum(base_harmful_acts * refusal_direction, axis=1)
    base_harmless_projs = backend.sum(base_harmless_acts * refusal_direction, axis=1)
    backend.eval(base_harmful_projs, base_harmless_projs)

    # Convert to lists
    base_harmful_projs_list = [
        float(backend.to_scalar(backend.take(base_harmful_projs, backend.array([i]), axis=0)))
        for i in range(len(harmful_prompts))
    ]
    base_harmless_projs_list = [
        float(backend.to_scalar(backend.take(base_harmless_projs, backend.array([i]), axis=0)))
        for i in range(len(harmless_prompts))
    ]

    # Compute base model refusal rates using instruct's threshold
    base_harmful_refusal_rate = sum(
        1 for p in base_harmful_projs_list if p > refusal_threshold
    ) / len(base_harmful_projs_list)
    base_harmless_refusal_rate = sum(
        1 for p in base_harmless_projs_list if p > refusal_threshold
    ) / len(base_harmless_projs_list)

    logger.info(
        "Base model (before): harmful refusal rate=%.2f%%, harmless refusal rate=%.2f%%",
        base_harmful_refusal_rate * 100,
        base_harmless_refusal_rate * 100,
    )

    # =========================================================================
    # STEP 3: Apply steering to harmful prompts ONLY and measure effect
    # =========================================================================
    logger.info("Applying alignment steering to harmful prompts (strength=%.2f)...", steering_strength)

    steering_results: list[SteeringResult] = []

    # Steer harmful prompts (add refusal direction to push into refusal zone)
    steered_harmful_projs_list = []
    for i, prompt in enumerate(harmful_prompts):
        act = base_harmful_by_layer[steering_layer][i]
        orig_proj, steered_proj, _ = compute_steered_projection(
            act, refusal_direction, steering_strength, backend
        )
        steered_harmful_projs_list.append(steered_proj)

        steering_results.append(SteeringResult(
            prompt=prompt[:80],
            original_projection=orig_proj,
            steered_projection=steered_proj,
            projection_increase=steered_proj - orig_proj,
            steering_strength=steering_strength,
        ))

    # Compute steered harmful refusal rate
    steered_harmful_refusal_rate = sum(
        1 for p in steered_harmful_projs_list if p > refusal_threshold
    ) / len(steered_harmful_projs_list)

    logger.info(
        "Base model (after steering): harmful refusal rate=%.2f%%",
        steered_harmful_refusal_rate * 100,
    )

    # =========================================================================
    # STEP 4: Compute transfer effectiveness metrics
    # =========================================================================
    refusal_rate_increase = steered_harmful_refusal_rate - base_harmful_refusal_rate

    # Transfer effectiveness: how close is steered base's refusal rate to instruct's?
    # 1.0 = perfect transfer (steered base refuses harmful at same rate as instruct)
    # 0.0 = no transfer (steered same as base)
    if abs(instruct_harmful_refusal_rate - base_harmful_refusal_rate) > 1e-6:
        transfer_effectiveness = (steered_harmful_refusal_rate - base_harmful_refusal_rate) / (
            instruct_harmful_refusal_rate - base_harmful_refusal_rate
        )
        transfer_effectiveness = max(0.0, min(1.0, transfer_effectiveness))
    else:
        # If instruct and base have same refusal rate, check if steering changed anything
        transfer_effectiveness = 1.0 if abs(refusal_rate_increase) < 1e-6 else 0.0

    logger.info("Refusal rate increase: +%.1f%%", refusal_rate_increase * 100)
    logger.info("Transfer effectiveness: %.1f%%", transfer_effectiveness * 100)

    # Aggregate metrics
    aggregate_metrics = {
        "refusal_threshold": refusal_threshold,
        "instruct_harmful_refusal_rate": instruct_harmful_refusal_rate,
        "instruct_harmless_refusal_rate": instruct_harmless_refusal_rate,
        "base_harmful_refusal_rate": base_harmful_refusal_rate,
        "base_harmless_refusal_rate": base_harmless_refusal_rate,
        "steered_harmful_refusal_rate": steered_harmful_refusal_rate,
        "refusal_rate_increase": refusal_rate_increase,
        "transfer_effectiveness": transfer_effectiveness,
        "steering_strength": steering_strength,
        "steering_layer": steering_layer,
        "mean_projection_increase": sum(r.projection_increase for r in steering_results) / len(steering_results),
    }

    result = AlignmentTransferResult(
        instruct_model_path=str(instruct_model_path),
        base_model_path=str(base_model_path),
        steering_layer=steering_layer,
        steering_strength=steering_strength,
        num_harmful_prompts=len(harmful_prompts),
        num_harmless_prompts=len(harmless_prompts),
        refusal_direction_strength=refusal_direction_result.strength,
        refusal_direction_explained_var=refusal_direction_result.explained_variance,
        refusal_threshold=refusal_threshold,
        instruct_harmful_refusal_rate=instruct_harmful_refusal_rate,
        instruct_harmless_refusal_rate=instruct_harmless_refusal_rate,
        base_harmful_refusal_rate=base_harmful_refusal_rate,
        base_harmless_refusal_rate=base_harmless_refusal_rate,
        steered_harmful_refusal_rate=steered_harmful_refusal_rate,
        refusal_rate_increase=refusal_rate_increase,
        transfer_effectiveness=transfer_effectiveness,
        steering_results=steering_results,
        aggregate_metrics=aggregate_metrics,
    )

    # Save results
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "instruct_model_path": result.instruct_model_path,
            "base_model_path": result.base_model_path,
            "steering_layer": result.steering_layer,
            "steering_strength": result.steering_strength,
            "num_harmful_prompts": result.num_harmful_prompts,
            "num_harmless_prompts": result.num_harmless_prompts,
            "refusal_direction_strength": result.refusal_direction_strength,
            "refusal_direction_explained_var": result.refusal_direction_explained_var,
            "refusal_threshold": result.refusal_threshold,
            "instruct_harmful_refusal_rate": result.instruct_harmful_refusal_rate,
            "instruct_harmless_refusal_rate": result.instruct_harmless_refusal_rate,
            "base_harmful_refusal_rate": result.base_harmful_refusal_rate,
            "base_harmless_refusal_rate": result.base_harmless_refusal_rate,
            "steered_harmful_refusal_rate": result.steered_harmful_refusal_rate,
            "refusal_rate_increase": result.refusal_rate_increase,
            "transfer_effectiveness": result.transfer_effectiveness,
            "aggregate_metrics": aggregate_metrics,
            "steering_results": [
                {
                    "prompt": r.prompt,
                    "original_projection": r.original_projection,
                    "steered_projection": r.steered_projection,
                    "projection_increase": r.projection_increase,
                    "crossed_threshold": r.steered_projection > refusal_threshold,
                }
                for r in steering_results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("ALIGNMENT TRANSFER SUMMARY")
    logger.info("=" * 60)
    logger.info("Instruct (donor): %s", Path(instruct_model_path).name)
    logger.info("Base (recipient): %s", Path(base_model_path).name)
    logger.info("Steering layer: %d", steering_layer)
    logger.info("Steering strength: %.2f", steering_strength)
    logger.info("-" * 60)
    logger.info("Refusal Direction:")
    logger.info("  Strength: %.4f", refusal_direction_result.strength)
    logger.info("  Explained variance: %.4f", refusal_direction_result.explained_variance)
    logger.info("  Threshold (95th pct harmless): %.4f", refusal_threshold)
    logger.info("-" * 60)
    logger.info("Harmful Refusal Rates (above threshold):")
    logger.info("  Instruct model:        %.1f%%", instruct_harmful_refusal_rate * 100)
    logger.info("  Base (before):         %.1f%%", base_harmful_refusal_rate * 100)
    logger.info("  Base (after steering): %.1f%%", steered_harmful_refusal_rate * 100)
    logger.info("-" * 60)
    logger.info("Transfer Metrics:")
    logger.info("  Refusal rate increase: +%.1f%%", refusal_rate_increase * 100)
    logger.info("  Transfer effectiveness: %.1f%%", transfer_effectiveness * 100)
    logger.info("=" * 60)

    return result


__all__ = [
    "AlignmentTransferResult",
    "SteeringResult",
    "run_alignment_transfer",
]
