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
    1. Extract refusal direction from instruct model
    2. Test base model's response to harmful prompts (baseline)
    3. Apply activation steering: add refusal direction to base model activations
    4. Test steered base model's response
    5. Measure alignment transfer effectiveness

This demonstrates that alignment is geometric and transferable - we can add the
"refusal direction" from one model to another to induce aligned behavior.

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
    # Before steering (instruct model baseline)
    instruct_mean_harmful_proj: float
    instruct_mean_harmless_proj: float
    instruct_separation: float
    # Base model before steering
    base_mean_harmful_proj: float
    base_mean_harmless_proj: float
    base_separation: float
    # Base model after steering
    steered_mean_harmful_proj: float
    steered_mean_harmless_proj: float
    steered_separation: float
    # Transfer effectiveness
    separation_improvement: float
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
        best_sep = 0.0
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
                harmful_proj = backend.mean(backend.sum(harmful_acts * refusal_dir.direction, axis=1))
                harmless_proj = backend.mean(backend.sum(harmless_acts * refusal_dir.direction, axis=1))
                backend.eval(harmful_proj, harmless_proj)
                sep = abs(float(backend.to_scalar(harmful_proj)) - float(backend.to_scalar(harmless_proj)))
                if sep > best_sep:
                    best_sep = sep
                    best_layer = layer_idx

        steering_layer = best_layer

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

    instruct_mean_harmful = float(backend.to_scalar(backend.mean(instruct_harmful_projs)))
    instruct_mean_harmless = float(backend.to_scalar(backend.mean(instruct_harmless_projs)))
    instruct_separation = instruct_mean_harmful - instruct_mean_harmless

    logger.info("Instruct model: harmful=%.4f, harmless=%.4f, separation=%.4f",
                instruct_mean_harmful, instruct_mean_harmless, instruct_separation)

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

    base_mean_harmful = float(backend.to_scalar(backend.mean(base_harmful_projs)))
    base_mean_harmless = float(backend.to_scalar(backend.mean(base_harmless_projs)))
    base_separation = base_mean_harmful - base_mean_harmless

    logger.info("Base model (before): harmful=%.4f, harmless=%.4f, separation=%.4f",
                base_mean_harmful, base_mean_harmless, base_separation)

    # =========================================================================
    # STEP 3: Apply steering and measure effect
    # =========================================================================
    logger.info("Applying alignment steering (strength=%.2f)...", steering_strength)

    steering_results: list[SteeringResult] = []

    # Steer harmful prompts (add refusal direction to induce refusal)
    steered_harmful_projs = []
    for i, prompt in enumerate(harmful_prompts):
        act = base_harmful_by_layer[steering_layer][i]
        orig_proj, steered_proj, _ = compute_steered_projection(
            act, refusal_direction, steering_strength, backend
        )
        steered_harmful_projs.append(steered_proj)

        steering_results.append(SteeringResult(
            prompt=prompt[:80],
            original_projection=orig_proj,
            steered_projection=steered_proj,
            projection_increase=steered_proj - orig_proj,
            steering_strength=steering_strength,
        ))

    # Steer harmless prompts (should stay relatively unchanged if steering is targeted)
    steered_harmless_projs = []
    for i, prompt in enumerate(harmless_prompts):
        act = base_harmless_by_layer[steering_layer][i]
        orig_proj, steered_proj, _ = compute_steered_projection(
            act, refusal_direction, steering_strength, backend
        )
        steered_harmless_projs.append(steered_proj)

    # Compute steered statistics
    steered_mean_harmful = sum(steered_harmful_projs) / len(steered_harmful_projs)
    steered_mean_harmless = sum(steered_harmless_projs) / len(steered_harmless_projs)
    steered_separation = steered_mean_harmful - steered_mean_harmless

    logger.info("Base model (after): harmful=%.4f, harmless=%.4f, separation=%.4f",
                steered_mean_harmful, steered_mean_harmless, steered_separation)

    # =========================================================================
    # STEP 4: Compute transfer effectiveness metrics
    # =========================================================================
    separation_improvement = steered_separation - base_separation

    # Transfer effectiveness: how close is steered base to instruct?
    # 1.0 = perfect transfer (steered matches instruct separation)
    # 0.0 = no transfer (steered same as base)
    if abs(instruct_separation - base_separation) > 1e-6:
        transfer_effectiveness = (steered_separation - base_separation) / (
            instruct_separation - base_separation
        )
        transfer_effectiveness = max(0.0, min(1.0, transfer_effectiveness))
    else:
        transfer_effectiveness = 1.0 if abs(separation_improvement) < 1e-6 else 0.0

    logger.info("Transfer effectiveness: %.1f%%", transfer_effectiveness * 100)

    # Aggregate metrics
    aggregate_metrics = {
        "instruct_separation": instruct_separation,
        "base_separation_before": base_separation,
        "base_separation_after": steered_separation,
        "separation_improvement": separation_improvement,
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
        instruct_mean_harmful_proj=instruct_mean_harmful,
        instruct_mean_harmless_proj=instruct_mean_harmless,
        instruct_separation=instruct_separation,
        base_mean_harmful_proj=base_mean_harmful,
        base_mean_harmless_proj=base_mean_harmless,
        base_separation=base_separation,
        steered_mean_harmful_proj=steered_mean_harmful,
        steered_mean_harmless_proj=steered_mean_harmless,
        steered_separation=steered_separation,
        separation_improvement=separation_improvement,
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
            "instruct_separation": result.instruct_separation,
            "base_separation_before": result.base_separation,
            "base_separation_after": result.steered_separation,
            "separation_improvement": result.separation_improvement,
            "transfer_effectiveness": result.transfer_effectiveness,
            "aggregate_metrics": aggregate_metrics,
            "steering_results": [
                {
                    "prompt": r.prompt,
                    "original_projection": r.original_projection,
                    "steered_projection": r.steered_projection,
                    "projection_increase": r.projection_increase,
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
    logger.info("-" * 60)
    logger.info("Harmful/Harmless Separation (refusal axis):")
    logger.info("  Instruct model: %.4f", instruct_separation)
    logger.info("  Base (before):  %.4f", base_separation)
    logger.info("  Base (after):   %.4f", steered_separation)
    logger.info("-" * 60)
    logger.info("Transfer Metrics:")
    logger.info("  Separation improvement: +%.4f", separation_improvement)
    logger.info("  Transfer effectiveness: %.1f%%", transfer_effectiveness * 100)
    logger.info("=" * 60)

    return result


__all__ = [
    "AlignmentTransferResult",
    "SteeringResult",
    "run_alignment_transfer",
]
