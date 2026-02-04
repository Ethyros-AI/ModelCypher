#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validation experiment: Loop Preservation Training.

This experiment trains two adapters on the same math task:
1. With loop preservation loss (preserves Δβ₁)
2. Without loop preservation (baseline)

Then compares the spectral entropy trajectories to verify that
loop preservation actually preserves topological structure.

Usage:
    poetry run python scripts/experiments/loop_preservation_validation.py

Results are saved to: experiments/loop_preservation_validation/
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Initialize backend before imports
from modelcypher.backends import initialize_default_backend
initialize_default_backend()

import mlx.core as mx
import numpy as np
from mlx_lm import load

from modelcypher.core.domain.training.loop_preservation import (
    LoopPreservationConfig,
    detect_highway_layer,
    compute_base_entropy_trajectory,
    compute_entropy_trajectory,
)
from modelcypher.core.domain.training.geometric_lora import analyze_model_geometry
from modelcypher.core.domain.training.geometric_lora_trainer import (
    derive_config_from_geometry,
    train_geometric_lora,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
OUTPUT_DIR = Path("experiments/loop_preservation_validation")
N_SAMPLES = 100  # Training samples
EPOCHS = 3


def generate_math_samples(n: int) -> list[dict]:
    """Generate simple math training samples."""
    import random
    samples = []

    for _ in range(n):
        op = random.choice(['+', '-'])
        if op == '+':
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            result = a + b
            prompt = f"Calculate: {a} + {b} = "
            completion = str(result)
        else:
            a = random.randint(10, 99)
            b = random.randint(1, a)
            result = a - b
            prompt = f"Calculate: {a} - {b} = "
            completion = str(result)

        samples.append({"prompt": prompt, "completion": completion})

    return samples


def compute_entropy_profile(model, tokenizer, prompts: list[str], n_layers: int) -> dict:
    """Compute spectral entropy at each layer for a set of prompts."""
    from modelcypher.core.domain.training.loop_preservation import _compute_spectral_entropy, _get_hidden_at_layer

    layer_entropies = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])

        for layer_idx in range(n_layers):
            hidden = _get_hidden_at_layer(model, input_ids, layer_idx)
            mx.eval(hidden)
            entropy = _compute_spectral_entropy(hidden)
            layer_entropies[layer_idx].append(entropy)

    # Average across prompts
    return {i: np.mean(vals) for i, vals in layer_entropies.items()}


def main():
    logger.info("=" * 60)
    logger.info("Loop Preservation Validation Experiment")
    logger.info("=" * 60)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # Load model
    logger.info(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", [])
    n_layers = len(layers)
    logger.info(f"Model has {n_layers} layers")

    # Generate training data
    logger.info(f"Generating {N_SAMPLES} training samples...")
    training_data = generate_math_samples(N_SAMPLES)
    data_path = run_dir / "training_data.jsonl"
    with open(data_path, "w") as f:
        for sample in training_data:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved training data to {data_path}")

    # Probe prompts for measurement
    probe_prompts = [
        "Calculate: 25 + 17 = ",
        "Calculate: 43 - 19 = ",
        "Calculate: 8 + 5 = ",
        "Calculate: 30 - 12 = ",
    ]

    # Compute base model geometry
    logger.info("\n=== Computing Base Model Geometry ===")
    highway_layer = detect_highway_layer(model, tokenizer, probe_prompts)
    base_delta_entropy = compute_base_entropy_trajectory(model, tokenizer, probe_prompts, highway_layer)

    geometries = analyze_model_geometry(model)
    first_geom = next(iter(geometries.values()))
    sigma_max = first_geom.sigma_max

    loop_config = LoopPreservationConfig(
        highway_layer=highway_layer,
        base_delta_entropy=base_delta_entropy,
        lambda_scale=1.0 / max(sigma_max, 1e-8),
    )

    logger.info(f"Highway layer: {highway_layer}")
    logger.info(f"Base ΔH: {base_delta_entropy:+.4f}")
    logger.info(f"λ (loss scale): {loop_config.lambda_scale:.6f}")

    # Compute base model entropy profile
    logger.info("\nComputing base model entropy profile...")
    base_profile = compute_entropy_profile(model, tokenizer, probe_prompts, n_layers)

    # Derive LoRA config
    lora_config = derive_config_from_geometry(
        model,
        learning_rate=1e-4,
        epochs=EPOCHS,
        batch_size=4,
    )

    results = {
        "model_path": MODEL_PATH,
        "n_layers": n_layers,
        "n_samples": N_SAMPLES,
        "epochs": EPOCHS,
        "highway_layer": highway_layer,
        "base_delta_entropy": base_delta_entropy,
        "lambda_scale": loop_config.lambda_scale,
        "base_entropy_profile": base_profile,
    }

    # ========================================
    # Experiment A: WITHOUT loop preservation
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT A: Training WITHOUT loop preservation")
    logger.info("=" * 60)

    # Reload model (fresh weights)
    model_a, _ = load(MODEL_PATH)

    adapter_a_path = run_dir / "adapter_without_loop_preservation"
    result_a = train_geometric_lora(
        model=model_a,
        tokenizer=tokenizer,
        training_data=training_data,
        output_path=adapter_a_path,
        config=lora_config,
        loop_config=None,  # No loop preservation
    )

    if result_a.success:
        logger.info(f"Training A complete: loss={result_a.final_loss:.4f}")
        profile_a = compute_entropy_profile(model_a, tokenizer, probe_prompts, n_layers)
        delta_a = profile_a[n_layers - 1] - profile_a[highway_layer]
        logger.info(f"Post-training ΔH (A): {delta_a:+.4f}")
        results["experiment_a"] = {
            "final_loss": result_a.final_loss,
            "entropy_profile": profile_a,
            "delta_entropy": delta_a,
            "delta_change": delta_a - base_delta_entropy,
        }
    else:
        logger.error(f"Training A failed: {result_a.error}")
        results["experiment_a"] = {"error": result_a.error}

    # ========================================
    # Experiment B: WITH loop preservation
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT B: Training WITH loop preservation")
    logger.info("=" * 60)

    # Reload model (fresh weights)
    model_b, _ = load(MODEL_PATH)

    adapter_b_path = run_dir / "adapter_with_loop_preservation"
    result_b = train_geometric_lora(
        model=model_b,
        tokenizer=tokenizer,
        training_data=training_data,
        output_path=adapter_b_path,
        config=lora_config,
        loop_config=loop_config,  # WITH loop preservation
    )

    if result_b.success:
        logger.info(f"Training B complete: loss={result_b.final_loss:.4f}")
        profile_b = compute_entropy_profile(model_b, tokenizer, probe_prompts, n_layers)
        delta_b = profile_b[n_layers - 1] - profile_b[highway_layer]
        logger.info(f"Post-training ΔH (B): {delta_b:+.4f}")
        results["experiment_b"] = {
            "final_loss": result_b.final_loss,
            "entropy_profile": profile_b,
            "delta_entropy": delta_b,
            "delta_change": delta_b - base_delta_entropy,
        }
    else:
        logger.error(f"Training B failed: {result_b.error}")
        results["experiment_b"] = {"error": result_b.error}

    # ========================================
    # Compare Results
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    if "experiment_a" in results and "experiment_b" in results:
        if "error" not in results["experiment_a"] and "error" not in results["experiment_b"]:
            base_dh = base_delta_entropy
            delta_a = results["experiment_a"]["delta_entropy"]
            delta_b = results["experiment_b"]["delta_entropy"]

            logger.info(f"Base model ΔH:           {base_dh:+.4f}")
            logger.info(f"Without preservation ΔH: {delta_a:+.4f} (change: {delta_a - base_dh:+.4f})")
            logger.info(f"With preservation ΔH:    {delta_b:+.4f} (change: {delta_b - base_dh:+.4f})")

            # Check if loop preservation helped
            a_degradation = abs(delta_a - base_dh)
            b_degradation = abs(delta_b - base_dh)

            if b_degradation < a_degradation:
                improvement = (a_degradation - b_degradation) / a_degradation * 100
                logger.info(f"\n✓ Loop preservation reduced entropy drift by {improvement:.1f}%")
                results["conclusion"] = f"Loop preservation reduced drift by {improvement:.1f}%"
            else:
                logger.info("\n✗ Loop preservation did not reduce drift (may need more training)")
                results["conclusion"] = "No improvement observed"

    # Save results
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
