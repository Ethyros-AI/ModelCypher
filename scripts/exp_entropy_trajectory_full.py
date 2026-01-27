#!/usr/bin/env python3
"""Verify the expand-then-compress hypothesis.

The model:
1. EXPANDS information into high-dim space (entropy ↑)
2. PROCESSES in expanded space (high entropy plateau)
3. COMPRESSES to output (entropy ↓)

We measure entropy at EVERY layer to capture the full trajectory.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import svd


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2
E_OVER_PI = np.e / np.pi
PI_OVER_E = np.pi / np.e


def compute_spectral_entropy(activations: np.ndarray, sqrt_eps: float) -> float:
    """Compute spectral entropy from activations."""
    if len(activations) < 2:
        return 0.0

    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)

    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0

    p = S_valid ** 2
    p = p / p.sum()

    return float(-np.sum(p * np.log(p + 1e-10)))


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("ENTROPY TRAJECTORY: EXPAND → PROCESS → COMPRESS")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    # Test prompts
    prompts = [
        "Question: 5 + 3 = ?\n\nAnswer:",
        "Question: 12 - 4 = ?\n\nAnswer:",
        "Question: 6 * 2 = ?\n\nAnswer:",
        "Question: 20 / 4 = ?\n\nAnswer:",
        "Question: If 3 workers finish in 12 days, how long for 4?\n\nAnswer:",
        "Question: A train travels 60 mph for 2 hours. Distance?\n\nAnswer:",
    ]

    # Collect activations at EVERY layer
    logger.info(f"\nCollecting activations from {len(prompts)} prompts at ALL {n_layers} layers...")

    layer_activations = {i: [] for i in range(n_layers)}
    post_norm_activations = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = model.model.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            layer_activations[layer_idx].append(
                np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
            )

        # Final norm
        hidden = model.model.norm(hidden)
        mx.eval(hidden)
        post_norm_activations.append(
            np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
        )

    # Compute entropy at each layer
    logger.info("\nComputing entropy trajectory...")

    entropy_trajectory = []
    for layer_idx in range(n_layers):
        acts = np.vstack(layer_activations[layer_idx])
        entropy = compute_spectral_entropy(acts, sqrt_eps)
        entropy_trajectory.append(entropy)

    # Post-norm entropy
    post_norm_acts = np.vstack(post_norm_activations)
    post_norm_entropy = compute_spectral_entropy(post_norm_acts, sqrt_eps)

    # Find peak and compression
    peak_idx = np.argmax(entropy_trajectory)
    peak_entropy = entropy_trajectory[peak_idx]
    initial_entropy = entropy_trajectory[0]
    final_entropy = entropy_trajectory[-1]

    # Expansion phase: 0 → peak
    expansion_rate = (peak_entropy - initial_entropy) / (peak_idx + 1)

    # Compression phase: peak → final
    compression_layers = n_layers - peak_idx - 1
    compression_rate = (peak_entropy - final_entropy) / max(compression_layers, 1)

    logger.info(f"\n{'=' * 50}")
    logger.info("ENTROPY TRAJECTORY ANALYSIS")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Initial (layer 0):  {initial_entropy:.4f}")
    logger.info(f"  Peak (layer {peak_idx}):    {peak_entropy:.4f}")
    logger.info(f"  Final (layer {n_layers-1}):  {final_entropy:.4f}")
    logger.info(f"  Post-norm:          {post_norm_entropy:.4f}")
    logger.info(f"\n  EXPANSION: layers 0-{peak_idx}")
    logger.info(f"    Rate: {expansion_rate:.4f} entropy/layer")
    logger.info(f"  COMPRESSION: layers {peak_idx}-{n_layers-1}")
    logger.info(f"    Rate: {compression_rate:.4f} entropy/layer")

    # Check if compression happens
    compression_detected = final_entropy < peak_entropy
    post_norm_compression = post_norm_entropy < final_entropy

    logger.info(f"\n  Compression in final layers: {compression_detected}")
    logger.info(f"  Compression in norm layer: {post_norm_compression}")

    # Print full trajectory
    logger.info(f"\n  Full trajectory:")
    for i, e in enumerate(entropy_trajectory):
        marker = " ← PEAK" if i == peak_idx else ""
        logger.info(f"    Layer {i:2d}: {e:.4f}{marker}")
    logger.info(f"    Post-norm: {post_norm_entropy:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "entropy_trajectory": entropy_trajectory,
        "post_norm_entropy": post_norm_entropy,
        "analysis": {
            "initial_entropy": initial_entropy,
            "peak_entropy": peak_entropy,
            "peak_layer": peak_idx,
            "final_entropy": final_entropy,
            "expansion_rate": expansion_rate,
            "compression_rate": compression_rate,
            "compression_detected": compression_detected,
            "post_norm_compression": post_norm_compression,
        },
    }

    output_path = Path("data/experiments/entropy_trajectory_full.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
