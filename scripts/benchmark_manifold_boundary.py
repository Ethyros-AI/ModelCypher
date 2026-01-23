#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Benchmark script for manifold boundary detection.
"""
Manifold Boundary Detection Benchmark

Compares two approaches to finding model utilization:
1. Variance-based: Infer from sample statistics (what we tried before)
2. Flood fill: Directly probe model response (new approach)

The key question: Which gives the TRUE boundary of the learned manifold?
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def initialize_backend():
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
    )

    return model, tokenizer, config


def generate_probe_texts(max_probes: int = 200) -> list[str]:
    """Generate probe texts from unified atlas."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    import random

    probes = UnifiedAtlasInventory.all_probes()
    texts = []
    for probe in probes:
        if probe.support_texts:
            for text in probe.support_texts:
                if text and len(text) > 5:
                    texts.append(text)

    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    if len(unique_texts) > max_probes:
        random.seed(42)
        unique_texts = random.sample(unique_texts, max_probes)

    logger.info("Using %d probe texts", len(unique_texts))
    return unique_texts


def collect_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: dict,
) -> dict[int, Any]:
    """Collect activations per layer across all probe texts."""
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    num_layers = config.get("num_hidden_layers", 0)
    layer_activations: dict[int, list] = {i: [] for i in range(num_layers)}

    logger.info("Collecting activations for %d probes", len(texts))

    for text in texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            for layer_idx, act in acts.items():
                layer_activations[layer_idx].append(act)
        except Exception as e:
            pass

    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            stacked = mx.stack(acts, axis=0)
            mx.eval(stacked)
            result[layer_idx] = stacked

    return result


def compare_approaches(
    model: Any,
    layer_activations: dict[int, Any],
    config: dict,
    layers_to_test: list[int],
    n_directions: int = 30,
    max_radius: float = 10.0,
    forward_mode: str = "full_model",
) -> dict:
    """Compare variance-based vs flood fill boundary detection."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_compression import (
        estimate_compression_potential,
    )
    from modelcypher.core.domain.geometry.manifold_boundary import (
        detect_layer_boundary,
    )

    backend = get_default_backend()
    results = {}

    for layer_idx in layers_to_test:
        if layer_idx not in layer_activations:
            continue

        activations = layer_activations[layer_idx]
        logger.info("\n=== LAYER %d ===", layer_idx)

        # Method 1: Variance-based
        logger.info("Method 1: Variance-based null space")
        start = time.time()
        variance_result = estimate_compression_potential(activations, backend)
        variance_time = time.time() - start

        logger.info(
            "  Utilized: %d/%d (%.1f%%), time=%.2fs",
            variance_result["intrinsic_dim"],
            variance_result["hidden_dim"],
            variance_result["utilized_fraction"] * 100,
            variance_time,
        )

        # Method 2: Flood fill boundary detection
        logger.info("Method 2: Flood fill boundary detection")
        start = time.time()
        try:
            boundary_result = detect_layer_boundary(
                model=model,
                layer_idx=layer_idx,
                activations=activations,
                config=config,
                backend=backend,
                n_directions=n_directions,
                max_radius=max_radius,
                forward_mode=forward_mode,
            )
            flood_time = time.time() - start

            logger.info(
                "  Mean radius: %.3f, min=%.3f, max=%.3f",
                boundary_result.mean_radius,
                boundary_result.min_radius,
                boundary_result.max_radius,
            )
            logger.info(
                "  Bounded: %d/%d, utilized_volume=%.1f%%, time=%.2fs",
                boundary_result.n_bounded,
                n_directions,
                boundary_result.utilized_volume_fraction * 100,
                flood_time,
            )

            results[layer_idx] = {
                "variance": {
                    "utilized_rank": variance_result["intrinsic_dim"],
                    "hidden_dim": variance_result["hidden_dim"],
                    "utilized_fraction": variance_result["utilized_fraction"],
                    "time": variance_time,
                },
                "flood_fill": {
                    "mean_radius": boundary_result.mean_radius,
                    "min_radius": boundary_result.min_radius,
                    "max_radius": boundary_result.max_radius,
                    "n_bounded": boundary_result.n_bounded,
                    "n_directions": n_directions,
                    "utilized_volume_fraction": boundary_result.utilized_volume_fraction,
                    "time": flood_time,
                },
            }

        except Exception as e:
            logger.warning("Flood fill failed: %s", e)
            results[layer_idx] = {
                "variance": {
                    "utilized_rank": variance_result["intrinsic_dim"],
                    "hidden_dim": variance_result["hidden_dim"],
                    "utilized_fraction": variance_result["utilized_fraction"],
                    "time": variance_time,
                },
                "flood_fill": {"error": str(e)},
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark manifold boundary detection")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--max-probes", type=int, default=200, help="Number of probes")
    parser.add_argument("--layers", type=str, default="0,5,10", help="Layers to test (comma-separated)")
    parser.add_argument("--directions", type=int, default=30, help="Number of directions to probe")
    parser.add_argument("--max-radius", type=float, default=10.0, help="Maximum radius to probe")
    parser.add_argument("--forward-mode", type=str, default="full_model", choices=["mlp", "full_model"],
                        help="Forward mode: 'mlp' for local MLP, 'full_model' for cascade through remaining layers")
    args = parser.parse_args()

    initialize_backend()

    # Parse layers
    layers_to_test = [int(x.strip()) for x in args.layers.split(",")]

    # Load model
    model, tokenizer, config = load_model(args.model)

    # Generate probes
    probes = generate_probe_texts(max_probes=args.max_probes)

    # Collect activations
    logger.info("\n=== COLLECTING ACTIVATIONS ===")
    start = time.time()
    layer_activations = collect_layer_activations(model, tokenizer, probes, config)
    logger.info("Activation collection took %.2fs", time.time() - start)

    # Compare approaches
    logger.info("\n=== COMPARING APPROACHES ===")
    results = compare_approaches(
        model, layer_activations, config,
        layers_to_test=layers_to_test,
        n_directions=args.directions,
        max_radius=args.max_radius,
        forward_mode=args.forward_mode,
    )

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info("%-8s %-20s %-30s", "Layer", "Variance (rank/dim)", "FloodFill (min/mean/max radius)")
    logger.info("-" * 60)
    for layer_idx, layer_results in results.items():
        variance = layer_results.get("variance", {})
        flood = layer_results.get("flood_fill", {})

        var_str = "%d/%d (%.1f%%)" % (
            variance.get("utilized_rank", 0),
            variance.get("hidden_dim", 0),
            variance.get("utilized_fraction", 0) * 100,
        )

        if "error" not in flood:
            flood_str = "%.2f/%.2f/%.2f (%d/%d bounded)" % (
                flood.get("min_radius", 0),
                flood.get("mean_radius", 0),
                flood.get("max_radius", 0),
                flood.get("n_bounded", 0),
                flood.get("n_directions", 0),
            )
        else:
            flood_str = "ERROR: " + flood.get("error", "")[:20]

        logger.info("%-8d %-20s %-30s", layer_idx, var_str, flood_str)


if __name__ == "__main__":
    main()
