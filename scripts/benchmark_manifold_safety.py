#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Benchmark script for combined manifold safety analysis.
"""
Manifold Safety Benchmark

Shows the layer-by-layer safety profile combining:
1. Variance analysis (intrinsic dimension)
2. Boundary analysis (flood fill)
3. Combined safety score

Identifies bottleneck layers where compression/transplant is risky.
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
        except Exception:
            pass

    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            stacked = mx.stack(acts, axis=0)
            mx.eval(stacked)
            result[layer_idx] = stacked

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark manifold safety analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--max-probes", type=int, default=200, help="Number of probes")
    parser.add_argument("--directions", type=int, default=30, help="Directions to probe")
    parser.add_argument("--max-radius", type=float, default=5.0, help="Maximum radius")
    parser.add_argument("--min-safe-radius", type=float, default=1.0, help="Minimum safe radius")
    parser.add_argument("--forward-mode", type=str, default="full_model", choices=["mlp", "full_model"],
                        help="Forward mode: 'mlp' for local MLP, 'full_model' for cascade through remaining layers")
    args = parser.parse_args()

    initialize_backend()

    # Load model
    model, tokenizer, config = load_model(args.model)

    # Generate probes
    probes = generate_probe_texts(max_probes=args.max_probes)

    # Collect activations
    logger.info("\n=== COLLECTING ACTIVATIONS ===")
    start = time.time()
    layer_activations = collect_layer_activations(model, tokenizer, probes, config)
    logger.info("Activation collection took %.2fs", time.time() - start)

    # Analyze safety
    logger.info("\n=== ANALYZING LAYER SAFETY ===")
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.manifold_safety import analyze_model_safety

    backend = get_default_backend()
    start = time.time()
    safety_results = analyze_model_safety(
        model=model,
        layer_activations=layer_activations,
        config=config,
        backend=backend,
        n_directions=args.directions,
        max_radius=args.max_radius,
        min_safe_radius=args.min_safe_radius,
        forward_mode=args.forward_mode,
    )
    logger.info("Safety analysis took %.2fs", time.time() - start)

    # Summary table
    logger.info("\n" + "=" * 100)
    logger.info("MANIFOLD SAFETY ANALYSIS")
    logger.info("=" * 100)
    logger.info("")
    logger.info(
        "%-6s | %-20s | %-20s | %-15s | %-10s",
        "Layer", "Variance (util/avail)", "Boundary (min/mean)", "Safe Rank", "Bottleneck"
    )
    logger.info("-" * 100)

    bottleneck_layers = []
    safe_layers = []
    total_safe_rank = 0
    total_hidden = 0

    for layer_idx, result in safety_results.items():
        var_str = "%d/%d (%.0f%%/%.0f%%)" % (
            result.variance_utilized_rank,
            result.variance_available_rank,
            (result.variance_utilized_rank / result.hidden_dim) * 100,
            (result.variance_available_rank / result.hidden_dim) * 100,
        )

        bound_str = "%.2f/%.2f" % (
            result.boundary_min_radius,
            result.boundary_mean_radius,
        )

        safe_str = "%d (%.0f%%)" % (
            result.safe_compression_rank,
            result.safe_compression_fraction * 100,
        )

        bottleneck_str = "YES" if result.is_bottleneck else "no"

        logger.info(
            "%-6d | %-20s | %-20s | %-15s | %-10s",
            layer_idx, var_str, bound_str, safe_str, bottleneck_str
        )

        if result.is_bottleneck:
            bottleneck_layers.append(layer_idx)
        else:
            safe_layers.append(layer_idx)

        total_safe_rank += result.safe_compression_rank
        total_hidden += result.hidden_dim

    logger.info("-" * 100)

    # Summary statistics
    logger.info("")
    logger.info("SUMMARY:")
    logger.info("  Total layers: %d", len(safety_results))
    logger.info("  Bottleneck layers: %s", bottleneck_layers if bottleneck_layers else "None")
    logger.info("  Safe layers: %d", len(safe_layers))
    logger.info("  Total safe compression rank: %d / %d (%.1f%%)",
                total_safe_rank, total_hidden, (total_safe_rank / total_hidden) * 100 if total_hidden > 0 else 0)

    # Implications
    logger.info("")
    logger.info("IMPLICATIONS:")
    if bottleneck_layers:
        logger.info("  ⚠️  Layers %s are bottlenecks - compression/transplant here is RISKY", bottleneck_layers)
        logger.info("      These layers are at the stability edge even at small perturbations.")
        logger.info("      Variance-based compression would fail here (as we saw with layer 4).")
    if safe_layers:
        logger.info("  ✓  Layers %s are safe for compression/transplant", safe_layers)
        logger.info("      These have both low-variance subspaces AND stable boundaries.")

    # Geometric insight
    logger.info("")
    logger.info("GEOMETRIC INSIGHT:")
    logger.info("  The 'safe subspace' is the intersection of:")
    logger.info("    - Variance null space (directions data doesn't use)")
    logger.info("    - Stable boundary region (directions model tolerates)")
    logger.info("  ")
    logger.info("  For transplants:")
    logger.info("    - Project source delta onto target's safe subspace")
    logger.info("    - This ensures transplant stays within target's manifold")
    logger.info("    - Bottleneck layers need special handling (or skip)")


if __name__ == "__main__":
    main()
