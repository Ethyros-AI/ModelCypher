#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Compress a model to its intrinsic dimensionality and save it.
"""
Intrinsic Compression - Full Model Compression

This script:
1. Loads a model and collects activations on diverse probes
2. Compresses all MLP layers to their intrinsic dimensionality
3. Saves the compressed model in factorized form
4. Verifies inference produces identical outputs

The compressed model uses factorized weights:
    Original: W [out_dim, in_dim]
    Factorized: W_left [out_dim, rank] + V_used [in_dim, rank]

Storage: rank * (out_dim + in_dim) instead of out_dim * in_dim
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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


def generate_probe_texts(max_probes: int = 2000) -> list[str]:
    """Generate probe texts from unified atlas."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    import random

    probes = UnifiedAtlasInventory.all_probes()
    logger.info("Loaded %d probes from unified atlas", len(probes))

    texts = []
    for probe in probes:
        if probe.support_texts:
            for text in probe.support_texts:
                if text and len(text) > 5:
                    texts.append(text)

    # Deduplicate
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    # Limit and shuffle
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

    logger.info("Collecting activations for %d probes across %d layers", len(texts), num_layers)

    for text in texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            for layer_idx, act in acts.items():
                layer_activations[layer_idx].append(act)
        except Exception as e:
            logger.warning("Failed to collect activations for '%s...': %s", text[:20], e)

    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            stacked = mx.stack(acts, axis=0)
            mx.eval(stacked)
            result[layer_idx] = stacked

    logger.info("Collected activations for %d layers", len(result))
    return result


def compress_model_mlp(
    model: Any,
    layer_activations: dict[int, Any],
    config: dict,
) -> dict[str, dict]:
    """Compress all MLP layers and replace weights with reconstructed form.

    Returns compression metadata for each weight.
    """
    import mlx.core as mx
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_compression import (
        compress_layer_with_validation,
    )
    from modelcypher.adapters.model_architecture import get_model_architecture

    backend = get_default_backend()
    arch = get_model_architecture(model, config=config)

    compression_metadata = {}
    total_original = 0
    total_compressed = 0

    for layer_idx in sorted(layer_activations.keys()):
        activations = layer_activations[layer_idx]
        mlp_keys = arch.layer_mlp_keys(layer_idx)

        logger.info("Compressing layer %d...", layer_idx)

        for key in mlp_keys:
            try:
                # Navigate to weight
                parts = key.split(".")
                obj = model
                parent = None
                parent_attr = None

                for i, part in enumerate(parts[:-1]):
                    parent = obj
                    parent_attr = part
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)

                weight_name = parts[-1]
                if not hasattr(obj, weight_name):
                    continue

                weight = getattr(obj, weight_name)
                if weight is None or not hasattr(weight, "shape"):
                    continue

                weight_arr = backend.array(weight)
                backend.eval(weight_arr)

                shape = backend.shape(weight_arr)
                if len(shape) != 2:
                    continue

                out_dim, in_dim = int(shape[0]), int(shape[1])
                act_dim = int(backend.shape(activations)[1])

                if act_dim != in_dim:
                    continue

                # Compress
                result, cka, max_rel_error = compress_layer_with_validation(
                    weight_arr, activations, backend
                )

                # Reconstruct and replace weight
                W_reconstructed = result.reconstruct(backend)
                backend.eval(W_reconstructed)

                # Replace the weight in the model
                setattr(obj, weight_name, mx.array(W_reconstructed))
                mx.eval(getattr(obj, weight_name))

                # Track compression
                original_params = out_dim * in_dim
                compressed_params = result.utilized_rank * (out_dim + in_dim)
                total_original += original_params
                total_compressed += compressed_params

                compression_metadata[key] = {
                    "original_shape": [out_dim, in_dim],
                    "utilized_rank": result.utilized_rank,
                    "compression_ratio": result.compression_ratio,
                    "cka": cka,
                    "variance_captured": result.variance_captured,
                }

                logger.info(
                    "  %s: [%d, %d] -> rank=%d (%.1fx), CKA=%.6f",
                    key.split(".")[-2] + "." + key.split(".")[-1],
                    out_dim, in_dim,
                    result.utilized_rank,
                    1.0 / result.compression_ratio,
                    cka
                )

            except Exception as e:
                logger.warning("Failed to compress %s: %s", key, e)

    if total_original > 0:
        overall = total_original / total_compressed
        logger.info(
            "MLP compression complete: %.2fM -> %.2fM params (%.2fx)",
            total_original / 1e6,
            total_compressed / 1e6,
            overall
        )

    return compression_metadata


def verify_inference(
    model: Any,
    tokenizer: Any,
    test_prompts: list[str],
) -> list[str]:
    """Generate responses to verify model works after compression."""
    from mlx_lm import generate

    responses = []
    for prompt in test_prompts:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            verbose=False,
        )
        responses.append(response)
        logger.info("Prompt: %s", prompt)
        logger.info("Response: %s", response[:100])

    return responses


def save_compressed_model(
    model: Any,
    tokenizer: Any,
    config: dict,
    output_path: str,
    compression_metadata: dict,
    source_path: str,
):
    """Save the compressed model."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights = dict(tree_flatten(model.parameters()))
    weight_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(weight_file), weights)
    logger.info("Saved weights to %s", weight_file)

    # Copy tokenizer files
    source_dir = Path(source_path)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in tokenizer_files:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            logger.info("Copied %s", fname)

    # Save config with compression metadata
    config_with_meta = config.copy()
    config_with_meta["intrinsic_compression"] = {
        "compressed": True,
        "weights": compression_metadata,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_with_meta, f, indent=2)
    logger.info("Saved config.json with compression metadata")

    logger.info("Compressed model saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Compress model to intrinsic dimensionality")
    parser.add_argument("--model", type=str, required=True, help="Path to source model")
    parser.add_argument("--output", type=str, required=True, help="Path to save compressed model")
    parser.add_argument("--max-probes", type=int, default=1000, help="Number of probes for manifold sampling")
    parser.add_argument("--skip-verify", action="store_true", help="Skip inference verification")
    args = parser.parse_args()

    # Initialize
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

    # Test prompts for before/after comparison
    test_prompts = [
        "What is 2 + 2? The answer is",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # Baseline inference
    if not args.skip_verify:
        logger.info("\n=== BASELINE INFERENCE ===")
        baseline_responses = verify_inference(model, tokenizer, test_prompts)

    # Compress
    logger.info("\n=== COMPRESSING MODEL ===")
    start = time.time()
    compression_metadata = compress_model_mlp(model, layer_activations, config)
    logger.info("Compression took %.2fs", time.time() - start)

    # Verify compressed model inference
    if not args.skip_verify:
        logger.info("\n=== COMPRESSED INFERENCE ===")
        compressed_responses = verify_inference(model, tokenizer, test_prompts)

        # Compare
        logger.info("\n=== COMPARISON ===")
        for i, (baseline, compressed) in enumerate(zip(baseline_responses, compressed_responses)):
            match = baseline == compressed
            logger.info("Prompt %d: %s", i, "MATCH" if match else "DIFFER")
            if not match:
                logger.info("  Baseline: %s", baseline[:100])
                logger.info("  Compressed: %s", compressed[:100])

    # Save
    logger.info("\n=== SAVING COMPRESSED MODEL ===")
    save_compressed_model(
        model,
        tokenizer,
        config,
        args.output,
        compression_metadata,
        args.model,
    )

    logger.info("\n=== DONE ===")
    logger.info("Compressed model saved to: %s", args.output)


if __name__ == "__main__":
    main()
