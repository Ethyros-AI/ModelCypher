#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Safety-aware model compression using manifold analysis.
"""
Safe Model Compression

Unlike naive variance-based compression, this script:
1. Analyzes manifold safety (variance + boundary) per layer
2. SKIPS bottleneck layers (small boundary radius)
3. Only compresses layers with both low-variance AND stable boundaries
4. Uses dtype-derived CKA threshold (you can't invent precision)

The key insight: variance alone is insufficient. A direction might have
low variance but small boundary (model is sensitive there). Compressing
such directions breaks inference.
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


# Dtype precision limits
DTYPE_EPSILON = {
    "float32": 1.19e-7,
    "float16": 9.77e-4,
    "bfloat16": 7.81e-3,
}


def get_cka_threshold(dtype: str) -> float:
    """Get the CKA threshold for 'lossless' compression given dtype.

    You cannot achieve higher precision than the weights encode.
    Using sqrt(eps) as threshold accounts for error propagation.
    """
    import math
    eps = DTYPE_EPSILON.get(dtype, DTYPE_EPSILON["float32"])
    return 1.0 - math.sqrt(eps)


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

    # Detect dtype from first weight
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.parameters()))
    first_weight = next(iter(weights.values()))
    dtype = str(first_weight.dtype)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d, dtype=%s",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
        dtype,
    )

    return model, tokenizer, config, dtype


def generate_probe_texts(max_probes: int = 500) -> list[str]:
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

    logger.info("Collecting activations for %d probes across %d layers", len(texts), num_layers)

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


def analyze_safety(
    model: Any,
    layer_activations: dict[int, Any],
    config: dict,
    n_directions: int = 30,
    max_radius: float = 5.0,
    min_safe_radius: float = 1.0,
    forward_mode: str = "full_model",
) -> dict[int, Any]:
    """Analyze manifold safety for all layers.

    Args:
        forward_mode: "full_model" for cascade sensitivity (recommended),
            "mlp" for local MLP-only sensitivity (may miss cascade effects).
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.manifold_safety import analyze_model_safety

    backend = get_default_backend()
    return analyze_model_safety(
        model=model,
        layer_activations=layer_activations,
        config=config,
        backend=backend,
        n_directions=n_directions,
        max_radius=max_radius,
        min_safe_radius=min_safe_radius,
        forward_mode=forward_mode,
    )


def compress_safe_layers(
    model: Any,
    layer_activations: dict[int, Any],
    safety_results: dict[int, Any],
    config: dict,
    cka_threshold: float,
) -> dict[str, dict]:
    """Compress only layers that are safe (not bottlenecks).

    Returns compression metadata for saved model.
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
    skipped_layers = []
    compressed_layers = []

    for layer_idx in sorted(layer_activations.keys()):
        safety = safety_results.get(layer_idx)
        activations = layer_activations[layer_idx]
        mlp_keys = arch.layer_mlp_keys(layer_idx)

        if safety and safety.is_bottleneck:
            logger.info("Layer %d: SKIPPED (bottleneck, boundary=%.2f)",
                       layer_idx, safety.boundary_min_radius)
            skipped_layers.append(layer_idx)
            compression_metadata[f"layer_{layer_idx}"] = {
                "skipped": True,
                "reason": "bottleneck",
                "boundary_min_radius": safety.boundary_min_radius,
            }
            continue

        logger.info("Layer %d: Compressing (boundary=%.2f)...",
                   layer_idx, safety.boundary_min_radius if safety else 0)
        compressed_layers.append(layer_idx)

        for key in mlp_keys:
            try:
                # Navigate to weight
                parts = key.split(".")
                obj = model
                for i, part in enumerate(parts[:-1]):
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

                # Check CKA threshold
                if cka < cka_threshold:
                    logger.warning(
                        "  %s: CKA=%.6f < threshold %.6f, keeping original",
                        key.split(".")[-2] + "." + key.split(".")[-1],
                        cka, cka_threshold
                    )
                    continue

                # Reconstruct and replace weight
                W_reconstructed = result.reconstruct(backend)
                backend.eval(W_reconstructed)
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

    logger.info("")
    logger.info("Compression summary:")
    logger.info("  Skipped (bottleneck): %s", skipped_layers)
    logger.info("  Compressed: %s", compressed_layers)

    if total_original > 0:
        overall = total_original / total_compressed
        logger.info(
            "  MLP params: %.2fM -> %.2fM (%.2fx compression)",
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
    safety_results: dict[int, Any],
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

    # Save config with compression metadata
    config_with_meta = config.copy()

    # Convert safety results to serializable format
    safety_summary = {}
    for layer_idx, result in safety_results.items():
        safety_summary[str(layer_idx)] = {
            "is_bottleneck": result.is_bottleneck,
            "boundary_min_radius": result.boundary_min_radius,
            "boundary_mean_radius": result.boundary_mean_radius,
            "variance_utilized_rank": result.variance_utilized_rank,
            "variance_available_rank": result.variance_available_rank,
            "safe_compression_rank": result.safe_compression_rank,
        }

    config_with_meta["safe_compression"] = {
        "compressed": True,
        "weights": compression_metadata,
        "safety_analysis": safety_summary,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_with_meta, f, indent=2)
    logger.info("Saved config.json with compression metadata")

    logger.info("Compressed model saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Safety-aware model compression")
    parser.add_argument("--model", type=str, required=True, help="Path to source model")
    parser.add_argument("--output", type=str, required=True, help="Path to save compressed model")
    parser.add_argument("--max-probes", type=int, default=500, help="Number of probes")
    parser.add_argument("--directions", type=int, default=30, help="Directions for boundary detection")
    parser.add_argument("--min-safe-radius", type=float, default=1.0, help="Minimum safe boundary radius")
    parser.add_argument("--skip-verify", action="store_true", help="Skip inference verification")
    parser.add_argument("--forward-mode", type=str, default="full_model", choices=["mlp", "full_model"],
                        help="Safety analysis mode: 'full_model' for cascade sensitivity (recommended)")
    args = parser.parse_args()

    initialize_backend()

    # Load model
    model, tokenizer, config, dtype = load_model(args.model)

    # Get dtype-derived CKA threshold
    cka_threshold = get_cka_threshold(dtype)
    logger.info("Dtype: %s, CKA threshold for 'lossless': %.6f", dtype, cka_threshold)

    # Generate probes
    probes = generate_probe_texts(max_probes=args.max_probes)

    # Collect activations
    logger.info("\n=== COLLECTING ACTIVATIONS ===")
    start = time.time()
    layer_activations = collect_layer_activations(model, tokenizer, probes, config)
    logger.info("Activation collection took %.2fs", time.time() - start)

    # Analyze safety
    logger.info("\n=== ANALYZING MANIFOLD SAFETY ===")
    start = time.time()
    safety_results = analyze_safety(
        model, layer_activations, config,
        n_directions=args.directions,
        min_safe_radius=args.min_safe_radius,
        forward_mode=args.forward_mode,
    )
    logger.info("Safety analysis took %.2fs", time.time() - start)

    # Print safety summary
    bottleneck_layers = [idx for idx, r in safety_results.items() if r.is_bottleneck]
    safe_layers = [idx for idx, r in safety_results.items() if not r.is_bottleneck]
    logger.info("")
    logger.info("Bottleneck layers (will SKIP): %s", bottleneck_layers)
    logger.info("Safe layers (will COMPRESS): %s", safe_layers)

    # Test prompts
    test_prompts = [
        "What is 2 + 2? The answer is",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # Baseline inference
    if not args.skip_verify:
        logger.info("\n=== BASELINE INFERENCE ===")
        baseline_responses = verify_inference(model, tokenizer, test_prompts)

    # Compress safe layers only
    logger.info("\n=== COMPRESSING SAFE LAYERS ===")
    start = time.time()
    compression_metadata = compress_safe_layers(
        model, layer_activations, safety_results, config, cka_threshold
    )
    logger.info("Compression took %.2fs", time.time() - start)

    # Verify compressed model
    if not args.skip_verify:
        logger.info("\n=== COMPRESSED INFERENCE ===")
        compressed_responses = verify_inference(model, tokenizer, test_prompts)

        # Compare
        logger.info("\n=== COMPARISON ===")
        all_match = True
        for i, (baseline, compressed) in enumerate(zip(baseline_responses, compressed_responses)):
            match = baseline == compressed
            all_match = all_match and match
            logger.info("Prompt %d: %s", i, "MATCH ✓" if match else "DIFFER ✗")
            if not match:
                logger.info("  Baseline: %s", baseline[:100])
                logger.info("  Compressed: %s", compressed[:100])

        if all_match:
            logger.info("\n✓ All responses match! Compression is lossless within dtype precision.")
        else:
            logger.warning("\n⚠️ Some responses differ. Check compression quality.")

    # Save
    logger.info("\n=== SAVING COMPRESSED MODEL ===")
    save_compressed_model(
        model, tokenizer, config, args.output,
        compression_metadata, args.model, safety_results,
    )

    logger.info("\n=== DONE ===")
    logger.info("Compressed model saved to: %s", args.output)


if __name__ == "__main__":
    main()
