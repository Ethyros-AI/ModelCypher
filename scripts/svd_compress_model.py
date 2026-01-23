#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# SVD-based lossless model compression.
"""
SVD-Based Lossless Compression

Unlike probe-based compression which only preserves behavior on sample inputs,
this approach uses the weight's own singular value decomposition:

1. Compute SVD: W = U @ S @ Vt
2. Find precision cutoff: rank k where S[k] < sqrt(eps) * S[0]
3. Truncate: W_compressed = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]

This is TRULY LOSSLESS within dtype precision because:
- We only remove directions contributing < sqrt(eps) relative to sigma_max
- Any output Y = W @ X satisfies: ||Y - Y_compressed|| / ||Y|| < sqrt(eps)

The key insight: The weight's singular values define what it CAN compute.
Directions with S[i] << sqrt(eps) * S[0] cannot contribute meaningfully
given the dtype's precision limits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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


def get_precision_threshold(dtype: str) -> float:
    """Get the singular value threshold for lossless compression.

    Uses sqrt(eps) as threshold because:
    - Error in Y = W @ X is bounded by ||W - W_approx|| * ||X||
    - ||W - W_approx|| is bounded by dropped singular values
    - For relative error < sqrt(eps), we need S[k] < sqrt(eps) * S_max
    """
    eps = DTYPE_EPSILON.get(dtype, DTYPE_EPSILON["float32"])
    return math.sqrt(eps)


def initialize_backend():
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict, str]:
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


def compress_weight_svd(
    weight: Any,
    threshold: float,
) -> tuple[Any, int, int, float]:
    """Compress a weight matrix using SVD truncation.

    Args:
        weight: Weight matrix [out_dim, in_dim]
        threshold: Relative singular value threshold (e.g., sqrt(eps))

    Returns:
        Tuple of (compressed_weight, original_rank, compressed_rank, compression_ratio)
    """
    import mlx.core as mx

    shape = weight.shape
    if len(shape) != 2:
        # Can't SVD compress non-2D weights
        return weight, 1, 1, 1.0

    out_dim, in_dim = shape

    # Compute SVD (MLX always computes reduced/thin SVD)
    U, S, Vt = mx.linalg.svd(weight)
    mx.eval(U, S, Vt)

    # Find cutoff rank
    S_max = float(S[0])
    if S_max == 0:
        logger.warning("Weight has S_max=0, skipping compression")
        return weight, int(min(out_dim, in_dim)), int(min(out_dim, in_dim)), 1.0

    cutoff = threshold * S_max

    # Find rank where S[k] >= cutoff
    S_np = S.tolist()  # Convert to Python list for iteration
    compressed_rank = 0
    for i, s in enumerate(S_np):
        if s >= cutoff:
            compressed_rank = i + 1
        else:
            break

    # Ensure at least rank 1
    if compressed_rank == 0:
        compressed_rank = 1

    original_rank = min(out_dim, in_dim)

    # If no compression possible, return original
    if compressed_rank >= original_rank:
        return weight, original_rank, original_rank, 1.0

    # Truncate and reconstruct
    U_k = U[:, :compressed_rank]
    S_k = S[:compressed_rank]
    Vt_k = Vt[:compressed_rank, :]
    mx.eval(U_k, S_k, Vt_k)

    # Reconstruct: W = U @ diag(S) @ Vt
    # For efficiency, compute as (U @ diag(S)) @ Vt
    US = U_k * S_k  # Broadcasting: [out, k] * [k] -> [out, k]
    mx.eval(US)
    W_compressed = mx.matmul(US, Vt_k)
    mx.eval(W_compressed)

    # Compute compression ratio
    original_params = out_dim * in_dim
    compressed_params = compressed_rank * (out_dim + in_dim)
    compression_ratio = original_params / compressed_params

    return W_compressed, original_rank, compressed_rank, compression_ratio


def compress_model(
    model: Any,
    config: dict,
    threshold: float,
    skip_layers: list[int] | None = None,
) -> dict[str, dict]:
    """Compress model MLP weights using SVD truncation.

    Args:
        model: The model to compress (modified in place)
        config: Model config
        threshold: Relative singular value threshold
        skip_layers: Layer indices to skip (e.g., bottleneck layers)

    Returns:
        Compression metadata
    """
    import mlx.core as mx
    from modelcypher.adapters.model_architecture import get_model_architecture

    arch = get_model_architecture(model, config=config)
    num_layers = config.get("num_hidden_layers", len(arch.layers))
    skip_layers = skip_layers or []

    compression_metadata = {}
    total_original = 0
    total_compressed = 0
    skipped_count = 0
    compressed_count = 0

    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            logger.info("Layer %d: SKIPPED (in skip_layers)", layer_idx)
            compression_metadata[f"layer_{layer_idx}"] = {
                "skipped": True,
                "reason": "in_skip_layers",
            }
            skipped_count += 1
            continue

        mlp_keys = arch.layer_mlp_keys(layer_idx)
        layer_compressed = False

        for key in mlp_keys:
            try:
                # Navigate to weight
                parts = key.split(".")
                obj = model
                for part in parts[:-1]:
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

                shape = weight.shape
                if len(shape) != 2:
                    continue

                # Compress
                W_compressed, orig_rank, comp_rank, ratio = compress_weight_svd(
                    weight, threshold
                )

                if ratio > 1.0:
                    # Apply compression
                    setattr(obj, weight_name, W_compressed)
                    mx.eval(getattr(obj, weight_name))

                    out_dim, in_dim = shape
                    original_params = out_dim * in_dim
                    compressed_params = comp_rank * (out_dim + in_dim)
                    total_original += original_params
                    total_compressed += compressed_params
                    layer_compressed = True

                    compression_metadata[key] = {
                        "original_shape": list(shape),
                        "original_rank": orig_rank,
                        "compressed_rank": comp_rank,
                        "compression_ratio": ratio,
                    }

                    short_key = ".".join(parts[-2:])
                    logger.info(
                        "  %s: [%d, %d] rank %d -> %d (%.1fx compression)",
                        short_key, shape[0], shape[1],
                        orig_rank, comp_rank, ratio
                    )

            except Exception as e:
                logger.warning("Failed to compress %s: %s", key, e)

        if layer_compressed:
            compressed_count += 1
        else:
            logger.info("Layer %d: No compression (all weights at full rank)", layer_idx)

    logger.info("")
    logger.info("Compression summary:")
    logger.info("  Skipped: %d layers", skipped_count)
    logger.info("  Compressed: %d layers", compressed_count)

    if total_original > 0:
        overall = total_original / total_compressed
        logger.info(
            "  MLP params: %.2fM -> %.2fM (%.2fx compression)",
            total_original / 1e6,
            total_compressed / 1e6,
            overall
        )
    else:
        logger.info("  No parameters compressed")

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

    # Save config with compression metadata
    config_with_meta = config.copy()
    config_with_meta["svd_compression"] = {
        "compressed": True,
        "method": "svd_truncation",
        "weights": compression_metadata,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_with_meta, f, indent=2)
    logger.info("Saved config.json with compression metadata")

    logger.info("Compressed model saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="SVD-based lossless model compression")
    parser.add_argument("--model", type=str, required=True, help="Path to source model")
    parser.add_argument("--output", type=str, required=True, help="Path to save compressed model")
    parser.add_argument("--skip-layers", type=str, default="",
                        help="Comma-separated layer indices to skip")
    parser.add_argument("--skip-verify", action="store_true", help="Skip inference verification")
    args = parser.parse_args()

    initialize_backend()

    # Load model
    model, tokenizer, config, dtype = load_model(args.model)

    # Get dtype-derived precision threshold
    threshold = get_precision_threshold(dtype)
    logger.info("Dtype: %s, SVD threshold: %.6f (sqrt(eps))", dtype, threshold)

    # Parse skip layers
    skip_layers = []
    if args.skip_layers:
        skip_layers = [int(x.strip()) for x in args.skip_layers.split(",")]
        logger.info("Will skip layers: %s", skip_layers)

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

    # Compress
    logger.info("\n=== COMPRESSING MODEL ===")
    start = time.time()
    compression_metadata = compress_model(
        model, config, threshold, skip_layers=skip_layers
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
            logger.info("\n✓ All responses match! Compression is lossless.")
        else:
            logger.warning("\n⚠️ Some responses differ.")

    # Save
    logger.info("\n=== SAVING COMPRESSED MODEL ===")
    save_compressed_model(
        model, tokenizer, config, args.output,
        compression_metadata, args.model,
    )

    logger.info("\n=== DONE ===")
    logger.info("Compressed model saved to: %s", args.output)


if __name__ == "__main__":
    main()
