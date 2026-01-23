#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# PCA-Based Layer Compression
"""
PCA-Based Layer Compression

Uses Principal Component Analysis of MLP OUTPUTS to compress layers.

Key insight: The WEIGHT matrix is full-rank, but the OUTPUT activations
live in a low-dimensional subspace. We project the output to the top-k
principal components.

For layer 14: SVD of weights gives 0.5% at rank-1, but PCA of outputs
gives 95.1% at PC-1. The compression opportunity is in the output space.

Usage:
    python pca_compress_layer.py \
        --model /path/to/model \
        --layer 14 \
        --rank 1 \
        --test
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SEMANTIC_PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other", "else"],
    "quantifiers": ["one", "two", "some", "all", "much", "many", "little", "few"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "feel", "see", "hear"],
    "speech": ["say", "words", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be", "live", "die"],
    "possession": ["have", "part"],
    "logical": ["not", "maybe", "can", "because", "if"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "moment"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside", "touch"],
    "taxonomy": ["kind of", "like"],
}


def get_prime_contexts() -> list[tuple[str, str, str]]:
    """Get semantic primes with minimal contexts."""
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime
            else:
                context = prime
            contexts.append((prime, context, category))
    return contexts


def collect_mlp_outputs(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> Any:
    """Collect MLP output activations for semantic primes."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    outputs = []

    for prime, context, category in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            h = model.model.embed_tokens(input_ids)
            mx.eval(h)

            for idx, layer in enumerate(model.model.layers):
                if idx < layer_idx:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result
                    mx.eval(h)
                elif idx == layer_idx:
                    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

                    if 'operator_norm' in layer_keys:
                        norm1 = layer['operator_norm']
                        norm2 = layer['ffn_norm']
                        mlp = layer['feed_forward']
                        if 'conv' in layer_keys:
                            self_attn = layer['conv']
                        else:
                            self_attn = layer['self_attn']
                    else:
                        raise ValueError(f"Unknown layer type")

                    # Forward through attention
                    h_normed = norm1(h)
                    mx.eval(h_normed)
                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    mx.eval(attn_out)
                    h_after_attn = h + attn_out
                    mx.eval(h_after_attn)

                    # MLP
                    h_before_mlp = norm2(h_after_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)

                    # Last token output
                    out = mlp_out[0, -1, :]
                    mx.eval(out)
                    outputs.append(out)
                    break

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    return mx.stack(outputs, axis=0) if outputs else None


def compute_output_pca(outputs: Any, rank: int) -> tuple[Any, float, list[float]]:
    """Compute PCA of output activations.

    Returns:
        P: Principal component matrix [hidden_dim, rank]
        variance_captured: Fraction of variance in top-k PCs
        eigenvalues_normalized: Top-20 eigenvalues as fraction of total
    """
    import mlx.core as mx

    # Center
    mean = mx.mean(outputs, axis=0)
    centered = outputs - mean
    mx.eval(centered)

    # Covariance
    n = outputs.shape[0]
    cov = (centered.T @ centered) / n
    mx.eval(cov)

    # Convert to numpy for eigendecomposition
    cov_f32 = cov.astype(mx.float32)
    mx.eval(cov_f32)
    cov_np = np.array(cov_f32)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_np)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance analysis
    total_var = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total_var

    # Top-k principal components
    P = eigenvectors[:, :rank]
    variance_captured = cumulative[rank - 1]

    # Normalized eigenvalues (top 20)
    eigenvalues_norm = (eigenvalues[:20] / total_var).tolist()

    # Convert to MLX
    P_mx = mx.array(P.astype(np.float32))
    mx.eval(P_mx)

    return P_mx, float(variance_captured), eigenvalues_norm


def compress_layer_pca(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    rank: int,
) -> tuple[float, Any]:
    """Compress MLP output using PCA projection.

    The idea: Instead of outputting to full hidden_dim, project to top-k PCs.

    Original: mlp_out = w2 @ hidden_state
    Compressed: mlp_out_approx = P @ (P.T @ w2 @ hidden_state)
              = (P @ P.T) @ mlp_out

    This is equivalent to projecting the output onto the manifold defined
    by the top-k principal components of the output distribution.

    Returns:
        variance_captured: Fraction of variance in approximation
        P: Principal component matrix for later use
    """
    import mlx.core as mx

    logger.info("Collecting MLP outputs for layer %d...", layer_idx)
    outputs = collect_mlp_outputs(model, tokenizer, layer_idx)

    if outputs is None:
        raise ValueError(f"No outputs collected for layer {layer_idx}")

    logger.info("Collected %d output samples", outputs.shape[0])

    # Compute PCA
    logger.info("Computing PCA with rank %d...", rank)
    P, variance_captured, eigenvalues = compute_output_pca(outputs, rank)

    logger.info("\nPCA analysis:")
    logger.info("  Variance in top %d PCs: %.1f%%", rank, variance_captured * 100)
    logger.info("\n  Top eigenvalue distribution:")
    for i, ev in enumerate(eigenvalues[:10]):
        bar = "█" * int(ev * 100)
        logger.info("    PC %2d: [%s] %.1f%%", i + 1, bar.ljust(50), ev * 100)

    # Now modify the MLP to project output onto the PC subspace
    # This is done by modifying w2: w2_new = P @ P.T @ w2
    layer = model.model.layers[layer_idx]
    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

    if 'feed_forward' in layer_keys:
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight  # [hidden_dim, intermediate_dim]
    else:
        raise ValueError(f"Cannot find MLP in layer {layer_idx}")

    original_dtype = w2.dtype
    w2_f32 = w2.astype(mx.float32)
    mx.eval(w2_f32)

    # Projection: w2_new = P @ P.T @ w2
    # This ensures output is in span(P)
    proj = P @ P.T  # [hidden_dim, hidden_dim]
    mx.eval(proj)

    w2_projected = proj @ w2_f32  # [hidden_dim, intermediate_dim]
    mx.eval(w2_projected)

    # Compute error
    diff = w2_f32 - w2_projected
    error = float(mx.linalg.norm(diff)) / float(mx.linalg.norm(w2_f32))
    logger.info("\n  Weight Frobenius error: %.1f%%", error * 100)

    # Convert back and apply
    w2_new = w2_projected.astype(original_dtype)
    mx.eval(w2_new)

    mlp['w2'].weight = w2_new
    mx.eval(model.parameters())

    return variance_captured, P


def save_compressed_model(
    model: Any,
    tokenizer: Any,
    source_path: str,
    output_path: str,
):
    """Save the compressed model."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(source_path)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    flat_params = tree_flatten(model.parameters())
    weights = {k: v for k, v in flat_params}

    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    logger.info("Saved compressed model to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="PCA-based layer compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--layer", type=int, required=True, help="Layer to compress")
    parser.add_argument("--rank", type=int, default=1, help="Number of PCs to keep")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    logger.info("\n=== Layer %d PCA Compression ===", args.layer)

    # Compress
    variance, P = compress_layer_pca(model, tokenizer, args.layer, args.rank)

    # Test inference
    if args.test:
        logger.info("\nTesting compressed model...")
        prompts = [
            "The answer to 2+2 is",
            "Hello, my name is",
            "The capital of France is",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
            logger.info("  %s -> %s", prompt, output[len(prompt):][:50])

    # Save
    if args.output:
        save_compressed_model(model, tokenizer, args.model, args.output)
    else:
        logger.info("\nNo output path specified, model not saved.")


if __name__ == "__main__":
    main()
