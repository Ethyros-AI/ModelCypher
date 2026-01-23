#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# SVD-Based Layer Compression
"""
SVD-Based Layer Compression

Uses Singular Value Decomposition to compress MLP layers.

Key insight: Layer 14 has 95.1% energy in 1 PCA component, but the energy is
in a LINEAR COMBINATION of output dimensions, not a single coordinate.
SVD captures this linear combination directly.

Usage:
    python svd_compress_layer.py \
        --model /path/to/model \
        --output /path/to/compressed \
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


def compute_svd_approximation(
    W: Any,
    rank: int,
) -> tuple[Any, Any, Any, float, float]:
    """Compute rank-k SVD approximation of weight matrix.

    Args:
        W: Weight matrix [out_dim, in_dim]
        rank: Number of singular values to keep

    Returns:
        U_k: Left singular vectors [out_dim, rank]
        S_k: Singular values [rank]
        Vt_k: Right singular vectors [rank, in_dim]
        energy_captured: Fraction of squared Frobenius norm captured
        reconstruction_error: Relative Frobenius error
    """
    import mlx.core as mx

    # Convert to float32 for numerical stability
    W_f32 = W.astype(mx.float32)
    mx.eval(W_f32)

    # Convert to numpy for SVD
    W_np = np.array(W_f32)

    # Full SVD
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)

    # Total energy (squared Frobenius norm = sum of squared singular values)
    total_energy = np.sum(S ** 2)

    # Keep top-k
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]

    # Energy captured
    captured_energy = np.sum(S_k ** 2)
    energy_fraction = captured_energy / total_energy

    # Reconstruction error
    W_approx = U_k @ np.diag(S_k) @ Vt_k
    error = np.linalg.norm(W_np - W_approx, 'fro') / np.linalg.norm(W_np, 'fro')

    # Convert back to MLX
    U_k_mx = mx.array(U_k.astype(np.float32))
    S_k_mx = mx.array(S_k.astype(np.float32))
    Vt_k_mx = mx.array(Vt_k.astype(np.float32))
    mx.eval(U_k_mx, S_k_mx, Vt_k_mx)

    return U_k_mx, S_k_mx, Vt_k_mx, energy_fraction, error


def analyze_singular_values(W: Any, top_k: int = 20) -> list[tuple[int, float, float]]:
    """Analyze singular value distribution.

    Returns list of (rank, cumulative_energy, singular_value) tuples.
    """
    import mlx.core as mx

    W_f32 = W.astype(mx.float32)
    mx.eval(W_f32)
    W_np = np.array(W_f32)

    _, S, _ = np.linalg.svd(W_np, full_matrices=False)

    total_energy = np.sum(S ** 2)
    cumulative = 0.0

    results = []
    for i in range(min(top_k, len(S))):
        cumulative += S[i] ** 2
        results.append((i + 1, cumulative / total_energy, float(S[i])))

    return results


def compress_layer_svd(
    model: Any,
    layer_idx: int,
    rank: int,
) -> tuple[float, float, int, int]:
    """Compress MLP w2 using SVD.

    Returns:
        energy_captured: Fraction of energy in rank-k approximation
        reconstruction_error: Relative Frobenius error
        original_params: Original parameter count
        compressed_params: Compressed parameter count (U + S + V)
    """
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

    if 'feed_forward' in layer_keys:
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight
    else:
        raise ValueError(f"Cannot find MLP in layer {layer_idx}")

    out_dim, in_dim = w2.shape
    original_params = out_dim * in_dim

    logger.info("W2 shape: [%d, %d] = %d params", out_dim, in_dim, original_params)

    # Analyze singular values first
    logger.info("\nSingular value analysis:")
    sv_analysis = analyze_singular_values(w2, top_k=10)
    for r, energy, sv in sv_analysis:
        logger.info("  Rank %2d: %.1f%% cumulative energy (σ=%.4f)", r, energy * 100, sv)

    # Compute SVD approximation
    U_k, S_k, Vt_k, energy_captured, error = compute_svd_approximation(w2, rank)

    # Compressed params: U [out_dim, rank] + S [rank] + Vt [rank, in_dim]
    compressed_params = out_dim * rank + rank + rank * in_dim

    logger.info("\nRank-%d approximation:", rank)
    logger.info("  Energy captured: %.1f%%", energy_captured * 100)
    logger.info("  Reconstruction error: %.1f%%", error * 100)
    logger.info("  Params: %d -> %d (%.1fx compression)",
                original_params, compressed_params, original_params / compressed_params)

    # Reconstruct and apply
    # W_approx = U @ diag(S) @ Vt
    W_approx = U_k @ mx.diag(S_k) @ Vt_k
    mx.eval(W_approx)

    # Convert back to original dtype
    W_approx = W_approx.astype(w2.dtype)
    mx.eval(W_approx)

    # Apply
    mlp['w2'].weight = W_approx
    mx.eval(model.parameters())

    return energy_captured, error, original_params, compressed_params


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
    parser = argparse.ArgumentParser(description="SVD-based layer compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output path (optional)")
    parser.add_argument("--layer", type=int, required=True, help="Layer to compress")
    parser.add_argument("--rank", type=int, default=1, help="SVD rank (default: 1)")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze singular values, don't compress")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    logger.info("\n=== Layer %d SVD Analysis ===", args.layer)

    if args.analyze_only:
        # Just analyze, don't modify
        layer = model.model.layers[args.layer]
        layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []
        if 'feed_forward' in layer_keys:
            w2 = layer['feed_forward']['w2'].weight
            logger.info("W2 shape: %s", w2.shape)
            sv_analysis = analyze_singular_values(w2, top_k=20)
            logger.info("\nSingular value spectrum:")
            for r, energy, sv in sv_analysis:
                bar = "█" * int(energy * 40)
                logger.info("  Rank %2d: [%s] %.1f%% (σ=%.4f)", r, bar.ljust(40), energy * 100, sv)
        return

    # Compress
    energy, error, orig, comp = compress_layer_svd(model, args.layer, args.rank)

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
        logger.info("\nSaving compressed model...")
        save_compressed_model(model, tokenizer, args.model, args.output)
        logger.info("Done!")
    else:
        logger.info("\nNo output path specified, model not saved.")


if __name__ == "__main__":
    main()
