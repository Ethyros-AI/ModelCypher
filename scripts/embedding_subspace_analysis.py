#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Embedding Subspace Analysis
"""
Embedding Subspace Analysis

Check if token embeddings live in the same subspace as the active dimensions.

If the model's "information highway" is 11-dimensional, then:
1. Embeddings should project into those 11 dims
2. Internal computation should stay in those 11 dims
3. Output head should read from those 11 dims

This would enable 100x compression.

Usage:
    python embedding_subspace_analysis.py \
        --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def analyze_embedding_subspace(model: Any, active_dims: list[int]) -> dict:
    """Check how much of embedding energy is in active dims."""
    import mlx.core as mx

    embed_weights = model.model.embed_tokens.weight
    mx.eval(embed_weights)

    embed_np = np.array(embed_weights.astype(mx.float32))
    vocab_size, hidden_dim = embed_np.shape

    # Energy per dimension (sum over all tokens)
    dim_energy = np.sum(embed_np ** 2, axis=0)
    total_energy = np.sum(dim_energy)

    # Energy in active dims
    active_energy = sum(dim_energy[d] for d in active_dims)

    # PCA on embeddings
    mean = embed_np.mean(axis=0)
    centered = embed_np - mean
    cov = (centered.T @ centered) / vocab_size

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        total_var = np.sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues) / total_var

        dims_90 = np.searchsorted(cumsum, 0.90) + 1
        dims_95 = np.searchsorted(cumsum, 0.95) + 1
        dims_99 = np.searchsorted(cumsum, 0.99) + 1
    except:
        dims_90 = dims_95 = dims_99 = hidden_dim

    return {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'active_energy_pct': active_energy / total_energy * 100,
        'dims_90': dims_90,
        'dims_95': dims_95,
        'dims_99': dims_99,
    }


def analyze_residual_stream(model: Any, tokenizer: Any, active_dims: list[int]) -> dict:
    """Track how much of residual stream energy is in active dims at each layer."""
    import mlx.core as mx

    # Use a variety of test inputs
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, my name is Claude and I am an AI assistant.",
        "What is the meaning of life?",
        "2 + 2 = 4",
        "The capital of France is Paris.",
    ]

    n_layers = len(model.model.layers)
    layer_active_pct = {i: [] for i in range(n_layers + 1)}  # +1 for embedding

    for text in test_texts:
        try:
            tokens = tokenizer.encode(text)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            h = model.model.embed_tokens(input_ids)
            mx.eval(h)

            # After embedding
            h_np = np.array(h[0, -1, :].astype(mx.float32))
            total_e = np.sum(h_np ** 2)
            active_e = sum(h_np[d] ** 2 for d in active_dims)
            layer_active_pct[0].append(active_e / total_e * 100 if total_e > 0 else 0)

            for idx, layer in enumerate(model.model.layers):
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)

                h_np = np.array(h[0, -1, :].astype(mx.float32))
                total_e = np.sum(h_np ** 2)
                active_e = sum(h_np[d] ** 2 for d in active_dims)
                layer_active_pct[idx + 1].append(active_e / total_e * 100 if total_e > 0 else 0)

        except Exception as e:
            logger.warning("Failed on '%s': %s", text[:20], e)
            continue

    # Average
    return {i: np.mean(pcts) if pcts else 0 for i, pcts in layer_active_pct.items()}


def main():
    parser = argparse.ArgumentParser(description="Embedding subspace analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    # The 11 active dimensions we found earlier
    active_dims = [98, 126, 188, 249, 374, 391, 457, 462, 827, 896, 1009]

    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    embed_stats = analyze_embedding_subspace(model, active_dims)
    print(f"Vocab size: {embed_stats['vocab_size']}")
    print(f"Hidden dim: {embed_stats['hidden_dim']}")
    print(f"Energy in active {len(active_dims)} dims: {embed_stats['active_energy_pct']:.1f}%")
    print(f"PCA dims for 90/95/99% variance: {embed_stats['dims_90']}/{embed_stats['dims_95']}/{embed_stats['dims_99']}")

    print("\n" + "=" * 60)
    print("RESIDUAL STREAM ENERGY IN ACTIVE DIMS")
    print("=" * 60)

    stream_stats = analyze_residual_stream(model, tokenizer, active_dims)

    print(f"\n{'Position':>12} | {'Active %':>10}")
    print("-" * 30)
    print(f"{'Embedding':>12} | {stream_stats[0]:>9.1f}%")
    for layer_idx in range(len(model.model.layers)):
        print(f"{'Layer '+str(layer_idx):>12} | {stream_stats[layer_idx+1]:>9.1f}%")

    # Check if active dims dominate after layer 7
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    pre_7_avg = np.mean([stream_stats[i] for i in range(8)])
    post_7_avg = np.mean([stream_stats[i] for i in range(8, len(stream_stats))])

    print(f"Average before layer 7: {pre_7_avg:.1f}% in active dims")
    print(f"Average after layer 7:  {post_7_avg:.1f}% in active dims")

    if post_7_avg > 90:
        print("\n✓ After layer 7, residual stream is >90% in active dims!")
        print("  This suggests 11-dim compression is feasible for layers 8+")
    else:
        print(f"\n✗ Residual stream is only {post_7_avg:.1f}% in active dims")
        print("  Need to understand where the other energy goes")


if __name__ == "__main__":
    main()
