#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Manifold Dimensionality by Layer
"""
Manifold Dimensionality by Layer

Track how the semantic manifold evolves through the network.

Questions:
1. Is the manifold dimension consistent across layers?
2. Where does compression happen (manifold shrinks)?
3. Do specific layers define the manifold structure?

Usage:
    python manifold_by_layer.py --model /path/to/model
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


SEMANTIC_CONCEPTS = [
    # Core concepts
    "apple", "orange", "fruit", "dog", "cat", "animal",
    "car", "house", "tree", "water", "fire", "sun",
    "good", "bad", "big", "small", "hot", "cold",
    "love", "hate", "think", "know", "see", "hear",
    "one", "two", "many", "all", "some", "none",
    "here", "there", "now", "then", "always", "never",
    "run", "walk", "speak", "write", "read", "sleep",
    "human", "child", "mother", "father", "friend", "enemy",
]


def compute_manifold_at_layer(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    concepts: list[str],
) -> dict:
    """Compute manifold properties at a specific layer."""
    import mlx.core as mx

    states = []

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        # Go through layers up to layer_idx
        for idx, layer in enumerate(model.model.layers):
            if idx <= layer_idx:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)
            else:
                break

        h_np = np.array(h[0, -1, :].astype(mx.float32))
        states.append(h_np)

    H = np.stack(states, axis=0)

    # Intrinsic dimensionality
    mean = H.mean(axis=0)
    centered = H - mean
    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2) if len(eigenvalues) > 0 else 0

    # Effective rank
    normalized = eigenvalues / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else eigenvalues
    normalized = normalized[normalized > 1e-15]
    entropy = -np.sum(normalized * np.log(normalized)) if len(normalized) > 0 else 0
    eff_rank = np.exp(entropy)

    # Total energy
    total_energy = np.sum(H ** 2) / len(H)

    return {
        'participation_ratio': pr,
        'effective_rank': eff_rank,
        'total_energy': total_energy,
        'top_eigenvalues': eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
    }


def compute_embedding_manifold(model: Any, tokenizer: Any, concepts: list[str]) -> dict:
    """Compute manifold at embedding layer (before any transformer layers)."""
    import mlx.core as mx

    states = []

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        h_np = np.array(h[0, -1, :].astype(mx.float32))
        states.append(h_np)

    H = np.stack(states, axis=0)

    mean = H.mean(axis=0)
    centered = H - mean
    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2) if len(eigenvalues) > 0 else 0
    total_energy = np.sum(H ** 2) / len(H)

    return {
        'participation_ratio': pr,
        'total_energy': total_energy,
    }


def main():
    parser = argparse.ArgumentParser(description="Manifold by layer")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    concepts = SEMANTIC_CONCEPTS

    print(f"\nAnalyzing manifold evolution across {n_layers} layers")
    print(f"Using {len(concepts)} semantic concepts")

    print("\n" + "=" * 70)
    print("MANIFOLD DIMENSIONALITY BY LAYER")
    print("=" * 70)
    print(f"{'Layer':>8} | {'Part.Ratio':>12} | {'Eff.Rank':>10} | {'Energy':>12}")
    print("-" * 70)

    # Embedding
    emb = compute_embedding_manifold(model, tokenizer, concepts)
    print(f"{'Embed':>8} | {emb['participation_ratio']:>12.1f} | {'N/A':>10} | {emb['total_energy']:>12.2f}")

    # Each layer
    layer_dims = []
    for layer_idx in range(n_layers):
        stats = compute_manifold_at_layer(model, tokenizer, layer_idx, concepts)
        layer_dims.append(stats['participation_ratio'])

        # Mark significant changes
        marker = ""
        if layer_idx > 0 and abs(layer_dims[-1] - layer_dims[-2]) > 1:
            if layer_dims[-1] > layer_dims[-2]:
                marker = " ↑"
            else:
                marker = " ↓"

        print(f"{layer_idx:>8} | {stats['participation_ratio']:>12.1f} | "
              f"{stats['effective_rank']:>10.1f} | {stats['total_energy']:>12.2f}{marker}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    min_dim = min(layer_dims)
    max_dim = max(layer_dims)
    final_dim = layer_dims[-1]

    print(f"Embedding manifold dim: {emb['participation_ratio']:.1f}")
    print(f"Min layer dim:          {min_dim:.1f}")
    print(f"Max layer dim:          {max_dim:.1f}")
    print(f"Final layer dim:        {final_dim:.1f}")

    # Find where compression happens
    print(f"\nManifold evolution:")
    for i, dim in enumerate(layer_dims):
        if i == 0:
            change = dim - emb['participation_ratio']
        else:
            change = dim - layer_dims[i-1]

        if abs(change) > 1:
            direction = "expands" if change > 0 else "compresses"
            print(f"  Layer {i}: {direction} by {abs(change):.1f} dims")


if __name__ == "__main__":
    main()
