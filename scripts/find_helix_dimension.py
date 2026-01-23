#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Find Helix Dimension
"""
Find the minimum helix dimension needed for 95% explained variance at each layer.

Usage:
    python find_helix_dimension.py --model /path/to/model
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


CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "good", "bad", "love", "hate",
    # More concepts for better statistics
    "red", "blue", "green", "yellow",
    "fast", "slow", "quiet", "loud",
]


def get_layer_delta(model: Any, tokenizer: Any, concepts: list[str], layer_idx: int) -> np.ndarray:
    """Get the delta (output - input) for a layer."""
    import mlx.core as mx

    deltas = []

    for word in concepts:
        tokens = tokenizer.encode(word)
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
                h_in = np.array(h[0, -1, :].astype(mx.float32))

                result = layer(h)
                h_out_full = result[0] if isinstance(result, tuple) else result
                mx.eval(h_out_full)

                h_out = np.array(h_out_full[0, -1, :].astype(mx.float32))
                deltas.append(h_out - h_in)
                break

    return np.stack(deltas)


def find_dims_for_variance(deltas: np.ndarray, target_variance: float = 0.95) -> int:
    """Find number of PCA dimensions needed for target variance."""
    deltas = np.nan_to_num(deltas, nan=0.0)

    mean = deltas.mean(axis=0)
    centered = deltas - mean
    cov = (centered.T @ centered) / len(deltas)
    cov = np.nan_to_num(cov, nan=0.0)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0

    total = np.sum(eigenvalues)
    cumsum = np.cumsum(eigenvalues) / total

    dims_needed = np.searchsorted(cumsum, target_variance) + 1
    return int(dims_needed)


def main():
    parser = argparse.ArgumentParser(description="Find helix dimension")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("HELIX DIMENSION BY LAYER")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim}D hidden")
    print(f"Target: 95% explained variance")
    print(f"\n{'Layer':>6} | {'Dims 95%':>10} | {'Dims 99%':>10} | {'Compression':>12}")
    print("-" * 50)

    dims_95 = []
    dims_99 = []

    for layer_idx in range(n_layers):
        deltas = get_layer_delta(model, tokenizer, CONCEPTS, layer_idx)

        d95 = find_dims_for_variance(deltas, 0.95)
        d99 = find_dims_for_variance(deltas, 0.99)

        dims_95.append(d95)
        dims_99.append(d99)

        # Compression ratio: hidden_dim / d95
        compression = hidden_dim / d95 if d95 > 0 else 0

        print(f"{layer_idx:>6} | {d95:>10} | {d99:>10} | {compression:>11.1f}x")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    max_d95 = max(dims_95)
    avg_d95 = np.mean(dims_95)

    print(f"Maximum helix dimension needed (95%): {max_d95}")
    print(f"Average helix dimension: {avg_d95:.1f}")
    print(f"\nIf we use {max_d95}D helix for all layers:")
    print(f"  Compression: {hidden_dim / max_d95:.1f}x on output projections")

    # Which layers have minimal dims?
    min_dim = min(d for d in dims_95 if d > 0)
    bottleneck_layers = [i for i, d in enumerate(dims_95) if d == min_dim]
    print(f"\nBottleneck layers (smallest helix = {min_dim}D): {bottleneck_layers}")


if __name__ == "__main__":
    main()
