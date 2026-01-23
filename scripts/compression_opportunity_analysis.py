#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Compression Opportunity Analysis
"""
Compression Opportunity Analysis

Unifies energy flow, alignment, and dimensionality to identify compression opportunities.

Key insight: Compression that "maintains energy conservation of 1" means:
1. For orthogonal layers (cos≈0): preserve full ||delta||² in minimal dimensions
2. For aligned/anti-aligned layers: preserve the cross term <h_in, delta> contribution

The compression opportunity is the ratio: effective_dim / hidden_dim
where effective_dim captures 100% of the energy contribution.

Usage:
    python compression_opportunity_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any

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
            elif prime in ["this", "the same", "other", "else"]:
                context = f"{prime} thing"
            elif prime in ["one", "two", "some", "all", "many", "much", "little", "few"]:
                context = f"{prime} things"
            elif prime in ["good", "bad", "big", "small", "true"]:
                context = f"It is {prime}"
            elif prime in ["think", "know", "want", "feel", "see", "hear"]:
                context = f"I {prime}"
            else:
                context = prime
            contexts.append((prime, context, category))
    return contexts


def collect_mlp_deltas(model: Any, tokenizer: Any, layer_idx: int):
    """Collect MLP deltas for all primes."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    deltas = []

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

                    # MLP delta
                    h_before_mlp = norm2(h_after_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)

                    delta = mlp_out[0, -1, :]
                    mx.eval(delta)
                    deltas.append(delta)
                    break

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    return mx.stack(deltas, axis=0) if deltas else None


def analyze_compression_opportunity(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> dict[str, Any]:
    """Analyze compression opportunity for a layer.

    Returns:
        - total_energy: sum of squared delta norms
        - dim_for_99pct: dimensions needed for 99% energy
        - dim_for_100pct: dimensions needed for 100% energy (non-zero eigenvalues)
        - energy_per_dim: cumulative energy captured by top k dims
        - compression_ratio: hidden_dim / effective_dim
    """
    import mlx.core as mx
    import numpy as np

    deltas = collect_mlp_deltas(model, tokenizer, layer_idx)
    if deltas is None:
        raise ValueError(f"No deltas collected for layer {layer_idx}")

    # Compute covariance
    mean_delta = mx.mean(deltas, axis=0)
    centered = deltas - mean_delta
    mx.eval(centered)

    # Covariance matrix
    n_samples = deltas.shape[0]
    cov = (centered.T @ centered) / n_samples
    mx.eval(cov)

    # Convert to numpy for eigendecomposition (bf16 -> float32 first)
    cov_f32 = cov.astype(mx.float32)
    mx.eval(cov_f32)
    cov_np = np.array(cov_f32)

    # Eigendecomposition (symmetric matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_np)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Energy analysis
    total_variance = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total_variance

    # Find dimensions for different energy thresholds
    dim_99 = np.searchsorted(cumulative, 0.99) + 1
    dim_999 = np.searchsorted(cumulative, 0.999) + 1
    dim_9999 = np.searchsorted(cumulative, 0.9999) + 1

    # Count non-zero eigenvalues (numerical rank)
    eps = 1e-10
    rank = np.sum(eigenvalues > eps * eigenvalues[0])

    # Top eigenvalue concentration
    top1_pct = eigenvalues[0] / total_variance * 100

    # Energy per dimension
    energy_per_dim = eigenvalues / total_variance

    # Total energy (Frobenius norm squared of deltas)
    total_energy = float(mx.sum(deltas * deltas))

    hidden_dim = deltas.shape[1]

    return {
        "total_energy": total_energy,
        "total_variance": float(total_variance),
        "hidden_dim": hidden_dim,
        "numerical_rank": int(rank),
        "dim_99": int(dim_99),
        "dim_999": int(dim_999),
        "dim_9999": int(dim_9999),
        "top1_pct": float(top1_pct),
        "top3_pct": float(np.sum(eigenvalues[:3]) / total_variance * 100),
        "top10_pct": float(np.sum(eigenvalues[:10]) / total_variance * 100),
        "compression_ratio_99": hidden_dim / dim_99,
        "compression_ratio_999": hidden_dim / dim_999,
        "compression_ratio_rank": hidden_dim / rank,
        "eigenvalues_top20": eigenvalues[:20].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compression opportunity analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=None, help="Analyze specific layer")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    if args.layer is not None:
        layers_to_analyze = [args.layer]
    else:
        layers_to_analyze = list(range(n_layers))

    print("\n" + "=" * 120)
    print("COMPRESSION OPPORTUNITY ANALYSIS - MLP Deltas")
    print("=" * 120)
    print(f"Hidden dimension: {hidden_dim}")
    print()
    print(f"{'Layer':>5} | {'Rank':>6} | {'99%':>5} | {'99.9%':>6} | {'99.99%':>7} | "
          f"{'Top1%':>7} | {'Top3%':>7} | {'Top10%':>8} | {'Compress@99%':>13}")
    print("-" * 120)

    all_data = []
    for layer_idx in layers_to_analyze:
        try:
            data = analyze_compression_opportunity(model, tokenizer, layer_idx)
            all_data.append((layer_idx, data))

            print(f"{layer_idx:>5} | {data['numerical_rank']:>6} | {data['dim_99']:>5} | "
                  f"{data['dim_999']:>6} | {data['dim_9999']:>7} | "
                  f"{data['top1_pct']:>6.1f}% | {data['top3_pct']:>6.1f}% | {data['top10_pct']:>7.1f}% | "
                  f"{data['compression_ratio_99']:>12.1f}x")

        except Exception as e:
            logger.error("Layer %d failed: %s", layer_idx, e)

    print("-" * 120)

    if all_data:
        print("\nSUMMARY:")

        # Best compression opportunities
        sorted_by_compression = sorted(all_data, key=lambda x: x[1]['compression_ratio_99'], reverse=True)

        print("\n  TOP COMPRESSION OPPORTUNITIES (by 99% energy retention):")
        for layer_idx, data in sorted_by_compression[:5]:
            print(f"    Layer {layer_idx:>2}: {data['compression_ratio_99']:>6.1f}x compression, "
                  f"top1={data['top1_pct']:.1f}%, rank={data['numerical_rank']}")

        # Find bottleneck layers (very low rank)
        print("\n  BOTTLENECK LAYERS (rank < 10):")
        for layer_idx, data in all_data:
            if data['numerical_rank'] < 10:
                print(f"    Layer {layer_idx:>2}: rank={data['numerical_rank']}, "
                      f"dims for 99%={data['dim_99']}, top1={data['top1_pct']:.1f}%")

        # Energy distribution
        print("\n  ENERGY CONCENTRATION:")
        for layer_idx, data in all_data:
            bar = "█" * int(data['top1_pct'] / 5) + "░" * (20 - int(data['top1_pct'] / 5))
            print(f"    Layer {layer_idx:>2}: [{bar}] {data['top1_pct']:>5.1f}% in dim 1")


if __name__ == "__main__":
    main()
