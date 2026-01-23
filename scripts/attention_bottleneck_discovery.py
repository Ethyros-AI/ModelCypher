#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Attention Bottleneck Discovery
"""
Attention Bottleneck Discovery

We found MLP has 1D bottlenecks. Does attention have similar structure?

Questions:
1. What is the effective rank of Q, K, V at each layer?
2. Do Q, K, V have bottleneck layers like MLP?
3. Do Q, K, V bottleneck together or independently?

This is key to understanding if attention follows the same geometric laws as MLP.

Usage:
    python attention_bottleneck_discovery.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
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
    "red", "blue", "green", "yellow",
    "fast", "slow", "quiet", "loud",
]


@dataclass
class AttentionProfile:
    """Per-layer attention analysis."""
    layer_idx: int
    is_attention_layer: bool

    # Q metrics
    q_effective_rank: float
    q_participation_ratio: float
    q_dims_95: int

    # K metrics
    k_effective_rank: float
    k_participation_ratio: float
    k_dims_95: int

    # V metrics
    v_effective_rank: float
    v_participation_ratio: float
    v_dims_95: int


def compute_effective_rank(activations: np.ndarray) -> dict:
    """Compute effective rank metrics for activation matrix.

    Args:
        activations: [n_samples, dim] matrix

    Returns:
        Dict with effective_rank, participation_ratio, dims_95
    """
    activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)

    # Center
    mean = activations.mean(axis=0)
    centered = activations - mean

    # Covariance
    cov = (centered.T @ centered) / len(activations)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return {'effective_rank': 0, 'participation_ratio': 0, 'dims_95': 0}

    # Participation ratio
    total = np.sum(eigenvalues)
    pr = total**2 / np.sum(eigenvalues**2) if total > 0 else 0

    # Shannon effective rank
    p = eigenvalues / total
    p_safe = np.where(p > 1e-10, p, 1e-10)
    entropy = -np.sum(p * np.log(p_safe))
    eff_rank = np.exp(entropy)

    # Dims for 95% variance
    cumsum = np.cumsum(eigenvalues) / total
    dims_95 = int(np.searchsorted(cumsum, 0.95) + 1)

    return {
        'effective_rank': eff_rank,
        'participation_ratio': pr,
        'dims_95': dims_95,
    }


def get_attention_activations(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
    layer_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Get Q, K, V activations at a specific layer.

    Returns:
        (Q, K, V) arrays of shape [n_concepts, dim] or (None, None, None) if not attention
    """
    import mlx.core as mx

    layer = model.model.layers[layer_idx]

    # Check if this is an attention layer
    if not hasattr(layer, 'self_attn'):
        return None, None, None

    attn = layer.self_attn

    # Get the projections
    q_proj = attn.q_proj.weight  # [q_dim, hidden_dim]
    k_proj = attn.k_proj.weight  # [kv_dim, hidden_dim]
    v_proj = attn.v_proj.weight  # [kv_dim, hidden_dim]

    q_list = []
    k_list = []
    v_list = []

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        # Get hidden state at this layer
        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, lyr in enumerate(model.model.layers):
            if idx < layer_idx:
                result = lyr(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)
            elif idx == layer_idx:
                # Get pre-attention hidden state
                h_in = h[0, -1, :]  # Last token
                mx.eval(h_in)

                # Compute Q, K, V projections
                q = h_in @ q_proj.T
                k = h_in @ k_proj.T
                v = h_in @ v_proj.T
                mx.eval(q, k, v)

                q_list.append(np.array(q.astype(mx.float32)))
                k_list.append(np.array(k.astype(mx.float32)))
                v_list.append(np.array(v.astype(mx.float32)))
                break

    return np.stack(q_list), np.stack(k_list), np.stack(v_list)


def analyze_attention_bottlenecks(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
) -> list[AttentionProfile]:
    """Analyze attention bottleneck structure across all layers."""
    n_layers = len(model.model.layers)
    profiles = []

    for layer_idx in range(n_layers):
        Q, K, V = get_attention_activations(model, tokenizer, concepts, layer_idx)

        if Q is None:
            # Not an attention layer
            profiles.append(AttentionProfile(
                layer_idx=layer_idx,
                is_attention_layer=False,
                q_effective_rank=0, q_participation_ratio=0, q_dims_95=0,
                k_effective_rank=0, k_participation_ratio=0, k_dims_95=0,
                v_effective_rank=0, v_participation_ratio=0, v_dims_95=0,
            ))
            continue

        q_metrics = compute_effective_rank(Q)
        k_metrics = compute_effective_rank(K)
        v_metrics = compute_effective_rank(V)

        profiles.append(AttentionProfile(
            layer_idx=layer_idx,
            is_attention_layer=True,
            q_effective_rank=q_metrics['effective_rank'],
            q_participation_ratio=q_metrics['participation_ratio'],
            q_dims_95=q_metrics['dims_95'],
            k_effective_rank=k_metrics['effective_rank'],
            k_participation_ratio=k_metrics['participation_ratio'],
            k_dims_95=k_metrics['dims_95'],
            v_effective_rank=v_metrics['effective_rank'],
            v_participation_ratio=v_metrics['participation_ratio'],
            v_dims_95=v_metrics['dims_95'],
        ))

    return profiles


def main():
    parser = argparse.ArgumentParser(description="Attention bottleneck discovery")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print(f"\n{'='*80}")
    print("ATTENTION BOTTLENECK DISCOVERY")
    print("="*80)
    print(f"Model: {n_layers} layers")
    print(f"Using {len(CONCEPTS)} concepts")

    # Analyze
    profiles = analyze_attention_bottlenecks(model, tokenizer, CONCEPTS)

    # Print results
    print(f"\n{'Layer':>6} | {'Type':>8} | {'Q dim':>8} | {'K dim':>8} | {'V dim':>8} | {'Q eff.rank':>10}")
    print("-" * 70)

    attn_layers = []
    q_dims = []
    k_dims = []
    v_dims = []

    for p in profiles:
        if p.is_attention_layer:
            layer_type = "ATTN"
            attn_layers.append(p.layer_idx)
            q_dims.append(p.q_dims_95)
            k_dims.append(p.k_dims_95)
            v_dims.append(p.v_dims_95)
        else:
            layer_type = "conv"

        print(f"{p.layer_idx:>6} | {layer_type:>8} | {p.q_dims_95:>8} | {p.k_dims_95:>8} | "
              f"{p.v_dims_95:>8} | {p.q_effective_rank:>10.1f}")

    # Summary
    print(f"\n{'='*80}")
    print("ATTENTION GEOMETRY SUMMARY")
    print("="*80)

    if attn_layers:
        print(f"Attention layers: {attn_layers}")
        print(f"\nQ dimension range: {min(q_dims)} - {max(q_dims)} (avg: {np.mean(q_dims):.1f})")
        print(f"K dimension range: {min(k_dims)} - {max(k_dims)} (avg: {np.mean(k_dims):.1f})")
        print(f"V dimension range: {min(v_dims)} - {max(v_dims)} (avg: {np.mean(v_dims):.1f})")

        # Find bottleneck layers
        q_min = min(q_dims)
        k_min = min(k_dims)
        v_min = min(v_dims)

        q_bottlenecks = [attn_layers[i] for i, d in enumerate(q_dims) if d == q_min]
        k_bottlenecks = [attn_layers[i] for i, d in enumerate(k_dims) if d == k_min]
        v_bottlenecks = [attn_layers[i] for i, d in enumerate(v_dims) if d == v_min]

        print(f"\nQ bottleneck layers ({q_min}D): {q_bottlenecks}")
        print(f"K bottleneck layers ({k_min}D): {k_bottlenecks}")
        print(f"V bottleneck layers ({v_min}D): {v_bottlenecks}")

        # Do Q, K, V bottleneck together?
        qkv_same = set(q_bottlenecks) == set(k_bottlenecks) == set(v_bottlenecks)

        print(f"\n{'='*80}")
        print("ATTENTION vs MLP COMPARISON")
        print("="*80)

        if qkv_same and len(q_bottlenecks) <= 3:
            print(f"""
✓ ATTENTION HAS BOTTLENECK STRUCTURE

Q, K, V all bottleneck at the same layers: {q_bottlenecks}
This suggests attention follows SIMILAR geometric laws to MLP.

The attention bottleneck dimension is {q_min}D.
(Compare to MLP which can be 1D at bottleneck layers)
""")
        elif q_min <= 5:
            print(f"""
~ ATTENTION HAS PARTIAL BOTTLENECK

Q, K, V have different bottleneck positions:
- Q: layers {q_bottlenecks} ({q_min}D)
- K: layers {k_bottlenecks} ({k_min}D)
- V: layers {v_bottlenecks} ({v_min}D)

This suggests Q, K, V have DIFFERENT geometric roles.
""")
        else:
            print(f"""
✗ ATTENTION HAS NO CLEAR BOTTLENECK

Minimum dimensions: Q={q_min}D, K={k_min}D, V={v_min}D
These are not extreme bottlenecks like MLP's 1D.

Attention may follow DIFFERENT geometric laws than MLP.
""")
    else:
        print("No attention layers found in this model.")


if __name__ == "__main__":
    main()
