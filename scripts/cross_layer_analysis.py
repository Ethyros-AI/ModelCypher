#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Cross-Layer Analysis
"""
Cross-Layer Analysis

Instead of per-layer energy, measure:
1. Which dimensions are ACTIVE across ALL layers
2. How much OVERLAP there is between layers
3. The CUMULATIVE subspace used by the residual stream

Hypothesis: Each layer might use different dimensions, but the
UNION might still be small. Or: layers might be REDUNDANT,
adding information that's already there.

Usage:
    python cross_layer_analysis.py --model /path/to/model
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


SEMANTIC_PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other", "else"],
    "quantifiers": ["one", "two", "some", "all", "much", "many", "little", "few"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "feel", "see", "hear"],
}


def get_prime_contexts() -> list[str]:
    contexts = []
    for primes in SEMANTIC_PRIMES.values():
        contexts.extend(primes)
    return contexts


def collect_all_deltas(model: Any, tokenizer: Any) -> dict[int, Any]:
    """Collect MLP deltas for ALL layers at once.

    Returns dict: layer_idx -> [n_samples, hidden_dim]
    """
    import mlx.core as mx

    contexts = get_prime_contexts()
    n_layers = len(model.model.layers)

    layer_deltas = {i: [] for i in range(n_layers)}

    for context in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            h = model.model.embed_tokens(input_ids)
            mx.eval(h)

            for idx, layer in enumerate(model.model.layers):
                h_in = mx.array(h)
                mx.eval(h_in)

                layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

                norm1 = layer['operator_norm']
                norm2 = layer['ffn_norm']
                mlp = layer['feed_forward']
                if 'conv' in layer_keys:
                    self_attn = layer['conv']
                else:
                    self_attn = layer['self_attn']

                h_normed = norm1(h)
                mx.eval(h_normed)
                attn_out = self_attn(h_normed)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                mx.eval(attn_out)
                h_attn = h + attn_out
                mx.eval(h_attn)

                h_before_mlp = norm2(h_attn)
                mx.eval(h_before_mlp)
                mlp_out = mlp(h_before_mlp)
                mx.eval(mlp_out)

                h = h_attn + mlp_out
                mx.eval(h)

                # Store MLP delta (last token)
                delta = mlp_out[0, -1, :]
                mx.eval(delta)
                layer_deltas[idx].append(delta)

        except Exception as e:
            logger.debug("Failed on '%s': %s", context, e)
            continue

    # Stack
    import mlx.core as mx
    for idx in layer_deltas:
        if layer_deltas[idx]:
            layer_deltas[idx] = mx.stack(layer_deltas[idx], axis=0)
            mx.eval(layer_deltas[idx])

    return layer_deltas


def analyze_dimension_usage(layer_deltas: dict[int, Any], threshold: float = 0.01) -> dict:
    """Analyze which dimensions are used across layers.

    For each dimension, compute:
    - Which layers use it (energy > threshold of that layer's total)
    - Total energy across all layers

    Returns analysis dict.
    """
    import mlx.core as mx

    hidden_dim = layer_deltas[0].shape[1]
    n_layers = len(layer_deltas)

    # Energy per dimension per layer
    energy_matrix = np.zeros((n_layers, hidden_dim))

    for layer_idx, deltas in layer_deltas.items():
        energy = mx.sum(deltas * deltas, axis=0)
        mx.eval(energy)
        energy_list = np.array(energy.tolist())
        total = energy_list.sum()
        if total > 0:
            energy_matrix[layer_idx] = energy_list / total

    # Which dimensions are "active" in each layer (> threshold of that layer's energy)
    active_dims_per_layer = {}
    for layer_idx in range(n_layers):
        active = np.where(energy_matrix[layer_idx] > threshold)[0]
        active_dims_per_layer[layer_idx] = set(active.tolist())

    # Union of all active dimensions
    all_active = set()
    for dims in active_dims_per_layer.values():
        all_active.update(dims)

    # Intersection (dimensions used by ALL layers)
    intersection = active_dims_per_layer[0].copy()
    for dims in active_dims_per_layer.values():
        intersection &= dims

    # Total energy per dimension (sum across layers)
    total_energy_per_dim = energy_matrix.sum(axis=0)
    total_energy_per_dim /= total_energy_per_dim.sum()

    # Rank by total energy
    sorted_dims = np.argsort(total_energy_per_dim)[::-1]

    # Cumulative energy
    cumulative = np.cumsum(total_energy_per_dim[sorted_dims])

    # Find how many dims for 95%, 99%
    dims_for_95 = np.searchsorted(cumulative, 0.95) + 1
    dims_for_99 = np.searchsorted(cumulative, 0.99) + 1

    return {
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "threshold": threshold,
        "active_per_layer": {k: len(v) for k, v in active_dims_per_layer.items()},
        "union_size": len(all_active),
        "intersection_size": len(intersection),
        "dims_for_95": dims_for_95,
        "dims_for_99": dims_for_99,
        "top_dims": sorted_dims[:20].tolist(),
        "top_energies": total_energy_per_dim[sorted_dims[:20]].tolist(),
        "energy_matrix": energy_matrix,
    }


def analyze_layer_redundancy(layer_deltas: dict[int, Any]) -> dict:
    """Measure how REDUNDANT each layer's delta is with previous layers.

    For each layer, compute:
    - How much of its delta is in the span of previous layers' deltas
    - How much is truly NEW information

    This tells us if layers are adding redundant information.
    """
    import mlx.core as mx

    n_layers = len(layer_deltas)
    hidden_dim = layer_deltas[0].shape[1]

    # For each layer, project onto subspace of all previous deltas
    redundancy = []
    new_info = []

    cumulative_subspace = None

    for layer_idx in range(n_layers):
        deltas = layer_deltas[layer_idx]
        deltas_f32 = deltas.astype(mx.float32)
        mx.eval(deltas_f32)
        deltas_np = np.array(deltas_f32)

        # Mean delta for this layer
        mean_delta = deltas_np.mean(axis=0)
        delta_norm = np.linalg.norm(mean_delta)

        if cumulative_subspace is None:
            # First layer - all information is new
            redundancy.append(0.0)
            new_info.append(1.0)
            cumulative_subspace = deltas_np.T  # [hidden_dim, n_samples]
        else:
            # Project mean_delta onto cumulative subspace
            # Using least squares: find c such that cumulative @ c ≈ mean_delta
            try:
                c, residuals, rank, s = np.linalg.lstsq(cumulative_subspace, mean_delta, rcond=None)
                projected = cumulative_subspace @ c
                residual = mean_delta - projected

                proj_norm = np.linalg.norm(projected)
                resid_norm = np.linalg.norm(residual)

                if delta_norm > 1e-8:
                    redundancy.append(proj_norm / delta_norm)
                    new_info.append(resid_norm / delta_norm)
                else:
                    redundancy.append(0.0)
                    new_info.append(0.0)
            except:
                redundancy.append(0.0)
                new_info.append(1.0)

            # Add this layer's deltas to cumulative
            cumulative_subspace = np.hstack([cumulative_subspace, deltas_np.T])

    return {
        "redundancy_per_layer": redundancy,
        "new_info_per_layer": new_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-layer analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Energy threshold for 'active' dimension")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    logger.info("Model: %d layers, %d hidden dim", n_layers, hidden_dim)

    # Collect all deltas
    logger.info("\nCollecting MLP deltas for all layers...")
    layer_deltas = collect_all_deltas(model, tokenizer)

    # Analyze dimension usage
    logger.info("\nAnalyzing dimension usage across layers...")
    usage = analyze_dimension_usage(layer_deltas, args.threshold)

    print("\n" + "=" * 80)
    print("CROSS-LAYER DIMENSION USAGE")
    print("=" * 80)
    print(f"Hidden dim: {usage['hidden_dim']}")
    print(f"Threshold: {usage['threshold']*100:.1f}% of layer energy")
    print()

    print("Active dimensions per layer:")
    for layer_idx, count in usage['active_per_layer'].items():
        bar = "█" * (count // 10)
        print(f"  Layer {layer_idx:>2}: {count:>4} dims {bar}")

    print()
    print(f"UNION of all active dims: {usage['union_size']}")
    print(f"INTERSECTION (used by ALL layers): {usage['intersection_size']}")
    print()
    print(f"Dims for 95% of TOTAL energy: {usage['dims_for_95']}")
    print(f"Dims for 99% of TOTAL energy: {usage['dims_for_99']}")
    print()

    print("Top 10 dimensions by TOTAL energy:")
    for i, (dim, energy) in enumerate(zip(usage['top_dims'][:10], usage['top_energies'][:10])):
        print(f"  {i+1}. Dim {dim}: {energy*100:.2f}%")

    # Analyze redundancy
    logger.info("\nAnalyzing layer redundancy...")
    redundancy = analyze_layer_redundancy(layer_deltas)

    print("\n" + "=" * 80)
    print("LAYER REDUNDANCY ANALYSIS")
    print("=" * 80)
    print("How much of each layer's delta is REDUNDANT with previous layers?")
    print()

    for layer_idx in range(n_layers):
        red = redundancy['redundancy_per_layer'][layer_idx]
        new = redundancy['new_info_per_layer'][layer_idx]
        red_bar = "█" * int(red * 20)
        new_bar = "░" * int(new * 20)
        print(f"  Layer {layer_idx:>2}: [{red_bar}{new_bar}] "
              f"redundant={red*100:>5.1f}%, new={new*100:>5.1f}%")

    # Summary
    total_red = sum(redundancy['redundancy_per_layer']) / n_layers
    total_new = sum(redundancy['new_info_per_layer']) / n_layers
    print()
    print(f"Average: {total_red*100:.1f}% redundant, {total_new*100:.1f}% new")


if __name__ == "__main__":
    main()
