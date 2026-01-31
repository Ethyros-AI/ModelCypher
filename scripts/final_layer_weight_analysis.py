#!/usr/bin/env python3
"""Analyze final layer weights to find dimension recovery signature.

If dimension recovery is learned, the final layer weights should show
different properties in base models vs specialist models.

Metrics:
1. Singular value distribution (how many dimensions are "active")
2. Weight matrix rank (effective rank via entropy)
3. Sparsity patterns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def analyze_weight_matrix(w: np.ndarray, name: str) -> dict:
    """Analyze properties of a weight matrix."""
    # SVD
    try:
        U, S, Vt = np.linalg.svd(w, full_matrices=False)
    except:
        return {"name": name, "error": "SVD failed"}

    # Normalized singular values
    S_norm = S / (S[0] + 1e-10)

    # Effective rank via entropy
    S_prob = S / (np.sum(S) + 1e-10)
    entropy = -np.sum(S_prob * np.log(S_prob + 1e-10))
    effective_rank = np.exp(entropy)

    # Participation ratio
    pr = np.sum(S) ** 2 / (np.sum(S ** 2) + 1e-10)

    # How many singular values to capture 90% of variance
    cumsum = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-10)
    rank_90 = int(np.searchsorted(cumsum, 0.9) + 1)

    # Sparsity (fraction of near-zero entries)
    threshold = 0.01 * np.max(np.abs(w))
    sparsity = np.mean(np.abs(w) < threshold)

    return {
        "name": name,
        "shape": w.shape,
        "effective_rank": effective_rank,
        "participation_ratio": pr,
        "rank_90": rank_90,
        "sparsity": sparsity,
        "top_sv": float(S[0]),
        "sv_decay_10": float(S[9] / S[0]) if len(S) > 9 else 0,
        "sv_decay_50": float(S[49] / S[0]) if len(S) > 49 else 0,
    }


def analyze_final_layers(model, n_layers: int = 3) -> list[dict]:
    """Analyze the last n layers of a model."""
    import mlx.core as mx

    base = getattr(model, "model", model)
    results = []

    # Analyze final n layers
    for i in range(max(0, len(base.layers) - n_layers), len(base.layers)):
        layer = base.layers[i]
        layer_name = f"L{i+1}"

        # Find weight matrices
        if hasattr(layer, 'feed_forward') or hasattr(layer, 'mlp'):
            ff = getattr(layer, 'feed_forward', getattr(layer, 'mlp', None))
            if ff:
                # Look for projections
                for proj_name in ['w1', 'w2', 'w3', 'up_proj', 'down_proj', 'gate_proj']:
                    if hasattr(ff, proj_name):
                        w = getattr(ff, proj_name).weight
                        w_np = np.array(w.astype(mx.float32))
                        results.append(analyze_weight_matrix(w_np, f"{layer_name}.{proj_name}"))

        # Check for attention projections
        for attn_name in ['self_attn', 'attention']:
            if hasattr(layer, attn_name):
                attn = getattr(layer, attn_name)
                for proj_name in ['o_proj', 'out_proj']:
                    if hasattr(attn, proj_name):
                        w = getattr(attn, proj_name).weight
                        w_np = np.array(w.astype(mx.float32))
                        results.append(analyze_weight_matrix(w_np, f"{layer_name}.{proj_name}"))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Model paths to analyze")
    args = parser.parse_args()

    print("=" * 80)
    print("FINAL LAYER WEIGHT ANALYSIS")
    print("=" * 80)

    all_results = {}

    for model_path in args.models:
        from mlx_lm import load

        model_name = Path(model_path).name
        print(f"\n{model_name}")
        print("-" * 60)

        model, _ = load(model_path)
        results = analyze_final_layers(model)
        all_results[model_name] = results

        if not results:
            print("  No weights extracted")
            continue

        print(f"{'Weight':>25} {'Shape':>15} {'EffRank':>8} {'PR':>8} {'R90':>6} {'Sparse':>8}")
        print("-" * 80)

        for r in results:
            if "error" in r:
                print(f"{r['name']:>25} ERROR: {r['error']}")
                continue

            shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
            print(f"{r['name']:>25} {shape_str:>15} {r['effective_rank']:>8.1f} {r['participation_ratio']:>8.1f} {r['rank_90']:>6d} {r['sparsity']:>8.3f}")

    # Cross-model comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON")
        print("=" * 80)

        # Compare effective rank of down_proj/w2 (output projection)
        print("\nOutput projection effective rank (higher = more dimensional recovery):")
        for model_name, results in all_results.items():
            for r in results:
                if 'down_proj' in r['name'] or 'w2' in r['name']:
                    print(f"  {model_name:40s}: {r['effective_rank']:.1f}")


if __name__ == "__main__":
    main()
