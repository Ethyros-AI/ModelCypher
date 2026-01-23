#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Dimensionality Curve Analysis
"""
Dimensionality Curve Analysis

THE HYPOTHESIS:
Dimensionality is not discrete (0, 1, 2, 3...).
It's continuous - a CURVE through layer-space.

What we call "1D bottleneck" is the minimum of this curve.
What we call "24D processing" is a local maximum.

The curve itself is the geodesic through constraint-space.

METHOD:
1. Compute continuous effective dimension at each layer
2. Plot the curve
3. Look for smooth structure, not steps

If dimensionality is discrete: we see plateaus and jumps
If dimensionality is continuous: we see a smooth curve

Usage:
    python dimensionality_curve.py --model /path/to/model
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

        inner_model = model.model if hasattr(model, 'model') else model

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(inner_model.layers):
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


def compute_continuous_dimension(deltas: np.ndarray) -> dict:
    """Compute multiple continuous dimensionality metrics."""
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    mean = deltas.mean(axis=0)
    centered = deltas - mean
    cov = (centered.T @ centered) / len(deltas)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return {
            'participation_ratio': 0.0,
            'effective_rank': 0.0,
            'dims_50': 0,
            'dims_90': 0,
            'dims_95': 0,
            'dims_99': 0,
            'spectral_decay': 0.0,
        }

    total = np.sum(eigenvalues)

    # Participation ratio (continuous)
    pr = total**2 / np.sum(eigenvalues**2) if total > 0 else 0

    # Shannon effective rank (continuous)
    p = eigenvalues / total
    p_safe = np.where(p > 1e-10, p, 1e-10)
    entropy = -np.sum(p * np.log(p_safe))
    eff_rank = np.exp(entropy)

    # Dims for various variance thresholds
    cumsum = np.cumsum(eigenvalues) / total
    dims_50 = int(np.searchsorted(cumsum, 0.50) + 1)
    dims_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    dims_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    dims_99 = int(np.searchsorted(cumsum, 0.99) + 1)

    # Spectral decay rate (how fast eigenvalues drop)
    if len(eigenvalues) > 1:
        log_eigs = np.log(eigenvalues[:min(10, len(eigenvalues))] + 1e-10)
        indices = np.arange(len(log_eigs))
        slope, _ = np.polyfit(indices, log_eigs, 1)
        spectral_decay = -slope  # Positive = fast decay = low dimension
    else:
        spectral_decay = 0.0

    return {
        'participation_ratio': float(pr),
        'effective_rank': float(eff_rank),
        'dims_50': dims_50,
        'dims_90': dims_90,
        'dims_95': dims_95,
        'dims_99': dims_99,
        'spectral_decay': float(spectral_decay),
    }


def main():
    parser = argparse.ArgumentParser(description="Dimensionality curve analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    print(f"\n{'='*80}")
    print("DIMENSIONALITY CURVE")
    print("="*80)
    print(f"Model: {n_layers} layers")

    # Compute dimensionality at each layer
    metrics_by_layer = []

    print(f"\n{'Layer':>6} | {'Part.Ratio':>10} | {'Eff.Rank':>10} | {'D50':>5} | {'D90':>5} | {'D95':>5} | {'Decay':>8}")
    print("-" * 75)

    for layer_idx in range(n_layers):
        deltas = get_layer_delta(model, tokenizer, CONCEPTS, layer_idx)
        metrics = compute_continuous_dimension(deltas)
        metrics['layer'] = layer_idx
        metrics_by_layer.append(metrics)

        print(f"{layer_idx:>6} | {metrics['participation_ratio']:>10.3f} | "
              f"{metrics['effective_rank']:>10.3f} | {metrics['dims_50']:>5} | "
              f"{metrics['dims_90']:>5} | {metrics['dims_95']:>5} | "
              f"{metrics['spectral_decay']:>8.3f}")

    # Plot the curve (ASCII)
    print(f"\n{'='*80}")
    print("DIMENSIONALITY CURVE (Participation Ratio)")
    print("="*80)

    pr_values = [m['participation_ratio'] for m in metrics_by_layer]
    max_pr = max(pr_values) if pr_values else 1
    min_pr = min(pr_values) if pr_values else 0

    # ASCII plot
    width = 50
    for i, pr in enumerate(pr_values):
        if max_pr > min_pr:
            bar_len = int((pr - min_pr) / (max_pr - min_pr) * width)
        else:
            bar_len = width // 2
        bar = '█' * bar_len
        print(f"L{i:02d} |{bar:<{width}}| {pr:.2f}")

    # Analysis
    print(f"\n{'='*80}")
    print("CURVE ANALYSIS")
    print("="*80)

    # Find local minima
    minima = []
    for i in range(1, len(pr_values) - 1):
        if pr_values[i] < pr_values[i-1] and pr_values[i] < pr_values[i+1]:
            minima.append((i, pr_values[i]))

    # Find local maxima
    maxima = []
    for i in range(1, len(pr_values) - 1):
        if pr_values[i] > pr_values[i-1] and pr_values[i] > pr_values[i+1]:
            maxima.append((i, pr_values[i]))

    print(f"\nLocal minima (bottlenecks): {minima}")
    print(f"Local maxima (expansions): {maxima}")

    # Check for smoothness
    derivatives = np.diff(pr_values)
    second_derivatives = np.diff(derivatives)

    smoothness = 1.0 / (1.0 + np.std(second_derivatives))

    print(f"\nCurve smoothness: {smoothness:.3f}")
    print(f"  (1.0 = perfectly smooth, 0.0 = discontinuous)")

    # The insight
    print(f"\n{'='*80}")
    print("THE DIMENSIONALITY CURVE")
    print("="*80)

    if smoothness > 0.5:
        print(f"""
THE CURVE IS SMOOTH.

Dimensionality isn't discrete - it's continuous.
What we call "1D bottleneck" is the minimum of a smooth curve.
What we call "24D processing" is a local maximum.

The curve shape:
  - Minimum: {min(pr_values):.2f} at layer {pr_values.index(min(pr_values))}
  - Maximum: {max(pr_values):.2f} at layer {pr_values.index(max(pr_values))}
  - Range: {max(pr_values) - min(pr_values):.2f}

This suggests dimensionality is a GEODESIC through constraint-space.
The curve is the path of minimum constraint for information flow.
""")
    else:
        print(f"""
THE CURVE HAS DISCONTINUITIES.

There may be discrete jumps in dimensionality.
Or the measurements are noisy.

Further analysis needed.
""")

    # Derivative analysis
    print(f"\n{'='*80}")
    print("RATE OF DIMENSIONAL CHANGE")
    print("="*80)

    print(f"\n{'Transition':>15} | {'Δ Dimension':>12} | Interpretation")
    print("-" * 50)

    for i, d in enumerate(derivatives):
        if abs(d) > 1.0:
            interp = "MAJOR shift"
        elif abs(d) > 0.5:
            interp = "Moderate shift"
        else:
            interp = "Gradual"

        print(f"L{i:02d} → L{i+1:02d}       | {d:>+12.3f} | {interp}")


if __name__ == "__main__":
    main()
