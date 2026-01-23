#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Residual Stream Analysis
"""
Residual Stream Analysis

THE INSIGHT FROM COMPRESSION EXPERIMENTS:
Gram preservation ≠ Generation preservation.
Even 5% reconstruction error destroys generation.

THE HYPOTHESIS:
The transmission layers aren't doing a TRANSFORM.
They're adding small RESIDUALS.

h_out = h_in + δ  (not h_out = T @ h_in)

If each layer adds a small δ, the cumulative effect is:
h_final = h_0 + Σ δ_i

To compress, we need to understand:
1. What is the structure of δ?
2. Can we predict δ from h?
3. Is δ low-rank?

METHOD:
1. Extract h_in and h_out for each layer
2. Compute δ = h_out - h_in
3. Analyze δ: magnitude, direction, rank
4. See if δ can be predicted from h_in

Usage:
    python residual_stream_analysis.py --model /path/to/model
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
    "apple", "orange", "banana", "car", "truck", "house", "tree", "book",
    "dog", "cat", "bird", "fish", "horse", "elephant", "tiger", "whale",
    "love", "hate", "fear", "joy", "anger", "peace", "war", "truth",
    "hot", "cold", "fast", "slow", "big", "small", "good", "bad",
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "think",
    "Paris", "Tokyo", "London", "mountain", "ocean", "forest", "desert", "city",
]


def get_all_layer_residuals(
    model: Any,
    tokenizer: Any,
    word: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get (h_in, h_out, delta) for each layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(word)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    inner_model = model.model if hasattr(model, 'model') else model

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    residuals = []

    for layer in inner_model.layers:
        h_in = np.array(h[0, -1, :].astype(mx.float32))

        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

        h_out = np.array(h[0, -1, :].astype(mx.float32))
        delta = h_out - h_in

        residuals.append((h_in, h_out, delta))

    return residuals


def analyze_residual_structure(
    all_residuals: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    n_layers: int,
) -> dict:
    """Analyze the structure of residuals across all samples."""
    analysis = {
        'mean_delta_norm': [],
        'mean_h_norm': [],
        'relative_change': [],
        'delta_h_correlation': [],  # How much does delta correlate with h_in?
        'delta_rank': [],  # Effective rank of delta across samples
        'delta_direction_consistency': [],  # Do all samples have similar delta direction?
    }

    for layer_idx in range(n_layers):
        # Collect deltas and h_ins for this layer
        deltas = np.stack([r[layer_idx][2] for r in all_residuals])
        h_ins = np.stack([r[layer_idx][0] for r in all_residuals])

        # Mean norms
        delta_norms = np.linalg.norm(deltas, axis=1)
        h_norms = np.linalg.norm(h_ins, axis=1)

        analysis['mean_delta_norm'].append(float(np.mean(delta_norms)))
        analysis['mean_h_norm'].append(float(np.mean(h_norms)))
        analysis['relative_change'].append(float(np.mean(delta_norms / (h_norms + 1e-10))))

        # Delta-H correlation
        # If delta = f(h_in), they should be correlated
        # Compute average cosine similarity between delta and h_in
        correlations = []
        for i in range(len(deltas)):
            d_norm = np.linalg.norm(deltas[i])
            h_norm = np.linalg.norm(h_ins[i])
            if d_norm > 1e-10 and h_norm > 1e-10:
                corr = np.dot(deltas[i], h_ins[i]) / (d_norm * h_norm)
                correlations.append(corr)
        analysis['delta_h_correlation'].append(float(np.mean(correlations)) if correlations else 0.0)

        # Delta rank (PCA on deltas)
        deltas_clean = np.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.abs(deltas_clean) > 1e-10):
            cov = (deltas_clean.T @ deltas_clean) / len(deltas_clean)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                if len(eigenvalues) > 0:
                    total = np.sum(eigenvalues)
                    cumsum = np.cumsum(eigenvalues) / total
                    rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
                else:
                    rank_90 = 0
            except:
                rank_90 = 0
        else:
            rank_90 = 0
        analysis['delta_rank'].append(rank_90)

        # Direction consistency
        # Normalize deltas and compute mean direction
        delta_normed = deltas_clean / (np.linalg.norm(deltas_clean, axis=1, keepdims=True) + 1e-10)
        mean_direction = np.mean(delta_normed, axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)

        # Cosine similarity with mean direction
        consistencies = [np.dot(d, mean_direction) for d in delta_normed]
        analysis['delta_direction_consistency'].append(float(np.mean(consistencies)))

    return analysis


def test_residual_prediction(
    all_residuals: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    layer_idx: int,
) -> dict:
    """Test if we can predict delta from h_in using a linear model."""
    # Collect data
    h_ins = np.stack([r[layer_idx][0] for r in all_residuals])
    deltas = np.stack([r[layer_idx][2] for r in all_residuals])

    # Clean data
    h_ins = np.nan_to_num(h_ins, nan=0.0, posinf=0.0, neginf=0.0)
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # Try to predict delta from h_in: delta ≈ h_in @ W
    # This tests if the layer is approximately linear
    try:
        W, residuals, rank, s = np.linalg.lstsq(h_ins, deltas, rcond=None)
        predicted = h_ins @ W

        # Prediction error
        error = np.mean(np.linalg.norm(predicted - deltas, axis=1))
        actual_norm = np.mean(np.linalg.norm(deltas, axis=1))
        rel_error = error / (actual_norm + 1e-10)

        return {
            'prediction_error': float(rel_error),
            'W_effective_rank': int(rank),
            'is_linear': rel_error < 0.5,
        }
    except:
        return {
            'prediction_error': 1.0,
            'W_effective_rank': 0,
            'is_linear': False,
        }


def test_cumulative_residual(
    all_residuals: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    start_layer: int,
    end_layer: int,
) -> dict:
    """Test if cumulative delta can be predicted from initial state."""
    # h_final = h_start + Σ δ_i
    # Can we predict Σ δ_i from h_start?

    h_starts = []
    cumulative_deltas = []

    for residuals in all_residuals:
        h_start = residuals[start_layer][0]  # h_in at start layer
        h_end = residuals[end_layer][1]  # h_out at end layer
        cumulative = h_end - h_start

        h_starts.append(h_start)
        cumulative_deltas.append(cumulative)

    h_starts = np.stack(h_starts)
    cumulative_deltas = np.stack(cumulative_deltas)

    # Clean
    h_starts = np.nan_to_num(h_starts, nan=0.0, posinf=0.0, neginf=0.0)
    cumulative_deltas = np.nan_to_num(cumulative_deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # Magnitude analysis
    cumulative_norms = np.linalg.norm(cumulative_deltas, axis=1)
    h_norms = np.linalg.norm(h_starts, axis=1)

    # Is cumulative delta low-rank?
    cov = (cumulative_deltas.T @ cumulative_deltas) / len(cumulative_deltas)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            total = np.sum(eigenvalues)
            cumsum = np.cumsum(eigenvalues) / total
            rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
            rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)
        else:
            rank_90 = rank_99 = 0
    except:
        rank_90 = rank_99 = 0

    # Can we predict cumulative from h_start?
    try:
        W, _, _, _ = np.linalg.lstsq(h_starts, cumulative_deltas, rcond=None)
        predicted = h_starts @ W
        error = np.mean(np.linalg.norm(predicted - cumulative_deltas, axis=1))
        actual_norm = np.mean(cumulative_norms)
        pred_error = error / (actual_norm + 1e-10)
    except:
        pred_error = 1.0

    return {
        'mean_cumulative_norm': float(np.mean(cumulative_norms)),
        'relative_change': float(np.mean(cumulative_norms / (h_norms + 1e-10))),
        'cumulative_rank_90': rank_90,
        'cumulative_rank_99': rank_99,
        'prediction_error': float(pred_error),
        'is_linear': pred_error < 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Residual stream analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("RESIDUAL STREAM ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Collect residuals for all concepts
    print(f"\nCollecting residuals for {len(CONCEPTS)} concepts...")
    all_residuals = []
    for word in CONCEPTS:
        residuals = get_all_layer_residuals(model, tokenizer, word)
        all_residuals.append(residuals)

    # Analyze structure
    print(f"\n{'='*80}")
    print("LAYER-BY-LAYER RESIDUAL STRUCTURE")
    print("="*80)

    analysis = analyze_residual_structure(all_residuals, n_layers)

    print(f"\n{'Layer':>6} | {'||δ||':>10} | {'||h||':>10} | {'δ/h':>8} | {'δ∠h':>8} | {'δ rank':>7} | {'Consist':>8}")
    print("-" * 80)

    for i in range(n_layers):
        print(f"{i:>6} | {analysis['mean_delta_norm'][i]:>10.2f} | "
              f"{analysis['mean_h_norm'][i]:>10.2f} | "
              f"{analysis['relative_change'][i]:>8.4f} | "
              f"{analysis['delta_h_correlation'][i]:>8.4f} | "
              f"{analysis['delta_rank'][i]:>7} | "
              f"{analysis['delta_direction_consistency'][i]:>8.4f}")

    # Identify transmission layers
    print(f"\n{'='*80}")
    print("TRANSMISSION vs COMPUTATION LAYERS")
    print("="*80)

    transmission_layers = [i for i in range(n_layers) if analysis['relative_change'][i] < 0.05]
    computation_layers = [i for i in range(n_layers) if analysis['relative_change'][i] >= 0.15]

    print(f"\nTransmission layers (δ/h < 5%): {transmission_layers if transmission_layers else 'None'}")
    print(f"Computation layers (δ/h >= 15%): {computation_layers if computation_layers else 'None'}")

    # Test linearity
    print(f"\n{'='*80}")
    print("LINEARITY TEST: Can δ be predicted from h_in?")
    print("="*80)

    print(f"\n{'Layer':>6} | {'Pred Error':>10} | {'Linear?':>8}")
    print("-" * 35)

    linear_layers = []
    for i in range(n_layers):
        result = test_residual_prediction(all_residuals, i)
        is_linear = '✓' if result['is_linear'] else ''
        print(f"{i:>6} | {result['prediction_error']:>10.4f} | {is_linear:>8}")
        if result['is_linear']:
            linear_layers.append(i)

    print(f"\nLinear layers: {linear_layers if linear_layers else 'None'}")

    # Test cumulative residual
    if transmission_layers:
        start = transmission_layers[0]
        end = transmission_layers[-1]
    else:
        start = n_layers // 4
        end = 3 * n_layers // 4

    print(f"\n{'='*80}")
    print(f"CUMULATIVE RESIDUAL (layers {start} to {end})")
    print("="*80)

    cumulative = test_cumulative_residual(all_residuals, start, end)

    print(f"\nCumulative ||Σδ||:         {cumulative['mean_cumulative_norm']:.2f}")
    print(f"Relative change (Σδ/h):   {cumulative['relative_change']:.4f}")
    print(f"Cumulative rank (90%):    {cumulative['cumulative_rank_90']}")
    print(f"Cumulative rank (99%):    {cumulative['cumulative_rank_99']}")
    print(f"Prediction error:         {cumulative['prediction_error']:.4f}")
    print(f"Is linear:                {'Yes' if cumulative['is_linear'] else 'No'}")

    # The insight
    print(f"\n{'='*80}")
    print("THE RESIDUAL STREAM GEOMETRY")
    print("="*80)

    if transmission_layers and len(transmission_layers) > n_layers // 2:
        print(f"""
TRANSMISSION DOMINATES: {len(transmission_layers)}/{n_layers} layers

For these layers:
- δ is small relative to h (< 5%)
- The state is mostly passed through

But here's the key insight:
- Even small δ matters for generation
- 5% error × 24 layers = 120% cumulative drift (worst case)
- We need to preserve the EXACT δ, not just the relationship

THE GEOMETRY:
- h moves through the space on a nearly-straight path
- Each layer adds a small correction δ
- The corrections are NOT independent - they're coordinated

COMPRESSION STRATEGY:
- Don't try to skip layers
- Instead, factorize each layer's δ into low-rank form
- δ_i = U_i @ V_i where U_i, V_i are thin matrices
""")
    else:
        print(f"""
DISTRIBUTED COMPUTATION

All layers contribute significantly.
The residual stream is not a simple wire.

COMPRESSION STRATEGY:
- Standard low-rank approximation per layer
- Can't skip entire layers
""")

    # Final analysis: Direction consistency
    print(f"\n{'='*80}")
    print("RESIDUAL DIRECTION ANALYSIS")
    print("="*80)

    consistent_layers = [i for i in range(n_layers) if analysis['delta_direction_consistency'][i] > 0.5]
    print(f"\nConsistent direction layers (>0.5): {consistent_layers if consistent_layers else 'None'}")

    if consistent_layers:
        print(f"""
These layers push ALL inputs in a similar direction!
This suggests a global bias, not input-dependent computation.

This is KEY for compression:
- If δ ≈ constant_direction × f(||h||)
- We can replace the layer with: h + scale(h) × fixed_δ
- Parameters: just one vector (fixed_δ) + one scalar function
""")


if __name__ == "__main__":
    main()
