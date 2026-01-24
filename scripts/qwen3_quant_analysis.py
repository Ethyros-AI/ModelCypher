#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Quantization Analysis
"""
Investigate what quantization does to model geometry.

Focus on:
1. Weight matrix singular value structure
2. How quantization affects our linear MLP approximation
3. Layer-specific quantization sensitivity
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def quantize_symmetric(W: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric quantization (standard for weights)."""
    W = W.astype(np.float32)
    abs_max = np.abs(W).max()
    if abs_max < 1e-10:
        return W.copy()
    scale = abs_max / (2**(bits-1) - 1)
    W_q = np.round(W / scale) * scale
    return W_q


def analyze_weight_svd(W: np.ndarray) -> Dict:
    """Analyze weight matrix SVD structure."""
    W = W.astype(np.float64)
    S = np.linalg.svd(W, compute_uv=False)

    eff_rank = np.sum(S > 0.01 * S[0]) if len(S) > 0 else 0
    total_energy = np.sum(S**2)
    top10_pct = np.sum(S[:10]**2) / total_energy * 100 if len(S) >= 10 else 100

    return {
        'top_sv': S[0],
        'sv_10': S[9] if len(S) > 9 else 0,
        'sv_100': S[99] if len(S) > 99 else 0,
        'eff_rank': eff_rank,
        'top10_energy_pct': top10_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    print(f"\n{'='*80}")
    print("QUANTIZATION GEOMETRY ANALYSIS")
    print("="*80)

    # Part 1: Analyze MLP weight geometry across layers
    print(f"\n{'='*80}")
    print("PART 1: MLP WEIGHT GEOMETRY (down_proj)")
    print("="*80)
    print(f"{'Layer':>5} {'Top SV':>10} {'SV[10]':>10} {'SV[100]':>10} {'Eff Rank':>10} {'Top10 E%':>10}")
    print("-"*80)

    layer_data = []

    for layer_idx in range(n_layers):
        layer = inner_model.layers[layer_idx]
        down = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

        geo = analyze_weight_svd(down)
        layer_data.append((layer_idx, geo))

        print(f"{layer_idx:>5} {geo['top_sv']:>10.2f} {geo['sv_10']:>10.4f} "
              f"{geo['sv_100']:>10.4f} {geo['eff_rank']:>10} {geo['top10_energy_pct']:>9.1f}%")

    # Part 2: Quantization distortion by layer
    print(f"\n{'='*80}")
    print("PART 2: QUANTIZATION DISTORTION BY LAYER (8-bit vs 4-bit)")
    print("="*80)
    print(f"{'Layer':>5} {'8-bit Err%':>12} {'4-bit Err%':>12} {'SV Change 8b':>14} {'SV Change 4b':>14}")
    print("-"*80)

    for layer_idx in range(n_layers):
        layer = inner_model.layers[layer_idx]
        down = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

        # Quantize
        down_8b = quantize_symmetric(down, 8)
        down_4b = quantize_symmetric(down, 4)

        # Frobenius error
        err_8b = np.linalg.norm(down - down_8b) / np.linalg.norm(down) * 100
        err_4b = np.linalg.norm(down - down_4b) / np.linalg.norm(down) * 100

        # Singular value change
        S_orig = np.linalg.svd(down.astype(np.float64), compute_uv=False)
        S_8b = np.linalg.svd(down_8b.astype(np.float64), compute_uv=False)
        S_4b = np.linalg.svd(down_4b.astype(np.float64), compute_uv=False)

        sv_change_8b = (S_8b[0] - S_orig[0]) / S_orig[0] * 100
        sv_change_4b = (S_4b[0] - S_orig[0]) / S_orig[0] * 100

        print(f"{layer_idx:>5} {err_8b:>11.4f}% {err_4b:>11.2f}% {sv_change_8b:>13.4f}% {sv_change_4b:>13.2f}%")

    # Part 3: Compare layer types
    print(f"\n{'='*80}")
    print("PART 3: LAYER TYPE ANALYSIS")
    print("="*80)

    # Categorize layers based on our previous findings
    layer_types = {
        'encoder': [0, 1, 2, 3, 4, 5],
        'gate': [6],
        'transition': [7, 8, 9, 10, 11, 12, 13],
        'transmission': [14, 15, 16, 17, 18, 19, 20, 21],
        'late_trans': [22, 23, 24, 25, 26, 27, 28],
        'decoder': [29, 30, 31, 32, 33, 34, 35],
    }

    for type_name, indices in layer_types.items():
        valid_indices = [i for i in indices if i < n_layers]
        if not valid_indices:
            continue

        avg_top_sv = np.mean([layer_data[i][1]['top_sv'] for i in valid_indices])
        avg_eff_rank = np.mean([layer_data[i][1]['eff_rank'] for i in valid_indices])
        avg_top10 = np.mean([layer_data[i][1]['top10_energy_pct'] for i in valid_indices])

        print(f"\n{type_name.upper()} (layers {valid_indices[0]}-{valid_indices[-1]}):")
        print(f"  Avg top SV: {avg_top_sv:.2f}")
        print(f"  Avg eff rank: {avg_eff_rank:.0f}")
        print(f"  Avg top-10 energy: {avg_top10:.1f}%")

    # Part 4: Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS FOR GEOMETRY-AWARE QUANTIZATION")
    print("="*80)

    print("""
FINDINGS:

1. SINGULAR VALUE STRUCTURE VARIES BY LAYER TYPE
   - Our earlier finding that Layer 6 has dominant SV (168.9) should
     be reflected in the weight geometry as well
   - Transmission layers likely have more uniform SV distribution

2. QUANTIZATION ERROR IS PREDICTABLE
   - Frobenius error scales with bit reduction (4-bit ~16x worse than 8-bit)
   - But spectral error (top SV change) may not track Frobenius error

3. THE OPPORTUNITY
   Our compression research showed:
   - Transmission layers (14-21) have LINEAR MLP behavior
   - T = Y @ pinv(X) captures the transformation exactly

   HYPOTHESIS: Instead of quantizing gate/up/down separately,
   we could quantize the COMPOSED transformation T:
   - Compute T at full precision during calibration
   - Quantize T (4096 x 4096 matrix)
   - At inference: y = T @ (x - mean) + mean

   ADVANTAGES:
   - T captures the actual geometric relationship
   - Quantization errors in T are interpretable
   - Single matrix instead of 3 MLP matrices
   - May allow more aggressive quantization

4. LAYER-ADAPTIVE QUANTIZATION
   Based on our layer categorization:
   - Encoder layers (0-5): Need higher precision (position-sensitive)
   - Gate layer (6): Preserve top singular mode exactly
   - Transmission (14-21): Can use aggressive quantization or T-based
   - Decoder (29-35): Need higher precision (output-sensitive)
""")


if __name__ == "__main__":
    main()
