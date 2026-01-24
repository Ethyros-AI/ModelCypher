#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Layer Compressibility Profiler
"""
GOAL: Measure the "compressibility" of each layer independently.

For each layer, we measure:
1. Calibration error (should be ~0 for all - MLP is linear)
2. Generalization error (varies by layer - the key metric!)
3. Effective rank of transformation
4. Singular value concentration

This will identify:
- ENCODER layers: high generalization error (position-dependent)
- TRANSMISSION layers: low generalization error (compressible)
- DECODER layers: high generalization error (output-dependent)

Usage:
    python qwen3_layer_compressibility.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_calibration_prompts() -> List[str]:
    """Generate calibration prompts."""
    prompts = []

    # Math (compact, predictable)
    for a in range(1, 16):
        for b in range(1, 16):
            prompts.append(f"{a} + {b} =")

    # Geography
    countries = ["France", "Japan", "Germany", "Italy", "Spain", "China", "India",
                 "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt"]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Code
    for kw in ["def", "class", "import", "return", "if", "for", "while"]:
        prompts.append(f"{kw} ")

    return prompts


def generate_heldout_prompts() -> List[str]:
    """Generate held-out prompts (different from calibration)."""
    return [
        # Geography (different countries)
        "The capital of Mongolia is",
        "The capital of Nepal is",
        "The capital of Chile is",
        # Math (different range)
        "25 + 37 =",
        "99 + 88 =",
        # Code (different patterns)
        "def factorial(",
        "async def process(",
        "class Database:",
        # Natural language (not in calibration)
        "The history of programming",
        "Why do birds fly",
        "Scientists believe that",
        "The speed of light is",
    ]


def profile_single_layer(model, tokenizer, layer_idx: int,
                         calibration: List[str], heldout: List[str]) -> Dict:
    """Profile compressibility of a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]
    layer = inner_model.layers[layer_idx]

    # Collect calibration data
    X_list, Y_list = [], []

    for prompt in calibration:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                mlp_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(mlp_in)

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)

                mlp_out_np = np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_list.append(mlp_out_np)
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X = np.stack(X_list, axis=1)
    Y = np.stack(Y_list, axis=1)

    # Fit linear model: Y = A @ X + bias
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    A = Y_c @ np.linalg.pinv(X_c)

    # Calibration error
    Y_pred_cal = A @ X_c
    cal_error = np.linalg.norm(Y_c - Y_pred_cal) / (np.linalg.norm(Y_c) + 1e-10)

    # SVD analysis
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    eff_rank = np.sum(S > 0.01 * S[0]) if len(S) > 0 else 0

    # Singular value concentration (what % of energy in top 100 dims?)
    total_energy = np.sum(S ** 2)
    top100_energy = np.sum(S[:100] ** 2) if len(S) >= 100 else total_energy
    sv_concentration = top100_energy / (total_energy + 1e-10)

    # Test on held-out data
    heldout_errors = []

    for prompt in heldout:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Ground truth MLP output
                mlp_out_true = layer.mlp(h_normed2)
                mx.eval(mlp_out_true)
                y_true = np.array(mlp_out_true[0, -1, :].astype(mx.float32)).astype(np.float64)

                # Predicted MLP output
                x_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                x_c = x_in - X_mean.flatten()
                y_pred = (A @ x_c) + Y_mean.flatten()

                # Relative error
                rel_err = np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-10)
                heldout_errors.append(rel_err)
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    gen_error = np.mean(heldout_errors) if heldout_errors else 1.0
    gen_error_std = np.std(heldout_errors) if heldout_errors else 0.0

    return {
        'layer': layer_idx,
        'calibration_error': cal_error,
        'generalization_error': gen_error,
        'generalization_std': gen_error_std,
        'effective_rank': eff_rank,
        'sv_concentration_100': sv_concentration,
        'top_singular_value': S[0] if len(S) > 0 else 0,
        'n_calibration': len(calibration),
        'n_heldout': len(heldout),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=35)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("LAYER COMPRESSIBILITY PROFILE")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Profiling layers {args.start_layer}-{min(args.end_layer, n_layers-1)}")

    calibration = generate_calibration_prompts()
    heldout = generate_heldout_prompts()

    print(f"Calibration: {len(calibration)} prompts")
    print(f"Held-out: {len(heldout)} prompts")

    print(f"\n{'='*80}")
    print(f"{'Layer':>5} {'Cal Err':>10} {'Gen Err':>10} {'Gen Std':>10} {'Eff Rank':>10} {'SV Conc':>10}")
    print("-"*80)

    results = []

    for layer_idx in range(args.start_layer, min(args.end_layer + 1, n_layers)):
        t0 = time.time()
        profile = profile_single_layer(model, tokenizer, layer_idx, calibration, heldout)
        t1 = time.time()

        results.append(profile)

        # Classify layer type based on generalization error
        gen_err = profile['generalization_error']
        if gen_err < 0.05:
            layer_type = "TRANSMISSION"
        elif gen_err < 0.15:
            layer_type = "transition"
        else:
            layer_type = "ENCODER/DEC"

        print(f"{layer_idx:>5} {profile['calibration_error']*100:>9.4f}% "
              f"{profile['generalization_error']*100:>9.2f}% "
              f"{profile['generalization_std']*100:>9.2f}% "
              f"{profile['effective_rank']:>10} "
              f"{profile['sv_concentration_100']*100:>9.1f}% "
              f"  [{layer_type}] ({t1-t0:.1f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    gen_errors = [r['generalization_error'] for r in results]

    # Find transmission range (gen_error < 5%)
    transmission_layers = [r['layer'] for r in results if r['generalization_error'] < 0.05]
    transition_layers = [r['layer'] for r in results if 0.05 <= r['generalization_error'] < 0.15]

    print(f"\nTransmission layers (gen_err < 5%): {transmission_layers}")
    print(f"Transition layers (5% <= gen_err < 15%): {transition_layers}")

    if transmission_layers:
        print(f"\nRecommended compression range: layers {min(transmission_layers)}-{max(transmission_layers)}")
        print(f"({len(transmission_layers)} layers can be compressed with 100% accuracy)")

    # Plot-like visualization
    print(f"\n{'='*80}")
    print("GENERALIZATION ERROR BY LAYER")
    print("="*80)

    max_bar = 50
    for r in results:
        bar_len = int(min(r['generalization_error'] * 100 * 2, max_bar))
        bar = "#" * bar_len
        print(f"L{r['layer']:02d} |{bar:<50}| {r['generalization_error']*100:.2f}%")


if __name__ == "__main__":
    main()
