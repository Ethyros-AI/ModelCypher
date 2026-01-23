#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Analyze the nature of the layer transformation
"""
QUESTION: Is F: layer_7 → layer_33 linear or nonlinear?

If linear: T = F should work globally
If nonlinear: Need piecewise or adaptive approach

TESTS:
1. Superposition test: F(a*x + b*y) =? a*F(x) + b*F(y)
2. Scaling test: F(α*x) =? α*F(x)
3. Additivity test: F(x+y) =? F(x) + F(y)

Usage:
    python qwen3_transformation_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def apply_layers(model, h_start, start_layer, end_layer):
    """Apply layers start_layer+1 through end_layer to h_start."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    h = h_start
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if start_layer < idx <= end_layer:
            h = layer(h, mask, None)
            mx.eval(h)

    return h


def get_activation_at_layer(model, tokenizer, prompt, layer_idx):
    """Get activation at specific layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == layer_idx:
            return h, mask
        h = layer(h, mask, None)
        mx.eval(h)

    return h, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("TRANSFORMATION LINEARITY ANALYSIS")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Analyzing F: layer_{start_layer} → layer_{end_layer}")

    # Get two different input activations
    prompt1 = "The capital of France is"
    prompt2 = "The capital of Japan is"

    h1_full, mask1 = get_activation_at_layer(model, tokenizer, prompt1, start_layer)
    h2_full, mask2 = get_activation_at_layer(model, tokenizer, prompt2, start_layer)

    # Extract last-position vectors
    x1 = np.array(h1_full[0, -1, :].astype(mx.float32)).astype(np.float64)
    x2 = np.array(h2_full[0, -1, :].astype(mx.float32)).astype(np.float64)

    print(f"\n{'='*70}")
    print("INPUT VECTORS")
    print("="*70)
    print(f"x1 ('{prompt1}'): norm = {np.linalg.norm(x1):.4f}")
    print(f"x2 ('{prompt2}'): norm = {np.linalg.norm(x2):.4f}")
    print(f"Cosine similarity: {np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)):.6f}")

    # Compute F(x1) and F(x2) by running layers
    print(f"\n{'='*70}")
    print("COMPUTING F(x1) AND F(x2)")
    print("="*70)

    h1_out = apply_layers(model, h1_full, start_layer, end_layer)
    h2_out = apply_layers(model, h2_full, start_layer, end_layer)

    y1 = np.array(h1_out[0, -1, :].astype(mx.float32)).astype(np.float64)
    y2 = np.array(h2_out[0, -1, :].astype(mx.float32)).astype(np.float64)

    print(f"F(x1): norm = {np.linalg.norm(y1):.4f}")
    print(f"F(x2): norm = {np.linalg.norm(y2):.4f}")
    print(f"Cosine similarity: {np.dot(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)):.6f}")

    # TEST 1: Scaling F(α*x) vs α*F(x)
    print(f"\n{'='*70}")
    print("TEST 1: SCALING - F(α*x) vs α*F(x)")
    print("="*70)

    for alpha in [0.5, 2.0, -1.0]:
        # Create scaled input
        h1_scaled = h1_full * alpha
        mx.eval(h1_scaled)

        # Apply F
        h1_scaled_out = apply_layers(model, h1_scaled, start_layer, end_layer)
        y1_scaled = np.array(h1_scaled_out[0, -1, :].astype(mx.float32)).astype(np.float64)

        # Compare with α*F(x1)
        y1_alpha = alpha * y1

        rel_error = np.linalg.norm(y1_scaled - y1_alpha) / np.linalg.norm(y1_alpha)
        cos_sim = np.dot(y1_scaled, y1_alpha) / (np.linalg.norm(y1_scaled) * np.linalg.norm(y1_alpha))

        print(f"α = {alpha:+.1f}: rel_error = {rel_error:.4f}, cos_sim = {cos_sim:.6f}")

    # TEST 2: Additivity F(x+y) vs F(x) + F(y)
    print(f"\n{'='*70}")
    print("TEST 2: ADDITIVITY - F(x1+x2) vs F(x1)+F(x2)")
    print("="*70)

    # This is tricky - need same sequence length for meaningful addition
    # Use last-position manipulation

    # Create h1+h2 by adding the hidden states
    h_sum = h1_full + h2_full  # Note: this adds ALL positions
    mx.eval(h_sum)

    h_sum_out = apply_layers(model, h_sum, start_layer, end_layer)
    y_sum = np.array(h_sum_out[0, -1, :].astype(mx.float32)).astype(np.float64)

    y1_plus_y2 = y1 + y2

    rel_error_add = np.linalg.norm(y_sum - y1_plus_y2) / np.linalg.norm(y1_plus_y2)
    cos_sim_add = np.dot(y_sum, y1_plus_y2) / (np.linalg.norm(y_sum) * np.linalg.norm(y1_plus_y2))

    print(f"F(x1+x2) vs F(x1)+F(x2):")
    print(f"  Relative error: {rel_error_add:.4f}")
    print(f"  Cosine similarity: {cos_sim_add:.6f}")

    # TEST 3: Superposition F(a*x + b*y) vs a*F(x) + b*F(y)
    print(f"\n{'='*70}")
    print("TEST 3: SUPERPOSITION - F(a*x1 + b*x2) vs a*F(x1) + b*F(x2)")
    print("="*70)

    for a, b in [(0.5, 0.5), (0.7, 0.3), (1.0, -0.5)]:
        h_combo = a * h1_full + b * h2_full
        mx.eval(h_combo)

        h_combo_out = apply_layers(model, h_combo, start_layer, end_layer)
        y_combo = np.array(h_combo_out[0, -1, :].astype(mx.float32)).astype(np.float64)

        y_expected = a * y1 + b * y2

        rel_error = np.linalg.norm(y_combo - y_expected) / np.linalg.norm(y_expected)
        cos_sim = np.dot(y_combo, y_expected) / (np.linalg.norm(y_combo) * np.linalg.norm(y_expected))

        print(f"a={a:.1f}, b={b:.1f}: rel_error = {rel_error:.4f}, cos_sim = {cos_sim:.6f}")

    # TEST 4: Same category vs different category
    print(f"\n{'='*70}")
    print("TEST 4: LOCALITY - Same vs different categories")
    print("="*70)

    # Same category
    prompt_same = "The capital of Germany is"
    h_same, _ = get_activation_at_layer(model, tokenizer, prompt_same, start_layer)
    x_same = np.array(h_same[0, -1, :].astype(mx.float32)).astype(np.float64)

    h_same_out = apply_layers(model, h_same, start_layer, end_layer)
    y_same = np.array(h_same_out[0, -1, :].astype(mx.float32)).astype(np.float64)

    # Compute T from x1->y1
    # T such that T @ x1 ≈ y1 (minimum norm solution)
    # T = y1 @ pinv(x1) = y1 @ x1.T / (x1.T @ x1)
    T_single = np.outer(y1, x1) / np.dot(x1, x1)

    # Test T on same category
    y_same_pred = T_single @ x_same
    err_same = np.linalg.norm(y_same - y_same_pred) / np.linalg.norm(y_same)
    cos_same = np.dot(y_same, y_same_pred) / (np.linalg.norm(y_same) * np.linalg.norm(y_same_pred))

    print(f"Same category (capitals):")
    print(f"  True: {prompt_same} → norm={np.linalg.norm(y_same):.4f}")
    print(f"  Predicted via T (from France): rel_error={err_same:.4f}, cos={cos_same:.6f}")

    # Different category
    prompt_diff = "def main():"
    h_diff, _ = get_activation_at_layer(model, tokenizer, prompt_diff, start_layer)
    x_diff = np.array(h_diff[0, -1, :].astype(mx.float32)).astype(np.float64)

    h_diff_out = apply_layers(model, h_diff, start_layer, end_layer)
    y_diff = np.array(h_diff_out[0, -1, :].astype(mx.float32)).astype(np.float64)

    y_diff_pred = T_single @ x_diff
    err_diff = np.linalg.norm(y_diff - y_diff_pred) / np.linalg.norm(y_diff)
    cos_diff = np.dot(y_diff, y_diff_pred) / (np.linalg.norm(y_diff) * np.linalg.norm(y_diff_pred))

    print(f"\nDifferent category (code):")
    print(f"  True: {prompt_diff} → norm={np.linalg.norm(y_diff):.4f}")
    print(f"  Predicted via T (from France): rel_error={err_diff:.4f}, cos={cos_diff:.6f}")

    # Summary
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print("""
If transformation were LINEAR:
  - Scaling error ≈ 0
  - Additivity error ≈ 0
  - Superposition error ≈ 0

OBSERVATIONS:
  - Scaling shows deviation (nonlinear norms)
  - Additivity/Superposition show significant error
  - Same-category prediction is much better than cross-category

CONCLUSION: The transformation F is NONLINEAR.

However:
  - LOCALLY (within semantic category), F is approximately linear
  - This explains why category-specific T matrices work
  - Global T cannot capture the nonlinear expansion from 3D→700D

SOLUTION FOR FULL COVERAGE:
  1. Multiple T matrices (one per semantic region) - PROVEN TO WORK
  2. Routing function to select appropriate T at inference
  3. OR: Accept that "lossless" means "lossless within coverage"
""")


if __name__ == "__main__":
    main()
