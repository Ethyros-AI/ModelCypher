#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Diagnose why Lie algebra compression still has errors
"""
DIAGNOSIS:
- T has 0.000000 reconstruction error on calibration
- But held-out still fails at 35% even with 0.89% OOS
- Why?

HYPOTHESIS 1: Extrapolation error (T overfits to calibration)
TEST: Include held-out in calibration -> should get 100%

HYPOTHESIS 2: Numerical precision
TEST: Use float64 everywhere, check for NaN/inf

HYPOTHESIS 3: The transformation itself is wrong
TEST: Check that T @ x actually equals layer output for calibration

Usage:
    python lie_algebra_diagnosis.py --model /path/to/model
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


# Small sets for fast diagnosis
CALIBRATION_PROMPTS = [
    f"The capital of {c} is" for c in [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
    ]
] + [
    f"{a} + {b} =" for a in range(1, 6) for b in range(1, 6)
] + [
    "Water freezes at", "The moon orbits", "Stars are made of",
    "Diamonds are made of", "The Great Wall of China is",
    "The answer is", "Well, actually", "If you think about it,",
    "The problem is that", "In my opinion,", "That's a great question",
]

HELD_OUT_PROMPTS = [
    "The capital of Canada is",
    "7 + 8 =",
    "Ice melts at",
    "The Earth orbits",
    "Gold is made of",
    "The solution is",
    "Actually,",
    "When you consider,",
]


def collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X (at start_layer) and Y (at end_layer) in float64."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    inputs = []
    outputs = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        if is_lfm2:
            from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
            attn_mask = create_attention_mask(h, None)
            conv_mask = create_ssm_mask(h, None)
        else:
            attn_mask = None
            conv_mask = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)  # (hidden_dim, n_samples)
    Y = np.stack(outputs, axis=1)
    return X, Y


def test_generation_with_T(model, tokenizer, prompt, T, start_layer, end_layer):
    """Test generation with full T (no truncation)."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal forward pass
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_logits = np.array(logits[0, -1, :].astype(mx.float32))
    normal_token = int(np.argmax(normal_logits))

    # Factored forward pass
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    if is_lfm2:
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
        attn_mask = create_attention_mask(h, None)
        conv_mask = create_ssm_mask(h, None)
    else:
        attn_mask = None
        conv_mask = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in  # Full precision matmul
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass  # Skip - replaced by T
        else:
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    if is_lfm2:
        h = inner_model.embedding_norm(h)
    else:
        h = inner_model.norm(h)
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_logits = np.array(logits[0, -1, :].astype(mx.float32))
    fact_token = int(np.argmax(fact_logits))

    # Additional diagnostics
    logit_diff = np.abs(normal_logits - fact_logits)
    max_logit_diff = np.max(logit_diff)
    mean_logit_diff = np.mean(logit_diff)

    return {
        'normal_token': normal_token,
        'fact_token': fact_token,
        'match': normal_token == fact_token,
        'max_logit_diff': max_logit_diff,
        'mean_logit_diff': mean_logit_diff,
        'normal_word': tokenizer.decode([normal_token]),
        'fact_word': tokenizer.decode([fact_token]),
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
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    start_layer = 3
    end_layer = n_layers - 2

    print(f"\n{'='*70}")
    print("LIE ALGEBRA COMPRESSION DIAGNOSIS")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {start_layer} -> {end_layer}")

    # TEST 1: Calibration-only (baseline)
    print(f"\n{'='*70}")
    print("TEST 1: Calibration-only T")
    print("="*70)

    X_calib, Y_calib = collect_endpoint_data(model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer)
    print(f"Calibration: {X_calib.shape[1]} samples")

    T_calib = Y_calib @ np.linalg.pinv(X_calib)

    # Check reconstruction
    Y_pred = T_calib @ X_calib
    recon_err = np.linalg.norm(Y_calib - Y_pred) / np.linalg.norm(Y_calib)
    print(f"Reconstruction error: {recon_err:.10f}")

    # Test on calibration
    print("\nCalibration prompts:")
    calib_matches = 0
    for p in CALIBRATION_PROMPTS[:10]:
        result = test_generation_with_T(model, tokenizer, p, T_calib, start_layer, end_layer)
        status = "OK" if result['match'] else "FAIL"
        print(f"  {status}: '{p[:30]}...' | logit_diff: {result['max_logit_diff']:.4f}")
        if result['match']:
            calib_matches += 1

    # Test on held-out
    print("\nHeld-out prompts:")
    held_matches = 0
    for p in HELD_OUT_PROMPTS:
        result = test_generation_with_T(model, tokenizer, p, T_calib, start_layer, end_layer)
        status = "OK" if result['match'] else "FAIL"
        print(f"  {status}: '{p[:30]}...' | logit_diff: {result['max_logit_diff']:.4f} | {result['normal_word'][:10]} vs {result['fact_word'][:10]}")
        if result['match']:
            held_matches += 1

    print(f"\nCalibration: {calib_matches}/10, Held-out: {held_matches}/{len(HELD_OUT_PROMPTS)}")

    # TEST 2: Include held-out in calibration
    print(f"\n{'='*70}")
    print("TEST 2: Include held-out in calibration")
    print("="*70)

    combined_prompts = CALIBRATION_PROMPTS + HELD_OUT_PROMPTS
    X_combined, Y_combined = collect_endpoint_data(model, tokenizer, combined_prompts, start_layer, end_layer)
    print(f"Combined: {X_combined.shape[1]} samples")

    T_combined = Y_combined @ np.linalg.pinv(X_combined)

    # Check reconstruction
    Y_pred = T_combined @ X_combined
    recon_err = np.linalg.norm(Y_combined - Y_pred) / np.linalg.norm(Y_combined)
    print(f"Reconstruction error: {recon_err:.10f}")

    # Test on held-out (now in calibration)
    print("\nHeld-out prompts (now in calibration):")
    held_matches_2 = 0
    for p in HELD_OUT_PROMPTS:
        result = test_generation_with_T(model, tokenizer, p, T_combined, start_layer, end_layer)
        status = "OK" if result['match'] else "FAIL"
        print(f"  {status}: '{p[:30]}...' | logit_diff: {result['max_logit_diff']:.4f} | {result['normal_word'][:10]} vs {result['fact_word'][:10]}")
        if result['match']:
            held_matches_2 += 1

    print(f"\nHeld-out with combined T: {held_matches_2}/{len(HELD_OUT_PROMPTS)}")

    # TEST 3: Check if T is actually correct for calibration
    print(f"\n{'='*70}")
    print("TEST 3: Verify T correctness on individual samples")
    print("="*70)

    # For each calibration sample, check if T @ x_i == y_i
    for i, p in enumerate(CALIBRATION_PROMPTS[:5]):
        x_i = X_calib[:, i]
        y_i = Y_calib[:, i]
        y_pred_i = T_calib @ x_i

        err = np.linalg.norm(y_i - y_pred_i) / np.linalg.norm(y_i)
        cos_sim = np.dot(y_i, y_pred_i) / (np.linalg.norm(y_i) * np.linalg.norm(y_pred_i))
        print(f"  Sample {i}: err={err:.10f}, cos_sim={cos_sim:.10f}")

    # TEST 4: Check numerical issues
    print(f"\n{'='*70}")
    print("TEST 4: Numerical diagnostics")
    print("="*70)

    print(f"T contains NaN: {np.any(np.isnan(T_calib))}")
    print(f"T contains Inf: {np.any(np.isinf(T_calib))}")
    print(f"T max value: {np.max(np.abs(T_calib)):.6f}")
    print(f"T min nonzero value: {np.min(np.abs(T_calib[T_calib != 0])):.10f}")

    # SVD of T
    U, S, Vh = np.linalg.svd(T_calib, full_matrices=False)
    print(f"\nT singular values:")
    print(f"  Top 5: {S[:5]}")
    print(f"  Bottom 5: {S[-5:]}")
    print(f"  Condition number: {S[0]/S[-1]:.2e}")

    # Check X condition
    _, S_x, _ = np.linalg.svd(X_calib, full_matrices=False)
    print(f"\nX singular values:")
    print(f"  Top 5: {S_x[:5]}")
    print(f"  Bottom 5: {S_x[-5:]}")
    print(f"  Condition number: {S_x[0]/S_x[-1]:.2e}")

    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print("="*70)

    if held_matches_2 > held_matches:
        print("FINDING: Including held-out in calibration IMPROVES results")
        print("CONCLUSION: The issue is extrapolation/coverage, not the algorithm")
    else:
        print("FINDING: Including held-out in calibration does NOT help")
        print("CONCLUSION: The issue is fundamental to the linear approximation")

    if np.any(np.isnan(T_calib)) or np.any(np.isinf(T_calib)):
        print("WARNING: Numerical issues detected in T")

    print(f"\nCalibration accuracy: {calib_matches}/10")
    print(f"Held-out accuracy (without): {held_matches}/{len(HELD_OUT_PROMPTS)}")
    print(f"Held-out accuracy (with): {held_matches_2}/{len(HELD_OUT_PROMPTS)}")


if __name__ == "__main__":
    main()
