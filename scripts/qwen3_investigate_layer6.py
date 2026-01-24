#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Investigate Layer 6 Anomaly
"""
Layer 6 shows only 75% individual accuracy while surrounding layers
(1-5, 7-8) show 100%. Why?

Possible causes:
1. Different activation patterns
2. Higher effective rank (needs more calibration)
3. Numerical issues
4. Structural anomaly in the architecture

This script investigates layer 6 in detail.
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


def generate_calibration_prompts() -> List[str]:
    """Generate calibration prompts."""
    prompts = []
    for a in range(1, 16):
        for b in range(1, 16):
            prompts.append(f"{a} + {b} =")
    countries = ["France", "Japan", "Germany", "Italy", "Spain", "China", "India",
                 "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt"]
    for c in countries:
        prompts.append(f"The capital of {c} is")
    for kw in ["def", "class", "import", "return", "if", "for", "while"]:
        prompts.append(f"{kw} ")
    return prompts


def generate_heldout_prompts() -> List[str]:
    """Held-out test prompts."""
    return [
        "The capital of Mongolia is",
        "The capital of Nepal is",
        "25 + 37 =",
        "99 + 88 =",
        "def factorial(",
        "async def process(",
        "class Database:",
        "The history of programming",
        "Why do birds fly",
        "Scientists believe that",
        "The speed of light is",
        "Write a function to",
    ]


def analyze_layer(model, tokenizer, layer_idx: int, calibration: List[str], heldout: List[str]):
    """Deep analysis of a layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]
    layer = inner_model.layers[layer_idx]

    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx} DEEP ANALYSIS")
    print("="*70)

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

    # Fit linear model
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    A = Y_c @ np.linalg.pinv(X_c)

    # SVD analysis
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"\nCalibration samples: {len(calibration)}")
    print(f"Hidden dim: {hidden_dim}")

    # Singular value analysis
    print(f"\nSingular value analysis:")
    print(f"  Top 5 singular values: {S[:5]}")
    print(f"  Effective rank (>1% of max): {np.sum(S > 0.01 * S[0])}")
    print(f"  Effective rank (>0.1% of max): {np.sum(S > 0.001 * S[0])}")
    print(f"  Condition number: {S[0] / S[-1] if S[-1] > 0 else np.inf:.2e}")

    # Compare to adjacent layers
    print(f"\n  For comparison, checking layers 5 and 7...")

    for compare_idx in [5, 7]:
        X_comp, Y_comp = [], []
        compare_layer = inner_model.layers[compare_idx]

        for prompt in calibration[:50]:  # Quick sample
            tokens = tokenizer.encode(prompt)
            if not tokens:
                continue

            input_ids = mx.array([tokens])
            h = inner_model.embed_tokens(input_ids)
            mx.eval(h)
            mask = create_attention_mask(h, None)

            for idx, l in enumerate(inner_model.layers):
                if idx == compare_idx:
                    h_normed = compare_layer.input_layernorm(h)
                    attn_out = compare_layer.self_attn(h_normed, mask=mask, cache=None)
                    mx.eval(attn_out)
                    h_post = h + attn_out
                    h_normed2 = compare_layer.post_attention_layernorm(h_post)
                    mx.eval(h_normed2)
                    mlp_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                    X_comp.append(mlp_in)
                    mlp_out = compare_layer.mlp(h_normed2)
                    mx.eval(mlp_out)
                    Y_comp.append(np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64))
                    break
                else:
                    h = l(h, mask, None)
                    mx.eval(h)

        X_comp = np.stack(X_comp, axis=1)
        Y_comp = np.stack(Y_comp, axis=1)
        X_comp_c = X_comp - X_comp.mean(axis=1, keepdims=True)
        Y_comp_c = Y_comp - Y_comp.mean(axis=1, keepdims=True)
        A_comp = Y_comp_c @ np.linalg.pinv(X_comp_c)
        _, S_comp, _ = np.linalg.svd(A_comp, full_matrices=False)

        print(f"  Layer {compare_idx}: top SV = {S_comp[0]:.4f}, eff rank = {np.sum(S_comp > 0.01 * S_comp[0])}")

    print(f"  Layer {layer_idx}: top SV = {S[0]:.4f}, eff rank = {np.sum(S > 0.01 * S[0])}")

    # Test on held-out with detailed output
    print(f"\nHeld-out test (detailed):")

    for prompt in heldout:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        # Original
        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

        # Compressed (only this layer)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l_iter in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Get ground truth for comparison
                mlp_out_true = layer.mlp(h_normed2)
                mx.eval(mlp_out_true)
                y_true = np.array(mlp_out_true[0, -1, :].astype(mx.float32)).astype(np.float64)

                # Compressed
                x_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                x_c = x_in - X_mean.flatten()
                y_pred = (A @ x_c) + Y_mean.flatten()

                # Error
                rel_err = np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-10)

                # Replace
                mlp_out_np = np.array(mlp_out_true.astype(mx.float32))
                mlp_out_np[0, -1, :] = y_pred.astype(np.float32)
                mlp_out = mx.array(mlp_out_np).astype(h.dtype)

                h = h_post + mlp_out
                mx.eval(h)
            else:
                h = l_iter(h, mask, None)
                mx.eval(h)

        h = inner_model.norm(h)
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)
        mx.eval(logits)

        comp_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

        match = "Y" if orig_token == comp_token else "X"
        orig_str = tokenizer.decode([orig_token])
        comp_str = tokenizer.decode([comp_token])

        print(f"  {match} '{prompt[:30]:<30}' err={rel_err*100:5.1f}% orig='{orig_str}' comp='{comp_str}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    calibration = generate_calibration_prompts()
    heldout = generate_heldout_prompts()

    # Analyze layer 6 and compare
    for layer_idx in [5, 6, 7]:
        analyze_layer(model, tokenizer, layer_idx, calibration, heldout)


if __name__ == "__main__":
    main()
