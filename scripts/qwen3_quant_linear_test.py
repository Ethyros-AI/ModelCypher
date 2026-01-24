#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Quantization vs Linear MLP Approximation Test
"""
KEY QUESTION: Does quantizing the MLP weights affect our linear approximation T?

Insight from compression research:
- T = Y @ pinv(X) captures MLP behavior exactly
- Token accuracy depends on T, not individual gate/up/down weights

This script tests:
1. Compute T from original MLP
2. Compute T from quantized MLP
3. Compare: Is T_quantized similar to T_original?
4. Test token accuracy with quantized MLP

HYPOTHESIS: If T is preserved despite weight quantization,
token accuracy may be higher than expected.
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


def generate_prompts() -> List[str]:
    """Generate test prompts."""
    prompts = []
    for a in range(1, 11):
        for b in range(1, 11):
            prompts.append(f"{a} + {b} =")
    for c in ["France", "Japan", "Germany", "Italy", "Spain"]:
        prompts.append(f"The capital of {c} is")
    return prompts


def generate_heldout() -> List[str]:
    return [
        "The capital of Mongolia is",
        "99 + 88 =",
        "def factorial(",
        "Scientists believe that",
        "The history of programming",
        "Why do birds fly",
        "23 * 17 =",
        "Explain quantum computing",
    ]


def quantize_weight(W, bits: int):
    """Symmetric quantization."""
    W = np.array(W).astype(np.float32)
    abs_max = np.abs(W).max()
    if abs_max < 1e-10:
        return W.copy()
    scale = abs_max / (2**(bits-1) - 1)
    return np.round(W / scale) * scale


def derive_T_matrix(model, tokenizer, layer_idx: int, prompts: List[str], use_quantized: bool = False, bits: int = 8):
    """Derive linear transformation T for the MLP."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    layer = inner_model.layers[layer_idx]

    # Get original weights
    gate_proj = np.array(layer.mlp.gate_proj.weight.astype(mx.float32))
    up_proj = np.array(layer.mlp.up_proj.weight.astype(mx.float32))
    down_proj = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

    # Quantize if requested
    if use_quantized:
        gate_proj = quantize_weight(gate_proj, bits)
        up_proj = quantize_weight(up_proj, bits)
        down_proj = quantize_weight(down_proj, bits)

    X_list, Y_list = [], []

    for prompt in prompts:
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

                x_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x_in)

                if use_quantized:
                    # Simulate quantized MLP
                    gate = x_in @ gate_proj.T
                    up = x_in @ up_proj.T
                    silu_gate = gate * (1 / (1 + np.exp(-np.clip(gate, -500, 500))))
                    hidden = silu_gate * up
                    y_out = hidden @ down_proj.T
                else:
                    # Use original MLP
                    mlp_out = layer.mlp(h_normed2)
                    mx.eval(mlp_out)
                    y_out = np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64)

                Y_list.append(y_out)
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X = np.stack(X_list, axis=1)
    Y = np.stack(Y_list, axis=1)

    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    # SVD-based pseudoinverse
    U_x, S_x, Vt_x = np.linalg.svd(X_c, full_matrices=False)
    threshold = 1e-6 * S_x[0] if len(S_x) > 0 else 1e-6
    S_x_inv = np.where(S_x > threshold, 1.0 / S_x, 0.0)
    T = Y_c @ (Vt_x.T * S_x_inv) @ U_x.T

    return {
        'T': T,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
    }


def test_token_accuracy(model, tokenizer, layer_idx: int, rule: Dict, test_prompts: List[str]):
    """Test token accuracy with compressed MLP."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    matches = 0

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        # Original
        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

        # Compressed
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Apply T rule
                h_np = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                mlp_out_last = rule['T'] @ (h_np - rule['X_mean']) + rule['Y_mean']

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)
                mlp_out_np = np.array(mlp_out.astype(mx.float32))
                mlp_out_np[0, -1, :] = mlp_out_last.astype(np.float32)
                mlp_out = mx.array(mlp_out_np).astype(h.dtype)

                h = h_post + mlp_out
                mx.eval(h)
            else:
                h = layer(h, mask, None)
                mx.eval(h)

        h = inner_model.norm(h)
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)
        mx.eval(logits)

        comp_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
        if orig_token == comp_token:
            matches += 1

    return matches / len(test_prompts) if test_prompts else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    prompts = generate_prompts()
    heldout = generate_heldout()

    print(f"\n{'='*80}")
    print("QUANTIZATION vs LINEAR MLP APPROXIMATION")
    print("="*80)
    print(f"Calibration: {len(prompts)} prompts")
    print(f"Held-out: {len(heldout)} prompts")

    # Test layers from different regions
    test_layers = [6, 15, 20, 25, 30]

    print(f"\n{'='*80}")
    print("T MATRIX PRESERVATION UNDER QUANTIZATION")
    print("="*80)
    print(f"{'Layer':>5} {'T_err 8b':>12} {'T_err 4b':>12} {'Acc orig T':>12} {'Acc 8b T':>12} {'Acc 4b T':>12}")
    print("-"*80)

    for layer_idx in test_layers:
        # Derive T from original
        t0 = time.time()
        T_orig = derive_T_matrix(model, tokenizer, layer_idx, prompts, use_quantized=False)

        # Derive T from 8-bit quantized
        T_8b = derive_T_matrix(model, tokenizer, layer_idx, prompts, use_quantized=True, bits=8)

        # Derive T from 4-bit quantized
        T_4b = derive_T_matrix(model, tokenizer, layer_idx, prompts, use_quantized=True, bits=4)

        # Compare T matrices
        T_err_8b = np.linalg.norm(T_orig['T'] - T_8b['T']) / np.linalg.norm(T_orig['T']) * 100
        T_err_4b = np.linalg.norm(T_orig['T'] - T_4b['T']) / np.linalg.norm(T_orig['T']) * 100

        # Test token accuracy
        acc_orig = test_token_accuracy(model, tokenizer, layer_idx, T_orig, heldout)
        acc_8b = test_token_accuracy(model, tokenizer, layer_idx, T_8b, heldout)
        acc_4b = test_token_accuracy(model, tokenizer, layer_idx, T_4b, heldout)

        print(f"{layer_idx:>5} {T_err_8b:>11.2f}% {T_err_4b:>11.2f}% "
              f"{acc_orig*100:>11.1f}% {acc_8b*100:>11.1f}% {acc_4b*100:>11.1f}%")

    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print("="*80)
    print("""
INTERPRETATION:

1. T_err measures how much quantization changes the linear transformation
   - If T_err is LOW despite high weight Frobenius error, quantization is OK
   - If T_err is HIGH, the geometric relationship is distorted

2. Token accuracy is what matters for inference
   - If acc_8b ≈ acc_orig despite T_err > 0, residual connections absorb error
   - If acc_4b << acc_orig, quantization has broken the layer

3. THE BIG IDEA:
   Instead of quantizing gate/up/down separately:
   - Compute T from full-precision MLP outputs (as we do for compression)
   - Quantize T directly (one 4096×4096 matrix)
   - For transmission layers, T IS the MLP (no SiLU, no three-matrix multiply)

   This is COMPRESSION + QUANTIZATION in one step:
   - Original: gate (12288×4096) + up (12288×4096) + down (4096×12288) = 150M params
   - T matrix: 4096×4096 = 16.8M params
   - If we quantize T to 4-bit: 16.8M × 0.5 bytes = 8.4MB per layer

   For 8 transmission layers: 67MB instead of 1.2GB (18× smaller)
""")


if __name__ == "__main__":
    main()
