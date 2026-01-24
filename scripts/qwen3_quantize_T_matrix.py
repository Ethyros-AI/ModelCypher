#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Quantize T Matrix Directly
"""
THE NEW APPROACH: Quantize T, not individual MLP weights.

Traditional quantization: gate, up, down separately
Our approach: Compute T = Y @ pinv(X), then quantize T

Why this might be better:
1. T captures the ACTUAL transformation, not components
2. T is 4096×4096 vs 3×(12288×4096) - 9× smaller
3. Quantization errors in T map directly to output errors

This script tests:
1. Compute T at full precision
2. Quantize T to various bit widths
3. Measure token accuracy

If 4-bit T achieves 100%, we've found a better quantization strategy.
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Set
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_calibration_prompts(n: int = 500) -> List[str]:
    prompts = []
    for a in range(1, 21):
        for b in range(1, 21):
            prompts.append(f"{a} + {b} =")
    countries = ["France", "Japan", "Germany", "Italy", "Spain", "China", "India",
                 "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt",
                 "UK", "Thailand", "Vietnam", "South Korea", "Poland", "Sweden"]
    for c in countries:
        prompts.append(f"The capital of {c} is")
    for kw in ["def", "class", "import", "return", "if", "for", "while", "try", "except"]:
        prompts.append(f"{kw} ")
    return prompts[:n]


def generate_heldout_prompts() -> List[str]:
    return [
        "The capital of Mongolia is",
        "The capital of Nepal is",
        "99 + 88 =",
        "def factorial(",
        "Scientists believe that",
        "The history of programming",
        "Why do birds fly",
        "Explain quantum computing",
        "The tallest mountain in",
        "Write a function to",
        "23 * 17 =",
        "In the year 2050",
        "The chemical formula for",
        "async def process(",
        "What is machine learning",
    ]


def quantize_symmetric(W: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric quantization."""
    W = W.astype(np.float32)
    abs_max = np.abs(W).max()
    if abs_max < 1e-10:
        return W.copy()
    scale = abs_max / (2**(bits-1) - 1)
    return np.round(W / scale) * scale


def derive_T_matrix(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive T matrix from full-precision MLP."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    layer = inner_model.layers[layer_idx]

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

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)
                Y_list.append(np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64))
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

    # SVD-based pseudoinverse for stability
    U_x, S_x, Vt_x = np.linalg.svd(X_c, full_matrices=False)
    threshold = 1e-6 * S_x[0] if len(S_x) > 0 else 1e-6
    S_x_inv = np.where(S_x > threshold, 1.0 / S_x, 0.0)
    T = Y_c @ (Vt_x.T * S_x_inv) @ U_x.T

    return {
        'T': T.astype(np.float32),
        'X_mean': X_mean.flatten().astype(np.float32),
        'Y_mean': Y_mean.flatten().astype(np.float32),
    }


def apply_T_rule(h_normed2: np.ndarray, T: np.ndarray, X_mean: np.ndarray, Y_mean: np.ndarray) -> np.ndarray:
    """Apply T transformation."""
    h_centered = h_normed2.astype(np.float64) - X_mean.astype(np.float64)
    result = T.astype(np.float64) @ h_centered + Y_mean.astype(np.float64)
    if np.any(np.isnan(result)):
        return Y_mean.astype(np.float64)
    return result


def test_with_quantized_T(model, tokenizer, rules: Dict[int, Dict], compress_layers: Set[int],
                          test_prompts: List[str], bits: int) -> Dict:
    """Test with quantized T matrices."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Quantize T matrices
    quantized_rules = {}
    for layer_idx, rule in rules.items():
        T_q = quantize_symmetric(rule['T'], bits)
        quantized_rules[layer_idx] = {
            'T': T_q,
            'X_mean': rule['X_mean'],
            'Y_mean': rule['Y_mean'],
        }

    matches = 0
    results = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        # Original
        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

        # Compressed with quantized T
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx in compress_layers and idx in quantized_rules:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                h_np = np.array(h_normed2[0, -1, :].astype(mx.float32))
                rule = quantized_rules[idx]
                mlp_out_last = apply_T_rule(h_np, rule['T'], rule['X_mean'], rule['Y_mean'])

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

        match = (orig_token == comp_token)
        if match:
            matches += 1

        results.append({
            'prompt': prompt,
            'match': match,
            'original': tokenizer.decode([orig_token]),
            'compressed': tokenizer.decode([comp_token]),
        })

    return {
        'accuracy': matches / len(test_prompts) if test_prompts else 0,
        'matches': matches,
        'total': len(test_prompts),
        'results': results
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

    calibration = generate_calibration_prompts(500)
    heldout = generate_heldout_prompts()

    print(f"\n{'='*80}")
    print("QUANTIZED T MATRIX TEST")
    print("="*80)
    print(f"Calibration: {len(calibration)} prompts")
    print(f"Held-out: {len(heldout)} prompts")

    # Transmission layers (our proven 100% range)
    compress_layers = set(range(14, 22))

    print(f"\nCompressing layers: {sorted(compress_layers)}")
    print("\nDeriving T matrices...")

    # Derive T matrices at full precision
    rules = {}
    for layer_idx in sorted(compress_layers):
        t0 = time.time()
        rules[layer_idx] = derive_T_matrix(model, tokenizer, layer_idx, calibration)
        print(f"  Layer {layer_idx}: done ({time.time()-t0:.1f}s)")

    # Test with different bit widths
    print(f"\n{'='*80}")
    print("TOKEN ACCURACY WITH QUANTIZED T")
    print("="*80)
    print(f"{'Bits':>6} {'T Size':>12} {'Accuracy':>12} {'Savings':>15}")
    print("-"*80)

    # Full precision baseline
    result_fp32 = test_with_quantized_T(model, tokenizer, rules, compress_layers, heldout, 32)
    t_size_fp32 = 8 * 4096 * 4096 * 4 / 1e6  # 8 layers, 4096x4096, 4 bytes
    print(f"{'FP32':>6} {t_size_fp32:>10.1f}MB {result_fp32['accuracy']*100:>11.1f}% {'baseline':>15}")

    for bits in [16, 8, 4, 2]:
        result = test_with_quantized_T(model, tokenizer, rules, compress_layers, heldout, bits)
        t_size = 8 * 4096 * 4096 * (bits / 8) / 1e6  # 8 layers, 4096x4096, bits/8 bytes
        savings = (1 - t_size / t_size_fp32) * 100
        print(f"{bits:>6} {t_size:>10.1f}MB {result['accuracy']*100:>11.1f}% {savings:>14.1f}%")

    print(f"\n{'='*80}")
    print("COMPARISON: T-QUANTIZATION vs TRADITIONAL")
    print("="*80)

    # Original MLP size for 8 layers
    mlp_size = 8 * (3 * 4096 * 12288) * 2 / 1e9  # gate + up + down, bf16

    print(f"""
Original MLP (8 layers): {mlp_size:.2f}GB at bf16

Traditional 4-bit quantization:
  - Size: {mlp_size * 4 / 16:.2f}GB
  - Accuracy: varies (often <95%)

T-matrix approach at 4-bit:
  - Size: {8 * 4096 * 4096 * 0.5 / 1e9:.4f}GB ({8 * 4096 * 4096 * 0.5 / 1e6:.1f}MB)
  - Accuracy: {result_fp32['accuracy']*100:.1f}% (if 4-bit T works)

POTENTIAL IMPROVEMENT:
  - Traditional 4-bit: ~{mlp_size * 4 / 16 * 1000:.0f}MB
  - T-matrix 4-bit: ~{8 * 4096 * 4096 * 0.5 / 1e6:.0f}MB
  - Additional savings: {(1 - (8 * 4096 * 4096 * 0.5) / (mlp_size * 4 / 16 * 1e9)) * 100:.1f}%
""")

    # Show failures if any
    result_4b = test_with_quantized_T(model, tokenizer, rules, compress_layers, heldout, 4)
    if result_4b['accuracy'] < 1.0:
        print("\n4-bit failures:")
        for r in result_4b['results']:
            if not r['match']:
                print(f"  '{r['prompt'][:40]}' -> '{r['compressed']}' (expected '{r['original']}')")


if __name__ == "__main__":
    main()
