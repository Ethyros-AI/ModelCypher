#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Token Accuracy Profiler
"""
KEY INSIGHT: MLP generalization error (60-80%) != token match accuracy (100%)

The residual connections and final normalization ABSORB MLP errors.
What matters is: does compressing this layer change the output token?

This script measures TOKEN MATCH ACCURACY for each layer individually:
- Compress layer N only
- Test on held-out prompts
- Does the output token match?

This tells us which layers are ACTUALLY compressible for inference.

Usage:
    python qwen3_token_accuracy_profile.py --model /path/to/model
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

    # Math
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


def derive_mlp_rule(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive MLP linear rule for a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]
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

    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    A = Y_c @ np.linalg.pinv(X_c)

    return {
        'A': A,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'hidden_dim': hidden_dim
    }


def apply_compressed_mlp(h_normed2: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply compressed MLP rule."""
    h_centered = h_normed2 - rule['X_mean']
    y_centered = rule['A'] @ h_centered
    return y_centered + rule['Y_mean']


def test_single_layer_accuracy(model, tokenizer, rule: Dict, layer_idx: int,
                                test_prompts: List[str]) -> Dict:
    """Test token accuracy when compressing only this layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    matches = 0
    results = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        # Original output
        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

        # Compressed inference (only compress this one layer)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                # Use original attention
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Compressed MLP for last position
                h_normed2_np = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                mlp_out_last = apply_compressed_mlp(h_normed2_np, rule)

                # Original MLP for other positions
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

        # Final output
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

    print(f"\n{'='*80}")
    print("TOKEN ACCURACY PROFILE (Single Layer Compression)")
    print("="*80)
    print(f"Model: {n_layers} layers")
    print(f"Testing layers {args.start_layer}-{min(args.end_layer, n_layers-1)}")

    calibration = generate_calibration_prompts()
    heldout = generate_heldout_prompts()

    print(f"Calibration: {len(calibration)} prompts")
    print(f"Held-out: {len(heldout)} prompts")

    print(f"\n{'='*80}")
    print(f"{'Layer':>5} {'Token Acc':>12} {'Matches':>10} {'Type':>15}")
    print("-"*80)

    results = []

    for layer_idx in range(args.start_layer, min(args.end_layer + 1, n_layers)):
        t0 = time.time()

        # Derive rule
        rule = derive_mlp_rule(model, tokenizer, layer_idx, calibration)

        # Test accuracy
        accuracy = test_single_layer_accuracy(model, tokenizer, rule, layer_idx, heldout)
        t1 = time.time()

        results.append({
            'layer': layer_idx,
            'accuracy': accuracy['accuracy'],
            'matches': accuracy['matches'],
            'total': accuracy['total'],
        })

        # Classify
        acc = accuracy['accuracy']
        if acc >= 1.0:
            layer_type = "COMPRESSIBLE"
        elif acc >= 0.9:
            layer_type = "near-lossless"
        elif acc >= 0.75:
            layer_type = "moderate"
        else:
            layer_type = "NOT COMPRESS"

        print(f"{layer_idx:>5} {acc*100:>11.1f}% {accuracy['matches']:>5}/{accuracy['total']:<4} {layer_type:>15} ({t1-t0:.1f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    compressible = [r['layer'] for r in results if r['accuracy'] >= 1.0]
    near_lossless = [r['layer'] for r in results if 0.9 <= r['accuracy'] < 1.0]
    not_compress = [r['layer'] for r in results if r['accuracy'] < 0.75]

    print(f"\n100% accuracy (COMPRESSIBLE): {compressible}")
    print(f">90% accuracy (near-lossless): {near_lossless}")
    print(f"<75% accuracy (NOT compressible): {not_compress}")

    if compressible:
        print(f"\nSAFE compression range: layers {min(compressible)}-{max(compressible)}")

    # Visual
    print(f"\n{'='*80}")
    print("TOKEN ACCURACY BY LAYER")
    print("="*80)

    for r in results:
        bar_len = int(r['accuracy'] * 50)
        bar = "#" * bar_len + "." * (50 - bar_len)
        marker = " *" if r['accuracy'] >= 1.0 else ""
        print(f"L{r['layer']:02d} |{bar}| {r['accuracy']*100:5.1f}%{marker}")


if __name__ == "__main__":
    main()
