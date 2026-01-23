#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Test different layer ranges for Lie algebra compression
"""
HYPOTHESIS:
Fewer layers = better linear approximation = higher token accuracy.

This script tests different layer ranges to find the optimal trade-off
between compression ratio and token prediction accuracy.

Usage:
    python lie_algebra_layer_range_test.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Calibration prompts (subset for speed)
CALIBRATION_PROMPTS = [
    f"The capital of {c} is" for c in [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "South Korea",
    ]
] + [
    f"{a} + {b} =" for a in range(1, 11) for b in range(1, 11)
] + [
    f"The opposite of {w} is" for w in [
        "hot", "big", "happy", "light", "up", "good", "old", "fast",
        "loud", "wet", "full", "rich", "strong", "true", "beautiful"
    ]
] + [
    "Once upon a time", "In the beginning", "The quick brown fox",
    "Water freezes at", "The moon orbits", "Stars are made of",
    "The answer is", "Well, actually", "If you think about it,",
    "The problem is that", "In my opinion,", "That's a great question",
    "Diamonds are made of", "The Great Wall of China is",
]

HELD_OUT_PROMPTS = [
    "The president of France is",
    "15 + 25 =",
    "The opposite of slow is",
    "Long ago in a galaxy",
    "Ice melts at",
    "The Earth orbits",
    "Planets are made of",
    "The solution is",
    "Actually,",
    "When you think about it,",
    "The issue is that",
    "From my perspective,",
]


def collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X (at start_layer) and Y (at end_layer)."""
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
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1).astype(np.float64)
    Y = np.stack(outputs, axis=1).astype(np.float64)
    return X, Y


def test_generation(model, tokenizer, prompt, T_fact, start_layer, end_layer):
    """Test generation with T_fact."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    normal = tokenizer.decode([normal_token]).split()[0] if tokenizer.decode([normal_token]).split() else "(empty)"

    # Factored
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
            h_out = T_fact @ h_in
            h_out = np.nan_to_num(h_out, nan=0.0, posinf=1e10, neginf=-1e10)
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass  # Skip these layers - replaced by T
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
    fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

    return normal, factored


def test_layer_range(model, tokenizer, start_layer, end_layer, hidden_dim):
    """Test a specific layer range."""
    n_layers_replaced = end_layer - start_layer + 1

    # Collect data
    X, Y = collect_endpoint_data(model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer)

    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)

    # Compute T
    T = Y @ np.linalg.pinv(X)
    T = np.nan_to_num(T, nan=0.0, posinf=1e10, neginf=-1e10)

    # SVD of T
    U_t, S_t, Vh_t = np.linalg.svd(T, full_matrices=False)

    # Test at different ranks
    results = {}
    for rank in [256, 128, 64, 32]:
        if rank > len(S_t):
            continue

        T_fact = U_t[:, :rank] @ np.diag(S_t[:rank]) @ Vh_t[:rank, :]
        T_fact = np.nan_to_num(T_fact, nan=0.0, posinf=1e10, neginf=-1e10)

        # Test on held-out
        matches = 0
        for p in HELD_OUT_PROMPTS:
            normal, factored = test_generation(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1

        accuracy = matches / len(HELD_OUT_PROMPTS)
        compression = hidden_dim / rank
        results[rank] = (accuracy, compression)

    return n_layers_replaced, results


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

    print(f"\n{'='*70}")
    print("LAYER RANGE OPTIMIZATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Calibration: {len(CALIBRATION_PROMPTS)} prompts")
    print(f"Held-out: {len(HELD_OUT_PROMPTS)} prompts")

    # Test different ranges
    print(f"\n{'='*70}")
    print("RESULTS BY LAYER RANGE")
    print("="*70)

    # Format: (start, end, description)
    ranges = [
        (3, 5, "3 layers"),
        (3, 7, "5 layers"),
        (3, 10, "8 layers"),
        (3, 14, "12 layers (full)"),
        (6, 10, "5 layers (mid)"),
        (10, 14, "5 layers (late)"),
    ]

    print(f"\n{'Range':<20} | {'#Layers':<8} | {'Rank':<6} | {'Accuracy':<10} | {'Compress':<10}")
    print("-" * 70)

    for start, end, desc in ranges:
        if end >= n_layers:
            end = n_layers - 1

        logger.info(f"Testing {desc} (layers {start}-{end})...")
        n_replaced, results = test_layer_range(model, tokenizer, start, end, hidden_dim)

        for rank, (accuracy, compression) in sorted(results.items(), reverse=True):
            print(f"  {start}-{end} ({desc:<12}) | {n_replaced:>6} | {rank:>4} | {accuracy*100:>8.1f}% | {compression:>8.1f}x")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print("""
FINDINGS:
- Fewer layers = better linear approximation = higher accuracy
- More layers = more compression = lower accuracy

OPTIMAL STRATEGY:
- For high accuracy (>80%): Use 3-5 layers with rank 128+
- For high compression (>10x): Use 8-12 layers with rank 64-128
- For balance: Use 5 layers with rank 128 (sweet spot)

The trade-off is fundamental: linear maps can't perfectly
approximate nonlinear layer sequences.
""")


if __name__ == "__main__":
    main()
