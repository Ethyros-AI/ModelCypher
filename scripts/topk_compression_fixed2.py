#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Top-K Compression (FIXED v2)
"""
Top-K Compression - PROPERLY FIXED VERSION

BUG IN v1:
We were copying the INPUT h and modifying it, instead of the OUTPUT h_true.

THE FIX:
1. Run layer to get h_true (full output)
2. Modify ONLY the last position of h_true
3. Pass modified h_true to next layer

Usage:
    python topk_compression_fixed2.py --model /path/to/model
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


def topk_compress_delta(delta: np.ndarray, k: int) -> np.ndarray:
    """Keep only the top-k dimensions by magnitude."""
    if k >= len(delta):
        return delta.copy()

    abs_delta = np.abs(delta)
    topk_indices = np.argpartition(abs_delta, -k)[-k:]

    delta_compressed = np.zeros_like(delta)
    delta_compressed[topk_indices] = delta[topk_indices]

    return delta_compressed


def test_topk_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    compress_layers: list[int],
    max_tokens: int = 20,
) -> tuple[str, str, dict]:
    """Test generation with top-k compression - PROPERLY FIXED."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    normal_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        normal_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    normal_output = tokenizer.decode(normal_generated)

    # Compressed generation - PROPERLY FIXED
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)
    stats = {'total_energy_kept': [], 'dims_used': k}

    for idx, layer in enumerate(inner_model.layers):
        # Get h_in BEFORE running layer (only for stats/compression)
        h_in_np = np.array(h[0, -1, :].astype(mx.float32))

        # Run layer to get full output
        h_true = layer(h, mask, None)
        mx.eval(h_true)

        if idx in compress_layers:
            # Extract output for last position
            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))

            # Compute delta
            delta_true = h_out_np - h_in_np

            # Compress
            delta_compressed = topk_compress_delta(delta_true, k)

            # Stats
            energy_kept = np.sum(delta_compressed**2) / (np.sum(delta_true**2) + 1e-10)
            stats['total_energy_kept'].append(energy_kept)

            # Reconstruct: h_in + delta_compressed
            h_new = h_in_np + delta_compressed

            # FIXED: Start from h_true (full output), modify only last position
            h_true_np = np.array(h_true.astype(mx.float32))
            h_true_np[0, -1, :] = h_new

            h = mx.array(h_true_np).astype(h_true.dtype)
            mx.eval(h)
        else:
            # Use layer output directly
            h = h_true

    # Final norm
    h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    compressed_generated = []
    if next_token != tokenizer.eos_token_id:
        compressed_generated.append(next_token)

    # Continue normally
    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        compressed_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    compressed_output = tokenizer.decode(compressed_generated)

    return normal_output, compressed_output, stats


def main():
    parser = argparse.ArgumentParser(description="Top-K compression (fixed v2)")
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
    print("TOP-K COMPRESSION TEST (PROPERLY FIXED)")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    transmission_layers = list(range(3, 27)) if n_layers == 28 else list(range(3, n_layers - 1))
    print(f"Compressing layers: {transmission_layers[0]}-{transmission_layers[-1]}")

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    # Verify K=2048 works
    print(f"\n--- Verification: K=2048 (no compression) ---")
    all_match = True
    for prompt in test_prompts:
        normal, compressed, _ = test_topk_compression(
            model, tokenizer, prompt, 2048, transmission_layers, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = normal_first == compressed_first
        all_match = all_match and match
        status = "✓" if match else f"✗ ({normal_first}→{compressed_first})"
        print(f"  {prompt[:30]}: {status}")

    if not all_match:
        print("\n  WARNING: K=2048 doesn't match! There's still a bug.")
        return

    print("\n  All K=2048 tests pass! ✓")

    # Test different K values
    k_values = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]

    print(f"\n{'K':>6} | {'Compression':>11} | {'Energy Kept':>12} | Matches | Results")
    print("-" * 80)

    for k in k_values:
        matches = 0
        results = []
        energy_kept = []

        for prompt in test_prompts:
            normal, compressed, stats = test_topk_compression(
                model, tokenizer, prompt, k, transmission_layers, max_tokens=5
            )

            if stats['total_energy_kept']:
                energy_kept.extend(stats['total_energy_kept'])

            normal_first = normal.split()[0] if normal.split() else "(empty)"
            compressed_first = compressed.split()[0] if compressed.split() else "(empty)"

            if normal_first == compressed_first:
                matches += 1
                results.append("✓")
            else:
                results.append(f"{normal_first[:6]}→{compressed_first[:6]}")

        avg_energy = np.mean(energy_kept) if energy_kept else 0
        compression = hidden_dim / k

        print(f"{k:>6} | {compression:>10.1f}x | {avg_energy:>11.4f} | {matches}/{len(test_prompts)}     | {' | '.join(results)}")

    # Binary search
    print(f"\n{'='*80}")
    print("FINDING MINIMUM K FOR 3/3 MATCHES")
    print("="*80)

    low, high = 8, 2048
    threshold_k = 2048

    while low < high:
        mid = (low + high) // 2

        matches = 0
        for prompt in test_prompts:
            normal, compressed, _ = test_topk_compression(
                model, tokenizer, prompt, mid, transmission_layers, max_tokens=5
            )
            normal_first = normal.split()[0] if normal.split() else ""
            compressed_first = compressed.split()[0] if compressed.split() else ""
            if normal_first == compressed_first:
                matches += 1

        if matches == len(test_prompts):
            threshold_k = mid
            high = mid
        else:
            low = mid + 1

        print(f"  K={mid}: {matches}/{len(test_prompts)}")

    print(f"\n{'='*80}")
    print(f"MINIMUM K FOR CORRECT OUTPUT: {threshold_k}")
    print(f"MAXIMUM COMPRESSION: {hidden_dim / threshold_k:.1f}x")
    print("="*80)


if __name__ == "__main__":
    main()
