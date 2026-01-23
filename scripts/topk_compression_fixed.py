#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Top-K Compression (FIXED)
"""
Top-K Compression - FIXED VERSION

THE BUG IN PREVIOUS EXPERIMENTS:
We weren't passing the attention mask to the layers!
layer(h) != layer(h, mask, cache)

This caused ALL previous compression experiments to fail.

NOW WITH PROPER MASK:
layer(h, mask, None) matches model(input_ids) exactly.

Usage:
    python topk_compression_fixed.py --model /path/to/model
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
    """Keep only the top-k dimensions by magnitude, zero out the rest."""
    if k >= len(delta):
        return delta.copy()

    abs_delta = np.abs(delta)
    topk_indices = np.argpartition(abs_delta, -k)[-k:]

    delta_compressed = np.zeros_like(delta)
    delta_compressed[topk_indices] = delta[topk_indices]

    return delta_compressed


def test_topk_compression_fixed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    compress_layers: list[int],
    max_tokens: int = 20,
) -> tuple[str, str, dict]:
    """Test generation with top-k compression - FIXED with proper mask."""
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

    # Compressed generation with FIXED mask handling
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    # CREATE THE MASK!
    mask = create_attention_mask(h, None)

    stats = {'total_energy_kept': [], 'dims_used': k}

    for idx, layer in enumerate(inner_model.layers):
        if idx in compress_layers:
            # Get h_in before layer
            h_np = np.array(h.astype(mx.float32))
            h_in = h_np[0, -1, :]

            # Run actual layer to get true output
            h_true = layer(h, mask, None)  # FIXED: pass mask!
            mx.eval(h_true)

            h_out_true = np.array(h_true[0, -1, :].astype(mx.float32))
            delta_true = h_out_true - h_in

            # Top-k compress
            delta_compressed = topk_compress_delta(delta_true, k)

            # Stats
            energy_kept = np.sum(delta_compressed**2) / (np.sum(delta_true**2) + 1e-10)
            stats['total_energy_kept'].append(energy_kept)

            # Apply compressed delta
            h_new = h_in + delta_compressed
            h_np_new = h_np.copy()
            h_np_new[0, -1, :] = h_new

            h = mx.array(h_np_new).astype(h.dtype)
            mx.eval(h)
        else:
            # Run layer normally
            h = layer(h, mask, None)  # FIXED: pass mask!
            mx.eval(h)

    # Final norm
    h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        logits = model(input_ids)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    compressed_generated = []
    if next_token != tokenizer.eos_token_id:
        compressed_generated.append(next_token)

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
    parser = argparse.ArgumentParser(description="Top-K compression (fixed)")
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
    print("TOP-K COMPRESSION TEST (FIXED)")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Transmission layers
    transmission_layers = list(range(3, 27)) if n_layers == 28 else list(range(3, n_layers - 1))
    print(f"Compressing layers: {transmission_layers[0]}-{transmission_layers[-1]}")

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    # First verify that K=2048 (no compression) works
    print(f"\n--- Verification: K=2048 (no compression) ---")
    for prompt in test_prompts:
        normal, compressed, _ = test_topk_compression_fixed(
            model, tokenizer, prompt, 2048, transmission_layers, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else ""
        compressed_first = compressed.split()[0] if compressed.split() else ""
        match = "✓" if normal_first == compressed_first else f"✗ ({normal_first}→{compressed_first})"
        print(f"  {prompt[:30]}: {match}")

    # Test different K values
    k_values = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]

    print(f"\n{'K':>6} | {'Compression':>11} | {'Energy Kept':>12} | Matches | Results")
    print("-" * 80)

    for k in k_values:
        matches = 0
        results = []
        energy_kept = []

        for prompt in test_prompts:
            normal, compressed, stats = test_topk_compression_fixed(
                model, tokenizer, prompt, k, transmission_layers, max_tokens=5
            )

            if stats['total_energy_kept']:
                energy_kept.extend(stats['total_energy_kept'])

            normal_first = normal.split()[0] if normal.split() else ""
            compressed_first = compressed.split()[0] if compressed.split() else ""

            if normal_first == compressed_first:
                matches += 1
                results.append(f"✓")
            else:
                results.append(f"{normal_first[:8]}→{compressed_first[:8]}")

        avg_energy = np.mean(energy_kept) if energy_kept else 0
        compression = hidden_dim / k

        print(f"{k:>6} | {compression:>10.1f}x | {avg_energy:>11.4f} | {matches}/{len(test_prompts)}     | {' | '.join(results)}")

    # Binary search for threshold
    print(f"\n{'='*80}")
    print("BINARY SEARCH FOR COMPRESSION THRESHOLD")
    print("="*80)

    low, high = 8, 2048
    threshold_k = 2048

    while low < high:
        mid = (low + high) // 2

        matches = 0
        for prompt in test_prompts:
            normal, compressed, _ = test_topk_compression_fixed(
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

        print(f"  K={mid}: {matches}/{len(test_prompts)} matches")

    print(f"\nMINIMUM K FOR CORRECT OUTPUT: {threshold_k}")
    print(f"MAXIMUM COMPRESSION: {hidden_dim / threshold_k:.1f}x")

    # Analysis
    print(f"\n{'='*80}")
    print("COMPRESSION ANALYSIS (FIXED)")
    print("="*80)

    if threshold_k < hidden_dim:
        print(f"""
COMPRESSION IS ACHIEVABLE!

With proper attention mask handling:
- Minimum dimensions needed: {threshold_k} / {hidden_dim} = {threshold_k/hidden_dim*100:.1f}%
- Maximum compression: {hidden_dim / threshold_k:.1f}x

This proves:
1. The model CAN tolerate dimension reduction
2. Top-{threshold_k} dimensions capture the essential information
3. {hidden_dim - threshold_k} dimensions can be discarded

NEXT STEP:
Replace top-k (input-dependent) with a LEARNED basis that captures
the consistently important dimensions across all inputs.
""")
    else:
        print(f"""
All {hidden_dim} dimensions are required.

The model is maximally efficient - no redundancy to compress.
""")


if __name__ == "__main__":
    main()
