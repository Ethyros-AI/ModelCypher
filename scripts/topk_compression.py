#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Top-K Compression
"""
Top-K Compression: The Simplest Test

THE PROBLEM WITH PREVIOUS APPROACHES:
Numerical overflow corrupts the decomposition.
Even perp_scale = 1.0 (keep everything) breaks generation.

THE SIMPLEST TEST:
What if we just keep the top-K dimensions of δ (by magnitude)?

This is:
1. Numerically stable (no matrix operations)
2. Easy to understand
3. A baseline for what compression is even possible

If top-K compression breaks at K=2040 (out of 2048), we know
the model is exquisitely sensitive to ALL dimensions.

If it works at K=100, we know most dimensions don't matter.

Usage:
    python topk_compression.py --model /path/to/model
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

    # Find indices of top-k by absolute value
    abs_delta = np.abs(delta)
    topk_indices = np.argpartition(abs_delta, -k)[-k:]

    # Create compressed delta
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
    """Test generation with top-k compression on specified layers."""
    import mlx.core as mx

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

    # Compressed generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    stats = {'total_energy_kept': [], 'dims_used': k}

    for idx, layer in enumerate(inner_model.layers):
        h_np = np.array(h.astype(mx.float32))
        h_in = h_np[0, -1, :]

        # Run actual layer
        result = layer(h)
        h_true = result[0] if isinstance(result, tuple) else result
        mx.eval(h_true)

        h_out_true = np.array(h_true[0, -1, :].astype(mx.float32))
        delta_true = h_out_true - h_in

        if idx in compress_layers:
            # Top-k compress
            delta_compressed = topk_compress_delta(delta_true, k)

            # Stats
            energy_kept = np.sum(delta_compressed**2) / (np.sum(delta_true**2) + 1e-10)
            stats['total_energy_kept'].append(energy_kept)

            # Apply
            h_new = h_in + delta_compressed
            h_np_new = h_np.copy()
            h_np_new[0, -1, :] = h_new

            h = mx.array(h_np_new).astype(h.dtype)
            mx.eval(h)
        else:
            h = h_true

    # Final norm
    if hasattr(inner_model, 'norm'):
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
    parser = argparse.ArgumentParser(description="Top-K compression")
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
    print("TOP-K COMPRESSION TEST")
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

            normal_first = normal.split()[0] if normal.split() else ""
            compressed_first = compressed.split()[0] if compressed.split() else ""

            if normal_first == compressed_first:
                matches += 1
                results.append(f"✓")
            else:
                results.append(f"{normal_first}→{compressed_first}")

        avg_energy = np.mean(energy_kept) if energy_kept else 0
        compression = hidden_dim / k

        print(f"{k:>6} | {compression:>10.1f}x | {avg_energy:>11.4f} | {matches}/{len(test_prompts)}     | {' | '.join(results)}")

    # Find the threshold
    print(f"\n{'='*80}")
    print("BINARY SEARCH FOR COMPRESSION THRESHOLD")
    print("="*80)

    # Binary search between working and non-working
    low, high = 8, 2048
    threshold_k = None

    while low < high:
        mid = (low + high) // 2

        # Test at mid
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
            # Works at mid, try smaller
            threshold_k = mid
            high = mid
        else:
            # Fails at mid, need more dims
            low = mid + 1

        print(f"  K={mid}: {matches}/{len(test_prompts)} matches")

    print(f"\nMINIMUM K FOR CORRECT OUTPUT: {threshold_k}")
    if threshold_k:
        print(f"MAXIMUM COMPRESSION: {hidden_dim / threshold_k:.1f}x")

    # Analysis
    print(f"\n{'='*80}")
    print("TOP-K COMPRESSION INSIGHT")
    print("="*80)

    if threshold_k and threshold_k < hidden_dim:
        print(f"""
COMPRESSION IS POSSIBLE!

The model can tolerate losing {hidden_dim - threshold_k} dimensions per layer.
Minimum dimensions needed: {threshold_k} / {hidden_dim} = {threshold_k/hidden_dim*100:.1f}%

This means:
1. NOT all dimensions are critical
2. There IS redundancy we can exploit
3. {hidden_dim / threshold_k:.1f}x compression is achievable with top-k

NEXT STEP:
Instead of top-k (which is input-dependent),
use a LEARNED basis that captures the critical dimensions.
""")
    else:
        print(f"""
THE MODEL IS HIGHLY SENSITIVE

Even small compression breaks generation.
This suggests:
1. Information is distributed across ALL dimensions
2. The helix structure means all dimensions participate
3. Compression requires preserving the FULL structure, not just top components
""")


if __name__ == "__main__":
    main()
