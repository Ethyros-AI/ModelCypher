#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Middle-Out Compression
"""
Middle-Out Compression

Start from the highest energy layer (layer 7) and work outward.

Layer 7 is the energy INJECTION point. It's orthogonal to the residual
stream, so compressing it preserves conservation automatically.

Then move to adjacent layers (6, 8), (5, 9), etc.
Test after each expansion.

Usage:
    python middle_out_compress.py \
        --model /path/to/model \
        --test
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SEMANTIC_PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other", "else"],
    "quantifiers": ["one", "two", "some", "all", "much", "many", "little", "few"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "feel", "see", "hear"],
}


def get_prime_contexts() -> list[str]:
    contexts = []
    for primes in SEMANTIC_PRIMES.values():
        contexts.extend(primes)
    return contexts


def find_layer_top_dims(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    n_dims: int = 1,
) -> tuple[list[int], float]:
    """Find top energy output dimensions for a layer."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    energy = np.zeros(hidden_dim)

    for context in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            h = model.model.embed_tokens(input_ids)
            mx.eval(h)

            for idx, layer in enumerate(model.model.layers):
                if idx < layer_idx:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result
                    mx.eval(h)
                elif idx == layer_idx:
                    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

                    norm1 = layer['operator_norm']
                    norm2 = layer['ffn_norm']
                    mlp = layer['feed_forward']
                    if 'conv' in layer_keys:
                        self_attn = layer['conv']
                    else:
                        self_attn = layer['self_attn']

                    h_normed = norm1(h)
                    mx.eval(h_normed)
                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    mx.eval(attn_out)
                    h_attn = h + attn_out
                    mx.eval(h_attn)

                    h_before_mlp = norm2(h_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)

                    delta = mlp_out[0, -1, :]
                    mx.eval(delta)
                    delta_np = np.array(delta.astype(mx.float32))
                    energy += delta_np ** 2
                    break

        except Exception:
            continue

    total = energy.sum()
    if total == 0:
        return [], 0.0

    # Sort by energy
    sorted_dims = np.argsort(energy)[::-1]
    top_dims = sorted_dims[:n_dims].tolist()
    top_energy_pct = sum(energy[d] for d in top_dims) / total

    return top_dims, top_energy_pct


def compress_layer_sparse(model: Any, layer_idx: int, keep_dims: list[int]):
    """Compress layer's w2 by zeroing out all rows except keep_dims."""
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight  # [hidden_dim, intermediate_dim]

    w2_sparse = mx.zeros_like(w2)
    for dim in keep_dims:
        row = w2[dim:dim+1, :]
        indices = mx.array([dim])
        w2_sparse = w2_sparse.at[indices].add(row)
    mx.eval(w2_sparse)

    mlp['w2'].weight = w2_sparse
    mx.eval(model.parameters())


def test_model_quality(model: Any, tokenizer: Any) -> tuple[bool, list[str]]:
    """Test if model produces coherent output."""
    from mlx_lm import generate

    prompts = [
        "The answer to 2+2 is",
        "Hello, my name is",
    ]

    outputs = []
    is_coherent = True

    for prompt in prompts:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=15, verbose=False)
        response = output[len(prompt):].strip()[:40]
        outputs.append(f"{prompt} -> {response}")

        # Check for repetition
        words = response.split()
        if len(words) >= 4:
            counts = Counter(words)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 4:
                is_coherent = False

    return is_coherent, outputs


def main():
    parser = argparse.ArgumentParser(description="Middle-out compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--center", type=int, default=7, help="Starting layer (highest energy)")
    parser.add_argument("--min-energy", type=float, default=80.0,
                        help="Minimum energy %% in top dims to compress")
    parser.add_argument("--n-dims", type=int, default=1, help="Dimensions to keep per layer")
    parser.add_argument("--test", action="store_true", help="Test after each step")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    logger.info("Model: %d layers, %d hidden dim", n_layers, hidden_dim)

    # Analyze all layers first
    print("\n" + "=" * 60)
    print("LAYER ENERGY ANALYSIS")
    print("=" * 60)

    layer_info = {}
    for layer_idx in range(n_layers):
        top_dims, energy_pct = find_layer_top_dims(model, tokenizer, layer_idx, args.n_dims)
        layer_info[layer_idx] = {
            'dims': top_dims,
            'energy_pct': energy_pct * 100,
        }
        marker = " ★" if energy_pct * 100 >= args.min_energy else ""
        print(f"  Layer {layer_idx:>2}: top-{args.n_dims} dims = {top_dims}, "
              f"{energy_pct*100:.1f}% energy{marker}")

    # Baseline
    if args.test:
        print("\n=== BASELINE ===")
        _, outputs = test_model_quality(model, tokenizer)
        for o in outputs:
            print(f"  {o}")

    # Middle-out compression
    print(f"\n=== MIDDLE-OUT FROM LAYER {args.center} ===")

    compressed = set()
    radius = 0

    while True:
        # Layers to try at this radius
        to_try = []
        if radius == 0:
            to_try = [args.center]
        else:
            if args.center - radius >= 0:
                to_try.append(args.center - radius)
            if args.center + radius < n_layers:
                to_try.append(args.center + radius)

        if not to_try:
            break

        for layer_idx in to_try:
            if layer_idx in compressed:
                continue

            info = layer_info[layer_idx]
            if info['energy_pct'] < args.min_energy:
                print(f"\n  Layer {layer_idx}: SKIP (only {info['energy_pct']:.1f}% energy in top-{args.n_dims})")
                continue

            print(f"\n  Layer {layer_idx}: compressing to dims {info['dims']} ({info['energy_pct']:.1f}% energy)")
            compress_layer_sparse(model, layer_idx, info['dims'])
            compressed.add(layer_idx)

            if args.test:
                is_coherent, outputs = test_model_quality(model, tokenizer)
                status = "✓" if is_coherent else "✗"
                print(f"    Status: {status}")
                for o in outputs:
                    print(f"      {o}")

                if not is_coherent:
                    print(f"\n    ⚠ Model degraded after compressing layer {layer_idx}")
                    # Could try to revert here if needed
                    break

        radius += 1
        if radius > n_layers:
            break

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Compressed layers: {sorted(compressed)}")
    print(f"Total compression: {len(compressed)} layers × {1024/args.n_dims:.0f}x = "
          f"{len(compressed) * 1024 / args.n_dims:.0f}x effective")


if __name__ == "__main__":
    main()
