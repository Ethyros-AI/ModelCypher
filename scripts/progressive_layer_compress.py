#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Progressive Layer Compression
"""
Progressive Layer Compression

Compress layers one at a time, testing after each.
Each layer gets its OWN active dimensions (not global).

This finds the MAXIMUM compressible layers before quality degrades.

Usage:
    python progressive_layer_compress.py \
        --model /path/to/model \
        --test
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


def find_layer_active_dims(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    threshold: float = 0.01,
) -> tuple[list[int], float]:
    """Find active dimensions for a specific layer.

    Returns:
        active_dims: Dimensions with >threshold of layer energy
        top_energy_pct: Percentage of energy in active dims
    """
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
    if total > 0:
        energy /= total

    active = np.where(energy > threshold)[0].tolist()
    top_energy = sum(energy[d] for d in active)

    return active, top_energy


def compress_layer_to_dims(model: Any, layer_idx: int, active_dims: list[int]):
    """Compress layer's w2 to only output to active dimensions."""
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight  # [hidden_dim, intermediate_dim]

    w2_sparse = mx.zeros_like(w2)
    for dim in active_dims:
        row = w2[dim:dim+1, :]
        indices = mx.array([dim])
        w2_sparse = w2_sparse.at[indices].add(row)
    mx.eval(w2_sparse)

    mlp['w2'].weight = w2_sparse
    mx.eval(model.parameters())


def test_model_quality(model: Any, tokenizer: Any) -> tuple[bool, list[str]]:
    """Test if model produces coherent output.

    Returns:
        is_coherent: True if outputs look reasonable
        outputs: List of test outputs
    """
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

        # Check for repetition (sign of broken model)
        words = response.split()
        if len(words) >= 4:
            # Check if same word repeats 4+ times
            from collections import Counter
            counts = Counter(words)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 4:
                is_coherent = False

    return is_coherent, outputs


def main():
    parser = argparse.ArgumentParser(description="Progressive layer compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Energy threshold for active dimension")
    parser.add_argument("--test", action="store_true", help="Test after each layer")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    logger.info("Model: %d layers, %d hidden dim", n_layers, hidden_dim)

    # First, analyze all layers
    logger.info("\nAnalyzing layer compression potential...")
    layer_info = []

    for layer_idx in range(n_layers):
        active_dims, energy_pct = find_layer_active_dims(
            model, tokenizer, layer_idx, args.threshold
        )
        compression = hidden_dim / len(active_dims) if active_dims else 0
        layer_info.append({
            'idx': layer_idx,
            'dims': active_dims,
            'n_dims': len(active_dims),
            'energy': energy_pct,
            'compression': compression,
        })
        logger.info("  Layer %2d: %3d dims, %.1f%% energy, %.1fx compression",
                    layer_idx, len(active_dims), energy_pct * 100, compression)

    # Sort by compression potential (highest first)
    # But only consider layers with >90% energy in active dims
    compressible = [l for l in layer_info if l['energy'] >= 0.90]
    compressible.sort(key=lambda x: x['compression'], reverse=True)

    print("\n" + "=" * 60)
    print("COMPRESSION CANDIDATES (>90% energy in active dims)")
    print("=" * 60)
    for l in compressible:
        print(f"  Layer {l['idx']:>2}: {l['n_dims']:>3} dims, "
              f"{l['energy']*100:.1f}% energy, {l['compression']:.1f}x")

    # Test baseline
    logger.info("\nBaseline test...")
    is_coherent, outputs = test_model_quality(model, tokenizer)
    print("\nBaseline outputs:")
    for o in outputs:
        print(f"  {o}")

    # Progressive compression
    compressed_layers = []
    total_compression = 0

    print("\n" + "=" * 60)
    print("PROGRESSIVE COMPRESSION")
    print("=" * 60)

    for l in compressible:
        layer_idx = l['idx']
        active_dims = l['dims']

        logger.info("\nCompressing layer %d to %d dims...", layer_idx, len(active_dims))
        compress_layer_to_dims(model, layer_idx, active_dims)
        compressed_layers.append(layer_idx)

        # Test
        is_coherent, outputs = test_model_quality(model, tokenizer)

        status = "✓" if is_coherent else "✗"
        print(f"\n  Layer {layer_idx} ({l['compression']:.1f}x): {status}")
        for o in outputs:
            print(f"    {o}")

        if not is_coherent:
            print(f"\n  ⚠ Model degraded at layer {layer_idx}")
            break

        total_compression += l['compression']

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully compressed layers: {sorted(compressed_layers[:-1] if not is_coherent else compressed_layers)}")
    print(f"Failed at layer: {layer_idx if not is_coherent else 'None'}")


if __name__ == "__main__":
    main()
