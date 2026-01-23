#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Combined Conservation Compression
"""
Combined Conservation Compression

Apply both compression strategies based on energy conservation:

1. Layers with conservation ≈ 1.0 and tiny delta: SKIP entirely
2. Layers with energy concentrated in few dims: Compress MLP w2

This combines:
- Layer removal (conservation = 1)
- Dimension reduction (energy concentration)

Usage:
    python combined_conservation_compress.py \
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


def analyze_layer_conservation(model: Any, tokenizer: Any, layer_idx: int) -> dict:
    """Measure conservation ratio and energy concentration for a layer."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    E_in_total = 0.0
    E_out_total = 0.0
    delta_energy = np.zeros(hidden_dim)
    n_samples = 0

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
                    h_in = h[0, -1, :]
                    E_in = float(mx.sum(h_in * h_in))

                    result = layer(h)
                    h_out_full = result[0] if isinstance(result, tuple) else result
                    mx.eval(h_out_full)

                    h_out = h_out_full[0, -1, :]
                    E_out = float(mx.sum(h_out * h_out))

                    delta = h_out - h_in
                    mx.eval(delta)
                    delta_np = np.array(delta.astype(mx.float32))

                    E_in_total += E_in
                    E_out_total += E_out
                    delta_energy += delta_np ** 2
                    n_samples += 1
                    break

        except Exception:
            continue

    if n_samples == 0:
        return {}

    # Conservation ratio
    ratio = E_out_total / E_in_total if E_in_total > 0 else 0

    # Energy concentration in delta
    total_delta_energy = delta_energy.sum()
    if total_delta_energy > 0:
        sorted_dims = np.argsort(delta_energy)[::-1]
        top1_pct = delta_energy[sorted_dims[0]] / total_delta_energy * 100
        top_dim = sorted_dims[0]
    else:
        top1_pct = 0
        top_dim = 0

    return {
        'ratio': ratio,
        'delta_energy': total_delta_energy / n_samples,
        'top1_pct': top1_pct,
        'top_dim': top_dim,
    }


def compress_layer_w2(model: Any, layer_idx: int, keep_dim: int):
    """Compress layer's w2 to single output dimension."""
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight

    w2_sparse = mx.zeros_like(w2)
    row = w2[keep_dim:keep_dim+1, :]
    indices = mx.array([keep_dim])
    w2_sparse = w2_sparse.at[indices].add(row)
    mx.eval(w2_sparse)

    mlp['w2'].weight = w2_sparse
    mx.eval(model.parameters())


def inference_with_modifications(
    model: Any,
    tokenizer: Any,
    prompt: str,
    skip_layers: set[int],
    max_tokens: int = 30,
) -> str:
    """Run inference with layer skipping."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    output_tokens = list(tokens)

    for _ in range(max_tokens):
        input_ids = mx.array([output_tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(model.model.layers):
            if idx in skip_layers:
                continue
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

        if hasattr(model.model, 'embedding_norm'):
            h = model.model.embedding_norm(h)
            mx.eval(h)

        embed_weights = model.model.embed_tokens.weight
        logits = h @ embed_weights.T
        mx.eval(logits)

        next_token = int(mx.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens)


def main():
    parser = argparse.ArgumentParser(description="Combined conservation compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    # Analyze all layers
    print("\n" + "=" * 70)
    print("LAYER CONSERVATION ANALYSIS")
    print("=" * 70)
    print(f"{'Layer':>5} | {'Ratio':>8} | {'||δ||²':>10} | {'Top1%':>8} | {'Action':>15}")
    print("-" * 70)

    skip_layers = set()
    compress_layers = {}

    for layer_idx in range(n_layers):
        stats = analyze_layer_conservation(model, tokenizer, layer_idx)
        if not stats:
            continue

        # Decision rules based on conservation
        ratio = stats['ratio']
        delta_e = stats['delta_energy']
        top1 = stats['top1_pct']

        # Perfect conservation + tiny delta → can skip
        if 0.995 <= ratio <= 1.005 and delta_e < 0.1:
            action = "SKIP (identity)"
            skip_layers.add(layer_idx)
        # High energy concentration → compress MLP
        elif top1 > 90:
            action = f"COMPRESS (dim {stats['top_dim']})"
            compress_layers[layer_idx] = int(stats['top_dim'])
        else:
            action = "keep"

        print(f"{layer_idx:>5} | {ratio:>8.3f} | {delta_e:>10.2f} | {top1:>7.1f}% | {action:>15}")

    print(f"\nLayers to skip: {sorted(skip_layers)}")
    print(f"Layers to compress: {compress_layers}")

    # Baseline test
    if args.test:
        print("\n=== BASELINE ===")
        prompts = ["2+2 is", "The capital of France is"]
        for p in prompts:
            out = generate(model, tokenizer, prompt=p, max_tokens=15, verbose=False)
            print(f"  {p} -> {out[len(p):][:40]}")

    # Apply compressions
    for layer_idx, dim in compress_layers.items():
        compress_layer_w2(model, layer_idx, dim)
        logger.info("Compressed layer %d to dim %d", layer_idx, dim)

    # Test with modifications
    if args.test:
        print("\n=== AFTER COMPRESSION + LAYER SKIPPING ===")
        for p in prompts:
            out = inference_with_modifications(model, tokenizer, p, skip_layers, 15)
            print(f"  {p} -> {out[len(p):][:40]}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"Layers skipped: {len(skip_layers)}/{n_layers} ({len(skip_layers)/n_layers*100:.0f}%)")
    print(f"Layers with 1024x MLP compression: {len(compress_layers)}")

    # Rough parameter savings
    params_per_layer = 3 * hidden_dim * 4608 + 1024 * 1024 * 3  # MLP + attention approx
    mlp_w2_per_layer = hidden_dim * 4608

    saved_from_skip = len(skip_layers) * params_per_layer
    saved_from_compress = len(compress_layers) * mlp_w2_per_layer * 1023 / 1024

    total_params = n_layers * params_per_layer
    total_saved = saved_from_skip + saved_from_compress

    print(f"\nEstimated savings:")
    print(f"  From layer removal: {saved_from_skip/1e6:.1f}M params")
    print(f"  From w2 compression: {saved_from_compress/1e6:.1f}M params")
    print(f"  Total: {total_saved/1e6:.1f}M of {total_params/1e6:.1f}M ({total_saved/total_params*100:.1f}%)")


if __name__ == "__main__":
    main()
