#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Analyze Then Compress
"""
Analyze Then Compress

1. Analyze ALL layers on original model
2. Identify which can be compressed (>90% variance in rank-k)
3. Apply compressions simultaneously

Usage:
    python analyze_then_compress.py \
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


def analyze_all_layers(model: Any, tokenizer: Any) -> dict:
    """Collect deltas for all layers in a SINGLE forward pass."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    n_layers = len(model.model.layers)

    # Collect deltas per layer
    layer_deltas = {i: [] for i in range(n_layers)}

    for context in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            h = model.model.embed_tokens(input_ids)
            mx.eval(h)

            for idx, layer in enumerate(model.model.layers):
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

                h = h_attn + mlp_out
                mx.eval(h)

                # Store MLP delta
                delta = mlp_out[0, -1, :]
                mx.eval(delta)
                layer_deltas[idx].append(np.array(delta.astype(mx.float32)))

        except Exception:
            continue

    # Stack and compute PCA for each layer
    results = {}

    for layer_idx in range(n_layers):
        deltas = layer_deltas[layer_idx]
        if not deltas:
            continue

        Delta = np.stack(deltas, axis=0)  # [n, hidden]
        Delta = np.nan_to_num(Delta, nan=0.0, posinf=0.0, neginf=0.0)

        # Original energy
        orig_energy = np.mean(np.sum(Delta * Delta, axis=1))

        # PCA
        mean = Delta.mean(axis=0)
        centered = Delta - mean

        if np.abs(centered).max() < 1e-10:
            results[layer_idx] = {'skip': True}
            continue

        cov = (centered.T @ centered) / len(Delta)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except:
            results[layer_idx] = {'skip': True}
            continue

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = np.sum(np.abs(eigenvalues))
        if total_var < 1e-10:
            results[layer_idx] = {'skip': True}
            continue

        # How many components for 90%, 95%, 99%?
        cumsum = np.cumsum(np.abs(eigenvalues)) / total_var
        dims_90 = np.searchsorted(cumsum, 0.90) + 1
        dims_95 = np.searchsorted(cumsum, 0.95) + 1
        dims_99 = np.searchsorted(cumsum, 0.99) + 1

        # Top-1 variance
        top1_var = eigenvalues[0] / total_var if total_var > 0 else 0

        results[layer_idx] = {
            'orig_energy': orig_energy,
            'top1_var': top1_var * 100,
            'dims_90': dims_90,
            'dims_95': dims_95,
            'dims_99': dims_99,
            'eigenvectors': eigenvectors,
            'eigenvalues': eigenvalues,
            'total_var': total_var,
        }

    return results


def compress_layers(
    model: Any,
    analysis: dict,
    layers_to_compress: list[int],
    rank: int,
) -> dict:
    """Apply compression to specified layers using pre-computed analysis."""
    import mlx.core as mx

    stats = {}

    for layer_idx in layers_to_compress:
        if layer_idx not in analysis or analysis[layer_idx].get('skip'):
            stats[layer_idx] = {'skipped': True}
            continue

        info = analysis[layer_idx]
        eigenvectors = info['eigenvectors']
        eigenvalues = info['eigenvalues']
        total_var = info['total_var']

        # Top-k principal components
        P = eigenvectors[:, :rank]

        # Variance kept
        var_kept = np.sum(np.abs(eigenvalues[:rank])) / total_var

        # Get w2
        layer = model.model.layers[layer_idx]
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight

        w2_f32 = w2.astype(mx.float32)
        mx.eval(w2_f32)
        w2_np = np.array(w2_f32)
        w2_np = np.nan_to_num(w2_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Project: w2_new = P @ P.T @ w2
        w2_proj = P @ (P.T @ w2_np)

        # Scale to preserve energy
        if var_kept > 0.01:
            scale = 1.0 / np.sqrt(var_kept)
            w2_scaled = w2_proj * scale
        else:
            w2_scaled = w2_proj

        w2_scaled = np.nan_to_num(w2_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply
        w2_new = mx.array(w2_scaled.astype(np.float32))
        if w2.dtype != mx.float32:
            w2_new = w2_new.astype(w2.dtype)
        mx.eval(w2_new)

        mlp['w2'].weight = w2_new

        stats[layer_idx] = {
            'var_kept': var_kept * 100,
            'compression': 1024 / rank,
        }

    mx.eval(model.parameters())
    return stats


def test_model(model: Any, tokenizer: Any) -> list[str]:
    """Test model outputs."""
    from mlx_lm import generate

    prompts = [
        "The answer to 2+2 is",
        "Hello, my name is",
    ]
    results = []

    for prompt in prompts:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        response = output[len(prompt):][:50]
        results.append(f"{prompt} -> {response}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze then compress")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--rank", type=int, default=1, help="Compression rank")
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="Minimum variance %% in rank-k for compression")
    parser.add_argument("--test", action="store_true", help="Test model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    # Analyze ALL layers first
    logger.info("Analyzing all layers...")
    analysis = analyze_all_layers(model, tokenizer)

    print("\n" + "=" * 70)
    print("LAYER COMPRESSION POTENTIAL (from original model)")
    print("=" * 70)
    print(f"{'Layer':>5} | {'Top1 Var%':>10} | {'90% dims':>8} | {'95% dims':>8} | {'99% dims':>8} | {'Orig E':>10}")
    print("-" * 70)

    compressible = []

    for layer_idx in range(n_layers):
        if layer_idx not in analysis or analysis[layer_idx].get('skip'):
            print(f"{layer_idx:>5} | {'SKIP':>10}")
            continue

        info = analysis[layer_idx]
        print(f"{layer_idx:>5} | {info['top1_var']:>9.1f}% | "
              f"{info['dims_90']:>8} | {info['dims_95']:>8} | {info['dims_99']:>8} | "
              f"{info['orig_energy']:>10.2f}")

        # Check if compressible at given rank
        if args.rank == 1 and info['top1_var'] >= args.threshold:
            compressible.append(layer_idx)
        elif args.rank > 1:
            # Check cumsum
            cumsum = np.cumsum(np.abs(info['eigenvalues'])) / info['total_var']
            if cumsum[args.rank - 1] * 100 >= args.threshold:
                compressible.append(layer_idx)

    print(f"\nLayers with >={args.threshold}% variance in rank-{args.rank}: {compressible}")

    if not compressible:
        print("No layers meet threshold for compression")
        return

    # Baseline test
    if args.test:
        print("\n=== BASELINE ===")
        for r in test_model(model, tokenizer):
            print(f"  {r}")

    # Compress
    print(f"\n=== COMPRESSING LAYERS {compressible} ===")
    stats = compress_layers(model, analysis, compressible, args.rank)

    for layer_idx in compressible:
        s = stats.get(layer_idx, {})
        if s.get('skipped'):
            print(f"  Layer {layer_idx}: skipped")
        else:
            print(f"  Layer {layer_idx}: {s['var_kept']:.1f}% variance kept, {s['compression']:.0f}x compression")

    # Test after
    if args.test:
        print("\n=== AFTER COMPRESSION ===")
        for r in test_model(model, tokenizer):
            print(f"  {r}")


if __name__ == "__main__":
    main()
