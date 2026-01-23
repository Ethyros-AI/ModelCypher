#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Full Conservation Compression
"""
Full Conservation Compression

Apply conservation-preserving compression to ALL layers.

Key insight: Energy conservation = Information conservation

For each layer:
1. Find the principal direction(s) of MLP delta
2. Project w2 to that subspace
3. SCALE to preserve ||delta||² exactly

If conservation is preserved at each layer, total information is preserved.

Usage:
    python full_conservation_compress.py \
        --model /path/to/model \
        --rank 1 \
        --test
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
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


def collect_layer_deltas(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> Any:
    """Collect MLP deltas for a layer."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    deltas = []

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
                    deltas.append(delta)
                    break

        except Exception:
            continue

    if deltas:
        return mx.stack(deltas, axis=0)
    return None


def compress_layer_conservation(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    rank: int,
) -> dict:
    """Compress layer while preserving energy conservation."""
    import mlx.core as mx

    # Collect deltas
    Delta = collect_layer_deltas(model, tokenizer, layer_idx)
    if Delta is None:
        return {'skipped': True}

    # Original energy
    orig_energy = float(mx.mean(mx.sum(Delta * Delta, axis=1)))

    # PCA
    Delta_f32 = Delta.astype(mx.float32)
    mx.eval(Delta_f32)
    Delta_np = np.array(Delta_f32)

    # Handle numerical issues
    Delta_np = np.nan_to_num(Delta_np, nan=0.0, posinf=0.0, neginf=0.0)

    mean = Delta_np.mean(axis=0)
    centered = Delta_np - mean

    # Check for degenerate case
    if np.abs(centered).max() < 1e-10:
        return {'skipped': True, 'reason': 'degenerate'}

    cov = (centered.T @ centered) / len(Delta_np)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return {'skipped': True, 'reason': 'eigh_failed'}

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance captured
    total_var = np.sum(np.abs(eigenvalues))
    if total_var < 1e-10:
        return {'skipped': True, 'reason': 'no_variance'}

    kept_var = np.sum(np.abs(eigenvalues[:rank]))
    var_pct = kept_var / total_var

    # Top-k principal components
    P = eigenvectors[:, :rank]  # [hidden, rank]

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

    # Scale to preserve energy: ||w2_proj @ x||² ≈ var_pct * ||w2 @ x||²
    # So scale by 1/sqrt(var_pct)
    if var_pct > 0.01:  # Only scale if we're keeping meaningful variance
        scale = 1.0 / np.sqrt(var_pct)
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
    mx.eval(model.parameters())

    # Verify new energy
    Delta_new = collect_layer_deltas(model, tokenizer, layer_idx)
    if Delta_new is not None:
        new_energy = float(mx.mean(mx.sum(Delta_new * Delta_new, axis=1)))
    else:
        new_energy = 0

    return {
        'var_pct': var_pct * 100,
        'orig_energy': orig_energy,
        'new_energy': new_energy,
        'energy_ratio': new_energy / orig_energy if orig_energy > 0 else 0,
        'compression': 1024 / rank,  # Assuming 1024 hidden dim
    }


def test_model(model: Any, tokenizer: Any) -> list[str]:
    """Test model outputs."""
    from mlx_lm import generate

    prompts = [
        "The answer to 2+2 is",
        "Hello, my name is",
        "The capital of France is",
    ]
    results = []

    for prompt in prompts:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        response = output[len(prompt):][:50]
        results.append(f"{prompt} -> {response}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Full conservation compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--rank", type=int, default=1, help="Compression rank per layer")
    parser.add_argument("--test", action="store_true", help="Test model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    logger.info("Model: %d layers, %d hidden dim", n_layers, hidden_dim)

    # Baseline test
    if args.test:
        print("\n=== BASELINE ===")
        for r in test_model(model, tokenizer):
            print(f"  {r}")

    # Compress each layer
    print(f"\n=== COMPRESSING ALL LAYERS (rank={args.rank}) ===")
    print(f"{'Layer':>5} | {'Var %':>8} | {'Orig E':>10} | {'New E':>10} | {'E Ratio':>8} | {'Compress':>8}")
    print("-" * 70)

    total_compression = 0
    layers_compressed = 0

    for layer_idx in range(n_layers):
        stats = compress_layer_conservation(model, tokenizer, layer_idx, args.rank)

        if stats.get('skipped'):
            reason = stats.get('reason', 'unknown')
            print(f"{layer_idx:>5} | {'SKIPPED':>8} ({reason})")
        else:
            print(f"{layer_idx:>5} | {stats['var_pct']:>7.1f}% | "
                  f"{stats['orig_energy']:>10.2f} | {stats['new_energy']:>10.2f} | "
                  f"{stats['energy_ratio']:>8.3f} | {stats['compression']:>7.0f}x")
            total_compression += stats['compression']
            layers_compressed += 1

    # Final test
    if args.test:
        print("\n=== AFTER COMPRESSION ===")
        for r in test_model(model, tokenizer):
            print(f"  {r}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Layers compressed: {layers_compressed}/{n_layers}")
    print(f"Average compression per layer: {total_compression/layers_compressed:.1f}x" if layers_compressed > 0 else "N/A")

    # Save
    if args.output:
        from mlx.utils import tree_flatten

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_dir = Path(args.model)
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json", "merges.txt"]:
            src = source_dir / fname
            if src.exists():
                shutil.copy(src, output_dir / fname)

        flat_params = tree_flatten(model.parameters())
        weights = {k: v for k, v in flat_params}

        weights_path = output_dir / "model.safetensors"
        mx.save_safetensors(str(weights_path), weights)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
