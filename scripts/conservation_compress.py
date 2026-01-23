#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Conservation-Exact Compression
"""
Conservation-Exact Compression

Hypothesis: Energy conservation = Information conservation

For each layer, decompose delta:
    delta = delta_parallel + delta_perp

Where delta_parallel is the component along h (determines cross term).

To preserve conservation EXACTLY:
1. Keep delta_parallel unchanged (preserves 2<h,δ>)
2. Scale delta_perp to preserve ||δ||²

The compression: represent delta_perp in a low-rank basis, but SCALE
to maintain the original perp energy.

Usage:
    python conservation_compress.py \
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


def analyze_layer_decomposition(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> dict:
    """Decompose layer's delta into parallel and perpendicular components."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    deltas = []
    h_ins = []

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
                    mx.eval(h_in)

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

                    mlp_delta = mlp_out[0, -1, :]
                    mx.eval(mlp_delta)

                    deltas.append(mlp_delta)
                    h_ins.append(h_in)
                    break

        except Exception:
            continue

    if not deltas:
        return {}

    Delta = mx.stack(deltas, axis=0)
    H = mx.stack(h_ins, axis=0)
    mx.eval(Delta)
    mx.eval(H)

    n = Delta.shape[0]

    # For each sample, decompose delta
    parallel_energies = []
    perp_energies = []
    cross_terms = []

    for i in range(n):
        h = H[i]
        d = Delta[i]

        h_norm_sq = float(mx.sum(h * h))
        if h_norm_sq < 1e-10:
            continue

        # delta_parallel = <d, h> / ||h||² * h
        dot = float(mx.sum(d * h))
        d_parallel = (dot / h_norm_sq) * h
        d_perp = d - d_parallel

        mx.eval(d_parallel)
        mx.eval(d_perp)

        par_e = float(mx.sum(d_parallel * d_parallel))
        perp_e = float(mx.sum(d_perp * d_perp))
        cross = 2 * dot

        parallel_energies.append(par_e)
        perp_energies.append(perp_e)
        cross_terms.append(cross)

    avg_parallel = np.mean(parallel_energies)
    avg_perp = np.mean(perp_energies)
    avg_cross = np.mean(cross_terms)
    total = avg_parallel + avg_perp

    return {
        'parallel_energy': avg_parallel,
        'perp_energy': avg_perp,
        'cross_term': avg_cross,
        'parallel_pct': avg_parallel / total * 100 if total > 0 else 0,
        'perp_pct': avg_perp / total * 100 if total > 0 else 0,
    }


def compress_layer_conservation_exact(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    rank: int = 1,
) -> dict:
    """Compress layer while preserving conservation EXACTLY.

    Strategy:
    1. Find principal direction(s) of MLP output delta
    2. Project w2 to only output in those directions
    3. SCALE the projection to preserve ||delta||²

    The cross term is preserved because we're keeping the same
    output direction (just scaled).
    """
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

    if not deltas:
        return {}

    Delta = mx.stack(deltas, axis=0)
    mx.eval(Delta)

    # PCA on delta
    Delta_f32 = Delta.astype(mx.float32)
    mx.eval(Delta_f32)
    Delta_np = np.array(Delta_f32)

    mean = Delta_np.mean(axis=0)
    centered = Delta_np - mean
    cov = (centered.T @ centered) / len(Delta_np)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top-k eigenvectors
    P = eigenvectors[:, :rank]  # [hidden, rank]

    # Original delta variance
    orig_var = np.sum(eigenvalues)
    kept_var = np.sum(eigenvalues[:rank])
    var_pct = kept_var / orig_var

    logger.info("  Layer %d: keeping %d/%d dims, %.1f%% variance",
                layer_idx, rank, len(eigenvalues), var_pct * 100)

    # Get w2
    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight  # [hidden, intermediate]

    w2_f32 = w2.astype(mx.float32)
    mx.eval(w2_f32)
    w2_np = np.array(w2_f32)

    # Project w2 to P subspace: w2_new = P @ P.T @ w2
    w2_proj = P @ (P.T @ w2_np)  # [hidden, intermediate]

    # SCALING for conservation: we need ||w2_proj @ x||² = ||w2 @ x||² for typical x
    # This requires scaling by 1/sqrt(var_pct)
    if var_pct > 0:
        scale = 1.0 / np.sqrt(var_pct)
        w2_scaled = w2_proj * scale
    else:
        w2_scaled = w2_proj

    # Apply
    w2_new = mx.array(w2_scaled.astype(np.float32))
    if w2.dtype != mx.float32:
        w2_new = w2_new.astype(w2.dtype)
    mx.eval(w2_new)

    mlp['w2'].weight = w2_new
    mx.eval(model.parameters())

    # Verify conservation is preserved
    new_deltas = []
    for context in get_prime_contexts()[:5]:
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
                    norm2 = layer['ffn_norm']
                    mlp = layer['feed_forward']
                    if 'conv' in layer_keys:
                        self_attn = layer['conv']
                    else:
                        self_attn = layer['self_attn']
                    norm1 = layer['operator_norm']

                    h_normed = norm1(h)
                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    h_before = norm2(h_attn)
                    mlp_out = mlp(h_before)
                    delta = mlp_out[0, -1, :]
                    mx.eval(delta)
                    new_deltas.append(float(mx.sum(delta * delta)))
                    break
        except:
            continue

    orig_energy = np.mean([float(mx.sum(d * d)) for d in deltas[:5]])
    new_energy = np.mean(new_deltas) if new_deltas else 0

    logger.info("  Original delta energy: %.2f", orig_energy)
    logger.info("  New delta energy: %.2f", new_energy)
    logger.info("  Energy ratio: %.3f", new_energy / orig_energy if orig_energy > 0 else 0)

    return {
        'variance_kept': var_pct,
        'orig_energy': orig_energy,
        'new_energy': new_energy,
        'energy_ratio': new_energy / orig_energy if orig_energy > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Conservation-exact compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=7, help="Layer to compress")
    parser.add_argument("--rank", type=int, default=1, help="Compression rank")
    parser.add_argument("--test", action="store_true", help="Test model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    # Analyze all layers
    print("\n" + "=" * 70)
    print("LAYER DECOMPOSITION (parallel vs perpendicular to h)")
    print("=" * 70)
    print(f"{'Layer':>5} | {'||δ_∥||²':>12} | {'||δ_⊥||²':>12} | {'∥ %':>8} | {'⊥ %':>8} | {'2<h,δ>':>10}")
    print("-" * 70)

    for layer_idx in range(n_layers):
        stats = analyze_layer_decomposition(model, tokenizer, layer_idx)
        if stats:
            print(f"{layer_idx:>5} | {stats['parallel_energy']:>12.2f} | "
                  f"{stats['perp_energy']:>12.2f} | {stats['parallel_pct']:>7.1f}% | "
                  f"{stats['perp_pct']:>7.1f}% | {stats['cross_term']:>10.2f}")

    # Compress
    print(f"\n\n=== Compressing layer {args.layer} with rank {args.rank} ===")

    if args.test:
        logger.info("\nBefore compression:")
        prompts = ["2+2 is", "Hello, my name is"]
        for p in prompts:
            out = generate(model, tokenizer, prompt=p, max_tokens=15, verbose=False)
            logger.info("  %s -> %s", p, out[len(p):][:40])

    stats = compress_layer_conservation_exact(model, tokenizer, args.layer, args.rank)

    if args.test:
        logger.info("\nAfter compression:")
        for p in prompts:
            out = generate(model, tokenizer, prompt=p, max_tokens=15, verbose=False)
            logger.info("  %s -> %s", p, out[len(p):][:40])


if __name__ == "__main__":
    main()
