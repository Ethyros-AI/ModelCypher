#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Conservation-Preserving Compression
"""
Conservation-Preserving Compression

Key insight: Energy conservation = Information conservation

E_out = E_in + ||delta||² + 2<h, delta>

The compression that works is one that PRESERVES:
1. ||delta||² (delta magnitude)
2. <h, delta> (cross term / alignment)

If both are preserved, E_out/E_in is preserved, so information is preserved.

Usage:
    python conservation_preserving_compress.py \
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


def measure_layer_conservation(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> dict:
    """Measure energy conservation components for a layer.

    Returns dict with:
        E_in: Input energy ||h_in||²
        E_out: Output energy ||h_out||²
        delta_norm_sq: ||delta||²
        cross_term: 2<h_in, delta>
        ratio: E_out / E_in
    """
    import mlx.core as mx

    contexts = get_prime_contexts()

    E_in_total = 0.0
    E_out_total = 0.0
    delta_norm_sq_total = 0.0
    cross_term_total = 0.0
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
                    h_in = h[0, -1, :]  # Last token
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

                    # h_out = h_attn + mlp_out (but we measure from h_in perspective)
                    # Actually for MLP contribution: h_in -> h_in + delta
                    # where delta = attn_out + mlp_out
                    attn_delta = attn_out[0, -1, :]
                    mlp_delta = mlp_out[0, -1, :]
                    mx.eval(attn_delta)
                    mx.eval(mlp_delta)

                    total_delta = attn_delta + mlp_delta
                    mx.eval(total_delta)

                    h_out = h_in + total_delta
                    mx.eval(h_out)

                    # Compute conservation terms
                    E_in = float(mx.sum(h_in * h_in))
                    E_out = float(mx.sum(h_out * h_out))
                    delta_norm_sq = float(mx.sum(total_delta * total_delta))
                    cross = float(mx.sum(h_in * total_delta))

                    E_in_total += E_in
                    E_out_total += E_out
                    delta_norm_sq_total += delta_norm_sq
                    cross_term_total += 2 * cross
                    n_samples += 1
                    break

        except Exception:
            continue

    if n_samples == 0:
        return {}

    # Average
    E_in_avg = E_in_total / n_samples
    E_out_avg = E_out_total / n_samples
    delta_avg = delta_norm_sq_total / n_samples
    cross_avg = cross_term_total / n_samples

    # Verify conservation equation: E_out = E_in + ||delta||² + 2<h,delta>
    predicted_E_out = E_in_avg + delta_avg + cross_avg
    equation_error = abs(E_out_avg - predicted_E_out) / E_out_avg

    return {
        'E_in': E_in_avg,
        'E_out': E_out_avg,
        'delta_norm_sq': delta_avg,
        'cross_term': cross_avg,
        'ratio': E_out_avg / E_in_avg if E_in_avg > 0 else 0,
        'equation_error': equation_error,
    }


def find_conservation_preserving_transform(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> tuple[Any, dict]:
    """Find compression that preserves conservation.

    For MLP delta to preserve conservation, we need:
    1. ||delta'||² = ||delta||² (preserve magnitude)
    2. <h, delta'> = <h, delta> (preserve cross term)

    Returns:
        transform: The compression transform
        stats: Conservation stats before/after
    """
    import mlx.core as mx

    contexts = get_prime_contexts()
    deltas = []
    h_inputs = []

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
                    h_inputs.append(h_in)
                    break

        except Exception:
            continue

    if not deltas:
        return None, {}

    # Stack
    Delta = mx.stack(deltas, axis=0)  # [n, hidden]
    H = mx.stack(h_inputs, axis=0)  # [n, hidden]
    mx.eval(Delta)
    mx.eval(H)

    # Key insight: We need to find P such that:
    # 1. ||P @ delta||² ≈ ||delta||²  (preserve magnitude)
    # 2. <h, P @ delta> ≈ <h, delta>  (preserve cross term)
    #
    # Condition 2 means: h.T @ P @ delta = h.T @ delta
    # This holds if P @ delta has the same projection onto h as delta
    #
    # One approach: P = I - (I - h @ h.T / ||h||²) @ Q
    # where Q projects out dimensions orthogonal to both h and delta
    #
    # Actually, simpler: decompose delta = delta_parallel + delta_perp
    # where delta_parallel = <h, delta> / ||h||² * h
    # We MUST keep delta_parallel (for cross term)
    # We can compress delta_perp

    n_samples = Delta.shape[0]
    hidden_dim = Delta.shape[1]

    # Compute mean h direction
    H_mean = mx.mean(H, axis=0)
    mx.eval(H_mean)
    H_norm = mx.sqrt(mx.sum(H_mean * H_mean))
    h_unit = H_mean / H_norm
    mx.eval(h_unit)

    # Decompose each delta
    # delta_parallel = <delta, h_unit> * h_unit
    # delta_perp = delta - delta_parallel

    dots = Delta @ h_unit  # [n]
    mx.eval(dots)

    Delta_parallel = mx.outer(dots, h_unit)  # [n, hidden]
    mx.eval(Delta_parallel)

    Delta_perp = Delta - Delta_parallel  # [n, hidden]
    mx.eval(Delta_perp)

    # Stats on parallel vs perpendicular
    parallel_energy = float(mx.sum(Delta_parallel * Delta_parallel))
    perp_energy = float(mx.sum(Delta_perp * Delta_perp))
    total_energy = parallel_energy + perp_energy

    logger.info("  Delta decomposition:")
    logger.info("    Parallel to h: %.1f%%", parallel_energy / total_energy * 100)
    logger.info("    Perpendicular: %.1f%%", perp_energy / total_energy * 100)

    # Now: can we compress delta_perp while maintaining its norm?
    # If we project delta_perp to a subspace, we need to SCALE to preserve norm

    # PCA on delta_perp
    Delta_perp_f32 = Delta_perp.astype(mx.float32)
    mx.eval(Delta_perp_f32)
    Delta_perp_np = np.array(Delta_perp_f32)

    # Covariance
    cov = (Delta_perp_np.T @ Delta_perp_np) / n_samples
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # How many components capture 99% of perp variance?
    cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components_99 = np.searchsorted(cumsum, 0.99) + 1
    n_components_95 = np.searchsorted(cumsum, 0.95) + 1

    logger.info("    Perp subspace: %d dims for 95%%, %d dims for 99%%",
                n_components_95, n_components_99)

    # The key: if we keep k components of delta_perp, we lose (1 - cumsum[k-1]) of perp energy
    # To preserve conservation, we'd need to SCALE the kept components
    # OR add a correction term

    return {
        'h_unit': h_unit,
        'parallel_pct': parallel_energy / total_energy,
        'perp_pct': perp_energy / total_energy,
        'perp_dims_95': n_components_95,
        'perp_dims_99': n_components_99,
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues,
    }


def main():
    parser = argparse.ArgumentParser(description="Conservation-preserving compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print("\n" + "=" * 80)
    print("ENERGY CONSERVATION BY LAYER")
    print("=" * 80)
    print(f"{'Layer':>5} | {'E_in':>10} | {'E_out':>10} | {'||δ||²':>10} | {'2<h,δ>':>10} | {'Ratio':>8} | {'Err':>6}")
    print("-" * 80)

    for layer_idx in range(n_layers):
        stats = measure_layer_conservation(model, tokenizer, layer_idx)
        if stats:
            print(f"{layer_idx:>5} | {stats['E_in']:>10.1f} | {stats['E_out']:>10.1f} | "
                  f"{stats['delta_norm_sq']:>10.1f} | {stats['cross_term']:>10.1f} | "
                  f"{stats['ratio']:>8.3f} | {stats['equation_error']*100:>5.2f}%")

    # Detailed analysis of select layers
    print("\n" + "=" * 80)
    print("CONSERVATION-PRESERVING COMPRESSION ANALYSIS")
    print("=" * 80)

    for layer_idx in [7, 14]:
        print(f"\n--- Layer {layer_idx} ---")
        result = find_conservation_preserving_transform(model, tokenizer, layer_idx)
        if result:
            print(f"  To preserve cross term, we MUST keep: delta component parallel to h")
            print(f"  Parallel component: {result['parallel_pct']*100:.1f}% of delta energy")
            print(f"  Perpendicular component: {result['perp_pct']*100:.1f}% of delta energy")
            print(f"  Perp can be compressed to: {result['perp_dims_95']} dims (95%) or {result['perp_dims_99']} dims (99%)")


if __name__ == "__main__":
    main()
