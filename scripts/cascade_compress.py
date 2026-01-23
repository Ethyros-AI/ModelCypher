#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Cascade Compression
"""
Cascade Compression

Iteratively compress layers and re-analyze energy flow.

Hypothesis: Compressing layer 7 shifts downstream energy distribution.
Layer 8's compensation might create NEW bottlenecks that weren't
visible in the original model.

Usage:
    python cascade_compress.py \
        --model /path/to/model \
        --output /path/to/compressed \
        --iterations 3
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
    "speech": ["say", "words", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be", "live", "die"],
    "possession": ["have", "part"],
    "logical": ["not", "maybe", "can", "because", "if"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "moment"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside", "touch"],
    "taxonomy": ["kind of", "like"],
}


def get_prime_contexts() -> list[tuple[str, str, str]]:
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            contexts.append((prime, prime, category))
    return contexts


def analyze_layer_concentration(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> tuple[float, int, float]:
    """Analyze MLP output energy concentration for a layer.

    Returns:
        top1_pct: Percentage of energy in top output dimension
        top1_dim: The dimension with most energy
        recon_error: Output reconstruction error with rank-1 PCA
    """
    import mlx.core as mx

    contexts = get_prime_contexts()
    outputs = []

    for prime, context, category in contexts:
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
                    h_after_attn = h + attn_out
                    mx.eval(h_after_attn)

                    h_before_mlp = norm2(h_after_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)

                    out = mlp_out[0, -1, :]
                    mx.eval(out)
                    outputs.append(out)
                    break
        except Exception:
            continue

    if not outputs:
        return 0.0, -1, 1.0

    outputs_arr = mx.stack(outputs, axis=0)
    mx.eval(outputs_arr)

    # Energy per output dimension
    energy = mx.sum(outputs_arr * outputs_arr, axis=0)
    mx.eval(energy)
    energy_list = energy.tolist()
    total_energy = sum(energy_list)

    top1_dim = max(range(len(energy_list)), key=lambda i: energy_list[i])
    top1_pct = energy_list[top1_dim] / total_energy * 100

    # PCA reconstruction error
    mean = mx.mean(outputs_arr, axis=0)
    centered = outputs_arr - mean
    mx.eval(centered)

    cov = (centered.T @ centered) / len(outputs)
    mx.eval(cov)

    cov_f32 = cov.astype(mx.float32)
    mx.eval(cov_f32)
    cov_np = np.array(cov_f32)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_np)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Rank-1 reconstruction error
    P = eigenvectors[:, :1]
    P_mx = mx.array(P.astype(np.float32))
    mx.eval(P_mx)

    outputs_f32 = outputs_arr.astype(mx.float32)
    mx.eval(outputs_f32)

    projected = outputs_f32 @ P_mx @ P_mx.T
    mx.eval(projected)

    diff = outputs_f32 - projected
    recon_error = float(mx.linalg.norm(diff)) / float(mx.linalg.norm(outputs_f32))

    return top1_pct, top1_dim, recon_error


def compress_layer_sparse(model: Any, layer_idx: int, dim: int) -> tuple[float, int]:
    """Compress MLP w2 by keeping only one output dimension."""
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight

    w2_sparse = mx.zeros_like(w2)
    row = w2[dim:dim+1, :]
    indices = mx.array([dim])
    w2_sparse = w2_sparse.at[indices].add(row)
    mx.eval(w2_sparse)

    diff = w2 - w2_sparse
    error = float(mx.linalg.norm(diff)) / float(mx.linalg.norm(w2))

    orig_params = w2.shape[0] * w2.shape[1]

    mlp['w2'].weight = w2_sparse
    mx.eval(model.parameters())

    return error, orig_params


def find_best_bottleneck(
    model: Any,
    tokenizer: Any,
    exclude_layers: set[int],
    threshold_top1: float = 90.0,
    threshold_error: float = 0.10,
) -> tuple[int, float, int, float] | None:
    """Find the best compressible layer.

    Returns:
        (layer_idx, top1_pct, top1_dim, recon_error) or None if no good candidate
    """
    n_layers = len(model.model.layers)

    candidates = []
    for layer_idx in range(n_layers):
        if layer_idx in exclude_layers:
            continue

        top1_pct, top1_dim, recon_error = analyze_layer_concentration(model, tokenizer, layer_idx)

        if top1_pct >= threshold_top1 and recon_error <= threshold_error:
            candidates.append((layer_idx, top1_pct, top1_dim, recon_error))

    if not candidates:
        return None

    # Sort by top1_pct descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def save_model(model: Any, tokenizer: Any, source_path: str, output_path: str):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(source_path)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    flat_params = tree_flatten(model.parameters())
    weights = {k: v for k, v in flat_params}

    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights)


def main():
    parser = argparse.ArgumentParser(description="Cascade compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--iterations", type=int, default=3, help="Max compression iterations")
    parser.add_argument("--threshold-top1", type=float, default=90.0,
                        help="Min top-1 concentration (%%)")
    parser.add_argument("--threshold-error", type=float, default=0.10,
                        help="Max reconstruction error")
    parser.add_argument("--test", action="store_true", help="Test after each iteration")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    logger.info("Model has %d layers", n_layers)

    # Initial analysis
    logger.info("\n=== INITIAL ANALYSIS ===")
    logger.info("%-6s | %8s | %8s | %12s", "Layer", "Top1%", "Top1 Dim", "Recon Error")
    logger.info("-" * 50)

    for layer_idx in range(n_layers):
        top1_pct, top1_dim, recon_error = analyze_layer_concentration(model, tokenizer, layer_idx)
        marker = " ★" if top1_pct >= args.threshold_top1 and recon_error <= args.threshold_error else ""
        logger.info("%-6d | %7.1f%% | %8d | %11.1f%%%s",
                    layer_idx, top1_pct, top1_dim, recon_error * 100, marker)

    # Cascade compression
    compressed_layers: set[int] = set()
    total_compression = 0

    for iteration in range(args.iterations):
        logger.info("\n=== ITERATION %d ===", iteration + 1)

        # Find best candidate
        candidate = find_best_bottleneck(
            model, tokenizer, compressed_layers,
            args.threshold_top1, args.threshold_error
        )

        if candidate is None:
            logger.info("No more compressible layers found")
            break

        layer_idx, top1_pct, top1_dim, recon_error = candidate
        logger.info("Selected layer %d: top1=%.1f%% (dim %d), recon_error=%.1f%%",
                    layer_idx, top1_pct, top1_dim, recon_error * 100)

        # Compress
        weight_error, orig_params = compress_layer_sparse(model, layer_idx, top1_dim)
        compressed_layers.add(layer_idx)
        total_compression += orig_params

        logger.info("Compressed layer %d: weight_error=%.1f%%, %d params zeroed",
                    layer_idx, weight_error * 100, orig_params)

        # Test
        if args.test:
            logger.info("Testing...")
            prompts = ["2+2 is", "The capital of France is"]
            for prompt in prompts:
                output = generate(model, tokenizer, prompt=prompt, max_tokens=15, verbose=False)
                logger.info("  %s -> %s", prompt, output[len(prompt):][:40])

        # Re-analyze to find new bottlenecks
        logger.info("\nRe-analyzing after compression...")
        new_candidates = []
        for l in range(n_layers):
            if l in compressed_layers:
                continue
            top1_pct, top1_dim, recon_error = analyze_layer_concentration(model, tokenizer, l)
            if top1_pct >= args.threshold_top1 and recon_error <= args.threshold_error:
                new_candidates.append((l, top1_pct, recon_error))

        if new_candidates:
            logger.info("New potential bottlenecks:")
            for l, pct, err in new_candidates:
                logger.info("  Layer %d: top1=%.1f%%, recon_error=%.1f%%", l, pct, err * 100)
        else:
            logger.info("No new bottlenecks emerged")

    # Final summary
    logger.info("\n=== SUMMARY ===")
    logger.info("Compressed layers: %s", sorted(compressed_layers))
    logger.info("Total params zeroed: %d", total_compression)

    # Save
    logger.info("\nSaving to %s", args.output)
    save_model(model, tokenizer, args.model, args.output)

    # Final test
    if args.test:
        logger.info("\nFinal test:")
        prompts = [
            "The answer to 2+2 is",
            "Hello, my name is",
            "The capital of France is",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
            logger.info("  %s -> %s", prompt, output[len(prompt):][:50])


if __name__ == "__main__":
    main()
