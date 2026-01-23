#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Global Subspace Compression
"""
Global Subspace Compression

Finding: Across ALL 16 layers, only 11 dimensions are "active" (>1% energy).
The UNION of active dimensions is just 11 out of 1024.

This script:
1. Identifies the globally active dimensions
2. Projects ALL MLP outputs to just those dimensions
3. Tests if the model still works

If this works, we can achieve ~100x compression on ALL layers.

Usage:
    python global_subspace_compress.py \
        --model /path/to/model \
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


def find_global_active_dims(
    model: Any,
    tokenizer: Any,
    threshold: float = 0.01,
) -> tuple[list[int], dict[int, list[int]]]:
    """Find globally active dimensions across all layers.

    Returns:
        global_dims: Union of all active dimensions
        per_layer_dims: Active dimensions for each layer
    """
    import mlx.core as mx

    contexts = get_prime_contexts()
    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    # Collect energy per dimension per layer
    energy_per_layer = {i: np.zeros(hidden_dim) for i in range(n_layers)}

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

                # Accumulate energy
                delta = mlp_out[0, -1, :]
                mx.eval(delta)
                delta_np = np.array(delta.astype(mx.float32))
                energy_per_layer[idx] += delta_np ** 2

        except Exception:
            continue

    # Normalize and find active dims
    per_layer_dims = {}
    global_dims = set()

    for layer_idx in range(n_layers):
        energy = energy_per_layer[layer_idx]
        total = energy.sum()
        if total > 0:
            energy /= total

        active = np.where(energy > threshold)[0].tolist()
        per_layer_dims[layer_idx] = active
        global_dims.update(active)

    return sorted(global_dims), per_layer_dims


def compress_to_subspace(
    model: Any,
    active_dims: list[int],
):
    """Compress all MLP w2 matrices to output only to active dimensions.

    For each layer's w2:
    - Keep only rows corresponding to active_dims
    - Zero out all other rows
    """
    import mlx.core as mx

    n_layers = len(model.model.layers)
    total_orig = 0
    total_kept = 0

    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight  # [hidden_dim, intermediate_dim]

        hidden_dim, intermediate_dim = w2.shape
        total_orig += hidden_dim * intermediate_dim

        # Create sparse version - only keep active dims
        w2_sparse = mx.zeros_like(w2)
        for dim in active_dims:
            row = w2[dim:dim+1, :]
            indices = mx.array([dim])
            w2_sparse = w2_sparse.at[indices].add(row)
        mx.eval(w2_sparse)

        total_kept += len(active_dims) * intermediate_dim

        # Apply
        mlp['w2'].weight = w2_sparse
        mx.eval(model.parameters())

    logger.info("Compression: %d -> %d effective params (%.1fx)",
                total_orig, total_kept, total_orig / total_kept)

    return total_orig, total_kept


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
    parser = argparse.ArgumentParser(description="Global subspace compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Energy threshold for active dimension")
    parser.add_argument("--test", action="store_true", help="Test after compression")
    parser.add_argument("--extended-test", action="store_true",
                        help="Run extended test prompts")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    logger.info("Model: %d layers, %d hidden dim", n_layers, hidden_dim)

    # Find active dimensions
    logger.info("\nFinding globally active dimensions...")
    global_dims, per_layer = find_global_active_dims(model, tokenizer, args.threshold)

    print("\n" + "=" * 60)
    print("GLOBAL ACTIVE DIMENSIONS")
    print("=" * 60)
    print(f"Threshold: {args.threshold*100:.1f}% of layer energy")
    print(f"Active dimensions: {len(global_dims)} out of {hidden_dim}")
    print(f"Dimensions: {global_dims}")
    print(f"Theoretical compression: {hidden_dim / len(global_dims):.1f}x")
    print()

    print("Per-layer breakdown:")
    for layer_idx in range(n_layers):
        dims = per_layer[layer_idx]
        print(f"  Layer {layer_idx:>2}: {len(dims):>3} dims - {dims}")

    # Compress
    logger.info("\nCompressing ALL layers to global subspace...")
    orig, kept = compress_to_subspace(model, global_dims)

    # Test
    if args.test:
        logger.info("\nTesting compressed model...")
        prompts = [
            "The answer to 2+2 is",
            "Hello, my name is",
            "The capital of France is",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
            logger.info("  %s -> %s", prompt, output[len(prompt):][:50])

    if args.extended_test:
        logger.info("\nExtended test...")
        prompts = [
            "What is the meaning of life?",
            "Explain quantum physics in simple terms:",
            "Write a haiku about coding:",
            "The quick brown fox",
            "Once upon a time",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=30, verbose=False)
            logger.info("  %s\n    -> %s\n", prompt, output[len(prompt):][:80])

    if args.output:
        save_model(model, tokenizer, args.model, args.output)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
