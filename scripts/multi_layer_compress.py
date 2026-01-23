#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Multi-Layer Compression
"""
Multi-Layer Compression

Compresses multiple layers based on their natural bottleneck structure.

Usage:
    python multi_layer_compress.py \
        --model /path/to/model \
        --output /path/to/compressed \
        --layers 7,14 \
        --test
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

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
    """Get semantic primes with minimal contexts."""
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime
            else:
                context = prime
            contexts.append((prime, context, category))
    return contexts


def find_dominant_output_dims(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    n_dims: int = 5,
) -> list[tuple[int, float]]:
    """Find the dominant output dimensions for the MLP in a layer."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    all_deltas = []

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
                    h_in = mx.array(h)
                    mx.eval(h_in)

                    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

                    if 'operator_norm' in layer_keys:
                        norm1 = layer['operator_norm']
                        norm2 = layer['ffn_norm']
                        mlp = layer['feed_forward']
                        if 'conv' in layer_keys:
                            self_attn = layer['conv']
                        else:
                            self_attn = layer['self_attn']
                    else:
                        raise ValueError(f"Unknown layer type: {layer_keys}")

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
                    all_deltas.append(mlp_delta)
                    break

        except Exception:
            continue

    if not all_deltas:
        raise ValueError(f"No deltas collected for layer {layer_idx}")

    Delta = mx.stack(all_deltas, axis=0)
    mx.eval(Delta)

    energy = mx.sum(Delta * Delta, axis=0)
    mx.eval(energy)

    energy_list = energy.tolist()
    total_energy = sum(energy_list)

    sorted_dims = sorted(
        range(len(energy_list)),
        key=lambda i: energy_list[i],
        reverse=True
    )

    return [
        (dim, energy_list[dim] / total_energy)
        for dim in sorted_dims[:n_dims]
    ]


def compress_layer_mlp(
    model: Any,
    layer_idx: int,
    dims_to_keep: list[int],
) -> tuple[float, int, int]:
    """Compress the MLP w2 matrix by keeping only specified output dimensions."""
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

    if 'feed_forward' in layer_keys:
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight
    else:
        raise ValueError(f"Cannot find MLP in layer {layer_idx}")

    # Create sparse w2
    w2_sparse = mx.zeros_like(w2)
    for dim in dims_to_keep:
        row = w2[dim:dim+1, :]
        indices = mx.array([dim])
        w2_sparse = w2_sparse.at[indices].add(row)
    mx.eval(w2_sparse)

    # Compute error
    diff = w2 - w2_sparse
    error = float(mx.linalg.norm(diff)) / float(mx.linalg.norm(w2))

    orig_params = w2.shape[0] * w2.shape[1]
    new_params = len(dims_to_keep) * w2.shape[1]

    # Apply compression
    mlp['w2'].weight = w2_sparse
    mx.eval(model.parameters())

    return error, orig_params, new_params


def save_compressed_model(
    model: Any,
    tokenizer: Any,
    source_path: str,
    output_path: str,
):
    """Save the compressed model."""
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

    logger.info("Saved compressed model to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-layer compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layers to compress (e.g., 7,14)")
    parser.add_argument("--n-dims", type=int, default=1,
                        help="Number of output dimensions to keep per layer")
    parser.add_argument("--test", action="store_true",
                        help="Run inference test after compression")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    layers = [int(l.strip()) for l in args.layers.split(",")]

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    total_orig = 0
    total_new = 0

    for layer_idx in layers:
        logger.info("\n=== Layer %d ===", layer_idx)

        # Find dominant dimensions
        dominant_dims = find_dominant_output_dims(model, tokenizer, layer_idx, n_dims=10)

        logger.info("Top dimensions by energy:")
        for i, (dim, frac) in enumerate(dominant_dims[:5]):
            logger.info("  %d. Dim %d: %.1f%%", i + 1, dim, frac * 100)

        dims_to_keep = [dim for dim, _ in dominant_dims[:args.n_dims]]
        energy = sum(frac for _, frac in dominant_dims[:args.n_dims])

        logger.info("Keeping %d dims (%.1f%% energy)", len(dims_to_keep), energy * 100)

        # Compress
        error, orig_params, new_params = compress_layer_mlp(model, layer_idx, dims_to_keep)
        compression = orig_params / new_params

        logger.info("Compression: %dx (error: %.1f%%)", int(compression), error * 100)

        total_orig += orig_params
        total_new += new_params

    logger.info("\n=== TOTAL ===")
    logger.info("Original params in compressed layers: %d", total_orig)
    logger.info("Compressed params: %d", total_new)
    logger.info("Overall compression: %.1fx", total_orig / total_new)

    # Test inference
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

    # Save
    logger.info("\nSaving compressed model...")
    save_compressed_model(model, tokenizer, args.model, args.output)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
