#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Delta-Based Model Compression
"""
Delta-Based Model Compression

Compresses transformer layers based on the finding that certain layers
have low-dimensional delta contributions.

Key finding: Layer 7's MLP output is 98.3% concentrated in dimension 249.
This allows us to zero out 99.4% of the w2 weight matrix with minimal impact.

Usage:
    python delta_compress_model.py \
        --model /path/to/model \
        --output /path/to/compressed \
        --layer 7
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Semantic primes for analysis (from manifold_rotation_analysis.py)
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
    """Get semantic primes with minimal contexts for activation."""
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime
            elif prime in ["this", "the same", "other", "else"]:
                context = f"{prime} thing"
            elif prime in ["one", "two", "some", "all", "many", "much", "little", "few"]:
                context = f"{prime} things"
            elif prime in ["good", "bad", "big", "small", "true"]:
                context = f"It is {prime}"
            elif prime in ["think", "know", "want", "feel", "see", "hear"]:
                context = f"I {prime}"
            elif prime in ["say"]:
                context = "I say"
            elif prime in ["words"]:
                context = "words"
            elif prime in ["do", "happen", "move"]:
                context = f"Things {prime}"
            elif prime in ["there is"]:
                context = "There is something"
            elif prime in ["be"]:
                context = "I am"
            elif prime in ["live", "die"]:
                context = f"People {prime}"
            elif prime in ["have", "part"]:
                context = "I have"
            elif prime in ["not"]:
                context = "not this"
            elif prime in ["maybe"]:
                context = "maybe"
            elif prime in ["can"]:
                context = "I can"
            elif prime in ["because", "if"]:
                context = prime
            elif prime in ["when", "now", "before", "after"]:
                context = prime
            elif prime in ["a long time", "a short time", "moment"]:
                context = prime
            elif prime in ["where", "here"]:
                context = prime
            elif prime in ["above", "below", "far", "near", "inside"]:
                context = prime
            elif prime in ["side", "touch"]:
                context = prime
            elif prime in ["kind of", "part of", "like"]:
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
    """Find the dominant output dimensions for the MLP in a layer.

    Returns list of (dim, energy_fraction) tuples.
    """
    import mlx.core as mx

    contexts = get_prime_contexts()
    all_deltas = []

    for prime, context, category in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            # Forward to layer
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

                    # Get layer components
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

                    # Forward through layer components
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

                    # Get MLP delta (last token)
                    mlp_delta = mlp_out[0, -1, :]
                    mx.eval(mlp_delta)
                    all_deltas.append(mlp_delta)
                    break

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    if not all_deltas:
        raise ValueError(f"No deltas collected for layer {layer_idx}")

    # Stack and compute energy per dimension
    Delta = mx.stack(all_deltas, axis=0)  # (n_samples, hidden_dim)
    mx.eval(Delta)

    energy = mx.sum(Delta * Delta, axis=0)  # (hidden_dim,)
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
    """Compress the MLP w2 matrix by keeping only specified output dimensions.

    Returns:
        Tuple of (weight_error, original_params, new_params)
    """
    import mlx.core as mx

    layer = model.model.layers[layer_idx]
    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

    if 'feed_forward' in layer_keys:
        mlp = layer['feed_forward']
        w2 = mlp['w2'].weight
    elif hasattr(layer, 'mlp'):
        mlp = layer.mlp
        w2 = mlp.down_proj.weight if hasattr(mlp, 'down_proj') else mlp.w2.weight
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

    # Params
    orig_params = w2.shape[0] * w2.shape[1]
    new_params = len(dims_to_keep) * w2.shape[1]

    # Apply compression
    if 'feed_forward' in layer_keys:
        mlp['w2'].weight = w2_sparse
    elif hasattr(mlp, 'down_proj'):
        mlp.down_proj.weight = w2_sparse
    else:
        mlp.w2.weight = w2_sparse
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

    # Copy config and tokenizer files
    source_dir = Path(source_path)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    # Flatten parameters using MLX utility
    flat_params = tree_flatten(model.parameters())
    weights = {k: v for k, v in flat_params}

    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    logger.info("Saved compressed model to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Delta-based model compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--layer", type=int, required=True, help="Layer to compress")
    parser.add_argument("--n-dims", type=int, default=1,
                        help="Number of output dimensions to keep (default: 1)")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically find best dimensions to keep")
    parser.add_argument("--test", action="store_true",
                        help="Run inference test after compression")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    # Load model
    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    # Find dominant dimensions
    logger.info("\nAnalyzing layer %d MLP output dimensions...", args.layer)
    dominant_dims = find_dominant_output_dims(model, tokenizer, args.layer, n_dims=10)

    logger.info("Top 10 output dimensions by energy:")
    for i, (dim, frac) in enumerate(dominant_dims):
        logger.info("  %d. Dim %d: %.1f%%", i + 1, dim, frac * 100)

    # Select dimensions to keep
    if args.auto:
        # Keep dimensions until we capture 95% of energy
        dims_to_keep = []
        cumulative = 0.0
        for dim, frac in dominant_dims:
            dims_to_keep.append(dim)
            cumulative += frac
            if cumulative >= 0.95:
                break
        logger.info("\nAuto-selected %d dimensions (%.1f%% energy)",
                    len(dims_to_keep), cumulative * 100)
    else:
        dims_to_keep = [dim for dim, _ in dominant_dims[:args.n_dims]]
        energy = sum(frac for _, frac in dominant_dims[:args.n_dims])
        logger.info("\nKeeping top %d dimensions (%.1f%% energy)",
                    args.n_dims, energy * 100)

    # Compress
    logger.info("\nCompressing layer %d MLP...", args.layer)
    error, orig_params, new_params = compress_layer_mlp(model, args.layer, dims_to_keep)
    compression = orig_params / new_params

    logger.info("  Weight Frobenius error: %.1f%%", error * 100)
    logger.info("  Original params: %d", orig_params)
    logger.info("  Compressed params: %d", new_params)
    logger.info("  Compression ratio: %.0fx", compression)

    # Test inference
    if args.test:
        logger.info("\nTesting compressed model...")
        prompts = [
            "The answer to 2+2 is",
            "Hello, my name is",
            "The capital of France is",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=15, verbose=False)
            logger.info("  %s -> %s", prompt, output[len(prompt):][:40])

    # Save
    logger.info("\nSaving compressed model...")
    save_compressed_model(model, tokenizer, args.model, args.output)

    logger.info("\nDone! Compressed model saved to %s", args.output)


if __name__ == "__main__":
    main()
