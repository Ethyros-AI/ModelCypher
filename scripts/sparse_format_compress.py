#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# True Sparse Format Compression
"""
True Sparse Format Compression

Stores only the non-zero rows of w2 matrices for actual disk savings.

Current approach zeros rows but stores full matrix (~10MB each).
This stores sparse data + indices for real compression.

Storage format:
    model.layers.N.feed_forward.w2.weight -> full matrix (deleted)
    model.layers.N.feed_forward.w2.sparse_data -> [k, intermediate] (the rows)
    model.layers.N.feed_forward.w2.sparse_indices -> [k] (which rows)

Inference requires custom module that reconstructs on the fly.

Usage:
    python sparse_format_compress.py \
        --model /path/to/model \
        --output /path/to/compressed \
        --layers 7
"""

from __future__ import annotations

import argparse
import json
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
}


def get_prime_contexts() -> list[tuple[str, str, str]]:
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            contexts.append((prime, prime, category))
    return contexts


def find_dominant_dims(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    n_dims: int = 1,
) -> list[tuple[int, float]]:
    """Find dominant output dimensions by energy."""
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
                    all_deltas.append(delta)
                    break
        except Exception:
            continue

    if not all_deltas:
        raise ValueError(f"No deltas for layer {layer_idx}")

    Delta = mx.stack(all_deltas, axis=0)
    mx.eval(Delta)

    energy = mx.sum(Delta * Delta, axis=0)
    mx.eval(energy)

    energy_list = energy.tolist()
    total = sum(energy_list)

    sorted_dims = sorted(range(len(energy_list)), key=lambda i: energy_list[i], reverse=True)

    return [(dim, energy_list[dim] / total) for dim in sorted_dims[:n_dims]]


def create_sparse_model(
    model: Any,
    tokenizer: Any,
    source_path: str,
    output_path: str,
    layers_to_compress: list[int],
    n_dims: int = 1,
):
    """Create a model with sparse w2 storage.

    For each compressed layer:
    - Find dominant output dimension(s)
    - Extract just those rows of w2
    - Store as sparse_data + sparse_indices

    Note: This creates a NON-STANDARD model format. Loading requires
    custom code to reconstruct the full w2 matrix.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer
    source_dir = Path(source_path)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    # Get all parameters
    flat_params = tree_flatten(model.parameters())
    params = {k: v for k, v in flat_params}

    # Compression metadata
    compression_info = {
        "format": "sparse_w2",
        "layers": {},
    }

    original_size = 0
    compressed_size = 0

    for layer_idx in layers_to_compress:
        logger.info("\nProcessing layer %d...", layer_idx)

        # Find dominant dimensions
        dom_dims = find_dominant_dims(model, tokenizer, layer_idx, n_dims)
        logger.info("  Dominant dims: %s", [(d, f"{e*100:.1f}%") for d, e in dom_dims])

        dims = [d for d, _ in dom_dims]
        total_energy = sum(e for _, e in dom_dims)

        # Get w2 weight
        w2_key = f"model.layers.{layer_idx}.feed_forward.w2.weight"
        if w2_key not in params:
            logger.warning("  Key %s not found", w2_key)
            continue

        w2 = params[w2_key]
        hidden_dim, intermediate_dim = w2.shape

        # Original size
        orig_bytes = hidden_dim * intermediate_dim * 2  # bf16 = 2 bytes
        original_size += orig_bytes

        # Extract sparse data
        sparse_rows = []
        for dim in dims:
            row = w2[dim, :]
            sparse_rows.append(row)
        sparse_data = mx.stack(sparse_rows, axis=0)  # [n_dims, intermediate_dim]
        mx.eval(sparse_data)

        sparse_indices = mx.array(dims, dtype=mx.int32)
        mx.eval(sparse_indices)

        # Compressed size
        comp_bytes = n_dims * intermediate_dim * 2 + n_dims * 4  # data + indices
        compressed_size += comp_bytes

        logger.info("  Original: %d bytes", orig_bytes)
        logger.info("  Compressed: %d bytes (%.1fx)", comp_bytes, orig_bytes / comp_bytes)
        logger.info("  Energy captured: %.1f%%", total_energy * 100)

        # Replace in params
        # Remove original w2
        del params[w2_key]

        # Add sparse representation
        sparse_data_key = f"model.layers.{layer_idx}.feed_forward.w2.sparse_data"
        sparse_idx_key = f"model.layers.{layer_idx}.feed_forward.w2.sparse_indices"
        params[sparse_data_key] = sparse_data
        params[sparse_idx_key] = sparse_indices

        compression_info["layers"][str(layer_idx)] = {
            "dims": dims,
            "energy": total_energy,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
        }

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info("Layers compressed: %s", layers_to_compress)
    logger.info("Original size (compressed layers): %d bytes (%.2f MB)",
                original_size, original_size / 1e6)
    logger.info("Compressed size: %d bytes (%.2f MB)",
                compressed_size, compressed_size / 1e6)
    logger.info("Compression ratio: %.1fx", original_size / compressed_size)

    # Save compression info
    info_path = output_dir / "compression_info.json"
    with open(info_path, "w") as f:
        json.dump(compression_info, f, indent=2)
    logger.info("Saved compression info to %s", info_path)

    # Save weights
    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), params)
    logger.info("Saved model to %s", weights_path)

    # Compute actual file size
    actual_size = weights_path.stat().st_size
    source_size = (source_dir / "model.safetensors").stat().st_size
    logger.info("\nActual file sizes:")
    logger.info("  Original: %.2f MB", source_size / 1e6)
    logger.info("  Compressed: %.2f MB", actual_size / 1e6)
    logger.info("  Savings: %.2f MB (%.1f%%)",
                (source_size - actual_size) / 1e6,
                (source_size - actual_size) / source_size * 100)


def main():
    parser = argparse.ArgumentParser(description="True sparse format compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layers to compress (e.g., '7' or '7,14')")
    parser.add_argument("--n-dims", type=int, default=1,
                        help="Number of output dimensions to keep per layer")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    layers = [int(l.strip()) for l in args.layers.split(",")]

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    logger.info("Compressing layers: %s", layers)
    create_sparse_model(model, tokenizer, args.model, args.output, layers, args.n_dims)


if __name__ == "__main__":
    main()
