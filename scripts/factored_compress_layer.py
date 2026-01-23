#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Factored Layer Compression
"""
Factored Layer Compression

Uses activation PCA to create a factored weight representation.

Key insight: If output lives in rank-k subspace, we can factor:
    w2 @ h = P @ (P.T @ w2) @ h = P @ w2_eff @ h

Storage:
    Original: w2 [hidden_dim, intermediate_dim] = 4.7M params
    Factored: P [hidden_dim, k] + w2_eff [k, intermediate_dim]
              = hidden_dim * k + k * intermediate_dim
              = k * (hidden_dim + intermediate_dim)

For k=1: 1 * (1024 + 4608) = 5632 params (800x compression)

Usage:
    python factored_compress_layer.py \
        --model /path/to/model \
        --layer 14 \
        --rank 1 \
        --test
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

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


def collect_mlp_outputs(model: Any, tokenizer: Any, layer_idx: int) -> Any:
    """Collect MLP output activations."""
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

    return mx.stack(outputs, axis=0) if outputs else None


def compute_output_pca(outputs: Any, rank: int) -> tuple[Any, float]:
    """Compute top-k principal components of outputs."""
    import mlx.core as mx

    mean = mx.mean(outputs, axis=0)
    centered = outputs - mean
    mx.eval(centered)

    n = outputs.shape[0]
    cov = (centered.T @ centered) / n
    mx.eval(cov)

    cov_f32 = cov.astype(mx.float32)
    mx.eval(cov_f32)
    cov_np = np.array(cov_f32)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_np)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total_var
    variance_captured = cumulative[rank - 1]

    P = eigenvectors[:, :rank]
    P_mx = mx.array(P.astype(np.float32))
    mx.eval(P_mx)

    return P_mx, float(variance_captured)


def compress_layer_factored(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    rank: int,
) -> tuple[float, int, int]:
    """Compress MLP using factored representation.

    Instead of storing w2 [hidden, intermediate], store:
    - P [hidden, rank]: output basis
    - w2_eff [rank, intermediate] = P.T @ w2: effective weight

    Inference: output = P @ w2_eff @ hidden_state

    The trick: We reconstruct w2_approx = P @ P.T @ w2 and store that.
    This is mathematically equivalent to the factored form but compatible
    with standard model format.

    Returns:
        variance_captured: Fraction of output variance captured
        original_params: Original w2 params
        effective_params: Factored representation params
    """
    import mlx.core as mx

    logger.info("Collecting MLP outputs for layer %d...", layer_idx)
    outputs = collect_mlp_outputs(model, tokenizer, layer_idx)

    if outputs is None:
        raise ValueError(f"No outputs for layer {layer_idx}")

    logger.info("Collected %d samples", outputs.shape[0])

    # Compute PCA
    P, variance_captured = compute_output_pca(outputs, rank)
    logger.info("Variance in top %d PCs: %.1f%%", rank, variance_captured * 100)

    # Get w2
    layer = model.model.layers[layer_idx]
    mlp = layer['feed_forward']
    w2 = mlp['w2'].weight  # [hidden_dim, intermediate_dim]

    hidden_dim, intermediate_dim = w2.shape
    original_params = hidden_dim * intermediate_dim

    # Compute factored representation
    # w2_eff = P.T @ w2  [rank, intermediate_dim]
    w2_f32 = w2.astype(mx.float32)
    mx.eval(w2_f32)

    w2_eff = P.T @ w2_f32  # [rank, intermediate_dim]
    mx.eval(w2_eff)

    # Effective params: P + w2_eff
    effective_params = hidden_dim * rank + rank * intermediate_dim

    logger.info("\nFactored representation:")
    logger.info("  P: [%d, %d] = %d params", hidden_dim, rank, hidden_dim * rank)
    logger.info("  w2_eff: [%d, %d] = %d params", rank, intermediate_dim, rank * intermediate_dim)
    logger.info("  Total: %d params (%.1fx compression)",
                effective_params, original_params / effective_params)

    # Reconstruct w2_approx = P @ w2_eff = P @ P.T @ w2
    w2_approx = P @ w2_eff
    mx.eval(w2_approx)

    # Check reconstruction quality on actual outputs
    # The key is: does P @ w2_eff @ h ≈ w2 @ h for typical h?
    logger.info("\nVerifying on activation samples...")

    # Collect intermediate hidden states (after SiLU gate)
    # We need the input to w2 to verify
    h_inputs = collect_mlp_inputs(model, tokenizer, layer_idx)
    if h_inputs is not None:
        # Original outputs
        orig_outputs = h_inputs @ w2_f32.T  # [n, hidden]
        mx.eval(orig_outputs)

        # Factored outputs
        factored_outputs = h_inputs @ w2_approx.T  # [n, hidden]
        mx.eval(factored_outputs)

        # Error on outputs
        output_diff = orig_outputs - factored_outputs
        output_error = float(mx.linalg.norm(output_diff)) / float(mx.linalg.norm(orig_outputs))
        logger.info("  Output reconstruction error: %.2f%%", output_error * 100)

    # Weight Frobenius error
    weight_diff = w2_f32 - w2_approx
    weight_error = float(mx.linalg.norm(weight_diff)) / float(mx.linalg.norm(w2_f32))
    logger.info("  Weight Frobenius error: %.1f%%", weight_error * 100)

    # Apply
    w2_new = w2_approx.astype(w2.dtype)
    mx.eval(w2_new)
    mlp['w2'].weight = w2_new
    mx.eval(model.parameters())

    return variance_captured, original_params, effective_params


def collect_mlp_inputs(model: Any, tokenizer: Any, layer_idx: int) -> Any:
    """Collect MLP intermediate hidden states (input to w2)."""
    import mlx.core as mx

    contexts = get_prime_contexts()
    inputs = []

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

                    # MLP structure: out = w2 @ silu(w1 @ h) * (w3 @ h)
                    # We need the hidden state AFTER the gate
                    w1 = mlp['w1']
                    w3 = mlp['w3']

                    h_last = h_before_mlp[0, -1:, :]  # [1, hidden]
                    mx.eval(h_last)

                    gate = w1(h_last)  # [1, intermediate]
                    mx.eval(gate)

                    up = w3(h_last)  # [1, intermediate]
                    mx.eval(up)

                    # SiLU gate
                    hidden = mx.sigmoid(gate) * gate * up  # SiLU(gate) * up
                    mx.eval(hidden)

                    inputs.append(hidden[0])  # [intermediate]
                    break
        except Exception as e:
            logger.debug("Failed: %s", e)
            continue

    return mx.stack(inputs, axis=0) if inputs else None


def save_compressed_model(model: Any, tokenizer: Any, source_path: str, output_path: str):
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

    logger.info("Saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Factored layer compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--layer", type=int, required=True, help="Layer to compress")
    parser.add_argument("--rank", type=int, default=1, help="Factorization rank")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    logger.info("\n=== Layer %d Factored Compression (rank=%d) ===", args.layer, args.rank)

    variance, orig, effective = compress_layer_factored(model, tokenizer, args.layer, args.rank)

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

    if args.output:
        save_compressed_model(model, tokenizer, args.model, args.output)


if __name__ == "__main__":
    main()
