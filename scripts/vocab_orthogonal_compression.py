#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Vocabulary-Orthogonal Compression
"""
Vocabulary-Orthogonal Compression

THE INSIGHT FROM PREVIOUS EXPERIMENTS:
- δ is only 20-35% in vocab space
- The vocab-relevant part is what matters for generation
- Our delta_basis doesn't capture vocab-relevant directions

THE NEW APPROACH:
Split δ into:
    δ = δ_vocab + δ_perp

Where:
    δ_vocab = (δ @ V.T) @ V  (projection onto vocab space)
    δ_perp = δ - δ_vocab    (orthogonal to vocab)

Then:
    - KEEP δ_vocab EXACTLY (this is what affects generation)
    - COMPRESS only δ_perp (can be lossy since it doesn't affect output)

This is EXACT for the parts that matter!

Compression ratio:
    - δ_vocab: 64D (vocab rank) - kept exactly
    - δ_perp: 2048-64 = 1984D - can be compressed to k dims

Total: 64 + k instead of 2048 = massive compression

Usage:
    python vocab_orthogonal_compression.py --model /path/to/model
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


CONCEPTS = [
    "apple", "orange", "banana", "car", "truck", "house", "tree", "book",
    "dog", "cat", "bird", "fish", "horse", "elephant", "tiger", "whale",
    "love", "hate", "fear", "joy", "anger", "peace", "war", "truth",
    "hot", "cold", "fast", "slow", "big", "small", "good", "bad",
]


def get_vocab_basis(model: Any, n_components: int = 128) -> np.ndarray:
    """Get the principal components of vocabulary embeddings."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    W = np.array(inner_model.embed_tokens.weight.astype(mx.float32))

    print(f"Vocabulary: {W.shape[0]} tokens, {W.shape[1]} dims")

    mean = W.mean(axis=0)
    centered = W - mean

    print("Computing vocabulary SVD...")
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)

    total_var = np.sum(S**2)
    explained = np.cumsum(S[:n_components]**2) / total_var
    print(f"Top {n_components} components explain {explained[-1]*100:.2f}% of vocabulary variance")

    return Vh[:n_components, :]


def collect_layer_data(model: Any, tokenizer: Any, words: list[str]):
    """Collect h_in and delta for all layers."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    all_h_ins = [[] for _ in range(n_layers)]
    all_deltas = [[] for _ in range(n_layers)]

    for word in words:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(inner_model.layers):
            h_in = np.array(h[0, -1, :].astype(mx.float32))

            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

            h_out = np.array(h[0, -1, :].astype(mx.float32))
            delta = h_out - h_in

            all_h_ins[idx].append(h_in)
            all_deltas[idx].append(delta)

    return all_h_ins, all_deltas


def decompose_delta(delta: np.ndarray, vocab_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose delta into vocab-parallel and vocab-orthogonal parts.

    δ = δ_vocab + δ_perp
    Where δ_vocab = V.T @ V @ δ (projection onto vocab space)
    """
    # vocab_basis: (vocab_rank, hidden_dim)
    # δ: (hidden_dim,)

    # Project onto vocab space
    coeffs = vocab_basis @ delta  # (vocab_rank,)
    delta_vocab = vocab_basis.T @ coeffs  # (hidden_dim,)

    # Orthogonal part
    delta_perp = delta - delta_vocab

    return delta_vocab, delta_perp


def test_vocab_orthogonal_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vocab_basis: np.ndarray,
    compress_perp: bool = True,
    perp_scale: float = 0.0,  # 0 = discard perp, 1 = keep perp
    max_tokens: int = 20,
) -> tuple[str, str, dict]:
    """
    Test generation with vocab-orthogonal compression.

    Strategy: Keep δ_vocab exactly, optionally discard/scale δ_perp.
    """
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    # Normal generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    normal_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        normal_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    normal_output = tokenizer.decode(normal_generated)

    # Compressed generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    stats = []
    transmission_layers = list(range(3, 27)) if n_layers == 28 else list(range(3, n_layers - 1))

    for idx, layer in enumerate(inner_model.layers):
        h_np = np.array(h.astype(mx.float32))
        h_in = h_np[0, -1, :]

        # Run actual layer
        result = layer(h)
        h_true = result[0] if isinstance(result, tuple) else result
        mx.eval(h_true)

        h_out_true = np.array(h_true[0, -1, :].astype(mx.float32))
        delta_true = h_out_true - h_in

        if compress_perp and idx in transmission_layers:
            # Decompose delta
            delta_vocab, delta_perp = decompose_delta(delta_true, vocab_basis)

            # Compress: keep vocab exactly, scale perp
            delta_compressed = delta_vocab + perp_scale * delta_perp

            # Stats
            vocab_ratio = np.linalg.norm(delta_vocab) / (np.linalg.norm(delta_true) + 1e-10)
            perp_ratio = np.linalg.norm(delta_perp) / (np.linalg.norm(delta_true) + 1e-10)

            stats.append({
                'layer': idx,
                'vocab_ratio': vocab_ratio,
                'perp_ratio': perp_ratio,
            })

            # Apply compressed delta
            h_new = h_in + delta_compressed
            h_np_new = h_np.copy()
            h_np_new[0, -1, :] = h_new

            h = mx.array(h_np_new).astype(h.dtype)
            mx.eval(h)
        else:
            h = h_true

    # Final norm
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        logits = model(input_ids)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    compressed_generated = []
    if next_token != tokenizer.eos_token_id:
        compressed_generated.append(next_token)

    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        compressed_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    compressed_output = tokenizer.decode(compressed_generated)

    return normal_output, compressed_output, stats


def main():
    parser = argparse.ArgumentParser(description="Vocab-orthogonal compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--vocab-rank", type=int, default=256, help="Vocab projection rank")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("VOCABULARY-ORTHOGONAL COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Get vocab basis
    print(f"\nComputing vocab basis (rank {args.vocab_rank})...")
    vocab_basis = get_vocab_basis(model, args.vocab_rank)

    # Test different perp_scale values
    print(f"\n{'='*80}")
    print("TESTING PERP SCALE VALUES")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    for perp_scale in [0.0, 0.5, 1.0]:
        print(f"\n--- perp_scale = {perp_scale} ---")

        matches = 0
        for prompt in test_prompts:
            normal, compressed, stats = test_vocab_orthogonal_compression(
                model, tokenizer, prompt,
                vocab_basis,
                compress_perp=True,
                perp_scale=perp_scale,
                max_tokens=8
            )

            if stats:
                avg_vocab = np.mean([s['vocab_ratio'] for s in stats])
                avg_perp = np.mean([s['perp_ratio'] for s in stats])

            normal_first = normal.split()[0] if normal.split() else ""
            compressed_first = compressed.split()[0] if compressed.split() else ""
            match = "✓" if normal_first == compressed_first else "✗"

            if normal_first == compressed_first:
                matches += 1

            print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match}")

        print(f"  Matches: {matches}/{len(test_prompts)}")

        # Compression ratio
        # Keep: vocab_rank per layer
        # Discard: (1 - perp_scale) * (hidden_dim - vocab_rank) per layer
        if perp_scale < 1.0:
            effective_dims = args.vocab_rank + perp_scale * (hidden_dim - args.vocab_rank)
            compression = hidden_dim / effective_dims
            print(f"  Effective compression: {compression:.1f}x")

    # Analysis
    print(f"\n{'='*80}")
    print("VOCAB-ORTHOGONAL INSIGHT")
    print("="*80)

    # Get decomposition stats
    _, _, stats = test_vocab_orthogonal_compression(
        model, tokenizer, "The capital of France is",
        vocab_basis, compress_perp=True, perp_scale=0.0, max_tokens=5
    )

    avg_vocab = np.mean([s['vocab_ratio'] for s in stats])
    avg_perp = np.mean([s['perp_ratio'] for s in stats])

    print(f"""
DELTA DECOMPOSITION:

Average across transmission layers:
  - ||δ_vocab|| / ||δ||: {avg_vocab:.4f} ({avg_vocab*100:.1f}%)
  - ||δ_perp|| / ||δ||:  {avg_perp:.4f} ({avg_perp*100:.1f}%)

THE CRITICAL QUESTION:
If {avg_perp*100:.1f}% of δ is orthogonal to vocab space,
why does discarding it break generation?

POSSIBILITIES:
1. The perp component affects FUTURE layers (propagation)
2. The vocab basis doesn't capture the full output space
3. The decomposition is correct but our compression loses phase/sign

NEXT STEP:
Try keeping more of vocab space (higher rank vocab basis).
The 9% explained variance is too low - we're missing most of the vocabulary structure.
""")


if __name__ == "__main__":
    main()
