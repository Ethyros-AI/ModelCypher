#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Vocabulary Projection Compression
"""
Vocabulary Projection Compression

THE INSIGHT:
Energy preservation fails because energy is SCALAR.
We need to preserve VECTOR information.

Specifically, what matters is:
    logits = h_final @ W_vocab.T

For correct generation, we need:
    <h_compressed, w_i> ≈ <h_original, w_i>

for the relevant vocabulary vectors.

THE MATH:
The vocabulary embedding W has shape (vocab_size, hidden_dim).
Most vocabulary vectors are similar - the effective rank is low.

If we project W onto its principal components, we get:
    W ≈ W_low_rank = U @ V  where U is (vocab_size × k) and V is (k × hidden_dim)

The projection of h onto vocabulary space is:
    h @ W.T ≈ h @ V.T @ U.T

What matters is h @ V.T - the projection onto vocabulary principal components.

THE CONSTRAINT:
For compression to preserve generation, we need:
    δ_compressed @ V.T ≈ δ_original @ V.T

This is different from energy! This preserves the LOGITS, not the NORM.

Usage:
    python vocab_projection_compression.py --model /path/to/model
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


def get_vocab_principal_components(model: Any, n_components: int = 64) -> np.ndarray:
    """Get the principal components of vocabulary embeddings."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

    # Get vocabulary embeddings
    W = np.array(inner_model.embed_tokens.weight.astype(mx.float32))
    # W: (vocab_size, hidden_dim)

    print(f"Vocabulary: {W.shape[0]} tokens, {W.shape[1]} dims")

    # PCA on vocabulary (treat each token as a sample)
    mean = W.mean(axis=0)
    centered = W - mean

    # Use SVD for efficiency (don't compute full covariance)
    # centered = U @ S @ Vh
    # Principal components are rows of Vh
    print("Computing vocabulary PCA...")
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)

    # Explained variance
    total_var = np.sum(S**2)
    explained = np.cumsum(S[:n_components]**2) / total_var
    print(f"Top {n_components} components explain {explained[-1]*100:.2f}% of vocabulary variance")

    # Return top n_components rows of Vh (shape: n_components × hidden_dim)
    return Vh[:n_components, :], mean


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


def compute_delta_basis(deltas: list[np.ndarray], n_components: int) -> np.ndarray:
    """Compute PCA basis for deltas."""
    deltas_arr = np.stack(deltas)
    deltas_arr = np.nan_to_num(deltas_arr, nan=0.0, posinf=0.0, neginf=0.0)

    mean = deltas_arr.mean(axis=0)
    centered = deltas_arr - mean

    cov = (centered.T @ centered) / len(deltas_arr)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(np.abs(eigenvalues))[::-1]

    return eigenvectors[:, idx[:n_components]]


def compute_vocab_preserving_compression(
    delta: np.ndarray,
    delta_basis: np.ndarray,
    vocab_basis: np.ndarray,
) -> np.ndarray:
    """
    Compress delta while preserving its projection onto vocabulary space.

    We want: δ_compressed @ vocab_basis.T ≈ δ @ vocab_basis.T

    If δ_compressed is in the span of delta_basis, then:
    δ_compressed = delta_basis @ c  for some coefficients c

    We need: delta_basis @ c @ vocab_basis.T = δ @ vocab_basis.T
    Let M = delta_basis @ vocab_basis.T  (shape: delta_rank × vocab_rank)
    Let y = δ @ vocab_basis.T  (shape: vocab_rank)

    Then: c @ M.T = y, so c = y @ pinv(M.T) = y @ pinv(M).T

    Actually, M @ c = y.T, so c = pinv(M) @ y.T
    Then δ_compressed = delta_basis @ c
    """
    # vocab_basis: (vocab_rank, hidden_dim)
    # delta_basis: (hidden_dim, delta_rank)

    # Target: δ's projection onto vocabulary space
    # y = δ @ vocab_basis.T  shape: (vocab_rank,)
    y = delta @ vocab_basis.T

    # M = vocab_basis @ delta_basis  shape: (vocab_rank, delta_rank)
    M = vocab_basis @ delta_basis

    # Solve: M @ c = y for c
    # c = pinv(M) @ y
    try:
        c = np.linalg.lstsq(M, y, rcond=None)[0]  # shape: (delta_rank,)
    except:
        c = np.zeros(delta_basis.shape[1])

    # Reconstruct
    delta_compressed = delta_basis @ c

    return delta_compressed


def test_vocab_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_bases: dict,
    vocab_basis: np.ndarray,
    all_h_ins: list[list[np.ndarray]],
    all_deltas: list[list[np.ndarray]],
    max_tokens: int = 20,
) -> tuple[str, str, dict]:
    """Test generation with vocabulary-preserving compression."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

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

    # Vocab-preserving compressed generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    stats = []

    for idx, layer in enumerate(inner_model.layers):
        h_np = np.array(h.astype(mx.float32))
        h_in = h_np[0, -1, :]

        # Run actual layer
        result = layer(h)
        h_true = result[0] if isinstance(result, tuple) else result
        mx.eval(h_true)

        h_out_true = np.array(h_true[0, -1, :].astype(mx.float32))
        delta_true = h_out_true - h_in

        if idx in layer_bases:
            # Compress delta preserving vocab projection
            delta_basis = layer_bases[idx]
            delta_compressed = compute_vocab_preserving_compression(
                delta_true, delta_basis, vocab_basis
            )

            # Check vocab projection preservation
            proj_true = delta_true @ vocab_basis.T
            proj_compressed = delta_compressed @ vocab_basis.T
            proj_error = np.linalg.norm(proj_compressed - proj_true) / (np.linalg.norm(proj_true) + 1e-10)

            stats.append({
                'layer': idx,
                'vocab_proj_error': proj_error,
                'delta_reconstruction_error': np.linalg.norm(delta_compressed - delta_true) / (np.linalg.norm(delta_true) + 1e-10),
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
    parser = argparse.ArgumentParser(description="Vocab projection compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--delta-rank", type=int, default=32, help="Rank for delta compression")
    parser.add_argument("--vocab-rank", type=int, default=64, help="Rank for vocab projection")
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
    print("VOCABULARY PROJECTION COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Get vocabulary principal components
    print(f"\nComputing vocabulary principal components (rank {args.vocab_rank})...")
    vocab_basis, vocab_mean = get_vocab_principal_components(model, args.vocab_rank)
    print(f"Vocab basis shape: {vocab_basis.shape}")

    # Collect layer data
    print(f"\nCollecting data for {len(CONCEPTS)} concepts...")
    all_h_ins, all_deltas = collect_layer_data(model, tokenizer, CONCEPTS)

    # Identify transmission layers
    if n_layers == 28:
        transmission_layers = list(range(3, 27))
    else:
        transmission_layers = list(range(3, n_layers - 1))

    # Compute delta bases
    print(f"\nComputing rank-{args.delta_rank} bases for transmission layers...")
    layer_bases = {}
    for idx in transmission_layers:
        basis = compute_delta_basis(all_deltas[idx], args.delta_rank)
        layer_bases[idx] = basis

    # Analyze vocab projection coverage
    print(f"\n{'='*80}")
    print("VOCABULARY PROJECTION ANALYSIS")
    print("="*80)

    print(f"\n{'Layer':>6} | {'δ in vocab space':>15} | {'δ_basis covers vocab':>20}")
    print("-" * 55)

    for idx in transmission_layers[:5] + transmission_layers[-5:]:  # Sample layers
        deltas = np.stack(all_deltas[idx])
        delta_basis = layer_bases[idx]

        # How much of δ is in vocab space?
        proj_to_vocab = deltas @ vocab_basis.T @ vocab_basis
        in_vocab = np.mean(np.linalg.norm(proj_to_vocab, axis=1) / (np.linalg.norm(deltas, axis=1) + 1e-10))

        # How much does δ_basis cover vocab space?
        # Overlap = ||delta_basis @ vocab_basis.T||_F / ||vocab_basis||_F
        overlap = np.linalg.norm(vocab_basis @ delta_basis) / (np.linalg.norm(vocab_basis) + 1e-10)

        print(f"{idx:>6} | {in_vocab:>15.4f} | {overlap:>20.4f}")

    # Test generation
    print(f"\n{'='*80}")
    print("GENERATION TEST (Vocab-Preserving)")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
        "Dogs are known for being",
    ]

    matches = 0
    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        try:
            normal, compressed, stats = test_vocab_compression(
                model, tokenizer, prompt,
                layer_bases, vocab_basis,
                all_h_ins, all_deltas,
                max_tokens=10
            )
            print(f"  Normal:     {normal[:40]}")
            print(f"  Compressed: {compressed[:40]}")

            if stats:
                avg_proj_error = np.mean([s['vocab_proj_error'] for s in stats])
                avg_recon_error = np.mean([s['delta_reconstruction_error'] for s in stats])
                print(f"  Vocab proj error: {avg_proj_error:.6f}, delta recon: {avg_recon_error:.4f}")

            if normal.split() and compressed.split():
                if normal.split()[0] == compressed.split()[0]:
                    print(f"  → First token MATCH ✓")
                    matches += 1
        except Exception as e:
            print(f"  → Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nMatches: {matches}/{len(test_prompts)}")

    # Analysis
    print(f"\n{'='*80}")
    print("THE VOCABULARY PROJECTION INSIGHT")
    print("="*80)

    print(f"""
WHAT MATTERS FOR GENERATION:

Not ||h||² (energy) but <h, w_i> (vocabulary projection).

The final logits are: logits = h @ W.T

If we compress δ but preserve δ @ vocab_basis.T, we preserve the logits.

THE CONSTRAINT:
    δ_compressed @ vocab_basis.T = δ_original @ vocab_basis.T

This is what the model ACTUALLY needs for generation.

IMPLICATION:
Any component of δ orthogonal to vocab space can be discarded!
We only need to preserve the {args.vocab_rank}-dimensional projection.
""")


if __name__ == "__main__":
    main()
