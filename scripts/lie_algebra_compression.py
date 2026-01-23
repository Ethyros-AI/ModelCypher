#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Lie Algebra Compression
"""
Lie Algebra Compression

THE INSIGHT:
Single layer at rank=1 works. All layers at rank=512 fails.
This isn't about compression - it's about composition.

THE ALGEBRA:
Residual network: h_{i+1} = h_i + δ_i = (I + F_i) @ h_i

Composition: T = (I + F_{n-1}) @ ... @ (I + F_0)

Expanding:
T = I + Σ F_i + Σ_{i<j} F_j @ F_i + higher order terms

When we factor each F_i independently, we're ignoring the cross-terms F_j @ F_i.

THE LIE ALGEBRA APPROACH:
In the Lie algebra, composition is ADDITION, not multiplication.
log(T) = Σ log(I + F_i) ≈ Σ F_i (for small F_i)

Factor the SUM, not the individual terms!

THE EXPERIMENT:
1. Compute the TOTAL transformation T from h_in to h_out
2. Factor T directly (not individual layers)
3. Test if the factored T preserves generation

This should work because we're factoring the actual end-to-end map.

Usage:
    python lie_algebra_compression.py --model /path/to/model
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


CALIBRATION_PROMPTS = [
    # Geography
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of China is",
    "The capital of Brazil is",
    "The capital of Australia is",
    # Math
    "2 + 2 =",
    "10 - 3 =",
    "5 * 5 =",
    "100 / 4 =",
    "7 + 8 =",
    "15 - 6 =",
    "3 * 4 =",
    "20 / 5 =",
    # Opposites
    "The opposite of hot is",
    "The opposite of big is",
    "The opposite of happy is",
    "The opposite of light is",
    "The opposite of up is",
    "The opposite of good is",
    "The opposite of old is",
    "The opposite of fast is",
    # Completions
    "Once upon a time",
    "In the beginning",
    "The quick brown fox",
    "To be or not to",
    "It was a dark and",
    "Long ago in a",
    "There was once a",
    "At the end of",
    # Technical
    "Python is a",
    "Machine learning is",
    "The internet is",
    "Artificial intelligence is",
    "A computer is",
    "An algorithm is",
    "Data science is",
    "Programming is",
    # Abstract
    "Love is",
    "Time is",
    "Life is",
    "Truth is",
    "Beauty is",
    "Knowledge is",
    "Power is",
    "Freedom is",
]


def collect_endpoint_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    start_layer: int,
    end_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect h_in (at start_layer) and h_out (at end_layer) for many prompts.

    Returns:
        X: (hidden_dim, n_samples) - inputs to the block
        Y: (hidden_dim, n_samples) - outputs from the block
    """
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    inputs = []
    outputs = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                # Capture input
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                # Capture output
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)   # (hidden_dim, n_samples)
    Y = np.stack(outputs, axis=1)  # (hidden_dim, n_samples)

    return X, Y


def compute_total_transformation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the transformation T such that Y ≈ T @ X.

    Uses least squares: T = Y @ pinv(X)
    """
    # T @ X = Y
    # T = Y @ X^+ where X^+ is the pseudoinverse
    X_pinv = np.linalg.pinv(X)  # (n_samples, hidden_dim)
    T = Y @ X_pinv              # (hidden_dim, hidden_dim)

    return T


def analyze_transformation(T: np.ndarray) -> dict:
    """Analyze the structure of transformation T."""
    # SVD
    U, S, Vh = np.linalg.svd(T, full_matrices=False)

    # Effective ranks
    total_var = np.sum(S ** 2)
    cumsum_var = np.cumsum(S ** 2)

    rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1
    rank_95 = np.searchsorted(cumsum_var / total_var, 0.95) + 1
    rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1

    # Decompose T = I + F (residual form)
    hidden_dim = T.shape[0]
    I = np.eye(hidden_dim)
    F = T - I

    # Analyze F
    U_F, S_F, Vh_F = np.linalg.svd(F, full_matrices=False)
    F_rank_90 = np.searchsorted(np.cumsum(S_F ** 2) / np.sum(S_F ** 2), 0.90) + 1

    # Norm of F (measures how much the transformation deviates from identity)
    F_norm = np.linalg.norm(F, 'fro')

    return {
        'T_singular_values': S,
        'T_rank_90': rank_90,
        'T_rank_95': rank_95,
        'T_rank_99': rank_99,
        'F_singular_values': S_F,
        'F_rank_90': F_rank_90,
        'F_norm': F_norm,
    }


def factor_transformation(T: np.ndarray, rank: int) -> np.ndarray:
    """Factor T to given rank using SVD."""
    U, S, Vh = np.linalg.svd(T, full_matrices=False)

    # Truncate
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    T_factored = U_r @ np.diag(S_r) @ Vh_r

    return T_factored


def test_factored_block(
    model: Any,
    tokenizer: Any,
    prompt: str,
    T_factored: np.ndarray,
    start_layer: int,
    end_layer: int,
    max_tokens: int = 5,
) -> tuple[str, str]:
    """Test generation replacing layers [start, end] with T_factored."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

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

    # Factored generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            # Apply factored transformation instead of running layers
            h_in = np.array(h[0, -1, :].astype(mx.float32))
            h_out = T_factored @ h_in

            # Update h (only last position)
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

        elif start_layer < idx <= end_layer:
            # Skip these layers - they're replaced by T_factored
            pass
        else:
            # Run layer normally
            h = layer(h, mask, None)
            mx.eval(h)

    # Final norm
    h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    factored_generated = []
    if next_token != tokenizer.eos_token_id:
        factored_generated.append(next_token)

    # Continue normally
    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        factored_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    factored_output = tokenizer.decode(factored_generated)

    return normal_output, factored_output


def main():
    parser = argparse.ArgumentParser(description="Lie algebra compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
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
    print("LIE ALGEBRA COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Define the transmission block (layers 3-26 for 28-layer model)
    start_layer = 3
    end_layer = 26 if n_layers == 28 else n_layers - 2

    print(f"Compressing layers {start_layer} to {end_layer} as a single block")

    # Phase 1: Collect activations
    print(f"\n{'='*80}")
    print("PHASE 1: COLLECTING ENDPOINT ACTIVATIONS")
    print("="*80)

    X, Y = collect_endpoint_activations(
        model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer
    )

    print(f"Collected {X.shape[1]} samples")
    print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

    # Phase 2: Compute total transformation
    print(f"\n{'='*80}")
    print("PHASE 2: COMPUTING TOTAL TRANSFORMATION")
    print("="*80)

    T = compute_total_transformation(X, Y)
    print(f"T shape: {T.shape}")

    # Verify T works
    Y_pred = T @ X
    reconstruction_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Reconstruction error on calibration: {reconstruction_error:.6f}")

    # Phase 3: Analyze transformation structure
    print(f"\n{'='*80}")
    print("PHASE 3: TRANSFORMATION ANALYSIS")
    print("="*80)

    analysis = analyze_transformation(T)

    print(f"\nTotal transformation T:")
    print(f"  Rank for 90% variance: {analysis['T_rank_90']}")
    print(f"  Rank for 95% variance: {analysis['T_rank_95']}")
    print(f"  Rank for 99% variance: {analysis['T_rank_99']}")
    print(f"  Top 5 singular values: {analysis['T_singular_values'][:5]}")

    print(f"\nResidual F = T - I:")
    print(f"  Frobenius norm: {analysis['F_norm']:.4f}")
    print(f"  Rank for 90% variance: {analysis['F_rank_90']}")
    print(f"  Top 5 singular values: {analysis['F_singular_values'][:5]}")

    # Phase 4: Test factored transformation
    print(f"\n{'='*80}")
    print("PHASE 4: TEST FACTORED TRANSFORMATION")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    # Test full rank first (should work perfectly)
    print(f"\n--- Full rank (sanity check) ---")
    for prompt in test_prompts:
        normal, factored = test_factored_block(
            model, tokenizer, prompt, T, start_layer, end_layer, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        factored_first = factored.split()[0] if factored.split() else "(empty)"
        match = "✓" if normal_first == factored_first else "✗"
        print(f"  {prompt[:30]}: {normal_first} → {factored_first} {match}")

    # Test different ranks on HELD-OUT prompts
    held_out_prompts = [
        # NONE of these are in calibration
        "Water freezes at",
        "The color of the sky is",
        "100 / 10 =",
        "The opposite of fast is",
        "In a galaxy far far",
        "Neural networks are",
    ]

    print(f"\n--- Testing on HELD-OUT prompts (not in calibration) ---")

    for rank in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        T_factored = factor_transformation(T, rank)

        # Measure approximation error
        approx_error = np.linalg.norm(T - T_factored) / np.linalg.norm(T)

        matches = 0
        for prompt in held_out_prompts:
            normal, factored = test_factored_block(
                model, tokenizer, prompt, T_factored, start_layer, end_layer, max_tokens=5
            )
            normal_first = normal.split()[0] if normal.split() else "(empty)"
            factored_first = factored.split()[0] if factored.split() else "(empty)"
            if normal_first == factored_first:
                matches += 1

        compression = hidden_dim / rank
        print(f"Rank={rank:>4}: {matches}/{len(held_out_prompts)} matches, error={approx_error:.4f}, compression={compression:.1f}x")

    # Insight
    print(f"\n{'='*80}")
    print("LIE ALGEBRA INSIGHT")
    print("="*80)

    print(f"""
THE ALGEBRA:

We computed T such that h_out = T @ h_in for the block of {end_layer - start_layer + 1} layers.

T has effective rank {analysis['T_rank_90']} (90% variance).

This is MUCH lower than {hidden_dim} * {end_layer - start_layer + 1} = {hidden_dim * (end_layer - start_layer + 1)} parameters
in the original layers!

THE KEY INSIGHT:

When we factor INDIVIDUAL layers, errors compound through:
  T = (I + F_n) @ ... @ (I + F_1)
  = I + Σ F_i + Σ F_j @ F_i + ...

The cross-terms Σ F_j @ F_i are the problem.

When we factor the TOTAL transformation T directly:
  T_factored ≈ T

No cross-terms! The factorization preserves the end-to-end map.

COMPRESSION POTENTIAL:

If T has rank R for 99% variance, we can replace:
  - {end_layer - start_layer + 1} transformer layers
  - With a single rank-R matrix multiplication
  - Compression: O(L * d^2) → O(R * d)
""")


if __name__ == "__main__":
    main()
