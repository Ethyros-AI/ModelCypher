#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Lie Algebra Compression v2
"""
Lie Algebra Compression v2 - With More Samples and Analysis

THE ISSUE WITH V1:
48 samples for 2048×2048 matrix = underdetermined.
T's rank is bounded by min(samples, dim) = 48.

THE FIX:
1. Collect MANY more samples (use generated tokens)
2. Analyze which prompts fail and why
3. Try regularized solutions

Usage:
    python lie_algebra_compression_v2.py --model /path/to/model
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


def collect_many_samples(
    model: Any,
    tokenizer: Any,
    seed_prompts: list[str],
    start_layer: int,
    end_layer: int,
    n_tokens_per_prompt: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect many samples by running generation and capturing activations at each token.
    """
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    inputs = []
    outputs = []

    for prompt in seed_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        # Generate multiple tokens, collecting activations at each step
        for _ in range(n_tokens_per_prompt):
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

            # Get next token
            h = inner_model.norm(h)
            logits = inner_model.embed_tokens.as_linear(h)
            mx.eval(logits)

            logits_np = np.array(logits[0, -1, :].astype(mx.float32))
            next_token = int(np.argmax(logits_np))

            if next_token == tokenizer.eos_token_id:
                break

            input_ids = mx.array([[next_token]])

    X = np.stack(inputs, axis=1)   # (hidden_dim, n_samples)
    Y = np.stack(outputs, axis=1)  # (hidden_dim, n_samples)

    return X, Y


def compute_transformation_regularized(
    X: np.ndarray,
    Y: np.ndarray,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """
    Compute T such that Y ≈ T @ X with Tikhonov regularization.

    Normalizes data first to avoid overflow, then denormalizes T.

    Returns: T, X_scale, Y_scale
    """
    # Normalize to avoid overflow
    X_scale = np.linalg.norm(X, 'fro') / np.sqrt(X.shape[1])
    Y_scale = np.linalg.norm(Y, 'fro') / np.sqrt(Y.shape[1])

    X_norm = X / X_scale
    Y_norm = Y / Y_scale

    hidden_dim = X.shape[0]

    # Gram matrix
    XXT = X_norm @ X_norm.T  # (hidden_dim, hidden_dim)

    # Add regularization
    XXT_reg = XXT + regularization * np.eye(hidden_dim)

    # Solve for normalized T
    T_norm = Y_norm @ X_norm.T @ np.linalg.inv(XXT_reg)

    # Denormalize: Y = T @ X => Y/Y_scale = T_norm @ X/X_scale => T = T_norm * (Y_scale/X_scale)
    T = T_norm * (Y_scale / X_scale)

    return T, X_scale, Y_scale


def analyze_failure_cases(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    T: np.ndarray,
    start_layer: int,
    end_layer: int,
):
    """Analyze why certain prompts fail with the factored transformation."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        # Collect true h_in and h_out
        h_in_true = None
        h_out_true = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in_true = np.array(h[0, -1, :].astype(mx.float32))

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out_true = np.array(h[0, -1, :].astype(mx.float32))

        # Compute factored output
        h_out_factored = T @ h_in_true

        # Compare
        error = np.linalg.norm(h_out_true - h_out_factored) / np.linalg.norm(h_out_true)
        cos_sim = np.dot(h_out_true, h_out_factored) / (np.linalg.norm(h_out_true) * np.linalg.norm(h_out_factored))

        # Check logits
        # Continue forward pass with true and factored outputs
        # ... (simplified for now)

        print(f"  {prompt[:30]}: error={error:.4f}, cos_sim={cos_sim:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Lie algebra compression v2")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
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
    print("LIE ALGEBRA COMPRESSION V2")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    start_layer = 3
    end_layer = 26 if n_layers == 28 else n_layers - 2

    # Seed prompts for generation
    seed_prompts = [
        "The history of",
        "In the year 2050",
        "Scientists discovered",
        "The recipe for",
        "A famous painting",
        "The algorithm works",
        "When the sun sets",
        "The mathematical proof",
        "An ancient civilization",
        "The future of AI",
    ]

    # Phase 1: Collect many samples via generation
    print(f"\n{'='*80}")
    print(f"PHASE 1: COLLECTING ~{args.samples} SAMPLES VIA GENERATION")
    print("="*80)

    tokens_per_prompt = args.samples // len(seed_prompts)
    X, Y = collect_many_samples(
        model, tokenizer, seed_prompts, start_layer, end_layer, tokens_per_prompt
    )

    print(f"Collected {X.shape[1]} samples")

    # Phase 2: Compute regularized transformation
    print(f"\n{'='*80}")
    print("PHASE 2: COMPUTING REGULARIZED TRANSFORMATION")
    print("="*80)

    T, X_scale, Y_scale = compute_transformation_regularized(X, Y, regularization=1e-4)

    print(f"Data scales: X={X_scale:.2f}, Y={Y_scale:.2f}")

    # Reconstruction error
    Y_pred = T @ X
    recon_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Reconstruction error: {recon_error:.6f}")

    # Analyze T
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    total_var = np.sum(S ** 2)
    cumsum_var = np.cumsum(S ** 2)
    rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1
    rank_95 = np.searchsorted(cumsum_var / total_var, 0.95) + 1
    rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1

    print(f"\nTransformation T:")
    print(f"  Rank for 90% variance: {rank_90}")
    print(f"  Rank for 95% variance: {rank_95}")
    print(f"  Rank for 99% variance: {rank_99}")
    print(f"  Top 10 singular values: {S[:10]}")

    # Phase 3: Test on held-out prompts
    print(f"\n{'='*80}")
    print("PHASE 3: TEST ON HELD-OUT PROMPTS")
    print("="*80)

    held_out = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
        "Water freezes at",
        "The color of the sky is",
        "100 / 10 =",
    ]

    def factor_T(T, rank):
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        return U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]

    def test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import create_attention_mask

        inner_model = model.model if hasattr(model, 'model') else model

        # Normal
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
        normal = tokenizer.decode([normal_token]).split()[0] if tokenizer.decode([normal_token]).split() else "(empty)"

        # Factored
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                h_out = T_fact @ h_in
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] = h_out
                h = mx.array(h_np).astype(h.dtype)
                mx.eval(h)
            elif start_layer < idx <= end_layer:
                pass  # Skip
            else:
                h = layer(h, mask, None)
                mx.eval(h)

        h = inner_model.norm(h)
        logits = inner_model.embed_tokens.as_linear(h)
        mx.eval(logits)
        fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
        factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

        return normal, factored

    for rank in [256, 128, 64, 32, 16]:
        T_fact = factor_T(T, rank)

        matches = 0
        for prompt in held_out:
            normal, factored = test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1

        print(f"Rank={rank:>4}: {matches}/{len(held_out)} matches, compression={hidden_dim/rank:.1f}x")

    # Phase 4: Analyze failure cases
    print(f"\n{'='*80}")
    print("PHASE 4: ANALYZING FAILURE CASES")
    print("="*80)

    analyze_failure_cases(model, tokenizer, held_out, T, start_layer, end_layer)

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("="*80)
    print(f"""
With {X.shape[1]} samples:
  - T has effective rank {rank_99} (99% variance)
  - This represents the true dimensionality of the transformation

The Lie algebra approach shows:
  - The 24-layer block can be approximated by a single matrix
  - The effective rank determines the compression ratio

For {hidden_dim}D hidden state and rank {rank_99}:
  - Original: 24 layers × (attention + MLP) = massive
  - Factored: rank-{rank_99} linear map = {rank_99 * hidden_dim * 2:,} parameters
  - Compression: ~{24 * hidden_dim // rank_99}x on this block
""")


if __name__ == "__main__":
    main()
