#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Jacobian Analysis: What IS the Transformation?
"""
INSIGHT: We've been fitting T to sample pairs (X, Y).
But T should capture the TRUE transformation, not an approximation.

The TRUE transformation is defined by the layer weights.
At each point h, the Jacobian J = ∂f/∂h tells us the LOCAL linear transformation.

If J is CONSTANT across the manifold → f is globally linear → T = J (exact!)
If J varies → f is nonlinear → need to understand HOW it varies

This script computes the Jacobian at multiple points and asks:
1. Is J constant (within numerical precision)?
2. If not, what's the structure of its variation?
3. Can we derive T analytically from the layer weights?

Usage:
    python qwen3_jacobian_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_jacobian_finite_diff(model, tokenizer, prompt: str,
                                  layer_idx: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute Jacobian ∂h_out/∂h_in for a single layer using finite differences.

    This tells us: how does a small change in input affect the output?
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    # Forward to get h_in at layer_idx
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == layer_idx:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            break
        h = layer(h, mask, None)
        mx.eval(h)

    # Compute Jacobian column by column using finite differences
    # J[:, i] = (f(h + epsilon * e_i) - f(h - epsilon * e_i)) / (2 * epsilon)

    J = np.zeros((hidden_dim, hidden_dim), dtype=np.float64)

    # Only compute a subset of columns to save time (sample 100 dimensions)
    sample_dims = np.random.choice(hidden_dim, min(100, hidden_dim), replace=False)

    for i in sample_dims:
        # Perturb in direction i
        h_plus = h_in.copy()
        h_plus[i] += epsilon

        h_minus = h_in.copy()
        h_minus[i] -= epsilon

        # Forward pass with perturbed input
        for sign, h_perturbed in [(1, h_plus), (-1, h_minus)]:
            input_ids = mx.array([tokens])
            h = inner_model.embed_tokens(input_ids)
            mx.eval(h)
            mask = create_attention_mask(h, None)

            for idx, layer in enumerate(inner_model.layers):
                if idx == layer_idx:
                    # Replace h with perturbed version
                    h_np = np.array(h.astype(mx.float32))
                    h_np[0, -1, :] = h_perturbed.astype(np.float32)
                    h = mx.array(h_np).astype(h.dtype)
                    mx.eval(h)

                    # Apply the layer
                    h = layer(h, mask, None)
                    mx.eval(h)

                    if sign == 1:
                        h_out_plus = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                    else:
                        h_out_minus = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                    break
                h = layer(h, mask, None)
                mx.eval(h)

        # Finite difference approximation of Jacobian column
        J[:, i] = (h_out_plus - h_out_minus) / (2 * epsilon)

    return J, sample_dims


def analyze_jacobian_structure(J: np.ndarray) -> dict:
    """Analyze the structure of a Jacobian matrix."""
    # SVD
    U, S, Vt = np.linalg.svd(J, full_matrices=False)

    # Effective rank (singular values > 1% of max)
    tol = 0.01 * S[0] if len(S) > 0 else 1e-10
    rank = np.sum(S > tol)

    # Condition number
    cond = S[0] / (S[-1] + 1e-10) if len(S) > 0 else np.inf

    # Check if close to identity
    I = np.eye(J.shape[0], J.shape[1])
    I_error = np.linalg.norm(J - I[:J.shape[0], :J.shape[1]]) / np.linalg.norm(I[:J.shape[0], :J.shape[1]])

    # Check if low-rank: can we approximate with top-k singular values?
    for k in [1, 5, 10, 50]:
        if k < len(S):
            J_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            approx_error = np.linalg.norm(J - J_approx) / np.linalg.norm(J)
        else:
            approx_error = 0

    return {
        'rank': rank,
        'condition': cond,
        'identity_error': I_error,
        'singular_values': S[:20],
        'top_sv_ratio': S[0] / (S[1] + 1e-10) if len(S) > 1 else np.inf
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*70}")
    print("JACOBIAN ANALYSIS")
    print("What IS the transformation?")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Analyzing layer: {args.layer}")

    # Test prompts - diverse to see if Jacobian varies
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "What is the meaning of",
        "1 + 1 =",
        "Once upon a time",
        "The speed of light is",
        "Actually, I think",
        "SELECT * FROM users WHERE",
    ]

    print(f"\nComputing Jacobians at {len(test_prompts)} different points...")
    print("(This tells us: is the transformation constant or input-dependent?)")

    jacobians = []
    analyses = []

    for prompt in test_prompts:
        print(f"\n  '{prompt[:30]}...'")
        J, sample_dims = compute_jacobian_finite_diff(model, tokenizer, prompt, args.layer)

        # Only analyze the sampled submatrix
        J_sub = J[np.ix_(sample_dims, sample_dims)]
        analysis = analyze_jacobian_structure(J_sub)

        jacobians.append(J_sub)
        analyses.append(analysis)

        print(f"    Rank: {analysis['rank']}")
        print(f"    Identity error: {analysis['identity_error']:.4f}")
        print(f"    Top SV ratio: {analysis['top_sv_ratio']:.2f}")

    # Compare Jacobians across prompts
    print(f"\n{'='*70}")
    print("JACOBIAN VARIATION ANALYSIS")
    print("="*70)

    # Compute pairwise differences
    J_mean = np.mean(jacobians, axis=0)

    print(f"\nMean Jacobian analysis:")
    mean_analysis = analyze_jacobian_structure(J_mean)
    print(f"  Rank: {mean_analysis['rank']}")
    print(f"  Identity error: {mean_analysis['identity_error']:.4f}")
    print(f"  Condition: {mean_analysis['condition']:.2e}")

    print(f"\nVariation from mean:")
    for i, (prompt, J) in enumerate(zip(test_prompts, jacobians)):
        variation = np.linalg.norm(J - J_mean) / np.linalg.norm(J_mean)
        print(f"  '{prompt[:25]}...': {variation*100:.2f}% deviation")

    # Key question: Is the Jacobian constant?
    variations = [np.linalg.norm(J - J_mean) / np.linalg.norm(J_mean) for J in jacobians]
    avg_variation = np.mean(variations)
    max_variation = np.max(variations)

    print(f"\n{'='*70}")
    print("KEY FINDING")
    print("="*70)

    if max_variation < 0.01:
        print(f"""
The Jacobian is CONSTANT (max variation: {max_variation*100:.2f}%)!

This means: The transformation IS linear on the manifold.
T = J (the Jacobian) is the EXACT transformation.
No sampling needed - just compute J from the weights.
""")
    elif max_variation < 0.10:
        print(f"""
The Jacobian varies SLIGHTLY (max variation: {max_variation*100:.2f}%)

This means: The transformation is APPROXIMATELY linear.
T = mean(J) is a good approximation.
The ~{max_variation*100:.0f}% variation explains our empirical error.
""")
    else:
        print(f"""
The Jacobian varies SIGNIFICANTLY (max variation: {max_variation*100:.2f}%)

This means: The transformation is NONLINEAR.
A single linear T cannot capture it exactly.
We need either:
  1. Multiple T's for different regions (MoE-style)
  2. A nonlinear approximation
  3. Or to accept approximation error
""")

    # Check: is the Jacobian close to identity + low-rank?
    print(f"\n{'='*70}")
    print("STRUCTURE OF THE JACOBIAN")
    print("="*70)

    # Decompose: J = I + R (identity plus residual)
    I_sub = np.eye(len(sample_dims))
    R_mean = J_mean - I_sub

    U_R, S_R, Vt_R = np.linalg.svd(R_mean, full_matrices=False)

    print(f"\nDecomposition: J = I + R")
    print(f"Residual R analysis:")
    print(f"  ||R|| / ||I||: {np.linalg.norm(R_mean) / np.linalg.norm(I_sub):.4f}")
    print(f"  Rank of R: {np.sum(S_R > 0.01 * S_R[0])}")
    print(f"  Top singular values of R: {S_R[:10]}")

    # How much of R is captured by low-rank approximation?
    for k in [1, 5, 10, 20]:
        if k < len(S_R):
            R_approx = U_R[:, :k] @ np.diag(S_R[:k]) @ Vt_R[:k, :]
            approx_error = np.linalg.norm(R_mean - R_approx) / np.linalg.norm(R_mean)
            print(f"  Rank-{k} approximation error: {approx_error*100:.1f}%")

    print(f"""
INSIGHT:

If J ≈ I + low-rank, the layer makes small, structured adjustments.
This is characteristic of the "transmission" layers.

The transformation T = I + UV' where U, V are derived from the
layer weights would be the EXACT closed-form solution.

No sampling. No calibration. Just linear algebra on weights.
""")


if __name__ == "__main__":
    main()
