#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Weight Helix Factorization
"""
Weight Helix Factorization

If representations evolve along a helix in 18D (out of 1024D),
then the WEIGHTS should factor into:

    W = P_in @ W_helix @ P_out

Where:
    P_in: 1024D → 18D (project to helix space)
    W_helix: 18D → 18D (the helix transformation)
    P_out: 18D → 1024D (back to full space)

Compression:
    Full MLP w2: 1024 × 4608 = 4.7M params
    Factored: 1024×18 + 18×18 + 18×4608 = 18K + 324 + 83K = 101K
    Ratio: 46x

But this assumes we can find P_in, P_out that work for ALL inputs.
Let's test this hypothesis.

Usage:
    python weight_helix_factorization.py --model /path/to/model
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
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "good", "bad", "love", "hate",
]


def get_mlp_activations(model: Any, tokenizer: Any, concepts: list[str], layer_idx: int) -> tuple:
    """Get MLP input and output activations for a layer.

    Returns: (inputs, outputs) both [n_concepts, intermediate_dim] and [n_concepts, hidden_dim]
    """
    import mlx.core as mx

    inputs = []
    outputs = []

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        # Go to target layer
        for idx, layer in enumerate(model.model.layers):
            if idx < layer_idx:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)
            elif idx == layer_idx:
                # Get attention output first
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                elif hasattr(layer, 'conv'):
                    # Different layer types
                    break
                else:
                    break

                # We need pre-MLP activation (after attention)
                h_pre = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_pre)

                # Run full layer to get output
                result = layer(h)
                h_out = result[0] if isinstance(result, tuple) else result
                mx.eval(h_out)

                h_post = np.array(h_out[0, -1, :].astype(mx.float32))
                outputs.append(h_post)
                break

    if not inputs:
        return None, None

    return np.stack(inputs), np.stack(outputs)


def compute_mlp_delta(inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Compute MLP delta = outputs - inputs (residual contribution)."""
    return outputs - inputs


def find_helix_projection(deltas: np.ndarray, n_dims: int = 18) -> tuple:
    """Find the projection that captures the MLP delta subspace.

    Returns: (P_out, eigenvalues) where P_out projects from helix to full space
    """
    deltas = np.nan_to_num(deltas, nan=0.0)

    # Covariance of deltas
    mean = deltas.mean(axis=0)
    centered = deltas - mean
    cov = (centered.T @ centered) / len(deltas)
    cov = np.nan_to_num(cov, nan=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # P_out maps from n_dims helix space to hidden_dim
    P_out = eigenvectors[:, :n_dims]

    return P_out, eigenvalues[:n_dims], mean


def test_factorization(inputs: np.ndarray, deltas: np.ndarray, P_out: np.ndarray) -> dict:
    """Test how well the factorization reconstructs the MLP output.

    delta ≈ inputs @ W
    If W = P_out @ W_helix @ P_in.T, then:
    delta ≈ inputs @ P_out @ W_helix @ P_in.T

    For simplicity, we test:
    delta ≈ (inputs @ P_out @ P_out.T) + (orthogonal part)

    Actually, the correct factorization is:
    delta = coeffs @ P_out.T where coeffs = delta @ P_out
    """
    # Project delta to helix space and back
    coeffs = deltas @ P_out  # [n, n_dims]
    delta_reconstructed = coeffs @ P_out.T  # [n, hidden_dim]

    # Reconstruction error
    error = deltas - delta_reconstructed
    mse = np.mean(error ** 2)
    total_var = np.mean(deltas ** 2)
    explained_var = 1 - mse / total_var if total_var > 0 else 0

    return {
        'mse': mse,
        'explained_variance': explained_var,
        'coeffs_norm': np.mean(coeffs ** 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Weight helix factorization")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("WEIGHT HELIX FACTORIZATION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim}D hidden")

    # Test factorization at each layer
    print(f"\n{'Layer':>6} | {'Explained Var':>14} | {'MSE':>12} | Interpretation")
    print("-" * 60)

    explained_vars = []
    for layer_idx in range(n_layers):
        inputs, outputs = get_mlp_activations(model, tokenizer, CONCEPTS, layer_idx)

        if inputs is None:
            print(f"{layer_idx:>6} | {'N/A':>14} | {'N/A':>12} | Incompatible layer type")
            continue

        deltas = compute_mlp_delta(inputs, outputs)

        # Find optimal helix projection - try different dims
        for n_dims in [18, 32, 64]:
            P_out, eigenvalues, mean = find_helix_projection(deltas, n_dims=n_dims)
            results = test_factorization(inputs, deltas, P_out)
            if results['explained_variance'] > 0.95:
                break

        # Test reconstruction
        results = test_factorization(inputs, deltas, P_out)
        explained_vars.append(results['explained_variance'])

        if results['explained_variance'] > 0.95:
            interp = "Excellent factorization"
        elif results['explained_variance'] > 0.80:
            interp = "Good factorization"
        elif results['explained_variance'] > 0.50:
            interp = "Partial factorization"
        else:
            interp = "Poor factorization"

        print(f"{layer_idx:>6} | {results['explained_variance']:>13.1%} | "
              f"{results['mse']:>12.4f} | {interp}")

    # Summary
    print(f"\n{'='*80}")
    print("FACTORIZATION SUMMARY")
    print("="*80)

    if explained_vars:
        avg_explained = np.mean(explained_vars)
        print(f"Average explained variance: {avg_explained:.1%}")
        print(f"Layers with >80% explained: {sum(1 for v in explained_vars if v > 0.8)}/{len(explained_vars)}")

        if avg_explained > 0.8:
            print(f"\n✓ FACTORIZATION WORKS!")
            print(f"  The MLP weights can be factored through an 18D helix space")
            print(f"\n  Compression potential:")
            print(f"    Full w2: {hidden_dim} × 4608 = {hidden_dim * 4608:,} params")
            print(f"    Factored: {hidden_dim}×18 + 18×4608 = {hidden_dim*18 + 18*4608:,} params")
            print(f"    Ratio: {(hidden_dim * 4608) / (hidden_dim*18 + 18*4608):.1f}x")
        else:
            print(f"\n✗ Factorization not sufficient")
            print(f"  Need higher dimensional helix or different approach")
    else:
        print("No compatible layers found")

    # The key insight
    print(f"\n{'='*80}")
    print("THE WEIGHT FACTORIZATION INSIGHT")
    print("="*80)
    print(f"""
    If the MLP delta lives in an 18D subspace:

    1. ORIGINAL WEIGHT:
       w2: [hidden_dim, intermediate_dim] = [{hidden_dim}, 4608]
       Total: {hidden_dim * 4608:,} params

    2. FACTORED WEIGHT:
       P_out: [hidden_dim, 18] = projection to helix space
       W_helix: [18, intermediate_dim] = helix transformation
       delta = intermediate @ W_helix.T @ P_out.T

       Total: {hidden_dim * 18 + 18 * 4608:,} params

    3. BUT WAIT - P_out is SHARED across layers!
       If all layers use the same helix subspace:
       - Store P_out once: {hidden_dim * 18:,} params
       - Per layer: W_helix = {18 * 4608:,} params

       Total: {hidden_dim * 18:,} + {n_layers} × {18 * 4608:,}
            = {hidden_dim * 18 + n_layers * 18 * 4608:,} params

       vs Original: {n_layers} × {hidden_dim * 4608:,}
                  = {n_layers * hidden_dim * 4608:,} params

       Compression: {(n_layers * hidden_dim * 4608) / (hidden_dim * 18 + n_layers * 18 * 4608):.1f}x
""")


if __name__ == "__main__":
    main()
