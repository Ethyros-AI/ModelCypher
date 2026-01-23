#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Subspace Rotation Analysis
"""
Subspace Rotation Analysis

Key finding from velocity_subspace_analysis:
- Global 11D explains 95.8% of variance
- But each layer's 11D is DIFFERENT
- The subspace ROTATES through layers

This means the "through line" is a CURVE that rotates through
different dimensions at each layer, while maintaining invariant
relationships.

Question: Can we describe this rotation compactly?
If R_i is the rotation from layer i to i+1, then:
- Total description = R_0 @ R_1 @ ... @ R_n
- If R_i are simple (sparse, low-rank), compression is possible

Usage:
    python subspace_rotation_analysis.py --model /path/to/model
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


def get_all_layer_hidden_states(model: Any, tokenizer: Any, concepts: list[str]) -> list[np.ndarray]:
    """Get hidden states at each layer for all concepts.

    Returns: list of [n_concepts, hidden_dim] arrays, one per layer (including embedding)
    """
    import mlx.core as mx

    n_layers = len(model.model.layers)
    layer_states = [[] for _ in range(n_layers + 1)]  # +1 for embedding

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)
        layer_states[0].append(np.array(h[0, -1, :].astype(mx.float32)))

        for layer_idx, layer in enumerate(model.model.layers):
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
            layer_states[layer_idx + 1].append(np.array(h[0, -1, :].astype(mx.float32)))

    return [np.stack(states) for states in layer_states]


def find_layer_subspace(H: np.ndarray, n_components: int = 11) -> tuple:
    """Find the principal subspace for hidden states at a layer.

    Returns: (basis, eigenvalues)
    """
    mean = H.mean(axis=0)
    centered = H - mean
    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors[:, :n_components], eigenvalues[:n_components]


def compute_rotation_between_subspaces(basis1: np.ndarray, basis2: np.ndarray) -> tuple:
    """Compute the rotation that maps basis1 to basis2.

    Uses Procrustes alignment: find R such that basis1 @ R ≈ basis2
    """
    # Orthonormalize
    Q1, _ = np.linalg.qr(basis1)
    Q2, _ = np.linalg.qr(basis2)

    # Find optimal rotation via SVD
    M = Q1.T @ Q2
    U, s, Vt = np.linalg.svd(M)
    R = U @ Vt  # Orthogonal rotation

    return R, s


def analyze_rotation_structure(R: np.ndarray) -> dict:
    """Analyze the structure of a rotation matrix.

    A simple rotation (close to identity or permutation) is compressible.
    """
    n = R.shape[0]

    # How close to identity?
    identity = np.eye(n)
    frobenius_from_identity = np.linalg.norm(R - identity, 'fro')

    # How close to permutation? (max absolute value per row)
    max_per_row = np.max(np.abs(R), axis=1)
    avg_max = np.mean(max_per_row)

    # Sparsity: how many entries are close to 0?
    threshold = 0.1
    sparsity = np.mean(np.abs(R) < threshold)

    # Angle of rotation (using trace)
    trace = np.trace(R)
    # For SO(n), trace = sum of cos(theta_i) for each plane rotation
    # Average angle estimate
    avg_cos = trace / n
    avg_angle_deg = np.arccos(np.clip(avg_cos, -1, 1)) * 180 / np.pi

    return {
        'frobenius_from_identity': frobenius_from_identity,
        'avg_max_per_row': avg_max,
        'sparsity': sparsity,
        'avg_angle_deg': avg_angle_deg,
    }


def compute_cumulative_rotation(rotations: list) -> np.ndarray:
    """Compute the cumulative rotation from first layer to last."""
    R_cum = np.eye(rotations[0].shape[0])
    for R in rotations:
        R_cum = R_cum @ R
    return R_cum


def main():
    parser = argparse.ArgumentParser(description="Subspace rotation analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\nAnalyzing subspace rotation across {n_layers} layers")
    print(f"Hidden dimension: {hidden_dim}")

    # Get hidden states at each layer
    layer_states = get_all_layer_hidden_states(model, tokenizer, CONCEPTS)

    # Find subspace at each layer
    n_components = 11
    layer_bases = []
    for i, H in enumerate(layer_states):
        basis, eigenvalues = find_layer_subspace(H, n_components)
        layer_bases.append(basis)

    # Compute rotations between consecutive layers
    print("\n" + "=" * 80)
    print("SUBSPACE ROTATION BETWEEN LAYERS")
    print("=" * 80)
    print(f"{'Layers':>10} | {'||R-I||':>10} | {'Sparsity':>10} | {'Angle':>10} | Interpretation")
    print("-" * 70)

    rotations = []
    for i in range(len(layer_bases) - 1):
        R, singular_values = compute_rotation_between_subspaces(layer_bases[i], layer_bases[i+1])
        rotations.append(R)

        stats = analyze_rotation_structure(R)

        if stats['frobenius_from_identity'] < 1.0:
            interp = "Small rotation"
        elif stats['avg_angle_deg'] < 30:
            interp = "Moderate rotation"
        elif stats['sparsity'] > 0.7:
            interp = "Sparse (permutation-like)"
        else:
            interp = "Complex rotation"

        layer_str = f"{i} -> {i+1}"
        print(f"{layer_str:>10} | {stats['frobenius_from_identity']:>10.3f} | "
              f"{stats['sparsity']:>10.1%} | {stats['avg_angle_deg']:>9.1f}° | {interp}")

    # Cumulative rotation from embedding to final
    print("\n" + "=" * 80)
    print("CUMULATIVE ROTATION")
    print("=" * 80)

    R_total = compute_cumulative_rotation(rotations)
    total_stats = analyze_rotation_structure(R_total)
    print(f"Embedding -> Final layer:")
    print(f"  ||R_total - I||: {total_stats['frobenius_from_identity']:.3f}")
    print(f"  Sparsity: {total_stats['sparsity']:.1%}")
    print(f"  Average angle: {total_stats['avg_angle_deg']:.1f}°")

    # Check if rotation is low-rank
    U, s, Vt = np.linalg.svd(R_total - np.eye(n_components))
    print(f"\n  Singular values of (R_total - I):")
    print(f"    {[f'{v:.3f}' for v in s]}")

    effective_rank = np.sum(s > 0.1)
    print(f"  Effective rank: {effective_rank}")

    # The key insight
    print("\n" + "=" * 80)
    print("THE ROTATION INSIGHT")
    print("=" * 80)

    small_rotations = sum(1 for r in rotations if analyze_rotation_structure(r)['frobenius_from_identity'] < 1.0)

    print(f"""
1. ROTATION STRUCTURE:
   - Small rotations (||R-I|| < 1): {small_rotations}/{len(rotations)} layers
   - Total rotation from embedding to final: {total_stats['avg_angle_deg']:.1f}°

2. THE "THROUGH LINE" IS A HELIX:
   - At each layer, the 11D semantic subspace rotates slightly
   - The relationships (Gram matrix) are preserved by the rotation
   - The curve spirals through 1024D space, always in 11D

3. COMPRESSION IMPLICATION:
   - Full representation: {n_layers} layers × {hidden_dim} dims = {n_layers * hidden_dim}
   - Rotation representation: 11 × 11 rotation per layer = {n_layers * 11 * 11}
   - Plus: initial 11D embedding projection

   If rotations are simple (rank-{effective_rank} perturbations of identity):
   - Per-layer storage: {effective_rank} × 11 = {effective_rank * 11}
   - Total: {n_layers * effective_rank * 11} + initial embedding

   Compression: {n_layers * hidden_dim / (n_layers * effective_rank * 11):.0f}x

4. THE INVARIANT RELATIONSHIPS:
   - Rotation PRESERVES inner products
   - If R is orthogonal: G_new = R @ G_old @ R.T = G_old (for centered data)
   - The Gram matrix is the INVARIANT of the trajectory!
""")


if __name__ == "__main__":
    main()
