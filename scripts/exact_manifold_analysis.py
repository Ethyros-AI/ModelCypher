#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Exact Manifold Analysis
"""
Exact Manifold Analysis

Find the EXACT semantic manifold, not a heuristic approximation.

Questions to answer:
1. What is the intrinsic dimensionality? (Not PCA 95%)
2. Are the 11 MLP active dims THE SAME as the semantic manifold?
3. Is this consistent across many more concepts?

Methods:
- Use MANY more semantic concepts
- Compute intrinsic dimensionality (participation ratio, etc.)
- Check alignment between MLP active dims and semantic PCA

Usage:
    python exact_manifold_analysis.py --model /path/to/model
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


# Extensive semantic concepts across categories
SEMANTIC_CONCEPTS = {
    "animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
                "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit"],
    "fruits": ["apple", "orange", "banana", "grape", "lemon", "peach", "pear",
               "mango", "cherry", "plum", "melon", "berry"],
    "objects": ["car", "house", "table", "chair", "book", "phone", "computer",
                "door", "window", "clock", "lamp", "bed", "cup", "plate"],
    "actions": ["run", "walk", "jump", "swim", "fly", "eat", "drink", "sleep",
                "think", "speak", "write", "read", "work", "play"],
    "properties": ["big", "small", "hot", "cold", "fast", "slow", "old", "new",
                   "good", "bad", "bright", "dark", "soft", "hard"],
    "abstracts": ["love", "hate", "fear", "hope", "truth", "beauty", "freedom",
                  "justice", "power", "peace", "time", "space"],
    "numbers": ["one", "two", "three", "four", "five", "ten", "hundred"],
    "relations": ["above", "below", "inside", "outside", "before", "after",
                  "with", "without", "between", "around"],
}

# The 11 active dimensions from MLP delta analysis
MLP_ACTIVE_DIMS = [98, 126, 188, 249, 374, 391, 457, 462, 827, 896, 1009]


def get_all_concepts() -> list[str]:
    """Get all semantic concepts."""
    all_concepts = []
    for concepts in SEMANTIC_CONCEPTS.values():
        all_concepts.extend(concepts)
    return all_concepts


def compute_hidden_states(model: Any, tokenizer: Any, words: list[str]) -> np.ndarray:
    """Compute hidden states for all words."""
    import mlx.core as mx

    states = []
    for word in words:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for layer in model.model.layers:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

        if hasattr(model.model, 'embedding_norm'):
            h = model.model.embedding_norm(h)
            mx.eval(h)

        h_np = np.array(h[0, -1, :].astype(mx.float32))
        states.append(h_np)

    return np.stack(states, axis=0)


def compute_intrinsic_dimensionality(H: np.ndarray) -> dict:
    """Compute intrinsic dimensionality using multiple methods.

    1. Participation ratio: sum(λ)² / sum(λ²)
    2. Effective rank: exp(entropy of normalized eigenvalues)
    3. Exact count: number of non-zero eigenvalues
    """
    # Center
    mean = H.mean(axis=0)
    centered = H - mean

    # Covariance
    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

    # Participation ratio
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    # Effective rank (exponential of entropy)
    normalized = eigenvalues / np.sum(eigenvalues)
    normalized = normalized[normalized > 1e-15]  # Avoid log(0)
    entropy = -np.sum(normalized * np.log(normalized))
    effective_rank = np.exp(entropy)

    # Count eigenvalues > 1% of max
    threshold = eigenvalues[0] * 0.01
    significant_count = np.sum(eigenvalues > threshold)

    # Count eigenvalues > 0.1% of total variance
    total_var = np.sum(eigenvalues)
    var_threshold = total_var * 0.001
    significant_by_var = np.sum(eigenvalues > var_threshold)

    return {
        'participation_ratio': participation_ratio,
        'effective_rank': effective_rank,
        'significant_eigenvalues': significant_count,
        'significant_by_variance': significant_by_var,
        'total_eigenvalues': len(eigenvalues),
        'top_eigenvalues': eigenvalues[:20].tolist(),
    }


def compute_pca_directions(H: np.ndarray, n_components: int) -> np.ndarray:
    """Get top-n PCA directions."""
    mean = H.mean(axis=0)
    centered = H - mean

    cov = (centered.T @ centered) / len(H)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]

    return eigenvectors[:, idx[:n_components]]


def check_alignment_with_mlp_dims(H: np.ndarray, mlp_dims: list[int]) -> dict:
    """Check if semantic PCA aligns with MLP active dimensions.

    If they're the same manifold:
    - Projecting H to MLP dims should capture most variance
    - PCA directions should have high weights on MLP dims
    """
    # Energy in MLP dims
    total_energy = np.sum(H ** 2)
    mlp_energy = np.sum(H[:, mlp_dims] ** 2)
    mlp_energy_pct = mlp_energy / total_energy * 100

    # PCA directions
    P = compute_pca_directions(H, len(mlp_dims))  # Top-k PCA vectors

    # For each PCA direction, what fraction of its weight is on MLP dims?
    alignment_per_pc = []
    for i in range(P.shape[1]):
        pc = P[:, i]
        total_weight = np.sum(pc ** 2)
        mlp_weight = np.sum(pc[mlp_dims] ** 2)
        alignment_per_pc.append(mlp_weight / total_weight * 100)

    # Which specific dimensions dominate each PC?
    dominant_dims_per_pc = []
    for i in range(min(P.shape[1], 5)):  # Top 5 PCs
        pc = P[:, i]
        top_dims = np.argsort(np.abs(pc))[::-1][:5]
        dominant_dims_per_pc.append(top_dims.tolist())

    return {
        'mlp_energy_pct': mlp_energy_pct,
        'alignment_per_pc': alignment_per_pc,
        'dominant_dims_per_pc': dominant_dims_per_pc,
    }


def main():
    parser = argparse.ArgumentParser(description="Exact manifold analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    # Get all concepts
    concepts = get_all_concepts()
    print(f"\nAnalyzing {len(concepts)} semantic concepts across {len(SEMANTIC_CONCEPTS)} categories")

    # Compute hidden states
    logger.info("Computing hidden states...")
    H = compute_hidden_states(model, tokenizer, concepts)
    print(f"Hidden state matrix: {H.shape}")

    # Intrinsic dimensionality
    print("\n" + "=" * 70)
    print("INTRINSIC DIMENSIONALITY")
    print("=" * 70)

    intrinsic = compute_intrinsic_dimensionality(H)
    print(f"Participation ratio:     {intrinsic['participation_ratio']:.1f}")
    print(f"Effective rank:          {intrinsic['effective_rank']:.1f}")
    print(f"Significant eigenvalues: {intrinsic['significant_eigenvalues']}")
    print(f"Significant by variance: {intrinsic['significant_by_variance']}")
    print(f"Total non-zero:          {intrinsic['total_eigenvalues']}")

    # The intrinsic dim is roughly the participation ratio or effective rank
    estimated_dim = int(round(intrinsic['participation_ratio']))
    print(f"\n→ Estimated intrinsic dimension: ~{estimated_dim}")

    # Alignment with MLP active dims
    print("\n" + "=" * 70)
    print(f"ALIGNMENT WITH MLP ACTIVE DIMS {MLP_ACTIVE_DIMS}")
    print("=" * 70)

    alignment = check_alignment_with_mlp_dims(H, MLP_ACTIVE_DIMS)
    print(f"Energy in MLP {len(MLP_ACTIVE_DIMS)} dims: {alignment['mlp_energy_pct']:.1f}%")
    print(f"\nPCA-MLP alignment per component:")
    for i, pct in enumerate(alignment['alignment_per_pc'][:10]):
        print(f"  PC{i}: {pct:.1f}% of weight on MLP dims")

    print(f"\nDominant dimensions per top PC:")
    for i, dims in enumerate(alignment['dominant_dims_per_pc']):
        mlp_overlap = [d for d in dims if d in MLP_ACTIVE_DIMS]
        print(f"  PC{i}: {dims} (MLP overlap: {mlp_overlap})")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if alignment['mlp_energy_pct'] > 50:
        print(f"✓ MLP active dims capture {alignment['mlp_energy_pct']:.1f}% of semantic energy")
        print("  The MLP dims ARE the semantic manifold!")
    else:
        print(f"✗ MLP active dims only capture {alignment['mlp_energy_pct']:.1f}%")
        print("  The semantic manifold uses different dimensions")

    print(f"\nIntrinsic dimension: {estimated_dim}")
    print(f"MLP active dims: {len(MLP_ACTIVE_DIMS)}")

    if abs(estimated_dim - len(MLP_ACTIVE_DIMS)) <= 3:
        print(f"\n→ MATCH: Intrinsic dim ({estimated_dim}) ≈ MLP dims ({len(MLP_ACTIVE_DIMS)})")
    else:
        print(f"\n→ MISMATCH: Intrinsic dim ({estimated_dim}) ≠ MLP dims ({len(MLP_ACTIVE_DIMS)})")


if __name__ == "__main__":
    main()
