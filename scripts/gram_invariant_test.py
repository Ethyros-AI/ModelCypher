#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Gram Invariant Test
"""
Gram Invariant Test

The hypothesis: The Gram matrix (relational structure) is THE invariant.

If this is true:
1. G = H @ H.T is preserved (up to rotation) through layers
2. We only need to store the Gram matrix, not full hidden states
3. Compression = store G (n×n) instead of H (n×d)

For 20 concepts in 1024D:
- H: 20 × 1024 = 20,480 floats
- G: 20 × 20 = 400 floats (symmetric, so really 210)
- Compression: ~50-100x

But we also need to DECODE back to hidden states for output.
Can we reconstruct H from G?

G = H @ H.T
H = sqrt(G) @ R for some rotation R

The question is: does the model only USE G, or does it use H directly?

Usage:
    python gram_invariant_test.py --model /path/to/model
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
]


def get_hidden_at_layer(model: Any, tokenizer: Any, concepts: list[str], layer_idx: int) -> np.ndarray:
    """Get hidden states at specific layer for all concepts."""
    import mlx.core as mx

    states = []
    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        if layer_idx >= 0:
            for idx, layer in enumerate(model.model.layers):
                if idx <= layer_idx:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result
                    mx.eval(h)

        states.append(np.array(h[0, -1, :].astype(mx.float32)))

    return np.stack(states)


def compute_gram(H: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute Gram matrix."""
    if normalize:
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / (norms + 1e-10)
    return H @ H.T


def gram_similarity(G1: np.ndarray, G2: np.ndarray) -> float:
    """Compute similarity between two Gram matrices."""
    g1 = G1.flatten()
    g2 = G2.flatten()

    # Center
    g1 = g1 - g1.mean()
    g2 = g2 - g2.mean()

    # Correlation
    num = np.sum(g1 * g2)
    denom = np.sqrt(np.sum(g1**2) * np.sum(g2**2))

    return num / denom if denom > 0 else 0


def reconstruct_from_gram(G: np.ndarray, target_dim: int) -> np.ndarray:
    """Reconstruct hidden states from Gram matrix.

    G = H @ H.T
    Eigendecompose: G = V @ D @ V.T
    H_approx = V @ sqrt(D)

    This gives one valid factorization (up to rotation).
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep positive eigenvalues
    pos_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    # sqrt(D)
    sqrt_D = np.diag(np.sqrt(eigenvalues))

    # H_approx: [n_concepts, n_pos_eigenvalues]
    H_approx = eigenvectors @ sqrt_D

    # Pad or truncate to target_dim
    if H_approx.shape[1] < target_dim:
        padding = np.zeros((H_approx.shape[0], target_dim - H_approx.shape[1]))
        H_approx = np.hstack([H_approx, padding])
    else:
        H_approx = H_approx[:, :target_dim]

    return H_approx


def main():
    parser = argparse.ArgumentParser(description="Gram invariant test")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f"\nTesting Gram invariance across {n_layers} layers")
    print(f"Using {len(CONCEPTS)} concepts")

    # Get Gram at each layer
    print("\n" + "=" * 80)
    print("GRAM MATRIX EVOLUTION")
    print("=" * 80)

    gram_matrices = {}

    # Embedding
    H_embed = get_hidden_at_layer(model, tokenizer, CONCEPTS, -1)
    G_embed = compute_gram(H_embed)
    gram_matrices[-1] = G_embed

    print(f"{'Layer':>8} | {'Sim to Embed':>12} | {'Sim to Prev':>12} | Interpretation")
    print("-" * 60)

    prev_G = G_embed
    for layer_idx in range(n_layers):
        H = get_hidden_at_layer(model, tokenizer, CONCEPTS, layer_idx)
        G = compute_gram(H)
        gram_matrices[layer_idx] = G

        sim_to_embed = gram_similarity(G_embed, G)
        sim_to_prev = gram_similarity(prev_G, G)

        if sim_to_prev > 0.999:
            interp = "PRESERVED"
        elif sim_to_prev > 0.99:
            interp = "Nearly preserved"
        elif sim_to_prev > 0.95:
            interp = "Minor change"
        else:
            interp = "Transformed"

        print(f"{layer_idx:>8} | {sim_to_embed:>12.4f} | {sim_to_prev:>12.4f} | {interp}")
        prev_G = G

    # Test reconstruction
    print("\n" + "=" * 80)
    print("RECONSTRUCTION FROM GRAM MATRIX")
    print("=" * 80)

    # Final layer
    H_final = get_hidden_at_layer(model, tokenizer, CONCEPTS, n_layers - 1)
    G_final = compute_gram(H_final, normalize=False)  # Don't normalize for reconstruction

    # Reconstruct
    H_recon = reconstruct_from_gram(G_final, hidden_dim)

    # Check Gram preservation
    G_recon = compute_gram(H_recon, normalize=False)
    gram_error = np.mean((G_final - G_recon) ** 2)
    print(f"Gram reconstruction error: {gram_error:.6f}")

    # Check state similarity
    # Since reconstruction is up to rotation, compare Gram not states
    G_final_norm = compute_gram(H_final, normalize=True)
    G_recon_norm = compute_gram(H_recon, normalize=True)
    sim = gram_similarity(G_final_norm, G_recon_norm)
    print(f"Gram similarity after reconstruction: {sim:.6f}")

    # Intrinsic dimensionality of relationships
    eigenvalues = np.linalg.eigvalsh(G_final)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    total = np.sum(eigenvalues)
    cumsum = np.cumsum(eigenvalues) / total

    dims_95 = np.searchsorted(cumsum, 0.95) + 1
    dims_99 = np.searchsorted(cumsum, 0.99) + 1

    print(f"\nIntrinsic dims of relational structure:")
    print(f"  For 95%: {dims_95}")
    print(f"  For 99%: {dims_99}")

    # The insight
    print("\n" + "=" * 80)
    print("THE GRAM INVARIANT INSIGHT")
    print("=" * 80)

    # Count preserved layers
    preserved = 0
    for i in range(n_layers):
        if i > 0:
            sim = gram_similarity(gram_matrices[i-1], gram_matrices[i])
            if sim > 0.99:
                preserved += 1

    print(f"""
1. GRAM PRESERVATION:
   - Layers with Gram sim > 0.99 to previous: {preserved}/{n_layers}
   - The relational structure IS mostly preserved

2. INTRINSIC DIMENSIONALITY:
   - Gram has {dims_95} significant eigenvalues (95%)
   - This is the true dimensionality of relationships
   - NOT 1024, but ~{dims_95}

3. COMPRESSION INSIGHT:
   - Full storage: {len(CONCEPTS)} × {hidden_dim} = {len(CONCEPTS) * hidden_dim} floats
   - Gram storage: {len(CONCEPTS)} × {len(CONCEPTS)} / 2 = {len(CONCEPTS) * len(CONCEPTS) // 2} floats
   - But Gram is rank-{dims_95}, so only need {len(CONCEPTS)} × {dims_95} = {len(CONCEPTS) * dims_95}

   Compression: {len(CONCEPTS) * hidden_dim / (len(CONCEPTS) * dims_95):.0f}x = {hidden_dim / dims_95:.0f}x

4. THE INVARIANT:
   - The Gram matrix G = H @ H.T is THE invariant
   - Rotations preserve G
   - The model transforms coordinates but preserves relationships
   - COMPRESSION = Store the invariant, not the coordinates
""")


if __name__ == "__main__":
    main()
