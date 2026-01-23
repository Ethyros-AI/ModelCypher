#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Relational Structure Analysis
"""
Relational Structure Analysis

The manifold isn't about specific dimensions - it's about the
INVARIANT RELATIONSHIPS between concepts.

Key insight from user:
- The model encodes across ALL dimensions
- But maintains invariant relationships
- The "through line" is the geometric structure, not coordinates
- Models "fall through" the manifold to valleys (some disastrous)

This script:
1. Extract the relational structure (distances/angles between concepts)
2. Track how this structure is preserved through layers
3. Find where the structure is DEFINED vs where it's PRESERVED

Usage:
    python relational_structure.py --model /path/to/model
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


# Semantic concepts with known relationships
CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "big", "small", "huge", "size",
    "good", "bad", "evil", "morality",
    "love", "hate", "anger", "emotion",
    "run", "walk", "move", "motion",
]

# Known invariant relationships (should have similar distances)
INVARIANT_PAIRS = [
    # IS-A relationships (should be similar distance)
    [("apple", "fruit"), ("orange", "fruit"), ("dog", "animal"), ("car", "vehicle")],
    # Opposites (should be similar distance)
    [("hot", "cold"), ("big", "small"), ("good", "bad"), ("love", "hate")],
    # Category members (should be similar distance to each other)
    [("apple", "orange"), ("dog", "cat"), ("car", "truck")],
]


def compute_gram_matrix(H: np.ndarray) -> np.ndarray:
    """Compute Gram matrix (all pairwise inner products)."""
    # Normalize to unit vectors
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-10)
    return H_norm @ H_norm.T


def compute_distance_matrix(H: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix."""
    n = H.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(H[i] - H[j])
    return D


def get_hidden_states_at_layer(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
    layer_idx: int,
) -> np.ndarray:
    """Get hidden states at specific layer."""
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

        h_np = np.array(h[0, -1, :].astype(mx.float32))
        states.append(h_np)

    return np.stack(states, axis=0)


def measure_invariant_preservation(
    D: np.ndarray,
    concepts: list[str],
    invariant_groups: list[list[tuple]],
) -> dict:
    """Measure how well invariant relationships are preserved.

    For each group of pairs that SHOULD have similar distances,
    compute the variance of distances within the group.
    Lower variance = better preservation of invariance.
    """
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    results = []
    for group in invariant_groups:
        distances = []
        for (c1, c2) in group:
            if c1 in concept_to_idx and c2 in concept_to_idx:
                i, j = concept_to_idx[c1], concept_to_idx[c2]
                distances.append(D[i, j])

        if len(distances) >= 2:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            cv = std_dist / mean_dist if mean_dist > 0 else 0  # Coefficient of variation
            results.append({
                'mean': mean_dist,
                'std': std_dist,
                'cv': cv,  # Lower = more invariant
            })

    if results:
        avg_cv = np.mean([r['cv'] for r in results])
        return {'avg_cv': avg_cv, 'groups': results}
    return {'avg_cv': 1.0, 'groups': []}


def compute_gram_similarity(G1: np.ndarray, G2: np.ndarray) -> float:
    """Compute similarity between two Gram matrices (CKA-like)."""
    # Flatten and compute correlation
    g1 = G1.flatten()
    g2 = G2.flatten()

    # Center
    g1 = g1 - g1.mean()
    g2 = g2 - g2.mean()

    # Correlation
    num = np.sum(g1 * g2)
    denom = np.sqrt(np.sum(g1**2) * np.sum(g2**2))

    return num / denom if denom > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Relational structure analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print(f"\nAnalyzing relational structure across {n_layers} layers")
    print(f"Using {len(CONCEPTS)} concepts with {len(INVARIANT_PAIRS)} invariant groups")

    # Get embedding Gram matrix as reference
    H_embed = get_hidden_states_at_layer(model, tokenizer, CONCEPTS, -1)
    G_embed = compute_gram_matrix(H_embed)
    D_embed = compute_distance_matrix(H_embed)

    print("\n" + "=" * 80)
    print("RELATIONAL STRUCTURE EVOLUTION")
    print("=" * 80)
    print(f"{'Layer':>8} | {'Gram Sim':>10} | {'Inv. CV':>10} | {'Interpretation':>30}")
    print("-" * 80)

    # Embedding baseline
    inv_embed = measure_invariant_preservation(D_embed, CONCEPTS, INVARIANT_PAIRS)
    print(f"{'Embed':>8} | {'1.000':>10} | {inv_embed['avg_cv']:>10.4f} | Initial relational structure")

    prev_G = G_embed
    gram_sims = []
    inv_cvs = []

    for layer_idx in range(n_layers):
        H = get_hidden_states_at_layer(model, tokenizer, CONCEPTS, layer_idx)
        G = compute_gram_matrix(H)
        D = compute_distance_matrix(H)

        # Compare to embedding
        gram_sim_to_embed = compute_gram_similarity(G_embed, G)

        # Compare to previous layer
        gram_sim_to_prev = compute_gram_similarity(prev_G, G)

        # Invariant preservation
        inv = measure_invariant_preservation(D, CONCEPTS, INVARIANT_PAIRS)

        gram_sims.append(gram_sim_to_embed)
        inv_cvs.append(inv['avg_cv'])

        # Interpretation
        if gram_sim_to_prev > 0.99:
            interp = "PRESERVES structure"
        elif gram_sim_to_prev > 0.9:
            interp = "Minor transformation"
        elif gram_sim_to_embed > gram_sims[-2] if len(gram_sims) > 1 else False:
            interp = "Refines toward embedding"
        else:
            interp = "TRANSFORMS structure"

        print(f"{layer_idx:>8} | {gram_sim_to_embed:>10.4f} | {inv['avg_cv']:>10.4f} | {interp}")

        prev_G = G

    # Find where structure is DEFINED vs PRESERVED
    print("\n" + "=" * 80)
    print("STRUCTURE ANALYSIS")
    print("=" * 80)

    # Find layers with biggest Gram changes
    gram_changes = [abs(gram_sims[i] - gram_sims[i-1]) if i > 0 else abs(gram_sims[0] - 1.0)
                    for i in range(len(gram_sims))]

    defining_layers = [i for i, change in enumerate(gram_changes) if change > 0.05]
    preserving_layers = [i for i, change in enumerate(gram_changes) if change < 0.01]

    print(f"\nStructure-DEFINING layers (Gram changes >5%): {defining_layers}")
    print(f"Structure-PRESERVING layers (Gram changes <1%): {preserving_layers}")

    # Find layer with best invariant preservation
    best_inv_layer = np.argmin(inv_cvs)
    print(f"\nBest invariant preservation at layer {best_inv_layer} (CV = {inv_cvs[best_inv_layer]:.4f})")

    # The insight
    print("\n" + "=" * 80)
    print("THE THROUGH LINE")
    print("=" * 80)
    print(f"""
The relational structure (Gram matrix) tracks the INVARIANT relationships.

- Embedding defines initial structure (CV = {inv_embed['avg_cv']:.4f})
- Structure-DEFINING layers: {defining_layers}
  (These transform the relational geometry)
- Structure-PRESERVING layers: {preserving_layers}
  (These maintain relationships while processing)

The "through line" is the path through high-D space that:
1. Gets DEFINED at layers {defining_layers}
2. Gets PRESERVED at layers {preserving_layers}
3. Maintains invariant relationships (CV stays low)

Compression = Find minimal parameterization of this path.
""")


if __name__ == "__main__":
    main()
