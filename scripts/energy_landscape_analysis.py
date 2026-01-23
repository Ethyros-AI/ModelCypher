#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Energy Landscape Analysis
"""
Energy Landscape Analysis

Inspired by LeCun's Energy-Based Models:
- The model defines an energy function E(x, y)
- Low energy = compatible (x, y) pairs
- The data lives on a manifold in this energy landscape

Key insight: Invariant relationships (apple-orange-fruit) ARE
the structure of the energy landscape. Compression should preserve
the TOPOLOGY of valleys, not specific weights.

This script:
1. Measures the energy landscape for semantic relationships
2. Checks if invariant relationships have consistent low energy
3. Explores if the landscape can be represented more compactly

Usage:
    python energy_landscape_analysis.py \
        --model /path/to/model
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


# Semantic relationships that MUST be invariant
INVARIANT_RELATIONS = {
    "is-a": [
        ("apple", "fruit"),
        ("orange", "fruit"),
        ("dog", "animal"),
        ("cat", "animal"),
        ("car", "vehicle"),
    ],
    "has-property": [
        ("fire", "hot"),
        ("ice", "cold"),
        ("sun", "bright"),
        ("night", "dark"),
    ],
    "part-of": [
        ("wheel", "car"),
        ("leaf", "tree"),
        ("finger", "hand"),
    ],
    "opposite": [
        ("hot", "cold"),
        ("big", "small"),
        ("up", "down"),
        ("good", "bad"),
    ],
}


def compute_hidden_state(model: Any, tokenizer: Any, text: str) -> Any:
    """Get final hidden state for text."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
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

    # Return last token's hidden state
    return h[0, -1, :]


def compute_energy(h1: Any, h2: Any) -> float:
    """Compute "energy" between two hidden states.

    Low energy = compatible/related
    High energy = incompatible/unrelated

    Using negative cosine similarity as energy.
    """
    import mlx.core as mx

    h1_norm = h1 / (mx.linalg.norm(h1) + 1e-8)
    h2_norm = h2 / (mx.linalg.norm(h2) + 1e-8)

    cos_sim = float(mx.sum(h1_norm * h2_norm))

    # Energy = -similarity (lower energy for more similar)
    return -cos_sim


def compute_energy_euclidean(h1: Any, h2: Any) -> float:
    """Energy as Euclidean distance."""
    import mlx.core as mx
    return float(mx.linalg.norm(h1 - h2))


def analyze_invariant_energies(model: Any, tokenizer: Any) -> dict:
    """Check if invariant relationships have consistent energy structure."""
    import mlx.core as mx

    results = {}

    for relation_type, pairs in INVARIANT_RELATIONS.items():
        energies = []

        for word1, word2 in pairs:
            h1 = compute_hidden_state(model, tokenizer, word1)
            h2 = compute_hidden_state(model, tokenizer, word2)
            mx.eval(h1)
            mx.eval(h2)

            energy = compute_energy(h1, h2)
            energies.append((word1, word2, energy))

        results[relation_type] = energies

    return results


def analyze_non_relations(model: Any, tokenizer: Any) -> list:
    """Measure energy for non-related word pairs (should be higher)."""
    import mlx.core as mx

    # Random unrelated pairs
    non_pairs = [
        ("apple", "democracy"),
        ("dog", "triangle"),
        ("sun", "algorithm"),
        ("car", "philosophy"),
        ("tree", "integer"),
    ]

    energies = []
    for word1, word2 in non_pairs:
        h1 = compute_hidden_state(model, tokenizer, word1)
        h2 = compute_hidden_state(model, tokenizer, word2)
        mx.eval(h1)
        mx.eval(h2)

        energy = compute_energy(h1, h2)
        energies.append((word1, word2, energy))

    return energies


def analyze_manifold_structure(model: Any, tokenizer: Any) -> dict:
    """Analyze the structure of the semantic manifold.

    If relationships are invariant, related words should cluster
    in the hidden space, forming a low-dimensional manifold.
    """
    import mlx.core as mx

    # Collect hidden states for all words
    all_words = set()
    for pairs in INVARIANT_RELATIONS.values():
        for w1, w2 in pairs:
            all_words.add(w1)
            all_words.add(w2)

    word_to_hidden = {}
    for word in all_words:
        h = compute_hidden_state(model, tokenizer, word)
        mx.eval(h)
        word_to_hidden[word] = np.array(h.astype(mx.float32))

    # Stack into matrix
    words = list(word_to_hidden.keys())
    H = np.stack([word_to_hidden[w] for w in words])  # [n_words, hidden_dim]

    # PCA to find manifold dimensionality
    mean = H.mean(axis=0)
    centered = H - mean
    cov = (centered.T @ centered) / len(H)

    try:
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        total_var = np.sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues) / total_var

        dims_90 = np.searchsorted(cumsum, 0.90) + 1
        dims_95 = np.searchsorted(cumsum, 0.95) + 1
        dims_99 = np.searchsorted(cumsum, 0.99) + 1
    except:
        dims_90 = dims_95 = dims_99 = -1

    return {
        'n_words': len(words),
        'hidden_dim': H.shape[1],
        'manifold_dims_90': dims_90,
        'manifold_dims_95': dims_95,
        'manifold_dims_99': dims_99,
    }


def main():
    parser = argparse.ArgumentParser(description="Energy landscape analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    print("\n" + "=" * 70)
    print("ENERGY LANDSCAPE ANALYSIS")
    print("=" * 70)
    print("Insight: Invariant relationships ARE the energy landscape structure")
    print("Low energy = compatible pairs, High energy = incompatible")
    print()

    # Analyze invariant relations
    print("INVARIANT RELATIONS (should have LOW energy):")
    print("-" * 50)

    invariant_results = analyze_invariant_energies(model, tokenizer)

    for relation_type, energies in invariant_results.items():
        print(f"\n  {relation_type.upper()}:")
        for w1, w2, e in energies:
            print(f"    ({w1}, {w2}): energy = {e:.4f}")

        avg_energy = np.mean([e for _, _, e in energies])
        print(f"    Average: {avg_energy:.4f}")

    # Analyze non-relations
    print("\n" + "-" * 50)
    print("NON-RELATIONS (should have HIGHER energy):")
    print("-" * 50)

    non_results = analyze_non_relations(model, tokenizer)
    for w1, w2, e in non_results:
        print(f"    ({w1}, {w2}): energy = {e:.4f}")

    avg_non = np.mean([e for _, _, e in non_results])
    print(f"    Average: {avg_non:.4f}")

    # Compare
    avg_invariant = np.mean([e for res in invariant_results.values() for _, _, e in res])

    print("\n" + "=" * 70)
    print("ENERGY SEPARATION")
    print("=" * 70)
    print(f"Average invariant energy:     {avg_invariant:.4f}")
    print(f"Average non-relation energy:  {avg_non:.4f}")
    print(f"Separation (non - invariant): {avg_non - avg_invariant:.4f}")

    if avg_non > avg_invariant:
        print("\n✓ Non-relations have HIGHER energy than invariant relations")
        print("  The model has learned the energy landscape!")
    else:
        print("\n✗ Energy structure is not as expected")

    # Manifold analysis
    print("\n" + "=" * 70)
    print("MANIFOLD STRUCTURE")
    print("=" * 70)

    manifold = analyze_manifold_structure(model, tokenizer)
    print(f"Words analyzed: {manifold['n_words']}")
    print(f"Hidden dimension: {manifold['hidden_dim']}")
    print(f"Manifold dims for 90% variance: {manifold['manifold_dims_90']}")
    print(f"Manifold dims for 95% variance: {manifold['manifold_dims_95']}")
    print(f"Manifold dims for 99% variance: {manifold['manifold_dims_99']}")

    print("\n" + "=" * 70)
    print("COMPRESSION IMPLICATION")
    print("=" * 70)
    print(f"If semantic relationships live in ~{manifold['manifold_dims_95']}-dim manifold,")
    print(f"then the model only needs to represent a {manifold['manifold_dims_95']}/{manifold['hidden_dim']}")
    print(f"= {manifold['hidden_dim']/manifold['manifold_dims_95']:.0f}x compressed energy landscape")


if __name__ == "__main__":
    main()
