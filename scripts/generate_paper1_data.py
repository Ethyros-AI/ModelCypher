#!/usr/bin/env python3
"""
Generate test data for Paper 1: Manifold Hypothesis of Agency.

This script extracts semantic prime embeddings and computes Gram matrices
for cross-model comparison experiments.

Usage:
    poetry run python scripts/generate_paper1_data.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_semantic_primes() -> list[str]:
    """Load semantic primes from data file or use defaults."""
    # Default NSM semantic primes (65 core primitives)
    primes = [
        # Substantives
        "I", "you", "someone", "something", "people", "body",
        # Determiners
        "this", "the same", "other", "one", "two", "some", "all", "much", "many",
        # Evaluators
        "good", "bad", "big", "small",
        # Descriptors
        "true",
        # Mental predicates
        "think", "know", "want", "feel", "see", "hear",
        # Speech
        "say", "words",
        # Actions, events, movement
        "do", "happen", "move",
        # Existence & possession
        "be", "there is", "have",
        # Life & death
        "live", "die",
        # Time
        "when", "now", "before", "after", "a long time", "a short time", "for some time", "moment",
        # Space
        "where", "here", "above", "below", "far", "near", "side", "inside", "touch",
        # Logical concepts
        "not", "maybe", "can", "because", "if",
        # Intensifier
        "very", "more",
        # Similarity
        "like",
    ]
    return primes


def load_control_words(n: int = 65) -> list[str]:
    """Load frequency-matched control words."""
    # Common English words matched for frequency
    controls = [
        "table", "chair", "house", "car", "book", "water", "food", "tree",
        "road", "city", "school", "work", "money", "time", "year", "day",
        "night", "morning", "evening", "week", "month", "family", "friend",
        "name", "place", "story", "game", "music", "movie", "art", "sport",
        "health", "life", "world", "country", "state", "group", "company",
        "system", "power", "point", "fact", "part", "case", "way", "number",
        "hand", "room", "face", "door", "window", "wall", "floor", "street",
        "child", "woman", "man", "girl", "boy", "baby", "parent", "student",
        "teacher",
    ]
    return controls[:n]


def extract_embeddings(model_id: str, words: list[str]) -> np.ndarray:
    """Extract token embeddings for given words from a model.
    
    This is a placeholder that returns random embeddings.
    Real implementation would use MLX or transformers to extract actual embeddings.
    """
    print(f"[PLACEHOLDER] Would extract embeddings for {len(words)} words from {model_id}")
    
    # Placeholder: return random embeddings
    # In production, this would call:
    #   from modelcypher.core.domain.geometry.concept_response_matrix import extract_token_embeddings
    embedding_dim = 768  # Typical embedding dimension
    embeddings = np.random.randn(len(words), embedding_dim).astype(np.float32)
    
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


def compute_gram_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute Gram matrix from embeddings."""
    # Mean-center
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    # Compute Gram matrix
    gram = embeddings_centered @ embeddings_centered.T
    
    return gram


def compute_cka(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two Gram matrices.
    
    CKA = <K_tilde, L_tilde>_F / (||K_tilde||_F * ||L_tilde||_F)
    where K_tilde = H @ K @ H (centering matrix H = I - 1/n * 11^T)
    """
    n = gram_a.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    
    K_centered = H @ gram_a @ H
    L_centered = H @ gram_b @ H
    
    # Frobenius inner product
    numerator = np.sum(K_centered * L_centered)
    denominator = np.sqrt(np.sum(K_centered * K_centered) * np.sum(L_centered * L_centered))
    
    if denominator < 1e-10:
        return 0.0
    
    return float(numerator / denominator)


def compute_pearson_upper_triangle(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """Compute Pearson correlation of upper triangle elements."""
    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(gram_a.shape[0], k=1)
    
    a_upper = gram_a[triu_indices]
    b_upper = gram_b[triu_indices]
    
    # Pearson correlation
    a_centered = a_upper - a_upper.mean()
    b_centered = b_upper - b_upper.mean()
    
    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    
    if denominator < 1e-10:
        return 0.0
    
    return float(numerator / denominator)


def generate_null_distribution(
    model_a: str,
    model_b: str,
    control_words: list[str],
    n_samples: int = 200,
    subset_size: int = 65,
) -> list[float]:
    """Generate null distribution by sampling random subsets of control words."""
    pearson_samples = []
    
    for i in range(n_samples):
        # Random subset
        indices = np.random.choice(len(control_words), size=min(subset_size, len(control_words)), replace=False)
        subset = [control_words[j] for j in indices]
        
        # Extract embeddings
        emb_a = extract_embeddings(model_a, subset)
        emb_b = extract_embeddings(model_b, subset)
        
        # Compute Gram matrices
        gram_a = compute_gram_matrix(emb_a)
        gram_b = compute_gram_matrix(emb_b)
        
        # Compute Pearson
        pearson = compute_pearson_upper_triangle(gram_a, gram_b)
        pearson_samples.append(pearson)
    
    return pearson_samples


def main():
    parser = argparse.ArgumentParser(description="Generate Paper 1 test data")
    parser.add_argument(
        "--models",
        type=str,
        default="tinyllama-1.1b,qwen2.5-0.5b,qwen2.5-1.5b",
        help="Comma-separated model IDs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiments/paper1",
        help="Output directory",
    )
    parser.add_argument(
        "--null-samples",
        type=int,
        default=200,
        help="Number of null distribution samples",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [m.strip() for m in args.models.split(",")]
    primes = load_semantic_primes()
    controls = load_control_words(200)  # Larger pool for null sampling
    
    print(f"Models: {models}")
    print(f"Primes: {len(primes)} words")
    print(f"Controls: {len(controls)} words")
    print(f"Output: {output_dir}")
    print()
    
    # Store embeddings and Gram matrices
    embeddings_dict = {}
    gram_dict = {}
    
    for model in models:
        print(f"Processing {model}...")
        
        # Extract prime embeddings
        prime_emb = extract_embeddings(model, primes)
        embeddings_dict[model] = prime_emb.tolist()
        
        # Compute Gram matrix
        gram = compute_gram_matrix(prime_emb)
        gram_dict[model] = gram
        
        # Save Gram matrix
        gram_path = output_dir / "gram_matrices" / f"{model.replace('/', '_')}_primes.npy"
        gram_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gram_path, gram)
        print(f"  Saved Gram matrix to {gram_path}")
    
    # Compute pairwise CKA and Pearson
    results = {
        "models": models,
        "anchor_type": "semantic_primes",
        "n_anchors": len(primes),
        "comparisons": [],
    }
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i >= j:
                continue
            
            gram_a = gram_dict[model_a]
            gram_b = gram_dict[model_b]
            
            cka = compute_cka(gram_a, gram_b)
            pearson = compute_pearson_upper_triangle(gram_a, gram_b)
            
            comparison = {
                "model_a": model_a,
                "model_b": model_b,
                "cka": cka,
                "pearson": pearson,
            }
            results["comparisons"].append(comparison)
            
            print(f"  {model_a} vs {model_b}: CKA={cka:.3f}, Pearson={pearson:.3f}")
    
    # Save results
    results_path = output_dir / "cka_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CKA results to {results_path}")
    
    # Generate null distribution for first pair
    if len(models) >= 2:
        print(f"\nGenerating null distribution ({args.null_samples} samples)...")
        null_samples = generate_null_distribution(
            models[0], models[1], controls, n_samples=args.null_samples
        )
        
        null_stats = {
            "model_a": models[0],
            "model_b": models[1],
            "n_samples": args.null_samples,
            "mean": float(np.mean(null_samples)),
            "std": float(np.std(null_samples)),
            "p5": float(np.percentile(null_samples, 5)),
            "p25": float(np.percentile(null_samples, 25)),
            "p50": float(np.percentile(null_samples, 50)),
            "p75": float(np.percentile(null_samples, 75)),
            "p95": float(np.percentile(null_samples, 95)),
            "samples": null_samples,
        }
        
        null_path = output_dir / "null_distributions" / f"{models[0].replace('/', '_')}_vs_{models[1].replace('/', '_')}.json"
        null_path.parent.mkdir(parents=True, exist_ok=True)
        with open(null_path, "w") as f:
            json.dump(null_stats, f, indent=2)
        print(f"Saved null distribution to {null_path}")
        print(f"  Null mean: {null_stats['mean']:.3f} ± {null_stats['std']:.3f}")
        print(f"  Null 95th percentile: {null_stats['p95']:.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
