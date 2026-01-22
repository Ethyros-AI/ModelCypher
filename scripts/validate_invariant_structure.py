#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Cross-Model Invariant Structure Validation
# Tests Paper 1 hypothesis: CKA ≈ 0.94 across model families

"""
Cross-Model Invariant Structure Validation

Validates the hypothesis from Paper 1 (Invariant Semantic Structure):

Key insight: CKA depends on what you're comparing:
- SHARED MANIFOLD (semantic primes): Aligned CKA = 1.0 (invariant structure)
- FULL REPRESENTATION: CKA < 1.0 (models explore different regions)

A law model and medical model share invariant structure on common concepts,
but each has unique knowledge the other hasn't learned.

Usage:
    poetry run python scripts/validate_invariant_structure.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import detect_default_backend_type, get_backend
from modelcypher.core.domain._backend import set_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import find_alignment

if TYPE_CHECKING:
    pass

# =============================================================================
# SEMANTIC PRIMES (65 items from Wierzbicka 1996 / Paper 1)
# =============================================================================

SEMANTIC_PRIMES = [
    # Substantives (6)
    "I", "you", "someone", "something", "people", "body",
    # Determiners (9)
    "this", "same", "other", "one", "two", "some", "all", "much", "many",
    # Evaluators (4)
    "good", "bad", "big", "small",
    # Descriptors (1)
    "true",
    # Mental Predicates (6)
    "think", "know", "want", "feel", "see", "hear",
    # Speech (2)
    "say", "words",
    # Actions/Events (3)
    "do", "happen", "move",
    # Existence/Possession (3)
    "be", "there", "have",
    # Life/Death (2)
    "live", "die",
    # Time (8)
    "when", "now", "before", "after", "long", "short", "time", "moment",
    # Space (9)
    "where", "here", "above", "below", "far", "near", "side", "inside", "touch",
    # Logical (5)
    "not", "maybe", "can", "because", "if",
    # Intensifier/Similarity (3)
    "very", "more", "like",
]

# =============================================================================
# MODEL PAIRS TO COMPARE
# =============================================================================

MODELS_DIR = Path("/path/to/models/mlx-community")

MODEL_PAIRS = [
    # Within-family comparisons
    {
        "model_a": "Qwen2.5-Coder-0.5B-Instruct-bf16",
        "model_b": "Qwen2.5-3B-Instruct-bf16",
        "type": "within_family",
        "description": "Same family, different sizes",
    },
    {
        "model_a": "Qwen2.5-3B-Instruct-bf16",
        "model_b": "Qwen2.5-Coder-3B-Instruct-bf16",
        "type": "same_base_diff_ft",
        "description": "Same base model, different fine-tuning",
    },
    # Cross-family comparisons (KEY TESTS)
    {
        "model_a": "Qwen2.5-Coder-3B-Instruct-bf16",
        "model_b": "granite-3b-code-instruct-128k-mlx",
        "type": "cross_family",
        "description": "Qwen vs Granite (different families)",
    },
]


def get_prime_embeddings(model, tokenizer, primes: list[str]):
    """Extract embeddings for semantic primes from model's embedding matrix."""
    import mlx.core as mx

    embeddings_list = []
    found_primes = []

    embed_matrix = model.model.embed_tokens.weight  # [vocab_size, hidden_dim]

    for prime in primes:
        # Get token ID for prime (lowercase)
        tokens = tokenizer.encode(prime.lower())
        # Use first token if multi-token
        if len(tokens) > 0:
            token_id = tokens[0]
            if token_id < embed_matrix.shape[0]:
                emb = embed_matrix[token_id]
                embeddings_list.append(emb)
                found_primes.append(prime)

    if len(embeddings_list) == 0:
        return None, []

    embeddings = mx.stack(embeddings_list, axis=0)
    mx.eval(embeddings)
    return embeddings, found_primes


def sample_random_vocabulary(tokenizer_a, tokenizer_b, n_samples: int = 100, seed: int = 42):
    """Sample random tokens from vocabulary intersection."""
    import random
    random.seed(seed)

    # Get vocabulary intersection
    vocab_a = set(range(tokenizer_a.vocab_size if hasattr(tokenizer_a, 'vocab_size') else 50000))
    vocab_b = set(range(tokenizer_b.vocab_size if hasattr(tokenizer_b, 'vocab_size') else 50000))

    # Use smaller vocabulary size
    max_vocab = min(len(vocab_a), len(vocab_b), 30000)  # Cap at 30k common tokens
    common_tokens = list(range(max_vocab))

    # Sample random tokens
    sampled = random.sample(common_tokens, min(n_samples, len(common_tokens)))
    return sampled


def compute_cross_model_cka(model_a_path: Path, model_b_path: Path, backend) -> dict:
    """Load two models and compute CKA on semantic prime embeddings."""
    import mlx.core as mx
    from mlx_lm import load

    print(f"  Loading {model_a_path.name}...")
    model_a, tokenizer_a = load(str(model_a_path))
    hidden_dim_a = model_a.model.embed_tokens.weight.shape[1]

    print(f"  Loading {model_b_path.name}...")
    model_b, tokenizer_b = load(str(model_b_path))
    hidden_dim_b = model_b.model.embed_tokens.weight.shape[1]

    print(f"  Hidden dims: {hidden_dim_a} vs {hidden_dim_b}")

    # Get embedding matrices for later random sampling
    embed_a = model_a.model.embed_tokens.weight
    embed_b = model_b.model.embed_tokens.weight

    # Extract embeddings for semantic primes
    print("  Extracting prime embeddings...")
    emb_a, primes_a = get_prime_embeddings(model_a, tokenizer_a, SEMANTIC_PRIMES)
    emb_b, primes_b = get_prime_embeddings(model_b, tokenizer_b, SEMANTIC_PRIMES)

    if emb_a is None or emb_b is None:
        return {"error": "Failed to extract embeddings"}

    # Find shared primes (intersection)
    shared_primes = set(primes_a) & set(primes_b)
    print(f"  Shared primes: {len(shared_primes)}/{len(SEMANTIC_PRIMES)}")

    if len(shared_primes) < 10:
        return {"error": f"Too few shared primes: {len(shared_primes)}"}

    # Reorder to match shared primes
    shared_list = sorted(shared_primes)
    idx_a = [primes_a.index(p) for p in shared_list]
    idx_b = [primes_b.index(p) for p in shared_list]

    emb_a_shared = mx.take(emb_a, mx.array(idx_a), axis=0)
    emb_b_shared = mx.take(emb_b, mx.array(idx_b), axis=0)
    mx.eval(emb_a_shared, emb_b_shared)

    print(f"  Embeddings A: {emb_a_shared.shape}")
    print(f"  Embeddings B: {emb_b_shared.shape}")

    # Convert to backend arrays
    emb_a_backend = backend.array(emb_a_shared)
    emb_b_backend = backend.array(emb_b_shared)
    backend.eval(emb_a_backend, emb_b_backend)

    # Compute raw CKA (before alignment)
    print("  Computing raw CKA...")
    raw_result = compute_cka(emb_a_backend, emb_b_backend, backend)
    raw_cka = raw_result.cka

    # Compute aligned CKA (after Procrustes alignment)
    print("  Computing aligned CKA (Procrustes)...")
    alignment = find_alignment(emb_a_backend, emb_b_backend, backend)
    aligned_cka = alignment.achieved_cka

    print(f"  [Semantic Primes] Raw CKA: {raw_cka:.4f}  |  Aligned CKA: {aligned_cka:.4f}")

    # Also sample random vocabulary to compare full representation space
    print("  Sampling random vocabulary for comparison...")
    random_tokens = sample_random_vocabulary(tokenizer_a, tokenizer_b, n_samples=200)

    # Get embeddings for random tokens
    emb_a_random = mx.take(embed_a, mx.array(random_tokens), axis=0)
    emb_b_random = mx.take(embed_b, mx.array(random_tokens), axis=0)
    mx.eval(emb_a_random, emb_b_random)

    emb_a_random_backend = backend.array(emb_a_random)
    emb_b_random_backend = backend.array(emb_b_random)
    backend.eval(emb_a_random_backend, emb_b_random_backend)

    # Compute CKA on random sample (approximates full representation)
    random_raw_result = compute_cka(emb_a_random_backend, emb_b_random_backend, backend)
    random_raw_cka = random_raw_result.cka

    random_alignment = find_alignment(emb_a_random_backend, emb_b_random_backend, backend)
    random_aligned_cka = random_alignment.achieved_cka

    print(f"  [Random Vocab]    Raw CKA: {random_raw_cka:.4f}  |  Aligned CKA: {random_aligned_cka:.4f}")

    # Clean up to free memory
    del model_a, model_b, tokenizer_a, tokenizer_b
    del emb_a, emb_b, emb_a_shared, emb_b_shared
    mx.clear_cache()

    return {
        # Semantic primes (shared manifold)
        "primes_raw_cka": raw_cka,
        "primes_aligned_cka": aligned_cka,
        "primes_numerical_deviation": alignment.numerical_deviation,
        "primes_is_perfect": alignment.is_perfect,
        # Random vocabulary (approximates full representation)
        "random_raw_cka": random_raw_cka,
        "random_aligned_cka": random_aligned_cka,
        "random_numerical_deviation": random_alignment.numerical_deviation,
        # Metadata
        "hsic_xy": raw_result.hsic_xy,
        "hsic_xx": raw_result.hsic_xx,
        "hsic_yy": raw_result.hsic_yy,
        "sample_count": raw_result.sample_count,
        "hidden_dim_a": hidden_dim_a,
        "hidden_dim_b": hidden_dim_b,
        "shared_primes": len(shared_primes),
        "random_samples": len(random_tokens),
    }


def main() -> None:
    print("=" * 70)
    print("CROSS-MODEL INVARIANT STRUCTURE VALIDATION")
    print("Testing Paper 1 hypothesis: CKA ≈ 0.94 across model families")
    print("=" * 70)
    print()

    # Initialize backend
    print("[1/3] Initializing backend...")
    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    print(f"      Backend: {backend_type}")
    print()

    # Check models exist
    print("[2/3] Checking models...")
    for pair in MODEL_PAIRS:
        path_a = MODELS_DIR / pair["model_a"]
        path_b = MODELS_DIR / pair["model_b"]
        if not path_a.exists():
            print(f"      WARNING: {pair['model_a']} not found")
        if not path_b.exists():
            print(f"      WARNING: {pair['model_b']} not found")
    print()

    # Run comparisons
    print("[3/3] Running CKA comparisons...")
    print()

    results = []
    within_family_ckas = []
    cross_family_ckas = []

    for i, pair in enumerate(MODEL_PAIRS):
        path_a = MODELS_DIR / pair["model_a"]
        path_b = MODELS_DIR / pair["model_b"]

        print(f"--- Comparison {i+1}/{len(MODEL_PAIRS)} ---")
        print(f"  {pair['model_a']} vs {pair['model_b']}")
        print(f"  Type: {pair['type']}")
        print(f"  Description: {pair['description']}")

        if not path_a.exists() or not path_b.exists():
            print("  SKIPPED: Model not found")
            print()
            continue

        try:
            result = compute_cross_model_cka(path_a, path_b, backend)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                results.append({
                    "model_a": pair["model_a"],
                    "model_b": pair["model_b"],
                    "type": pair["type"],
                    "description": pair["description"],
                    **result,
                })

                # Track aligned CKA on semantic primes (shared manifold)
                primes_aligned = result["primes_aligned_cka"]
                if pair["type"] in ("within_family", "same_base_diff_ft"):
                    within_family_ckas.append(primes_aligned)
                elif pair["type"] == "cross_family":
                    cross_family_ckas.append(primes_aligned)

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    for r in results:
        print(f"  {r['model_a'][:30]:30} vs {r['model_b'][:30]:30}")
        print(f"      Semantic Primes (shared):  Raw={r['primes_raw_cka']:.4f}  Aligned={r['primes_aligned_cka']:.4f}")
        print(f"      Random Vocab (full repr):  Raw={r['random_raw_cka']:.4f}  Aligned={r['random_aligned_cka']:.4f}")
        print()

    print()
    print("--- Aggregate Statistics (Aligned CKA) ---")

    if within_family_ckas:
        mean_within = sum(within_family_ckas) / len(within_family_ckas)
        print(f"  Within-family mean aligned CKA: {mean_within:.4f} (Paper 1 claims: 0.96 ± 0.02)")
    else:
        mean_within = None
        print("  Within-family: No data")

    if cross_family_ckas:
        mean_cross = sum(cross_family_ckas) / len(cross_family_ckas)
        print(f"  Cross-family mean aligned CKA:  {mean_cross:.4f} (Paper 1 claims: 0.94 ± 0.01)")
    else:
        mean_cross = None
        print("  Cross-family: No data")

    print()

    # Validation verdict
    print("--- Paper 1 Validation ---")
    validated = True

    if mean_within is not None:
        if mean_within >= 0.90:
            print(f"  Within-family: PASS (CKA={mean_within:.4f} >= 0.90)")
        else:
            print(f"  Within-family: FAIL (CKA={mean_within:.4f} < 0.90)")
            validated = False

    if mean_cross is not None:
        if mean_cross >= 0.85:
            print(f"  Cross-family:  PASS (CKA={mean_cross:.4f} >= 0.85)")
        else:
            print(f"  Cross-family:  FAIL (CKA={mean_cross:.4f} < 0.85)")
            validated = False

    print()
    if validated and (mean_within or mean_cross):
        print("  VERDICT: Paper 1 hypothesis VALIDATED")
    elif not results:
        print("  VERDICT: INCONCLUSIVE (no results)")
    else:
        print("  VERDICT: Paper 1 hypothesis NOT validated")

    # Save results
    output_dir = Path(__file__).parent.parent / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cross_model_cka.json"

    output_data = {
        "experiment": "cross_model_invariant_structure",
        "date": datetime.now().isoformat(),
        "semantic_primes_count": len(SEMANTIC_PRIMES),
        "results": results,
        "summary": {
            "within_family_mean": mean_within,
            "cross_family_mean": mean_cross,
            "paper_1_validated": validated,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
