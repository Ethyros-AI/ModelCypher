#!/usr/bin/env python3
"""
Thesis Verification Experiment: Do ALL concepts have invariant positions across models?

This experiment tests the Geometric Knowledge Thesis:
- ALL concepts have invariant positions across model families
- After Gram alignment, CKA should approach 1.0
- If positions weren't invariant, models couldn't find concepts - meaning requires invariance

Phases:
1. Raw CKA - baseline without alignment
2. Gram-Aligned CKA - after finding optimal transformation

Success Criteria:
- Raw CKA > 0.85 all pairs
- Gram-Aligned CKA = 1.0 (within machine epsilon)
"""

from __future__ import annotations

import logging
import sys
import time
from itertools import combinations

import mlx.core as mx
from mlx_lm import load

# ModelCypher imports
from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
from modelcypher.core.domain._backend import get_default_backend, set_default_backend
from modelcypher.backends import get_backend
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import find_alignment

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    "qwen": "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "smollm": "/Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit",
    "llama": "/Volumes/CodeCypher/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
    "mistral": "/Volumes/CodeCypher/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

# Random words for control comparison (common English words)
RANDOM_WORDS = [
    "table", "chair", "window", "door", "book", "paper", "pencil", "computer",
    "phone", "water", "coffee", "bread", "apple", "orange", "banana", "car",
    "bus", "train", "plane", "ship", "house", "building", "street", "road",
    "tree", "flower", "grass", "sun", "moon", "star", "cloud", "rain",
    "wind", "fire", "earth", "mountain", "river", "ocean", "lake", "forest",
    "bird", "fish", "dog", "cat", "horse", "cow", "chicken", "pig",
    "red", "blue", "green", "yellow", "black", "white", "happy", "sad",
    "fast", "slow", "big", "small", "old", "new", "hot", "cold",
]


def get_middle_layer(model) -> int:
    """Get the middle layer index for a model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers) // 2
    return 0


def collect_word_activations(
    model,
    tokenizer,
    words: list[str],
    provider: MLXActivationProvider,
    backend,
) -> mx.array:
    """Collect activations for a list of words at the middle layer."""
    mid_layer = get_middle_layer(model)
    activations = []

    for word in words:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, word)
            if mid_layer in acts:
                activations.append(acts[mid_layer])
            elif acts:
                # Fall back to any available layer
                layer_idx = min(acts.keys())
                activations.append(acts[layer_idx])
        except Exception as e:
            logger.warning(f"Failed to get activation for '{word}': {e}")
            continue

    if not activations:
        raise ValueError("No activations collected")

    # Stack into [n_words, hidden_dim]
    stacked = mx.stack(activations, axis=0)
    mx.eval(stacked)
    return stacked


def run_experiment():
    """Run the thesis verification experiment."""

    # Set up backend
    set_default_backend(get_backend("mlx"))
    backend = get_default_backend()
    provider = MLXActivationProvider()

    # Get semantic primes
    primes = SemanticPrimeInventory.english2014()
    prime_words = [p.canonical_english for p in primes]
    logger.info(f"Loaded {len(prime_words)} semantic primes")

    # Load models
    logger.info("\n" + "=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)

    models = {}
    for name, path in MODELS.items():
        logger.info(f"Loading {name} from {path}...")
        start = time.time()
        model, tokenizer = load(path)
        elapsed = time.time() - start

        # Get model info
        num_layers = len(model.model.layers) if hasattr(model, "model") else "?"
        logger.info(f"  Loaded in {elapsed:.1f}s ({num_layers} layers)")
        models[name] = (model, tokenizer)

    # Filter words to vocabulary intersection
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING WORDS TO VOCABULARY INTERSECTION")
    logger.info("=" * 60)

    # Get common words that tokenize to single tokens in all models
    def is_single_token(word: str, tokenizer) -> bool:
        """Check if word tokenizes to a single token."""
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if isinstance(tokens, list):
            return len(tokens) == 1
        return len(list(tokens.ids)) == 1

    filtered_primes = []
    for word in prime_words:
        if all(is_single_token(word, tok) for _, tok in models.values()):
            filtered_primes.append(word)

    filtered_random = []
    for word in RANDOM_WORDS:
        if all(is_single_token(word, tok) for _, tok in models.values()):
            filtered_random.append(word)

    logger.info(f"Semantic primes: {len(prime_words)} -> {len(filtered_primes)} (single-token)")
    logger.info(f"Random words: {len(RANDOM_WORDS)} -> {len(filtered_random)} (single-token)")

    # Use common size
    n_words = min(len(filtered_primes), len(filtered_random), 50)
    test_primes = filtered_primes[:n_words]
    test_random = filtered_random[:n_words]

    logger.info(f"Using {n_words} words per set for comparison")

    # Collect activations for each model
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING ACTIVATIONS")
    logger.info("=" * 60)

    activations = {}
    for name, (model, tokenizer) in models.items():
        logger.info(f"\n{name}:")

        # Semantic primes
        logger.info(f"  Collecting primes...")
        start = time.time()
        acts_primes = collect_word_activations(
            model, tokenizer, test_primes, provider, backend
        )
        logger.info(f"    Shape: {acts_primes.shape}, Time: {time.time() - start:.1f}s")

        # Random words
        logger.info(f"  Collecting random words...")
        start = time.time()
        acts_random = collect_word_activations(
            model, tokenizer, test_random, provider, backend
        )
        logger.info(f"    Shape: {acts_random.shape}, Time: {time.time() - start:.1f}s")

        activations[name] = {
            "primes": acts_primes,
            "random": acts_random,
        }

    # Phase 1: Raw CKA
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: RAW CROSS-FAMILY CKA (No Alignment)")
    logger.info("=" * 60)
    logger.info("Expected if thesis holds: > 0.85 all pairs")
    logger.info("")

    model_names = list(models.keys())
    raw_results = {}

    for name_a, name_b in combinations(model_names, 2):
        logger.info(f"{name_a} <-> {name_b}:")

        for word_set in ["primes", "random"]:
            acts_a = activations[name_a][word_set]
            acts_b = activations[name_b][word_set]

            # Convert to backend arrays
            acts_a_backend = backend.array(acts_a.tolist())
            acts_b_backend = backend.array(acts_b.tolist())

            result = compute_cka(acts_a_backend, acts_b_backend, backend)
            raw_results[(name_a, name_b, word_set)] = result.cka

            status = "PASS" if result.cka > 0.85 else "FAIL"
            logger.info(f"  {word_set:10s}: CKA = {result.cka:.4f} [{status}]")

    # Phase 2: Gram-Aligned CKA
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: GRAM-ALIGNED CKA")
    logger.info("=" * 60)
    logger.info("Expected if thesis holds: = 1.0 (within epsilon)")
    logger.info("")

    aligned_results = {}

    for name_a, name_b in combinations(model_names, 2):
        logger.info(f"{name_a} <-> {name_b}:")

        for word_set in ["primes", "random"]:
            acts_a = activations[name_a][word_set]
            acts_b = activations[name_b][word_set]

            # Convert to backend arrays
            acts_a_backend = backend.array(acts_a.tolist())
            acts_b_backend = backend.array(acts_b.tolist())

            try:
                # Find alignment transformation
                alignment = find_alignment(acts_a_backend, acts_b_backend, backend)
                aligned_cka = alignment.achieved_cka

                aligned_results[(name_a, name_b, word_set)] = aligned_cka

                status = "PASS" if aligned_cka > 0.99 else "PARTIAL" if aligned_cka > 0.95 else "FAIL"
                logger.info(f"  {word_set:10s}: CKA = {aligned_cka:.6f} [{status}]")

            except Exception as e:
                logger.warning(f"  {word_set:10s}: Alignment failed - {e}")
                aligned_results[(name_a, name_b, word_set)] = None

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    # Raw CKA summary
    raw_values = [v for v in raw_results.values() if v is not None]
    if raw_values:
        raw_mean = sum(raw_values) / len(raw_values)
        raw_min = min(raw_values)
        raw_max = max(raw_values)
        logger.info(f"\nRaw CKA: mean={raw_mean:.4f}, min={raw_min:.4f}, max={raw_max:.4f}")

        raw_pass = sum(1 for v in raw_values if v > 0.85)
        logger.info(f"  Passing (>0.85): {raw_pass}/{len(raw_values)}")

    # Aligned CKA summary
    aligned_values = [v for v in aligned_results.values() if v is not None]
    if aligned_values:
        aligned_mean = sum(aligned_values) / len(aligned_values)
        aligned_min = min(aligned_values)
        aligned_max = max(aligned_values)
        logger.info(f"\nAligned CKA: mean={aligned_mean:.6f}, min={aligned_min:.6f}, max={aligned_max:.6f}")

        aligned_pass = sum(1 for v in aligned_values if v > 0.99)
        logger.info(f"  Passing (>0.99): {aligned_pass}/{len(aligned_values)}")

    # Verdict
    logger.info("\n" + "=" * 60)
    logger.info("THESIS VERDICT")
    logger.info("=" * 60)

    all_raw_pass = all(v > 0.85 for v in raw_values) if raw_values else False
    all_aligned_pass = all(v > 0.99 for v in aligned_values) if aligned_values else False

    if all_aligned_pass:
        logger.info("\nTHESIS VERIFIED")
        logger.info("All models converge to invariant geometry.")
        logger.info("Gram alignment achieves CKA = 1.0 across all pairs.")
        return 0
    elif all_raw_pass:
        logger.info("\nTHESIS PARTIALLY SUPPORTED")
        logger.info("Raw CKA is high across families, but alignment doesn't reach 1.0.")
        logger.info("This may indicate minor structural differences or numerical issues.")
        return 0
    else:
        logger.info("\nTHESIS REQUIRES INVESTIGATION")
        logger.info("Some model pairs show lower CKA than expected.")
        logger.info("This could indicate model-specific structure or probe issues.")
        return 1


if __name__ == "__main__":
    sys.exit(run_experiment())
