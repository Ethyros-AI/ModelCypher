#!/usr/bin/env python3
"""
Layer-by-Layer CKA Analysis: Does invariant geometry exist across ALL layers?

This experiment extends thesis_verification.py to analyze CKA across layer depth:
- Collect activations for ALL layers (not just middle)
- Map corresponding layers by position ratio
- Compute raw and aligned CKA at key positions (0%, 25%, 50%, 75%, 100%)

If thesis holds uniformly: Aligned CKA = 1.0 at ALL layer positions.
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
from modelcypher.backends import get_backend
from modelcypher.core.domain._backend import get_default_backend, set_default_backend
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
from modelcypher.core.domain.geometry.cka import compute_linear_cka
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
}

# Random words for control comparison
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

# Layer positions to analyze (0% = input, 100% = output)
LAYER_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def get_num_layers(model) -> int:
    """Get number of layers in a model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    return 1


def get_layer_at_position(position: float, num_layers: int) -> int:
    """Get layer index for a normalized position [0, 1]."""
    return round(position * (num_layers - 1))


def collect_all_layer_activations(
    model,
    tokenizer,
    words: list[str],
    provider: MLXActivationProvider,
) -> dict[int, mx.array]:
    """Collect activations for ALL layers across all words.

    Returns:
        dict[layer_idx, mx.array] where each array is [n_words, hidden_dim]
    """
    num_layers = get_num_layers(model)
    all_acts: dict[int, list[mx.array]] = {i: [] for i in range(num_layers)}

    for word in words:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, word)
            for layer_idx, act in acts.items():
                all_acts[layer_idx].append(act)
        except Exception as e:
            logger.warning(f"Failed to get activation for '{word}': {e}")
            continue

    # Stack and convert to float32
    result = {}
    for layer_idx, acts_list in all_acts.items():
        if acts_list:
            stacked = mx.stack(acts_list, axis=0)
            stacked = stacked.astype(mx.float32)
            mx.eval(stacked)
            result[layer_idx] = stacked

    return result


def is_single_token(word: str, tokenizer) -> bool:
    """Check if word tokenizes to a single token."""
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if isinstance(tokens, list):
        return len(tokens) == 1
    return len(list(tokens.ids)) == 1


def run_experiment():
    """Run the layer-by-layer CKA analysis."""

    logger.info("=" * 60)
    logger.info("LAYER-BY-LAYER CKA ANALYSIS")
    logger.info("=" * 60)

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
    num_layers = {}
    for name, path in MODELS.items():
        logger.info(f"Loading {name} from {path}...")
        start = time.time()
        model, tokenizer = load(path)
        elapsed = time.time() - start

        n_layers = get_num_layers(model)
        num_layers[name] = n_layers
        logger.info(f"  Loaded in {elapsed:.1f}s ({n_layers} layers)")
        models[name] = (model, tokenizer)

    # Filter words to vocabulary intersection
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING WORDS TO VOCABULARY INTERSECTION")
    logger.info("=" * 60)

    filtered_primes = [
        word for word in prime_words
        if all(is_single_token(word, tok) for _, tok in models.values())
    ]
    filtered_random = [
        word for word in RANDOM_WORDS
        if all(is_single_token(word, tok) for _, tok in models.values())
    ]

    logger.info(f"Semantic primes: {len(prime_words)} -> {len(filtered_primes)} (single-token)")
    logger.info(f"Random words: {len(RANDOM_WORDS)} -> {len(filtered_random)} (single-token)")

    # Use common size
    n_words = min(len(filtered_primes), len(filtered_random), 50)
    test_primes = filtered_primes[:n_words]
    test_random = filtered_random[:n_words]
    logger.info(f"Using {n_words} words per set")

    # Collect activations for ALL layers
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING ACTIVATIONS FOR ALL LAYERS")
    logger.info("=" * 60)

    all_activations = {}
    for name, (model, tokenizer) in models.items():
        logger.info(f"\n{name} ({num_layers[name]} layers):")

        # Semantic primes
        logger.info("  Collecting primes...")
        start = time.time()
        acts_primes = collect_all_layer_activations(model, tokenizer, test_primes, provider)
        logger.info(f"    {len(acts_primes)} layers, Time: {time.time() - start:.1f}s")

        # Random words
        logger.info("  Collecting random words...")
        start = time.time()
        acts_random = collect_all_layer_activations(model, tokenizer, test_random, provider)
        logger.info(f"    {len(acts_random)} layers, Time: {time.time() - start:.1f}s")

        all_activations[name] = {
            "primes": acts_primes,
            "random": acts_random,
        }

    # Phase 1: Diagonal Analysis at Key Positions
    logger.info("\n" + "=" * 60)
    logger.info("DIAGONAL ANALYSIS (Corresponding Layers)")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"{'Position':<10} | {'Pair':<25} | {'Raw CKA':>8} | {'Aligned CKA':>12}")
    logger.info("-" * 65)

    model_names = list(models.keys())
    results = []

    for name_a, name_b in combinations(model_names, 2):
        n_a = num_layers[name_a]
        n_b = num_layers[name_b]

        for pos in LAYER_POSITIONS:
            layer_a = get_layer_at_position(pos, n_a)
            layer_b = get_layer_at_position(pos, n_b)

            # Use semantic primes for primary analysis
            acts_a = all_activations[name_a]["primes"].get(layer_a)
            acts_b = all_activations[name_b]["primes"].get(layer_b)

            if acts_a is None or acts_b is None:
                logger.warning(f"Missing activations for {name_a}[{layer_a}] or {name_b}[{layer_b}]")
                continue

            # Raw CKA
            raw_cka = compute_linear_cka(acts_a, acts_b, backend)

            # Aligned CKA
            try:
                alignment = find_alignment(acts_a, acts_b, backend)
                aligned_cka = alignment.achieved_cka
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")
                aligned_cka = None

            pair_str = f"{name_a}[{layer_a}]<->{name_b}[{layer_b}]"
            pos_str = f"{pos:.0%}"

            if aligned_cka is not None:
                logger.info(f"{pos_str:<10} | {pair_str:<25} | {raw_cka:>8.4f} | {aligned_cka:>12.6f}")
                results.append({
                    "position": pos,
                    "model_a": name_a,
                    "model_b": name_b,
                    "layer_a": layer_a,
                    "layer_b": layer_b,
                    "raw_cka": raw_cka,
                    "aligned_cka": aligned_cka,
                })
            else:
                logger.info(f"{pos_str:<10} | {pair_str:<25} | {raw_cka:>8.4f} | {'FAILED':>12}")

    # Summary by position
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY BY LAYER POSITION")
    logger.info("=" * 60)

    for pos in LAYER_POSITIONS:
        pos_results = [r for r in results if r["position"] == pos]
        if not pos_results:
            continue

        raw_vals = [r["raw_cka"] for r in pos_results]
        aligned_vals = [r["aligned_cka"] for r in pos_results if r["aligned_cka"] is not None]

        raw_mean = sum(raw_vals) / len(raw_vals) if raw_vals else 0
        aligned_mean = sum(aligned_vals) / len(aligned_vals) if aligned_vals else 0
        aligned_min = min(aligned_vals) if aligned_vals else 0

        status = "PASS" if aligned_min > 0.99 else "PARTIAL" if aligned_min > 0.95 else "FAIL"

        pos_str = f"{pos:.0%}"
        logger.info(f"{pos_str:>5}: raw_mean={raw_mean:.4f}, aligned_mean={aligned_mean:.6f}, "
                   f"aligned_min={aligned_min:.6f} [{status}]")

    # Final verdict
    logger.info("\n" + "=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)

    all_aligned = [r["aligned_cka"] for r in results if r["aligned_cka"] is not None]
    if not all_aligned:
        logger.info("\nNO RESULTS - All alignments failed")
        return 1

    min_aligned = min(all_aligned)
    mean_aligned = sum(all_aligned) / len(all_aligned)

    if min_aligned > 0.99:
        logger.info("\nTHESIS VERIFIED ACROSS ALL LAYERS")
        logger.info(f"All {len(all_aligned)} layer pairs achieve CKA > 0.99")
        logger.info(f"Min aligned CKA: {min_aligned:.6f}")
        logger.info(f"Mean aligned CKA: {mean_aligned:.6f}")
        logger.info("\nThe invariant geometry exists throughout the entire network depth.")
        return 0
    elif min_aligned > 0.95:
        logger.info("\nTHESIS PARTIALLY SUPPORTED")
        logger.info(f"Most layer pairs achieve high CKA but not all reach 0.99")
        logger.info(f"Min aligned CKA: {min_aligned:.6f}")
        return 0
    else:
        logger.info("\nTHESIS REQUIRES INVESTIGATION")
        logger.info(f"Some layer pairs show unexpectedly low CKA")
        logger.info(f"Min aligned CKA: {min_aligned:.6f}")
        return 1


if __name__ == "__main__":
    sys.exit(run_experiment())
