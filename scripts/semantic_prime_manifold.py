#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Semantic Prime Manifold Discovery
"""
Semantic Prime Manifold Discovery

The key insight from linguistic universals research (Wierzbicka, Goddard):
- There are ~65 semantic primes that exist in ALL human languages
- All other concepts are COMPOSED from these primes
- They sit at the bottom of the conceptual hierarchy

Therefore:
- The semantic manifold IS the span of semantic prime activations
- Any direction orthogonal to this span is non-semantic (syntax, position, noise)
- ~65 probes should span the entire semantic space

This gives us:
1. A theoretically-grounded, finite probe set
2. Complete coverage by construction (primes span all semantics)
3. Likely << 65 effective dimensions (primes have internal structure)

References:
- Wierzbicka, A. (1996). Semantics: Primes and Universals. Oxford University Press.
- Goddard, C. (2011). Semantic Analysis. Oxford University Press.
- NSM (Natural Semantic Metalanguage) project: https://nsm-approach.net/

Usage:
    python semantic_prime_manifold.py --model /path/to/model --layer 8
    python semantic_prime_manifold.py --model /path/to/model --all-layers
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC PRIMES (NSM - Natural Semantic Metalanguage)
# Based on Wierzbicka/Goddard's cross-linguistic research
# These ~65 concepts exist in ALL known human languages
# =============================================================================

SEMANTIC_PRIMES = {
    # SUBSTANTIVES (entities)
    "substantives": [
        "I",
        "you",
        "someone",
        "something",
        "people",
        "body",
    ],

    # DETERMINERS
    "determiners": [
        "this",
        "the same",
        "other",
    ],

    # QUANTIFIERS
    "quantifiers": [
        "one",
        "two",
        "some",
        "all",
        "many",
        "much",
    ],

    # EVALUATORS
    "evaluators": [
        "good",
        "bad",
        "big",
        "small",
    ],

    # DESCRIPTORS
    "descriptors": [
        "true",
        "more",
        "very",
    ],

    # MENTAL PREDICATES
    "mental": [
        "think",
        "know",
        "want",
        "feel",
        "see",
        "hear",
    ],

    # SPEECH
    "speech": [
        "say",
        "words",
    ],

    # ACTIONS/EVENTS
    "actions": [
        "do",
        "happen",
        "move",
    ],

    # EXISTENCE/POSSESSION
    "existence": [
        "there is",
        "be",
        "live",
        "die",
        "have",
    ],

    # LOGICAL CONCEPTS
    "logical": [
        "not",
        "maybe",
        "can",
        "because",
        "if",
    ],

    # TIME
    "time": [
        "when",
        "now",
        "before",
        "after",
        "a long time",
        "a short time",
        "moment",
    ],

    # SPACE
    "space": [
        "where",
        "here",
        "above",
        "below",
        "far",
        "near",
        "side",
        "inside",
        "touch",
    ],

    # SIMILARITY/TAXONOMY
    "taxonomy": [
        "kind of",
        "part of",
        "like",
    ],
}


def get_all_primes() -> list[str]:
    """Get flat list of all semantic primes."""
    primes = []
    for category, items in SEMANTIC_PRIMES.items():
        primes.extend(items)
    return primes


def get_prime_contexts() -> list[tuple[str, str, str]]:
    """Get semantic primes with minimal contexts for activation.

    Returns list of (prime, context, category) tuples.
    Using minimal context to get pure semantic activation.
    """
    contexts = []

    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            # Use prime in minimal sentence context
            # Goal: activate the semantic concept, not syntax
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime  # Bare noun/pronoun
            elif prime in ["this", "the same", "other"]:
                context = f"{prime} thing"
            elif prime in ["one", "two", "some", "all", "many", "much"]:
                context = f"{prime} things"
            elif prime in ["good", "bad", "big", "small", "true"]:
                context = f"It is {prime}"
            elif prime in ["more", "very"]:
                context = f"{prime} good"
            elif prime in ["think", "know", "want", "feel", "see", "hear"]:
                context = f"I {prime}"
            elif prime in ["say"]:
                context = "I say"
            elif prime in ["words"]:
                context = "words"
            elif prime in ["do", "happen", "move"]:
                context = f"Things {prime}"
            elif prime in ["there is"]:
                context = "There is something"
            elif prime in ["be"]:
                context = "I am"
            elif prime in ["live", "die"]:
                context = f"People {prime}"
            elif prime in ["have"]:
                context = "I have"
            elif prime in ["not"]:
                context = "not this"
            elif prime in ["maybe"]:
                context = "maybe"
            elif prime in ["can"]:
                context = "I can"
            elif prime in ["because"]:
                context = "because"
            elif prime in ["if"]:
                context = "if"
            elif prime in ["when", "now", "before", "after"]:
                context = prime
            elif prime in ["a long time", "a short time", "moment"]:
                context = prime
            elif prime in ["where", "here"]:
                context = prime
            elif prime in ["above", "below", "far", "near", "inside"]:
                context = prime
            elif prime in ["side"]:
                context = "side"
            elif prime in ["touch"]:
                context = "touch"
            elif prime in ["kind of", "part of", "like"]:
                context = prime
            else:
                context = prime

            contexts.append((prime, context, category))

    return contexts


@dataclass
class ManifoldResult:
    """Result of semantic prime manifold analysis."""
    layer_idx: int
    hidden_dim: int
    n_primes: int
    manifold_dim: int  # Effective dimension of prime span
    compression_ratio: float
    variance_captured: float
    projection: "Array"  # [hidden_dim, manifold_dim]
    prime_activations: "Array"  # [n_primes, hidden_dim]
    eigenvalues: list[float]
    category_dimensions: dict[str, int]  # Dimensions per category


def initialize_backend() -> "Backend":
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    return get_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
    )

    return model, tokenizer, config


def collect_prime_activations(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
) -> tuple["Array", list[tuple[str, str, str]]]:
    """Collect activations for all semantic primes at a layer.

    Returns:
        Tuple of (activations [n_primes, hidden_dim], prime_contexts)
    """
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    contexts = get_prime_contexts()

    activations = []
    successful_contexts = []

    for prime, context, category in contexts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, context)
            if layer_idx in acts:
                activations.append(acts[layer_idx])
                successful_contexts.append((prime, context, category))
        except Exception as e:
            logger.warning("Failed to get activation for '%s': %s", context, e)

    if not activations:
        raise ValueError(f"No activations collected for layer {layer_idx}")

    X = mx.stack(activations, axis=0)
    mx.eval(X)

    logger.info(
        "Collected %d/%d prime activations at layer %d",
        len(activations), len(contexts), layer_idx
    )

    return X, successful_contexts


def compute_manifold_from_primes(
    activations: "Array",
    backend: "Backend",
    target_variance: float = 0.99,
) -> tuple["Array", int, float, list[float]]:
    """Compute manifold basis from semantic prime activations.

    The manifold IS the span of these activations.

    Args:
        activations: [n_primes, hidden_dim]
        backend: Backend for tensor ops
        target_variance: Variance to capture (default 99%)

    Returns:
        Tuple of (projection P [hidden_dim, k], k, variance_captured, eigenvalues)
    """
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(activations), b)
    b.eval(X)

    shape = b.shape(X)
    n_samples = int(shape[0])
    hidden_dim = int(shape[1])

    # Center the data
    mean = b.mean(X, axis=0)
    b.eval(mean)
    X_centered = X - mean
    b.eval(X_centered)

    # Compute Gram matrix (more efficient when n < d)
    G = b.matmul(X_centered, b.transpose(X_centered))
    b.eval(G)

    # Eigendecomposition
    eigenvalues, eigenvectors = b.eigh(G)
    b.eval(eigenvalues, eigenvectors)

    # Sort descending
    indices = b.argsort(-eigenvalues)
    b.eval(indices)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    b.eval(eigenvalues, eigenvectors)

    # Convert to feature space: v_feature = X^T @ v_gram / sqrt(lambda)
    eigenvalues_list = [max(0, e) for e in eigenvalues.tolist()]
    total_var = sum(eigenvalues_list)

    if total_var == 0:
        logger.warning("Zero total variance, returning identity")
        return b.eye(hidden_dim), hidden_dim, 1.0, eigenvalues_list

    # Find dimension for target variance
    cumvar = 0.0
    manifold_dim = 0
    for i, eig in enumerate(eigenvalues_list):
        cumvar += eig
        manifold_dim = i + 1
        if cumvar / total_var >= target_variance:
            break

    # Ensure at least 1 dimension
    manifold_dim = max(manifold_dim, 1)

    # Build projection matrix
    valid_mask = eigenvalues > 1e-10
    n_valid = min(manifold_dim, int(b.to_scalar(b.sum(b.astype(valid_mask, "float32")))))

    sqrt_eigs = b.sqrt(b.maximum(eigenvalues[:n_valid], b.array([1e-10])))
    b.eval(sqrt_eigs)
    V_scaled = eigenvectors[:, :n_valid] / sqrt_eigs
    b.eval(V_scaled)

    # Project to feature space
    P = b.matmul(b.transpose(X_centered), V_scaled)
    b.eval(P)

    # Normalize columns
    norms = b.sqrt(b.sum(P * P, axis=0))
    b.eval(norms)
    P = P / b.maximum(norms, b.array([1e-10]))
    b.eval(P)

    # Variance captured
    variance_captured = sum(eigenvalues_list[:manifold_dim]) / total_var

    return P, manifold_dim, variance_captured, eigenvalues_list


def analyze_category_dimensions(
    activations: "Array",
    contexts: list[tuple[str, str, str]],
    projection: "Array",
    backend: "Backend",
) -> dict[str, int]:
    """Analyze how many dimensions each category contributes.

    This tells us which semantic categories have distinct geometric structure.
    """
    b = backend

    # Group activations by category
    categories = {}
    for i, (prime, context, category) in enumerate(contexts):
        if category not in categories:
            categories[category] = []
        categories[category].append(i)

    category_dims = {}

    for category, indices in categories.items():
        if len(indices) < 2:
            category_dims[category] = 1
            continue

        # Extract category activations
        cat_acts = activations[indices]
        b.eval(cat_acts)

        # Center
        mean = b.mean(cat_acts, axis=0)
        b.eval(mean)
        cat_centered = cat_acts - mean
        b.eval(cat_centered)

        # SVD to find intrinsic dimension
        try:
            U, S, Vt = b.svd(cat_centered, full_matrices=False)
            b.eval(S)

            # Count significant singular values (> 1% of max)
            S_list = S.tolist()
            S_max = max(S_list) if S_list else 0
            threshold = 0.01 * S_max
            n_significant = sum(1 for s in S_list if s > threshold)
            category_dims[category] = max(1, n_significant)
        except Exception:
            category_dims[category] = len(indices)

    return category_dims


def compute_cka(X: "Array", Y: "Array", backend: "Backend") -> float:
    """Compute CKA between two activation matrices."""
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(X), b)
    Y = _promote_precision_float32(b.array(Y), b)
    b.eval(X, Y)

    # Center
    X = X - b.mean(X, axis=0)
    Y = Y - b.mean(Y, axis=0)
    b.eval(X, Y)

    # Gram matrices
    K_X = b.matmul(X, b.transpose(X))
    K_Y = b.matmul(Y, b.transpose(Y))
    b.eval(K_X, K_Y)

    # Center Gram matrices (HSIC)
    n = int(b.shape(K_X)[0])
    H = b.eye(n) - b.ones((n, n)) / n
    b.eval(H)

    K_X_centered = b.matmul(H, b.matmul(K_X, H))
    K_Y_centered = b.matmul(H, b.matmul(K_Y, H))
    b.eval(K_X_centered, K_Y_centered)

    # HSIC values
    hsic_xy = b.sum(K_X_centered * K_Y_centered)
    hsic_xx = b.sum(K_X_centered * K_X_centered)
    hsic_yy = b.sum(K_Y_centered * K_Y_centered)
    b.eval(hsic_xy, hsic_xx, hsic_yy)

    # CKA
    denom = b.sqrt(hsic_xx * hsic_yy)
    b.eval(denom)

    if float(b.to_scalar(denom)) < 1e-10:
        return 0.0

    cka = hsic_xy / denom
    b.eval(cka)

    return float(b.to_scalar(cka))


def analyze_layer(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
    target_variance: float = 0.99,
) -> ManifoldResult:
    """Analyze semantic prime manifold for a single layer."""
    hidden_dim = config.get("hidden_size", 0)

    # Collect prime activations
    activations, contexts = collect_prime_activations(
        model, tokenizer, config, layer_idx, backend
    )

    n_primes = int(backend.shape(activations)[0])

    # Compute manifold
    projection, manifold_dim, variance_captured, eigenvalues = compute_manifold_from_primes(
        activations, backend, target_variance=target_variance
    )

    # Analyze category contributions
    category_dims = analyze_category_dimensions(
        activations, contexts, projection, backend
    )

    compression_ratio = hidden_dim / manifold_dim

    logger.info(
        "Layer %d: %d primes -> %dD manifold (%.1fx compression, %.2f%% variance)",
        layer_idx, n_primes, manifold_dim, compression_ratio, variance_captured * 100
    )

    # Log category breakdown
    for cat, dim in sorted(category_dims.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d dimensions", cat, dim)

    return ManifoldResult(
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        n_primes=n_primes,
        manifold_dim=manifold_dim,
        compression_ratio=compression_ratio,
        variance_captured=variance_captured,
        projection=projection,
        prime_activations=activations,
        eigenvalues=eigenvalues,
        category_dimensions=category_dims,
    )


def verify_manifold_completeness(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    projection: "Array",
    backend: "Backend",
    test_concepts: list[str],
) -> tuple[float, float, list[tuple[str, float]]]:
    """Verify that derived concepts live in the prime manifold.

    If semantic primes truly span all semantics, any concept should
    be reconstructable from the manifold projection.

    Returns:
        Tuple of (mean_cka, mean_reconstruction_error, per_concept_errors)
    """
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    b = backend
    P = projection

    errors = []
    activations_original = []
    activations_projected = []

    for concept in test_concepts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, concept)
            if layer_idx not in acts:
                continue

            a = b.array(acts[layer_idx])
            b.eval(a)

            # Project to manifold and back
            a_in_manifold = b.matmul(b.transpose(P), b.reshape(a, (-1, 1)))
            b.eval(a_in_manifold)
            a_proj = b.matmul(P, a_in_manifold)
            a_proj = b.reshape(a_proj, (-1,))
            b.eval(a_proj)

            # Reconstruction error
            diff = a - a_proj
            b.eval(diff)

            error = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))
            norm = float(b.to_scalar(b.sqrt(b.sum(a * a))))

            rel_error = error / norm if norm > 0 else 0
            errors.append((concept, rel_error))

            activations_original.append(a)
            activations_projected.append(a_proj)

        except Exception as e:
            logger.warning("Failed to verify '%s': %s", concept, e)

    if not errors:
        return 0.0, 1.0, []

    mean_error = sum(e for _, e in errors) / len(errors)

    # Compute CKA between original and projected
    if activations_original:
        X_orig = mx.stack(activations_original, axis=0)
        X_proj = mx.stack(activations_projected, axis=0)
        mx.eval(X_orig, X_proj)
        cka = compute_cka(X_orig, X_proj, b)
    else:
        cka = 0.0

    return cka, mean_error, errors


def main():
    parser = argparse.ArgumentParser(
        description="Semantic prime manifold discovery"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to analyze")
    parser.add_argument("--all-layers", action="store_true", help="Analyze all layers")
    parser.add_argument("--target-variance", type=float, default=0.99,
                        help="Target variance to capture (default: 0.99)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify with derived concepts")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.layer is None and not args.all_layers:
        parser.error("Must specify --layer or --all-layers")

    # Initialize
    backend = initialize_backend()
    model, tokenizer, config = load_model(args.model)

    num_layers = config.get("num_hidden_layers", 0)
    hidden_dim = config.get("hidden_size", 0)

    # List primes
    all_primes = get_all_primes()
    logger.info("")
    logger.info("=== SEMANTIC PRIMES (%d total) ===", len(all_primes))
    for category, primes in SEMANTIC_PRIMES.items():
        logger.info("  %s: %s", category, ", ".join(primes))

    # Determine layers
    if args.all_layers:
        layers_to_analyze = list(range(num_layers))
    else:
        layers_to_analyze = [args.layer]

    logger.info("")
    logger.info("Will analyze %d layer(s): %s", len(layers_to_analyze), layers_to_analyze)

    results: dict[int, ManifoldResult] = {}

    for layer_idx in layers_to_analyze:
        logger.info("")
        logger.info("=" * 60)
        logger.info("LAYER %d", layer_idx)
        logger.info("=" * 60)

        result = analyze_layer(
            model, tokenizer, config, layer_idx, backend,
            target_variance=args.target_variance
        )
        results[layer_idx] = result

    # Verify with derived concepts
    if args.verify:
        logger.info("")
        logger.info("=== VERIFICATION: Derived Concepts ===")

        # Test concepts that should be COMPOSED from primes
        derived_concepts = [
            # Emotions (composed from feel + evaluators)
            "happy", "sad", "angry", "afraid",
            # Actions (composed from do + mental)
            "run", "eat", "sleep", "work",
            # Complex concepts
            "love", "hate", "truth", "freedom",
            # Abstract
            "mathematics", "philosophy", "science",
            # Concrete
            "tree", "water", "fire", "stone",
        ]

        for layer_idx, result in results.items():
            cka, mean_error, per_concept = verify_manifold_completeness(
                model, tokenizer, config, layer_idx,
                result.projection, backend, derived_concepts
            )

            logger.info("")
            logger.info("Layer %d verification:", layer_idx)
            logger.info("  CKA (original vs projected): %.4f", cka)
            logger.info("  Mean reconstruction error: %.4f%%", mean_error * 100)

            if cka >= 0.99:
                logger.info("  COMPLETE: Derived concepts live in prime manifold")
            elif cka >= 0.95:
                logger.info("  MOSTLY COMPLETE: 95%%+ CKA")
            else:
                logger.info("  INCOMPLETE: Some concepts outside prime manifold")

                # Show worst reconstructions
                worst = sorted(per_concept, key=lambda x: -x[1])[:5]
                logger.info("  Worst reconstructions:")
                for concept, error in worst:
                    logger.info("    %s: %.2f%% error", concept, error * 100)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("Layer | Primes | Hidden | Manifold | Compression | Variance")
    logger.info("-" * 65)

    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        logger.info(
            "%5d | %6d | %6d | %8d | %10.1fx | %.2f%%",
            layer_idx, r.n_primes, r.hidden_dim, r.manifold_dim,
            r.compression_ratio, r.variance_captured * 100
        )

    # Identify layers by manifold dimension
    if len(results) > 1:
        sorted_by_dim = sorted(results.items(), key=lambda x: x[1].manifold_dim)
        highway_layers = [idx for idx, r in sorted_by_dim[:3]]
        ramp_layers = [idx for idx, r in sorted_by_dim[-3:]]

        logger.info("")
        logger.info("Highway layers (lowest manifold dim): %s", highway_layers)
        logger.info("Ramp layers (highest manifold dim): %s", ramp_layers)

    # Eigenvalue analysis
    if len(results) == 1:
        layer_idx = list(results.keys())[0]
        r = results[layer_idx]
        logger.info("")
        logger.info("=== EIGENVALUE SPECTRUM (Layer %d) ===", layer_idx)
        eigenvalues = r.eigenvalues[:20]  # Top 20
        total = sum(r.eigenvalues)
        cumvar = 0.0
        for i, eig in enumerate(eigenvalues):
            cumvar += eig
            pct = eig / total * 100 if total > 0 else 0
            cumpct = cumvar / total * 100 if total > 0 else 0
            bar = "#" * int(pct * 2)
            logger.info("  PC%2d: %6.2f%% (cum: %5.1f%%) %s", i + 1, pct, cumpct, bar)

    # Save results
    if args.output:
        output_data = {}
        for layer_idx, r in results.items():
            output_data[layer_idx] = {
                "n_primes": r.n_primes,
                "hidden_dim": r.hidden_dim,
                "manifold_dim": r.manifold_dim,
                "compression_ratio": r.compression_ratio,
                "variance_captured": r.variance_captured,
                "eigenvalues": r.eigenvalues[:50],  # Top 50
                "category_dimensions": r.category_dimensions,
            }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("")
        logger.info("Results saved to %s", args.output)

    logger.info("")
    logger.info("DONE")


if __name__ == "__main__":
    main()
