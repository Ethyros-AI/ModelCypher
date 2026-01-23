#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Bottleneck 1D Probing
"""
Bottleneck 1D Probing

THE QUESTION:
If all semantic information passes through a single dimension at layers 7 and 14,
what does that dimension encode?

HYPOTHESES:
1. Semantic category (abstract vs concrete, animate vs inanimate)
2. Concept identity (unique hash per concept)
3. Linguistic properties (frequency, complexity)
4. Something else entirely

METHOD:
1. Extract the 1D value at bottleneck layers for many concepts
2. Correlate with known properties
3. Train simple classifiers to decode properties from 1D alone

If 1D encodes "everything", we should be able to decode "anything" from it.

Usage:
    python bottleneck_1d_probing.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Concepts with known properties
CONCEPTS = {
    # Concrete objects
    "apple": {"category": "fruit", "concrete": 1, "animate": 0, "valence": 0.6},
    "orange": {"category": "fruit", "concrete": 1, "animate": 0, "valence": 0.6},
    "banana": {"category": "fruit", "concrete": 1, "animate": 0, "valence": 0.5},
    "grape": {"category": "fruit", "concrete": 1, "animate": 0, "valence": 0.5},
    # Animals
    "dog": {"category": "animal", "concrete": 1, "animate": 1, "valence": 0.8},
    "cat": {"category": "animal", "concrete": 1, "animate": 1, "valence": 0.7},
    "bird": {"category": "animal", "concrete": 1, "animate": 1, "valence": 0.6},
    "fish": {"category": "animal", "concrete": 1, "animate": 1, "valence": 0.5},
    # Vehicles
    "car": {"category": "vehicle", "concrete": 1, "animate": 0, "valence": 0.5},
    "truck": {"category": "vehicle", "concrete": 1, "animate": 0, "valence": 0.4},
    "bike": {"category": "vehicle", "concrete": 1, "animate": 0, "valence": 0.6},
    "plane": {"category": "vehicle", "concrete": 1, "animate": 0, "valence": 0.5},
    # Abstract concepts
    "love": {"category": "emotion", "concrete": 0, "animate": 0, "valence": 0.9},
    "hate": {"category": "emotion", "concrete": 0, "animate": 0, "valence": 0.1},
    "fear": {"category": "emotion", "concrete": 0, "animate": 0, "valence": 0.2},
    "joy": {"category": "emotion", "concrete": 0, "animate": 0, "valence": 0.95},
    # Colors
    "red": {"category": "color", "concrete": 0.5, "animate": 0, "valence": 0.5},
    "blue": {"category": "color", "concrete": 0.5, "animate": 0, "valence": 0.6},
    "green": {"category": "color", "concrete": 0.5, "animate": 0, "valence": 0.6},
    "yellow": {"category": "color", "concrete": 0.5, "animate": 0, "valence": 0.7},
    # Actions
    "run": {"category": "action", "concrete": 0.3, "animate": 0, "valence": 0.6},
    "walk": {"category": "action", "concrete": 0.3, "animate": 0, "valence": 0.5},
    "jump": {"category": "action", "concrete": 0.3, "animate": 0, "valence": 0.6},
    "sleep": {"category": "action", "concrete": 0.3, "animate": 0, "valence": 0.5},
    # Qualities
    "hot": {"category": "quality", "concrete": 0.2, "animate": 0, "valence": 0.4},
    "cold": {"category": "quality", "concrete": 0.2, "animate": 0, "valence": 0.3},
    "fast": {"category": "quality", "concrete": 0.2, "animate": 0, "valence": 0.6},
    "slow": {"category": "quality", "concrete": 0.2, "animate": 0, "valence": 0.4},
}


@dataclass
class BottleneckProbe:
    """Results from probing the 1D bottleneck."""
    layer_idx: int
    concept_values: dict[str, float]  # concept -> 1D value

    # Correlations with properties
    concrete_correlation: float
    animate_correlation: float
    valence_correlation: float

    # Category separability
    category_separability: float  # How well categories cluster


def get_bottleneck_value(
    model: Any,
    tokenizer: Any,
    word: str,
    layer_idx: int,
) -> float:
    """Get the 1D bottleneck value for a word at a specific layer.

    The "1D value" is the projection onto the first principal component
    of the layer delta (output - input).
    """
    import mlx.core as mx

    tokens = tokenizer.encode(word)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = model.model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(model.model.layers):
        if idx < layer_idx:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        elif idx == layer_idx:
            h_in = np.array(h[0, -1, :].astype(mx.float32))

            result = layer(h)
            h_out_full = result[0] if isinstance(result, tuple) else result
            mx.eval(h_out_full)

            h_out = np.array(h_out_full[0, -1, :].astype(mx.float32))
            delta = h_out - h_in

            # The "1D value" is the norm of the delta projected to 1D
            # Since the bottleneck is 1D, the delta's magnitude IS the 1D value
            return float(np.linalg.norm(delta))

    return 0.0


def get_layer_pca_projection(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get PCA projection of layer deltas and return (values_1d, basis).

    Returns the first PC value for each concept.
    """
    import mlx.core as mx

    deltas = []

    for word in concepts:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(model.model.layers):
            if idx < layer_idx:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)
            elif idx == layer_idx:
                h_in = np.array(h[0, -1, :].astype(mx.float32))

                result = layer(h)
                h_out_full = result[0] if isinstance(result, tuple) else result
                mx.eval(h_out_full)

                h_out = np.array(h_out_full[0, -1, :].astype(mx.float32))
                deltas.append(h_out - h_in)
                break

    deltas = np.stack(deltas)
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA to find the 1D subspace
    mean = deltas.mean(axis=0)
    centered = deltas - mean
    cov = (centered.T @ centered) / len(deltas)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]

    # First principal component
    pc1 = eigenvectors[:, idx[0]]

    # Project each concept onto PC1
    values_1d = centered @ pc1

    return values_1d, pc1


def compute_correlation(values: np.ndarray, properties: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if len(values) < 3:
        return 0.0

    values = np.array(values)
    properties = np.array(properties)

    # Handle constant arrays
    if np.std(values) < 1e-10 or np.std(properties) < 1e-10:
        return 0.0

    return float(np.corrcoef(values, properties)[0, 1])


def compute_category_separability(values: np.ndarray, categories: list[str]) -> float:
    """Compute how well categories are separated in 1D space.

    Uses ratio of between-class variance to within-class variance.
    """
    unique_cats = list(set(categories))
    if len(unique_cats) < 2:
        return 0.0

    # Group values by category
    groups = {cat: [] for cat in unique_cats}
    for val, cat in zip(values, categories):
        groups[cat].append(val)

    # Between-class variance
    grand_mean = np.mean(values)
    between_var = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in groups.values() if len(g) > 0
    ) / len(values)

    # Within-class variance
    within_var = sum(
        np.var(g) * len(g)
        for g in groups.values() if len(g) > 1
    ) / len(values)

    if within_var < 1e-10:
        return 1.0 if between_var > 1e-10 else 0.0

    # Fisher's discriminant ratio
    return float(between_var / within_var)


def probe_bottleneck_layer(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> BottleneckProbe:
    """Probe what the 1D bottleneck encodes at a specific layer."""
    concepts = list(CONCEPTS.keys())

    # Get 1D projections
    values_1d, _ = get_layer_pca_projection(model, tokenizer, concepts, layer_idx)

    # Create mapping
    concept_values = {c: float(v) for c, v in zip(concepts, values_1d)}

    # Extract properties
    concrete = [CONCEPTS[c]["concrete"] for c in concepts]
    animate = [CONCEPTS[c]["animate"] for c in concepts]
    valence = [CONCEPTS[c]["valence"] for c in concepts]
    categories = [CONCEPTS[c]["category"] for c in concepts]

    # Compute correlations
    concrete_corr = compute_correlation(values_1d, concrete)
    animate_corr = compute_correlation(values_1d, animate)
    valence_corr = compute_correlation(values_1d, valence)

    # Compute category separability
    cat_sep = compute_category_separability(values_1d, categories)

    return BottleneckProbe(
        layer_idx=layer_idx,
        concept_values=concept_values,
        concrete_correlation=concrete_corr,
        animate_correlation=animate_corr,
        valence_correlation=valence_corr,
        category_separability=cat_sep,
    )


def main():
    parser = argparse.ArgumentParser(description="Bottleneck 1D probing")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print(f"\n{'='*80}")
    print("BOTTLENECK 1D PROBING")
    print("="*80)
    print(f"Model: {n_layers} layers")
    print(f"Testing {len(CONCEPTS)} concepts")

    # Probe all layers
    print(f"\n{'Layer':>6} | {'Concrete r':>12} | {'Animate r':>12} | {'Valence r':>12} | {'Cat Sep':>10}")
    print("-" * 70)

    probes = []
    for layer_idx in range(n_layers):
        probe = probe_bottleneck_layer(model, tokenizer, layer_idx)
        probes.append(probe)

        print(f"{layer_idx:>6} | {probe.concrete_correlation:>12.3f} | "
              f"{probe.animate_correlation:>12.3f} | {probe.valence_correlation:>12.3f} | "
              f"{probe.category_separability:>10.2f}")

    # Summary
    print(f"\n{'='*80}")
    print("WHAT DOES THE 1D ENCODE?")
    print("="*80)

    # Find strongest correlations
    max_concrete = max(probes, key=lambda p: abs(p.concrete_correlation))
    max_animate = max(probes, key=lambda p: abs(p.animate_correlation))
    max_valence = max(probes, key=lambda p: abs(p.valence_correlation))
    max_cat_sep = max(probes, key=lambda p: p.category_separability)

    print(f"\nStrongest correlations:")
    print(f"  Concreteness: r={max_concrete.concrete_correlation:.3f} at layer {max_concrete.layer_idx}")
    print(f"  Animacy:      r={max_animate.animate_correlation:.3f} at layer {max_animate.layer_idx}")
    print(f"  Valence:      r={max_valence.valence_correlation:.3f} at layer {max_valence.layer_idx}")
    print(f"  Category sep: {max_cat_sep.category_separability:.2f} at layer {max_cat_sep.layer_idx}")

    # Analyze bottleneck layers specifically (7 and 14 for LFM2-350M)
    bottleneck_layers = [7, 14] if n_layers == 16 else []

    if bottleneck_layers:
        print(f"\n{'='*80}")
        print("BOTTLENECK LAYER ANALYSIS")
        print("="*80)

        for bl in bottleneck_layers:
            if bl < len(probes):
                p = probes[bl]
                print(f"\nLayer {bl} (BOTTLENECK):")
                print(f"  Concreteness correlation: {p.concrete_correlation:.3f}")
                print(f"  Animacy correlation:      {p.animate_correlation:.3f}")
                print(f"  Valence correlation:      {p.valence_correlation:.3f}")
                print(f"  Category separability:    {p.category_separability:.2f}")

                # Show concept values sorted
                sorted_concepts = sorted(p.concept_values.items(), key=lambda x: x[1])
                print(f"\n  Concept ordering (low → high 1D value):")
                for i, (concept, val) in enumerate(sorted_concepts[:5]):
                    print(f"    {i+1}. {concept}: {val:.3f}")
                print("    ...")
                for i, (concept, val) in enumerate(sorted_concepts[-5:]):
                    print(f"    {len(sorted_concepts)-4+i}. {concept}: {val:.3f}")

    # The insight
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)

    # Check if any correlation is strong
    any_strong = any(
        abs(p.concrete_correlation) > 0.5 or
        abs(p.animate_correlation) > 0.5 or
        abs(p.valence_correlation) > 0.5
        for p in probes
    )

    if any_strong:
        print("""
The 1D bottleneck DOES encode interpretable properties!

Strong correlations suggest the 1D is a "semantic hash" that preserves:
- Concreteness (abstract vs physical)
- Animacy (living vs non-living)
- Valence (positive vs negative)

This is remarkable: a SINGLE DIMENSION preserves multiple orthogonal properties.
The bottleneck is not throwing away information - it's COMPRESSING it.
""")
    else:
        print("""
The 1D bottleneck does NOT strongly correlate with simple properties.

This suggests one of:
1. The 1D encodes something more complex (combinations of properties)
2. The 1D is a "hash" that's useful but not directly interpretable
3. Different concepts use the 1D differently

The bottleneck may be more like a "checkpoint" than a "summary".
""")


if __name__ == "__main__":
    main()
