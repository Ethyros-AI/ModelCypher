#!/usr/bin/env python3
"""
Cross-Model Geodesic Invariance Test

Tests whether the Pythagorean structure is invariant across models using
proper GEODESIC geometry instead of Euclidean Procrustes alignment.

Key differences from original:
- Uses Fréchet mean instead of arithmetic mean
- Uses geodesic distances instead of Euclidean
- Compares geodesic structural signatures across models
- No Procrustes (which assumes Euclidean space)

The hypothesis: If mathematical structure is encoded in the manifold,
the GEODESIC relationships should be consistent across models, even
though the embedding spaces have different scales and rotations.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    frechet_mean,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent


@dataclass
class GeodesicStructure:
    """Geodesic structural signature for a model."""
    model_name: str
    # Normalized geodesic ratios (scale-invariant)
    ratio_5_to_6_from_34: float  # geodesic(5→34) / geodesic(6→34)
    ratio_9_16_to_9_25: float    # geodesic(9↔16) / geodesic(9↔25)
    ratio_16_25_to_9_25: float   # geodesic(16↔25) / geodesic(9↔25)
    # Geodesic triangle inequality excess (curvature signature)
    triangle_excess_345: float   # (g(3,4) + g(4,5)) - g(3,5) normalized
    triangle_excess_sq: float    # (g(9,16) + g(16,25)) - g(9,25) normalized


def load_model(model_path: str):
    """Load MLX model and tokenizer."""
    from mlx_lm import load
    return load(model_path)


def extract_embedding(model, tokenizer, text: str, backend):
    """Extract embedding using Backend protocol."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    inner = getattr(model, 'model', model)
    x = inner.embed_tokens(input_ids) if hasattr(inner, 'embed_tokens') else inner.wte(input_ids)

    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    for layer in layers:
        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

    result = x[0, -1, :]
    mx.eval(result)
    result = result.astype(mx.float32)
    mx.eval(result)
    return backend.array(result)


def get_embeddings(model, tokenizer, numbers: list[int], backend) -> dict:
    """Get embeddings for a list of numbers."""
    embeddings = {}
    for n in numbers:
        emb = extract_embedding(model, tokenizer, f"The number {n}.", backend)
        backend.eval(emb)
        embeddings[n] = emb
    return embeddings


def compute_geodesic_structure(embeddings: dict, backend) -> dict:
    """Compute scale-invariant geodesic structural features."""
    rg = RiemannianGeometry(backend)

    numbers = sorted(embeddings.keys())
    points = backend.stack([embeddings[n] for n in numbers])
    backend.eval(points)

    # Compute geodesic distance matrix
    geo_result = rg.geodesic_distances(points, k_neighbors=min(10, len(numbers)-1))
    geo_dist = geo_result.distances
    backend.eval(geo_dist)

    def get_idx(n):
        return numbers.index(n)

    def get_geo(a, b):
        return float(backend.to_numpy(geo_dist[get_idx(a), get_idx(b)]))

    # Compute Frechet mean of (3,4)
    e3, e4 = embeddings[3], embeddings[4]
    pair_34 = backend.stack([e3, e4])
    backend.eval(pair_34)
    frechet_34 = frechet_mean(pair_34, backend=backend)
    backend.eval(frechet_34)

    # Add Frechet mean to point cloud for geodesic calculation
    points_ext = backend.concatenate([points, backend.reshape(frechet_34, (1, -1))], axis=0)
    backend.eval(points_ext)

    geo_ext = rg.geodesic_distances(points_ext, k_neighbors=min(10, len(numbers)))
    geo_dist_ext = geo_ext.distances
    backend.eval(geo_dist_ext)

    frechet_idx = len(numbers)

    def get_geo_to_frechet(n):
        return float(backend.to_numpy(geo_dist_ext[get_idx(n), frechet_idx]))

    # Geodesic distances
    geo_5_to_34 = get_geo_to_frechet(5)
    geo_6_to_34 = get_geo_to_frechet(6)

    geo_3_4 = get_geo(3, 4)
    geo_4_5 = get_geo(4, 5)
    geo_3_5 = get_geo(3, 5)

    geo_9_16 = get_geo(9, 16)
    geo_16_25 = get_geo(16, 25)
    geo_9_25 = get_geo(9, 25)

    # Scale-invariant ratios
    ratio_5_to_6 = geo_5_to_34 / (geo_6_to_34 + 1e-10)
    ratio_9_16_to_9_25 = geo_9_16 / (geo_9_25 + 1e-10)
    ratio_16_25_to_9_25 = geo_16_25 / (geo_9_25 + 1e-10)

    # Triangle inequality excess (normalized by perimeter)
    perimeter_345 = geo_3_4 + geo_4_5 + geo_3_5
    excess_345 = (geo_3_4 + geo_4_5 - geo_3_5) / (perimeter_345 + 1e-10)

    perimeter_sq = geo_9_16 + geo_16_25 + geo_9_25
    excess_sq = (geo_9_16 + geo_16_25 - geo_9_25) / (perimeter_sq + 1e-10)

    return {
        "ratio_5_to_6_from_34": ratio_5_to_6,
        "ratio_9_16_to_9_25": ratio_9_16_to_9_25,
        "ratio_16_25_to_9_25": ratio_16_25_to_9_25,
        "triangle_excess_345": excess_345,
        "triangle_excess_sq": excess_sq,
        # Raw values for reference
        "geo_5_to_34": geo_5_to_34,
        "geo_6_to_34": geo_6_to_34,
        "geo_9_16": geo_9_16,
        "geo_16_25": geo_16_25,
        "geo_9_25": geo_9_25,
    }


def compare_structures(struct_a: dict, struct_b: dict) -> dict:
    """Compare geodesic structures between two models."""
    # Compare scale-invariant features
    ratio_keys = ["ratio_5_to_6_from_34", "ratio_9_16_to_9_25", "ratio_16_25_to_9_25"]
    excess_keys = ["triangle_excess_345", "triangle_excess_sq"]

    ratio_diffs = []
    for key in ratio_keys:
        diff = abs(struct_a[key] - struct_b[key])
        ratio_diffs.append(diff)

    excess_diffs = []
    for key in excess_keys:
        diff = abs(struct_a[key] - struct_b[key])
        excess_diffs.append(diff)

    # Similarity score (1 = identical, 0 = very different)
    avg_ratio_diff = sum(ratio_diffs) / len(ratio_diffs)
    avg_excess_diff = sum(excess_diffs) / len(excess_diffs)

    # Convert to similarity (assuming typical differences < 1)
    ratio_similarity = max(0, 1 - avg_ratio_diff)
    excess_similarity = max(0, 1 - avg_excess_diff * 10)  # excess is smaller scale

    overall_similarity = (ratio_similarity + excess_similarity) / 2

    return {
        "ratio_similarity": ratio_similarity,
        "excess_similarity": excess_similarity,
        "overall_similarity": overall_similarity,
        "ratio_diffs": dict(zip(ratio_keys, ratio_diffs)),
        "excess_diffs": dict(zip(excess_keys, excess_diffs)),
    }


def main():
    import argparse
    import gc

    parser = argparse.ArgumentParser(description="Cross-model geodesic invariance test")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "cross_model_geodesic.json"))
    args = parser.parse_args()

    backend = get_default_backend()

    logger.info("=" * 70)
    logger.info("CROSS-MODEL GEODESIC INVARIANCE TEST")
    logger.info("Comparing scale-invariant geodesic structures")
    logger.info("=" * 70)

    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 24, 25]

    all_structures = {}

    for model_path in args.models:
        name = Path(model_path).name
        logger.info(f"\n--- Processing {name} ---")

        model, tokenizer = load_model(model_path)
        embeddings = get_embeddings(model, tokenizer, numbers, backend)
        structure = compute_geodesic_structure(embeddings, backend)

        all_structures[name] = structure

        logger.info(f"  5/6 ratio: {structure['ratio_5_to_6_from_34']:.4f}")
        logger.info(f"  Triangle excess (3,4,5): {structure['triangle_excess_345']:.4f}")
        logger.info(f"  Triangle excess (9,16,25): {structure['triangle_excess_sq']:.4f}")

        # Free memory
        del model, tokenizer, embeddings
        gc.collect()

    # Compare all pairs
    logger.info("\n" + "=" * 70)
    logger.info("PAIRWISE COMPARISONS")
    logger.info("=" * 70)

    model_names = list(all_structures.keys())
    comparisons = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            comparison = compare_structures(all_structures[name_a], all_structures[name_b])

            logger.info(f"\n{name_a} vs {name_b}:")
            logger.info(f"  Ratio similarity: {comparison['ratio_similarity']:.4f}")
            logger.info(f"  Excess similarity: {comparison['excess_similarity']:.4f}")
            logger.info(f"  Overall similarity: {comparison['overall_similarity']:.4f}")

            comparisons.append({
                "model_a": name_a,
                "model_b": name_b,
                **comparison
            })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INVARIANCE SUMMARY")
    logger.info("=" * 70)

    if comparisons:
        avg_similarity = sum(c["overall_similarity"] for c in comparisons) / len(comparisons)
        avg_ratio_sim = sum(c["ratio_similarity"] for c in comparisons) / len(comparisons)

        logger.info(f"\nAverage overall similarity: {avg_similarity:.4f}")
        logger.info(f"Average ratio similarity: {avg_ratio_sim:.4f}")

        if avg_similarity > 0.7:
            logger.info("\n>>> GEODESIC STRUCTURE IS INVARIANT ACROSS MODELS")
        elif avg_similarity > 0.5:
            logger.info("\n>>> PARTIAL INVARIANCE: Some geodesic features are consistent")
        else:
            logger.info("\n>>> LIMITED INVARIANCE: Geodesic structures vary significantly")

    # Save results
    results = {
        "structures": all_structures,
        "comparisons": comparisons,
        "summary": {
            "avg_overall_similarity": avg_similarity if comparisons else 0,
            "avg_ratio_similarity": avg_ratio_sim if comparisons else 0,
        }
    }

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
