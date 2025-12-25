#!/usr/bin/env python3
"""
Riemannian Pythagorean Test

Uses ModelCypher's proper geometric infrastructure:
- ConceptVolume: numbers as probability clouds, not points
- Curvature estimation: high-D space is curved
- Geodesic distances: respects manifold geometry
- Intrinsic dimension: find the true dimensionality

This is the correct way to test if a² + b² = c² is encoded geometrically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3] / "ModelCypher" / "src"))

from modelcypher.core.domain.geometry.riemannian_density import (
    RiemannianDensityEstimator,
    RiemannianDensityConfig,
    InfluenceType,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    SectionalCurvatureEstimator,
    CurvatureConfig,
)
from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
    BootstrapConfiguration,
)

SCRIPT_DIR = Path(__file__).parent.parent


def load_model(model_path: str):
    from mlx_lm import load
    return load(model_path)


def extract_embedding(model, tokenizer, text: str) -> np.ndarray:
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model

    if hasattr(inner, 'embed_tokens'):
        x = inner.embed_tokens(input_ids)
    else:
        x = inner.wte(input_ids)

    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    for layer in layers:
        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

    result = x[0, -1, :]
    mx.eval(result)
    result = result.astype(mx.float32)
    mx.eval(result)
    return np.array(result, dtype=np.float32)


def get_number_activations(model, tokenizer, n: int, num_variations: int = 10) -> np.ndarray:
    """Get multiple activations for a number to form a concept cloud.

    Uses different phrasings to capture the concept's distribution.
    """
    templates = [
        f"The number {n}.",
        f"The value {n}.",
        f"Consider {n}.",
        f"{n} is a number.",
        f"When we have {n} items.",
        f"The quantity {n}.",
        f"There are {n} of them.",
        f"We count to {n}.",
        f"The result is {n}.",
        f"Exactly {n}.",
        f"{n} appears here.",
        f"The integer {n}.",
    ]

    activations = []
    for i in range(min(num_variations, len(templates))):
        emb = extract_embedding(model, tokenizer, templates[i])
        activations.append(emb)

    return np.stack(activations)


def test_concept_volumes(model, tokenizer) -> dict:
    """Test Pythagorean structure using proper concept volumes."""

    estimator = RiemannianDensityEstimator(RiemannianDensityConfig(
        influence_type=InfluenceType.GAUSSIAN,
        use_curvature_correction=True,
        k_neighbors=8,
    ))

    # Get concept volumes for key numbers
    numbers = [3, 4, 5, 6, 9, 16, 25]
    concept_volumes = {}

    print("Extracting concept volumes (probability clouds)...")
    for n in numbers:
        activations = get_number_activations(model, tokenizer, n, num_variations=10)
        volume = estimator.estimate_concept_volume(f"number_{n}", activations)
        concept_volumes[n] = volume
        print(f"  {n}: geodesic_radius={volume.geodesic_radius:.4f}, "
              f"effective_radius={volume.effective_radius:.4f}")

    # Test 1: Geodesic distances between Pythagorean-related numbers
    print("\n--- Geodesic Distance Analysis ---")

    # Valid triple: 3² + 4² = 5²
    geodist_3_to_5 = concept_volumes[3].geodesic_distance(concept_volumes[5].centroid)
    geodist_4_to_5 = concept_volumes[4].geodesic_distance(concept_volumes[5].centroid)
    geodist_3_to_4 = concept_volumes[3].geodesic_distance(concept_volumes[4].centroid)

    # Invalid: 3² + 4² ≠ 6²
    geodist_3_to_6 = concept_volumes[3].geodesic_distance(concept_volumes[6].centroid)
    geodist_4_to_6 = concept_volumes[4].geodesic_distance(concept_volumes[6].centroid)

    print(f"  3 → 5: {geodist_3_to_5:.4f}")
    print(f"  4 → 5: {geodist_4_to_5:.4f}")
    print(f"  3 → 4: {geodist_3_to_4:.4f}")
    print(f"  3 → 6: {geodist_3_to_6:.4f}")
    print(f"  4 → 6: {geodist_4_to_6:.4f}")

    # Test 2: Volume overlap analysis
    print("\n--- Volume Overlap Analysis ---")

    # Pythagorean relationship should show specific overlap pattern
    relations = {}
    for pair in [(3, 5), (4, 5), (3, 6), (4, 6), (9, 25), (16, 25)]:
        a, b = pair
        relation = estimator.compute_relation(concept_volumes[a], concept_volumes[b])
        relations[pair] = {
            "bhattacharyya": relation.bhattacharyya_coefficient,
            "geodesic_dist": relation.geodesic_centroid_distance,
            "subspace_align": relation.subspace_alignment,
            "curvature_div": relation.curvature_divergence,
        }
        print(f"  {a} ↔ {b}: Bhattacharyya={relation.bhattacharyya_coefficient:.4f}, "
              f"subspace_align={relation.subspace_alignment:.4f}")

    # Test 3: Curvature analysis at key points
    print("\n--- Local Curvature Analysis ---")

    curvature_estimator = SectionalCurvatureEstimator(CurvatureConfig(
        num_directions=20,
        use_parallel_transport=True,
    ))

    curvature_info = {}
    for n, vol in concept_volumes.items():
        if vol.local_curvature:
            K = vol.local_curvature.mean_sectional
            sign = vol.local_curvature.sign.value
            curvature_info[n] = {"mean_K": K, "sign": sign}
            print(f"  {n}: K={K:.6f} ({sign})")

    # Test 4: Check if squared numbers form a submanifold
    print("\n--- Squared Numbers Submanifold Analysis ---")

    squared_centroids = np.stack([concept_volumes[n].centroid for n in [9, 16, 25]])

    # Estimate intrinsic dimension of squared numbers
    try:
        id_estimate = IntrinsicDimensionEstimator.estimate_two_nn(
            squared_centroids.tolist(),
            TwoNNConfiguration(
                use_regression=True,
                bootstrap=BootstrapConfiguration(resamples=100, confidence_level=0.95)
            )
        )
        print(f"  Squared numbers intrinsic dim: {id_estimate.intrinsic_dimension:.2f}")
        if id_estimate.ci:
            print(f"  95% CI: [{id_estimate.ci.lower:.2f}, {id_estimate.ci.upper:.2f}]")
    except Exception as e:
        print(f"  Could not estimate intrinsic dimension: {e}")
        id_estimate = None

    # Test 5: Pythagorean constraint on probability clouds
    print("\n--- Pythagorean Constraint Test (Volume-based) ---")

    # The key insight: in the correct formulation, the probability masses
    # should satisfy geometric constraints

    # Get the probability that 5's cloud overlaps the "a² + b² manifold"
    # defined by the relationship between 3 and 4

    # Vector from 3 to 5 vs vector from 4 to 5
    v_3_5 = concept_volumes[5].centroid - concept_volumes[3].centroid
    v_4_5 = concept_volumes[5].centroid - concept_volumes[4].centroid
    v_3_4 = concept_volumes[4].centroid - concept_volumes[3].centroid

    # vs 6
    v_3_6 = concept_volumes[6].centroid - concept_volumes[3].centroid
    v_4_6 = concept_volumes[6].centroid - concept_volumes[4].centroid

    # Project 5 onto the 3-4 line and measure residual
    t_5 = np.dot(v_3_5, v_3_4) / (np.dot(v_3_4, v_3_4) + 1e-10)
    proj_5 = concept_volumes[3].centroid + t_5 * v_3_4
    residual_5 = np.linalg.norm(concept_volumes[5].centroid - proj_5)

    # Same for 6
    t_6 = np.dot(v_3_6, v_3_4) / (np.dot(v_3_4, v_3_4) + 1e-10)
    proj_6 = concept_volumes[3].centroid + t_6 * v_3_4
    residual_6 = np.linalg.norm(concept_volumes[6].centroid - proj_6)

    print(f"  5's residual from (3,4) line: {residual_5:.4f}")
    print(f"  6's residual from (3,4) line: {residual_6:.4f}")
    print(f"  5's projection parameter t: {t_5:.4f}")
    print(f"  6's projection parameter t: {t_6:.4f}")

    # 5 should be at a specific geometric position based on 3² + 4² = 5²
    # In the ideal case, t ≈ 0.6 (weighted by 4/(3+4))
    ideal_t = 4.0 / 7.0  # based on 3,4 relationship
    t_5_error = abs(t_5 - ideal_t)

    print(f"  Ideal t (from Pythagorean): {ideal_t:.4f}")
    print(f"  5's t error from ideal: {t_5_error:.4f}")

    # Test 6: Mahalanobis distance (accounts for cloud shape)
    print("\n--- Mahalanobis Distance Analysis ---")

    mahal_5_from_3 = concept_volumes[3].mahalanobis_distance(concept_volumes[5].centroid)
    mahal_5_from_4 = concept_volumes[4].mahalanobis_distance(concept_volumes[5].centroid)
    mahal_6_from_3 = concept_volumes[3].mahalanobis_distance(concept_volumes[6].centroid)
    mahal_6_from_4 = concept_volumes[4].mahalanobis_distance(concept_volumes[6].centroid)

    print(f"  Mahal(3 → 5): {mahal_5_from_3:.4f}")
    print(f"  Mahal(4 → 5): {mahal_5_from_4:.4f}")
    print(f"  Mahal(3 → 6): {mahal_6_from_3:.4f}")
    print(f"  Mahal(4 → 6): {mahal_6_from_4:.4f}")

    # Key metric: sum of Mahal distances
    mahal_sum_5 = mahal_5_from_3 + mahal_5_from_4
    mahal_sum_6 = mahal_6_from_3 + mahal_6_from_4

    print(f"  Sum Mahal for 5: {mahal_sum_5:.4f}")
    print(f"  Sum Mahal for 6: {mahal_sum_6:.4f}")
    print(f"  5 is {'CLOSER' if mahal_sum_5 < mahal_sum_6 else 'FARTHER'} to (3,4) than 6")

    # Compile results
    results = {
        "geodesic_distances": {
            "3_to_5": float(geodist_3_to_5),
            "4_to_5": float(geodist_4_to_5),
            "3_to_4": float(geodist_3_to_4),
            "3_to_6": float(geodist_3_to_6),
            "4_to_6": float(geodist_4_to_6),
        },
        "relations": {str(k): v for k, v in relations.items()},
        "curvature": curvature_info,
        "projection_analysis": {
            "5_residual": float(residual_5),
            "6_residual": float(residual_6),
            "5_t": float(t_5),
            "6_t": float(t_6),
            "ideal_t": float(ideal_t),
            "5_t_error": float(t_5_error),
        },
        "mahalanobis": {
            "5_from_3": float(mahal_5_from_3),
            "5_from_4": float(mahal_5_from_4),
            "6_from_3": float(mahal_6_from_3),
            "6_from_4": float(mahal_6_from_4),
            "sum_5": float(mahal_sum_5),
            "sum_6": float(mahal_sum_6),
            "5_closer_than_6": bool(mahal_sum_5 < mahal_sum_6),
        },
    }

    if id_estimate:
        results["intrinsic_dimension"] = {
            "squared_numbers": id_estimate.intrinsic_dimension,
            "ci_lower": id_estimate.ci.lower if id_estimate.ci else None,
            "ci_upper": id_estimate.ci.upper if id_estimate.ci else None,
        }

    return results


def test_full_number_manifold(model, tokenizer) -> dict:
    """Estimate intrinsic dimension of the number manifold."""

    print("\n--- Full Number Manifold Analysis ---")

    # Get concept clouds for 1-25
    numbers = list(range(1, 26))
    all_centroids = []

    for n in numbers:
        activations = get_number_activations(model, tokenizer, n, num_variations=5)
        centroid = np.mean(activations, axis=0)
        all_centroids.append(centroid)

    centroids_matrix = np.stack(all_centroids)

    # Estimate intrinsic dimension
    try:
        id_estimate = IntrinsicDimensionEstimator.estimate_two_nn(
            centroids_matrix.tolist(),
            TwoNNConfiguration(
                use_regression=True,
                bootstrap=BootstrapConfiguration(resamples=200, confidence_level=0.95)
            )
        )
        print(f"  Numbers 1-25 intrinsic dimension: {id_estimate.intrinsic_dimension:.2f}")
        if id_estimate.ci:
            print(f"  95% CI: [{id_estimate.ci.lower:.2f}, {id_estimate.ci.upper:.2f}]")
    except Exception as e:
        print(f"  Could not estimate: {e}")
        id_estimate = None

    # Estimate global curvature
    curvature_estimator = SectionalCurvatureEstimator(CurvatureConfig(
        num_directions=30,
    ))

    profile = curvature_estimator.estimate_manifold_profile(
        centroids_matrix,
        k_neighbors=min(15, len(numbers) - 1)
    )

    print(f"  Global mean curvature: {profile.global_mean:.6f}")
    print(f"  Curvature variance: {profile.global_variance:.6f}")
    print(f"  Dominant sign: {profile.dominant_sign.value}")

    return {
        "intrinsic_dimension": id_estimate.intrinsic_dimension if id_estimate else None,
        "ci_lower": id_estimate.ci.lower if id_estimate and id_estimate.ci else None,
        "ci_upper": id_estimate.ci.upper if id_estimate and id_estimate.ci else None,
        "global_mean_curvature": float(profile.global_mean),
        "curvature_variance": float(profile.global_variance),
        "dominant_sign": profile.dominant_sign.value,
    }


def convert_for_json(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    model_name = Path(args.model).name
    output_path = args.output or str(SCRIPT_DIR / "results" / f"riemannian_{model_name}.json")

    print("=" * 70)
    print("RIEMANNIAN PYTHAGOREAN TEST")
    print("Using proper geometric infrastructure:")
    print("  - Concepts as probability clouds (ConceptVolume)")
    print("  - Curvature-aware distances (geodesic)")
    print("  - Shape-aware distances (Mahalanobis)")
    print(f"Model: {model_name}")
    print("=" * 70)

    model, tokenizer = load_model(args.model)

    results = {
        "model": model_name,
        "pythagorean": test_concept_volumes(model, tokenizer),
        "manifold": test_full_number_manifold(model, tokenizer),
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mahal = results["pythagorean"]["mahalanobis"]
    print(f"5 closer to (3,4) than 6 (Mahalanobis): {mahal['5_closer_than_6']}")

    proj = results["pythagorean"]["projection_analysis"]
    print(f"5's projection error from ideal: {proj['5_t_error']:.4f}")

    if results["manifold"]["intrinsic_dimension"]:
        print(f"Number manifold intrinsic dimension: {results['manifold']['intrinsic_dimension']:.2f}")

    print(f"Manifold curvature sign: {results['manifold']['dominant_sign']}")

    # Verdict
    if mahal['5_closer_than_6'] and proj['5_t_error'] < 0.3:
        print("\n>>> RIEMANNIAN STRUCTURE DETECTED: Pythagorean relationship encoded!")
    elif mahal['5_closer_than_6']:
        print("\n>>> PARTIAL: Proximity structure exists but position not precisely constrained")
    else:
        print("\n>>> NOT DETECTED: No clear geometric encoding of Pythagorean relationship")

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
