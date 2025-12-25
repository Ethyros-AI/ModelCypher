#!/usr/bin/env python3
"""
Pythagorean Probe using ModelCypher Infrastructure

Uses proper GPU-accelerated extraction via LocalInferenceEngine
and RiemannianDensityEstimator for curvature-aware analysis.

Fast version: extracts once, analyzes offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3] / "ModelCypher" / "src"))

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.core.domain.entropy.hidden_state_extractor import ExtractorConfig
from modelcypher.core.domain.geometry.riemannian_density import (
    RiemannianDensityEstimator,
    RiemannianDensityConfig,
    InfluenceType,
)
from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
    BootstrapConfiguration,
)

SCRIPT_DIR = Path(__file__).parent.parent


def extract_number_activations(
    engine: LocalInferenceEngine,
    model_path: str,
    numbers: list[int],
    num_variations: int = 5,
) -> dict[int, np.ndarray]:
    """Extract activations for numbers using LocalInferenceEngine.

    Uses capture_hidden_states which returns {layer_idx: activation_list}.
    We extract from the last layer only for efficiency.
    """

    templates = [
        "The number {n}.",
        "The value {n}.",
        "Consider {n}.",
        "{n} is a number.",
        "There are {n} items.",
    ]

    activations = {}

    # First, figure out how many layers there are (by doing one extraction)
    test_states = engine.capture_hidden_states(model_path, "test", target_layers=None)
    if not test_states:
        raise RuntimeError("Could not extract hidden states from model")
    last_layer = max(test_states.keys())
    print(f"  Model has {last_layer + 1} layers, extracting from layer {last_layer}")

    for n in numbers:
        number_activations = []
        for i in range(min(num_variations, len(templates))):
            prompt = templates[i].format(n=n)

            try:
                # Only capture the last layer for efficiency
                hidden_states = engine.capture_hidden_states(
                    model_path,
                    prompt,
                    target_layers={last_layer},
                )

                if hidden_states and last_layer in hidden_states:
                    activation = np.array(hidden_states[last_layer], dtype=np.float32)
                    number_activations.append(activation)
            except Exception as e:
                print(f"  Warning: Failed to extract for '{prompt}': {e}")
                continue

        if number_activations:
            activations[n] = np.stack(number_activations)
            print(f"  {n}: {len(number_activations)} activations, shape={activations[n].shape}")

    return activations


def analyze_pythagorean_structure(activations: dict[int, np.ndarray]) -> dict:
    """Analyze Pythagorean structure from pre-extracted activations."""

    results = {}

    # 1. Compute centroids
    centroids = {n: np.mean(acts, axis=0) for n, acts in activations.items()}

    # 2. Test: Is 5 closer to (3,4) centroid than 6?
    centroid_34 = (centroids[3] + centroids[4]) / 2
    dist_5 = np.linalg.norm(centroids[5] - centroid_34)
    dist_6 = np.linalg.norm(centroids[6] - centroid_34)

    results["position_test"] = {
        "dist_5_to_34": float(dist_5),
        "dist_6_to_34": float(dist_6),
        "5_closer_than_6": bool(dist_5 < dist_6),
    }
    print(f"\nPosition test: 5→(3,4)={dist_5:.4f}, 6→(3,4)={dist_6:.4f}")
    print(f"  5 {'CLOSER' if dist_5 < dist_6 else 'FARTHER'} than 6")

    # 3. Riemannian analysis (without expensive curvature)
    estimator = RiemannianDensityEstimator(RiemannianDensityConfig(
        influence_type=InfluenceType.GAUSSIAN,
        use_curvature_correction=False,  # Skip expensive curvature for speed
        k_neighbors=min(5, min(len(v) for v in activations.values()) - 1),
    ))

    # Create concept volumes
    volumes = {}
    for n, acts in activations.items():
        if len(acts) >= 2:
            volumes[n] = estimator.estimate_concept_volume(f"number_{n}", acts)

    # 4. Mahalanobis distances (shape-aware)
    if 3 in volumes and 4 in volumes and 5 in volumes and 6 in volumes:
        mahal_5_from_3 = volumes[3].mahalanobis_distance(volumes[5].centroid)
        mahal_5_from_4 = volumes[4].mahalanobis_distance(volumes[5].centroid)
        mahal_6_from_3 = volumes[3].mahalanobis_distance(volumes[6].centroid)
        mahal_6_from_4 = volumes[4].mahalanobis_distance(volumes[6].centroid)

        sum_5 = mahal_5_from_3 + mahal_5_from_4
        sum_6 = mahal_6_from_3 + mahal_6_from_4

        results["mahalanobis"] = {
            "5_from_3": float(mahal_5_from_3),
            "5_from_4": float(mahal_5_from_4),
            "6_from_3": float(mahal_6_from_3),
            "6_from_4": float(mahal_6_from_4),
            "sum_5": float(sum_5),
            "sum_6": float(sum_6),
            "5_closer_mahal": bool(sum_5 < sum_6),
        }
        print(f"\nMahalanobis: sum(5)={sum_5:.4f}, sum(6)={sum_6:.4f}")
        print(f"  5 {'CLOSER' if sum_5 < sum_6 else 'FARTHER'} (shape-aware)")

    # 5. Triangle structure (9, 16, 25)
    if all(n in centroids for n in [9, 16, 25]):
        d_9_16 = np.linalg.norm(centroids[9] - centroids[16])
        d_16_25 = np.linalg.norm(centroids[16] - centroids[25])
        d_9_25 = np.linalg.norm(centroids[9] - centroids[25])

        # Normalized ratios
        total = d_9_16 + d_16_25 + d_9_25
        results["squared_triangle"] = {
            "d_9_16": float(d_9_16),
            "d_16_25": float(d_16_25),
            "d_9_25": float(d_9_25),
            "ratios": [d_9_16/total, d_16_25/total, d_9_25/total],
        }
        print(f"\nSquared triangle (9,16,25): {d_9_16:.4f}, {d_16_25:.4f}, {d_9_25:.4f}")

    # 6. Intrinsic dimension
    if len(centroids) >= 10:
        centroid_matrix = np.stack([centroids[n] for n in sorted(centroids.keys())])
        try:
            id_est = IntrinsicDimensionEstimator.estimate_two_nn(
                centroid_matrix.tolist(),
                TwoNNConfiguration(use_regression=True)
            )
            results["intrinsic_dimension"] = float(id_est.intrinsic_dimension)
            print(f"\nIntrinsic dimension: {id_est.intrinsic_dimension:.2f}")
        except Exception as e:
            print(f"\nCould not estimate intrinsic dimension: {e}")

    # 7. Bhattacharyya overlap (for volumes)
    if 3 in volumes and 5 in volumes and 6 in volumes:
        rel_3_5 = estimator.compute_relation(volumes[3], volumes[5])
        rel_3_6 = estimator.compute_relation(volumes[3], volumes[6])

        results["overlap"] = {
            "bhattacharyya_3_5": float(rel_3_5.bhattacharyya_coefficient),
            "bhattacharyya_3_6": float(rel_3_6.bhattacharyya_coefficient),
            "subspace_align_3_5": float(rel_3_5.subspace_alignment),
            "subspace_align_3_6": float(rel_3_6.subspace_alignment),
        }
        print(f"\nVolume overlap: 3↔5={rel_3_5.bhattacharyya_coefficient:.4f}, "
              f"3↔6={rel_3_6.bhattacharyya_coefficient:.4f}")

    return results


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
    output_path = args.output or str(SCRIPT_DIR / "results" / f"probe_{model_name}.json")

    print("=" * 70)
    print("PYTHAGOREAN PROBE (ModelCypher Infrastructure)")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Initialize engine
    engine = LocalInferenceEngine()

    # Numbers to test
    numbers = [3, 4, 5, 6, 9, 12, 13, 16, 25]

    print("\nExtracting activations...")
    activations = extract_number_activations(engine, args.model, numbers, num_variations=5)

    print("\nAnalyzing structure...")
    results = analyze_pythagorean_structure(activations)
    results["model"] = model_name

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tests_passed = 0
    if results.get("position_test", {}).get("5_closer_than_6"):
        print("✓ Position test: 5 closer to (3,4) than 6")
        tests_passed += 1
    else:
        print("✗ Position test: FAILED")

    if results.get("mahalanobis", {}).get("5_closer_mahal"):
        print("✓ Mahalanobis test: 5 closer (shape-aware)")
        tests_passed += 1
    else:
        print("✗ Mahalanobis test: FAILED")

    print(f"\nTests passed: {tests_passed}/2")

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
