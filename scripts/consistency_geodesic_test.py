#!/usr/bin/env python3
"""Experiment 31: ConsistencyMeasure - Geodesic vs Euclidean.

Compare Euclidean cosine distance (current) vs geodesic distance
for semantic consistency measurement.

Key question: Does geodesic achieve higher effect size
(better separation of implications vs contradictions)?
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Test cases: (statement, implications, contradictions)
TEST_CASES = [
    {
        "statement": "The capital of France is Paris.",
        "implications": [
            "Paris is a capital city.",
            "France has a capital.",
            "Paris is in France.",
        ],
        "contradictions": [
            "The capital of France is London.",
            "Paris is not a capital.",
            "France has no capital city.",
        ],
    },
    {
        "statement": "Water freezes at 0 degrees Celsius.",
        "implications": [
            "Water can turn into ice.",
            "Temperature affects water's state.",
            "0 degrees is the freezing point.",
        ],
        "contradictions": [
            "Water never freezes.",
            "Water freezes at 100 degrees.",
            "Temperature has no effect on water.",
        ],
    },
    {
        "statement": "2 + 2 equals 4.",
        "implications": [
            "4 is the sum of 2 and 2.",
            "2 plus 2 is four.",
            "Adding 2 twice gives 4.",
        ],
        "contradictions": [
            "2 + 2 equals 5.",
            "2 plus 2 is three.",
            "Adding numbers doesn't work.",
        ],
    },
    {
        "statement": "The sun rises in the east.",
        "implications": [
            "The sun appears from the eastern horizon.",
            "East is where sunrise occurs.",
            "Morning light comes from the east.",
        ],
        "contradictions": [
            "The sun rises in the west.",
            "The sun never rises.",
            "East and west are the same direction.",
        ],
    },
    {
        "statement": "Cats are mammals.",
        "implications": [
            "Cats are warm-blooded animals.",
            "Cats give birth to live young.",
            "Cats belong to the mammal class.",
        ],
        "contradictions": [
            "Cats are reptiles.",
            "Cats lay eggs.",
            "Cats are not animals.",
        ],
    },
]


@dataclass
class ConsistencyResult:
    implication_consistency: float
    contradiction_distance: float
    consistency_score: float
    knowledge_confidence: float  # Effect size
    n_implications: int
    n_contradictions: int


def euclidean_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine_similarity) in Euclidean space."""
    a_flat = a.flatten()
    b_flat = b.flatten()

    a_norm = np.linalg.norm(a_flat)
    b_norm = np.linalg.norm(b_flat)

    if a_norm < 1e-10 or b_norm < 1e-10:
        return 1.0

    similarity = np.dot(a_flat, b_flat) / (a_norm * b_norm)
    return 1.0 - similarity


def geodesic_distance(a: np.ndarray, b: np.ndarray, all_points: np.ndarray, k_neighbors: int = 5) -> float:
    """Compute geodesic distance between a and b using all_points as manifold samples."""
    n = all_points.shape[0]

    # Chord distances
    chord_dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = all_points[i] - all_points
        chord_dists[i] = np.sqrt(np.sum(diff * diff, axis=1))

    # Build k-NN graph
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
        for j in neighbors:
            adjacency[i, j] = chord_dists[i, j]
            adjacency[j, i] = chord_dists[j, i]

    # Geodesic distances via shortest path
    sparse_adj = csr_matrix(adjacency)
    geo_dists = shortest_path(sparse_adj, directed=False)

    # Handle inf
    finite_mask = np.isfinite(geo_dists)
    if not np.all(finite_mask):
        max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
        geo_dists[~finite_mask] = max_finite * 2

    # Find indices of a and b (they should be first two points)
    return float(geo_dists[0, 1])


def compute_consistency_euclidean(
    original: np.ndarray,
    implications: List[np.ndarray],
    contradictions: List[np.ndarray],
) -> ConsistencyResult:
    """Compute consistency using Euclidean cosine distance."""
    # Distances to implications
    impl_distances = [euclidean_cosine_distance(original, impl) for impl in implications]
    avg_impl_dist = sum(impl_distances) / len(impl_distances)
    implication_consistency = 1.0 - min(1.0, avg_impl_dist)

    # Distances to contradictions
    contra_distances = [euclidean_cosine_distance(original, contra) for contra in contradictions]
    avg_contra_dist = sum(contra_distances) / len(contra_distances)
    contradiction_distance = avg_contra_dist

    # Consistency score
    consistency_score = implication_consistency * min(1.0, contradiction_distance)

    # Effect size (Cohen's d)
    all_dists = impl_distances + contra_distances
    impl_mean = sum(impl_distances) / len(impl_distances)
    contra_mean = sum(contra_distances) / len(contra_distances)
    variance = sum((d - sum(all_dists)/len(all_dists))**2 for d in all_dists) / len(all_dists)
    std = math.sqrt(variance) if variance > 0 else 1.0
    effect_size = abs(contra_mean - impl_mean) / std if std > 0 else 0.0
    knowledge_confidence = min(1.0, effect_size / 1.5)

    return ConsistencyResult(
        implication_consistency=implication_consistency,
        contradiction_distance=contradiction_distance,
        consistency_score=consistency_score,
        knowledge_confidence=knowledge_confidence,
        n_implications=len(implications),
        n_contradictions=len(contradictions),
    )


def compute_consistency_geodesic(
    original: np.ndarray,
    implications: List[np.ndarray],
    contradictions: List[np.ndarray],
    k_neighbors: int = 3,
) -> ConsistencyResult:
    """Compute consistency using geodesic distance on the manifold."""
    # Stack all points for manifold estimation
    all_points = np.vstack([
        original.reshape(1, -1),
        np.vstack(implications),
        np.vstack(contradictions),
    ])

    n_impl = len(implications)
    n_contra = len(contradictions)

    # Compute chord distances
    n = all_points.shape[0]
    chord_dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = all_points[i] - all_points
        chord_dists[i] = np.sqrt(np.sum(diff * diff, axis=1))

    # Build k-NN graph
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
        for j in neighbors:
            adjacency[i, j] = chord_dists[i, j]
            adjacency[j, i] = chord_dists[j, i]

    # Geodesic distances
    sparse_adj = csr_matrix(adjacency)
    geo_dists = shortest_path(sparse_adj, directed=False)

    # Handle inf
    finite_mask = np.isfinite(geo_dists)
    if not np.all(finite_mask):
        max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
        geo_dists[~finite_mask] = max_finite * 2

    # Distances from original (index 0) to implications (indices 1 to n_impl)
    impl_indices = list(range(1, 1 + n_impl))
    impl_distances = [float(geo_dists[0, i]) for i in impl_indices]

    # Distances from original to contradictions
    contra_indices = list(range(1 + n_impl, 1 + n_impl + n_contra))
    contra_distances = [float(geo_dists[0, i]) for i in contra_indices]

    # Normalize distances to [0, 1] range based on max distance
    max_dist = max(max(impl_distances), max(contra_distances)) if impl_distances and contra_distances else 1.0
    if max_dist > 1e-10:
        impl_distances_norm = [d / max_dist for d in impl_distances]
        contra_distances_norm = [d / max_dist for d in contra_distances]
    else:
        impl_distances_norm = impl_distances
        contra_distances_norm = contra_distances

    avg_impl_dist = sum(impl_distances_norm) / len(impl_distances_norm)
    implication_consistency = 1.0 - min(1.0, avg_impl_dist)

    avg_contra_dist = sum(contra_distances_norm) / len(contra_distances_norm)
    contradiction_distance = avg_contra_dist

    consistency_score = implication_consistency * min(1.0, contradiction_distance)

    # Effect size
    all_dists = impl_distances_norm + contra_distances_norm
    impl_mean = sum(impl_distances_norm) / len(impl_distances_norm)
    contra_mean = sum(contra_distances_norm) / len(contra_distances_norm)
    variance = sum((d - sum(all_dists)/len(all_dists))**2 for d in all_dists) / len(all_dists)
    std = math.sqrt(variance) if variance > 0 else 1.0
    effect_size = abs(contra_mean - impl_mean) / std if std > 0 else 0.0
    knowledge_confidence = min(1.0, effect_size / 1.5)

    return ConsistencyResult(
        implication_consistency=implication_consistency,
        contradiction_distance=contradiction_distance,
        consistency_score=consistency_score,
        knowledge_confidence=knowledge_confidence,
        n_implications=n_impl,
        n_contradictions=n_contra,
    )


class ConsistencyGeodesicTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_representation(self, text: str, layer_idx: int) -> np.ndarray:
        """Get activation representation for text at specified layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        hidden = self.model.model.embed_tokens(input_ids)
        mx.eval(hidden)

        for i, layer in enumerate(self.model.model.layers):
            hidden = layer(hidden)
            mx.eval(hidden)
            if i == layer_idx:
                act = hidden[0, -1, :]
                mx.eval(act)
                return np.array(act.tolist(), dtype=np.float32)

        return np.zeros(1024, dtype=np.float32)

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 31: CONSISTENCY GEODESIC VS EUCLIDEAN")
        logger.info("=" * 60)

        results = {
            "n_test_cases": len(TEST_CASES),
            "n_layers": self.n_layers,
            "by_case": [],
            "by_layer": {},
            "summary": {},
        }

        # Test on middle layer
        layer_idx = self.n_layers // 2
        logger.info(f"\nUsing layer {layer_idx}")

        euclidean_effects = []
        geodesic_effects = []
        euclidean_scores = []
        geodesic_scores = []

        for case_idx, case in enumerate(TEST_CASES):
            logger.info(f"\nCase {case_idx + 1}: {case['statement'][:50]}...")

            # Get representations
            orig_repr = self.get_representation(case["statement"], layer_idx)
            impl_reprs = [self.get_representation(impl, layer_idx) for impl in case["implications"]]
            contra_reprs = [self.get_representation(contra, layer_idx) for contra in case["contradictions"]]

            # Euclidean
            euclidean_result = compute_consistency_euclidean(orig_repr, impl_reprs, contra_reprs)

            # Geodesic
            geodesic_result = compute_consistency_geodesic(orig_repr, impl_reprs, contra_reprs)

            euclidean_effects.append(euclidean_result.knowledge_confidence)
            geodesic_effects.append(geodesic_result.knowledge_confidence)
            euclidean_scores.append(euclidean_result.consistency_score)
            geodesic_scores.append(geodesic_result.consistency_score)

            case_result = {
                "statement": case["statement"],
                "euclidean": {
                    "implication_consistency": euclidean_result.implication_consistency,
                    "contradiction_distance": euclidean_result.contradiction_distance,
                    "consistency_score": euclidean_result.consistency_score,
                    "effect_size": euclidean_result.knowledge_confidence,
                },
                "geodesic": {
                    "implication_consistency": geodesic_result.implication_consistency,
                    "contradiction_distance": geodesic_result.contradiction_distance,
                    "consistency_score": geodesic_result.consistency_score,
                    "effect_size": geodesic_result.effect_size if hasattr(geodesic_result, 'effect_size') else geodesic_result.knowledge_confidence,
                },
                "delta_effect_size": geodesic_result.knowledge_confidence - euclidean_result.knowledge_confidence,
            }
            results["by_case"].append(case_result)

            logger.info(f"  Euclidean: score={euclidean_result.consistency_score:.3f}, effect={euclidean_result.knowledge_confidence:.3f}")
            logger.info(f"  Geodesic:  score={geodesic_result.consistency_score:.3f}, effect={geodesic_result.knowledge_confidence:.3f}")
            logger.info(f"  Delta effect: {case_result['delta_effect_size']:+.3f}")

        # Summary
        results["summary"] = {
            "mean_euclidean_effect": float(np.mean(euclidean_effects)),
            "mean_geodesic_effect": float(np.mean(geodesic_effects)),
            "mean_effect_delta": float(np.mean(geodesic_effects) - np.mean(euclidean_effects)),
            "mean_euclidean_score": float(np.mean(euclidean_scores)),
            "mean_geodesic_score": float(np.mean(geodesic_scores)),
            "n_geodesic_higher_effect": sum(1 for g, e in zip(geodesic_effects, euclidean_effects) if g > e),
            "n_euclidean_higher_effect": sum(1 for g, e in zip(geodesic_effects, euclidean_effects) if e > g),
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Mean Euclidean effect size: {results['summary']['mean_euclidean_effect']:.4f}")
        logger.info(f"Mean Geodesic effect size: {results['summary']['mean_geodesic_effect']:.4f}")
        logger.info(f"Effect size delta (geo - euc): {results['summary']['mean_effect_delta']:+.4f}")
        logger.info(f"Cases where geodesic > euclidean: {results['summary']['n_geodesic_higher_effect']}/{len(TEST_CASES)}")

        # Interpretation
        if results["summary"]["mean_effect_delta"] > 0.05:
            logger.info("\nINTERPRETATION: Geodesic achieves HIGHER effect size - better separation")
            results["conclusion"] = "geodesic_better"
        elif results["summary"]["mean_effect_delta"] < -0.05:
            logger.info("\nINTERPRETATION: Euclidean achieves higher effect size")
            results["conclusion"] = "euclidean_better"
        else:
            logger.info("\nINTERPRETATION: No significant difference between methods")
            results["conclusion"] = "no_difference"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = ConsistencyGeodesicTest(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/consistency_euclidean_vs_geodesic.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
