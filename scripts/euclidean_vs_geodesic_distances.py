#!/usr/bin/env python3
"""Experiment 28: Euclidean vs Geodesic Distance Comparison.

Quantify the difference between Euclidean (chord) and geodesic distances
on real model activations to understand where linear assumptions break down.

Key metrics:
1. Correlation between chord and geodesic distances
2. Ratio distribution (geodesic / chord)
3. Max deviation normalized by mean chord distance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_chord_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    n = points.shape[0]
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = points[i] - points
        dists[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return dists


def compute_geodesic_distance_matrix(
    points: np.ndarray,
    k_neighbors: int = 10,
) -> np.ndarray:
    """Compute geodesic distances via k-NN graph shortest paths."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n = points.shape[0]

    # Build k-NN graph
    chord_dists = compute_chord_distance_matrix(points)

    # Create sparse adjacency matrix with k nearest neighbors
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]  # Skip self
        for j in neighbors:
            adjacency[i, j] = chord_dists[i, j]
            adjacency[j, i] = chord_dists[j, i]  # Symmetric

    # Shortest paths give geodesic distances
    sparse_adj = csr_matrix(adjacency)
    geo_dists = shortest_path(sparse_adj, directed=False)

    # Handle disconnected components (inf -> max finite * 2)
    finite_mask = np.isfinite(geo_dists)
    if not np.all(finite_mask):
        max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
        geo_dists[~finite_mask] = max_finite * 2

    return geo_dists.astype(np.float32)


def analyze_distance_comparison(
    chord_dists: np.ndarray,
    geo_dists: np.ndarray,
) -> Dict:
    """Analyze the relationship between chord and geodesic distances."""
    # Flatten upper triangle (excluding diagonal)
    n = chord_dists.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    chord_flat = chord_dists[triu_indices]
    geo_flat = geo_dists[triu_indices]

    # Filter out any inf/nan
    valid = np.isfinite(chord_flat) & np.isfinite(geo_flat) & (chord_flat > 1e-10)
    chord_valid = chord_flat[valid]
    geo_valid = geo_flat[valid]

    if len(chord_valid) < 10:
        return {"error": "too_few_valid_pairs", "n_valid": len(chord_valid)}

    # 1. Correlation
    correlation = float(np.corrcoef(chord_valid, geo_valid)[0, 1])

    # 2. Ratio distribution
    ratios = geo_valid / chord_valid
    ratio_stats = {
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "median": float(np.median(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "p5": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
    }

    # 3. Max deviation
    abs_diff = np.abs(geo_valid - chord_valid)
    mean_chord = np.mean(chord_valid)
    max_deviation = float(np.max(abs_diff) / mean_chord)
    mean_deviation = float(np.mean(abs_diff) / mean_chord)

    # 4. Where does geodesic differ most?
    # Find pairs with largest ratio deviation from 1.0
    ratio_deviation = np.abs(ratios - 1.0)
    top_deviations_idx = np.argsort(ratio_deviation)[-10:]
    top_deviations = [
        {
            "chord": float(chord_valid[i]),
            "geodesic": float(geo_valid[i]),
            "ratio": float(ratios[i]),
        }
        for i in top_deviations_idx
    ]

    return {
        "correlation": correlation,
        "ratio_stats": ratio_stats,
        "max_deviation_normalized": max_deviation,
        "mean_deviation_normalized": mean_deviation,
        "n_pairs": len(chord_valid),
        "top_deviations": top_deviations,
    }


class EuclideanGeodesicComparison:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_layer_activations(
        self,
        prompts: List[str],
        layer_idx: int,
    ) -> np.ndarray:
        """Get activations from a specific layer for given prompts."""
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Forward pass with hook to capture activations
            hidden = self.model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i, layer in enumerate(self.model.model.layers):
                hidden = layer(hidden)
                mx.eval(hidden)
                if i == layer_idx:
                    # Take last token's hidden state
                    act = hidden[0, -1, :]
                    mx.eval(act)
                    activations.append(np.array(act.tolist(), dtype=np.float32))
                    break

        return np.array(activations)

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 28: EUCLIDEAN VS GEODESIC DISTANCES")
        logger.info("=" * 60)

        # Test prompts covering different semantic categories
        test_prompts = [
            # Math
            "What is 2 + 2?",
            "Calculate 15 times 7.",
            "What is the square root of 144?",
            "Solve for x: 2x + 5 = 11",
            "What is 100 divided by 4?",
            # Geography
            "What is the capital of France?",
            "Which continent is Brazil in?",
            "What is the largest ocean?",
            "Name a country in Africa.",
            "What river flows through Egypt?",
            # Science
            "What is H2O?",
            "How many planets are in our solar system?",
            "What gas do plants produce?",
            "What is the speed of light?",
            "What is photosynthesis?",
            # Language
            "What is the opposite of hot?",
            "Define the word 'serendipity'.",
            "What is a synonym for happy?",
            "What is the plural of child?",
            "Name a verb in English.",
            # Logic
            "If all cats are animals, is a cat an animal?",
            "What comes next: 2, 4, 6, 8, ?",
            "Is a square a rectangle?",
            "If today is Monday, what day is tomorrow?",
            "True or false: All birds can fly.",
        ]

        results = {
            "n_prompts": len(test_prompts),
            "n_layers": self.n_layers,
            "by_layer": {},
            "summary": {},
        }

        # Test on early, middle, and late layers
        test_layers = [
            0,  # Early
            self.n_layers // 4,
            self.n_layers // 2,  # Middle
            3 * self.n_layers // 4,
            self.n_layers - 1,  # Late
        ]

        all_correlations = []
        all_mean_ratios = []
        all_max_deviations = []

        for layer_idx in test_layers:
            logger.info(f"\nAnalyzing layer {layer_idx}...")

            # Get activations
            activations = self.get_layer_activations(test_prompts, layer_idx)
            logger.info(f"  Activations shape: {activations.shape}")

            # Compute distances
            chord_dists = compute_chord_distance_matrix(activations)
            geo_dists = compute_geodesic_distance_matrix(activations, k_neighbors=5)

            # Analyze
            analysis = analyze_distance_comparison(chord_dists, geo_dists)

            if "error" not in analysis:
                logger.info(f"  Correlation: {analysis['correlation']:.4f}")
                logger.info(f"  Mean ratio (geo/chord): {analysis['ratio_stats']['mean']:.4f}")
                logger.info(f"  Max deviation (normalized): {analysis['max_deviation_normalized']:.4f}")

                all_correlations.append(analysis["correlation"])
                all_mean_ratios.append(analysis["ratio_stats"]["mean"])
                all_max_deviations.append(analysis["max_deviation_normalized"])

            results["by_layer"][str(layer_idx)] = analysis

        # Summary statistics
        if all_correlations:
            results["summary"] = {
                "mean_correlation": float(np.mean(all_correlations)),
                "std_correlation": float(np.std(all_correlations)),
                "mean_ratio": float(np.mean(all_mean_ratios)),
                "mean_max_deviation": float(np.mean(all_max_deviations)),
            }

            logger.info("\n" + "=" * 60)
            logger.info("SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Mean correlation across layers: {results['summary']['mean_correlation']:.4f}")
            logger.info(f"Mean geodesic/chord ratio: {results['summary']['mean_ratio']:.4f}")
            logger.info(f"Mean max deviation: {results['summary']['mean_max_deviation']:.4f}")

            # Interpretation
            if results["summary"]["mean_correlation"] > 0.99:
                logger.info("\nINTERPRETATION: Distances are highly correlated - manifold is nearly flat")
            elif results["summary"]["mean_correlation"] > 0.95:
                logger.info("\nINTERPRETATION: Moderate curvature detected - geodesic provides some additional structure")
            else:
                logger.info("\nINTERPRETATION: Significant curvature - geodesic distances capture non-linear structure")

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = EuclideanGeodesicComparison(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/euclidean_vs_geodesic_distances.json"
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
