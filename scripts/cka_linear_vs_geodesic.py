#!/usr/bin/env python3
"""Experiment 30: CKA Comparison (Linear vs Geodesic RBF).

Compare linear CKA (Euclidean dot-product Gram) vs geodesic CKA (RBF over k-NN graph)
to identify where geodesic CKA captures structure that linear CKA misses.
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


# Test prompts for collecting activations
TEST_PROMPTS = [
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


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA using dot-product Gram matrices."""
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    K_X = X_c @ X_c.T
    K_Y = Y_c @ Y_c.T

    # HSIC
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


def compute_geodesic_cka(
    X: np.ndarray,
    Y: np.ndarray,
    k_neighbors: int = 5,
) -> float:
    """Compute geodesic CKA using RBF kernel over k-NN graph distances."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    def geodesic_gram(points: np.ndarray) -> np.ndarray:
        n = points.shape[0]

        # Chord distances
        chord_dists = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = points[i] - points
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

        # RBF kernel with auto sigma
        geo_sq = geo_dists ** 2
        valid_sq = geo_sq[np.triu_indices(n, k=1)]
        sigma = np.median(valid_sq[valid_sq > 1e-10]) if np.any(valid_sq > 1e-10) else 1.0
        sigma = max(sigma, 1e-10)

        K = np.exp(-geo_sq / (2 * sigma))
        return K

    K_X = geodesic_gram(X)
    K_Y = geodesic_gram(Y)

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


class CKAComparison:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_all_layer_activations(
        self,
        prompts: List[str],
    ) -> Dict[int, np.ndarray]:
        """Get activations from all layers for given prompts."""
        import mlx.core as mx

        layer_activations = {i: [] for i in range(self.n_layers)}

        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            hidden = self.model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i, layer in enumerate(self.model.model.layers):
                hidden = layer(hidden)
                mx.eval(hidden)
                act = hidden[0, -1, :]
                mx.eval(act)
                layer_activations[i].append(np.array(act.tolist(), dtype=np.float32))

        return {i: np.array(acts) for i, acts in layer_activations.items()}

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 30: CKA LINEAR VS GEODESIC")
        logger.info("=" * 60)

        # Collect activations
        logger.info(f"\nCollecting activations from {len(TEST_PROMPTS)} prompts...")
        layer_activations = self.get_all_layer_activations(TEST_PROMPTS)

        results = {
            "n_prompts": len(TEST_PROMPTS),
            "n_layers": self.n_layers,
            "layer_pairs": [],
            "summary": {},
        }

        # Compare CKA for layer pairs
        # Focus on adjacent layers and some distant pairs
        layer_pairs = []

        # Adjacent layers
        for i in range(0, self.n_layers - 1, 2):
            layer_pairs.append((i, i + 1))

        # Distant pairs
        layer_pairs.extend([
            (0, self.n_layers // 2),
            (self.n_layers // 4, 3 * self.n_layers // 4),
            (0, self.n_layers - 1),
        ])

        all_linear_ckas = []
        all_geodesic_ckas = []
        all_deltas = []

        logger.info(f"\nComparing CKA for {len(layer_pairs)} layer pairs...")

        for layer_a, layer_b in layer_pairs:
            acts_a = layer_activations[layer_a]
            acts_b = layer_activations[layer_b]

            linear_cka = compute_linear_cka(acts_a, acts_b)
            geodesic_cka = compute_geodesic_cka(acts_a, acts_b)
            delta = geodesic_cka - linear_cka

            all_linear_ckas.append(linear_cka)
            all_geodesic_ckas.append(geodesic_cka)
            all_deltas.append(delta)

            pair_result = {
                "layer_a": layer_a,
                "layer_b": layer_b,
                "linear_cka": linear_cka,
                "geodesic_cka": geodesic_cka,
                "delta": delta,
            }
            results["layer_pairs"].append(pair_result)

            logger.info(f"  Layers {layer_a}-{layer_b}: linear={linear_cka:.4f}, geodesic={geodesic_cka:.4f}, Δ={delta:+.4f}")

        # Summary
        results["summary"] = {
            "mean_linear_cka": float(np.mean(all_linear_ckas)),
            "mean_geodesic_cka": float(np.mean(all_geodesic_ckas)),
            "mean_delta": float(np.mean(all_deltas)),
            "max_delta": float(np.max(all_deltas)),
            "min_delta": float(np.min(all_deltas)),
            "n_positive_delta": sum(1 for d in all_deltas if d > 0.01),
            "n_negative_delta": sum(1 for d in all_deltas if d < -0.01),
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Mean linear CKA: {results['summary']['mean_linear_cka']:.4f}")
        logger.info(f"Mean geodesic CKA: {results['summary']['mean_geodesic_cka']:.4f}")
        logger.info(f"Mean delta (geo - linear): {results['summary']['mean_delta']:+.4f}")
        logger.info(f"Max delta: {results['summary']['max_delta']:+.4f}")
        logger.info(f"Pairs where geodesic > linear (+0.01): {results['summary']['n_positive_delta']}")
        logger.info(f"Pairs where geodesic < linear (-0.01): {results['summary']['n_negative_delta']}")

        # Interpretation
        if results["summary"]["mean_delta"] > 0.05:
            logger.info("\nINTERPRETATION: Geodesic CKA consistently HIGHER than linear - manifold has positive curvature structure")
        elif results["summary"]["mean_delta"] < -0.05:
            logger.info("\nINTERPRETATION: Geodesic CKA consistently LOWER than linear - unusual, may indicate noise sensitivity")
        else:
            logger.info("\nINTERPRETATION: Linear and geodesic CKA are similar - manifold is approximately flat")

        # Find most interesting pairs (largest delta)
        sorted_pairs = sorted(results["layer_pairs"], key=lambda x: abs(x["delta"]), reverse=True)
        logger.info("\nMost interesting layer pairs (largest |delta|):")
        for pair in sorted_pairs[:5]:
            logger.info(f"  Layers {pair['layer_a']}-{pair['layer_b']}: Δ={pair['delta']:+.4f}")

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = CKAComparison(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/cka_linear_vs_geodesic.json"
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
