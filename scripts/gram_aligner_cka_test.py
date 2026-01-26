#!/usr/bin/env python3
"""Experiment 33: GramAligner - Linear vs Sampled Geodesic CKA.

Compare linear CKA (current diagnostic) vs geodesic CKA
to identify alignment issues that linear CKA might miss.

Key question: When linear_cka ≈ 1.0 but geodesic_cka < 1.0, what does that indicate?
"""

from __future__ import annotations

import json
import logging
import sys
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


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (dot-product Gram)."""
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    K_X = X_c @ X_c.T
    K_Y = Y_c @ Y_c.T

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


def compute_geodesic_cka(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5) -> float:
    """Compute geodesic CKA (RBF over k-NN distances)."""

    def geodesic_gram(points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        chord_dists = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = points[i] - points
            chord_dists[i] = np.sqrt(np.sum(diff * diff, axis=1))

        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
            for j in neighbors:
                adjacency[i, j] = chord_dists[i, j]
                adjacency[j, i] = chord_dists[j, i]

        sparse_adj = csr_matrix(adjacency)
        geo_dists = shortest_path(sparse_adj, directed=False)

        finite_mask = np.isfinite(geo_dists)
        if not np.all(finite_mask):
            max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
            geo_dists[~finite_mask] = max_finite * 2

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


def find_linear_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find linear alignment transform F such that source @ F ≈ target."""
    # Least squares: source @ F = target
    # F = pinv(source) @ target
    F = np.linalg.lstsq(source, target, rcond=None)[0]
    return F


class GramAlignerCKATest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_layer_activations(self, prompts: List[str], layer_idx: int) -> np.ndarray:
        """Get activations for prompts at a specific layer."""
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            hidden = self.model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i, layer in enumerate(self.model.model.layers):
                hidden = layer(hidden)
                mx.eval(hidden)
                if i == layer_idx:
                    act = hidden[0, -1, :]
                    mx.eval(act)
                    activations.append(np.array(act.tolist(), dtype=np.float32))
                    break

        return np.array(activations)

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 33: GRAM ALIGNER CKA COMPARISON")
        logger.info("=" * 60)

        # Test prompts
        prompts = [
            "What is 2 + 2?",
            "Calculate 15 times 7.",
            "What is the square root of 144?",
            "What is the capital of France?",
            "Which continent is Brazil in?",
            "What is the largest ocean?",
            "What is H2O?",
            "How many planets are there?",
            "What gas do plants produce?",
            "What is the opposite of hot?",
            "What is a synonym for happy?",
            "What is the plural of child?",
            "If all cats are animals, is a cat an animal?",
            "What comes next: 2, 4, 6, 8, ?",
            "Is a square a rectangle?",
            "The sun rises in the east.",
            "Water freezes at zero degrees.",
            "Cats are mammals.",
            "Birds have wings.",
            "The Earth is round.",
            "Paris is in France.",
            "Tokyo is in Japan.",
            "The moon orbits the Earth.",
            "Plants need sunlight.",
            "Fish live in water.",
        ]

        results = {
            "n_prompts": len(prompts),
            "n_layers": self.n_layers,
            "alignment_pairs": [],
            "summary": {},
        }

        # Test various alignment pairs
        layer_pairs = [
            (0, 1),  # Adjacent early
            (self.n_layers // 4, self.n_layers // 4 + 1),  # Adjacent early-mid
            (self.n_layers // 2, self.n_layers // 2 + 1),  # Adjacent mid
            (3 * self.n_layers // 4, 3 * self.n_layers // 4 + 1),  # Adjacent late
            (0, self.n_layers // 2),  # Distant: first to middle
            (self.n_layers // 4, 3 * self.n_layers // 4),  # Distant: quarter to three-quarter
            (0, self.n_layers - 1),  # Most distant: first to last
        ]

        logger.info(f"\nCollecting activations for {len(prompts)} prompts...")

        all_linear_ckas = []
        all_geodesic_ckas = []
        all_deltas = []
        high_linear_low_geodesic = []

        for source_layer, target_layer in layer_pairs:
            logger.info(f"\nAlignment: layer {source_layer} → layer {target_layer}")

            source_acts = self.get_layer_activations(prompts, source_layer)
            target_acts = self.get_layer_activations(prompts, target_layer)

            # Find linear alignment
            F = find_linear_alignment(source_acts, target_acts)
            aligned_source = source_acts @ F

            # Compute CKA metrics
            linear_cka = compute_linear_cka(aligned_source, target_acts)
            geodesic_cka = compute_geodesic_cka(aligned_source, target_acts)
            delta = linear_cka - geodesic_cka

            all_linear_ckas.append(linear_cka)
            all_geodesic_ckas.append(geodesic_cka)
            all_deltas.append(delta)

            # Track cases where linear CKA is high but geodesic is lower
            if linear_cka > 0.9 and delta > 0.1:
                high_linear_low_geodesic.append({
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "linear_cka": linear_cka,
                    "geodesic_cka": geodesic_cka,
                    "delta": delta,
                })

            pair_result = {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "linear_cka": linear_cka,
                "geodesic_cka": geodesic_cka,
                "delta": delta,
            }
            results["alignment_pairs"].append(pair_result)

            logger.info(f"  Linear CKA: {linear_cka:.4f}")
            logger.info(f"  Geodesic CKA: {geodesic_cka:.4f}")
            logger.info(f"  Delta (linear - geodesic): {delta:+.4f}")

        # Summary
        results["summary"] = {
            "mean_linear_cka": float(np.mean(all_linear_ckas)),
            "mean_geodesic_cka": float(np.mean(all_geodesic_ckas)),
            "mean_delta": float(np.mean(all_deltas)),
            "max_delta": float(np.max(all_deltas)),
            "n_high_linear_low_geodesic": len(high_linear_low_geodesic),
            "high_linear_low_geodesic_cases": high_linear_low_geodesic,
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Mean Linear CKA: {results['summary']['mean_linear_cka']:.4f}")
        logger.info(f"Mean Geodesic CKA: {results['summary']['mean_geodesic_cka']:.4f}")
        logger.info(f"Mean Delta (linear - geodesic): {results['summary']['mean_delta']:+.4f}")
        logger.info(f"Max Delta: {results['summary']['max_delta']:+.4f}")
        logger.info(f"Cases with high linear but low geodesic: {len(high_linear_low_geodesic)}")

        if high_linear_low_geodesic:
            logger.info("\nCases where linear CKA is HIGH but geodesic is LOWER:")
            for case in high_linear_low_geodesic:
                logger.info(f"  Layers {case['source_layer']}→{case['target_layer']}: "
                           f"linear={case['linear_cka']:.4f}, geodesic={case['geodesic_cka']:.4f}, Δ={case['delta']:+.4f}")

        # Interpretation
        if results["summary"]["mean_delta"] > 0.1:
            logger.info("\nINTERPRETATION: Linear CKA consistently HIGHER than geodesic")
            logger.info("  → Linear alignment looks good in Euclidean space but may miss manifold structure")
            results["conclusion"] = "geodesic_more_discriminative"
        elif results["summary"]["mean_delta"] < -0.1:
            logger.info("\nINTERPRETATION: Geodesic CKA higher than linear")
            logger.info("  → Geodesic captures additional manifold structure")
            results["conclusion"] = "geodesic_captures_more"
        else:
            logger.info("\nINTERPRETATION: Linear and geodesic CKA are similar")
            logger.info("  → Activation manifold is approximately flat")
            results["conclusion"] = "similar"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = GramAlignerCKATest(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/gram_aligner_cka_comparison.json"
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
