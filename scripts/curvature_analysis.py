#!/usr/bin/env python3
"""Experiment 29: Curvature Analysis.

Analyze where the activation manifold is curved (geodesic != Euclidean).
Key question: Do high-curvature regions correlate with model capabilities?
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


CATEGORY_QUESTIONS = {
    "math": [
        ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
        ("What is 8 x 7?", ["48", "54", "56", "64"], 2),
        ("What is 100 / 4?", ["20", "25", "30", "40"], 1),
    ],
    "geography": [
        ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
        ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
        ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ],
    "science": [
        ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
        ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
        ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ],
    "language": [
        ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
        ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
        ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ],
    "logic": [
        ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
        ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
        ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ],
}


def compute_local_curvature(
    points: np.ndarray,
    k_neighbors: int = 5,
) -> np.ndarray:
    """Estimate local curvature at each point.

    Curvature is estimated by comparing geodesic to chord distances
    for local neighborhoods. High ratio = high curvature.
    """
    n = points.shape[0]

    # Compute chord distances
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

    # Local curvature: mean geodesic/chord ratio for k nearest neighbors
    curvatures = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
        ratios = []
        for j in neighbors:
            if chord_dists[i, j] > 1e-10:
                ratios.append(geo_dists[i, j] / chord_dists[i, j])
        curvatures[i] = np.mean(ratios) - 1.0 if ratios else 0.0  # Subtract 1 so flat = 0

    return curvatures


class CurvatureAnalysis:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def evaluate_question(
        self,
        question: str,
        choices: List[str],
        correct_idx: int,
    ) -> Tuple[bool, int]:
        """Evaluate a question and return (is_correct, prediction)."""
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])
                    break
            else:
                choice_tokens.append(0)

        scores = [float(next_logits[t].item()) for t in choice_tokens]
        prediction = int(np.argmax(scores))

        return prediction == correct_idx, prediction

    def get_activations_for_questions(
        self,
        layer_idx: int,
    ) -> Tuple[np.ndarray, List[bool], List[str]]:
        """Get activations and correctness for all questions."""
        import mlx.core as mx

        activations = []
        correct_flags = []
        categories = []

        for cat, questions in CATEGORY_QUESTIONS.items():
            for question, choices, correct_idx in questions:
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "Answer:"

                tokens = self.tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

                # Forward pass
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

                is_correct, _ = self.evaluate_question(question, choices, correct_idx)
                correct_flags.append(is_correct)
                categories.append(cat)

        return np.array(activations), correct_flags, categories

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 29: CURVATURE ANALYSIS")
        logger.info("=" * 60)

        results = {
            "n_layers": self.n_layers,
            "n_questions": sum(len(q) for q in CATEGORY_QUESTIONS.values()),
            "by_layer": {},
            "summary": {},
        }

        # Test layers
        test_layers = [
            0,
            self.n_layers // 4,
            self.n_layers // 2,
            3 * self.n_layers // 4,
            self.n_layers - 1,
        ]

        curvature_correct_corrs = []
        curvature_by_category = {cat: [] for cat in CATEGORY_QUESTIONS.keys()}

        for layer_idx in test_layers:
            logger.info(f"\nAnalyzing layer {layer_idx}...")

            activations, correct_flags, categories = self.get_activations_for_questions(layer_idx)
            logger.info(f"  Activations shape: {activations.shape}")

            # Compute local curvature
            curvatures = compute_local_curvature(activations, k_neighbors=3)

            # Correlation with correctness
            correct_arr = np.array(correct_flags, dtype=np.float32)
            if np.std(curvatures) > 1e-10 and np.std(correct_arr) > 1e-10:
                corr = float(np.corrcoef(curvatures, correct_arr)[0, 1])
            else:
                corr = 0.0

            curvature_correct_corrs.append(corr)

            # Curvature by category
            cat_curvatures = {}
            for cat in CATEGORY_QUESTIONS.keys():
                cat_mask = [c == cat for c in categories]
                cat_curvature = np.mean(curvatures[cat_mask])
                cat_curvatures[cat] = float(cat_curvature)
                curvature_by_category[cat].append(cat_curvature)

            # Curvature stats
            layer_results = {
                "curvature_mean": float(np.mean(curvatures)),
                "curvature_std": float(np.std(curvatures)),
                "curvature_min": float(np.min(curvatures)),
                "curvature_max": float(np.max(curvatures)),
                "correlation_with_correctness": corr,
                "curvature_by_category": cat_curvatures,
                "accuracy": float(np.mean(correct_arr)),
            }

            logger.info(f"  Mean curvature: {layer_results['curvature_mean']:.4f}")
            logger.info(f"  Corr(curvature, correct): {corr:.4f}")
            logger.info(f"  Accuracy: {layer_results['accuracy']:.0%}")

            results["by_layer"][str(layer_idx)] = layer_results

        # Summary
        results["summary"] = {
            "mean_curvature_correct_correlation": float(np.mean(curvature_correct_corrs)),
            "category_curvature_means": {
                cat: float(np.mean(curv_list))
                for cat, curv_list in curvature_by_category.items()
            },
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Mean correlation (curvature vs correctness): {results['summary']['mean_curvature_correct_correlation']:.4f}")
        logger.info("\nCurvature by category (mean across layers):")
        for cat, mean_curv in sorted(results["summary"]["category_curvature_means"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {cat}: {mean_curv:.4f}")

        # Interpretation
        corr = results["summary"]["mean_curvature_correct_correlation"]
        if abs(corr) > 0.3:
            if corr > 0:
                logger.info("\nINTERPRETATION: Higher curvature correlates with CORRECT answers")
            else:
                logger.info("\nINTERPRETATION: Higher curvature correlates with INCORRECT answers")
        else:
            logger.info("\nINTERPRETATION: No strong correlation between curvature and correctness")

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = CurvatureAnalysis(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/curvature_analysis.json"
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
