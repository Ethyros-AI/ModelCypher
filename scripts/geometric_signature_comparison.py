#!/usr/bin/env python3
"""Experiment 44: Geometric Signature Comparison.

Phase 9 - Stage 2: Do correct vs incorrect math have different geometry?

The hypothesis: If math capability requires specific geometric structure,
correct answers should have different SVD patterns than incorrect ones.

This experiment:
1. Captures activation SVD when model answers math questions
2. Compares SVD ratios for correct vs incorrect answers
3. Identifies which SVD dimensions correlate with correctness

If we find dimensions that differ, those are targets for fundamental alignment.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# All math questions (fundamentals + benchmark)
MATH_QUESTIONS = [
    # Fundamentals (Level 1)
    ("What is 1 + 1?", ["1", "2", "3", "4"], 1, "fundamental"),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1, "fundamental"),
    ("What is 3 + 3?", ["5", "6", "7", "8"], 1, "fundamental"),
    ("What is 2 × 2?", ["2", "4", "6", "8"], 1, "fundamental"),
    ("What is 3 × 3?", ["6", "9", "12", "15"], 1, "fundamental"),
    ("What is 5 + 5?", ["8", "9", "10", "11"], 2, "fundamental"),
    ("What is 10 - 5?", ["3", "4", "5", "6"], 2, "fundamental"),
    ("What is 4 ÷ 2?", ["1", "2", "3", "4"], 1, "fundamental"),
    # Basic operations (Level 2)
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1, "basic"),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2, "basic"),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1, "basic"),
    ("What is 9 × 6?", ["45", "54", "56", "63"], 1, "basic"),
    ("What is 25 + 17?", ["32", "42", "52", "62"], 1, "basic"),
    # Complex (Level 3)
    ("What is 3²?", ["6", "9", "12", "27"], 1, "complex"),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1, "complex"),
    ("What is 7²?", ["14", "21", "49", "56"], 2, "complex"),
]

# Constants for geometric analysis
CONSTANTS = {
    "pi_over_e": np.pi / np.e,
    "e_over_pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "inv_phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "inv_sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}


class GeometricSignatureComparison:
    """Compare geometric signatures of correct vs incorrect math responses."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _evaluate_and_capture(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float, np.ndarray]:
        """Evaluate question and capture activation SVD."""
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get prediction via forward pass
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

        scores = np.array([float(next_logits[t].item()) for t in choice_tokens[:len(choices)]])
        prediction = int(np.argmax(scores))
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        confidence = float(probs[prediction])

        # Get weight SVD as geometric proxy
        mid = self.n_layers // 2
        layer = self.model.model.layers[mid]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        w_np = np.array(w.tolist(), dtype=np.float32)

        try:
            _, S, _ = np.linalg.svd(w_np, full_matrices=False)
            S = S[:50]  # Top 50 singular values
        except:
            S = np.array([])

        return prediction == correct_idx, confidence, S

    def _compute_svd_features(self, S: np.ndarray) -> Dict:
        """Compute features from SVD."""
        if len(S) < 2:
            return {
                "top_ratios": [],
                "const_matches": {name: 0 for name in CONSTANTS},
                "total_const_matches": 0,
                "spectral_entropy": 0,
                "effective_rank": 0,
            }

        # Top 10 ratios
        top_ratios = []
        for i in range(min(10, len(S) - 1)):
            if S[i+1] > 1e-10:
                top_ratios.append(S[i] / S[i+1])

        # Constant matches
        const_matches = {name: 0 for name in CONSTANTS}
        for i in range(len(S) - 1):
            if S[i+1] > 1e-10:
                ratio = S[i] / S[i+1]
                for name, const in CONSTANTS.items():
                    if abs(ratio - const) / const < 0.05:
                        const_matches[name] += 1

        # Spectral entropy
        S_norm = S / (S.sum() + 1e-10)
        spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))

        # Effective rank
        effective_rank = np.exp(spectral_entropy)

        return {
            "top_ratios": top_ratios,
            "const_matches": const_matches,
            "total_const_matches": sum(const_matches.values()),
            "spectral_entropy": spectral_entropy,
            "effective_rank": effective_rank,
            "svd_values": S[:20].tolist() if len(S) > 20 else S.tolist(),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 44: GEOMETRIC SIGNATURE COMPARISON")
        logger.info("=" * 60)
        logger.info("\nDo correct vs incorrect math have different geometry?\n")

        correct_features = []
        incorrect_features = []
        all_results = []

        for q, choices, correct_idx, level in MATH_QUESTIONS:
            is_correct, confidence, S = self._evaluate_and_capture(q, choices, correct_idx)
            features = self._compute_svd_features(S)

            result = {
                "question": q,
                "level": level,
                "correct": is_correct,
                "confidence": confidence,
                "features": features,
            }
            all_results.append(result)

            if is_correct:
                correct_features.append(features)
            else:
                incorrect_features.append(features)

            status = "✓" if is_correct else "✗"
            logger.info(f"  [{status}] {q[:40]}... (const: {features['total_const_matches']})")

        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON")
        logger.info("=" * 60)

        n_correct = len(correct_features)
        n_incorrect = len(incorrect_features)
        logger.info(f"\nCorrect: {n_correct}, Incorrect: {n_incorrect}")

        results = {
            "all_results": all_results,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "comparison": {},
        }

        if n_correct > 0 and n_incorrect > 0:
            # Compare constant matches
            correct_const = [f["total_const_matches"] for f in correct_features]
            incorrect_const = [f["total_const_matches"] for f in incorrect_features]

            mean_correct = np.mean(correct_const)
            mean_incorrect = np.mean(incorrect_const)

            logger.info(f"\nConstant Matches:")
            logger.info(f"  Correct answers: {mean_correct:.2f} mean")
            logger.info(f"  Incorrect answers: {mean_incorrect:.2f} mean")

            # Statistical test
            if len(correct_const) > 1 and len(incorrect_const) > 1:
                t_stat, p_value = stats.ttest_ind(correct_const, incorrect_const)
                logger.info(f"  t-test: t={t_stat:.3f}, p={p_value:.3f}")
                results["comparison"]["const_matches"] = {
                    "mean_correct": mean_correct,
                    "mean_incorrect": mean_incorrect,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            # Compare spectral entropy
            correct_entropy = [f["spectral_entropy"] for f in correct_features]
            incorrect_entropy = [f["spectral_entropy"] for f in incorrect_features]

            mean_correct_ent = np.mean(correct_entropy)
            mean_incorrect_ent = np.mean(incorrect_entropy)

            logger.info(f"\nSpectral Entropy:")
            logger.info(f"  Correct answers: {mean_correct_ent:.3f} mean")
            logger.info(f"  Incorrect answers: {mean_incorrect_ent:.3f} mean")

            if len(correct_entropy) > 1 and len(incorrect_entropy) > 1:
                t_stat, p_value = stats.ttest_ind(correct_entropy, incorrect_entropy)
                logger.info(f"  t-test: t={t_stat:.3f}, p={p_value:.3f}")
                results["comparison"]["spectral_entropy"] = {
                    "mean_correct": mean_correct_ent,
                    "mean_incorrect": mean_incorrect_ent,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            # Compare effective rank
            correct_rank = [f["effective_rank"] for f in correct_features]
            incorrect_rank = [f["effective_rank"] for f in incorrect_features]

            mean_correct_rank = np.mean(correct_rank)
            mean_incorrect_rank = np.mean(incorrect_rank)

            logger.info(f"\nEffective Rank:")
            logger.info(f"  Correct answers: {mean_correct_rank:.2f} mean")
            logger.info(f"  Incorrect answers: {mean_incorrect_rank:.2f} mean")

            if len(correct_rank) > 1 and len(incorrect_rank) > 1:
                t_stat, p_value = stats.ttest_ind(correct_rank, incorrect_rank)
                logger.info(f"  t-test: t={t_stat:.3f}, p={p_value:.3f}")
                results["comparison"]["effective_rank"] = {
                    "mean_correct": mean_correct_rank,
                    "mean_incorrect": mean_incorrect_rank,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

            # Compare by constant type
            logger.info(f"\nBy Constant Type:")
            for const_name in CONSTANTS.keys():
                correct_const_type = [f["const_matches"][const_name] for f in correct_features]
                incorrect_const_type = [f["const_matches"][const_name] for f in incorrect_features]

                mean_c = np.mean(correct_const_type)
                mean_i = np.mean(incorrect_const_type)

                if mean_c > 0 or mean_i > 0:
                    logger.info(f"  {const_name}: correct={mean_c:.2f}, incorrect={mean_i:.2f}")

            # Compare top ratios
            logger.info(f"\nTop SVD Ratios (first 5):")
            if correct_features and incorrect_features:
                correct_top = np.mean([f["top_ratios"][:5] for f in correct_features if len(f["top_ratios"]) >= 5], axis=0)
                incorrect_top = np.mean([f["top_ratios"][:5] for f in incorrect_features if len(f["top_ratios"]) >= 5], axis=0)

                if len(correct_top) > 0 and len(incorrect_top) > 0:
                    for i, (c, inc) in enumerate(zip(correct_top, incorrect_top)):
                        logger.info(f"  Ratio {i+1}: correct={c:.3f}, incorrect={inc:.3f}")
                        results["comparison"][f"ratio_{i+1}"] = {
                            "correct": float(c),
                            "incorrect": float(inc),
                        }

        # Analysis by level
        logger.info(f"\n{'='*60}")
        logger.info("BY DIFFICULTY LEVEL")
        logger.info("=" * 60)

        for level in ["fundamental", "basic", "complex"]:
            level_results = [r for r in all_results if r["level"] == level]
            n_level = len(level_results)
            n_correct_level = sum(1 for r in level_results if r["correct"])

            if n_level > 0:
                logger.info(f"\n{level.upper()}:")
                logger.info(f"  Accuracy: {n_correct_level}/{n_level} ({n_correct_level/n_level:.0%})")

                const_matches = [r["features"]["total_const_matches"] for r in level_results]
                logger.info(f"  Mean const matches: {np.mean(const_matches):.2f}")

                results[f"level_{level}"] = {
                    "n_questions": n_level,
                    "n_correct": n_correct_level,
                    "accuracy": n_correct_level / n_level,
                    "mean_const_matches": float(np.mean(const_matches)),
                }

        # Conclusion
        logger.info(f"\n{'='*60}")
        logger.info("CONCLUSION")
        logger.info("=" * 60)

        significant_diffs = []
        for metric, data in results.get("comparison", {}).items():
            if isinstance(data, dict) and data.get("significant"):
                significant_diffs.append(metric)

        if significant_diffs:
            conclusion = "geometry_differs"
            logger.info(f"\n*** SIGNIFICANT GEOMETRIC DIFFERENCES FOUND ***")
            logger.info(f"Metrics that differ (p<0.05): {significant_diffs}")
            logger.info("This suggests correct/incorrect math use different geometric patterns.")
            logger.info("These dimensions are targets for fundamental alignment.")
        else:
            conclusion = "no_significant_difference"
            logger.info(f"\n*** NO SIGNIFICANT GEOMETRIC DIFFERENCES ***")
            logger.info("Correct and incorrect math have similar geometric signatures.")
            logger.info("The issue may be more about representation than geometry.")

        results["conclusion"] = conclusion
        results["significant_differences"] = significant_diffs

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = GeometricSignatureComparison(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/geometric_signature_comparison.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
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
