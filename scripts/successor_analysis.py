#!/usr/bin/env python3
"""Experiment 54: Successor Function Analysis.

Compression concentrates misunderstanding.
Rotation transforms but doesn't teach.

The real question: What DOES the model compute when it gets 1+n = n?
Does it have the concept of "successor" (adding 1 = incrementing)?

If 1+n = n consistently, then the model treats "+1" as identity.
If 1+n ≈ n with noise, then there's a weak successor signal.

Let's find what the model's internal representation of "+1" actually IS.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SuccessorAnalyzer:
    """Analyze what the model computes for '+1' operations."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _get_hidden_states(self, prompt: str, layer_frac: float = 0.5) -> np.ndarray:
        """Get logits as representation proxy (simpler than extracting hidden states)."""
        return self._get_logits(prompt)

    def _get_logits(self, prompt: str) -> np.ndarray:
        """Get output logits."""
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def _predict_number(self, prompt: str) -> Tuple[int, float]:
        """Get the predicted number and its probability."""
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        probs = []
        for num in range(21):
            num_str = str(num)
            token_ids = self.tokenizer.encode(num_str)
            if token_ids:
                prob = float(next_logits[token_ids[-1]].item())
                probs.append((num, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[0][0], probs[0][1]

    def analyze_successor_function(self) -> Dict:
        """Analyze what '+1' actually does in the model."""
        results = {
            "n_values": {},
            "successor_differences": [],
            "successor_direction": None,
        }

        # For each n, get representations of:
        # - "n" (the number)
        # - "n+1=" (adding 1 to n)
        # - model's prediction for n+1=

        n_reps = []
        n_plus_1_reps = []

        for n in range(1, 11):
            logger.info(f"\nAnalyzing n={n}:")

            # Get representation of just "n"
            n_rep = self._get_hidden_states(str(n))

            # Get representation when computing "n+1="
            plus_1_prompt = f"{n}+1="
            plus_1_rep = self._get_hidden_states(plus_1_prompt)

            # What does the model predict?
            predicted, conf = self._predict_number(plus_1_prompt)
            expected = n + 1
            correct = predicted == expected

            logger.info(f"  {n}+1 = {predicted} (expected {expected}) {'✓' if correct else '✗'}")

            # The "successor direction" should be plus_1_rep - n_rep
            successor_diff = plus_1_rep - n_rep

            results["n_values"][str(n)] = {
                "predicted": int(predicted),
                "expected": expected,
                "correct": correct,
                "confidence": float(conf),
            }
            results["successor_differences"].append(successor_diff)

            n_reps.append(n_rep)
            n_plus_1_reps.append(plus_1_rep)

        # Stack all successor differences
        successor_diffs = np.vstack(results["successor_differences"])

        # Is there a COMMON successor direction across all n?
        # PCA to find it
        centered = successor_diffs - successor_diffs.mean(axis=0)
        U, S, Vt = svd(centered, full_matrices=False)

        # The first principal component is the "main" successor direction
        main_successor_dir = Vt[0]

        # How much variance does it explain?
        total_var = (S ** 2).sum()
        explained_var = S[0] ** 2 / total_var if total_var > 0 else 0

        logger.info(f"\n\nSuccessor direction analysis:")
        logger.info(f"  Variance explained by 1st PC: {explained_var:.1%}")

        # How consistent are the individual successor diffs with this direction?
        consistencies = []
        for i, diff in enumerate(successor_diffs):
            diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
            consistency = abs(np.dot(diff_norm, main_successor_dir))
            consistencies.append(consistency)
            n = i + 1
            logger.info(f"  n={n}: consistency with main direction: {consistency:.3f}")

        mean_consistency = np.mean(consistencies)
        logger.info(f"\n  Mean consistency: {mean_consistency:.3f}")

        results["main_successor_direction_variance"] = float(explained_var)
        results["consistencies"] = [float(c) for c in consistencies]
        results["mean_consistency"] = float(mean_consistency)

        # Also check: are n and n+1 representations related in a consistent way?
        logger.info(f"\n\nRelationship between n and n+1 representations:")
        n_stack = np.vstack(n_reps)
        n_plus_1_stack = np.vstack(n_plus_1_reps[:-1])  # 1 less because we don't have 11
        n_shifted = np.vstack(n_reps[1:])  # n_reps shifted by 1 (so n_shifted[0] = n_reps[1] = "2")

        # If +1 works: n_plus_1_stack should equal n_shifted
        # i.e., rep(1+1=) should equal rep(2)
        for i in range(len(n_shifted)):
            n = i + 1
            n_plus_1_rep = n_plus_1_reps[i]
            n_next_rep = n_reps[i + 1]

            sim = np.dot(n_plus_1_rep, n_next_rep) / (np.linalg.norm(n_plus_1_rep) * np.linalg.norm(n_next_rep) + 1e-10)
            logger.info(f"  sim(rep({n}+1=), rep({n+1})) = {sim:.3f}")

        return results

    def compare_operations(self) -> Dict:
        """Compare +1, +2, +3 to see if the model has different 'adder' circuits."""
        results = {}

        for delta in [1, 2, 3]:
            logger.info(f"\n\nAnalyzing +{delta} operation:")
            diffs = []
            accuracies = []

            for n in range(1, 8):  # smaller range to fit within 10
                prompt = f"{n}+{delta}="
                rep = self._get_hidden_states(prompt)
                base_rep = self._get_hidden_states(str(n))
                diff = rep - base_rep
                diffs.append(diff)

                predicted, _ = self._predict_number(prompt)
                expected = n + delta
                accuracies.append(predicted == expected)

            diffs = np.vstack(diffs)
            centered = diffs - diffs.mean(axis=0)
            U, S, Vt = svd(centered, full_matrices=False)

            explained_var = S[0] ** 2 / (S ** 2).sum() if (S ** 2).sum() > 0 else 0
            accuracy = np.mean(accuracies)

            logger.info(f"  +{delta} accuracy: {accuracy:.0%}")
            logger.info(f"  +{delta} direction consistency: {explained_var:.1%}")

            results[f"+{delta}"] = {
                "accuracy": float(accuracy),
                "direction_consistency": float(explained_var),
            }

        return results

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 54: SUCCESSOR FUNCTION ANALYSIS")
        logger.info("=" * 60)
        logger.info("\nWhat does '+1' actually compute in this model?\n")

        successor_results = self.analyze_successor_function()
        operation_results = self.compare_operations()

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        mean_cons = successor_results["mean_consistency"]
        var_explained = successor_results["main_successor_direction_variance"]

        if mean_cons < 0.5:
            logger.info(f"\n*** NO UNIFIED SUCCESSOR CONCEPT ***")
            logger.info(f"Mean consistency: {mean_cons:.3f}")
            logger.info(f"The model doesn't have a consistent '+1' direction.")
            logger.info(f"Each 'n+1' is computed differently - no unified successor function.")
            conclusion = "no_unified_successor"
        elif var_explained < 0.5:
            logger.info(f"\n*** WEAK SUCCESSOR DIRECTION ***")
            logger.info(f"Variance explained: {var_explained:.1%}")
            logger.info(f"There's some structure but it's not dominant.")
            conclusion = "weak_successor"
        else:
            logger.info(f"\n*** SUCCESSOR DIRECTION EXISTS ***")
            logger.info(f"Mean consistency: {mean_cons:.3f}")
            logger.info(f"Variance explained: {var_explained:.1%}")
            conclusion = "successor_exists"

        results = {
            "successor_analysis": successor_results,
            "operation_comparison": operation_results,
            "conclusion": conclusion,
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = SuccessorAnalyzer(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/successor_analysis.json"
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
