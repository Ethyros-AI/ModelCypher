#!/usr/bin/env python3
"""Experiment 55: What Makes Correct Arithmetic Correct?

The model gets some arithmetic right (1+1=2, 4+1=5) and most wrong.
What's DIFFERENT about the correct cases?

Is there a "correctness subspace" where arithmetic works?
Can we identify what makes some computations succeed while others fail?
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


# Extended test set to find more correct/incorrect pairs
ARITHMETIC_FACTS = []
for a in range(1, 11):
    for b in range(1, 11):
        ARITHMETIC_FACTS.append((f"{a}+{b}=", a + b, "addition"))
        if a >= b:
            ARITHMETIC_FACTS.append((f"{a}-{b}=", a - b, "subtraction"))


class CorrectVsIncorrectAnalyzer:
    """Analyze what distinguishes correct from incorrect arithmetic."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _get_logits(self, prompt: str) -> np.ndarray:
        """Get output logits."""
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def _predict_number(self, prompt: str) -> Tuple[int, float, np.ndarray]:
        """Get the predicted number, its probability, and full logit vector."""
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]
        logit_vec = np.array(next_logits.tolist(), dtype=np.float32)

        probs = []
        for num in range(21):
            num_str = str(num)
            token_ids = self.tokenizer.encode(num_str)
            if token_ids:
                prob = float(next_logits[token_ids[-1]].item())
                probs.append((num, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[0][0], probs[0][1], logit_vec

    def evaluate_all_facts(self) -> Tuple[List[Dict], List[Dict]]:
        """Evaluate all facts and separate into correct/incorrect."""
        correct_facts = []
        incorrect_facts = []

        for prompt, expected, op_type in ARITHMETIC_FACTS:
            predicted, conf, logits = self._predict_number(prompt)
            fact = {
                "prompt": prompt,
                "expected": expected,
                "predicted": int(predicted),
                "confidence": float(conf),
                "op_type": op_type,
                "logits": logits,
            }
            if predicted == expected:
                correct_facts.append(fact)
            else:
                incorrect_facts.append(fact)

        return correct_facts, incorrect_facts

    def analyze_differences(self, correct: List[Dict], incorrect: List[Dict]) -> Dict:
        """Analyze what distinguishes correct from incorrect."""
        results = {}

        # Separate by operation type
        for op_type in ["addition", "subtraction"]:
            c_facts = [f for f in correct if f["op_type"] == op_type]
            i_facts = [f for f in incorrect if f["op_type"] == op_type]

            if not c_facts or not i_facts:
                continue

            logger.info(f"\n{op_type.upper()}: {len(c_facts)} correct, {len(i_facts)} incorrect")

            # Get logit vectors
            c_logits = np.vstack([f["logits"] for f in c_facts])
            i_logits = np.vstack([f["logits"] for f in i_facts])

            # 1. Mean logit differences
            c_mean = c_logits.mean(axis=0)
            i_mean = i_logits.mean(axis=0)
            diff = c_mean - i_mean

            # Top dimensions where correct differs from incorrect
            top_diff_dims = np.argsort(np.abs(diff))[-10:][::-1]
            logger.info(f"  Top diff dimensions: {top_diff_dims}")

            # 2. Variance comparison
            c_var = c_logits.var(axis=0).mean()
            i_var = i_logits.var(axis=0).mean()
            logger.info(f"  Correct variance: {c_var:.4f}")
            logger.info(f"  Incorrect variance: {i_var:.4f}")

            # 3. Find the "correctness direction"
            # Direction from incorrect centroid to correct centroid
            correctness_dir = diff / (np.linalg.norm(diff) + 1e-10)

            # Project all facts onto this direction
            c_projections = c_logits @ correctness_dir
            i_projections = i_logits @ correctness_dir

            logger.info(f"  Correct projection: {c_projections.mean():.4f} ± {c_projections.std():.4f}")
            logger.info(f"  Incorrect projection: {i_projections.mean():.4f} ± {i_projections.std():.4f}")

            # Is there separation?
            c_min = c_projections.min()
            i_max = i_projections.max()
            separation = c_min - i_max
            logger.info(f"  Separation gap: {separation:.4f} ({'separable' if separation > 0 else 'overlapping'})")

            # 4. SVD analysis
            c_centered = c_logits - c_mean
            i_centered = i_logits - i_mean

            _, S_c, _ = svd(c_centered, full_matrices=False)
            _, S_i, _ = svd(i_centered, full_matrices=False)

            # Effective dimensionality
            S_c_norm = S_c / S_c.sum()
            S_i_norm = S_i / S_i.sum()
            eff_dim_c = np.exp(-np.sum(S_c_norm * np.log(S_c_norm + 1e-10)))
            eff_dim_i = np.exp(-np.sum(S_i_norm * np.log(S_i_norm + 1e-10)))

            logger.info(f"  Correct effective dim: {eff_dim_c:.2f}")
            logger.info(f"  Incorrect effective dim: {eff_dim_i:.2f}")

            results[op_type] = {
                "n_correct": len(c_facts),
                "n_incorrect": len(i_facts),
                "correct_variance": float(c_var),
                "incorrect_variance": float(i_var),
                "correct_eff_dim": float(eff_dim_c),
                "incorrect_eff_dim": float(eff_dim_i),
                "separation_gap": float(separation),
                "separable": separation > 0,
            }

        return results

    def find_patterns_in_correct(self, correct: List[Dict], incorrect: List[Dict]) -> Dict:
        """Find patterns in what makes facts correct."""
        results = {"correct_facts": [], "patterns": {}}

        logger.info("\n\nCORRECT FACTS:")
        for f in sorted(correct, key=lambda x: x["prompt"]):
            logger.info(f"  {f['prompt']}{f['expected']} (conf: {f['confidence']:.2f})")
            results["correct_facts"].append(f"{f['prompt']}{f['expected']}")

        # Extract numbers from correct addition facts
        add_correct = [f for f in correct if f["op_type"] == "addition"]
        if add_correct:
            operand_sums = []
            for f in add_correct:
                # Parse "a+b="
                parts = f["prompt"].replace("=", "").split("+")
                a, b = int(parts[0]), int(parts[1])
                operand_sums.append(a + b)
                logger.info(f"  {a}+{b}={a+b}: operands sum to {a+b}")

            results["patterns"]["operand_sums"] = operand_sums

        return results

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 55: WHAT MAKES CORRECT ARITHMETIC CORRECT?")
        logger.info("=" * 60)

        correct, incorrect = self.evaluate_all_facts()

        total = len(correct) + len(incorrect)
        logger.info(f"\nTotal facts: {total}")
        logger.info(f"Correct: {len(correct)} ({len(correct)/total:.1%})")
        logger.info(f"Incorrect: {len(incorrect)} ({len(incorrect)/total:.1%})")

        diff_analysis = self.analyze_differences(correct, incorrect)
        pattern_analysis = self.find_patterns_in_correct(correct, incorrect)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        # Check if there's a separable correctness dimension
        for op, data in diff_analysis.items():
            if data.get("separable"):
                logger.info(f"\n*** {op.upper()}: CORRECTNESS IS SEPARABLE ***")
                logger.info(f"There's a direction in logit space that separates correct from incorrect!")
                logger.info(f"This could be exploited for intervention.")
            else:
                logger.info(f"\n*** {op.upper()}: CORRECTNESS IS NOT SEPARABLE ***")
                logger.info(f"Correct and incorrect facts overlap in logit space.")

            # Dimensionality insight
            c_dim = data["correct_eff_dim"]
            i_dim = data["incorrect_eff_dim"]
            if c_dim < i_dim:
                logger.info(f"Correct facts are MORE concentrated (dim {c_dim:.1f} vs {i_dim:.1f})")
            else:
                logger.info(f"Incorrect facts are MORE concentrated (dim {i_dim:.1f} vs {c_dim:.1f})")

        results = {
            "total_facts": total,
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "accuracy": len(correct) / total,
            "difference_analysis": diff_analysis,
            "pattern_analysis": pattern_analysis,
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = CorrectVsIncorrectAnalyzer(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/correct_vs_incorrect.json"
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
