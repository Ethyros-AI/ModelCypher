#!/usr/bin/env python3
"""Experiment 43: Fundamental Arithmetic Check.

Phase 9 - Stage 1: Are the fundamentals locked in?

The hypothesis: Math at 20% isn't "no capability" - it's "corrupted foundation."

This experiment checks if TRIVIAL arithmetic (2+2, 3×3) is structurally locked in:
1. Accuracy: Does it get the right answer?
2. Consistency: Same answer every time? (run N times)
3. Confidence: How sure is it?
4. Geometric signature: SVD ratios during computation

If fundamentals are NOT locked in, that explains why gradient-guided learning
couldn't improve math - you can't build on a broken foundation.
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


# Level 1: Fundamentals (should be trivially easy)
FUNDAMENTAL_QUESTIONS = [
    ("What is 1 + 1?", ["1", "2", "3", "4"], 1),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    ("What is 3 + 3?", ["5", "6", "7", "8"], 1),
    ("What is 2 × 2?", ["2", "4", "6", "8"], 1),
    ("What is 3 × 3?", ["6", "9", "12", "15"], 1),
    ("What is 5 + 5?", ["8", "9", "10", "11"], 2),
    ("What is 10 - 5?", ["3", "4", "5", "6"], 2),
    ("What is 4 ÷ 2?", ["1", "2", "3", "4"], 1),
]

# Level 2: Basic operations (from the benchmark - what we test at 20%)
BASIC_QUESTIONS = [
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
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


class FundamentalArithmeticChecker:
    """Check if fundamentals are structurally locked in."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float, int]:
        """Evaluate a question, return (correct, confidence, prediction)."""
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

        scores = np.array([float(next_logits[t].item()) for t in choice_tokens[:len(choices)]])
        prediction = int(np.argmax(scores))

        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        confidence = float(probs[prediction])

        return prediction == correct_idx, confidence, prediction

    def _get_weight_svd(self) -> np.ndarray:
        """Get SVD of middle layer weights as geometric proxy."""
        import mlx.core as mx

        # Get weight from middle layer
        mid = self.n_layers // 2
        layer = self.model.model.layers[mid]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        w_np = np.array(w.tolist(), dtype=np.float32)

        # Compute SVD of weight
        try:
            _, S, _ = np.linalg.svd(w_np, full_matrices=False)
            return S[:50]  # Top 50 singular values
        except:
            return np.array([])

    def _count_constant_matches(self, S: np.ndarray, threshold: float = 0.05) -> Dict[str, int]:
        """Count how many SVD ratios match each constant."""
        if len(S) < 2:
            return {name: 0 for name in CONSTANTS}

        counts = {name: 0 for name in CONSTANTS}

        for i in range(len(S) - 1):
            if S[i+1] > 1e-10:
                ratio = S[i] / S[i+1]
                for name, const in CONSTANTS.items():
                    if abs(ratio - const) / const < threshold:
                        counts[name] += 1

        return counts

    def check_consistency(self, question: str, choices: List[str], correct_idx: int, n_runs: int = 5) -> Dict:
        """Check if the model gives consistent answers."""
        results = []
        for _ in range(n_runs):
            correct, confidence, prediction = self._evaluate_question(question, choices, correct_idx)
            results.append({
                "correct": correct,
                "confidence": confidence,
                "prediction": prediction,
            })

        predictions = [r["prediction"] for r in results]
        most_common = max(set(predictions), key=predictions.count)
        consistency = predictions.count(most_common) / len(predictions)

        return {
            "consistency": consistency,
            "most_common_prediction": most_common,
            "correct_prediction": correct_idx,
            "all_correct": all(r["correct"] for r in results),
            "mean_confidence": np.mean([r["confidence"] for r in results]),
            "std_confidence": np.std([r["confidence"] for r in results]),
        }

    def check_fundamental(self, question: str, choices: List[str], correct_idx: int) -> Dict:
        """Full check of a fundamental operation."""
        # Basic evaluation
        correct, confidence, prediction = self._evaluate_question(question, choices, correct_idx)

        # Consistency check
        consistency_result = self.check_consistency(question, choices, correct_idx, n_runs=5)

        # Geometric signature (use weight SVD as proxy)
        svd = self._get_weight_svd()
        const_matches = self._count_constant_matches(svd)

        # Determine if "locked in"
        is_locked_in = (
            consistency_result["all_correct"] and
            consistency_result["consistency"] >= 0.95 and
            consistency_result["mean_confidence"] >= 0.90
        )

        return {
            "question": question,
            "correct_answer": choices[correct_idx],
            "model_answer": choices[prediction],
            "correct": correct,
            "confidence": confidence,
            "consistency": consistency_result["consistency"],
            "all_runs_correct": consistency_result["all_correct"],
            "mean_confidence": consistency_result["mean_confidence"],
            "constant_matches": const_matches,
            "total_const_matches": sum(const_matches.values()),
            "is_locked_in": is_locked_in,
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 43: FUNDAMENTAL ARITHMETIC CHECK")
        logger.info("=" * 60)
        logger.info("\nAre the fundamentals structurally locked in?\n")

        results = {
            "fundamental": {"questions": [], "summary": {}},
            "basic": {"questions": [], "summary": {}},
        }

        # Check Level 1: Fundamentals
        logger.info("=" * 40)
        logger.info("LEVEL 1: FUNDAMENTALS (trivial arithmetic)")
        logger.info("=" * 40)

        for q, choices, correct_idx in FUNDAMENTAL_QUESTIONS:
            result = self.check_fundamental(q, choices, correct_idx)
            results["fundamental"]["questions"].append(result)

            status = "LOCKED IN" if result["is_locked_in"] else (
                "CORRECT" if result["correct"] else "WRONG"
            )
            logger.info(f"  {q}")
            logger.info(f"    Answer: {result['model_answer']} ({'✓' if result['correct'] else '✗'})")
            logger.info(f"    Confidence: {result['confidence']:.2f}, Consistency: {result['consistency']:.2f}")
            logger.info(f"    Status: [{status}]")

        # Summarize fundamentals
        fund_correct = sum(1 for r in results["fundamental"]["questions"] if r["correct"])
        fund_locked = sum(1 for r in results["fundamental"]["questions"] if r["is_locked_in"])
        fund_mean_conf = np.mean([r["mean_confidence"] for r in results["fundamental"]["questions"]])
        fund_mean_const = np.mean([r["total_const_matches"] for r in results["fundamental"]["questions"]])

        results["fundamental"]["summary"] = {
            "n_questions": len(FUNDAMENTAL_QUESTIONS),
            "n_correct": fund_correct,
            "n_locked_in": fund_locked,
            "accuracy": fund_correct / len(FUNDAMENTAL_QUESTIONS),
            "locked_in_rate": fund_locked / len(FUNDAMENTAL_QUESTIONS),
            "mean_confidence": fund_mean_conf,
            "mean_const_matches": fund_mean_const,
        }

        logger.info(f"\nFundamentals Summary:")
        logger.info(f"  Accuracy: {fund_correct}/{len(FUNDAMENTAL_QUESTIONS)} ({results['fundamental']['summary']['accuracy']:.0%})")
        logger.info(f"  Locked In: {fund_locked}/{len(FUNDAMENTAL_QUESTIONS)} ({results['fundamental']['summary']['locked_in_rate']:.0%})")
        logger.info(f"  Mean Confidence: {fund_mean_conf:.2f}")

        # Check Level 2: Basic Operations
        logger.info("\n" + "=" * 40)
        logger.info("LEVEL 2: BASIC OPERATIONS (benchmark questions)")
        logger.info("=" * 40)

        for q, choices, correct_idx in BASIC_QUESTIONS:
            result = self.check_fundamental(q, choices, correct_idx)
            results["basic"]["questions"].append(result)

            status = "LOCKED IN" if result["is_locked_in"] else (
                "CORRECT" if result["correct"] else "WRONG"
            )
            logger.info(f"  {q}")
            logger.info(f"    Answer: {result['model_answer']} ({'✓' if result['correct'] else '✗'})")
            logger.info(f"    Confidence: {result['confidence']:.2f}, Consistency: {result['consistency']:.2f}")
            logger.info(f"    Status: [{status}]")

        # Summarize basic
        basic_correct = sum(1 for r in results["basic"]["questions"] if r["correct"])
        basic_locked = sum(1 for r in results["basic"]["questions"] if r["is_locked_in"])
        basic_mean_conf = np.mean([r["mean_confidence"] for r in results["basic"]["questions"]])
        basic_mean_const = np.mean([r["total_const_matches"] for r in results["basic"]["questions"]])

        results["basic"]["summary"] = {
            "n_questions": len(BASIC_QUESTIONS),
            "n_correct": basic_correct,
            "n_locked_in": basic_locked,
            "accuracy": basic_correct / len(BASIC_QUESTIONS),
            "locked_in_rate": basic_locked / len(BASIC_QUESTIONS),
            "mean_confidence": basic_mean_conf,
            "mean_const_matches": basic_mean_const,
        }

        logger.info(f"\nBasic Operations Summary:")
        logger.info(f"  Accuracy: {basic_correct}/{len(BASIC_QUESTIONS)} ({results['basic']['summary']['accuracy']:.0%})")
        logger.info(f"  Locked In: {basic_locked}/{len(BASIC_QUESTIONS)} ({results['basic']['summary']['locked_in_rate']:.0%})")
        logger.info(f"  Mean Confidence: {basic_mean_conf:.2f}")

        # Final analysis
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS")
        logger.info("=" * 60)

        fund_acc = results["fundamental"]["summary"]["accuracy"]
        basic_acc = results["basic"]["summary"]["accuracy"]
        fund_lock = results["fundamental"]["summary"]["locked_in_rate"]
        basic_lock = results["basic"]["summary"]["locked_in_rate"]

        if fund_acc == 1.0 and fund_lock == 1.0:
            conclusion = "fundamentals_locked_in"
            logger.info("\n*** FUNDAMENTALS ARE LOCKED IN ***")
            logger.info("The model has basic arithmetic structurally correct.")
            if basic_acc < 0.8:
                logger.info(f"But basic operations at {basic_acc:.0%} suggests difficulty with complexity, not foundation.")
        elif fund_acc == 1.0 and fund_lock < 1.0:
            conclusion = "fundamentals_correct_not_locked"
            logger.info("\n*** FUNDAMENTALS CORRECT BUT NOT LOCKED ***")
            logger.info("The model gets fundamentals right but with low consistency/confidence.")
            logger.info("This suggests unstable representation - needs alignment.")
        elif fund_acc < 1.0:
            conclusion = "fundamentals_broken"
            logger.info("\n*** FUNDAMENTALS ARE BROKEN ***")
            logger.info(f"The model gets {fund_acc:.0%} on trivial arithmetic like 2+2.")
            logger.info("This explains why math can't be improved - the foundation is wrong.")
            logger.info("Must align fundamentals FIRST before any capability can be added.")
        else:
            conclusion = "unknown"

        # Compare geometric signatures
        if results["fundamental"]["summary"]["mean_const_matches"] > 0:
            fund_const = results["fundamental"]["summary"]["mean_const_matches"]
            basic_const = results["basic"]["summary"]["mean_const_matches"]
            logger.info(f"\nGeometric Signature:")
            logger.info(f"  Fundamentals mean constant matches: {fund_const:.1f}")
            logger.info(f"  Basic ops mean constant matches: {basic_const:.1f}")
            if fund_const > basic_const * 1.2:
                logger.info("  → Fundamentals have MORE geometric structure")
            elif basic_const > fund_const * 1.2:
                logger.info("  → Basic ops have MORE geometric structure")
            else:
                logger.info("  → Similar geometric structure")

        results["conclusion"] = conclusion
        results["analysis"] = {
            "fundamentals_accuracy": fund_acc,
            "fundamentals_locked_in_rate": fund_lock,
            "basic_accuracy": basic_acc,
            "basic_locked_in_rate": basic_lock,
            "foundation_status": "solid" if fund_acc == 1.0 and fund_lock == 1.0 else (
                "unstable" if fund_acc == 1.0 else "broken"
            ),
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = FundamentalArithmeticChecker(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/fundamental_arithmetic_check.json"
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
