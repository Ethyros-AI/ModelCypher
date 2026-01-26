#!/usr/bin/env python3
"""Benchmark Aligned Model on MMLU-style Questions.

Tests whether surgical SVD alignment produces real capability improvement
on a diverse set of multiple-choice questions, not just our 5 test prompts.

Usage:
    poetry run python scripts/benchmark_aligned_model.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 10 \
        --output data/benchmark/mmlu_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# MMLU-style benchmark questions (diverse subjects)
# Each tuple: (question, choices, correct_answer_index)
BENCHMARK_QUESTIONS = [
    # Math
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),

    # Geography
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
    ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
    ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),

    # Science
    ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
    ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),

    # History
    ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
    ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
    ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),

    # Logic/Reasoning
    ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
    ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
    ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
    ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),

    # Language
    ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
    ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
    ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
    ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),

    # Common sense
    ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
    ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
    ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
    ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
    ("How many days are in a week?", ["5", "6", "7", "8"], 2),
]

CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}


@dataclass
class BenchmarkResult:
    initial_accuracy: float
    final_accuracy: float
    initial_matches: int
    final_matches: int
    iterations: int
    trajectory: List[Dict]
    per_category_results: Dict[str, Dict]


class BenchmarkEvaluator:
    """Evaluate model on MMLU-style benchmark before/after alignment."""

    def __init__(self, model, tokenizer, proximity_threshold: float = 0.10):
        self.model = model
        self.tokenizer = tokenizer
        self.proximity_threshold = proximity_threshold
        self.n_layers = len(model.model.layers)

    def _get_mlp_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        if hasattr(mlp, 'gate_proj'):
            w = mlp.gate_proj.weight
        elif hasattr(mlp, 'w1'):
            w = mlp.w1.weight
        else:
            w = mlp.weight

        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_mlp_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        new_weight = mx.array(weights.astype(np.float32))

        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight

        mx.eval(new_weight)

    def _count_matches(self, S: np.ndarray) -> int:
        count = 0
        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 6)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        if abs(ratio - const_val) / const_val < 0.05:
                            count += 1
                            break
        return count

    def _count_total_matches(self, layer_indices: List[int]) -> int:
        total = 0
        for layer_idx in layer_indices:
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)
            total += self._count_matches(S)
        return total

    def _surgical_align_layer(self, layer_idx: int, max_targets: int = 2) -> int:
        W = self._get_mlp_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        min_sv = S[0] * 1e-6
        targets = []

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > max(1e-10, min_sv):
                    ratio = S[i] / S[j]
                    for const_name, const_val in CONSTANTS.items():
                        error = abs(ratio - const_val) / const_val
                        if error < self.proximity_threshold:
                            targets.append((i, j, const_val))
                            break

        if not targets:
            return 0

        S_modified = S.copy()
        aligned = 0

        for i, j, target_val in targets[:max_targets]:
            if S_modified[j] < min_sv:
                continue
            new_val = target_val * S_modified[j]
            if new_val > S[0] * 10 or new_val < min_sv:
                continue
            S_modified[i] = new_val
            aligned += 1

        if aligned > 0:
            if not np.all(np.isfinite(S_modified)):
                return 0
            W_modified = U @ np.diag(S_modified) @ Vt
            if not np.all(np.isfinite(W_modified)):
                return 0
            self._set_mlp_weight(layer_idx, W_modified)

        return aligned

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, int]:
        """Evaluate a single multiple-choice question. Returns (correct, predicted_idx)."""
        import mlx.core as mx

        # Format as multiple choice
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get logits for next token
        logits = self.model(input_ids)
        mx.eval(logits)

        # Get probabilities for A, B, C, D tokens
        next_logits = logits[0, -1, :]

        # Find token IDs for A, B, C, D
        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            # Try different tokenizations
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])  # Last token
                    break
            else:
                choice_tokens.append(0)  # Fallback

        # Get scores for each choice
        scores = [float(next_logits[t].item()) for t in choice_tokens]

        # Predict highest scoring choice
        predicted_idx = int(np.argmax(scores))
        correct = predicted_idx == correct_idx

        return correct, predicted_idx

    def evaluate_benchmark(self) -> Tuple[float, Dict[str, Dict]]:
        """Evaluate on full benchmark. Returns (accuracy, per_category_results)."""

        categories = {
            "math": BENCHMARK_QUESTIONS[0:5],
            "geography": BENCHMARK_QUESTIONS[5:10],
            "science": BENCHMARK_QUESTIONS[10:15],
            "history": BENCHMARK_QUESTIONS[15:20],
            "logic": BENCHMARK_QUESTIONS[20:25],
            "language": BENCHMARK_QUESTIONS[25:30],
            "common_sense": BENCHMARK_QUESTIONS[30:35],
        }

        total_correct = 0
        total_questions = 0
        per_category = {}

        for category, questions in categories.items():
            cat_correct = 0
            for question, choices, correct_idx in questions:
                is_correct, _ = self._evaluate_question(question, choices, correct_idx)
                if is_correct:
                    cat_correct += 1
                    total_correct += 1
                total_questions += 1

            per_category[category] = {
                "correct": cat_correct,
                "total": len(questions),
                "accuracy": cat_correct / len(questions),
            }

        overall_accuracy = total_correct / total_questions
        return overall_accuracy, per_category

    def run(
        self,
        n_iterations: int = 10,
        layer_indices: Optional[List[int]] = None,
    ) -> BenchmarkResult:
        """Run benchmark before/after surgical alignment."""

        if layer_indices is None:
            mid = self.n_layers // 2
            layer_indices = list(range(mid - 3, mid + 4))

        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK EVALUATION")
        logger.info(f"Questions: {len(BENCHMARK_QUESTIONS)}")
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"Layers: {layer_indices}")
        logger.info("=" * 60)

        # Initial evaluation
        initial_accuracy, initial_per_cat = self.evaluate_benchmark()
        initial_matches = self._count_total_matches(layer_indices)

        logger.info(f"\nInitial state:")
        logger.info(f"  Accuracy: {initial_accuracy:.1%} ({int(initial_accuracy * len(BENCHMARK_QUESTIONS))}/{len(BENCHMARK_QUESTIONS)})")
        logger.info(f"  Matches: {initial_matches}")
        logger.info("  Per category:")
        for cat, results in initial_per_cat.items():
            logger.info(f"    {cat}: {results['correct']}/{results['total']} ({results['accuracy']:.0%})")

        trajectory = []

        for iteration in range(n_iterations):
            # Surgical alignment
            for layer_idx in layer_indices:
                self._surgical_align_layer(layer_idx, max_targets=2)

            matches = self._count_total_matches(layer_indices)
            accuracy, per_cat = self.evaluate_benchmark()

            trajectory.append({
                "iteration": iteration + 1,
                "matches": matches,
                "accuracy": accuracy,
            })

            logger.info(f"  Iter {iteration+1}: {accuracy:.1%} accuracy, {matches} matches")

        # Final evaluation
        final_accuracy, final_per_cat = self.evaluate_benchmark()
        final_matches = self._count_total_matches(layer_indices)

        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Accuracy: {initial_accuracy:.1%} → {final_accuracy:.1%}")
        logger.info(f"Matches: {initial_matches} → {final_matches}")
        logger.info("\nPer category comparison:")
        for cat in initial_per_cat:
            init = initial_per_cat[cat]['accuracy']
            final = final_per_cat[cat]['accuracy']
            delta = final - init
            logger.info(f"  {cat}: {init:.0%} → {final:.0%} ({delta:+.0%})")

        return BenchmarkResult(
            initial_accuracy=initial_accuracy,
            final_accuracy=final_accuracy,
            initial_matches=initial_matches,
            final_matches=final_matches,
            iterations=n_iterations,
            trajectory=trajectory,
            per_category_results={
                "initial": initial_per_cat,
                "final": final_per_cat,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--proximity", type=float, default=0.10)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    from mlx_lm import load

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    evaluator = BenchmarkEvaluator(
        model=model,
        tokenizer=tokenizer,
        proximity_threshold=args.proximity,
    )

    result = evaluator.run(n_iterations=args.iterations)

    # Save results
    output_path = args.output or f"data/benchmark/mmlu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "experiment": "mmlu_benchmark",
        "n_questions": len(BENCHMARK_QUESTIONS),
        "initial_accuracy": result.initial_accuracy,
        "final_accuracy": result.final_accuracy,
        "accuracy_improvement": result.final_accuracy - result.initial_accuracy,
        "initial_matches": result.initial_matches,
        "final_matches": result.final_matches,
        "iterations": result.iterations,
        "trajectory": result.trajectory,
        "per_category": result.per_category_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    if result.final_accuracy > result.initial_accuracy:
        improvement = (result.final_accuracy - result.initial_accuracy) * 100
        logger.info(f"\nSUCCESS: Accuracy improved by {improvement:.1f}%")
    elif result.final_accuracy == result.initial_accuracy:
        logger.info(f"\nNO CHANGE: Accuracy unchanged")
    else:
        degradation = (result.initial_accuracy - result.final_accuracy) * 100
        logger.info(f"\nDEGRADED: Accuracy decreased by {degradation:.1f}%")


if __name__ == "__main__":
    main()
