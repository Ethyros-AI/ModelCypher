#!/usr/bin/env python3
"""Iteration Tracking Test - When Exactly Does Improvement/Degradation Happen?

The original experiment showed:
- After 10 iterations: language 60% → 100%, geography 100% → 80%

Question: At which iteration does each category CHANGE?
- Does language improve before geography degrades?
- Or do they change simultaneously?
- Is there a window where language improved but geography hasn't degraded yet?

This tells us whether the changes are fundamentally entangled or sequential.
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


CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "sqrt3": np.sqrt(3),
}

BENCHMARK_QUESTIONS = [
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
    ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
    ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
    ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
    ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
    ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
    ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
    ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
    ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
    ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
    ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
    ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
    ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
    ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
    ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
    ("How many days are in a week?", ["5", "6", "7", "8"], 2),
]


class IterationTrackingTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_weights = {}

    def _get_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        new_weight = mx.array(weights.astype(np.float32))
        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight
        mx.eval(new_weight)

    def _cache_weights(self, layers: List[int]):
        self._original_weights = {i: self._get_weight(i).copy() for i in layers}

    def _reset_weights(self, layers: List[int]):
        for i in layers:
            if i in self._original_weights:
                self._set_weight(i, self._original_weights[i])

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> bool:
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
        return int(np.argmax(scores)) == correct_idx

    def evaluate_by_category(self) -> Dict[str, float]:
        categories = {
            "math": BENCHMARK_QUESTIONS[0:5],
            "geography": BENCHMARK_QUESTIONS[5:10],
            "science": BENCHMARK_QUESTIONS[10:15],
            "history": BENCHMARK_QUESTIONS[15:20],
            "logic": BENCHMARK_QUESTIONS[20:25],
            "language": BENCHMARK_QUESTIONS[25:30],
            "common_sense": BENCHMARK_QUESTIONS[30:35],
        }
        results = {}
        for cat, questions in categories.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx))
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def align_layer_full(self, layer_idx: int, max_targets: int = 2) -> int:
        """Apply FULL alignment (original method) - not scaled."""
        W = self._get_weight(layer_idx)  # Get CURRENT weights
        U, S, Vt = svd(W, full_matrices=False)
        min_sv = S[0] * 1e-6

        targets = []
        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > max(1e-10, min_sv):
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        if abs(ratio - const_val) / const_val < 0.10:
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
            S_modified[i] = new_val  # FULL alignment, not interpolated
            aligned += 1

        if aligned > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)

        return aligned

    def run_experiment(self, n_iterations: int = 15) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("ITERATION TRACKING TEST")
        logger.info("=" * 60)

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: language={initial['language']:.0%}, geography={initial['geography']:.0%}")

        history = [{"iteration": 0, **initial}]

        logger.info("\nIteration-by-iteration tracking:")
        logger.info("  Iter | Language | Geography | Math | Science | Logic | History | CS | Overall")
        logger.info("-" * 85)

        # Track first change for each category
        first_change = {}

        for iteration in range(1, n_iterations + 1):
            # Apply one iteration of full alignment
            for layer_idx in layers:
                self.align_layer_full(layer_idx, max_targets=2)

            result = self.evaluate_by_category()
            history.append({"iteration": iteration, **result})

            # Track first changes
            for cat in result:
                if cat not in first_change and cat != "overall":
                    if result[cat] != initial[cat]:
                        first_change[cat] = {
                            "iteration": iteration,
                            "initial": initial[cat],
                            "new": result[cat],
                            "direction": "+" if result[cat] > initial[cat] else "-"
                        }

            # Log
            logger.info(
                f"  {iteration:4d} | "
                f"{result['language']:7.0%} | "
                f"{result['geography']:9.0%} | "
                f"{result['math']:4.0%} | "
                f"{result['science']:7.0%} | "
                f"{result['logic']:5.0%} | "
                f"{result['history']:7.0%} | "
                f"{result['common_sense']:2.0%} | "
                f"{result['overall']:7.1%}"
            )

        # Analysis
        logger.info("\n" + "=" * 60)
        logger.info("WHEN CHANGES HAPPENED")
        logger.info("=" * 60)

        for cat in sorted(first_change.keys(), key=lambda x: first_change[x]["iteration"]):
            fc = first_change[cat]
            logger.info(
                f"  {cat}: iteration {fc['iteration']} "
                f"({fc['initial']:.0%} → {fc['new']:.0%}, {fc['direction']})"
            )

        # Key question: does language improve BEFORE geography degrades?
        lang_first = first_change.get("language", {}).get("iteration", 999)
        geo_first = first_change.get("geography", {}).get("iteration", 999)

        logger.info("\n" + "=" * 60)
        logger.info("KEY FINDING")
        logger.info("=" * 60)

        if lang_first < geo_first:
            logger.info(f"Language improved at iteration {lang_first}")
            logger.info(f"Geography degraded at iteration {geo_first}")
            logger.info("*** WINDOW EXISTS: language improved before geography degraded! ***")
        elif lang_first > geo_first:
            logger.info(f"Geography degraded at iteration {geo_first}")
            logger.info(f"Language improved at iteration {lang_first}")
            logger.info("*** DEGRADATION FIRST: geography degraded before language improved ***")
        elif lang_first == geo_first and lang_first != 999:
            logger.info(f"Both changed at iteration {lang_first}")
            logger.info("*** SIMULTANEOUS: changes are fundamentally entangled ***")
        else:
            logger.info("One or both categories didn't change")

        self._reset_weights(layers)

        return {
            "initial": initial,
            "history": history,
            "first_change": first_change,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = IterationTrackingTest(model, tokenizer)
    results = test.run_experiment(n_iterations=15)

    # Save
    output_path = "data/iteration_tracking_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
