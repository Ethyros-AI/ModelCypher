#!/usr/bin/env python3
"""Crystallization Test - Sharpen Existing Structure, Don't Create New.

Key insight: The model already has latent geometric structure.
Our job isn't to force new structure, but to "crystallize" what's there.

Method:
1. Find SV ratios that are VERY CLOSE to constants (within 1-3%)
2. Apply a TINY nudge to make them exact
3. The nudge should be so small it's essentially noise-level

This is like "denoising" the geometric structure.
If the patterns are real, clarifying them should ONLY help.

Success: Improvement (any amount) with ZERO degradation.
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
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}

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
    # Logic
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


class CrystallizationTest:
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

    def crystallize_layer(
        self,
        layer_idx: int,
        proximity_threshold: float,  # Only touch ratios this close to constants
        max_change_per_sv: float,    # Maximum fractional change to any SV
        max_targets: int = 3
    ) -> int:
        """Crystallize near-constant ratios with minimal perturbation."""
        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)
        min_sv = S[0] * 1e-6

        targets = []
        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 8)):
                if S[j] > max(1e-10, min_sv):
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        error = abs(ratio - const_val) / const_val
                        if error < proximity_threshold:
                            # Calculate required change
                            target_si = const_val * S[j]
                            relative_change = abs(target_si - S[i]) / S[i]
                            targets.append((i, j, const_val, error, relative_change))
                            break

        if not targets:
            return 0

        # Sort by smallest relative change needed (most "almost there")
        targets.sort(key=lambda x: x[4])

        S_modified = S.copy()
        crystallized = 0

        for i, j, target_val, error, rel_change in targets[:max_targets]:
            if S_modified[j] < min_sv:
                continue

            # Only apply if change is within allowed limit
            if rel_change > max_change_per_sv:
                continue

            target_si = target_val * S_modified[j]
            if target_si > S[0] * 2 or target_si < min_sv:
                continue

            S_modified[i] = target_si
            crystallized += 1

        if crystallized > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)
        return crystallized

    def run_experiment(self, n_iterations: int = 5) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("CRYSTALLIZATION TEST - Sharpen Don't Rebuild")
        logger.info("=" * 60)

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test increasingly tight proximity thresholds
        for proximity in [0.03, 0.02, 0.01, 0.005]:
            for max_change in [0.05, 0.03, 0.01]:
                self._reset_weights(layers)
                logger.info(f"\n--- Proximity {proximity:.1%}, Max change {max_change:.1%} ---")

                for _ in range(n_iterations):
                    for layer_idx in layers:
                        self.crystallize_layer(
                            layer_idx,
                            proximity_threshold=proximity,
                            max_change_per_sv=max_change,
                            max_targets=3
                        )

                final = self.evaluate_by_category()
                changes = {k: final[k] - initial[k] for k in initial}
                degraded = [k for k, v in changes.items() if v < -0.01]
                improved = [k for k, v in changes.items() if v > 0.01]

                key = f"prox_{proximity}_change_{max_change}"
                results[key] = {
                    "final": final,
                    "changes": changes,
                    "degraded": degraded,
                    "improved": improved,
                }

                status = "✓" if not degraded else "✗"
                logger.info(f"{status} Final: overall={final['overall']:.1%}, degraded={degraded}, improved={improved}")

        # Find best configuration
        logger.info("\n" + "=" * 60)
        logger.info("BEST CONFIGURATIONS (no degradation)")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items() if not v["degraded"]]
        if no_degrade:
            # Sort by overall improvement
            no_degrade.sort(key=lambda x: x[1]["changes"]["overall"], reverse=True)
            for key, data in no_degrade[:5]:
                logger.info(f"{key}: +{data['changes']['overall']:.1%} overall, improved={data['improved']}")
        else:
            logger.info("No configuration achieved zero degradation")
            # Show least degradation
            by_degradation = sorted(results.items(), key=lambda x: len(x[1]["degraded"]))
            for key, data in by_degradation[:3]:
                logger.info(f"{key}: degraded={data['degraded']}, improved={data['improved']}")

        self._reset_weights(layers)

        return {
            "initial": initial,
            "results": results,
            "no_degradation_configs": [k for k, v in results.items() if not v["degraded"]],
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = CrystallizationTest(model, tokenizer)
    results = test.run_experiment(n_iterations=5)

    # Save
    output_path = "data/crystallization_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
