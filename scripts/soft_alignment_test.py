#!/usr/bin/env python3
"""Soft Alignment Test - Nudge Don't Force.

Hypothesis: Hard alignment (forcing exact ratios) destroys information.
Soft alignment (nudging toward ratios) should integrate without destruction.

Test: Compare hard vs soft alignment on per-category benchmark.
Success criterion: No category degrades while overall improves.
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

# Same benchmark as before
BENCHMARK_QUESTIONS = [
    # Math (5)
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    # Geography (5)
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
    ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
    ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    # Science (5)
    ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
    ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    # History (5)
    ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
    ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
    ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    # Logic (5)
    ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
    ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
    ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
    ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    # Language (5)
    ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
    ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
    ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
    ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    # Common sense (5)
    ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
    ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
    ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
    ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
    ("How many days are in a week?", ["5", "6", "7", "8"], 2),
]


class SoftAlignmentTest:
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

    def hard_align_layer(self, layer_idx: int, max_targets: int = 2) -> int:
        """Original hard alignment - forces exact ratios."""
        W = self._get_weight(layer_idx)
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
            # HARD: Force exact ratio
            S_modified[i] = new_val
            aligned += 1

        if aligned > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)
        return aligned

    def soft_align_layer(self, layer_idx: int, alpha: float = 0.5, max_targets: int = 2) -> int:
        """Soft alignment - interpolates toward target ratio."""
        W = self._get_weight(layer_idx)
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
            target_si = target_val * S_modified[j]
            if target_si > S[0] * 10 or target_si < min_sv:
                continue
            # SOFT: Interpolate between original and target
            S_modified[i] = alpha * target_si + (1 - alpha) * S[i]
            aligned += 1

        if aligned > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)
        return aligned

    def run_comparison(self, n_iterations: int = 10) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        logger.info("=" * 60)
        logger.info("HARD vs SOFT ALIGNMENT COMPARISON")
        logger.info("=" * 60)

        # Test HARD alignment
        self._cache_weights(layers)
        logger.info("\n--- HARD ALIGNMENT (α=1.0, force exact) ---")
        initial = self.evaluate_by_category()
        logger.info(f"Initial: {initial}")

        for i in range(n_iterations):
            for layer_idx in layers:
                self.hard_align_layer(layer_idx)

        hard_result = self.evaluate_by_category()
        logger.info(f"After hard: {hard_result}")

        hard_changes = {k: hard_result[k] - initial[k] for k in initial}
        hard_degraded = [k for k, v in hard_changes.items() if v < -0.01]
        logger.info(f"Hard degraded: {hard_degraded}")

        # Test SOFT alignment with different alpha values
        soft_results = {}
        for alpha in [0.3, 0.5, 0.7]:
            self._reset_weights(layers)
            logger.info(f"\n--- SOFT ALIGNMENT (α={alpha}) ---")

            for i in range(n_iterations):
                for layer_idx in layers:
                    self.soft_align_layer(layer_idx, alpha=alpha)

            soft_result = self.evaluate_by_category()
            logger.info(f"After soft α={alpha}: {soft_result}")

            soft_changes = {k: soft_result[k] - initial[k] for k in initial}
            soft_degraded = [k for k, v in soft_changes.items() if v < -0.01]
            logger.info(f"Soft α={alpha} degraded: {soft_degraded}")

            soft_results[alpha] = {
                "result": soft_result,
                "changes": soft_changes,
                "degraded": soft_degraded,
            }

        self._reset_weights(layers)

        return {
            "initial": initial,
            "hard": {"result": hard_result, "changes": hard_changes, "degraded": hard_degraded},
            "soft": soft_results,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = SoftAlignmentTest(model, tokenizer)
    results = test.run_comparison(n_iterations=10)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nInitial overall: {results['initial']['overall']:.1%}")
    logger.info(f"Hard overall: {results['hard']['result']['overall']:.1%} (degraded: {results['hard']['degraded']})")

    for alpha, data in results['soft'].items():
        logger.info(f"Soft α={alpha} overall: {data['result']['overall']:.1%} (degraded: {data['degraded']})")

    # Find best alpha (highest overall with no degradation)
    best_alpha = None
    best_overall = results['initial']['overall']
    for alpha, data in results['soft'].items():
        if not data['degraded'] and data['result']['overall'] > best_overall:
            best_alpha = alpha
            best_overall = data['result']['overall']

    if best_alpha:
        logger.info(f"\nBEST: Soft α={best_alpha} improves to {best_overall:.1%} with NO degradation")
    else:
        logger.info(f"\nNo soft alpha achieved improvement without degradation")

    # Save results
    output_path = "data/soft_alignment_test.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
