#!/usr/bin/env python3
"""Dormant Activation Test - Light Up Unused Regions.

Key insight: "Learning is lighting up new patterns and regions."

Hypothesis: The model has dormant capacity in SV indices that DON'T
currently have constant ratios. Creating structure there might activate
new capability without disturbing existing knowledge.

Method:
1. Find SV index ranges with NO near-constant ratios (dormant)
2. Restructure those dormant SVs to HAVE constant ratios
3. Measure if this activates new capability

This is the opposite of crystallization - we're not sharpening
existing patterns, we're creating new ones in unused space.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


class DormantActivationTest:
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

    def find_active_indices(self, S: np.ndarray, proximity: float = 0.10) -> Set[int]:
        """Find indices that participate in near-constant ratios."""
        active = set()
        for i in range(min(len(S) - 1, 50)):
            for j in range(i + 1, min(len(S), i + 10)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        if abs(ratio - const_val) / const_val < proximity:
                            active.add(i)
                            active.add(j)
                            break
        return active

    def find_dormant_indices(self, S: np.ndarray, proximity: float = 0.10) -> List[int]:
        """Find indices NOT participating in any constant ratio."""
        active = self.find_active_indices(S, proximity)
        min_sv = S[0] * 1e-4  # Only consider reasonably-sized SVs

        dormant = []
        for i in range(min(len(S), 50)):
            if i not in active and S[i] > min_sv:
                dormant.append(i)
        return dormant

    def activate_dormant_layer(
        self,
        layer_idx: int,
        max_activations: int = 3,
        scale_factor: float = 1.0
    ) -> int:
        """Create constant ratios among dormant indices."""
        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        dormant = self.find_dormant_indices(S)
        if len(dormant) < 2:
            return 0

        S_modified = S.copy()
        activated = 0

        # Create constant ratios among dormant indices
        const_vals = list(CONSTANTS.values())
        for i in range(min(len(dormant) - 1, max_activations)):
            idx_i = dormant[i]
            idx_j = dormant[i + 1]

            # Choose a constant to create
            target_const = const_vals[i % len(const_vals)]

            # Set S[idx_i] = target_const * S[idx_j]
            # But scaled to stay in reasonable range
            reference_sv = S_modified[idx_j]
            new_sv = target_const * reference_sv * scale_factor

            # Ensure we stay within bounds
            if new_sv > S[0] * 0.5:  # Don't exceed 50% of max
                new_sv = S[0] * 0.5
            if new_sv < S[-1] * 2:  # Don't go below 2x min
                new_sv = S[-1] * 2

            S_modified[idx_i] = new_sv
            activated += 1

        if activated > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)
        return activated

    def analyze_layer_structure(self, layer_idx: int) -> Dict:
        """Analyze the active/dormant structure of a layer."""
        W = self._get_weight(layer_idx)
        _, S, _ = svd(W, full_matrices=False)

        active = self.find_active_indices(S)
        dormant = self.find_dormant_indices(S)

        # Compute what % of variance is in active vs dormant
        total_var = np.sum(S ** 2)
        active_var = np.sum(S[list(active)] ** 2) if active else 0
        dormant_var = np.sum(S[dormant] ** 2) if dormant else 0

        return {
            "n_active": len(active),
            "n_dormant": len(dormant),
            "active_variance_pct": active_var / total_var * 100,
            "dormant_variance_pct": dormant_var / total_var * 100,
            "active_indices": sorted(list(active))[:10],
            "dormant_indices": dormant[:10],
        }

    def run_experiment(self, n_iterations: int = 5) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("DORMANT ACTIVATION TEST - Light Up New Regions")
        logger.info("=" * 60)

        # First, analyze structure
        logger.info("\nLayer structure analysis:")
        for layer_idx in layers:
            analysis = self.analyze_layer_structure(layer_idx)
            logger.info(
                f"  Layer {layer_idx}: {analysis['n_active']} active, "
                f"{analysis['n_dormant']} dormant, "
                f"active={analysis['active_variance_pct']:.1f}% var, "
                f"dormant={analysis['dormant_variance_pct']:.1f}% var"
            )

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test different scale factors
        for scale in [0.5, 1.0, 2.0]:
            for max_act in [1, 3, 5]:
                self._reset_weights(layers)
                logger.info(f"\n--- Scale {scale}, Max activations {max_act} ---")

                total_activated = 0
                for _ in range(n_iterations):
                    for layer_idx in layers:
                        total_activated += self.activate_dormant_layer(
                            layer_idx,
                            max_activations=max_act,
                            scale_factor=scale
                        )

                final = self.evaluate_by_category()
                changes = {k: final[k] - initial[k] for k in initial}
                degraded = [k for k, v in changes.items() if v < -0.01]
                improved = [k for k, v in changes.items() if v > 0.01]

                key = f"scale_{scale}_max_{max_act}"
                results[key] = {
                    "final": final,
                    "changes": changes,
                    "degraded": degraded,
                    "improved": improved,
                    "total_activated": total_activated,
                }

                status = "✓" if not degraded else "✗"
                logger.info(
                    f"{status} Final: overall={final['overall']:.1%}, "
                    f"degraded={degraded}, improved={improved}, "
                    f"activated={total_activated}"
                )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items() if not v["degraded"]]
        has_improve = [(k, v) for k, v in results.items() if v["improved"]]
        both = [(k, v) for k, v in results.items() if not v["degraded"] and v["improved"]]

        logger.info(f"Configs with NO degradation: {len(no_degrade)}")
        logger.info(f"Configs with improvement: {len(has_improve)}")
        logger.info(f"Configs with BOTH (the goal): {len(both)}")

        if both:
            logger.info("\nSUCCESS - Found configurations with improvement and no degradation:")
            for key, data in both:
                logger.info(f"  {key}: improved={data['improved']}")
        elif no_degrade:
            logger.info("\nNo improvement achieved, but these configs preserved quality:")
            for key, data in no_degrade[:3]:
                logger.info(f"  {key}")

        self._reset_weights(layers)

        return {
            "initial": initial,
            "results": results,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = DormantActivationTest(model, tokenizer)
    results = test.run_experiment(n_iterations=5)

    # Save
    output_path = "data/dormant_activation_test.json"
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
