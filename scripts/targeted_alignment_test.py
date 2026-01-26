#!/usr/bin/env python3
"""Targeted Alignment Test - Light Up New Patterns, Don't Extinguish Old.

Hypothesis: Different SVD dimensions encode different capabilities.
If we can identify which dimensions are "weak" vs "strong", we can:
1. Strengthen weak dimensions (add new patterns)
2. Leave strong dimensions alone (preserve existing knowledge)

Method:
1. For each category, identify which SV indices affect that category
2. Find indices that are "weak" (affecting underperforming categories)
3. Apply alignment ONLY to weak indices
4. Leave "strong" indices untouched

Success: Improve weak categories without degrading strong ones.
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

# Benchmark questions by category
CATEGORY_QUESTIONS = {
    "math": [
        ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
        ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
        ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
        ("What is 3²?", ["6", "9", "12", "27"], 1),
        ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    ],
    "geography": [
        ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
        ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
        ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
        ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
        ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    ],
    "science": [
        ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
        ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
        ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
        ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
        ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    ],
    "history": [
        ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
        ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
        ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
        ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
        ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    ],
    "logic": [
        ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
        ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
        ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
        ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
        ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    ],
    "language": [
        ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
        ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
        ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
        ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
        ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    ],
    "common_sense": [
        ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
        ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
        ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
        ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
        ("How many days are in a week?", ["5", "6", "7", "8"], 2),
    ],
}


class TargetedAlignmentTest:
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

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float]:
        """Returns (is_correct, confidence_margin)."""
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
        predicted_idx = int(np.argmax(scores))

        # Confidence = difference between top and second score
        sorted_scores = sorted(scores, reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0

        return predicted_idx == correct_idx, confidence

    def evaluate_by_category(self) -> Dict[str, float]:
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx)[0])
            results[cat] = correct / len(questions)
        results["overall"] = sum(v for k, v in results.items() if k != "overall") / len(CATEGORY_QUESTIONS)
        return results

    def identify_weak_strong_categories(self) -> Tuple[List[str], List[str]]:
        """Identify which categories are weak vs strong."""
        scores = self.evaluate_by_category()
        weak = [cat for cat, score in scores.items() if cat != "overall" and score < 0.7]
        strong = [cat for cat, score in scores.items() if cat != "overall" and score >= 0.7]
        return weak, strong

    def probe_sv_sensitivity(
        self,
        layer_idx: int,
        category: str,
        sv_indices: List[int],
        delta: float = 0.1
    ) -> Dict[int, float]:
        """Measure how sensitive a category is to perturbations at each SV index."""
        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)
        original_S = S.copy()

        questions = CATEGORY_QUESTIONS[category]
        sensitivities = {}

        for sv_idx in sv_indices:
            # Perturb this singular value
            S_perturbed = original_S.copy()
            S_perturbed[sv_idx] *= (1 + delta)

            W_perturbed = U @ np.diag(S_perturbed) @ Vt
            if not np.all(np.isfinite(W_perturbed)):
                sensitivities[sv_idx] = 0.0
                continue

            self._set_weight(layer_idx, W_perturbed)

            # Measure change in category
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx)[0])
            score = correct / len(questions)

            sensitivities[sv_idx] = score

            # Reset
            self._set_weight(layer_idx, W)

        return sensitivities

    def find_category_neutral_indices(
        self,
        layer_idx: int,
        strong_categories: List[str],
        n_indices: int = 20
    ) -> List[int]:
        """Find SV indices that don't strongly affect any strong category."""
        W = self._get_weight(layer_idx)
        _, S, _ = svd(W, full_matrices=False)

        # Look at first n_indices singular values
        indices_to_test = list(range(min(n_indices, len(S))))

        # For each strong category, measure sensitivity
        safe_indices = set(indices_to_test)

        for cat in strong_categories:
            base_questions = CATEGORY_QUESTIONS[cat]
            base_score = sum(1 for q, c, idx in base_questions if self._evaluate_question(q, c, idx)[0]) / len(base_questions)

            sensitivities = self.probe_sv_sensitivity(layer_idx, cat, indices_to_test)

            # Remove indices where perturbation significantly changes score
            for sv_idx, perturbed_score in sensitivities.items():
                if abs(perturbed_score - base_score) > 0.2:  # >20% change = sensitive
                    safe_indices.discard(sv_idx)

        return list(safe_indices)

    def targeted_align_layer(
        self,
        layer_idx: int,
        safe_indices: List[int],
        max_targets: int = 2
    ) -> int:
        """Apply alignment only to safe (category-neutral) indices."""
        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)
        min_sv = S[0] * 1e-6

        targets = []
        # Only look for targets among SAFE indices
        for i in safe_indices:
            for j in safe_indices:
                if j > i and j < len(S) and S[j] > max(1e-10, min_sv):
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
            S_modified[i] = new_val
            aligned += 1

        if aligned > 0 and np.all(np.isfinite(S_modified)):
            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer_idx, W_modified)
        return aligned

    def run_experiment(self, n_iterations: int = 10) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("TARGETED ALIGNMENT - Light Up New, Preserve Old")
        logger.info("=" * 60)

        # Step 1: Identify weak vs strong categories
        initial = self.evaluate_by_category()
        logger.info(f"\nInitial scores: {initial}")

        weak_cats, strong_cats = self.identify_weak_strong_categories()
        logger.info(f"\nWeak categories (<70%): {weak_cats}")
        logger.info(f"Strong categories (≥70%): {strong_cats}")

        # Step 2: For each layer, find category-neutral indices
        logger.info("\nFinding category-neutral SV indices...")
        layer_safe_indices = {}

        for layer_idx in layers:
            safe = self.find_category_neutral_indices(layer_idx, strong_cats, n_indices=15)
            layer_safe_indices[layer_idx] = safe
            logger.info(f"  Layer {layer_idx}: {len(safe)} safe indices")

        # Reset after probing
        self._reset_weights(layers)

        # Step 3: Apply targeted alignment only to safe indices
        logger.info("\n--- TARGETED ALIGNMENT (safe indices only) ---")

        for iteration in range(n_iterations):
            total_aligned = 0
            for layer_idx in layers:
                aligned = self.targeted_align_layer(
                    layer_idx,
                    layer_safe_indices.get(layer_idx, []),
                    max_targets=2
                )
                total_aligned += aligned

            if iteration % 3 == 0:
                result = self.evaluate_by_category()
                logger.info(f"Iteration {iteration+1}: {result}")

        final = self.evaluate_by_category()
        logger.info(f"\nFinal: {final}")

        changes = {k: final[k] - initial[k] for k in initial}
        degraded = [k for k, v in changes.items() if v < -0.01]
        improved = [k for k, v in changes.items() if v > 0.01]

        logger.info(f"\nChanges: {changes}")
        logger.info(f"Degraded: {degraded}")
        logger.info(f"Improved: {improved}")

        # Compare with blind alignment
        self._reset_weights(layers)
        logger.info("\n--- BLIND ALIGNMENT (for comparison) ---")

        for iteration in range(n_iterations):
            for layer_idx in layers:
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

                S_modified = S.copy()
                for i, j, target_val in targets[:2]:
                    if S_modified[j] < min_sv:
                        continue
                    new_val = target_val * S_modified[j]
                    if new_val > S[0] * 10 or new_val < min_sv:
                        continue
                    S_modified[i] = new_val

                if np.all(np.isfinite(S_modified)):
                    W_modified = U @ np.diag(S_modified) @ Vt
                    if np.all(np.isfinite(W_modified)):
                        self._set_weight(layer_idx, W_modified)

        blind_final = self.evaluate_by_category()
        blind_changes = {k: blind_final[k] - initial[k] for k in initial}
        blind_degraded = [k for k, v in blind_changes.items() if v < -0.01]

        logger.info(f"Blind final: {blind_final}")
        logger.info(f"Blind degraded: {blind_degraded}")

        self._reset_weights(layers)

        return {
            "initial": initial,
            "weak_categories": weak_cats,
            "strong_categories": strong_cats,
            "targeted": {
                "final": final,
                "changes": changes,
                "degraded": degraded,
                "improved": improved,
            },
            "blind": {
                "final": blind_final,
                "changes": blind_changes,
                "degraded": blind_degraded,
            },
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = TargetedAlignmentTest(model, tokenizer)
    results = test.run_experiment(n_iterations=10)

    logger.info("\n" + "=" * 60)
    logger.info("CONCLUSION")
    logger.info("=" * 60)

    targeted_deg = len(results["targeted"]["degraded"])
    blind_deg = len(results["blind"]["degraded"])

    if targeted_deg < blind_deg:
        logger.info(f"SUCCESS: Targeted alignment degraded {targeted_deg} categories vs blind's {blind_deg}")
    elif targeted_deg == 0 and len(results["targeted"]["improved"]) > 0:
        logger.info("SUCCESS: Improvement with ZERO degradation!")
    else:
        logger.info("PARTIAL: Targeted alignment didn't fully solve degradation")

    # Save
    output_path = "data/targeted_alignment_test.json"
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
