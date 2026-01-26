#!/usr/bin/env python3
"""Experiment 22: All-Category Optimization.

Find a direction that improves ALL weak categories while preserving ALL strong ones.

Categories:
- Weak (to improve): math (20%), language (60%), logic (60%)
- Strong (to preserve): geography (100%), history (100%), common_sense (100%), science (80%)

Method:
1. Compute gradients for all categories
2. Build preservation subspace from strong category gradients
3. Find combined improvement direction in orthogonal subspace
4. Apply and measure
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


class AllCategoryOptimization:
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
        prediction = int(np.argmax(scores))

        scores_np = np.array(scores)
        probs = np.exp(scores_np - np.max(scores_np))
        probs = probs / probs.sum()
        loss = -np.log(probs[correct_idx] + 1e-10)

        return prediction == correct_idx, float(loss)

    def evaluate_by_category(self) -> Dict[str, float]:
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx)[0])
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def compute_loss_gradient(
        self,
        layer_idx: int,
        category: str,
        epsilon: float = 0.01
    ) -> np.ndarray:
        W = self._original_weights[layer_idx]
        questions = CATEGORY_QUESTIONS[category]

        self._set_weight(layer_idx, W)
        base_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)

        U, S, Vt = svd(W, full_matrices=False)
        k = min(20, len(S))
        gradient = np.zeros(k)

        for i in range(k):
            S_perturbed = S.copy()
            S_perturbed[i] += epsilon * S[i]

            W_perturbed = U @ np.diag(S_perturbed) @ Vt
            if np.all(np.isfinite(W_perturbed)):
                self._set_weight(layer_idx, W_perturbed)
                perturbed_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)
                gradient[i] = (perturbed_loss - base_loss) / (epsilon * S[i])

        self._set_weight(layer_idx, W)
        return gradient

    def compute_weighted_improvement_direction(
        self,
        gradients: Dict[str, np.ndarray],
        weak_cats: List[str],
        baselines: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute weighted improvement direction based on how much room for improvement.

        Categories with lower baseline get higher weight.
        """
        direction = np.zeros_like(gradients[weak_cats[0]])

        for cat in weak_cats:
            # Weight by inverse of baseline (more room = more weight)
            room = 1.0 - baselines[cat]
            weight = room / sum(1.0 - baselines[c] for c in weak_cats)
            direction += weight * gradients[cat]

        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return direction

    def project_to_safe_subspace(
        self,
        direction: np.ndarray,
        preserve_gradients: List[np.ndarray]
    ) -> np.ndarray:
        """Project direction into subspace orthogonal to all preserve gradients."""
        result = direction.copy()

        for preserve_grad in preserve_gradients:
            norm = np.linalg.norm(preserve_grad)
            if norm > 1e-10:
                preserve_normalized = preserve_grad / norm
                projection = np.dot(result, preserve_normalized) * preserve_normalized
                result = result - projection

        return result

    def apply_modification(
        self,
        layer_idx: int,
        direction: np.ndarray,
        scale: float
    ) -> bool:
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        S_modified = S.copy()
        for i in range(min(len(direction), len(S))):
            S_modified[i] -= scale * direction[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            return True
        return False

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layer_idx = mid

        self._cache_weights([layer_idx])

        logger.info("=" * 60)
        logger.info("ALL-CATEGORY OPTIMIZATION")
        logger.info("=" * 60)
        logger.info(f"Testing layer: {layer_idx}")

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial accuracies:")
        for cat, acc in sorted(initial.items(), key=lambda x: x[1]):
            if cat != "overall":
                logger.info(f"  {cat}: {acc:.0%}")
        logger.info(f"  overall: {initial['overall']:.0%}")

        # Define weak (improve) and strong (preserve) categories
        weak_cats = ["math", "language", "logic"]
        strong_cats = ["geography", "history", "common_sense", "science"]

        logger.info(f"\nWeak categories (improve): {weak_cats}")
        logger.info(f"Strong categories (preserve): {strong_cats}")

        results = {
            "layer": layer_idx,
            "initial": initial,
            "weak_cats": weak_cats,
            "strong_cats": strong_cats,
            "experiments": {},
        }

        # Compute all gradients
        logger.info("\nComputing gradients...")
        gradients = {}
        for cat in weak_cats + strong_cats:
            logger.info(f"  {cat}...")
            gradients[cat] = self.compute_loss_gradient(layer_idx, cat)

        # Compute weighted improvement direction
        improve_dir = self.compute_weighted_improvement_direction(
            gradients, weak_cats, initial
        )
        logger.info(f"\nComputed weighted improvement direction")
        logger.info(f"  Weights based on room for improvement:")
        for cat in weak_cats:
            room = 1.0 - initial[cat]
            weight = room / sum(1.0 - initial[c] for c in weak_cats)
            logger.info(f"    {cat}: {weight:.1%} (room={room:.0%})")

        # Compute preservation gradients and project
        preserve_grads = [gradients[cat] for cat in strong_cats]
        safe_dir = self.project_to_safe_subspace(improve_dir, preserve_grads)
        safe_norm = np.linalg.norm(safe_dir)

        logger.info(f"\nSafe direction norm after projection: {safe_norm:.2%}")

        if safe_norm < 0.1:
            logger.info("Safe direction too small - all categories are entangled")
            results["failed"] = "entangled"
            return results

        safe_dir = safe_dir / safe_norm

        # Test different scales
        scales = [0.05, 0.1, 0.15, 0.2, 0.3]
        best_result = None
        best_overall_gain = -float('inf')

        for scale in scales:
            self._reset_weights([layer_idx])
            logger.info(f"\n--- Scale: {scale} ---")

            success = self.apply_modification(layer_idx, safe_dir, scale)
            if not success:
                logger.info("  Modification failed")
                continue

            final = self.evaluate_by_category()
            changes = {k: final[k] - initial[k] for k in initial}

            # Log all categories
            logger.info("  Results:")
            for cat in weak_cats:
                status = "✓" if changes[cat] > 0 else ("=" if changes[cat] == 0 else "✗")
                logger.info(f"    {cat}: {final[cat]:.0%} ({changes[cat]:+.0%}) {status}")
            for cat in strong_cats:
                status = "✓" if changes[cat] >= -0.01 else "✗"
                logger.info(f"    {cat}: {final[cat]:.0%} ({changes[cat]:+.0%}) {status}")

            # Check success criteria
            weak_improved = sum(1 for cat in weak_cats if changes[cat] > 0)
            strong_preserved = sum(1 for cat in strong_cats if changes[cat] >= -0.01)

            logger.info(f"  Improved {weak_improved}/{len(weak_cats)} weak categories")
            logger.info(f"  Preserved {strong_preserved}/{len(strong_cats)} strong categories")

            # Calculate overall gain (sum of weak improvements, penalize strong degradation)
            weak_gain = sum(changes[cat] for cat in weak_cats)
            strong_loss = sum(min(0, changes[cat]) for cat in strong_cats)
            overall_gain = weak_gain + strong_loss * 2  # Penalize preservation failure

            results["experiments"][f"scale_{scale}"] = {
                "final": final,
                "changes": changes,
                "weak_improved": weak_improved,
                "strong_preserved": strong_preserved,
                "overall_gain": overall_gain,
            }

            # Track best (must preserve all strong)
            if strong_preserved == len(strong_cats) and overall_gain > best_overall_gain:
                best_overall_gain = overall_gain
                best_result = {
                    "scale": scale,
                    "final": final,
                    "changes": changes,
                    "weak_improved": weak_improved,
                }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if best_result:
            logger.info(f"\nBest configuration (scale={best_result['scale']}):")
            logger.info(f"  Improved {best_result['weak_improved']}/{len(weak_cats)} weak categories:")
            for cat in weak_cats:
                logger.info(f"    {cat}: {initial[cat]:.0%} → {best_result['final'][cat]:.0%} "
                           f"({best_result['changes'][cat]:+.0%})")
            logger.info(f"  All {len(strong_cats)} strong categories preserved")

            if best_result['weak_improved'] == len(weak_cats):
                logger.info("\n*** FULL SUCCESS: All weak improved, all strong preserved! ***")
            elif best_result['weak_improved'] > 0:
                logger.info(f"\n** Partial success: {best_result['weak_improved']}/{len(weak_cats)} improved **")

            results["best"] = best_result
            results["success_level"] = best_result['weak_improved']
        else:
            logger.info("\nNo configuration maintained all strong categories")
            results["success_level"] = 0

        self._reset_weights([layer_idx])
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = AllCategoryOptimization(model, tokenizer)
    results = test.run_experiment()

    output_path = "data/experiments/all_category_optimization.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
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
