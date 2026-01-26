#!/usr/bin/env python3
"""Experiment 26: Iterative Orthogonal Refinement.

Test if multiple iterations of orthogonal gradient descent compound improvements.

Method:
1. Apply orthogonal gradient modification
2. Re-compute gradients on modified model
3. Apply again
4. Track improvement trajectory

Key question: Does improvement plateau or continue?
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


class IterativeOrthogonal:
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

    def compute_loss_direction(
        self,
        layer_idx: int,
        category: str,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Compute gradient direction on CURRENT weights (not cached original)."""
        W = self._get_weight(layer_idx)  # Get current weights
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

        self._set_weight(layer_idx, W)  # Restore current

        if np.linalg.norm(gradient) > 1e-10:
            return -gradient / np.linalg.norm(gradient)
        return gradient

    def compute_orthogonal_perturbation(
        self,
        improve_direction: np.ndarray,
        preserve_directions: List[np.ndarray]
    ) -> np.ndarray:
        result = improve_direction.copy()
        for preserve_dir in preserve_directions:
            if np.linalg.norm(preserve_dir) > 1e-10:
                projection = np.dot(result, preserve_dir) * preserve_dir
                result = result - projection
        if np.linalg.norm(result) > 1e-10:
            return result / np.linalg.norm(result)
        return result

    def apply_single_iteration(
        self,
        layer_idx: int,
        improve_category: str,
        preserve_categories: List[str],
        scale: float
    ) -> Tuple[bool, float]:
        """Apply one iteration of orthogonal gradient descent."""
        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        improve_dir = self.compute_loss_direction(layer_idx, improve_category)

        preserve_dirs = []
        for cat in preserve_categories:
            preserve_dir = self.compute_loss_direction(layer_idx, cat)
            preserve_dirs.append(preserve_dir)

        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, preserve_dirs)
        ortho_norm = np.linalg.norm(ortho_dir)

        if ortho_norm < 0.1:
            return False, ortho_norm

        S_modified = S.copy()
        for i in range(len(ortho_dir)):
            S_modified[i] += scale * ortho_dir[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            return True, ortho_norm
        return False, ortho_norm

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layer_idx = mid

        self._cache_weights([layer_idx])

        logger.info("=" * 60)
        logger.info("ITERATIVE ORTHOGONAL REFINEMENT")
        logger.info("=" * 60)
        logger.info(f"Testing layer: {layer_idx}")

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial accuracies:")
        for cat, acc in sorted(initial.items(), key=lambda x: x[1]):
            if cat != "overall":
                logger.info(f"  {cat}: {acc:.0%}")

        results = {
            "layer": layer_idx,
            "initial": initial,
            "experiments": {},
        }

        # Test configuration
        improve_cat = "language"
        preserve_cats = ["geography", "history"]
        scale = 0.5  # Smaller scale per iteration
        max_iterations = 5

        logger.info(f"\nConfiguration:")
        logger.info(f"  Improve: {improve_cat}")
        logger.info(f"  Preserve: {preserve_cats}")
        logger.info(f"  Scale per iteration: {scale}")
        logger.info(f"  Max iterations: {max_iterations}")

        # Track trajectory
        trajectory = [initial.copy()]
        ortho_norms = []

        self._reset_weights([layer_idx])

        for iteration in range(max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")

            success, ortho_norm = self.apply_single_iteration(
                layer_idx, improve_cat, preserve_cats, scale
            )
            ortho_norms.append(float(ortho_norm))

            if not success:
                logger.info(f"  Failed: orthogonal component too small ({ortho_norm:.3f})")
                break

            current = self.evaluate_by_category()
            trajectory.append(current.copy())

            changes_from_initial = {k: current[k] - initial[k] for k in initial}
            changes_from_prev = {k: current[k] - trajectory[-2][k] for k in initial}

            logger.info(f"  Ortho norm: {ortho_norm:.3f}")
            logger.info(f"  {improve_cat}: {current[improve_cat]:.0%} "
                       f"(Δ={changes_from_prev[improve_cat]:+.0%}, "
                       f"total={changes_from_initial[improve_cat]:+.0%})")
            for cat in preserve_cats:
                logger.info(f"  {cat}: {current[cat]:.0%} "
                           f"(Δ={changes_from_prev[cat]:+.0%}, "
                           f"total={changes_from_initial[cat]:+.0%})")

            # Check if improvement stalled
            if changes_from_prev[improve_cat] == 0 and iteration > 0:
                logger.info("  Improvement stalled")
                break

            # Check if preservation failed
            if any(changes_from_initial[cat] < -0.01 for cat in preserve_cats):
                logger.info("  Preservation failed!")
                break

        # Final analysis
        logger.info("\n" + "=" * 60)
        logger.info("TRAJECTORY ANALYSIS")
        logger.info("=" * 60)

        final = trajectory[-1]
        total_changes = {k: final[k] - initial[k] for k in initial}

        logger.info(f"\nTotal iterations completed: {len(trajectory) - 1}")
        logger.info(f"\nFinal changes from initial:")
        logger.info(f"  {improve_cat}: {initial[improve_cat]:.0%} → {final[improve_cat]:.0%} "
                   f"({total_changes[improve_cat]:+.0%})")
        for cat in preserve_cats:
            logger.info(f"  {cat}: {initial[cat]:.0%} → {final[cat]:.0%} "
                       f"({total_changes[cat]:+.0%})")

        # Check success
        improved = total_changes[improve_cat] > 0
        preserved = all(total_changes[cat] >= -0.01 for cat in preserve_cats)

        if improved and preserved:
            logger.info("\n*** SUCCESS: Improved target while preserving others! ***")
        elif preserved:
            logger.info("\nPreservation maintained, but no improvement")
        else:
            logger.info("\nFailed: preservation degraded")

        results["trajectory"] = trajectory
        results["ortho_norms"] = ortho_norms
        results["final_changes"] = total_changes
        results["improved"] = improved
        results["preserved"] = preserved
        results["success"] = improved and preserved
        results["n_iterations"] = len(trajectory) - 1

        # Also test single large step for comparison
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: Single large step")
        logger.info("=" * 60)

        self._reset_weights([layer_idx])
        success, _ = self.apply_single_iteration(layer_idx, improve_cat, preserve_cats, scale=1.0)

        if success:
            single_step = self.evaluate_by_category()
            single_changes = {k: single_step[k] - initial[k] for k in initial}
            logger.info(f"\n  Single step (scale=1.0):")
            logger.info(f"    {improve_cat}: {single_step[improve_cat]:.0%} ({single_changes[improve_cat]:+.0%})")
            for cat in preserve_cats:
                logger.info(f"    {cat}: {single_step[cat]:.0%} ({single_changes[cat]:+.0%})")

            results["single_step_comparison"] = {
                "final": single_step,
                "changes": single_changes,
            }

        self._reset_weights([layer_idx])
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = IterativeOrthogonal(model, tokenizer)
    results = test.run_experiment()

    output_path = "data/experiments/iterative_orthogonal.json"
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
