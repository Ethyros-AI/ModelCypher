#!/usr/bin/env python3
"""Experiment 18: Gradient-Guided Selective Modification.

Previous experiments showed:
- Weight modification: degradation before improvement
- Activation steering: same tradeoff when amplifying

Question: Can gradient information reveal "safe" modification directions?

Method:
1. Compute gradient of loss for different categories
2. Find weight perturbations that improve one category
   but are ORTHOGONAL to other categories' gradients
3. Apply only those orthogonal perturbations

Key insight: Gradients contain semantic separation information
that raw SVD indices lack.
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

ALL_QUESTIONS = []
for cat, qs in CATEGORY_QUESTIONS.items():
    for q, choices, idx in qs:
        ALL_QUESTIONS.append((q, choices, idx, cat))


class GradientGuidedModification:
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
        """Evaluate a question, return (correct, loss)."""
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

        # Compute cross-entropy loss for the correct answer
        scores_np = np.array(scores)
        probs = np.exp(scores_np - np.max(scores_np))
        probs = probs / probs.sum()
        loss = -np.log(probs[correct_idx] + 1e-10)

        return prediction == correct_idx, float(loss)

    def evaluate_by_category(self) -> Dict[str, float]:
        """Evaluate all categories."""
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
        """
        Compute the direction of loss change for a category.

        Uses finite differences to approximate gradient direction.
        Returns a direction in weight space that decreases loss.
        """
        W = self._original_weights[layer_idx]
        questions = CATEGORY_QUESTIONS[category]

        # Compute baseline loss
        self._set_weight(layer_idx, W)
        base_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)

        # Use SVD basis for perturbation directions
        U, S, Vt = svd(W, full_matrices=False)

        # Compute gradient in top-k SVD directions
        k = min(20, len(S))
        gradient = np.zeros_like(S[:k])

        for i in range(k):
            # Perturb in direction of i-th singular vector
            S_perturbed = S.copy()
            S_perturbed[i] += epsilon * S[i]  # Relative perturbation

            W_perturbed = U @ np.diag(S_perturbed) @ Vt
            if np.all(np.isfinite(W_perturbed)):
                self._set_weight(layer_idx, W_perturbed)
                perturbed_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)
                gradient[i] = (perturbed_loss - base_loss) / (epsilon * S[i])

        # Reset
        self._set_weight(layer_idx, W)

        # Return normalized direction that decreases loss
        if np.linalg.norm(gradient) > 1e-10:
            return -gradient / np.linalg.norm(gradient)
        return gradient

    def compute_orthogonal_perturbation(
        self,
        improve_direction: np.ndarray,
        preserve_directions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Find component of improve_direction orthogonal to all preserve_directions.

        This is the direction that improves one category without
        affecting others (in the gradient sense).
        """
        result = improve_direction.copy()

        for preserve_dir in preserve_directions:
            if np.linalg.norm(preserve_dir) > 1e-10:
                # Project out the preserve direction
                projection = np.dot(result, preserve_dir) * preserve_dir
                result = result - projection

        if np.linalg.norm(result) > 1e-10:
            return result / np.linalg.norm(result)
        return result

    def apply_orthogonal_modification(
        self,
        layer_idx: int,
        improve_category: str,
        preserve_categories: List[str],
        scale: float
    ) -> bool:
        """
        Apply modification that improves one category while orthogonal to others.
        """
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)
        k = min(20, len(S))

        # Compute improve direction
        logger.info(f"    Computing gradient for {improve_category}...")
        improve_dir = self.compute_loss_direction(layer_idx, improve_category)

        # Compute preserve directions
        preserve_dirs = []
        for cat in preserve_categories:
            logger.info(f"    Computing gradient for {cat}...")
            preserve_dir = self.compute_loss_direction(layer_idx, cat)
            preserve_dirs.append(preserve_dir)

        # Find orthogonal component
        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, preserve_dirs)

        if np.linalg.norm(ortho_dir) < 0.1:
            logger.info(f"    Orthogonal component too small - directions are entangled")
            return False

        # Apply perturbation in orthogonal direction
        S_modified = S.copy()
        for i in range(len(ortho_dir)):
            S_modified[i] += scale * ortho_dir[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            return True
        return False

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layers = [mid]  # Focus on middle layer for speed

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("GRADIENT-GUIDED SELECTIVE MODIFICATION")
        logger.info("=" * 60)
        logger.info(f"Testing layer: {layers[0]}")

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test: improve language, preserve geography
        test_configs = [
            ("language", ["geography"]),
            ("language", ["geography", "history"]),
            ("math", ["geography"]),
        ]

        for improve_cat, preserve_cats in test_configs:
            key = f"improve_{improve_cat}_preserve_{'_'.join(preserve_cats)}"
            logger.info(f"\n--- {key} ---")

            for scale in [0.1, 0.5, 1.0]:
                self._reset_weights(layers)

                logger.info(f"\n  Scale: {scale}")
                success = self.apply_orthogonal_modification(
                    layers[0], improve_cat, preserve_cats, scale
                )

                if not success:
                    logger.info(f"  Failed to find orthogonal direction")
                    continue

                final = self.evaluate_by_category()
                changes = {k: final[k] - initial[k] for k in initial}

                improved_target = changes[improve_cat] > 0.01
                preserved = all(changes[cat] >= -0.01 for cat in preserve_cats)

                logger.info(f"  Results: {improve_cat}={final[improve_cat]:.0%} "
                           f"({changes[improve_cat]:+.0%})")
                for cat in preserve_cats:
                    logger.info(f"    {cat}={final[cat]:.0%} ({changes[cat]:+.0%})")

                if improved_target and preserved:
                    logger.info(f"  *** SUCCESS: improved {improve_cat} without degrading {preserve_cats} ***")

                scale_key = f"{key}_scale{scale}"
                results[scale_key] = {
                    "final": final,
                    "changes": changes,
                    "improved_target": improved_target,
                    "preserved": preserved,
                    "success": improved_target and preserved,
                }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        successes = [(k, v) for k, v in results.items() if v.get("success", False)]
        if successes:
            logger.info(f"\n*** {len(successes)} SUCCESSES ***")
            for key, data in successes:
                logger.info(f"  {key}")
        else:
            logger.info("\nNo configurations achieved selective improvement")

        self._reset_weights(layers)

        return {
            "layer": layers[0],
            "initial": initial,
            "results": results,
            "success": len(successes) > 0,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = GradientGuidedModification(model, tokenizer)
    results = test.run_experiment()

    # Save
    output_path = "data/gradient_guided_modification.json"
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
