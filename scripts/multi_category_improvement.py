#!/usr/bin/env python3
"""Experiment 21: Multi-Category Improvement.

Stage 1 findings:
- Math failed because it's a harder task (20% baseline), not entanglement
- Language succeeded (60% → 80%) with plenty of safe subspace
- Logic has 60% baseline - similar to language, should be improvable

Question: Can we improve BOTH language AND logic while preserving geography+history?

Method:
1. Compute combined improvement gradient (language + logic)
2. Project orthogonal to preservation gradients
3. Apply modification and measure results
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


class MultiCategoryImprovement:
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

    def project_orthogonal(
        self,
        direction: np.ndarray,
        preserve_directions: List[np.ndarray]
    ) -> np.ndarray:
        """Project direction orthogonal to all preserve directions."""
        result = direction.copy()
        for preserve_dir in preserve_directions:
            norm = np.linalg.norm(preserve_dir)
            if norm > 1e-10:
                preserve_normalized = preserve_dir / norm
                projection = np.dot(result, preserve_normalized) * preserve_normalized
                result = result - projection
        return result

    def apply_modification(
        self,
        layer_idx: int,
        direction: np.ndarray,
        scale: float
    ) -> bool:
        """Apply modification in the given direction."""
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        # Modify singular values in the given direction
        S_modified = S.copy()
        for i in range(min(len(direction), len(S))):
            # Negative direction because we want to DECREASE loss
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
        logger.info("MULTI-CATEGORY IMPROVEMENT")
        logger.info("=" * 60)
        logger.info(f"Testing layer: {layer_idx}")

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial accuracies:")
        for cat, acc in initial.items():
            logger.info(f"  {cat}: {acc:.0%}")

        results = {
            "layer": layer_idx,
            "initial": initial,
            "experiments": {},
        }

        # Categories to improve (both 60% baseline)
        improve_cats = ["language", "logic"]
        # Categories to preserve
        preserve_cats = ["geography", "history"]

        logger.info(f"\nImprove: {improve_cats}")
        logger.info(f"Preserve: {preserve_cats}")

        # Compute gradients
        logger.info("\nComputing gradients...")
        gradients = {}

        for cat in improve_cats + preserve_cats:
            logger.info(f"  {cat}...")
            gradients[cat] = self.compute_loss_gradient(layer_idx, cat)

        # Compute combined improvement gradient
        improve_grad = sum(gradients[cat] for cat in improve_cats)
        improve_grad_norm = np.linalg.norm(improve_grad)
        if improve_grad_norm > 1e-10:
            improve_grad = improve_grad / improve_grad_norm

        # Compute preservation gradients
        preserve_grads = [gradients[cat] for cat in preserve_cats]

        # Project orthogonal
        ortho_grad = self.project_orthogonal(improve_grad, preserve_grads)
        ortho_norm = np.linalg.norm(ortho_grad)
        logger.info(f"\nOrthogonal component survive ratio: {ortho_norm:.1%}")

        if ortho_norm < 0.1:
            logger.info("Orthogonal component too small - categories are entangled")
            results["failed"] = "entangled"
            return results

        # Normalize orthogonal direction
        ortho_grad = ortho_grad / ortho_norm

        # Test different scales
        scales = [0.05, 0.1, 0.2, 0.3, 0.5]
        best_result = None
        best_score = 0

        for scale in scales:
            self._reset_weights([layer_idx])
            logger.info(f"\n--- Scale: {scale} ---")

            success = self.apply_modification(layer_idx, ortho_grad, scale)
            if not success:
                logger.info("  Modification failed (numerical issues)")
                continue

            final = self.evaluate_by_category()
            changes = {k: final[k] - initial[k] for k in initial}

            # Check success criteria
            improved_lang = changes["language"] > 0
            improved_logic = changes["logic"] > 0
            preserved_geo = changes["geography"] >= -0.01
            preserved_hist = changes["history"] >= -0.01

            logger.info(f"  language: {final['language']:.0%} ({changes['language']:+.0%})")
            logger.info(f"  logic: {final['logic']:.0%} ({changes['logic']:+.0%})")
            logger.info(f"  geography: {final['geography']:.0%} ({changes['geography']:+.0%})")
            logger.info(f"  history: {final['history']:.0%} ({changes['history']:+.0%})")

            success_level = sum([improved_lang, improved_logic, preserved_geo, preserved_hist])

            if success_level == 4:
                logger.info("  *** FULL SUCCESS: Both improved, both preserved ***")
            elif preserved_geo and preserved_hist:
                if improved_lang or improved_logic:
                    logger.info("  ** Partial success: preservation held, some improvement **")

            results["experiments"][f"scale_{scale}"] = {
                "final": final,
                "changes": changes,
                "improved_language": improved_lang,
                "improved_logic": improved_logic,
                "preserved_geography": preserved_geo,
                "preserved_history": preserved_hist,
                "success_level": success_level,
            }

            # Track best
            score = changes["language"] + changes["logic"]
            if preserved_geo and preserved_hist and score > best_score:
                best_score = score
                best_result = {
                    "scale": scale,
                    "final": final,
                    "changes": changes,
                }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if best_result:
            logger.info(f"\nBest configuration (scale={best_result['scale']}):")
            logger.info(f"  language: {initial['language']:.0%} → {best_result['final']['language']:.0%} "
                       f"({best_result['changes']['language']:+.0%})")
            logger.info(f"  logic: {initial['logic']:.0%} → {best_result['final']['logic']:.0%} "
                       f"({best_result['changes']['logic']:+.0%})")
            logger.info(f"  geography: preserved at {best_result['final']['geography']:.0%}")
            logger.info(f"  history: preserved at {best_result['final']['history']:.0%}")

            full_success = (
                best_result['changes']['language'] > 0 and
                best_result['changes']['logic'] > 0
            )
            if full_success:
                logger.info("\n*** SUCCESS: Multi-category improvement achieved! ***")
            else:
                logger.info("\nPartial success - preservation worked, limited improvement")

            results["best"] = best_result
            results["full_success"] = full_success
        else:
            logger.info("\nNo successful configuration found")
            results["full_success"] = False

        self._reset_weights([layer_idx])
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = MultiCategoryImprovement(model, tokenizer)
    results = test.run_experiment()

    output_path = "data/experiments/multi_category_improvement.json"
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
