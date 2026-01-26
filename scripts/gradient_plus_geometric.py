#!/usr/bin/env python3
"""Experiment 25: Gradient + Geometric Alignment.

Combine gradient guidance with geometric structure analysis.

Insight: Exp 18 used gradients to find safe directions.
What if we then apply geometric alignment ONLY in those safe directions?

Method:
1. Find orthogonal subspace (safe from preservation degradation)
2. Project weights into that subspace
3. Apply SVD geometric alignment to the projected weights
4. Transform back and measure
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


class GradientPlusGeometric:
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

        if np.linalg.norm(gradient) > 1e-10:
            return -gradient / np.linalg.norm(gradient)
        return gradient

    def find_nearest_constant_ratio(self, ratio: float) -> Tuple[str, float, float]:
        """Find which constant ratio is closest to the given ratio."""
        best_const = None
        best_error = float('inf')
        for name, val in CONSTANTS.items():
            error = abs(ratio - val) / val
            if error < best_error:
                best_error = error
                best_const = name
        return best_const, CONSTANTS[best_const], best_error

    def apply_geometric_alignment_in_safe_subspace(
        self,
        layer_idx: int,
        preserve_categories: List[str],
        alignment_strength: float = 0.5
    ) -> Dict:
        """
        Apply geometric alignment only in directions orthogonal to preserve gradients.

        Method:
        1. Compute preserve gradients
        2. Find orthogonal subspace
        3. In that subspace, adjust SVs toward constant ratios
        4. Leave preserved directions unchanged
        """
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)
        k = min(20, len(S))

        # Compute preservation gradients
        preserve_grads = []
        for cat in preserve_categories:
            grad = self.compute_loss_direction(layer_idx, cat)
            preserve_grads.append(grad)

        # Build safe subspace mask
        # Dimensions strongly aligned with preserve gradients are "unsafe"
        safe_mask = np.ones(k, dtype=bool)

        for grad in preserve_grads:
            for i in range(k):
                # If this SV dimension has strong gradient, it's not safe
                if abs(grad[i]) > 0.3:  # Threshold for "strong" gradient
                    safe_mask[i] = False

        n_safe = np.sum(safe_mask)
        logger.info(f"    Safe dimensions: {n_safe}/{k}")

        if n_safe < 2:
            return {"success": False, "reason": "too_few_safe_dimensions"}

        # Apply geometric alignment only in safe dimensions
        S_modified = S.copy()
        adjustments = []

        safe_indices = np.where(safe_mask)[0]
        for i in range(len(safe_indices) - 1):
            idx1 = safe_indices[i]
            idx2 = safe_indices[i + 1]

            if S[idx2] > 1e-10:
                current_ratio = S[idx1] / S[idx2]
                nearest_const, target_ratio, error = self.find_nearest_constant_ratio(current_ratio)

                if error < 0.15:  # Only adjust if close to a constant
                    # Adjust toward the constant ratio
                    target_s1 = S[idx2] * target_ratio
                    adjustment = (target_s1 - S[idx1]) * alignment_strength
                    S_modified[idx1] += adjustment
                    adjustments.append({
                        "idx1": int(idx1),
                        "idx2": int(idx2),
                        "original_ratio": float(current_ratio),
                        "target_const": nearest_const,
                        "target_ratio": float(target_ratio),
                        "adjustment": float(adjustment),
                    })

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            return {
                "success": True,
                "n_safe_dimensions": int(n_safe),
                "n_adjustments": len(adjustments),
                "adjustments": adjustments,
            }
        return {"success": False, "reason": "numerical_issues"}

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layer_idx = mid

        self._cache_weights([layer_idx])

        logger.info("=" * 60)
        logger.info("GRADIENT + GEOMETRIC ALIGNMENT")
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

        # Test configurations
        preserve_configs = [
            ["geography"],
            ["geography", "history"],
            ["geography", "history", "common_sense"],
        ]

        strength_configs = [0.3, 0.5, 0.7, 1.0]

        for preserve_cats in preserve_configs:
            for strength in strength_configs:
                self._reset_weights([layer_idx])
                key = f"preserve_{'_'.join(preserve_cats)}_strength_{strength}"
                logger.info(f"\n--- {key} ---")

                alignment_result = self.apply_geometric_alignment_in_safe_subspace(
                    layer_idx, preserve_cats, strength
                )

                if not alignment_result["success"]:
                    logger.info(f"  Failed: {alignment_result.get('reason', 'unknown')}")
                    continue

                logger.info(f"  Applied {alignment_result['n_adjustments']} adjustments "
                           f"in {alignment_result['n_safe_dimensions']} safe dimensions")

                final = self.evaluate_by_category()
                changes = {k: final[k] - initial[k] for k in initial}

                # Check preservation
                preserved = all(changes[cat] >= -0.01 for cat in preserve_cats)
                improved_any = any(changes[cat] > 0 for cat in final if cat not in preserve_cats and cat != "overall")

                logger.info(f"  Results:")
                for cat in final:
                    if cat != "overall":
                        status = "✓" if changes[cat] >= 0 else "✗"
                        logger.info(f"    {cat}: {final[cat]:.0%} ({changes[cat]:+.0%}) {status}")

                if preserved and improved_any:
                    logger.info("  *** SUCCESS: preserved and improved! ***")

                results["experiments"][key] = {
                    "alignment_result": {k: v for k, v in alignment_result.items() if k != "adjustments"},
                    "final": final,
                    "changes": changes,
                    "preserved": preserved,
                    "improved_any": improved_any,
                    "success": preserved and improved_any,
                }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        successes = [(k, v) for k, v in results["experiments"].items() if v.get("success", False)]
        if successes:
            logger.info(f"\n*** {len(successes)} SUCCESSES ***")
            for key, data in successes:
                logger.info(f"  {key}")
            results["overall_success"] = True
        else:
            logger.info("\nNo configurations achieved selective improvement")
            results["overall_success"] = False

        self._reset_weights([layer_idx])
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = GradientPlusGeometric(model, tokenizer)
    results = test.run_experiment()

    output_path = "data/experiments/gradient_plus_geometric.json"
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
