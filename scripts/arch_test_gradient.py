#!/usr/bin/env python3
"""Experiment 24: Architecture Test - Qwen.

Test if gradient-guided modification works across different architectures.

Key question: Is this a universal property or LFM2-specific?
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


class ArchTestGradient:
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._detect_architecture()
        self._original_weights = {}

    def _detect_architecture(self):
        """Detect model architecture and set appropriate accessors."""
        # Try different architecture patterns
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            self.layers = self.model.layers
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")

        self.n_layers = len(self.layers)
        logger.info(f"Detected {self.n_layers} layers")

        # Detect MLP accessor
        test_layer = self.layers[0]
        if hasattr(test_layer, 'feed_forward'):
            self.mlp_accessor = 'feed_forward'
        elif hasattr(test_layer, 'mlp'):
            self.mlp_accessor = 'mlp'
        else:
            raise ValueError(f"Unknown MLP accessor for layer: {type(test_layer)}")

        # Detect weight accessor
        mlp = getattr(test_layer, self.mlp_accessor)
        if hasattr(mlp, 'gate_proj'):
            self.weight_accessor = 'gate_proj'
        elif hasattr(mlp, 'w1'):
            self.weight_accessor = 'w1'
        elif hasattr(mlp, 'fc1'):
            self.weight_accessor = 'fc1'
        elif hasattr(mlp, 'c_fc'):
            self.weight_accessor = 'c_fc'
        else:
            raise ValueError(f"Unknown weight accessor for MLP: {type(mlp)}")

        logger.info(f"Using MLP accessor: {self.mlp_accessor}, weight accessor: {self.weight_accessor}")

    def _get_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.layers[layer_idx]
        mlp = getattr(layer, self.mlp_accessor)
        w = getattr(mlp, self.weight_accessor).weight
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.layers[layer_idx]
        mlp = getattr(layer, self.mlp_accessor)
        proj = getattr(mlp, self.weight_accessor)
        new_weight = mx.array(weights.astype(np.float32))
        proj.weight = new_weight
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

    def apply_orthogonal_modification(
        self,
        layer_idx: int,
        improve_category: str,
        preserve_categories: List[str],
        scale: float
    ) -> bool:
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        logger.info(f"    Computing gradient for {improve_category}...")
        improve_dir = self.compute_loss_direction(layer_idx, improve_category)

        preserve_dirs = []
        for cat in preserve_categories:
            logger.info(f"    Computing gradient for {cat}...")
            preserve_dir = self.compute_loss_direction(layer_idx, cat)
            preserve_dirs.append(preserve_dir)

        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, preserve_dirs)

        if np.linalg.norm(ortho_dir) < 0.1:
            logger.info(f"    Orthogonal component too small - directions are entangled")
            return False

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
        layer_idx = mid

        self._cache_weights([layer_idx])

        logger.info("=" * 60)
        logger.info(f"ARCHITECTURE TEST - {self.model_name}")
        logger.info("=" * 60)
        logger.info(f"Model layers: {self.n_layers}")
        logger.info(f"Testing layer: {layer_idx}")

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial accuracies:")
        for cat, acc in sorted(initial.items(), key=lambda x: x[1]):
            if cat != "overall":
                logger.info(f"  {cat}: {acc:.0%}")
        logger.info(f"  overall: {initial['overall']:.0%}")

        results = {
            "model": self.model_name,
            "n_layers": self.n_layers,
            "layer": layer_idx,
            "initial": initial,
            "experiments": {},
        }

        # Find a weak category to improve
        # Choose the weakest category that's not already at 0%
        weak_cats = [(cat, acc) for cat, acc in initial.items()
                     if cat != "overall" and 0 < acc < 1.0]
        weak_cats.sort(key=lambda x: x[1])

        if not weak_cats:
            logger.info("No weak categories to improve!")
            results["error"] = "no_weak_categories"
            return results

        # Find strong categories to preserve
        strong_cats = [cat for cat, acc in initial.items()
                       if cat != "overall" and acc >= 0.8]

        if not strong_cats:
            strong_cats = ["geography", "history"]  # Default

        logger.info(f"\nWeak categories: {[c for c, _ in weak_cats[:3]]}")
        logger.info(f"Strong categories (to preserve): {strong_cats[:2]}")

        # Test improvement configurations
        test_configs = []
        for cat, _ in weak_cats[:2]:
            test_configs.append((cat, strong_cats[:2]))

        for improve_cat, preserve_cats in test_configs:
            key = f"improve_{improve_cat}_preserve_{'_'.join(preserve_cats)}"
            logger.info(f"\n--- {key} ---")

            for scale in [0.5, 1.0, 1.5]:
                self._reset_weights([layer_idx])
                logger.info(f"\n  Scale: {scale}")

                success = self.apply_orthogonal_modification(
                    layer_idx, improve_cat, preserve_cats, scale
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
                    logger.info(f"  *** SUCCESS ***")

                scale_key = f"{key}_scale{scale}"
                results["experiments"][scale_key] = {
                    "final": final,
                    "changes": changes,
                    "improved_target": improved_target,
                    "preserved": preserved,
                    "success": improved_target and preserved,
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
            results["success"] = True
        else:
            logger.info("\nNo configurations achieved selective improvement")
            results["success"] = False

        self._reset_weights([layer_idx])
        return results


def main():
    from mlx_lm import load

    # Try to find a Qwen model
    qwen_paths = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Math-1.5B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16",
    ]

    model = None
    model_path = None

    for path in qwen_paths:
        if Path(path).exists():
            model_path = path
            break

    if model_path:
        logger.info(f"Loading model: {model_path}")
        try:
            model, tokenizer = load(model_path)
            model_name = Path(model_path).name
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")
            model = None

    if model is None:
        # Fallback to LFM2 for demonstration
        logger.info("No Qwen model found, using LFM2-350M for demonstration")
        model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
        model, tokenizer = load(model_path)
        model_name = "LFM2-350M (fallback)"

    test = ArchTestGradient(model, tokenizer, model_name)
    results = test.run_experiment()

    output_path = "data/experiments/arch_test_gradient.json"
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
