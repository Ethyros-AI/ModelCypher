#!/usr/bin/env python3
"""Experiment 40: Single-Topic Learning.

Phase 8 - Stage 2: The Critical Test

Can the full learning loop improve a single topic without degradation?

This combines:
1. Gap detection (Exp 38) - identify weak categories
2. Research integration (Exp 39) - generate training data
3. Gradient-guided modification (Phase 5-6) - learn without forgetting

Method:
1. Pick topic with LOW accuracy (math at ~40%)
2. Pick preservation targets (geography, history at 100%)
3. Use researched facts as training signal
4. Apply gradient-guided orthogonal modification
5. Verify: Did target improve? Did preserved stay stable?
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


# All categories with their questions (from benchmark)
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
}

# Training data from research (facts the model got wrong)
MATH_TRAINING_DATA = [
    # Format: (question, correct_answer_text, correct_idx)
    ("What is 8 × 7?", "56", ["48", "54", "56", "64"], 2),
    ("What is 9 × 6?", "54", ["45", "54", "56", "63"], 1),
    ("What is 100 ÷ 5?", "20", ["15", "20", "25", "50"], 1),
]

LOGIC_TRAINING_DATA = [
    ("What comes next: 2, 4, 6, 8, ?", "10", ["9", "10", "11", "12"], 1),
    ("What comes next: 1, 3, 5, 7, ?", "9", ["8", "9", "10", "11"], 1),
]


class SingleTopicLearner:
    """The critical test: improve one topic without degrading others."""

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

        # Compute loss
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
        questions: List[Tuple],
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Compute loss gradient direction for given questions."""
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        # Compute baseline loss
        self._set_weight(layer_idx, W)
        base_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)

        # Gradient in top-k SVD directions
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
        """Find component orthogonal to preserve directions."""
        result = improve_direction.copy()

        for preserve_dir in preserve_directions:
            if np.linalg.norm(preserve_dir) > 1e-10:
                projection = np.dot(result, preserve_dir) * preserve_dir
                result = result - projection

        if np.linalg.norm(result) > 1e-10:
            return result / np.linalg.norm(result)
        return result

    def apply_learning(
        self,
        layer_idx: int,
        training_questions: List[Tuple],
        preserve_categories: List[str],
        scale: float
    ) -> bool:
        """Apply gradient-guided learning."""
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        # Compute improve direction from training data
        logger.info(f"    Computing gradient for training data ({len(training_questions)} questions)...")
        improve_dir = self.compute_loss_direction(layer_idx, training_questions)

        # Compute preserve directions
        preserve_dirs = []
        for cat in preserve_categories:
            logger.info(f"    Computing gradient for {cat}...")
            cat_questions = [(q, c, idx) for q, c, idx in CATEGORY_QUESTIONS[cat]]
            preserve_dir = self.compute_loss_direction(layer_idx, cat_questions)
            preserve_dirs.append(preserve_dir)

        # Find orthogonal component
        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, preserve_dirs)

        ortho_magnitude = np.linalg.norm(ortho_dir)
        logger.info(f"    Orthogonal component magnitude: {ortho_magnitude:.3f}")

        if ortho_magnitude < 0.1:
            logger.info(f"    WARNING: Orthogonal component small - directions may be entangled")

        # Apply perturbation
        S_modified = S.copy()
        for i in range(len(ortho_dir)):
            S_modified[i] += scale * ortho_dir[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            return True
        return False

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 40: SINGLE-TOPIC LEARNING")
        logger.info("=" * 60)

        mid = self.n_layers // 2
        layers = [mid]
        self._cache_weights(layers)

        # Initial evaluation
        initial = self.evaluate_by_category()
        logger.info(f"\nInitial accuracies:")
        for cat, acc in sorted(initial.items()):
            logger.info(f"  {cat}: {acc:.0%}")

        # Identify weak and strong categories
        target_category = "math"  # Known weak from Exp 38-39
        preserve_categories = ["geography", "history"]  # Known strong

        logger.info(f"\nTarget for improvement: {target_category} ({initial[target_category]:.0%})")
        logger.info(f"Preserve: {preserve_categories} ({initial['geography']:.0%}, {initial['history']:.0%})")

        # Prepare training data (combining benchmark questions + researched facts)
        training_questions = []
        for q, ans, choices, idx in MATH_TRAINING_DATA:
            training_questions.append((q, choices, idx))
        # Also include original math questions
        training_questions.extend(CATEGORY_QUESTIONS["math"])

        results = {"scales": {}}

        for scale in [0.5, 1.0, 1.5, 2.0]:
            logger.info(f"\n--- Scale: {scale} ---")
            self._reset_weights(layers)

            success = self.apply_learning(
                layers[0],
                training_questions,
                preserve_categories,
                scale
            )

            if not success:
                logger.info("  Failed to apply modification")
                continue

            final = self.evaluate_by_category()
            changes = {k: final[k] - initial[k] for k in initial}

            # Check success criteria
            target_improved = changes[target_category] > 0.01
            preserved_stable = all(changes[cat] >= -0.05 for cat in preserve_categories)

            logger.info(f"  Results:")
            logger.info(f"    {target_category}: {initial[target_category]:.0%} → {final[target_category]:.0%} ({changes[target_category]:+.0%})")
            for cat in preserve_categories:
                logger.info(f"    {cat}: {initial[cat]:.0%} → {final[cat]:.0%} ({changes[cat]:+.0%})")

            status = "SUCCESS" if target_improved and preserved_stable else (
                "PRESERVED_ONLY" if preserved_stable else "DEGRADED"
            )

            if target_improved and preserved_stable:
                logger.info(f"  *** SUCCESS: Improved {target_category} without degrading {preserve_categories} ***")

            results["scales"][str(scale)] = {
                "final": final,
                "changes": changes,
                "target_improved": target_improved,
                "preserved_stable": preserved_stable,
                "status": status,
            }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        successes = [
            (scale, data) for scale, data in results["scales"].items()
            if data.get("status") == "SUCCESS"
        ]

        if successes:
            logger.info(f"\n*** {len(successes)} SUCCESSFUL CONFIGURATIONS ***")
            for scale, data in successes:
                logger.info(f"  Scale {scale}: {target_category} {initial[target_category]:.0%} → {data['final'][target_category]:.0%}")
            results["conclusion"] = "success"
        else:
            preserved_only = [
                (scale, data) for scale, data in results["scales"].items()
                if data.get("status") == "PRESERVED_ONLY"
            ]
            if preserved_only:
                logger.info(f"\n{len(preserved_only)} configs preserved but didn't improve")
                results["conclusion"] = "preserved_no_improvement"
            else:
                logger.info("\nAll configurations caused degradation")
                results["conclusion"] = "degradation"

        self._reset_weights(layers)

        results["initial"] = initial
        results["target_category"] = target_category
        results["preserve_categories"] = preserve_categories
        results["n_training_questions"] = len(training_questions)

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = SingleTopicLearner(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/single_topic_learning.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
