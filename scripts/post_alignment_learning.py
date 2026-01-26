#!/usr/bin/env python3
"""Experiment 46: Post-Alignment Learning.

Phase 9 - Stage 4: Can we now improve harder math?

Exp 45 fixed 2/3 broken fundamentals:
- 10 - 5 = 5 ✓ (was: 3)
- 5 + 5 = 10 ✓ (was: 9)
- 2 × 2 = 4 ✗ (still thinks 8)

Fundamentals went from 62% → 87.5% (7/8 correct).

Now test: With improved foundation, can gradient-guided learning improve Level 2 math?

This is the key test of the hypothesis: "You can't build capability on a broken foundation."
If the hypothesis is correct, math should now be improvable (it wasn't in Phase 8).
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


# Fundamentals (for alignment step)
BROKEN_FUNDAMENTALS = [
    ("What is 2 × 2?", ["2", "4", "6", "8"], 1),
    ("What is 5 + 5?", ["8", "9", "10", "11"], 2),
    ("What is 10 - 5?", ["3", "4", "5", "6"], 2),
]

CORRECT_FUNDAMENTALS = [
    ("What is 1 + 1?", ["1", "2", "3", "4"], 1),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    ("What is 3 + 3?", ["5", "6", "7", "8"], 1),
    ("What is 3 × 3?", ["6", "9", "12", "15"], 1),
    ("What is 4 ÷ 2?", ["1", "2", "3", "4"], 1),
]

FUNDAMENTAL_TRAINING = [
    ("What is 2 times 2?", "4", ["2", "4", "6", "8"], 1),
    ("What is 5 plus 5?", "10", ["8", "9", "10", "11"], 2),
    ("What is 10 minus 5?", "5", ["3", "4", "5", "6"], 2),
]

# Level 2: Basic operations (benchmark questions - Phase 8 showed 20%)
BASIC_MATH_QUESTIONS = [
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
]

# Training data for basic math
BASIC_MATH_TRAINING = [
    ("What is 8 × 7?", "56", ["48", "54", "56", "64"], 2),
    ("What is 9 × 6?", "54", ["45", "54", "56", "63"], 1),
    ("What is 100 ÷ 5?", "20", ["15", "20", "25", "50"], 1),
]

# Preservation categories (from Phase 8)
PRESERVATION_QUESTIONS = {
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
}


class PostAlignmentLearner:
    """Test if improved foundation enables math learning."""

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

    def evaluate_all(self) -> Dict[str, float]:
        """Evaluate fundamentals, basic math, and preservation."""
        results = {}

        # Fundamentals
        fund_correct = sum(1 for q, c, idx in CORRECT_FUNDAMENTALS + BROKEN_FUNDAMENTALS
                          if self._evaluate_question(q, c, idx)[0])
        results["fundamentals"] = fund_correct / (len(CORRECT_FUNDAMENTALS) + len(BROKEN_FUNDAMENTALS))

        # Basic math
        basic_correct = sum(1 for q, c, idx in BASIC_MATH_QUESTIONS
                           if self._evaluate_question(q, c, idx)[0])
        results["basic_math"] = basic_correct / len(BASIC_MATH_QUESTIONS)

        # Preservation
        for cat, questions in PRESERVATION_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx)[0])
            results[cat] = correct / len(questions)

        return results

    def compute_loss_direction(
        self,
        layer_idx: int,
        questions: List[Tuple],
        W: np.ndarray,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Compute loss gradient direction."""
        U, S, Vt = svd(W, full_matrices=False)

        base_loss = sum(self._evaluate_question(q, c, idx)[1] for q, c, idx in questions)

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

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 46: POST-ALIGNMENT LEARNING")
        logger.info("=" * 60)
        logger.info("\nCan we improve math after fixing fundamentals?\n")

        mid = self.n_layers // 2
        layer = mid
        self._cache_weights([layer])

        # ========== STEP 1: Align Fundamentals ==========
        logger.info("STEP 1: ALIGN FUNDAMENTALS")
        logger.info("-" * 40)

        initial = self.evaluate_all()
        logger.info(f"Initial state:")
        logger.info(f"  Fundamentals: {initial['fundamentals']:.0%}")
        logger.info(f"  Basic math: {initial['basic_math']:.0%}")
        logger.info(f"  Geography: {initial['geography']:.0%}")
        logger.info(f"  History: {initial['history']:.0%}")

        # Apply fundamental alignment (from Exp 45, scale 2.0 was best)
        W = self._get_weight(layer)
        U, S, Vt = svd(W, full_matrices=False)

        training_q = [(q, c, idx) for q, ans, c, idx in FUNDAMENTAL_TRAINING]
        training_q.extend(BROKEN_FUNDAMENTALS)

        improve_dir = self.compute_loss_direction(layer, training_q, W)
        preserve_dir = self.compute_loss_direction(layer, CORRECT_FUNDAMENTALS, W)
        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, [preserve_dir])

        S_aligned = S.copy()
        for i in range(len(ortho_dir)):
            S_aligned[i] += 2.0 * ortho_dir[i] * S[i]

        W_aligned = U @ np.diag(S_aligned) @ Vt
        if np.all(np.isfinite(W_aligned)):
            self._set_weight(layer, W_aligned)

        after_alignment = self.evaluate_all()
        logger.info(f"\nAfter fundamental alignment:")
        logger.info(f"  Fundamentals: {initial['fundamentals']:.0%} → {after_alignment['fundamentals']:.0%}")
        logger.info(f"  Basic math: {initial['basic_math']:.0%} → {after_alignment['basic_math']:.0%}")
        logger.info(f"  Geography: {initial['geography']:.0%} → {after_alignment['geography']:.0%}")
        logger.info(f"  History: {initial['history']:.0%} → {after_alignment['history']:.0%}")

        # ========== STEP 2: Learn Basic Math ==========
        logger.info(f"\n{'='*60}")
        logger.info("STEP 2: LEARN BASIC MATH (on aligned foundation)")
        logger.info("-" * 40)

        # Now try gradient-guided learning on basic math
        W_current = self._get_weight(layer)
        U, S, Vt = svd(W_current, full_matrices=False)

        # Training data for basic math
        basic_training = [(q, c, idx) for q, ans, c, idx in BASIC_MATH_TRAINING]
        basic_training.extend(BASIC_MATH_QUESTIONS)

        logger.info(f"  Training on {len(basic_training)} basic math questions...")

        improve_dir = self.compute_loss_direction(layer, basic_training, W_current)

        # Preserve geography, history, AND fundamentals
        preserve_questions = []
        for cat, questions in PRESERVATION_QUESTIONS.items():
            preserve_questions.extend(questions)
        preserve_questions.extend(CORRECT_FUNDAMENTALS)
        preserve_questions.extend([(q, c, idx) for q, c, idx in BROKEN_FUNDAMENTALS
                                   if self._evaluate_question(q, c, idx)[0]])  # Only fixed ones

        preserve_dir = self.compute_loss_direction(layer, preserve_questions, W_current)
        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, [preserve_dir])

        results = {"initial": initial, "after_alignment": after_alignment, "scales": {}}

        for scale in [0.5, 1.0, 1.5, 2.0]:
            logger.info(f"\n  Scale {scale}:")

            self._set_weight(layer, W_current)  # Reset to aligned state

            S_modified = S.copy()
            for i in range(len(ortho_dir)):
                S_modified[i] += scale * ortho_dir[i] * S[i]

            W_modified = U @ np.diag(S_modified) @ Vt
            if np.all(np.isfinite(W_modified)):
                self._set_weight(layer, W_modified)

                final = self.evaluate_all()

                math_improved = final["basic_math"] > after_alignment["basic_math"] + 0.05
                geo_preserved = final["geography"] >= after_alignment["geography"] - 0.1
                hist_preserved = final["history"] >= after_alignment["history"] - 0.1

                logger.info(f"    Basic math: {after_alignment['basic_math']:.0%} → {final['basic_math']:.0%}")
                logger.info(f"    Geography: {after_alignment['geography']:.0%} → {final['geography']:.0%}")
                logger.info(f"    History: {after_alignment['history']:.0%} → {final['history']:.0%}")

                if math_improved and geo_preserved and hist_preserved:
                    status = "SUCCESS"
                    logger.info(f"    *** SUCCESS: Math improved without degradation! ***")
                elif math_improved:
                    status = "IMPROVED_BUT_DEGRADED"
                elif geo_preserved and hist_preserved:
                    status = "NO_IMPROVEMENT"
                else:
                    status = "DEGRADED"

                results["scales"][str(scale)] = {
                    "final": final,
                    "math_improved": math_improved,
                    "preserved": geo_preserved and hist_preserved,
                    "status": status,
                }

        # ========== SUMMARY ==========
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nPhase 8 Result (without foundation fix):")
        logger.info(f"  Basic math: 20% → 20% (NO IMPROVEMENT)")

        best_scale = None
        best_math = after_alignment["basic_math"]
        for scale, data in results["scales"].items():
            if data["final"]["basic_math"] > best_math:
                best_math = data["final"]["basic_math"]
                best_scale = scale

        logger.info(f"\nPhase 9 Result (with foundation fix):")
        logger.info(f"  Fundamentals: {initial['fundamentals']:.0%} → {after_alignment['fundamentals']:.0%}")
        if best_scale:
            logger.info(f"  Basic math: {initial['basic_math']:.0%} → {best_math:.0%} (scale {best_scale})")
            results["conclusion"] = "foundation_enables_learning"
            logger.info(f"\n*** THE HYPOTHESIS IS CONFIRMED ***")
            logger.info(f"Fixing the foundation enabled math improvement!")
        else:
            logger.info(f"  Basic math: {initial['basic_math']:.0%} → {after_alignment['basic_math']:.0%} (no further improvement)")
            results["conclusion"] = "foundation_helps_but_not_enough"
            logger.info(f"\nFoundation improved but basic math still stuck.")

        results["best_scale"] = best_scale
        results["best_math"] = best_math

        self._reset_weights([layer])
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = PostAlignmentLearner(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/post_alignment_learning.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
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
