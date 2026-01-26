#!/usr/bin/env python3
"""Experiment 45: Fundamental Alignment.

Phase 9 - Stage 3: Fix the broken fundamentals.

Exp 43 found these broken fundamentals:
- 2×2 = 8 (should be 4)
- 5+5 = 9 (should be 10)
- 10-5 = 3 (should be 5)

This experiment uses gradient-guided modification to:
1. Improve: The broken fundamentals (2×2, 5+5, 10-5)
2. Preserve: The correct fundamentals (1+1, 2+2, 3+3, 3×3, 4÷2)

If this works, we've fixed the foundation and can then build capability on top.
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


# BROKEN fundamentals (need to be fixed)
BROKEN_FUNDAMENTALS = [
    ("What is 2 × 2?", ["2", "4", "6", "8"], 1),  # Model says 8, should be 4 (index 1)
    ("What is 5 + 5?", ["8", "9", "10", "11"], 2),  # Model says 9, should be 10 (index 2)
    ("What is 10 - 5?", ["3", "4", "5", "6"], 2),  # Model says 3, should be 5 (index 2)
]

# CORRECT fundamentals (need to be preserved)
CORRECT_FUNDAMENTALS = [
    ("What is 1 + 1?", ["1", "2", "3", "4"], 1),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    ("What is 3 + 3?", ["5", "6", "7", "8"], 1),
    ("What is 3 × 3?", ["6", "9", "12", "15"], 1),
    ("What is 4 ÷ 2?", ["1", "2", "3", "4"], 1),
]

# All fundamentals for verification
ALL_FUNDAMENTALS = BROKEN_FUNDAMENTALS + CORRECT_FUNDAMENTALS

# Training data to reinforce correct answers
TRAINING_DATA = [
    # Reinforce correct 2×2 = 4
    ("What is 2 times 2?", "4", ["2", "4", "6", "8"], 1),
    ("What is 2 multiplied by 2?", "4", ["2", "4", "6", "8"], 1),
    # Reinforce correct 5+5 = 10
    ("What is 5 plus 5?", "10", ["8", "9", "10", "11"], 2),
    ("What is five plus five?", "10", ["8", "9", "10", "11"], 2),
    # Reinforce correct 10-5 = 5
    ("What is 10 minus 5?", "5", ["3", "4", "5", "6"], 2),
    ("What is ten minus five?", "5", ["3", "4", "5", "6"], 2),
]


class FundamentalAligner:
    """Fix the broken fundamentals through gradient-guided alignment."""

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

    def evaluate_fundamentals(self) -> Dict[str, float]:
        """Evaluate all fundamentals."""
        broken_correct = sum(1 for q, c, idx in BROKEN_FUNDAMENTALS if self._evaluate_question(q, c, idx)[0])
        correct_correct = sum(1 for q, c, idx in CORRECT_FUNDAMENTALS if self._evaluate_question(q, c, idx)[0])
        total_correct = broken_correct + correct_correct

        return {
            "broken_accuracy": broken_correct / len(BROKEN_FUNDAMENTALS),
            "preserved_accuracy": correct_correct / len(CORRECT_FUNDAMENTALS),
            "total_accuracy": total_correct / len(ALL_FUNDAMENTALS),
            "broken_correct": broken_correct,
            "preserved_correct": correct_correct,
        }

    def compute_loss_direction(
        self,
        layer_idx: int,
        questions: List[Tuple],
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Compute loss gradient direction for given questions."""
        W = self._original_weights[layer_idx]
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

    def apply_alignment(
        self,
        layer_idx: int,
        scale: float
    ) -> bool:
        """Apply gradient-guided alignment to fix broken fundamentals."""
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        # Compute gradient to IMPROVE broken fundamentals
        logger.info(f"    Computing gradient to fix broken fundamentals...")
        training_q = [(q, c, idx) for q, ans, c, idx in TRAINING_DATA]
        training_q.extend(BROKEN_FUNDAMENTALS)
        improve_dir = self.compute_loss_direction(layer_idx, training_q)

        # Compute gradient to PRESERVE correct fundamentals
        logger.info(f"    Computing gradient to preserve correct fundamentals...")
        preserve_dir = self.compute_loss_direction(layer_idx, CORRECT_FUNDAMENTALS)

        # Find orthogonal component
        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, [preserve_dir])

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
        logger.info("EXPERIMENT 45: FUNDAMENTAL ALIGNMENT")
        logger.info("=" * 60)
        logger.info("\nFixing the broken fundamentals: 2×2, 5+5, 10-5\n")

        mid = self.n_layers // 2
        layers = [mid]
        self._cache_weights(layers)

        # Initial evaluation
        initial = self.evaluate_fundamentals()
        logger.info("Initial status:")
        logger.info(f"  Broken fundamentals: {initial['broken_accuracy']:.0%} ({initial['broken_correct']}/{len(BROKEN_FUNDAMENTALS)})")
        logger.info(f"  Preserved fundamentals: {initial['preserved_accuracy']:.0%} ({initial['preserved_correct']}/{len(CORRECT_FUNDAMENTALS)})")
        logger.info(f"  Total: {initial['total_accuracy']:.0%}")

        # Log which are wrong
        logger.info("\nBroken fundamentals (to fix):")
        for q, c, idx in BROKEN_FUNDAMENTALS:
            correct, _ = self._evaluate_question(q, c, idx)
            status = "✓" if correct else "✗"
            logger.info(f"  [{status}] {q} = {c[idx]}")

        logger.info("\nCorrect fundamentals (to preserve):")
        for q, c, idx in CORRECT_FUNDAMENTALS:
            correct, _ = self._evaluate_question(q, c, idx)
            status = "✓" if correct else "✗"
            logger.info(f"  [{status}] {q} = {c[idx]}")

        results = {"initial": initial, "scales": {}}

        # Try multiple scales
        for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
            logger.info(f"\n{'='*60}")
            logger.info(f"SCALE: {scale}")
            logger.info("=" * 60)
            self._reset_weights(layers)

            success = self.apply_alignment(layers[0], scale)

            if not success:
                logger.info("  FAILED to apply alignment")
                continue

            final = self.evaluate_fundamentals()

            broken_improved = final["broken_accuracy"] > initial["broken_accuracy"]
            preserved_stable = final["preserved_accuracy"] >= initial["preserved_accuracy"] - 0.1

            logger.info(f"\nResults:")
            logger.info(f"  Broken: {initial['broken_accuracy']:.0%} → {final['broken_accuracy']:.0%}")
            logger.info(f"  Preserved: {initial['preserved_accuracy']:.0%} → {final['preserved_accuracy']:.0%}")

            # Check individual questions
            logger.info(f"\nBroken fundamentals after alignment:")
            for q, c, idx in BROKEN_FUNDAMENTALS:
                correct, _ = self._evaluate_question(q, c, idx)
                status = "✓ FIXED" if correct else "✗ still wrong"
                logger.info(f"  [{status}] {q} = {c[idx]}")

            logger.info(f"\nPreserved fundamentals after alignment:")
            for q, c, idx in CORRECT_FUNDAMENTALS:
                correct, _ = self._evaluate_question(q, c, idx)
                status = "✓" if correct else "✗ DEGRADED"
                logger.info(f"  [{status}] {q} = {c[idx]}")

            if broken_improved and preserved_stable:
                status = "SUCCESS"
                logger.info(f"\n*** SUCCESS: Fundamentals improved without degradation! ***")
            elif broken_improved:
                status = "IMPROVED_BUT_DEGRADED"
                logger.info(f"\n*** PARTIAL: Broken improved but preserved degraded ***")
            elif preserved_stable:
                status = "NO_IMPROVEMENT"
                logger.info(f"\n*** NO IMPROVEMENT: Broken still broken, preserved stable ***")
            else:
                status = "DEGRADED"
                logger.info(f"\n*** DEGRADED: Both got worse ***")

            results["scales"][str(scale)] = {
                "final": final,
                "broken_improved": broken_improved,
                "preserved_stable": preserved_stable,
                "status": status,
            }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        successes = [s for s, data in results["scales"].items() if data["status"] == "SUCCESS"]
        if successes:
            logger.info(f"\n*** {len(successes)} SUCCESSFUL SCALES: {successes} ***")
            results["conclusion"] = "fundamentals_fixed"
            logger.info("The broken fundamentals have been fixed!")
            logger.info("The model can now be trained on higher-level math.")
        else:
            improved = [s for s, data in results["scales"].items() if data["broken_improved"]]
            if improved:
                logger.info(f"\n{len(improved)} scales showed improvement but with preservation issues")
                results["conclusion"] = "partial_fix"
            else:
                logger.info(f"\nNo scales improved the broken fundamentals")
                results["conclusion"] = "unfixable_via_gradient"

        self._reset_weights(layers)
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = FundamentalAligner(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/fundamental_alignment.json"
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
