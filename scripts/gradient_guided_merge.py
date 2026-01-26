#!/usr/bin/env python3
"""Experiment 27: Gradient-Guided Merge.

Apply gradient-guided principles to model merging/capability transfer.

Connection to ModelCypher: The original goal is model merging. If gradient
information can separate semantic capabilities, it might guide which
knowledge to transfer.

Method (Conceptual - same architecture for simplicity):
1. Load source model (better at math)
2. Load target model (baseline)
3. Identify capability gradients in source's "math improvement" direction
4. Find transfer direction orthogonal to target's preservation gradients
5. Apply transfer

This is a proof-of-concept using the same architecture (LFM2-350M vs LFM2-700M)
to test if gradient-guided transfer can work.
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


class GradientGuidedMerge:
    def __init__(self, target_model, target_tokenizer, source_model, source_tokenizer):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer

        self.target_n_layers = len(target_model.model.layers)
        self.source_n_layers = len(source_model.model.layers)

        self._target_original_weights = {}

    def _get_weight(self, model, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_target_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.target_model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        new_weight = mx.array(weights.astype(np.float32))
        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight
        mx.eval(new_weight)

    def _cache_target_weights(self, layers: List[int]):
        self._target_original_weights = {i: self._get_weight(self.target_model, i).copy() for i in layers}

    def _reset_target_weights(self, layers: List[int]):
        for i in layers:
            if i in self._target_original_weights:
                self._set_target_weight(i, self._target_original_weights[i])

    def _evaluate_question(self, model, tokenizer, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float]:
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = tokenizer.encode(variant)
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

    def evaluate_by_category(self, model, tokenizer) -> Dict[str, float]:
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(model, tokenizer, q, c, idx)[0])
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def compute_weight_difference_direction(
        self,
        target_layer_idx: int,
        source_layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the direction of weight difference in SVD space.

        Returns the SVD components and the difference direction.
        """
        W_target = self._target_original_weights[target_layer_idx]
        W_source = self._get_weight(self.source_model, source_layer_idx)

        # Handle dimension mismatch (common in cross-architecture)
        if W_target.shape != W_source.shape:
            logger.warning(f"Shape mismatch: target {W_target.shape} vs source {W_source.shape}")
            # Project source to target dimensions
            min_rows = min(W_target.shape[0], W_source.shape[0])
            min_cols = min(W_target.shape[1], W_source.shape[1])
            W_source_proj = W_source[:min_rows, :min_cols]
            W_target_proj = W_target[:min_rows, :min_cols]

            # Compute in the shared subspace
            W_diff = W_source_proj - W_target_proj
        else:
            W_diff = W_source - W_target

        # SVD of target for basis
        U, S, Vt = svd(W_target, full_matrices=False)

        # Project difference onto target's SVD basis
        # W_diff ≈ U @ diff_coeffs @ Vt
        diff_coeffs = U.T @ W_diff @ Vt.T

        # The diagonal of diff_coeffs represents the change in each singular direction
        k = min(20, len(S))
        direction = np.diag(diff_coeffs)[:k]

        if np.linalg.norm(direction) > 1e-10:
            direction = direction / np.linalg.norm(direction)

        return U, S, Vt, direction

    def compute_preservation_direction(
        self,
        layer_idx: int,
        category: str,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Compute gradient direction for preserving a category on target."""
        W = self._target_original_weights[layer_idx]
        questions = CATEGORY_QUESTIONS[category]

        self._set_target_weight(layer_idx, W)
        base_loss = sum(self._evaluate_question(self.target_model, self.target_tokenizer, q, c, idx)[1]
                       for q, c, idx in questions)

        U, S, Vt = svd(W, full_matrices=False)
        k = min(20, len(S))
        gradient = np.zeros(k)

        for i in range(k):
            S_perturbed = S.copy()
            S_perturbed[i] += epsilon * S[i]

            W_perturbed = U @ np.diag(S_perturbed) @ Vt
            if np.all(np.isfinite(W_perturbed)):
                self._set_target_weight(layer_idx, W_perturbed)
                perturbed_loss = sum(self._evaluate_question(self.target_model, self.target_tokenizer, q, c, idx)[1]
                                    for q, c, idx in questions)
                gradient[i] = (perturbed_loss - base_loss) / (epsilon * S[i])

        self._set_target_weight(layer_idx, W)

        if np.linalg.norm(gradient) > 1e-10:
            return gradient / np.linalg.norm(gradient)
        return gradient

    def project_orthogonal(
        self,
        direction: np.ndarray,
        preserve_directions: List[np.ndarray]
    ) -> np.ndarray:
        result = direction.copy()
        for preserve_dir in preserve_directions:
            if np.linalg.norm(preserve_dir) > 1e-10:
                projection = np.dot(result, preserve_dir) * preserve_dir
                result = result - projection
        return result

    def apply_guided_transfer(
        self,
        target_layer_idx: int,
        transfer_direction: np.ndarray,
        scale: float
    ) -> bool:
        """Apply transfer in the guided direction."""
        W = self._target_original_weights[target_layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        S_modified = S.copy()
        for i in range(min(len(transfer_direction), len(S))):
            S_modified[i] += scale * transfer_direction[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_target_weight(target_layer_idx, W_modified)
            return True
        return False

    def run_experiment(self) -> Dict:
        # Use middle layers
        target_layer = self.target_n_layers // 2
        source_layer = self.source_n_layers // 2

        self._cache_target_weights([target_layer])

        logger.info("=" * 60)
        logger.info("GRADIENT-GUIDED MERGE")
        logger.info("=" * 60)
        logger.info(f"Target layers: {self.target_n_layers}, Source layers: {self.source_n_layers}")
        logger.info(f"Testing target layer {target_layer}, source layer {source_layer}")

        # Evaluate baselines
        logger.info("\nEvaluating source model...")
        source_baseline = self.evaluate_by_category(self.source_model, self.source_tokenizer)

        logger.info("\nEvaluating target model...")
        target_baseline = self.evaluate_by_category(self.target_model, self.target_tokenizer)

        logger.info("\nSource model baseline:")
        for cat, acc in sorted(source_baseline.items(), key=lambda x: x[1]):
            if cat != "overall":
                logger.info(f"  {cat}: {acc:.0%}")

        logger.info("\nTarget model baseline:")
        for cat, acc in sorted(target_baseline.items(), key=lambda x: x[1]):
            if cat != "overall":
                logger.info(f"  {cat}: {acc:.0%}")

        # Identify what source is better at
        source_better = [cat for cat in CATEGORY_QUESTIONS
                        if source_baseline[cat] > target_baseline[cat]]
        target_better = [cat for cat in CATEGORY_QUESTIONS
                        if target_baseline[cat] > source_baseline[cat]]

        logger.info(f"\nSource better at: {source_better}")
        logger.info(f"Target better at: {target_better}")

        results = {
            "source_baseline": source_baseline,
            "target_baseline": target_baseline,
            "source_better": source_better,
            "target_better": target_better,
            "experiments": {},
        }

        if not source_better:
            logger.info("Source model is not better at anything - cannot transfer")
            results["error"] = "source_not_better"
            return results

        # Strategy: Transfer source's advantage while preserving target's strengths
        transfer_cat = source_better[0] if source_better else "math"
        preserve_cats = target_better[:2] if target_better else ["geography"]

        logger.info(f"\nTransfer strategy:")
        logger.info(f"  Transfer from source: {transfer_cat}")
        logger.info(f"  Preserve in target: {preserve_cats}")

        # Compute weight difference direction
        logger.info("\nComputing weight difference direction...")
        U, S, Vt, diff_direction = self.compute_weight_difference_direction(
            target_layer, source_layer
        )

        # Compute preservation gradients
        logger.info("Computing preservation gradients...")
        preserve_dirs = []
        for cat in preserve_cats:
            logger.info(f"  {cat}...")
            preserve_dir = self.compute_preservation_direction(target_layer, cat)
            preserve_dirs.append(preserve_dir)

        # Project difference direction orthogonal to preservation
        guided_direction = self.project_orthogonal(diff_direction, preserve_dirs)
        guided_norm = np.linalg.norm(guided_direction)

        logger.info(f"\nGuided direction norm after projection: {guided_norm:.2%}")

        if guided_norm < 0.1:
            logger.info("Guided direction too small - transfer would harm preservation")
            results["error"] = "direction_too_small"
            return results

        guided_direction = guided_direction / guided_norm

        # Test different transfer scales
        scales = [0.5, 1.0, 2.0]

        for scale in scales:
            self._reset_target_weights([target_layer])
            logger.info(f"\n--- Scale: {scale} ---")

            success = self.apply_guided_transfer(target_layer, guided_direction, scale)
            if not success:
                logger.info("  Transfer failed (numerical issues)")
                continue

            final = self.evaluate_by_category(self.target_model, self.target_tokenizer)
            changes = {k: final[k] - target_baseline[k] for k in target_baseline}

            logger.info("  Results:")
            for cat in CATEGORY_QUESTIONS:
                status = "✓" if changes[cat] >= 0 else "✗"
                logger.info(f"    {cat}: {final[cat]:.0%} ({changes[cat]:+.0%}) {status}")

            # Check success
            transferred = changes.get(transfer_cat, 0) > 0
            preserved = all(changes[cat] >= -0.01 for cat in preserve_cats)

            if transferred and preserved:
                logger.info("  *** SUCCESS: Transferred capability while preserving! ***")

            results["experiments"][f"scale_{scale}"] = {
                "final": final,
                "changes": changes,
                "transferred": transferred,
                "preserved": preserved,
                "success": transferred and preserved,
            }

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        successes = [(k, v) for k, v in results["experiments"].items() if v.get("success", False)]
        if successes:
            logger.info(f"\n*** {len(successes)} SUCCESSES ***")
            for key, _ in successes:
                logger.info(f"  {key}")
            results["overall_success"] = True
        else:
            logger.info("\nNo configurations achieved successful transfer")
            results["overall_success"] = False

        self._reset_target_weights([target_layer])
        return results


def main():
    from mlx_lm import load

    # Load target (smaller model)
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"Loading target model: {target_path}")
    target_model, target_tokenizer = load(target_path)

    # Load source (larger model, hopefully better)
    source_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16"
    logger.info(f"Loading source model: {source_path}")

    if not Path(source_path).exists():
        logger.warning(f"Source model not found, using same model (will be a no-op)")
        source_model, source_tokenizer = target_model, target_tokenizer
    else:
        source_model, source_tokenizer = load(source_path)

    test = GradientGuidedMerge(target_model, target_tokenizer, source_model, source_tokenizer)
    results = test.run_experiment()

    output_path = "data/experiments/gradient_guided_merge.json"
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
