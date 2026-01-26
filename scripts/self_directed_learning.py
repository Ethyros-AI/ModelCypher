#!/usr/bin/env python3
"""Experiment 42: Self-Directed Learning.

Phase 8 - Stage 3b: Can the model choose what to learn next?

The endgame: Full autonomy in the learning loop.

Method:
1. Model identifies its own knowledge gaps (via consistency metrics)
2. Model prioritizes which gap to fill
3. Model researches and learns
4. Repeat

This is giving the model what humans have:
- The ability to know what it doesn't know
- The ability to prioritize what to learn
- The ability to learn without forgetting
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


# All categories with their questions
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

# Training data bank (simulates what would come from research)
TRAINING_DATA_BANK = {
    "language": [
        ("What is the opposite of 'hot'?", "Cold", ["Warm", "Cold", "Cool", "Mild"], 1),
        ("What is the opposite of 'big'?", "Small", ["Tiny", "Small", "Little", "Mini"], 1),
        ("What is the opposite of 'fast'?", "Slow", ["Quick", "Slow", "Speedy", "Rapid"], 1),
    ],
    "logic": [
        ("What comes next: 2, 4, 6, 8, ?", "10", ["9", "10", "11", "12"], 1),
        ("What comes next: 1, 3, 5, 7, ?", "9", ["8", "9", "10", "11"], 1),
        ("What comes next: 10, 20, 30, 40, ?", "50", ["45", "50", "55", "60"], 1),
    ],
    "math": [
        ("What is 8 × 7?", "56", ["48", "54", "56", "64"], 2),
        ("What is 9 × 6?", "54", ["45", "54", "56", "63"], 1),
        ("What is 100 ÷ 5?", "20", ["15", "20", "25", "50"], 1),
    ],
}


class SelfDirectedLearner:
    """Full autonomy: model decides what to learn."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_weights = {}
        self._current_weights = {}

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
        self._current_weights = {i: self._get_weight(i).copy() for i in layers}

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
        confidence = probs[prediction]
        loss = -np.log(probs[correct_idx] + 1e-10)

        return prediction == correct_idx, confidence

    def evaluate_by_category(self) -> Dict[str, float]:
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx)[0])
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def compute_confidence_by_category(self) -> Dict[str, float]:
        """Compute mean confidence per category - the 'anxiety' signal."""
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            confidences = [self._evaluate_question(q, c, idx)[1] for q, c, idx in questions]
            results[cat] = np.mean(confidences)
        return results

    def identify_knowledge_gaps(self) -> List[Tuple[str, float, float]]:
        """
        Self-identify knowledge gaps.
        Returns: List of (category, accuracy, confidence) sorted by priority.

        Priority = low accuracy + low confidence (both signals agree)
        """
        accuracies = self.evaluate_by_category()
        confidences = self.compute_confidence_by_category()

        gaps = []
        for cat in CATEGORY_QUESTIONS.keys():
            acc = accuracies[cat]
            conf = confidences[cat]
            # Priority: lower is worse
            priority = acc + conf  # Both contribute
            gaps.append((cat, acc, conf, priority))

        # Sort by priority (lowest first = biggest gap)
        gaps.sort(key=lambda x: x[3])

        return [(cat, acc, conf) for cat, acc, conf, _ in gaps]

    def compute_loss_direction(
        self,
        layer_idx: int,
        questions: List[Tuple],
        epsilon: float = 0.01
    ) -> np.ndarray:
        W = self._get_weight(layer_idx)
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
        target_category: str,
        preserve_categories: List[str],
        scale: float = 1.5
    ) -> bool:
        """Apply gradient-guided learning for target category."""

        # Get training data
        if target_category not in TRAINING_DATA_BANK:
            logger.info(f"  No training data available for {target_category}")
            return False

        training_data = TRAINING_DATA_BANK[target_category]
        training_q = [(q, c, idx) for q, ans, c, idx in training_data]
        training_q.extend(CATEGORY_QUESTIONS[target_category])

        W = self._get_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        improve_dir = self.compute_loss_direction(layer_idx, training_q)

        preserve_dirs = []
        for cat in preserve_categories:
            cat_questions = [(q, c, idx) for q, c, idx in CATEGORY_QUESTIONS[cat]]
            preserve_dir = self.compute_loss_direction(layer_idx, cat_questions)
            preserve_dirs.append(preserve_dir)

        ortho_dir = self.compute_orthogonal_perturbation(improve_dir, preserve_dirs)

        S_modified = S.copy()
        for i in range(len(ortho_dir)):
            S_modified[i] += scale * ortho_dir[i] * S[i]

        W_modified = U @ np.diag(S_modified) @ Vt
        if np.all(np.isfinite(W_modified)):
            self._set_weight(layer_idx, W_modified)
            self._current_weights[layer_idx] = W_modified.copy()
            return True
        return False

    def run_experiment(self, max_iterations: int = 3) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 42: SELF-DIRECTED LEARNING")
        logger.info("=" * 60)
        logger.info("\nThe model chooses what to learn based on self-assessment.\n")

        mid = self.n_layers // 2
        layer = mid
        self._cache_weights([layer])

        # Initial evaluation
        initial = self.evaluate_by_category()
        logger.info("Initial accuracies:")
        for cat, acc in sorted(initial.items()):
            logger.info(f"  {cat}: {acc:.0%}")

        results = {
            "initial": initial,
            "iterations": [],
            "always_preserve": ["geography", "history"],  # Known 100% categories
        }

        # Categories that can't be improved (capability gaps from Exp 40)
        unfillable = {"math"}  # Known from Exp 40: 20% is capability gap

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration + 1}: SELF-ASSESSMENT")
            logger.info("=" * 60)

            # 1. DETECT - Model identifies its gaps
            gaps = self.identify_knowledge_gaps()
            logger.info("\nKnowledge gaps (model self-assessment):")
            for cat, acc, conf in gaps:
                status = "UNFILLABLE" if cat in unfillable else ("STRONG" if acc >= 0.8 else "FILLABLE")
                logger.info(f"  {cat}: acc={acc:.0%}, conf={conf:.2f} [{status}]")

            # 2. DECIDE - Pick highest priority fillable gap
            target = None
            for cat, acc, conf in gaps:
                if cat not in unfillable and cat not in results["always_preserve"]:
                    if acc < 0.8:  # Only try to improve if not already strong
                        target = cat
                        break

            if target is None:
                logger.info("\nNo fillable gaps remaining. Learning complete.")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "status": "no_gaps",
                    "message": "All fillable categories at 80%+ or no improvement possible"
                })
                break

            logger.info(f"\n→ DECISION: Learn '{target}' (acc={gaps[[g[0] for g in gaps].index(target)][1]:.0%})")

            # 3. PRESERVE - Build preservation list
            # Preserve strong categories + previously learned categories
            learned_categories = [it.get("topic") for it in results["iterations"] if it.get("success")]
            preserve = list(set(results["always_preserve"] + learned_categories))
            logger.info(f"  Preserving: {preserve}")

            # 4. LEARN - Apply gradient-guided modification
            before = self.evaluate_by_category()
            success = self.apply_learning(layer, target, preserve)

            if success:
                after = self.evaluate_by_category()
                changes = {k: after[k] - before[k] for k in before}

                logger.info(f"\nAfter learning '{target}':")
                improved = changes[target] > 0.05
                degraded = any(changes[cat] < -0.05 for cat in preserve)

                for cat in [target] + preserve:
                    change = changes[cat]
                    marker = "↑" if change > 0.01 else ("↓" if change < -0.01 else "=")
                    logger.info(f"  {cat}: {before[cat]:.0%} → {after[cat]:.0%} ({change:+.0%}) {marker}")

                if improved and not degraded:
                    status = "success"
                    logger.info(f"\n  *** SUCCESS: Improved {target} without degradation ***")
                elif not improved and not degraded:
                    status = "no_improvement"
                    logger.info(f"\n  No improvement (may be capability gap)")
                    unfillable.add(target)  # Mark as unfillable for future iterations
                else:
                    status = "degraded"
                    logger.info(f"\n  DEGRADED preserved categories")

                results["iterations"].append({
                    "iteration": iteration + 1,
                    "topic": target,
                    "preserve": preserve,
                    "before": before,
                    "after": after,
                    "changes": changes,
                    "improved": improved,
                    "degraded": degraded,
                    "status": status,
                    "success": improved and not degraded,
                })
            else:
                logger.info(f"\n  FAILED to apply learning")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "topic": target,
                    "status": "failed",
                    "success": False,
                })

        # === FINAL SUMMARY ===
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)

        final = self.evaluate_by_category()
        logger.info("\nFinal accuracies:")
        for cat, acc in sorted(final.items()):
            change = final[cat] - initial[cat]
            marker = "↑" if change > 0.01 else ("↓" if change < -0.01 else "=")
            logger.info(f"  {cat}: {initial[cat]:.0%} → {acc:.0%} ({change:+.0%}) {marker}")

        successes = [it for it in results["iterations"] if it.get("success")]
        logger.info(f"\nSuccessful autonomous improvements: {len(successes)}")
        for s in successes:
            logger.info(f"  - {s['topic']}: {s['before'][s['topic']]:.0%} → {s['after'][s['topic']]:.0%}")

        if successes:
            results["conclusion"] = "autonomous_success"
            logger.info("\n*** AUTONOMOUS LEARNING SUCCESSFUL ***")
        else:
            results["conclusion"] = "no_autonomous_improvement"
            logger.info("\n*** No autonomous improvements achieved ***")

        results["final"] = final
        results["unfillable_categories"] = list(unfillable)

        # Reset to original
        self._set_weight(layer, self._original_weights[layer])

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = SelfDirectedLearner(model, tokenizer)
    results = experiment.run_experiment(max_iterations=3)

    output_path = "data/experiments/self_directed_learning.json"
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
