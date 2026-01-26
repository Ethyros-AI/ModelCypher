#!/usr/bin/env python3
"""Experiment 62: Sharpness-Based Training.

The geometry tells us:
- Symbolic already has correct answer as top-1 (16% confident)
- Counting has 60% confidence
- We need to increase symbolic confidence 3x

The target is NOT arbitrary - it comes from measuring what
"working" looks like (counting) and matching that.

Loss = (target_sharpness - current_sharpness)²
where sharpness is measured as logit gap or concentration.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Training pairs with expected outputs
TRAINING_PAIRS = [
    ("4+1=", "5"),
    ("5+1=", "6"),
    ("6+1=", "7"),
    ("7+1=", "8"),
    ("8+1=", "9"),
    ("3+1=", "4"),
    ("2+1=", "3"),
    ("1+1=", "2"),
]


class SharpnessTrainer:
    """Train for sharpness, not just correctness."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Target sharpness from counting (measured in Exp 61)
        self.target_gap = 1.00  # Counting mean gap
        self.target_concentration = 1.61  # Counting mean concentration

        # Numerical precision
        self.eps = np.finfo(np.float32).eps

    def get_logits(self, prompt):
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return logits[0, -1, :]

    def measure_sharpness(self, logits):
        """Measure logit sharpness metrics."""
        import mlx.core as mx

        # Sort to get top values
        sorted_logits = mx.sort(logits)[::-1]
        mx.eval(sorted_logits)

        top1 = float(sorted_logits[0].item())
        top2 = float(sorted_logits[1].item())

        gap = top1 - top2

        # Probability concentration
        probs = mx.softmax(logits)
        mx.eval(probs)
        max_prob = float(mx.max(probs).item())
        concentration = max_prob / (1 - max_prob + 1e-10)

        return gap, concentration, max_prob

    def evaluate(self):
        """Evaluate accuracy and sharpness."""
        import mlx.core as mx

        results = []
        for prompt, expected in TRAINING_PAIRS:
            logits = self.get_logits(prompt)
            mx.eval(logits)

            # Accuracy
            top_token = int(mx.argmax(logits).item())
            predicted = self.tokenizer.decode([top_token]).strip()
            correct = expected in predicted or predicted == expected

            # Sharpness
            gap, concentration, max_prob = self.measure_sharpness(logits)

            # Target token probability
            target_tokens = self.tokenizer.encode(expected)
            if target_tokens:
                target_id = target_tokens[0]
                probs = mx.softmax(logits)
                mx.eval(probs)
                target_prob = float(probs[target_id].item())
            else:
                target_prob = 0.0

            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "gap": gap,
                "concentration": concentration,
                "max_prob": max_prob,
                "target_prob": target_prob,
            })

        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_gap = np.mean([r["gap"] for r in results])
        mean_conc = np.mean([r["concentration"] for r in results])
        mean_target = np.mean([r["target_prob"] for r in results])

        return {
            "accuracy": accuracy,
            "mean_gap": mean_gap,
            "mean_concentration": mean_conc,
            "mean_target_prob": mean_target,
            "details": results,
        }

    def sharpness_loss_fn(self, model):
        """Loss based on matching target sharpness."""
        import mlx.core as mx

        total_loss = mx.array(0.0)

        for prompt, expected in TRAINING_PAIRS:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            next_logits = logits[0, -1, :]

            # Get target token
            target_tokens = self.tokenizer.encode(expected)
            if not target_tokens:
                continue
            target_id = target_tokens[0]

            # Loss 1: Target token should be top-1 (cross-entropy)
            log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
            ce_loss = -log_probs[target_id]

            # Loss 2: Sharpness should match target
            # Gap loss: (current_gap - target_gap)²
            sorted_logits = mx.sort(next_logits)[::-1]
            current_gap = sorted_logits[0] - sorted_logits[1]
            gap_loss = (current_gap - self.target_gap) ** 2

            # Total loss: CE + sharpness matching
            # Weight sharpness loss by geometry-derived factor
            # The gap ratio is 3.08x, so we weight gap loss accordingly
            loss = ce_loss + (1.0 / 3.08) * gap_loss

            total_loss = total_loss + loss

        return total_loss / len(TRAINING_PAIRS)

    def training_step(self, lr):
        """Single training step."""
        import mlx.core as mx
        import mlx.optimizers as optim

        loss, grads = mx.value_and_grad(self.sharpness_loss_fn)(self.model)
        mx.eval(loss, grads)

        optimizer = optim.SGD(learning_rate=lr)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return float(loss.item())

    def run_experiment(self):
        """Run sharpness-based training."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT 62: SHARPNESS-BASED TRAINING")
        logger.info("=" * 60)
        logger.info(f"\nTarget sharpness (from counting geometry):")
        logger.info(f"  Gap: {self.target_gap:.2f}")
        logger.info(f"  Concentration: {self.target_concentration:.2f}")

        # Initial evaluation
        initial_eval = self.evaluate()
        logger.info(f"\n=== INITIAL STATE ===")
        logger.info(f"Accuracy: {initial_eval['accuracy']:.0%}")
        logger.info(f"Mean gap: {initial_eval['mean_gap']:.2f} (target: {self.target_gap:.2f})")
        logger.info(f"Mean concentration: {initial_eval['mean_concentration']:.2f} (target: {self.target_concentration:.2f})")
        logger.info(f"Mean target prob: {initial_eval['mean_target_prob']:.1%}")

        for r in initial_eval["details"]:
            logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({'✓' if r['correct'] else '✗'}) "
                       f"gap={r['gap']:.2f}, p={r['target_prob']:.1%}")

        results = {
            "target_sharpness": {
                "gap": self.target_gap,
                "concentration": self.target_concentration,
            },
            "initial": {
                "accuracy": initial_eval["accuracy"],
                "mean_gap": initial_eval["mean_gap"],
                "mean_concentration": initial_eval["mean_concentration"],
                "mean_target_prob": initial_eval["mean_target_prob"],
            },
            "iterations": [],
        }

        # Training - LR from geometry
        # The gap needs to change by ~0.67 (from 0.33 to 1.00)
        # A single step should change logits by O(lr × grad)
        # Start with a small lr and see what happens
        lr = 1e-5  # Start here

        logger.info(f"\n=== TRAINING ===")
        logger.info(f"Learning rate: {lr}")

        prev_gap = initial_eval["mean_gap"]

        for iteration in range(50):
            loss = self.training_step(lr)

            # Evaluate every 5 iterations
            if iteration % 5 == 0:
                eval_result = self.evaluate()
                gap_change = eval_result["mean_gap"] - prev_gap

                logger.info(f"Iter {iteration}: loss={loss:.4f}, "
                           f"acc={eval_result['accuracy']:.0%}, "
                           f"gap={eval_result['mean_gap']:.2f} (Δ={gap_change:+.3f}), "
                           f"p_target={eval_result['mean_target_prob']:.1%}")

                results["iterations"].append({
                    "iteration": iteration,
                    "loss": loss,
                    "accuracy": eval_result["accuracy"],
                    "mean_gap": eval_result["mean_gap"],
                    "mean_target_prob": eval_result["mean_target_prob"],
                })

                # Geometry-derived stopping: when gap reaches target
                if eval_result["mean_gap"] >= self.target_gap:
                    logger.info(f"STOPPED: Gap reached target ({eval_result['mean_gap']:.2f} >= {self.target_gap:.2f})")
                    results["stop_reason"] = "target_reached"
                    break

                # Stop if gap decreases (wrong direction)
                if gap_change < -0.1:
                    logger.info(f"STOPPED: Gap decreasing (Δ={gap_change:.3f})")
                    results["stop_reason"] = "gap_decreasing"
                    break

                prev_gap = eval_result["mean_gap"]
        else:
            results["stop_reason"] = "max_iterations"

        # Final evaluation
        final_eval = self.evaluate()
        logger.info(f"\n=== FINAL STATE ===")
        logger.info(f"Accuracy: {final_eval['accuracy']:.0%}")
        logger.info(f"Mean gap: {final_eval['mean_gap']:.2f} (target: {self.target_gap:.2f})")
        logger.info(f"Mean target prob: {final_eval['mean_target_prob']:.1%}")

        for r in final_eval["details"]:
            logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({'✓' if r['correct'] else '✗'}) "
                       f"gap={r['gap']:.2f}, p={r['target_prob']:.1%}")

        results["final"] = {
            "accuracy": final_eval["accuracy"],
            "mean_gap": final_eval["mean_gap"],
            "mean_concentration": final_eval["mean_concentration"],
            "mean_target_prob": final_eval["mean_target_prob"],
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {initial_eval['accuracy']:.0%} → {final_eval['accuracy']:.0%}")
        logger.info(f"Gap: {initial_eval['mean_gap']:.2f} → {final_eval['mean_gap']:.2f}")
        logger.info(f"Target prob: {initial_eval['mean_target_prob']:.1%} → {final_eval['mean_target_prob']:.1%}")

        if final_eval["accuracy"] > initial_eval["accuracy"]:
            logger.info("\n*** SHARPNESS TRAINING IMPROVED ACCURACY ***")
            results["conclusion"] = "success"
        else:
            logger.info("\n*** SHARPNESS TRAINING DID NOT IMPROVE ***")
            results["conclusion"] = "failed"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    trainer = SharpnessTrainer(model, tokenizer)
    results = trainer.run_experiment()

    output_path = "data/experiments/sharpness_training.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
