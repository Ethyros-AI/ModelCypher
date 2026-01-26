#!/usr/bin/env python3
"""Experiment 63: Targeted Sharpness Training.

Previous experiment increased sharpness but on WRONG tokens.
We need to increase sharpness specifically on the CORRECT token.

The loss should:
1. Make target token's logit the highest
2. Make gap between target and second-highest match counting's gap

This is more targeted than just "be sharper".
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


class TargetedSharpnessTrainer:
    """Train for sharpness on the CORRECT token."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.target_gap = 1.00  # From counting geometry

    def get_logits(self, prompt):
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return logits[0, -1, :]

    def evaluate(self):
        """Evaluate accuracy and targeted sharpness."""
        import mlx.core as mx

        results = []
        for prompt, expected in TRAINING_PAIRS:
            logits = self.get_logits(prompt)
            mx.eval(logits)

            # Get target token
            target_tokens = self.tokenizer.encode(expected)
            target_id = target_tokens[0] if target_tokens else -1

            # Top prediction
            top_token = int(mx.argmax(logits).item())
            predicted = self.tokenizer.decode([top_token]).strip()
            correct = expected in predicted or predicted == expected

            # Target token's logit vs max other logit
            logits_np = np.array(logits.tolist())
            target_logit = logits_np[target_id]

            # Max logit excluding target
            mask = np.ones_like(logits_np, dtype=bool)
            mask[target_id] = False
            max_other = logits_np[mask].max()

            # Target gap: how much higher is target than runner-up?
            target_gap = target_logit - max_other

            # Probabilities
            probs = mx.softmax(logits)
            mx.eval(probs)
            target_prob = float(probs[target_id].item())

            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "target_gap": target_gap,
                "target_prob": target_prob,
            })

        accuracy = sum(r["correct"] for r in results) / len(results)
        mean_target_gap = np.mean([r["target_gap"] for r in results])
        mean_target_prob = np.mean([r["target_prob"] for r in results])

        return {
            "accuracy": accuracy,
            "mean_target_gap": mean_target_gap,
            "mean_target_prob": mean_target_prob,
            "details": results,
        }

    def targeted_loss_fn(self, model):
        """Loss that targets sharpness ON the correct token."""
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

            # Loss 1: Cross-entropy (target should be top)
            log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
            ce_loss = -log_probs[target_id]

            # Loss 2: Target gap should match counting's gap
            # target_gap = target_logit - max_other_logit
            target_logit = next_logits[target_id]

            # Create mask to exclude target
            n_logits = next_logits.shape[0]
            mask = mx.ones(n_logits)
            mask = mask.at[target_id].add(-1.0)  # mask[target_id] = 0

            # Max other = max(logits * mask + (1-mask) * -inf)
            masked_logits = next_logits * mask + (1 - mask) * (-1e10)
            max_other = mx.max(masked_logits)

            current_target_gap = target_logit - max_other

            # We want target_gap >= target_gap_goal
            # Loss = max(0, target_gap_goal - current_target_gap)²
            # This only penalizes when gap is too small
            gap_deficit = mx.maximum(mx.array(0.0), mx.array(self.target_gap) - current_target_gap)
            gap_loss = gap_deficit ** 2

            # Combined loss
            # Weight by 1/gap_ratio since gap is smaller than CE typically
            loss = ce_loss + 0.5 * gap_loss

            total_loss = total_loss + loss

        return total_loss / len(TRAINING_PAIRS)

    def training_step(self, lr):
        """Single training step."""
        import mlx.core as mx
        import mlx.optimizers as optim

        loss, grads = mx.value_and_grad(self.targeted_loss_fn)(self.model)
        mx.eval(loss, grads)

        optimizer = optim.SGD(learning_rate=lr)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return float(loss.item())

    def run_experiment(self):
        """Run targeted sharpness training."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT 63: TARGETED SHARPNESS TRAINING")
        logger.info("=" * 60)
        logger.info(f"\nTarget gap (from counting): {self.target_gap:.2f}")
        logger.info("Sharpness measured on CORRECT token, not any token")

        # Initial evaluation
        initial_eval = self.evaluate()
        logger.info(f"\n=== INITIAL STATE ===")
        logger.info(f"Accuracy: {initial_eval['accuracy']:.0%}")
        logger.info(f"Mean target gap: {initial_eval['mean_target_gap']:.2f} (target: {self.target_gap:.2f})")
        logger.info(f"Mean target prob: {initial_eval['mean_target_prob']:.1%}")

        for r in initial_eval["details"]:
            logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({'✓' if r['correct'] else '✗'}) "
                       f"tgt_gap={r['target_gap']:.2f}, p={r['target_prob']:.1%}")

        results = {
            "target_gap": self.target_gap,
            "initial": {
                "accuracy": initial_eval["accuracy"],
                "mean_target_gap": initial_eval["mean_target_gap"],
                "mean_target_prob": initial_eval["mean_target_prob"],
            },
            "iterations": [],
        }

        # Training
        lr = 1e-5

        logger.info(f"\n=== TRAINING ===")
        logger.info(f"Learning rate: {lr}")

        best_accuracy = initial_eval["accuracy"]
        no_improvement = 0

        for iteration in range(100):
            loss = self.training_step(lr)

            # Evaluate every 5 iterations
            if iteration % 5 == 0:
                eval_result = self.evaluate()

                logger.info(f"Iter {iteration}: loss={loss:.4f}, "
                           f"acc={eval_result['accuracy']:.0%}, "
                           f"tgt_gap={eval_result['mean_target_gap']:.2f}, "
                           f"p_target={eval_result['mean_target_prob']:.1%}")

                results["iterations"].append({
                    "iteration": iteration,
                    "loss": loss,
                    "accuracy": eval_result["accuracy"],
                    "mean_target_gap": eval_result["mean_target_gap"],
                    "mean_target_prob": eval_result["mean_target_prob"],
                })

                # Check for improvement in accuracy
                if eval_result["accuracy"] > best_accuracy:
                    best_accuracy = eval_result["accuracy"]
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Stop if no improvement for 4 eval cycles (20 iterations)
                if no_improvement >= 4:
                    logger.info(f"STOPPED: No accuracy improvement for 20 iterations")
                    results["stop_reason"] = "no_improvement"
                    break

                # Stop if target gap reached and accuracy improved
                if eval_result["mean_target_gap"] >= self.target_gap and eval_result["accuracy"] > initial_eval["accuracy"]:
                    logger.info(f"STOPPED: Target gap reached with improved accuracy")
                    results["stop_reason"] = "target_reached"
                    break

        else:
            results["stop_reason"] = "max_iterations"

        # Final evaluation
        final_eval = self.evaluate()
        logger.info(f"\n=== FINAL STATE ===")
        logger.info(f"Accuracy: {final_eval['accuracy']:.0%}")
        logger.info(f"Mean target gap: {final_eval['mean_target_gap']:.2f}")
        logger.info(f"Mean target prob: {final_eval['mean_target_prob']:.1%}")

        for r in final_eval["details"]:
            logger.info(f"  '{r['prompt']}' → '{r['predicted']}' ({'✓' if r['correct'] else '✗'}) "
                       f"tgt_gap={r['target_gap']:.2f}, p={r['target_prob']:.1%}")

        results["final"] = {
            "accuracy": final_eval["accuracy"],
            "mean_target_gap": final_eval["mean_target_gap"],
            "mean_target_prob": final_eval["mean_target_prob"],
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {initial_eval['accuracy']:.0%} → {final_eval['accuracy']:.0%}")
        logger.info(f"Target gap: {initial_eval['mean_target_gap']:.2f} → {final_eval['mean_target_gap']:.2f}")
        logger.info(f"Target prob: {initial_eval['mean_target_prob']:.1%} → {final_eval['mean_target_prob']:.1%}")

        if final_eval["accuracy"] > initial_eval["accuracy"]:
            logger.info("\n*** TARGETED SHARPNESS TRAINING IMPROVED ACCURACY ***")
            results["conclusion"] = "success"
        else:
            logger.info("\n*** TARGETED SHARPNESS TRAINING DID NOT IMPROVE ***")
            results["conclusion"] = "failed"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    trainer = TargetedSharpnessTrainer(model, tokenizer)
    results = trainer.run_experiment()

    output_path = "data/experiments/targeted_sharpness_training.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
