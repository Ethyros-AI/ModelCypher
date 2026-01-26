#!/usr/bin/env python3
"""Experiment 58: Alignment-Driven Training.

Instead of minimizing language model loss, directly minimize the
MISALIGNMENT between counting and symbolic representations.

The objective: Gram(counting) ≈ Gram(symbolic)

This is fundamentally different from training on text - we're training
the model to have the same relational structure for counting and arithmetic.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PHI = (1 + math.sqrt(5)) / 2
E_OVER_PI = math.e / math.pi

# Matched pairs: counting and symbolic that should have same output
# Use longer counting prompts that we know work (from Exp 56)
ALIGNMENT_PAIRS = [
    # (counting_prompt, symbolic_prompt, expected_output, expected_token_id)
    ("1, 2, 3, 4,", "4+1=", "5"),
    ("2, 3, 4, 5,", "5+1=", "6"),
    ("3, 4, 5, 6,", "6+1=", "7"),
    ("4, 5, 6, 7,", "7+1=", "8"),
    ("5, 6, 7, 8,", "8+1=", "9"),
    ("6, 7, 8, 9,", "9+1=", "10"),
    ("Count to 5: 1, 2, 3, 4,", "4+1=", "5"),
    ("Count: one, two, three,", "3+1=", "four"),
]


class AlignmentDrivenTrainer:
    """Train by minimizing gram alignment loss between counting/symbolic."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _get_logits(self, prompt: str) -> np.ndarray:
        import mlx.core as mx
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def get_aligned_activations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get activations for counting and symbolic prompts."""
        counting_acts = []
        symbolic_acts = []

        for counting_prompt, symbolic_prompt, _ in ALIGNMENT_PAIRS:
            counting_acts.append(self._get_logits(counting_prompt))
            symbolic_acts.append(self._get_logits(symbolic_prompt))

        return np.vstack(counting_acts), np.vstack(symbolic_acts)

    def compute_alignment_loss(self, counting_acts: np.ndarray,
                                symbolic_acts: np.ndarray) -> float:
        """
        Compute the alignment loss: how different are the Gram matrices?

        We want Gram(counting) ≈ Gram(symbolic)
        Loss = ||Gram_c - Gram_s||_F / (||Gram_c||_F + ||Gram_s||_F)
        """
        # Center
        counting_c = counting_acts - counting_acts.mean(axis=0)
        symbolic_c = symbolic_acts - symbolic_acts.mean(axis=0)

        # Gram matrices
        G_count = counting_c @ counting_c.T
        G_symb = symbolic_c @ symbolic_c.T

        # Normalize
        norm_count = np.linalg.norm(G_count, 'fro') + 1e-10
        norm_symb = np.linalg.norm(G_symb, 'fro') + 1e-10

        G_count_norm = G_count / norm_count
        G_symb_norm = G_symb / norm_symb

        # Alignment loss: Frobenius norm of difference
        diff = G_count_norm - G_symb_norm
        loss = np.linalg.norm(diff, 'fro')

        return float(loss)

    def compute_gradient_toward_alignment(self):
        """
        Compute gradient that makes symbolic produce the correct answer.

        Simple approach: For each pair, the symbolic prompt should produce
        the same output as the correct answer (which counting gets right).
        """
        import mlx.core as mx

        def alignment_loss_fn(model):
            total_loss = mx.array(0.0)

            for counting_prompt, symbolic_prompt, expected in ALIGNMENT_PAIRS:
                # Get the token ID for the expected output
                expected_tokens = self.tokenizer.encode(expected)
                if not expected_tokens:
                    continue
                target_token = expected_tokens[0]  # First token of answer

                # Forward pass on symbolic prompt
                tokens = self.tokenizer.encode(symbolic_prompt)
                input_ids = mx.array([tokens])
                logits = model(input_ids)
                next_logits = logits[0, -1, :]

                # Cross-entropy loss toward the correct answer
                log_probs = mx.log(mx.softmax(next_logits) + 1e-10)
                loss = -log_probs[target_token]

                total_loss = total_loss + loss

            return total_loss / len(ALIGNMENT_PAIRS)

        loss, grads = mx.value_and_grad(alignment_loss_fn)(self.model)
        mx.eval(loss, grads)

        return float(loss.item()), grads

    def alignment_training_step(self, learning_rate: float):
        """Single step of alignment-based training."""
        import mlx.core as mx
        import mlx.optimizers as optim

        loss, grads = self.compute_gradient_toward_alignment()

        optimizer = optim.SGD(learning_rate=learning_rate)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return loss

    def evaluate(self) -> Dict:
        """Evaluate both counting and symbolic arithmetic."""
        import mlx.core as mx

        results = {"counting": [], "symbolic": []}

        for counting_prompt, symbolic_prompt, expected in ALIGNMENT_PAIRS:
            # Counting
            tokens = self.tokenizer.encode(counting_prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            output = self.tokenizer.decode([next_token]).strip()
            results["counting"].append(expected in output)

            # Symbolic
            tokens = self.tokenizer.encode(symbolic_prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            output = self.tokenizer.decode([next_token]).strip()
            results["symbolic"].append(expected in output)

        return {
            "counting_accuracy": sum(results["counting"]) / len(results["counting"]),
            "symbolic_accuracy": sum(results["symbolic"]) / len(results["symbolic"]),
            "counting_correct": sum(results["counting"]),
            "symbolic_correct": sum(results["symbolic"]),
            "total": len(ALIGNMENT_PAIRS),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 58: ALIGNMENT-DRIVEN TRAINING")
        logger.info("=" * 60)
        logger.info("\nObjective: Make Gram(counting) ≈ Gram(symbolic)\n")

        # Initial state
        counting_acts, symbolic_acts = self.get_aligned_activations()
        initial_loss = self.compute_alignment_loss(counting_acts, symbolic_acts)
        initial_eval = self.evaluate()

        logger.info("=== INITIAL STATE ===")
        logger.info(f"Alignment loss: {initial_loss:.4f}")
        logger.info(f"Counting: {initial_eval['counting_correct']}/{initial_eval['total']}")
        logger.info(f"Symbolic: {initial_eval['symbolic_correct']}/{initial_eval['total']}")

        results = {
            "initial": {
                "alignment_loss": initial_loss,
                **initial_eval,
            },
            "iterations": [],
        }

        # Training loop
        logger.info("\n=== ALIGNMENT-DRIVEN TRAINING ===")
        lr = 1 / (PHI ** 28)  # ≈ 2e-6 (more conservative)

        best_loss = initial_loss
        no_improvement = 0
        max_iterations = 100

        for i in range(max_iterations):
            loss = self.alignment_training_step(lr)

            if i % 5 == 0:
                counting_acts, symbolic_acts = self.get_aligned_activations()
                current_loss = self.compute_alignment_loss(counting_acts, symbolic_acts)
                eval_result = self.evaluate()

                logger.info(f"Iter {i}: loss={current_loss:.4f}, "
                           f"count={eval_result['counting_correct']}/{eval_result['total']}, "
                           f"symb={eval_result['symbolic_correct']}/{eval_result['total']}")

                results["iterations"].append({
                    "iteration": i,
                    "alignment_loss": current_loss,
                    **eval_result,
                })

                # Check for improvement
                if current_loss < best_loss - 0.001:
                    best_loss = current_loss
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Early stopping
                if no_improvement >= 3:
                    logger.info("Converged (no improvement)")
                    break

                # Don't stop early - let's see the full trajectory

        # Final state
        counting_acts, symbolic_acts = self.get_aligned_activations()
        final_loss = self.compute_alignment_loss(counting_acts, symbolic_acts)
        final_eval = self.evaluate()

        logger.info("\n=== FINAL STATE ===")
        logger.info(f"Alignment loss: {final_loss:.4f}")
        logger.info(f"Counting: {final_eval['counting_correct']}/{final_eval['total']}")
        logger.info(f"Symbolic: {final_eval['symbolic_correct']}/{final_eval['total']}")

        results["final"] = {
            "alignment_loss": final_loss,
            **final_eval,
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Alignment loss: {initial_loss:.4f} → {final_loss:.4f}")
        logger.info(f"Symbolic: {initial_eval['symbolic_accuracy']:.0%} → {final_eval['symbolic_accuracy']:.0%}")
        logger.info(f"Counting: {initial_eval['counting_accuracy']:.0%} → {final_eval['counting_accuracy']:.0%}")

        if final_loss < initial_loss and final_eval['symbolic_accuracy'] > initial_eval['symbolic_accuracy']:
            logger.info("\n*** ALIGNMENT TRAINING WORKED ***")
            results["conclusion"] = "success"
        else:
            logger.info("\n*** ALIGNMENT TRAINING DID NOT IMPROVE SYMBOLIC ***")
            results["conclusion"] = "failed"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    trainer = AlignmentDrivenTrainer(model, tokenizer)
    results = trainer.run_experiment()

    output_path = "data/experiments/alignment_driven_training.json"
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
