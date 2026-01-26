#!/usr/bin/env python3
"""Train the model to correlate symbolic arithmetic with counting.

The model already knows counting (100%). We're just teaching it that
"2+1=" triggers the same circuit as "what comes after 2".

This shouldn't disturb anything - it's reinforcing the invariant
relationship that already exists, just extending it to symbolic notation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_training_data(path: str) -> List[Dict]:
    """Load the math equivalence training data."""
    with open(path) as f:
        return json.load(f)


def format_for_training(examples: List[Dict]) -> List[str]:
    """Format examples for causal LM training."""
    formatted = []
    for ex in examples:
        # Simple format: instruction + input → output
        if ex["instruction"] and ex["input"]:
            text = f"{ex['instruction']}\n{ex['input']}\n{ex['output']}"
        elif ex["instruction"]:
            text = f"{ex['instruction']}\n{ex['output']}"
        elif ex["input"]:
            text = f"{ex['input']}{ex['output']}"
        else:
            text = ex["output"]
        formatted.append(text)
    return formatted


class MathCorrelationTrainer:
    """Train model to correlate symbolic arithmetic with counting."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_loss(self, text: str):
        """Compute cross-entropy loss for a training example."""
        import mlx.core as mx
        import mlx.nn as nn

        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return None

        input_ids = mx.array([tokens[:-1]])
        target_ids = mx.array([tokens[1:]])

        logits = self.model(input_ids)
        mx.eval(logits)

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction='mean'
        )
        return loss

    def train_step(self, texts: List[str], learning_rate: float = 1e-5):
        """Single training step on a batch of texts."""
        import mlx.core as mx
        import mlx.optimizers as optim

        def loss_fn(model):
            total_loss = mx.array(0.0)
            count = 0
            for text in texts:
                tokens = self.tokenizer.encode(text)
                if len(tokens) < 2:
                    continue
                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])

                logits = model(input_ids)

                loss = mx.mean(
                    mx.sum(
                        -mx.take_along_axis(
                            mx.log(mx.softmax(logits, axis=-1) + 1e-10),
                            target_ids[:, :, None],
                            axis=-1
                        ).squeeze(-1),
                        axis=-1
                    )
                )
                total_loss = total_loss + loss
                count += 1

            return total_loss / max(count, 1)

        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        mx.eval(loss, grads)

        # Apply gradients
        optimizer = optim.SGD(learning_rate=learning_rate)
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        return float(loss.item())

    def evaluate_arithmetic(self) -> Dict:
        """Quick evaluation on arithmetic."""
        import mlx.core as mx

        tests = [
            ("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
            ("1+2=", "3"), ("2+2=", "4"), ("3+3=", "6"), ("5+5=", "10"),
            ("10-5=", "5"), ("7-3=", "4"),
        ]

        correct = 0
        results = []
        for prompt, expected in tests:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)

            next_token = int(mx.argmax(logits[0, -1, :]).item())
            output = self.tokenizer.decode([next_token]).strip()

            is_correct = expected in output or output == expected
            if is_correct:
                correct += 1
            results.append(f"{prompt}{output} ({'✓' if is_correct else '✗'})")

        return {
            "accuracy": correct / len(tests),
            "correct": correct,
            "total": len(tests),
            "results": results,
        }

    def evaluate_counting(self) -> Dict:
        """Verify counting still works."""
        import mlx.core as mx

        tests = [
            ("1, 2, 3, 4,", "5"),
            ("A, B, C,", "D"),
            ("Count: one, two, three,", "four"),
        ]

        correct = 0
        results = []
        for prompt, expected in tests:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Generate a few tokens
            for _ in range(3):
                logits = self.model(input_ids)
                mx.eval(logits)
                next_token = int(mx.argmax(logits[0, -1, :]).item())
                input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            output = self.tokenizer.decode(input_ids[0].tolist()[len(tokens):]).strip()

            is_correct = expected.lower() in output.lower()
            if is_correct:
                correct += 1
            results.append(f"'{prompt}' → '{output}' ({'✓' if is_correct else '✗'})")

        return {
            "accuracy": correct / len(tests),
            "results": results,
        }

    def train(self, training_data: List[str], epochs: int = 3, batch_size: int = 8,
              learning_rate: float = 1e-5):
        """Train the model."""
        logger.info(f"Training on {len(training_data)} examples")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        # Initial evaluation
        logger.info("\n=== BEFORE TRAINING ===")
        arith_before = self.evaluate_arithmetic()
        count_before = self.evaluate_counting()
        logger.info(f"Arithmetic: {arith_before['correct']}/{arith_before['total']} ({arith_before['accuracy']:.0%})")
        for r in arith_before['results']:
            logger.info(f"  {r}")
        logger.info(f"Counting: {count_before['accuracy']:.0%}")
        for r in count_before['results']:
            logger.info(f"  {r}")

        # Training loop
        for epoch in range(epochs):
            logger.info(f"\n=== EPOCH {epoch + 1}/{epochs} ===")

            # Shuffle data
            np.random.shuffle(training_data)

            total_loss = 0
            n_batches = 0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                loss = self.train_step(batch, learning_rate)
                total_loss += loss
                n_batches += 1

                if n_batches % 50 == 0:
                    logger.info(f"  Batch {n_batches}: loss = {loss:.4f}")

            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            # Evaluate after epoch
            arith = self.evaluate_arithmetic()
            count = self.evaluate_counting()
            logger.info(f"Arithmetic: {arith['correct']}/{arith['total']} ({arith['accuracy']:.0%})")
            logger.info(f"Counting: {count['accuracy']:.0%}")

        # Final evaluation
        logger.info("\n=== AFTER TRAINING ===")
        arith_after = self.evaluate_arithmetic()
        count_after = self.evaluate_counting()
        logger.info(f"Arithmetic: {arith_after['correct']}/{arith_after['total']} ({arith_after['accuracy']:.0%})")
        for r in arith_after['results']:
            logger.info(f"  {r}")
        logger.info(f"Counting: {count_after['accuracy']:.0%}")
        for r in count_after['results']:
            logger.info(f"  {r}")

        return {
            "before": {"arithmetic": arith_before, "counting": count_before},
            "after": {"arithmetic": arith_after, "counting": count_after},
            "improvement": arith_after["accuracy"] - arith_before["accuracy"],
            "counting_preserved": count_after["accuracy"] >= count_before["accuracy"] - 0.1,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("Loading training data...")
    data_path = "data/training/math_equivalence_training.json"
    raw_data = load_training_data(data_path)
    training_data = format_for_training(raw_data)
    logger.info(f"Loaded {len(training_data)} training examples")

    # Conservative training - stop before overfitting
    training_subset = training_data[:500]
    logger.info(f"Using {len(training_subset)} examples")

    trainer = MathCorrelationTrainer(model, tokenizer)
    results = trainer.train(
        training_subset,
        epochs=1,  # Just 1 epoch - that's where we saw best results
        batch_size=4,
        learning_rate=5e-6,
    )

    # Save results
    output_path = "data/experiments/math_correlation_training.json"
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

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    if results["improvement"] > 0:
        logger.info(f"✓ Arithmetic improved by {results['improvement']:.0%}")
    else:
        logger.info(f"✗ Arithmetic did not improve")

    if results["counting_preserved"]:
        logger.info(f"✓ Counting ability preserved")
    else:
        logger.info(f"✗ Counting ability degraded")


if __name__ == "__main__":
    main()
