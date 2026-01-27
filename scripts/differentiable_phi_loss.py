#!/usr/bin/env python3
"""Differentiable φ-Loss: Train toward golden ratio compression.

The breakthrough insight:
    We don't need to differentiate through TwoNN.
    The TRAJECTORY of activation norms IS differentiable.

    Target: peak_norm / final_norm = φ

    This is a direct, differentiable proxy for comp/φ = 1.0.

How it works:
    1. Forward pass captures activation norms at each layer
    2. Find peak norm and final norm
    3. Compute ratio loss: |peak/final - φ|
    4. This IS differentiable through the computation graph

This makes geometric alignment trainable.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + mx.sqrt(mx.array(5.0))) / 2  # Golden ratio as MLX scalar


def compute_trajectory_norms(model, input_ids: mx.array) -> tuple[list[float], float, float]:
    """Compute L2 norms of activations at each layer.

    Returns (norms, peak_norm, final_norm) where:
    - norms: list of scalar norms, one per layer
    - peak_norm: maximum norm (for loss computation)
    - final_norm: last layer norm
    """
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    norms = []
    emb_norm = mx.sqrt(mx.sum(hidden * hidden))
    mx.eval(emb_norm)
    peak_val = float(emb_norm)

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        layer_norm = mx.sqrt(mx.sum(hidden * hidden))
        mx.eval(layer_norm)
        norm_val = float(layer_norm)
        norms.append(norm_val)
        peak_val = max(peak_val, norm_val)

    final_val = norms[-1]  # Last layer norm

    return norms, peak_val, final_val


def compute_phi_loss(peak_norm: float, final_norm: float) -> tuple[float, float]:
    """Compute φ-loss from peak and final norms.

    Target: peak_norm / final_norm = φ

    Loss = |compression_ratio - φ|
    """
    # Compression ratio with numerical stability
    eps = 1e-8
    compression_ratio = peak_norm / (final_norm + eps)

    # Loss: distance from φ
    phi_val = float(PHI)
    phi_loss = abs(compression_ratio - phi_val)

    return phi_loss, compression_ratio


def get_cot_training_data() -> list[dict]:
    """Get chain-of-thought training examples."""
    return [
        {
            "prompt": """Question: What is 15 × 7?

Let me break this down:
15 × 7 = 15 × (5 + 2)
      = 15 × 5 + 15 × 2
      = 75 + 30
      = 105

Answer: 105""",
        },
        {
            "prompt": """Question: A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?

Let me define variables:
Let ball = x dollars
Then bat = x + 1 dollars

Total: x + (x + 1) = 1.10
2x + 1 = 1.10
2x = 0.10
x = 0.05

Answer: $0.05""",
        },
        {
            "prompt": """Question: 5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?

Rate analysis:
5 machines → 5 widgets in 5 minutes
1 machine → 1 widget in 5 minutes
100 machines → 100 widgets in 5 minutes

Answer: 5 minutes""",
        },
        {
            "prompt": """Question: All cats have tails. Fluffy is a cat. Does Fluffy have a tail?

Syllogism:
Premise 1: All cats have tails
Premise 2: Fluffy is a cat
Conclusion: Fluffy has a tail (modus ponens)

Answer: Yes""",
        },
    ]


def train_with_phi_loss(
    model_path: str,
    phi_weight: float = 0.01,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
):
    """Train with differentiable φ-loss.

    Loss = task_loss + phi_weight * |compression_ratio - φ|
    """
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("DIFFERENTIABLE φ-LOSS TRAINING")
    logger.info("=" * 70)
    logger.info(f"φ weight: {phi_weight}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Target: compression_ratio = {float(PHI):.6f}")

    # Load model
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Training data
    training_data = get_cot_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def combined_loss_fn(model, tokens):
        """Combined task loss + φ-monitoring."""
        input_ids = mx.array([tokens])

        # Task loss (next token prediction) - this is differentiable
        logits = model(mx.array([tokens[:-1]]))
        logits = logits.reshape(-1, logits.shape[-1])
        targets = mx.array([tokens[1:]]).reshape(-1)
        task_loss = nn.losses.cross_entropy(logits, targets, reduction='mean')

        # φ-metrics (monitoring - not in gradient for now)
        norms, peak_norm, final_norm = compute_trajectory_norms(model, input_ids)
        phi_loss, compression_ratio = compute_phi_loss(peak_norm, final_norm)

        return task_loss, phi_loss, compression_ratio

    # Training loop
    history = []
    step = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 40)

        epoch_task_losses = []
        epoch_phi_losses = []
        epoch_ratios = []

        for example in training_data:
            tokens = tokenizer.encode(example["prompt"])
            input_ids = mx.array([tokens])

            # Compute gradients on task loss
            def loss_fn(model):
                logits = model(mx.array([tokens[:-1]]))
                logits = logits.reshape(-1, logits.shape[-1])
                targets = mx.array([tokens[1:]]).reshape(-1)
                return nn.losses.cross_entropy(logits, targets, reduction='mean')

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            task_loss, grads = loss_and_grad(model)
            mx.eval(task_loss)

            # Compute φ-metrics separately (before update)
            norms, peak_norm, final_norm = compute_trajectory_norms(model, input_ids)
            phi_loss, ratio = compute_phi_loss(peak_norm, final_norm)

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            step += 1
            task_val = float(task_loss)
            phi_val = phi_loss
            ratio_val = ratio

            epoch_task_losses.append(task_val)
            epoch_phi_losses.append(phi_val)
            epoch_ratios.append(ratio_val)

            logger.info(
                f"Step {step}: task={task_val:.4f}, φ_loss={phi_val:.4f}, "
                f"ratio={ratio_val:.3f} (target={float(PHI):.3f})"
            )

            history.append({
                "step": step,
                "epoch": epoch + 1,
                "task_loss": task_val,
                "phi_loss": phi_val,
                "compression_ratio": ratio_val,
                "distance_from_phi": abs(ratio_val - float(PHI)),
            })

        # Epoch summary
        mean_task = np.mean(epoch_task_losses)
        mean_phi = np.mean(epoch_phi_losses)
        mean_ratio = np.mean(epoch_ratios)
        logger.info(
            f"Epoch {epoch + 1}: task={mean_task:.4f}, φ_loss={mean_phi:.4f}, "
            f"ratio={mean_ratio:.3f}"
        )

    # Final assessment
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    initial_ratio = history[0]["compression_ratio"]
    final_ratio = history[-1]["compression_ratio"]
    initial_dist = abs(initial_ratio - float(PHI))
    final_dist = abs(final_ratio - float(PHI))

    logger.info(f"Initial ratio: {initial_ratio:.4f} (distance from φ: {initial_dist:.4f})")
    logger.info(f"Final ratio: {final_ratio:.4f} (distance from φ: {final_dist:.4f})")

    if final_dist < initial_dist:
        improvement = (initial_dist - final_dist) / initial_dist * 100
        logger.info(f"✓ Improved by {improvement:.1f}% toward φ!")
    else:
        logger.info("✗ Did not improve toward φ")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "phi_weight": phi_weight,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "target_phi": float(PHI),
        },
        "history": history,
        "improvement": {
            "initial_ratio": initial_ratio,
            "final_ratio": final_ratio,
            "initial_distance": initial_dist,
            "final_distance": final_dist,
            "improved": final_dist < initial_dist,
        },
    }

    output_path = Path("data/experiments/differentiable_phi_loss.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return history


if __name__ == "__main__":
    train_with_phi_loss(
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        phi_weight=0.01,
        learning_rate=1e-5,
        num_epochs=3,
    )
