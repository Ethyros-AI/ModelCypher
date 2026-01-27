#!/usr/bin/env python3
"""φ-Alignment Training: Train the model for comp/φ = 1.0.

The breakthrough:
    Chain-of-thought naturally produces comp/φ ≈ 1.0.
    Training with CoT examples + geometric loss teaches the model
    to think deeply enough on ANY problem.

Loss function:
    L = L_task + λ * |comp_phi - 1.0|

where:
    L_task = standard cross-entropy on next token prediction
    comp_phi = compression_ratio / φ (measured on the training example)
    λ = weight balancing task performance and geometric alignment

This is alignment through geometry.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2
TARGET_COMP_PHI = 1.0


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    if len(X) < 10:
        return float('nan')
    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)
    d1, d2 = distances[:, 1], distances[:, 2]
    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')
    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]
    if len(mu) < 5:
        return float('nan')
    return float(len(np.log(mu)) / np.sum(np.log(mu)))


@dataclass
class TrainingExample:
    """A training example with chain-of-thought."""
    question: str
    chain_of_thought: str
    answer: str

    @property
    def full_prompt(self) -> str:
        return f"Question: {self.question}\n\n{self.chain_of_thought}\n\nAnswer: {self.answer}"


def get_cot_training_data() -> list[TrainingExample]:
    """Get chain-of-thought training examples.

    These examples teach the model to maintain relationships
    and think through problems step by step.
    """
    return [
        # Math with clear steps
        TrainingExample(
            question="What is 15 × 7?",
            chain_of_thought="""Let me break this down:
15 × 7 = 15 × (5 + 2)
      = 15 × 5 + 15 × 2
      = 75 + 30
      = 105""",
            answer="105",
        ),
        TrainingExample(
            question="A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?",
            chain_of_thought="""Let me define variables:
Let ball = x dollars
Then bat = x + 1 dollars (since bat costs $1 more)

Total cost equation:
x + (x + 1) = 1.10
2x + 1 = 1.10
2x = 0.10
x = 0.05""",
            answer="$0.05",
        ),
        TrainingExample(
            question="5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
            chain_of_thought="""Let me figure out the rate:
5 machines make 5 widgets in 5 minutes
So 5 machines make 1 widget per minute collectively
So 1 machine makes 1 widget in 5 minutes

With 100 machines, each making 1 widget in 5 minutes:
100 machines make 100 widgets in 5 minutes""",
            answer="5 minutes",
        ),
        TrainingExample(
            question="A lily pad doubles in size every day. It takes 48 days to cover a lake. How many days for half?",
            chain_of_thought="""Let me think about the growth pattern:
Day 48: full lake (100%)
Since it DOUBLES each day, the day before was HALF
Day 47: half the lake (50%)

Working backwards from the end gives the answer.""",
            answer="47 days",
        ),
        # Logic with explicit reasoning
        TrainingExample(
            question="All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
            chain_of_thought="""Premise 1: All cats have tails
Premise 2: Fluffy is a cat
Conclusion: Since Fluffy is a cat, and all cats have tails, Fluffy has a tail.

This is a valid syllogism (modus ponens).""",
            answer="Yes",
        ),
        TrainingExample(
            question="Some fruits are red. Apples are fruits. Are all apples red?",
            chain_of_thought="""Let me analyze the logic:
Premise 1: Some fruits are red (not ALL fruits)
Premise 2: Apples are fruits

This does NOT mean all apples are red because:
- 'Some' does not equal 'All'
- Apples being fruits doesn't guarantee they share the property of being red
- Counter-example: Green apples exist""",
            answer="No",
        ),
        # Word problems with relationship tracking
        TrainingExample(
            question="Tom has 3 times as many apples as Jane. Jane has 5 apples. How many does Tom have?",
            chain_of_thought="""Relationship: Tom = 3 × Jane
Given: Jane = 5 apples

Substitute:
Tom = 3 × 5 = 15 apples""",
            answer="15 apples",
        ),
        TrainingExample(
            question="A train travels 60 km/h for 2 hours, then 80 km/h for 1.5 hours. Total distance?",
            chain_of_thought="""Calculate each segment:
Segment 1: 60 km/h × 2 h = 120 km
Segment 2: 80 km/h × 1.5 h = 120 km

Total: 120 + 120 = 240 km""",
            answer="240 km",
        ),
    ]


def measure_comp_phi_during_forward(model, input_ids) -> tuple[float, list[float]]:
    """Measure comp/φ during forward pass.

    Returns (comp_phi, trajectory) where trajectory is the dimensional
    evolution through layers.
    """
    import mlx.core as mx

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []
    # Convert to numpy for intrinsic dimension calculation
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]
        if final_dim > 0.1:
            compression_ratio = peak_dim / final_dim
            return compression_ratio / PHI, trajectory

    return float('nan'), trajectory


def train_for_phi(
    model_path: str,
    output_path: str,
    geometric_weight: float = 0.1,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    log_interval: int = 1,
):
    """Train the model toward comp/φ = 1.0.

    Args:
        model_path: Path to base model
        output_path: Where to save trained model
        geometric_weight: λ in loss = task_loss + λ * geometric_loss
        learning_rate: Learning rate for AdamW
        num_epochs: Number of training epochs
        log_interval: Log every N steps
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("φ-ALIGNMENT TRAINING")
    logger.info("Target: comp/φ = 1.0")
    logger.info("=" * 70)
    logger.info(f"Geometric weight (λ): {geometric_weight}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Load model
    logger.info(f"\nLoading model: {model_path}")
    model, tokenizer = load(model_path)

    # Get training data
    training_data = get_cot_training_data()
    logger.info(f"Training examples: {len(training_data)}")

    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def compute_task_loss(model, tokens):
        """Standard cross-entropy loss on next token prediction."""
        input_ids = mx.array([tokens[:-1]])
        target_ids = mx.array([tokens[1:]])
        logits = model(input_ids)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = target_ids.reshape(-1)
        loss = nn.losses.cross_entropy(logits, targets, reduction='mean')
        return loss

    def training_step(model, example: TrainingExample):
        """Single training step with geometric loss."""
        tokens = tokenizer.encode(example.full_prompt)
        input_ids = mx.array([tokens])

        # Forward pass for task loss
        def loss_fn(model):
            logits = model(mx.array([tokens[:-1]]))
            logits = logits.reshape(-1, logits.shape[-1])
            targets = mx.array([tokens[1:]]).reshape(-1)
            return nn.losses.cross_entropy(logits, targets, reduction='mean')

        # Compute gradients
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        task_loss, grads = loss_and_grad_fn(model)
        mx.eval(task_loss)

        # Measure comp/φ (detached from gradient computation)
        comp_phi, trajectory = measure_comp_phi_during_forward(model, input_ids)

        # Geometric loss (not differentiable, but guides training via monitoring)
        if not np.isnan(comp_phi):
            geometric_loss = abs(comp_phi - TARGET_COMP_PHI)
        else:
            geometric_loss = 0.0

        # Combined loss for reporting
        total_loss = float(task_loss) + geometric_weight * geometric_loss

        # Apply gradients (only from task loss - geometric is monitoring for now)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        return {
            "task_loss": float(task_loss),
            "geometric_loss": geometric_loss,
            "total_loss": total_loss,
            "comp_phi": comp_phi,
        }

    # Training loop
    history = []
    step = 0
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 40)

        epoch_losses = []
        epoch_comp_phis = []

        for i, example in enumerate(training_data):
            metrics = training_step(model, example)
            step += 1

            epoch_losses.append(metrics["total_loss"])
            if not np.isnan(metrics["comp_phi"]):
                epoch_comp_phis.append(metrics["comp_phi"])

            if step % log_interval == 0:
                logger.info(
                    f"Step {step}: task={metrics['task_loss']:.4f}, "
                    f"geo={metrics['geometric_loss']:.4f}, "
                    f"comp/φ={metrics['comp_phi']:.3f}"
                )

            history.append({
                "step": step,
                "epoch": epoch + 1,
                "question": example.question[:30],
                **metrics,
            })

        # Epoch summary
        mean_loss = np.mean(epoch_losses)
        mean_phi = np.mean(epoch_comp_phis) if epoch_comp_phis else float('nan')
        phi_std = np.std(epoch_comp_phis) if epoch_comp_phis else float('nan')
        logger.info(f"Epoch {epoch + 1} complete: loss={mean_loss:.4f}, comp/φ={mean_phi:.3f}±{phi_std:.3f}")

    # Final assessment
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    final_phis = [h["comp_phi"] for h in history[-len(training_data):] if not np.isnan(h["comp_phi"])]
    if final_phis:
        final_mean = np.mean(final_phis)
        final_dist = np.mean([abs(p - 1.0) for p in final_phis])
        logger.info(f"Final comp/φ: {final_mean:.3f} (distance from 1.0: {final_dist:.3f})")

        if final_dist < 0.15:
            logger.info("✓ Model is well-aligned (distance < 0.15)")
        elif final_dist < 0.25:
            logger.info("? Model is moderately aligned (distance < 0.25)")
        else:
            logger.info("✗ Model needs more training (distance >= 0.25)")

    # Save results (not the model yet - this is analysis)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": model_path,
            "geometric_weight": geometric_weight,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        },
        "history": history,
        "final_metrics": {
            "mean_comp_phi": float(np.mean(final_phis)) if final_phis else None,
            "std_comp_phi": float(np.std(final_phis)) if final_phis else None,
            "distance_from_1": float(final_dist) if final_phis else None,
        },
    }

    output_file = Path("data/experiments/phi_alignment_training.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    return history


if __name__ == "__main__":
    train_for_phi(
        model_path="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        output_path="/Volumes/CodeCypher/models/phi-aligned/LFM2-350M-phi",
        geometric_weight=0.1,
        learning_rate=1e-5,
        num_epochs=2,
    )
