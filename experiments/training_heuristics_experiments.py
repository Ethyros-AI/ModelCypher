#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiments for geometry-derived training heuristics.
# Compares industry standard approaches vs geometric alternatives.
#
# Usage:
#   poetry run python experiments/training_heuristics_experiments.py --experiment gradient_clipping
#   poetry run python experiments/training_heuristics_experiments.py --experiment all

"""
Training Heuristics Experiments

Tests five training heuristics:
1. Gradient clipping: none vs global (1.0) vs spectral (σ_max)
2. Warmup: linear vs none vs adaptive (BB stability)
3. LR schedules: BB + cosine vs BB + none
4. Batch size: fixed vs gradient noise derived
5. Dropout: fixed vs effective rank derived
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    name: str
    model_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    batch_size: int = 32
    sequence_length: int = 64
    num_steps: int = 500
    learning_rate: float = 0.001
    warmup_steps: int = 50
    weight_decay: float = 0.0
    gradient_clip_mode: str = "none"
    global_clip_value: float = 1.0
    use_cosine_decay: bool = False
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a training experiment."""
    config: ExperimentConfig
    loss_history: list = field(default_factory=list)
    lr_history: list = field(default_factory=list)
    gradient_norms: dict = field(default_factory=dict)
    bb_stability: list = field(default_factory=list)
    final_loss: float = 0.0
    convergence_step: int = -1  # Step where loss < threshold
    total_time: float = 0.0


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for experiments."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ExperimentModel(nn.Module):
    """Small transformer model for experiments."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, vocab_size: int = 1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = [SimpleTransformerBlock(dim) for _ in range(num_layers)]
        self.output = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


def generate_synthetic_data(batch_size: int, seq_length: int, vocab_size: int = 1000):
    """Generate synthetic language modeling data."""
    # Input: random tokens
    x = mx.random.randint(0, vocab_size, shape=(batch_size, seq_length))
    # Target: shifted by 1 (next token prediction)
    y = mx.random.randint(0, vocab_size, shape=(batch_size, seq_length))
    return x, y


def run_training_experiment(
    config: ExperimentConfig,
    model: nn.Module,
    optimizer,
    data_fn: Callable,
) -> ExperimentResult:
    """Run a single training experiment."""
    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer

    result = ExperimentResult(config=config)
    mx.random.seed(config.seed)

    # Initialize optimizer if geometric
    if isinstance(optimizer, GeometricOptimizer):
        optimizer.init_from_model(model)
        geometric_base_lr = optimizer.base_lr
        logger.info(f"Geometric base LR: {geometric_base_lr:.6f}")
    else:
        geometric_base_lr = config.learning_rate

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    start_time = time.time()
    convergence_threshold = 3.0  # Loss threshold for "converged"

    for step in range(config.num_steps):
        x, y = data_fn()
        mx.eval(x, y)

        # Compute loss and gradients
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss, grads)
        loss_val = float(loss.item())

        # Learning rate scheduling
        if isinstance(optimizer, GeometricOptimizer):
            if config.warmup_steps > 0 and step < config.warmup_steps:
                # Linear warmup
                warmup_factor = (step + 1) / config.warmup_steps
                optimizer.base_lr = geometric_base_lr * warmup_factor
            else:
                optimizer.base_lr = geometric_base_lr

            # Optional cosine decay
            if config.use_cosine_decay and step >= config.warmup_steps:
                decay_steps = config.num_steps - config.warmup_steps
                progress = (step - config.warmup_steps) / decay_steps
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                optimizer.base_lr = geometric_base_lr * cosine_factor

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Record metrics
        result.loss_history.append(loss_val)

        if isinstance(optimizer, GeometricOptimizer):
            result.lr_history.append(optimizer.base_lr)
            result.bb_stability.append(optimizer.get_bb_stability())

        # Check convergence
        if loss_val < convergence_threshold and result.convergence_step < 0:
            result.convergence_step = step

        # Logging
        if step % 50 == 0 or step == config.num_steps - 1:
            lr = optimizer.base_lr if isinstance(optimizer, GeometricOptimizer) else config.learning_rate
            logger.info(f"Step {step}: loss={loss_val:.4f}, lr={lr:.6f}")

    result.total_time = time.time() - start_time
    result.final_loss = result.loss_history[-1]

    # Get gradient statistics if available
    if isinstance(optimizer, GeometricOptimizer):
        result.gradient_norms = optimizer.get_gradient_stats()

    return result


def run_gradient_clipping_experiment(output_dir: Path):
    """Experiment 1: Gradient clipping comparison."""
    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer

    logger.info("=" * 60)
    logger.info("EXPERIMENT: Gradient Clipping Comparison")
    logger.info("=" * 60)

    results = {}

    for clip_mode in ["none", "global", "spectral"]:
        logger.info(f"\n--- Running with gradient_clip_mode={clip_mode} ---")

        config = ExperimentConfig(
            name=f"gradient_clip_{clip_mode}",
            gradient_clip_mode=clip_mode,
            global_clip_value=1.0,
            num_steps=300,
            warmup_steps=30,
        )

        model = ExperimentModel(
            dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(
            base_decay=config.weight_decay,
            gradient_clip_mode=clip_mode,
            global_clip_value=config.global_clip_value,
        )

        def data_fn():
            return generate_synthetic_data(config.batch_size, config.sequence_length)

        result = run_training_experiment(config, model, optimizer, data_fn)
        results[clip_mode] = result

        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Convergence step: {result.convergence_step}")
        logger.info(f"Time: {result.total_time:.2f}s")

        if result.gradient_norms:
            logger.info("Gradient clipping ratios per layer:")
            for key, stats in result.gradient_norms.items():
                logger.info(f"  {key}: clip_ratio={stats['clip_ratio']:.2%}, mean={stats['mean_norm']:.4f}")

    # Save results
    save_experiment_results(results, output_dir / "gradient_clipping_results.json")
    return results


def run_warmup_experiment(output_dir: Path):
    """Experiment 2: Warmup comparison."""
    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer

    logger.info("=" * 60)
    logger.info("EXPERIMENT: Warmup Comparison")
    logger.info("=" * 60)

    results = {}

    warmup_configs = [
        ("linear_50", 50),
        ("linear_100", 100),
        ("none", 0),
    ]

    for name, warmup_steps in warmup_configs:
        logger.info(f"\n--- Running with warmup={name} ({warmup_steps} steps) ---")

        config = ExperimentConfig(
            name=f"warmup_{name}",
            warmup_steps=warmup_steps,
            num_steps=300,
        )

        model = ExperimentModel(
            dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=config.weight_decay)

        def data_fn():
            return generate_synthetic_data(config.batch_size, config.sequence_length)

        result = run_training_experiment(config, model, optimizer, data_fn)
        results[name] = result

        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Early loss (step 10): {result.loss_history[10] if len(result.loss_history) > 10 else 'N/A'}")
        logger.info(f"BB stable at step: {next((i for i, s in enumerate(result.bb_stability) if s < 1e-4), -1)}")

    save_experiment_results(results, output_dir / "warmup_results.json")
    return results


def run_lr_schedule_experiment(output_dir: Path):
    """Experiment 3: LR schedule comparison."""
    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer

    logger.info("=" * 60)
    logger.info("EXPERIMENT: LR Schedule Comparison")
    logger.info("=" * 60)

    results = {}

    schedule_configs = [
        ("bb_only", False),
        ("bb_plus_cosine", True),
    ]

    for name, use_cosine in schedule_configs:
        logger.info(f"\n--- Running with schedule={name} ---")

        config = ExperimentConfig(
            name=f"schedule_{name}",
            use_cosine_decay=use_cosine,
            num_steps=400,
            warmup_steps=40,
        )

        model = ExperimentModel(
            dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=config.weight_decay)

        def data_fn():
            return generate_synthetic_data(config.batch_size, config.sequence_length)

        result = run_training_experiment(config, model, optimizer, data_fn)
        results[name] = result

        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"LR at end: {result.lr_history[-1] if result.lr_history else 'N/A'}")

    save_experiment_results(results, output_dir / "lr_schedule_results.json")
    return results


def run_batch_size_experiment(output_dir: Path):
    """Experiment 4: Batch size and gradient noise analysis."""
    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer

    logger.info("=" * 60)
    logger.info("EXPERIMENT: Batch Size Analysis")
    logger.info("=" * 60)

    results = {}

    batch_sizes = [8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        logger.info(f"\n--- Running with batch_size={batch_size} ---")

        config = ExperimentConfig(
            name=f"batch_{batch_size}",
            batch_size=batch_size,
            num_steps=200,
            warmup_steps=20,
        )

        model = ExperimentModel(
            dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=config.weight_decay)

        def data_fn():
            return generate_synthetic_data(batch_size, config.sequence_length)

        result = run_training_experiment(config, model, optimizer, data_fn)
        results[f"batch_{batch_size}"] = result

        # Estimate gradient noise scale
        noise_scale = estimate_gradient_noise_scale(model, batch_size, config.sequence_length)

        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Gradient noise scale: {noise_scale:.4f}")
        logger.info(f"Suggested critical batch: ~{noise_scale:.0f}")

    save_experiment_results(results, output_dir / "batch_size_results.json")
    return results


def estimate_gradient_noise_scale(model, batch_size: int, seq_length: int, num_samples: int = 8) -> float:
    """Estimate gradient noise scale B_simple = Var(g) / ||E[g]||²."""

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")

    # Collect gradients from multiple mini-batches
    grad_list = []
    for _ in range(num_samples):
        x, y = generate_synthetic_data(batch_size, seq_length)
        mx.eval(x, y)
        _, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        mx.eval(grads)

        # Flatten gradients to vector
        flat_grads = tree_flatten(grads)
        grad_vec = []
        for _, g in flat_grads:
            if g is not None:
                grad_vec.append(np.array(g).flatten())
        grad_vec = np.concatenate(grad_vec)
        grad_list.append(grad_vec)

    grad_array = np.array(grad_list)
    mean_grad = np.mean(grad_array, axis=0)
    variance = np.mean(np.sum((grad_array - mean_grad) ** 2, axis=1))
    mean_norm_sq = np.sum(mean_grad ** 2)

    if mean_norm_sq < 1e-10:
        return float('inf')

    return variance / mean_norm_sq


def run_dropout_experiment(output_dir: Path):
    """Experiment 5: Dropout and effective rank analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Dropout Analysis")
    logger.info("=" * 60)

    # Note: Full dropout experiment requires activation hooks
    # This is a placeholder showing the analysis approach

    logger.info("Dropout experiment requires activation effective rank hooks.")
    logger.info("See docs/research/training_heuristics_analysis.md for implementation details.")

    return {}


def save_experiment_results(results: dict, path: Path):
    """Save experiment results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for name, result in results.items():
        if isinstance(result, ExperimentResult):
            serializable[name] = {
                "config": {
                    "name": result.config.name,
                    "num_steps": result.config.num_steps,
                    "warmup_steps": result.config.warmup_steps,
                    "batch_size": result.config.batch_size,
                    "gradient_clip_mode": result.config.gradient_clip_mode,
                    "use_cosine_decay": result.config.use_cosine_decay,
                },
                "final_loss": result.final_loss,
                "convergence_step": result.convergence_step,
                "total_time": result.total_time,
                "loss_history": result.loss_history[:50] + ["...truncated..."] + result.loss_history[-10:] if len(result.loss_history) > 60 else result.loss_history,
                "gradient_norms": result.gradient_norms,
            }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Training heuristics experiments")
    parser.add_argument(
        "--experiment",
        choices=["gradient_clipping", "warmup", "lr_schedule", "batch_size", "dropout", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "gradient_clipping": run_gradient_clipping_experiment,
        "warmup": run_warmup_experiment,
        "lr_schedule": run_lr_schedule_experiment,
        "batch_size": run_batch_size_experiment,
        "dropout": run_dropout_experiment,
    }

    if args.experiment == "all":
        for name, func in experiments.items():
            try:
                func(args.output_dir)
            except Exception as e:
                logger.error(f"Experiment {name} failed: {e}")
    else:
        experiments[args.experiment](args.output_dir)


if __name__ == "__main__":
    main()
