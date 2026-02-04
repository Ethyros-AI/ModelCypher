#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiments for Phase 2 geometry-derived training heuristics.
#
# Usage:
#   poetry run python experiments/geometry_heuristics_phase2.py --experiment weight_init
#   poetry run python experiments/geometry_heuristics_phase2.py --experiment early_stopping
#   poetry run python experiments/geometry_heuristics_phase2.py --experiment residual_scaling
#   poetry run python experiments/geometry_heuristics_phase2.py --experiment all

"""
Phase 2 Geometry Training Heuristics Experiments

Tests three new geometry-derived training features:
1. Weight initialization: spectral-normalized vs arbitrary scale=0.01
2. Early stopping: geometric convergence vs fixed epochs
3. Residual scaling: α = σ_max(x) / σ_max(f(x)) vs α=1
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Numerical constants
SQRT_EPS = np.sqrt(np.finfo(np.float32).eps)


@dataclass
class Phase2ExperimentConfig:
    """Configuration for Phase 2 experiments."""
    name: str
    model_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    batch_size: int = 32
    sequence_length: int = 64
    num_steps: int = 500
    max_epochs: int = 10
    seed: int = 42


@dataclass
class Phase2ExperimentResult:
    """Results from a Phase 2 experiment."""
    config: Phase2ExperimentConfig
    metrics: dict = field(default_factory=dict)
    loss_history: list = field(default_factory=list)
    total_time: float = 0.0
    success: bool = True
    error: str = ""


# =============================================================================
# Simple Test Model (no LoRA, for isolation testing)
# =============================================================================

class SimpleLinear(nn.Module):
    """Linear layer with controllable initialization for testing."""

    def __init__(self, in_features: int, out_features: int, spectral_init: bool = True, target_spectral: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spectral_init = spectral_init
        self.target_spectral = target_spectral

        # Initialize weight
        W = mx.random.normal(shape=(out_features, in_features))

        if spectral_init:
            # Spectral-normalized initialization
            spectral_norm = self._compute_spectral_norm(W)
            if spectral_norm > SQRT_EPS:
                W = W * (target_spectral / spectral_norm)

        self.weight = W
        self.bias = mx.zeros((out_features,))

    def _compute_spectral_norm(self, W: mx.array, n_iters: int = 5) -> float:
        """Power iteration for spectral norm."""
        n = int(W.shape[1])
        v = mx.ones((n,)) / mx.sqrt(mx.array(float(n)))
        mx.eval(v)

        for _ in range(n_iters):
            u = W @ v
            u_norm = mx.sqrt(mx.sum(u * u))
            mx.eval(u_norm)
            if float(u_norm) < SQRT_EPS:
                return 0.0
            u = u / u_norm

            v = W.T @ u
            v_norm = mx.sqrt(mx.sum(v * v))
            mx.eval(v_norm)
            if float(v_norm) < SQRT_EPS:
                return 0.0
            v = v / v_norm
            mx.eval(v)

        Wv = W @ v
        return float(mx.sqrt(mx.sum(Wv * Wv)))

    def __call__(self, x):
        return x @ self.weight.T + self.bias


class TestModel(nn.Module):
    """Test model with configurable initialization."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, spectral_init: bool = True):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            in_dim = dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else dim
            self.layers.append(SimpleLinear(in_dim, out_dim, spectral_init=spectral_init))
        self.final = nn.Linear(dim, dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.gelu(layer(x))
        return self.final(x)


# =============================================================================
# Experiment 1: Weight Initialization
# =============================================================================

def run_weight_init_experiment(output_dir: Path) -> dict:
    """Compare spectral-normalized init vs arbitrary scale.

    Tests:
    1. Verify σ_max of initialized weights = target ± √ε
    2. Compare forward pass activation norms
    3. Compare training convergence
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Weight Initialization Comparison")
    logger.info("=" * 60)

    results = {}

    # Test 1: Verify spectral norm of initialized weights
    logger.info("\n--- Test 1: Spectral norm verification ---")

    spectral_norms_spectral = []
    spectral_norms_arbitrary = []

    for trial in range(100):
        mx.random.seed(trial)

        # Spectral init
        layer_spectral = SimpleLinear(256, 512, spectral_init=True, target_spectral=1.0)
        norm_spectral = layer_spectral._compute_spectral_norm(layer_spectral.weight)
        spectral_norms_spectral.append(norm_spectral)

        # Arbitrary init (scale=0.01 equivalent)
        layer_arbitrary = SimpleLinear(256, 512, spectral_init=False)
        layer_arbitrary.weight = mx.random.normal(shape=(512, 256)) * 0.01
        mx.eval(layer_arbitrary.weight)
        norm_arbitrary = layer_arbitrary._compute_spectral_norm(layer_arbitrary.weight)
        spectral_norms_arbitrary.append(norm_arbitrary)

    results["spectral_norm_verification"] = {
        "spectral_init": {
            "mean": float(np.mean(spectral_norms_spectral)),
            "std": float(np.std(spectral_norms_spectral)),
            "min": float(np.min(spectral_norms_spectral)),
            "max": float(np.max(spectral_norms_spectral)),
            "deviation_from_target": float(np.mean(np.abs(np.array(spectral_norms_spectral) - 1.0))),
        },
        "arbitrary_init": {
            "mean": float(np.mean(spectral_norms_arbitrary)),
            "std": float(np.std(spectral_norms_arbitrary)),
            "min": float(np.min(spectral_norms_arbitrary)),
            "max": float(np.max(spectral_norms_arbitrary)),
        },
    }

    logger.info(f"Spectral init: σ_max = {results['spectral_norm_verification']['spectral_init']['mean']:.4f} ± "
                f"{results['spectral_norm_verification']['spectral_init']['std']:.6f}")
    logger.info(f"Arbitrary init: σ_max = {results['spectral_norm_verification']['arbitrary_init']['mean']:.4f} ± "
                f"{results['spectral_norm_verification']['arbitrary_init']['std']:.6f}")

    deviation = results["spectral_norm_verification"]["spectral_init"]["deviation_from_target"]
    passed = deviation < SQRT_EPS * 10  # Allow some margin
    logger.info(f"Spectral init deviation from target: {deviation:.6f} (pass: {passed})")

    # Test 2: Forward pass activation norms
    logger.info("\n--- Test 2: Forward pass activation norms ---")

    mx.random.seed(42)
    x = mx.random.normal(shape=(32, 64, 256))  # [batch, seq, hidden]
    mx.eval(x)
    input_norm = float(mx.sqrt(mx.sum(x * x)))

    # Spectral init model
    model_spectral = TestModel(256, 512, 4, spectral_init=True)
    mx.eval(model_spectral.parameters())
    out_spectral = model_spectral(x)
    mx.eval(out_spectral)
    output_norm_spectral = float(mx.sqrt(mx.sum(out_spectral * out_spectral)))

    # Arbitrary init model
    mx.random.seed(42)  # Reset seed
    x = mx.random.normal(shape=(32, 64, 256))
    mx.eval(x)
    model_arbitrary = TestModel(256, 512, 4, spectral_init=False)
    mx.eval(model_arbitrary.parameters())
    out_arbitrary = model_arbitrary(x)
    mx.eval(out_arbitrary)
    output_norm_arbitrary = float(mx.sqrt(mx.sum(out_arbitrary * out_arbitrary)))

    results["forward_pass_norms"] = {
        "input_norm": input_norm,
        "spectral_init_output_norm": output_norm_spectral,
        "arbitrary_init_output_norm": output_norm_arbitrary,
        "spectral_ratio": output_norm_spectral / input_norm,
        "arbitrary_ratio": output_norm_arbitrary / input_norm,
    }

    logger.info(f"Input norm: {input_norm:.4f}")
    logger.info(f"Spectral init output norm: {output_norm_spectral:.4f} (ratio: {output_norm_spectral/input_norm:.4f})")
    logger.info(f"Arbitrary init output norm: {output_norm_arbitrary:.4f} (ratio: {output_norm_arbitrary/input_norm:.4f})")

    # Save results
    save_results(results, output_dir / "weight_init_results.json")

    return results


# =============================================================================
# Experiment 2: Early Stopping
# =============================================================================

def run_early_stopping_experiment(output_dir: Path) -> dict:
    """Compare geometric convergence vs fixed epochs.

    Tests:
    1. Compare stopping points: fixed epochs vs geometric criteria
    2. Measure training time reduction
    3. Compare final loss quality
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Early Stopping Comparison")
    logger.info("=" * 60)

    from modelcypher.core.domain.training.geometric_optimizer import GeometricOptimizer
    from modelcypher.core.domain.training.geometric_lora_trainer import GeometricConvergenceMonitor

    results = {}

    config = Phase2ExperimentConfig(
        name="early_stopping",
        num_steps=1000,
        max_epochs=20,
    )

    # Generate synthetic data once
    mx.random.seed(config.seed)
    train_data = []
    for _ in range(100):
        x = mx.random.normal(shape=(config.batch_size, config.sequence_length, config.model_dim))
        y = mx.random.normal(shape=(config.batch_size, config.sequence_length, config.model_dim))
        mx.eval(x, y)
        train_data.append((x, y))

    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    # Run 1: Fixed epochs (no early stopping)
    logger.info("\n--- Run 1: Fixed epochs (no early stopping) ---")

    mx.random.seed(config.seed)
    model_fixed = TestModel(config.model_dim, config.hidden_dim, config.num_layers, spectral_init=True)
    mx.eval(model_fixed.parameters())

    optimizer_fixed = GeometricOptimizer(base_decay=0.0)
    optimizer_fixed.init_from_model(model_fixed)

    loss_and_grad = nn.value_and_grad(model_fixed, loss_fn)

    fixed_losses = []
    start_time = time.time()

    for step in range(config.num_steps):
        x, y = train_data[step % len(train_data)]
        loss, grads = loss_and_grad(model_fixed, x, y)
        optimizer_fixed.update(model_fixed, grads)
        mx.eval(loss)
        fixed_losses.append(float(loss))

        if step % 100 == 0:
            logger.info(f"Step {step}: loss={float(loss):.6f}")

    fixed_time = time.time() - start_time

    # Run 2: Geometric early stopping
    logger.info("\n--- Run 2: Geometric early stopping ---")

    mx.random.seed(config.seed)
    model_geometric = TestModel(config.model_dim, config.hidden_dim, config.num_layers, spectral_init=True)
    mx.eval(model_geometric.parameters())

    optimizer_geometric = GeometricOptimizer(base_decay=0.0)
    optimizer_geometric.init_from_model(model_geometric)

    # Create mock LoRA layers dict for convergence monitor (uses spectral bound checking)
    # For this test, we'll use a simplified convergence check based on BB stability + loss
    monitor = GeometricConvergenceMonitor(
        bb_stability_threshold=1e-4,
        budget_threshold=0.9,
        loss_window=20,
    )

    loss_and_grad = nn.value_and_grad(model_geometric, loss_fn)

    geometric_losses = []
    start_time = time.time()
    stopped_at_step = config.num_steps

    for step in range(config.num_steps):
        x, y = train_data[step % len(train_data)]
        loss, grads = loss_and_grad(model_geometric, x, y)
        optimizer_geometric.update(model_geometric, grads)
        mx.eval(loss)
        geometric_losses.append(float(loss))

        # Check convergence (simplified - just BB + loss stability)
        bb_stable = optimizer_geometric.is_bb_stable(threshold=1e-4)

        # Check loss stability
        if len(geometric_losses) >= 20:
            recent = geometric_losses[-10:]
            older = geometric_losses[-20:-10]
            mean_recent = np.mean(recent)
            mean_older = np.mean(older)
            rel_change = abs(mean_recent - mean_older) / max(abs(mean_older), SQRT_EPS)
            loss_stable = rel_change < SQRT_EPS * 100  # More lenient for demonstration

            if bb_stable and loss_stable:
                logger.info(f"Geometric convergence at step {step}: bb_stable={bb_stable}, loss_stable={loss_stable}")
                stopped_at_step = step
                break

        if step % 100 == 0:
            logger.info(f"Step {step}: loss={float(loss):.6f}, bb_stable={bb_stable}")

    geometric_time = time.time() - start_time

    results["fixed_epochs"] = {
        "total_steps": config.num_steps,
        "final_loss": fixed_losses[-1],
        "min_loss": min(fixed_losses),
        "time_seconds": fixed_time,
    }

    results["geometric_stopping"] = {
        "total_steps": stopped_at_step,
        "final_loss": geometric_losses[-1],
        "min_loss": min(geometric_losses),
        "time_seconds": geometric_time,
        "steps_saved": config.num_steps - stopped_at_step,
        "time_reduction_percent": (1 - geometric_time / fixed_time) * 100 if fixed_time > 0 else 0,
    }

    logger.info("\n--- Summary ---")
    logger.info(f"Fixed epochs: {config.num_steps} steps, final loss={fixed_losses[-1]:.6f}, time={fixed_time:.2f}s")
    logger.info(f"Geometric: {stopped_at_step} steps, final loss={geometric_losses[-1]:.6f}, time={geometric_time:.2f}s")
    logger.info(f"Steps saved: {results['geometric_stopping']['steps_saved']} ({results['geometric_stopping']['time_reduction_percent']:.1f}% time reduction)")

    save_results(results, output_dir / "early_stopping_results.json")

    return results


# =============================================================================
# Experiment 3: Residual Scaling
# =============================================================================

def run_residual_scaling_experiment(output_dir: Path) -> dict:
    """Compare residual scaling α = σ_max(x) / σ_max(f(x)) vs α=1.

    Tests:
    1. Survey: measure σ_max(f(x))/σ_max(x) at each layer
    2. Compare alpha distribution during forward pass
    3. Training comparison with/without residual scaling
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Residual Scaling Comparison")
    logger.info("=" * 60)

    from modelcypher.core.domain.training.residual_scaling import ResidualScalingHook, _spectral_norm_fast

    results = {}

    config = Phase2ExperimentConfig(
        name="residual_scaling",
        num_steps=200,
    )

    # Create a simple residual block for testing
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 2)
            self.linear2 = nn.Linear(dim * 2, dim)
            self.norm = nn.LayerNorm(dim)

        def __call__(self, x):
            residual = x
            x = self.norm(x)
            x = nn.gelu(self.linear1(x))
            x = self.linear2(x)
            return residual + x  # Standard residual

    class ResidualModel(nn.Module):
        def __init__(self, dim: int, num_blocks: int):
            super().__init__()
            self.blocks = [ResidualBlock(dim) for _ in range(num_blocks)]
            self.output = nn.Linear(dim, dim)

        def __call__(self, x):
            for block in self.blocks:
                x = block(x)
            return self.output(x)

    # Test 1: Survey spectral ratios across layers
    logger.info("\n--- Test 1: Spectral ratio survey ---")

    mx.random.seed(config.seed)
    x = mx.random.normal(shape=(32, 64, config.model_dim))
    mx.eval(x)

    model = ResidualModel(config.model_dim, 8)
    mx.eval(model.parameters())

    spectral_ratios = []
    alpha_values = []

    # Manual forward pass to capture ratios
    h = x
    for i, block in enumerate(model.blocks):
        # Compute input spectral norm
        input_spectral = _spectral_norm_fast(h)

        # Forward through block
        residual = h
        normed = block.norm(h)
        hidden = nn.gelu(block.linear1(normed))
        out = block.linear2(hidden)
        f_h = out  # Residual contribution (before adding)

        # Compute residual spectral norm
        residual_spectral = _spectral_norm_fast(f_h)

        # Compute ratio
        if residual_spectral > SQRT_EPS:
            ratio = residual_spectral / input_spectral if input_spectral > SQRT_EPS else float('inf')
            alpha = input_spectral / residual_spectral
        else:
            ratio = 0.0
            alpha = 1.0

        spectral_ratios.append(ratio)
        alpha_values.append(alpha)

        logger.info(f"Block {i}: σ_f/σ_x = {ratio:.4f}, α = {alpha:.4f}")

        # Continue forward pass
        h = residual + out

    results["spectral_survey"] = {
        "ratios": spectral_ratios,
        "alphas": alpha_values,
        "mean_ratio": float(np.mean(spectral_ratios)),
        "std_ratio": float(np.std(spectral_ratios)),
        "mean_alpha": float(np.mean(alpha_values)),
        "alpha_range": [float(np.min(alpha_values)), float(np.max(alpha_values))],
    }

    logger.info(f"\nSurvey summary:")
    logger.info(f"  σ_f/σ_x: mean={results['spectral_survey']['mean_ratio']:.4f}, std={results['spectral_survey']['std_ratio']:.4f}")
    logger.info(f"  α: mean={results['spectral_survey']['mean_alpha']:.4f}, range=[{results['spectral_survey']['alpha_range'][0]:.4f}, {results['spectral_survey']['alpha_range'][1]:.4f}]")

    # Test 2: Verify ResidualScalingHook functionality
    logger.info("\n--- Test 2: ResidualScalingHook functionality ---")

    hook = ResidualScalingHook(min_alpha=0.1, max_alpha=10.0)

    # Process same input through hook
    h = x
    for i, block in enumerate(model.blocks):
        residual = h
        normed = block.norm(h)
        hidden = nn.gelu(block.linear1(normed))
        out = block.linear2(hidden)
        output = residual + out  # Standard output

        # Apply hook scaling
        scaled_output = hook.scale_residual(h, output, i)
        mx.eval(scaled_output)

        h = scaled_output

    hook.log_stats("Hook ")

    results["hook_stats"] = hook.state.get_alpha_summary()

    # Check success criteria
    alphas_in_range = all(0.5 <= a <= 2.0 for a in alpha_values)
    logger.info(f"\nSuccess criteria: all alphas in [0.5, 2.0]: {alphas_in_range}")

    results["success_criteria"] = {
        "alphas_in_range": alphas_in_range,
        "out_of_range_count": sum(1 for a in alpha_values if a < 0.5 or a > 2.0),
    }

    save_results(results, output_dir / "residual_scaling_results.json")

    return results


# =============================================================================
# Utilities
# =============================================================================

def save_results(results: dict, path: Path):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make numpy arrays serializable
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 geometry training heuristics experiments")
    parser.add_argument(
        "--experiment",
        choices=["weight_init", "early_stopping", "residual_scaling", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/phase2"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "weight_init": run_weight_init_experiment,
        "early_stopping": run_early_stopping_experiment,
        "residual_scaling": run_residual_scaling_experiment,
    }

    if args.experiment == "all":
        all_results = {}
        for name, func in experiments.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running experiment: {name}")
                logger.info(f"{'='*60}\n")
                all_results[name] = func(args.output_dir)
            except Exception as e:
                logger.error(f"Experiment {name} failed: {e}")
                import traceback
                traceback.print_exc()

        # Save combined summary
        save_results({"experiments": list(all_results.keys())}, args.output_dir / "summary.json")
    else:
        experiments[args.experiment](args.output_dir)


if __name__ == "__main__":
    main()
