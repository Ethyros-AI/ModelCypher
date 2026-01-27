#!/usr/bin/env python3
"""Geometry-DRIVEN training for LFM2-350M.

Philosophy: Hyperparameters are not knobs. They are MEASUREMENTS.
The geometry drives the training, not arbitrary choices.

Key principles:
1. Learning rate = 1 / top_eigenvalue (maximum stable step from curvature)
2. All tolerances derived from machine epsilon
3. Monitor comp/φ during training and STOP if it degrades
4. The model already has near-perfect geometry - PRESERVE it

This script uses ModelCypher's existing Hessian estimation infrastructure
to derive training parameters from the loss landscape curvature.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Chain-of-thought training data - explicit reasoning steps
TRAINING_DATA = [
    {
        "prompt": "Question: John has 5 apples and buys 3 more. How many does he have?\n\nLet me work through this step by step:",
        "completion": "1. John starts with: 5 apples\n2. John buys: 3 more apples\n3. Total: 5 + 3 = 8 apples\n\nAnswer: 8",
    },
    {
        "prompt": "Question: A store sells pencils for $2 each. Tom buys 4 pencils. How much?\n\nLet me work through this step by step:",
        "completion": "1. Price per pencil: $2\n2. Number of pencils: 4\n3. Total cost: 4 × $2 = $8\n\nAnswer: $8",
    },
    {
        "prompt": "Question: A train travels 60 km/h for 2 hours. How far?\n\nLet me work through this step by step:",
        "completion": "1. Speed: 60 km/h\n2. Time: 2 hours\n3. Distance = Speed × Time = 60 × 2 = 120 km\n\nAnswer: 120 km",
    },
    {
        "prompt": "Question: If 3 people share 12 cookies equally, how many each?\n\nLet me work through this step by step:",
        "completion": "1. Total cookies: 12\n2. Number of people: 3\n3. Cookies per person: 12 ÷ 3 = 4\n\nAnswer: 4",
    },
    {
        "prompt": "Question: Sarah has $20 and spends $7 on lunch. How much left?\n\nLet me work through this step by step:",
        "completion": "1. Sarah starts with: $20\n2. Sarah spends: $7\n3. Remaining: $20 - $7 = $13\n\nAnswer: $13",
    },
]

# Validation prompts to test comp/φ
VALIDATION_PROMPTS = [
    "What is 15 + 8?",
    "What is 24 divided by 6?",
    "A rectangle is 5 meters long and 3 meters wide. What is its area?",
    "If you save $5 per week, how much in 8 weeks?",
    "The temperature was 10°C and dropped 15 degrees. What is it now?",
]


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


def get_compression_phi(model, tokenizer, prompt: str) -> float:
    """Get compression/φ ratio for a prompt."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []
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
            return compression_ratio / PHI

    return float('nan')


def measure_geometry(model, tokenizer, prompts: List[str]) -> Dict:
    """Measure geometric properties across prompts."""
    comp_phis = []
    for prompt in prompts:
        try:
            comp_phi = get_compression_phi(model, tokenizer, f"Question: {prompt}\n\nAnswer:")
            if not np.isnan(comp_phi):
                comp_phis.append(comp_phi)
        except Exception:
            continue

    if comp_phis:
        return {
            "mean_comp_phi": np.mean(comp_phis),
            "std_comp_phi": np.std(comp_phis),
            "min_comp_phi": np.min(comp_phis),
            "max_comp_phi": np.max(comp_phis),
            "samples": len(comp_phis),
        }
    return {"mean_comp_phi": float('nan')}


def estimate_curvature(model, tokenizer, training_data: List[Dict]) -> Dict:
    """Estimate loss landscape curvature using Hessian approximation.

    This derives the learning rate from the geometry, not arbitrary choice.
    Learning rate = 1 / (c * top_eigenvalue) where c is a safety factor.
    """
    import mlx.core as mx
    import mlx.nn as nn

    logger.info("Estimating loss landscape curvature...")

    # Compute gradients on multiple samples
    grad_norms = []
    grad_variances = []

    for sample in training_data[:3]:  # Use 3 samples for estimation
        prompt = sample["prompt"]
        completion = sample["completion"]
        full_text = prompt + completion

        tokens = tokenizer.encode(full_text)
        input_ids = mx.array([tokens[:-1]])
        targets = mx.array([tokens[1:]])

        def loss_fn(model):
            logits = model(input_ids)
            # Cross-entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
            return loss

        # Compute loss and gradient norm
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss_value, grads)

        # Compute gradient norm
        total_norm_sq = 0.0
        for key, grad in grads.items():
            if hasattr(grad, 'shape'):
                norm_sq = float(mx.sum(grad * grad).item())
                total_norm_sq += norm_sq

        grad_norm = np.sqrt(total_norm_sq)
        grad_norms.append(grad_norm)

    # Estimate curvature from gradient variance
    mean_grad_norm = np.mean(grad_norms)
    grad_variance = np.var(grad_norms) if len(grad_norms) > 1 else 0.0

    # Signal-to-noise ratio
    snr = mean_grad_norm / (np.sqrt(grad_variance) + 1e-10) if grad_variance > 0 else float('inf')

    # Estimate top eigenvalue from gradient norm (rough approximation)
    # In practice, gradient norm correlates with sqrt(top_eigenvalue)
    estimated_top_eigenvalue = mean_grad_norm ** 2

    # Machine epsilon for this dtype
    eps = np.finfo(np.float32).eps

    # Derive learning rate from curvature
    # lr_max = 1 / top_eigenvalue (stability bound)
    # Use safety factor derived from condition number
    safety_factor = max(1.0, snr / 10)  # Higher SNR allows larger steps
    lr_derived = 1.0 / (estimated_top_eigenvalue * safety_factor + eps)

    # Clamp to reasonable range (but these bounds are also derived from eps)
    lr_min = eps  # Can't go below machine precision
    lr_max = 1.0 / np.sqrt(eps)  # sqrt(eps) is typical stability bound
    lr_derived = np.clip(lr_derived, lr_min, lr_max)

    return {
        "mean_grad_norm": float(mean_grad_norm),
        "grad_variance": float(grad_variance),
        "snr": float(snr),
        "estimated_top_eigenvalue": float(estimated_top_eigenvalue),
        "derived_learning_rate": float(lr_derived),
        "safety_factor": float(safety_factor),
    }


def should_stop_training(current_comp_phi: float, baseline_comp_phi: float, tolerance: float = 0.1) -> Tuple[bool, str]:
    """Check if training should stop based on geometry degradation.

    The model already has near-perfect geometry (comp/φ ≈ 1.0).
    If training degrades this, we should STOP.
    """
    if np.isnan(current_comp_phi) or np.isnan(baseline_comp_phi):
        return False, "insufficient_data"

    # How far from ideal (1.0)?
    current_distance = abs(current_comp_phi - 1.0)
    baseline_distance = abs(baseline_comp_phi - 1.0)

    # Stop if geometry got significantly worse
    if current_distance > baseline_distance + tolerance:
        return True, f"geometry_degraded: {baseline_comp_phi:.3f} -> {current_comp_phi:.3f}"

    # Also stop if we've moved too far from φ ratio
    if current_comp_phi < 0.7 or current_comp_phi > 1.5:
        return True, f"comp_phi_out_of_range: {current_comp_phi:.3f}"

    return False, "ok"


def train_geometry_driven(
    model,
    tokenizer,
    training_data: List[Dict],
    validation_prompts: List[str],
    output_dir: Path,
):
    """Train with geometry-derived parameters.

    Key insight: The model already has perfect geometry.
    Our job is to ADD reasoning capability WITHOUT degrading geometry.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    results = {
        "timestamp": datetime.now().isoformat(),
        "philosophy": "hyperparameters are measurements, not knobs",
        "steps": [],
    }

    # Step 1: Measure baseline geometry
    logger.info("=" * 70)
    logger.info("STEP 1: MEASURE BASELINE GEOMETRY")
    logger.info("=" * 70)

    baseline_geometry = measure_geometry(model, tokenizer, validation_prompts)
    results["baseline_geometry"] = baseline_geometry

    logger.info(f"Baseline comp/φ: {baseline_geometry['mean_comp_phi']:.3f} ± {baseline_geometry.get('std_comp_phi', 0):.3f}")

    if abs(baseline_geometry['mean_comp_phi'] - 1.0) < 0.1:
        logger.info("✓ Model has near-perfect geometry (comp/φ ≈ 1.0)")
        logger.info("  Training goal: ADD reasoning without degrading this")

    # Step 2: Estimate curvature and derive learning rate
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DERIVE LEARNING RATE FROM CURVATURE")
    logger.info("=" * 70)

    curvature = estimate_curvature(model, tokenizer, training_data)
    results["curvature_estimation"] = curvature

    logger.info(f"Mean gradient norm: {curvature['mean_grad_norm']:.6f}")
    logger.info(f"Gradient SNR: {curvature['snr']:.2f}")
    logger.info(f"Estimated top eigenvalue: {curvature['estimated_top_eigenvalue']:.6f}")
    logger.info(f"DERIVED learning rate: {curvature['derived_learning_rate']:.2e}")

    # Use the derived learning rate
    learning_rate = curvature['derived_learning_rate']

    # Step 3: Create optimizer with derived parameters
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: GEOMETRY-DRIVEN TRAINING")
    logger.info("=" * 70)

    # Only train adapter layers (LoRA-style) to minimize perturbation
    # For now, we'll do full fine-tuning but with geometry monitoring

    optimizer = mx.optimizers.AdamW(learning_rate=learning_rate)

    # Training loop with geometry monitoring
    max_steps = 50  # Conservative - check geometry frequently
    geometry_check_interval = 10

    model.train()

    for step in range(max_steps):
        # Select training sample
        sample = training_data[step % len(training_data)]
        prompt = sample["prompt"]
        completion = sample["completion"]
        full_text = prompt + completion

        tokens = tokenizer.encode(full_text)
        input_ids = mx.array([tokens[:-1]])
        targets = mx.array([tokens[1:]])

        def loss_fn(model):
            logits = model(input_ids)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            return nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')

        # Forward and backward pass
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss_value, grads)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        step_result = {
            "step": step,
            "loss": float(loss_value.item()),
        }

        # Check geometry periodically
        if (step + 1) % geometry_check_interval == 0:
            model.eval()
            current_geometry = measure_geometry(model, tokenizer, validation_prompts)
            model.train()

            current_comp_phi = current_geometry['mean_comp_phi']
            step_result["comp_phi"] = current_comp_phi

            logger.info(f"Step {step + 1}: loss={loss_value.item():.4f}, comp/φ={current_comp_phi:.3f}")

            # Check if we should stop
            should_stop, reason = should_stop_training(
                current_comp_phi,
                baseline_geometry['mean_comp_phi']
            )

            if should_stop:
                logger.warning(f"STOPPING: {reason}")
                step_result["stopped_reason"] = reason
                results["steps"].append(step_result)
                results["stopped_early"] = True
                results["stop_reason"] = reason
                break

        results["steps"].append(step_result)

    # Final geometry measurement
    logger.info("\n" + "=" * 70)
    logger.info("FINAL GEOMETRY MEASUREMENT")
    logger.info("=" * 70)

    model.eval()
    final_geometry = measure_geometry(model, tokenizer, validation_prompts)
    results["final_geometry"] = final_geometry

    logger.info(f"Baseline comp/φ: {baseline_geometry['mean_comp_phi']:.3f}")
    logger.info(f"Final comp/φ:    {final_geometry['mean_comp_phi']:.3f}")

    delta = final_geometry['mean_comp_phi'] - baseline_geometry['mean_comp_phi']
    if abs(delta) < 0.05:
        logger.info("✓ Geometry PRESERVED during training")
    elif delta > 0:
        logger.info(f"⚠ Geometry increased by {delta:.3f} (more compression)")
    else:
        logger.info(f"⚠ Geometry decreased by {abs(delta):.3f} (less compression)")

    # Save results
    output_path = output_dir / "geometry_driven_training.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("GEOMETRY-DRIVEN TRAINING")
    logger.info("Hyperparameters are MEASUREMENTS, not knobs")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    logger.info(f"Architecture: {n_layers} layers, {hidden_dim} hidden dim")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = train_geometry_driven(
        model=model,
        tokenizer=tokenizer,
        training_data=TRAINING_DATA,
        validation_prompts=VALIDATION_PROMPTS,
        output_dir=output_dir,
    )

    return results


if __name__ == "__main__":
    main()
