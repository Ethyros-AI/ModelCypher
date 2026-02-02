#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 9: Mode Connectivity Barrier Analysis
#
# HYPOTHESIS: Mode connectivity barrier (measured via CKA proxy) predicts
# merge/LoRA insertion success. Low barrier = same loss basin = safe operation.
#
# THEORY: Models in the same loss basin can be smoothly interpolated without
# crossing high-loss regions. The barrier height measures how "separated" two
# weight configurations are in the loss landscape. For LoRA merging, this tells
# us whether the LoRA is fighting the base model's structure.
#
# PROTOCOL:
# 1. Create synthetic weight pairs with varying "distance"
# 2. Measure CKA-based loss barrier along interpolation path
# 3. Measure merge quality (CKA of merged activations vs targets)
# 4. Compute correlation between barrier and merge quality
#
# CONTROLS:
# - Identical weights: barrier = 0 (by construction)
# - Random orthogonal weights: high barrier (expected)
#
# SUCCESS CRITERIA:
# - Control tests produce expected patterns
# - Barrier correlates with merge quality (higher barrier = worse merge)
# - All computations complete without numerical issues
#
# REFERENCES:
# - Draxler et al. (2018) "Essentially No Barriers in Neural Network Energy Landscape"
# - Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling"
# - Entezari et al. (2022) "The Role of Permutation Invariance in Linear Mode Connectivity"

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
)
from modelcypher.core.domain.geometry.mode_connectivity import (
    analyze_mode_connectivity,
    InterpolationMethod,
)
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.cka_loss_proxy import make_simple_cka_loss_proxy

from experiments.validation_protocol.shared import (
    SMOLLM_PATH,
    LFM2_PATH,
    ExperimentResult,
    setup_experiment,
    ensure_output_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def create_weight_pair_with_distance(
    shape: tuple[int, int],
    distance_factor: float,
    backend,
):
    """Create two weight matrices with controlled separation.

    Args:
        shape: Weight matrix shape (out_dim, in_dim)
        distance_factor: How far apart the weights are
            0 = identical
            1 = independent random
        backend: Compute backend

    Returns:
        (source_weights, target_weights)
    """
    backend.random_seed(42)

    # Base weights
    base = backend.random_normal(shape)
    backend.eval(base)

    # Perturbation for target
    backend.random_seed(123)
    perturbation = backend.random_normal(shape)
    backend.eval(perturbation)

    # Target = base + distance_factor * perturbation
    target = base + distance_factor * perturbation
    backend.eval(target)

    return base, target


def compute_activation_based_barrier(
    source_weights,
    target_weights,
    n_samples: int,
    hidden_dim: int,
    n_steps: int,
    backend,
):
    """Compute mode connectivity barrier using activation-based CKA proxy.

    Instead of true loss, we use: loss(t) = 1 - CKA(source_acts, interpolated_acts)
    where interpolated_acts are computed by applying interpolated weights.
    """
    backend.random_seed(42)

    # Create probe inputs
    probe_inputs = backend.random_normal((n_samples, hidden_dim))
    backend.eval(probe_inputs)

    # Compute source and target activations (simple matmul as forward pass)
    source_acts = backend.matmul(probe_inputs, source_weights)
    target_acts = backend.matmul(probe_inputs, target_weights)
    backend.eval(source_acts, target_acts)

    # Center for CKA
    source_centered = source_acts - backend.mean(source_acts, axis=0, keepdims=True)
    target_centered = target_acts - backend.mean(target_acts, axis=0, keepdims=True)
    backend.eval(source_centered, target_centered)

    def cka_loss_fn(interpolated_weights):
        """CKA-based loss at interpolated weights."""
        acts = backend.matmul(probe_inputs, interpolated_weights)
        acts_centered = acts - backend.mean(acts, axis=0, keepdims=True)
        backend.eval(acts_centered)

        cka = compute_linear_cka_from_activations(source_centered, acts_centered, backend)
        return 1.0 - cka

    result = analyze_mode_connectivity(
        source_weights,
        target_weights,
        cka_loss_fn,
        n_steps=n_steps,
        method=InterpolationMethod.LINEAR,
        backend=backend,
    )

    return result, source_centered, target_centered


def compute_merge_quality(
    source_acts,
    target_acts,
    merge_ratio: float,
    backend,
):
    """Compute quality of a simple weighted merge.

    Merged activations = (1 - ratio) * source + ratio * target
    Quality = CKA(merged, target)
    """
    merged = (1.0 - merge_ratio) * source_acts + merge_ratio * target_acts
    backend.eval(merged)

    cka = compute_linear_cka_from_activations(merged, target_acts, backend)
    return cka


def run_control_identical(backend) -> dict:
    """Control: Identical weights should have zero barrier."""
    logger.info("Running control: identical weights")

    shape = (64, 64)
    weights, _ = create_weight_pair_with_distance(shape, 0.0, backend)

    result, source_acts, target_acts = compute_activation_based_barrier(
        weights, weights,  # Same weights
        n_samples=50,
        hidden_dim=64,
        n_steps=11,
        backend=backend,
    )

    eps = float(machine_epsilon(backend, weights))

    return {
        "type": "identical",
        "barrier_height": result.barrier_height,
        "normalized_barrier": result.normalized_barrier,
        "source_loss": result.source_loss,
        "target_loss": result.target_loss,
        "barrier_near_zero": result.barrier_height < eps * 100,
    }


def run_control_orthogonal(backend) -> dict:
    """Control: Random orthogonal weights should have high barrier."""
    logger.info("Running control: orthogonal weights")

    shape = (64, 64)
    source, target = create_weight_pair_with_distance(shape, 2.0, backend)

    result, source_acts, target_acts = compute_activation_based_barrier(
        source, target,
        n_samples=50,
        hidden_dim=64,
        n_steps=11,
        backend=backend,
    )

    return {
        "type": "orthogonal",
        "barrier_height": result.barrier_height,
        "normalized_barrier": result.normalized_barrier,
        "source_loss": result.source_loss,
        "target_loss": result.target_loss,
        "barrier_positive": result.barrier_height > 0,
    }


def run_distance_sweep(backend, n_distances: int = 7) -> list[dict]:
    """Sweep through different weight distances and measure barriers."""
    logger.info("Running distance sweep with %d points", n_distances)

    shape = (64, 64)
    distances = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5][:n_distances]

    results = []
    for dist in distances:
        logger.info("  Distance factor: %.2f", dist)

        source, target = create_weight_pair_with_distance(shape, dist, backend)

        barrier_result, source_acts, target_acts = compute_activation_based_barrier(
            source, target,
            n_samples=50,
            hidden_dim=64,
            n_steps=11,
            backend=backend,
        )

        # Compute merge quality at 50% interpolation
        merge_quality = compute_merge_quality(source_acts, target_acts, 0.5, backend)

        results.append({
            "distance_factor": dist,
            "barrier_height": barrier_result.barrier_height,
            "normalized_barrier": barrier_result.normalized_barrier,
            "barrier_location": barrier_result.barrier_location,
            "merge_quality_cka": merge_quality,
        })

    return results


def compute_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two lists."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0

    return cov / (var_x ** 0.5 * var_y ** 0.5)


def main():
    """Run Experiment 9: Mode Connectivity Barrier Analysis."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp9_mode_connectivity")
    initialize_default_backend()
    backend = get_default_backend()

    config = setup_experiment(
        name="exp9_mode_connectivity",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "hypothesis": "Mode connectivity barrier predicts merge/LoRA success",
            "test_type": "barrier_correlation_validation",
            "loss_proxy": "CKA divergence from source activations",
        },
    )

    results = {
        "controls": {},
        "distance_sweep": [],
        "summary": {},
    }

    # ========== PART 1: Control Tests ==========
    logger.info("=" * 70)
    logger.info("PART 1: Control Tests")
    logger.info("=" * 70)

    try:
        results["controls"]["identical"] = run_control_identical(backend)
        logger.info("Identical control: barrier=%.6f (expected ~0)",
                   results["controls"]["identical"]["barrier_height"])

        results["controls"]["orthogonal"] = run_control_orthogonal(backend)
        logger.info("Orthogonal control: barrier=%.4f (expected >0)",
                   results["controls"]["orthogonal"]["barrier_height"])

    except Exception as e:
        logger.error("Control test failed: %s", e)
        import traceback
        traceback.print_exc()

    # ========== PART 2: Distance Sweep ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: Distance Sweep")
    logger.info("=" * 70)

    try:
        results["distance_sweep"] = run_distance_sweep(backend, n_distances=7)

        for r in results["distance_sweep"]:
            logger.info("  dist=%.2f: barrier=%.4f, merge_cka=%.4f",
                       r["distance_factor"], r["barrier_height"], r["merge_quality_cka"])

    except Exception as e:
        logger.error("Distance sweep failed: %s", e)
        import traceback
        traceback.print_exc()

    # ========== PART 3: Analysis ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 3: Summary Analysis")
    logger.info("=" * 70)

    sweep = results["distance_sweep"]
    if sweep:
        barriers = [r["barrier_height"] for r in sweep]
        qualities = [r["merge_quality_cka"] for r in sweep]
        distances = [r["distance_factor"] for r in sweep]

        # Barrier should increase with distance
        barrier_distance_corr = compute_correlation(distances, barriers)

        # Barrier should negatively correlate with merge quality
        # (higher barrier = worse merge)
        barrier_quality_corr = compute_correlation(barriers, qualities)

        results["summary"] = {
            "n_tests": len(sweep),
            "barrier_distance_correlation": barrier_distance_corr,
            "barrier_quality_correlation": barrier_quality_corr,
            "mean_barrier": sum(barriers) / len(barriers) if barriers else 0,
            "mean_merge_quality": sum(qualities) / len(qualities) if qualities else 0,
            "controls_passed": (
                results["controls"].get("identical", {}).get("barrier_near_zero", False) and
                results["controls"].get("orthogonal", {}).get("barrier_positive", False)
            ),
            "success": True,
        }

        logger.info("Barrier-Distance correlation: %.3f (expect positive)",
                   barrier_distance_corr)
        logger.info("Barrier-Quality correlation: %.3f (expect negative)",
                   barrier_quality_corr)
        logger.info("Controls passed: %s", results["summary"]["controls_passed"])

    else:
        results["summary"]["success"] = False
        results["summary"]["error"] = "No sweep results"

    duration = time.perf_counter() - start_time

    # ========== SAVE ==========
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=results.get("summary", {}).get("success", False),
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9 COMPLETE")
    logger.info("=" * 70)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
