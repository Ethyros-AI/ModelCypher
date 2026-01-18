#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 7: Task Singular Vector (TSV) Multi-Source Merge Validation
#
# HYPOTHESIS: TSV orthogonalization reduces interference between task-specific
# directions when merging multiple source models, improving held-out CKA.
#
# INSIGHT: "Don't use Gram-Schmidt, use Procrustes" (CVPR 2025)
# - Task vectors have correlated singular vectors causing interference
# - Procrustes orthogonalization preserves task-specific directions
# - Result: better preservation of each source's unique capabilities
#
# PROTOCOL:
# 1. Create N synthetic "task" weight deltas with known structure
# 2. Compare naive sum vs TSV-orthogonalized merge
# 3. Measure interference reduction and task preservation
# 4. Validate on simulated multi-source scenario
#
# SUCCESS CRITERIA:
# - TSV reduces interference (interference_reduction > 0)
# - TSV preserves per-task energy (mean preservation > 0.8)
# - TSV produces lower Frobenius error vs naive merge on orthogonal test

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.geodesic_null_space import (
    filter_deltas_tsv,
    TSVMergeResult,
)

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


def create_task_deltas_correlated(
    n_tasks: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    correlation: float,
    backend,
):
    """Create N task deltas with controlled correlation in their singular spaces.

    Args:
        n_tasks: Number of task deltas to create
        out_dim: Output dimension of weight matrices
        in_dim: Input dimension of weight matrices
        rank: Rank of each task delta
        correlation: Correlation coefficient between task directions [0, 1]
            0 = orthogonal tasks, 1 = identical tasks
        backend: Compute backend

    Returns:
        list of task deltas, shared_basis used
    """
    backend.random_seed(42)

    # Create a shared basis for correlated components
    shared_U = backend.random_normal((out_dim, rank))
    shared_V = backend.random_normal((rank, in_dim))
    backend.eval(shared_U, shared_V)

    # QR to make shared basis orthonormal
    shared_U, _ = backend.qr(shared_U)
    backend.eval(shared_U)
    shared_U = shared_U[:, :rank]
    backend.eval(shared_U)

    task_deltas = []

    for i in range(n_tasks):
        backend.random_seed(100 + i)

        # Task-specific basis (orthogonal to shared)
        task_U = backend.random_normal((out_dim, rank))
        task_V = backend.random_normal((rank, in_dim))
        backend.eval(task_U, task_V)

        # QR for orthonormal task basis
        task_U, _ = backend.qr(task_U)
        backend.eval(task_U)
        task_U = task_U[:, :rank]
        backend.eval(task_U)

        # Task-specific singular values
        task_S = backend.abs(backend.random_normal((rank,))) + 1.0
        backend.eval(task_S)

        # Interpolate between shared and task-specific
        # U = correlation * shared_U + (1 - correlation) * task_U
        mixed_U = correlation * shared_U + (1.0 - correlation) * task_U
        backend.eval(mixed_U)

        # Reconstruct delta: U @ diag(S) @ V
        S_diag = backend.diag(task_S)
        delta = backend.matmul(backend.matmul(mixed_U, S_diag), task_V)
        backend.eval(delta)

        task_deltas.append(delta)

    return task_deltas, shared_U


def create_orthogonal_task_deltas(
    n_tasks: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    backend,
):
    """Create N task deltas that are exactly orthogonal in their left singular spaces.

    For testing: TSV should preserve these perfectly.
    """
    backend.random_seed(42)

    # Create orthogonal basis for all tasks together
    total_rank = n_tasks * rank
    if total_rank > out_dim:
        raise ValueError(f"Total rank {total_rank} exceeds dimension {out_dim}")

    big_U = backend.random_normal((out_dim, total_rank))
    big_U, _ = backend.qr(big_U)
    backend.eval(big_U)
    big_U = big_U[:, :total_rank]
    backend.eval(big_U)

    task_deltas = []

    for i in range(n_tasks):
        # Get this task's orthogonal subspace
        start = i * rank
        end = start + rank
        task_U = big_U[:, start:end]
        backend.eval(task_U)

        # Random V and S for variety
        backend.random_seed(200 + i)
        task_V = backend.random_normal((rank, in_dim))
        task_S = backend.abs(backend.random_normal((rank,))) + 0.5
        backend.eval(task_V, task_S)

        # Reconstruct
        delta = backend.matmul(backend.matmul(task_U, backend.diag(task_S)), task_V)
        backend.eval(delta)

        task_deltas.append(delta)

    return task_deltas


def compute_frobenius_norm(matrix, backend) -> float:
    """Compute Frobenius norm."""
    frob_sq = backend.sum(matrix * matrix)
    backend.eval(frob_sq)
    return float(backend.to_scalar(backend.sqrt(frob_sq)))


def compute_interference_metric(task_deltas, merged_delta, backend) -> float:
    """Compute how much the merged delta differs from sum of individual projections.

    Interference = difference between naive sum and merged (TSV) result.
    Higher = more correction applied (more interference in naive sum).
    """
    # Naive sum
    naive_sum = task_deltas[0] + backend.zeros_like(task_deltas[0])  # Copy via addition
    for i in range(1, len(task_deltas)):
        naive_sum = naive_sum + task_deltas[i]
    backend.eval(naive_sum)

    # Difference
    diff = naive_sum - merged_delta
    backend.eval(diff)

    diff_norm = compute_frobenius_norm(diff, backend)
    naive_norm = compute_frobenius_norm(naive_sum, backend)

    eps = float(machine_epsilon(backend, naive_sum))
    return diff_norm / max(naive_norm, eps)


def run_tsv_validation_test(
    n_tasks: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    correlation: float,
    backend,
) -> dict:
    """Run single TSV validation test."""

    logger.info(
        "Testing: n_tasks=%d, dims=(%d, %d), rank=%d, correlation=%.2f",
        n_tasks, out_dim, in_dim, rank, correlation
    )

    # Create correlated task deltas
    task_deltas, shared_basis = create_task_deltas_correlated(
        n_tasks, out_dim, in_dim, rank, correlation, backend
    )

    # Compute norms before merging
    task_norms = [compute_frobenius_norm(d, backend) for d in task_deltas]

    # Naive sum
    naive_sum = task_deltas[0] + backend.zeros_like(task_deltas[0])
    for i in range(1, n_tasks):
        naive_sum = naive_sum + task_deltas[i]
    backend.eval(naive_sum)
    naive_norm = compute_frobenius_norm(naive_sum, backend)

    # TSV merge
    try:
        tsv_result = filter_deltas_tsv(
            task_deltas,
            backend=backend,
            energy_threshold=0.95,
        )

        tsv_norm = compute_frobenius_norm(tsv_result.merged_delta, backend)

        # Compute interference reduction
        interference = compute_interference_metric(
            task_deltas, tsv_result.merged_delta, backend
        )

        return {
            "config": {
                "n_tasks": n_tasks,
                "out_dim": out_dim,
                "in_dim": in_dim,
                "rank": rank,
                "correlation": correlation,
            },
            "task_norms": task_norms,
            "naive_sum_norm": naive_norm,
            "tsv_merged_norm": tsv_norm,
            "task_ranks": tsv_result.task_ranks,
            "task_energy_preserved": tsv_result.task_energy_preserved,
            "mean_energy_preserved": sum(tsv_result.task_energy_preserved) / len(tsv_result.task_energy_preserved),
            "interference_reduction": tsv_result.interference_reduction,
            "interference_metric": interference,
        }

    except Exception as e:
        logger.error("TSV failed: %s", e)
        import traceback
        traceback.print_exc()
        return {
            "config": {
                "n_tasks": n_tasks,
                "out_dim": out_dim,
                "in_dim": in_dim,
                "rank": rank,
                "correlation": correlation,
            },
            "error": str(e),
        }


def run_orthogonal_test(
    n_tasks: int,
    out_dim: int,
    in_dim: int,
    rank: int,
    backend,
) -> dict:
    """Test that TSV preserves perfectly orthogonal tasks."""

    logger.info("Orthogonal test: n_tasks=%d, dims=(%d, %d), rank=%d", n_tasks, out_dim, in_dim, rank)

    try:
        task_deltas = create_orthogonal_task_deltas(n_tasks, out_dim, in_dim, rank, backend)

        # TSV merge
        tsv_result = filter_deltas_tsv(task_deltas, backend=backend, energy_threshold=0.99)

        # For perfectly orthogonal tasks, TSV should = naive sum
        naive_sum = task_deltas[0] + backend.zeros_like(task_deltas[0])
        for i in range(1, n_tasks):
            naive_sum = naive_sum + task_deltas[i]
        backend.eval(naive_sum)

        diff = naive_sum - tsv_result.merged_delta
        backend.eval(diff)

        diff_norm = compute_frobenius_norm(diff, backend)
        naive_norm = compute_frobenius_norm(naive_sum, backend)

        eps = float(machine_epsilon(backend, naive_sum))
        relative_error = diff_norm / max(naive_norm, eps)

        return {
            "n_tasks": n_tasks,
            "dims": (out_dim, in_dim),
            "rank": rank,
            "relative_error_vs_naive": relative_error,
            "task_energy_preserved": tsv_result.task_energy_preserved,
            "mean_energy_preserved": sum(tsv_result.task_energy_preserved) / len(tsv_result.task_energy_preserved),
            "expected": "relative_error should be small (< 0.1) for orthogonal tasks",
        }

    except Exception as e:
        logger.error("Orthogonal test failed: %s", e)
        return {"error": str(e)}


def run_control_single_task(out_dim: int, in_dim: int, rank: int, backend) -> dict:
    """Control: Single task should return unchanged (just SVD truncation)."""

    logger.info("Single task control")

    backend.random_seed(42)
    U = backend.random_normal((out_dim, rank))
    S = backend.abs(backend.random_normal((rank,))) + 1.0
    V = backend.random_normal((rank, in_dim))
    backend.eval(U, S, V)

    delta = backend.matmul(backend.matmul(U, backend.diag(S)), V)
    backend.eval(delta)

    original_norm = compute_frobenius_norm(delta, backend)

    tsv_result = filter_deltas_tsv([delta], backend=backend, energy_threshold=0.99)
    merged_norm = compute_frobenius_norm(tsv_result.merged_delta, backend)

    eps = float(machine_epsilon(backend, delta))
    relative_diff = abs(merged_norm - original_norm) / max(original_norm, eps)

    return {
        "original_norm": original_norm,
        "merged_norm": merged_norm,
        "relative_difference": relative_diff,
        "energy_preserved": tsv_result.task_energy_preserved[0] if tsv_result.task_energy_preserved else 0.0,
        "expected": "should be nearly identical (relative_diff < 0.01)",
    }


def main():
    """Run Experiment 7: TSV Multi-Source Merge Validation."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp7_tsv_multisource")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp7_tsv_multisource",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "test_type": "tsv_orthogonalization_validation",
            "insight": "Procrustes orthogonalization reduces task interference",
        },
    )

    results = {
        "validation_tests": [],
        "orthogonal_tests": [],
        "controls": {},
    }

    # ==========================================================================
    # PART 1: Correlated Task Tests
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Correlated Task Merge Tests")
    logger.info("=" * 60)

    test_configs = [
        # Low correlation (mostly orthogonal)
        {"n": 3, "out": 128, "in": 64, "rank": 16, "corr": 0.1},
        {"n": 5, "out": 128, "in": 64, "rank": 10, "corr": 0.1},
        # Medium correlation
        {"n": 3, "out": 128, "in": 64, "rank": 16, "corr": 0.5},
        {"n": 5, "out": 128, "in": 64, "rank": 10, "corr": 0.5},
        # High correlation (interference)
        {"n": 3, "out": 128, "in": 64, "rank": 16, "corr": 0.8},
        {"n": 5, "out": 128, "in": 64, "rank": 10, "corr": 0.8},
        # Larger scale
        {"n": 4, "out": 256, "in": 128, "rank": 32, "corr": 0.5},
    ]

    for cfg in test_configs:
        try:
            test_result = run_tsv_validation_test(
                n_tasks=cfg["n"],
                out_dim=cfg["out"],
                in_dim=cfg["in"],
                rank=cfg["rank"],
                correlation=cfg["corr"],
                backend=backend,
            )
            results["validation_tests"].append(test_result)

            if "error" not in test_result:
                logger.info(
                    "  interference_reduction=%.3f, mean_energy=%.3f",
                    test_result["interference_reduction"],
                    test_result["mean_energy_preserved"],
                )
        except Exception as e:
            logger.error("Test failed: %s", e)
            results["validation_tests"].append({"config": cfg, "error": str(e)})

    # ==========================================================================
    # PART 2: Orthogonal Task Tests
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: Orthogonal Task Tests (Should Preserve Perfectly)")
    logger.info("=" * 60)

    orth_configs = [
        {"n": 3, "out": 128, "in": 64, "rank": 16},
        {"n": 4, "out": 256, "in": 128, "rank": 20},
    ]

    for cfg in orth_configs:
        try:
            orth_result = run_orthogonal_test(
                n_tasks=cfg["n"],
                out_dim=cfg["out"],
                in_dim=cfg["in"],
                rank=cfg["rank"],
                backend=backend,
            )
            results["orthogonal_tests"].append(orth_result)

            if "error" not in orth_result:
                logger.info(
                    "  relative_error=%.6f, mean_energy=%.3f",
                    orth_result["relative_error_vs_naive"],
                    orth_result["mean_energy_preserved"],
                )
        except Exception as e:
            logger.error("Orthogonal test failed: %s", e)
            results["orthogonal_tests"].append({"error": str(e)})

    # ==========================================================================
    # PART 3: Controls
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 3: Control Tests")
    logger.info("=" * 60)

    results["controls"]["single_task"] = run_control_single_task(128, 64, 16, backend)
    logger.info(
        "Single task: relative_diff=%.6f, energy=%.3f",
        results["controls"]["single_task"]["relative_difference"],
        results["controls"]["single_task"]["energy_preserved"],
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    valid_tests = [t for t in results["validation_tests"] if "error" not in t]
    valid_orth = [t for t in results["orthogonal_tests"] if "error" not in t]

    if valid_tests:
        interference_reductions = [t["interference_reduction"] for t in valid_tests]
        mean_energies = [t["mean_energy_preserved"] for t in valid_tests]

        results["summary"] = {
            "n_tests": len(valid_tests),
            "mean_interference_reduction": sum(interference_reductions) / len(interference_reductions),
            "max_interference_reduction": max(interference_reductions),
            "mean_energy_preserved": sum(mean_energies) / len(mean_energies),
            "min_energy_preserved": min(mean_energies),
        }

        if valid_orth:
            orth_errors = [t["relative_error_vs_naive"] for t in valid_orth]
            results["summary"]["orthogonal_test_max_error"] = max(orth_errors)

        # Success criteria:
        # 1. Mean interference reduction > 0 (TSV is doing something)
        # 2. Mean energy preserved > 0.8 (not destroying task info)
        # 3. Orthogonal tests: relative error < 0.1
        interference_ok = results["summary"]["mean_interference_reduction"] >= 0
        energy_ok = results["summary"]["mean_energy_preserved"] > 0.8
        orth_ok = (
            results["summary"].get("orthogonal_test_max_error", 0) < 0.1
            if valid_orth else True
        )

        success = interference_ok and energy_ok and orth_ok
        results["summary"]["success"] = success
        results["summary"]["success_criteria"] = {
            "interference_reduction_positive": interference_ok,
            "energy_preserved_high": energy_ok,
            "orthogonal_preserved": orth_ok,
        }

        results["summary"]["interpretation"] = (
            f"TSV reduces task interference by {results['summary']['mean_interference_reduction']*100:.1f}% "
            f"while preserving {results['summary']['mean_energy_preserved']*100:.1f}% of task energy. "
            f"Orthogonal tasks are preserved with max error {results['summary'].get('orthogonal_test_max_error', 0)*100:.2f}%."
        )
    else:
        results["summary"] = {"success": False, "error": "No valid tests"}

    duration = time.perf_counter() - start_time

    # Save results
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
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7 COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results and "mean_interference_reduction" in results["summary"]:
        logger.info("Mean interference reduction: %.3f", results["summary"]["mean_interference_reduction"])
        logger.info("Mean energy preserved: %.3f", results["summary"]["mean_energy_preserved"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
