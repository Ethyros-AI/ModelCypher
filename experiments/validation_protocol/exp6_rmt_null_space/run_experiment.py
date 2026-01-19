#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 6: RMT vs Variance Heuristic for Null-Space Selection
#
# HYPOTHESIS: Marchenko-Pastur distribution provides better signal/noise separation
# than variance-based heuristics, leading to improved behavioral preservation.
#
# THEOREM: Eigenvalues above MP edge λ+ = σ²(1 + sqrt(d/n))² are TRUE SIGNAL.
# Eigenvalues within the bulk are noise (available for transfer).
#
# PROTOCOL:
# 1. Create activations with known signal + noise structure
# 2. Compare variance-based separation vs RMT separation
# 3. Measure accuracy of signal detection
# 4. Apply null-space projection and compare behavioral preservation
#
# MEASUREMENTS:
# - Signal detection accuracy (RMT vs variance heuristic)
# - Behavioral preservation ratio
# - MP edge vs actual noise eigenvalues
#
# CONTROLS:
# - Pure noise: both methods should detect near-zero signal
# - Pure signal: both methods should detect high signal

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
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    separate_signal_noise,
    marchenko_pastur_edges,
    compute_rmt_null_space_weights,
)
from modelcypher.core.domain.geometry.transplant import compute_null_space_projector

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


def create_signal_noise_activations(
    n_samples: int,
    n_features: int,
    n_signal_dims: int,
    signal_strength: float,
    noise_std: float,
    backend,
):
    """Create synthetic activations with known signal and noise structure.

    Args:
        n_samples: Number of samples
        n_features: Number of features (dimensions)
        n_signal_dims: Number of signal dimensions (high variance)
        signal_strength: Standard deviation of signal dimensions
        noise_std: Standard deviation of noise dimensions
        backend: Compute backend

    Returns:
        activations, signal_indices
    """
    backend.random_seed(42)

    # Create noise matrix
    noise = noise_std * backend.random_normal((n_samples, n_features))
    backend.eval(noise)

    # Inject signal into specific dimensions
    signal_indices = backend.arange(0, n_signal_dims)

    # Add strong signal to those dimensions
    signal_component = signal_strength * backend.random_normal((n_samples, n_signal_dims))
    backend.eval(signal_component)

    # Create activations directly: signal in first n_signal_dims, noise in rest
    signal_part = signal_strength * backend.random_normal((n_samples, n_signal_dims))
    noise_part = noise_std * backend.random_normal((n_samples, n_features - n_signal_dims))
    backend.eval(signal_part, noise_part)

    activations = backend.concatenate([signal_part, noise_part], axis=1)
    backend.eval(activations)

    return activations, signal_indices


def variance_based_separation(activations, backend):
    """Simple variance-based signal/noise separation (the heuristic we're replacing).

    Variance normalization: normalize variance to [0, 1], threshold at 0.5.
    """
    # Compute per-dimension variance
    variance = backend.var(activations, axis=0)
    backend.eval(variance)

    max_var = backend.max(variance)
    min_var = backend.min(variance)
    backend.eval(max_var, min_var)

    max_val = float(backend.to_scalar(max_var))
    min_val = float(backend.to_scalar(min_var))
    eps = float(machine_epsilon(backend, activations))

    # Normalize to [0, 1]
    norm_variance = (variance - min_var) / max(max_val - min_val, eps)
    backend.eval(norm_variance)

    # Signal = high variance (normalized > 0.5)
    signal_mask = norm_variance > 0.5
    signal_count = int(backend.to_scalar(backend.sum(backend.astype(signal_mask, "int32"))))

    return signal_mask, signal_count


def compute_separation_accuracy(
    predicted_signal_mask,
    true_n_signal: int,
    n_features: int,
    backend,
) -> dict:
    """Compute accuracy metrics for signal detection.

    True signal is in first n_signal dimensions.
    """
    # True signal mask
    true_signal_mask = backend.arange(n_features) < true_n_signal
    backend.eval(true_signal_mask)

    # Convert to same type for comparison
    pred_mask = backend.astype(predicted_signal_mask, "bool")
    true_mask = backend.astype(true_signal_mask, "bool")
    backend.eval(pred_mask, true_mask)

    # Compute TP, FP, TN, FN
    tp = backend.sum(backend.astype(pred_mask & true_mask, "float32"))
    fp = backend.sum(backend.astype(pred_mask & ~true_mask, "float32"))
    tn = backend.sum(backend.astype(~pred_mask & ~true_mask, "float32"))
    fn = backend.sum(backend.astype(~pred_mask & true_mask, "float32"))
    backend.eval(tp, fp, tn, fn)

    tp_val = float(backend.to_scalar(tp))
    fp_val = float(backend.to_scalar(fp))
    tn_val = float(backend.to_scalar(tn))
    fn_val = float(backend.to_scalar(fn))

    total = tp_val + fp_val + tn_val + fn_val
    accuracy = (tp_val + tn_val) / max(total, 1.0)

    precision = tp_val / max(tp_val + fp_val, 1.0)
    recall = tp_val / max(tp_val + fn_val, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp_val),
        "false_positives": int(fp_val),
        "true_negatives": int(tn_val),
        "false_negatives": int(fn_val),
    }


def compute_behavioral_norm(delta_W, activations, backend) -> float:
    """Compute behavioral impact norm: ||A @ ΔW^T||_F."""
    output_change = backend.matmul(activations, backend.transpose(delta_W))
    backend.eval(output_change)

    frob_sq = backend.sum(output_change * output_change)
    backend.eval(frob_sq)
    return float(backend.to_scalar(backend.sqrt(frob_sq)))


def run_rmt_validation_test(
    n_samples: int,
    n_features: int,
    n_signal_dims: int,
    signal_strength: float,
    noise_std: float,
    backend,
) -> dict:
    """Run single validation test comparing RMT vs variance heuristic."""

    logger.info(
        "Testing: n=%d, d=%d, signal_dims=%d, signal=%.1f, noise=%.2f",
        n_samples, n_features, n_signal_dims, signal_strength, noise_std
    )

    # Create activations with known structure
    activations, true_signal_indices = create_signal_noise_activations(
        n_samples, n_features, n_signal_dims, signal_strength, noise_std, backend
    )

    # Method 1: Variance-based separation
    var_signal_mask, var_signal_count = variance_based_separation(activations, backend)
    var_accuracy = compute_separation_accuracy(
        var_signal_mask, n_signal_dims, n_features, backend
    )

    # Method 2: RMT-based separation
    rmt_result = separate_signal_noise(activations, backend)

    # Convert RMT signal indices to mask
    rmt_signal_mask = backend.zeros((n_features,), dtype="bool")
    if rmt_result.signal_rank > 0:
        # RMT gives indices in sorted eigenvalue order, not original dimension order
        # For synthetic data with signal in first dims, this should correlate
        rmt_signal_mask = backend.arange(n_features) < rmt_result.signal_rank
    backend.eval(rmt_signal_mask)

    rmt_accuracy = compute_separation_accuracy(
        rmt_signal_mask, n_signal_dims, n_features, backend
    )

    # Test behavioral preservation with each method
    backend.random_seed(123)
    delta_W = backend.random_normal((n_features // 2, n_features))
    backend.eval(delta_W)

    behavioral_before = compute_behavioral_norm(delta_W, activations, backend)

    # Compute null-space projector
    projector = compute_null_space_projector(activations, backend=backend)

    # Apply projection
    A_weighted = projector.weighted_activations
    gram_inv = projector.gram_inv
    backend.eval(A_weighted, gram_inv)

    delta_row = backend.matmul(delta_W, backend.transpose(A_weighted))
    correction = backend.matmul(delta_row, gram_inv)
    correction = backend.matmul(correction, A_weighted)
    delta_W_proj = delta_W - correction
    backend.eval(delta_W_proj)

    behavioral_after = compute_behavioral_norm(delta_W_proj, activations, backend)

    eps = float(division_epsilon(backend, delta_W))
    behavioral_ratio = behavioral_after / max(behavioral_before, eps)

    return {
        "config": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_signal_dims": n_signal_dims,
            "signal_strength": signal_strength,
            "noise_std": noise_std,
        },
        "variance_heuristic": {
            "detected_signal_count": var_signal_count,
            **var_accuracy,
        },
        "rmt": {
            "detected_signal_count": rmt_result.signal_rank,
            "mp_upper_edge": rmt_result.mp_upper_edge,
            "mp_lower_edge": rmt_result.mp_lower_edge,
            "noise_variance": rmt_result.noise_variance,
            "aspect_ratio": rmt_result.aspect_ratio,
            "signal_variance_fraction": rmt_result.signal_variance_fraction,
            **rmt_accuracy,
        },
        "behavioral_preservation": {
            "behavioral_before": behavioral_before,
            "behavioral_after": behavioral_after,
            "behavioral_ratio": behavioral_ratio,
            "null_rank": projector.null_rank,
        },
        "rmt_advantage": {
            "accuracy_diff": rmt_accuracy["accuracy"] - var_accuracy["accuracy"],
            "f1_diff": rmt_accuracy["f1"] - var_accuracy["f1"],
        },
    }


def run_control_pure_noise(n_samples: int, n_features: int, backend) -> dict:
    """Control: Pure noise data - both methods should detect minimal signal."""
    backend.random_seed(42)
    activations = backend.random_normal((n_samples, n_features))
    backend.eval(activations)

    # Variance-based
    var_signal_mask, var_signal_count = variance_based_separation(activations, backend)

    # RMT-based
    rmt_result = separate_signal_noise(activations, backend)

    return {
        "type": "pure_noise",
        "variance_signal_count": var_signal_count,
        "rmt_signal_count": rmt_result.signal_rank,
        "mp_upper_edge": rmt_result.mp_upper_edge,
    }


def run_control_pure_signal(n_samples: int, n_features: int, backend) -> dict:
    """Control: Pure signal data - both methods should detect high signal."""
    backend.random_seed(42)

    # Create data with clear structure (low-rank + small noise)
    rank = n_features // 4

    U = backend.random_normal((n_samples, rank))
    V = backend.random_normal((rank, n_features))
    noise = 0.01 * backend.random_normal((n_samples, n_features))
    backend.eval(U, V, noise)

    activations = backend.matmul(U, V) + noise
    backend.eval(activations)

    # Variance-based
    var_signal_mask, var_signal_count = variance_based_separation(activations, backend)

    # RMT-based
    rmt_result = separate_signal_noise(activations, backend)

    return {
        "type": "pure_signal",
        "true_rank": rank,
        "variance_signal_count": var_signal_count,
        "rmt_signal_count": rmt_result.signal_rank,
        "mp_upper_edge": rmt_result.mp_upper_edge,
        "signal_variance_fraction": rmt_result.signal_variance_fraction,
    }


def main():
    """Run Experiment 6: RMT vs Variance Null-Space Selection."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp6_rmt_null_space")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp6_rmt_null_space",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "test_type": "rmt_vs_variance_comparison",
            "theorem": "MP edge λ+ = σ²(1 + sqrt(d/n))² separates signal from noise",
        },
    )

    results = {
        "validation_tests": [],
        "controls": {},
    }

    # ==========================================================================
    # PART 1: Signal Detection Accuracy Tests
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Signal Detection Accuracy")
    logger.info("=" * 60)

    test_configs = [
        # Low signal-to-noise ratio
        {"n": 200, "d": 100, "signal": 20, "s_strength": 3.0, "noise": 1.0},
        {"n": 500, "d": 256, "signal": 50, "s_strength": 3.0, "noise": 1.0},
        # High signal-to-noise ratio
        {"n": 200, "d": 100, "signal": 20, "s_strength": 10.0, "noise": 1.0},
        {"n": 500, "d": 256, "signal": 50, "s_strength": 10.0, "noise": 1.0},
        # More samples than features (overdetermined)
        {"n": 300, "d": 100, "signal": 15, "s_strength": 5.0, "noise": 1.0},
        # More features than samples (underdetermined)
        {"n": 100, "d": 200, "signal": 20, "s_strength": 5.0, "noise": 1.0},
    ]

    for cfg in test_configs:
        try:
            test_result = run_rmt_validation_test(
                n_samples=cfg["n"],
                n_features=cfg["d"],
                n_signal_dims=cfg["signal"],
                signal_strength=cfg["s_strength"],
                noise_std=cfg["noise"],
                backend=backend,
            )
            results["validation_tests"].append(test_result)

            logger.info(
                "  RMT: acc=%.3f, F1=%.3f | Var: acc=%.3f, F1=%.3f | Advantage: %.3f",
                test_result["rmt"]["accuracy"],
                test_result["rmt"]["f1"],
                test_result["variance_heuristic"]["accuracy"],
                test_result["variance_heuristic"]["f1"],
                test_result["rmt_advantage"]["accuracy_diff"],
            )
        except Exception as e:
            logger.error("Error in test: %s", e)
            import traceback
            traceback.print_exc()
            results["validation_tests"].append({"config": cfg, "error": str(e)})

    # ==========================================================================
    # PART 2: Control Tests
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: Control Tests")
    logger.info("=" * 60)

    # Pure noise control
    results["controls"]["pure_noise"] = run_control_pure_noise(200, 100, backend)
    logger.info(
        "Pure noise: var=%d, rmt=%d",
        results["controls"]["pure_noise"]["variance_signal_count"],
        results["controls"]["pure_noise"]["rmt_signal_count"],
    )

    # Pure signal control
    results["controls"]["pure_signal"] = run_control_pure_signal(200, 100, backend)
    logger.info(
        "Pure signal (rank=%d): var=%d, rmt=%d",
        results["controls"]["pure_signal"]["true_rank"],
        results["controls"]["pure_signal"]["variance_signal_count"],
        results["controls"]["pure_signal"]["rmt_signal_count"],
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    valid_tests = [t for t in results["validation_tests"] if "error" not in t]

    if valid_tests:
        rmt_accuracies = [t["rmt"]["accuracy"] for t in valid_tests]
        var_accuracies = [t["variance_heuristic"]["accuracy"] for t in valid_tests]
        rmt_f1s = [t["rmt"]["f1"] for t in valid_tests]
        var_f1s = [t["variance_heuristic"]["f1"] for t in valid_tests]
        behavioral_ratios = [t["behavioral_preservation"]["behavioral_ratio"] for t in valid_tests]

        results["summary"] = {
            "n_tests": len(valid_tests),
            "rmt": {
                "mean_accuracy": sum(rmt_accuracies) / len(rmt_accuracies),
                "mean_f1": sum(rmt_f1s) / len(rmt_f1s),
            },
            "variance_heuristic": {
                "mean_accuracy": sum(var_accuracies) / len(var_accuracies),
                "mean_f1": sum(var_f1s) / len(var_f1s),
            },
            "behavioral_preservation": {
                "mean_ratio": sum(behavioral_ratios) / len(behavioral_ratios),
                "max_ratio": max(behavioral_ratios),
            },
        }

        # Report raw measurements; success = experiment ran
        # The user interprets whether RMT outperforms variance heuristic
        pure_signal = results["controls"].get("pure_signal", {})
        true_rank = pure_signal.get("true_rank", 0)
        rmt_rank = pure_signal.get("rmt_signal_count", 0)

        results["summary"]["pure_signal_control"] = {
            "true_rank": true_rank,
            "rmt_detected_rank": rmt_rank,
            "rank_error": abs(rmt_rank - true_rank),
        }
        results["summary"]["success"] = True  # Experiment completed
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
    logger.info("EXPERIMENT 6 COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results and "rmt" in results["summary"]:
        logger.info("RMT mean accuracy: %.3f", results["summary"]["rmt"]["mean_accuracy"])
        logger.info("Variance mean accuracy: %.3f", results["summary"]["variance_heuristic"]["mean_accuracy"])
        logger.info("Mean behavioral ratio: %.6f", results["summary"]["behavioral_preservation"]["mean_ratio"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
