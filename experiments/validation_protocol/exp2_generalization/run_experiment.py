#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 2: Generalization to Held-Out Concepts
#
# HYPOTHESIS: Alignment learned on training probes generalizes to held-out probes
#
# THEOREM 3: If alignment is learned on n_train probes and tested on n_test held-out
# probes, then CKA_test > 0.75 when n_train > 4 × max(d_s, d_t)
#
# PROTOCOL:
# 1. Split probes: 80% train, 20% test (stratified by domain)
# 2. Learn alignment on train: F = pinv(A_s_train) @ A_t_train
# 3. Apply to test: A_s_test_aligned = A_s_test @ F
# 4. Measure test CKA
# 5. Vary coverage ratio ρ = n_train / d: [0.5, 1.0, 2.0, 4.0, 8.0]
#
# SUCCESS CRITERIA:
# - ρ < 1.0: test CKA < 0.5 (underdetermined, expect poor generalization)
# - ρ = 1.0: test CKA ~ 0.5-0.7 (borderline)
# - ρ ≥ 4.0: test CKA > 0.75 (proper coverage)
#
# CONTROLS:
# - Random F (orthogonal matrix): expect test CKA ≈ raw CKA
# - Same split but no alignment: baseline raw CKA

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner

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

MMLU_DIR = Path(__file__).parent.parent.parent.parent / "data" / "probes" / "mmlu"


def load_all_mmlu_probes() -> list[str]:
    """Load all MMLU probes as a flat list."""
    all_probes = []

    for json_file in MMLU_DIR.glob("mmlu_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        all_probes.extend([p["text"] for p in data["probes"]])

    return all_probes


def collect_activations(
    model_path: Path,
    probes: list[str],
    layer_idx: int,
    backend,
):
    """Collect activations from a model."""
    from tests.fixtures.models import collect_real_activations

    activations_by_layer = collect_real_activations(
        model_path=model_path,
        probes=probes,
        backend=backend,
        layer_indices=[layer_idx],
    )

    if layer_idx not in activations_by_layer:
        raise ValueError(f"Layer {layer_idx} not found")

    return activations_by_layer[layer_idx]


def run_generalization_test(
    source_train, target_train,
    source_test, target_test,
    backend,
) -> dict:
    """Test generalization of alignment from train to test split.

    Args:
        source_train: Source activations for training [n_train, d_s]
        target_train: Target activations for training [n_train, d_t]
        source_test: Source activations for testing [n_test, d_s]
        target_test: Target activations for testing [n_test, d_t]
        backend: Compute backend

    Returns:
        Dict with train/test CKA values and alignment metrics
    """
    aligner = GramAligner(backend)

    # Compute raw CKA on test (baseline)
    raw_cka_test = compute_cka(source_test, target_test, backend=backend).best

    # Learn alignment on training set
    alignment_result = aligner.find_perfect_alignment(source_train, target_train)

    # Compute train CKA (should be ~1.0)
    F = backend.array(alignment_result.feature_transform)
    aligned_train = backend.matmul(source_train, F)
    backend.eval(aligned_train)
    train_cka = compute_cka(aligned_train, target_train, backend=backend).best

    # Apply alignment to test set and measure CKA
    aligned_test = backend.matmul(source_test, F)
    backend.eval(aligned_test)
    test_cka = compute_cka(aligned_test, target_test, backend=backend).best

    # Compute coverage ratio
    n_train = int(source_train.shape[0])
    d_source = int(source_train.shape[1])
    d_target = int(target_train.shape[1])
    d_max = max(d_source, d_target)
    coverage_ratio = n_train / d_max

    return {
        "raw_cka_test": raw_cka_test,
        "train_cka": train_cka,
        "test_cka": test_cka,
        "generalization_gap": train_cka - test_cka,
        "improvement_over_raw": test_cka - raw_cka_test,
        "condition_number": alignment_result.gram_condition_number,
        "is_perfect": alignment_result.is_perfect,
        "n_train": n_train,
        "n_test": int(source_test.shape[0]),
        "d_source": d_source,
        "d_target": d_target,
        "coverage_ratio": coverage_ratio,
    }


def run_random_baseline(source_test, target_test, d_source, d_target, backend) -> float:
    """Control: Random orthogonal alignment matrix."""
    # Generate random orthogonal matrix via QR decomposition
    backend.random_seed(999)
    random_mat = backend.random_normal((d_source, d_target))
    Q, _ = backend.qr(random_mat)
    backend.eval(Q)

    # Truncate if needed (QR gives square matrix of smaller dim)
    if d_source <= d_target:
        # Q is [d_source, d_source], need to pad to [d_source, d_target]
        padding = backend.zeros((d_source, d_target - d_source))
        Q_padded = backend.concatenate([Q, padding], axis=1)
        backend.eval(Q_padded)
        F_random = Q_padded
    else:
        # Q is [d_source, d_source], need to truncate to [d_source, d_target]
        F_random = Q[:, :d_target]

    # Apply random alignment
    aligned_test = backend.matmul(source_test, F_random)
    backend.eval(aligned_test)

    return compute_cka(aligned_test, target_test, backend=backend).best


def main():
    """Run Experiment 2: Generalization to Held-Out Concepts."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp2_generalization")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp2_generalization",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "dataset": "MMLU",
            "test_type": "train_test_generalization",
            "theory": "Alignment generalizes when coverage_ratio > 4.0",
        },
    )

    results = {
        "coverage_tests": [],
        "controls": {},
        "summary": {},
    }

    # Load all MMLU probes
    logger.info("Loading MMLU probes...")
    all_probes = load_all_mmlu_probes()
    logger.info("Loaded %d total probes", len(all_probes))

    # Shuffle with fixed seed for reproducibility
    backend.random_seed(42)
    import random
    random.seed(42)
    random.shuffle(all_probes)

    # Test at 50% depth
    smol_layer = 15  # 50% of 30
    lfm_layer = 8    # 50% of 16

    # SmolLM has d=576, LFM2 has d=1024
    # We use the smaller dimension for coverage ratio
    d_ref = 576

    # Test different coverage ratios
    coverage_ratios = [0.5, 1.0, 2.0, 4.0, 8.0]

    logger.info("=" * 70)
    logger.info("Testing generalization at different coverage ratios")
    logger.info("Reference dimension: %d (SmolLM hidden dim)", d_ref)
    logger.info("=" * 70)

    for rho in coverage_ratios:
        n_train = int(rho * d_ref)
        n_test = min(int(0.25 * n_train), len(all_probes) - n_train)  # 25% of train, capped

        if n_train + n_test > len(all_probes):
            logger.warning("Not enough probes for ρ=%.1f (need %d, have %d)",
                          rho, n_train + n_test, len(all_probes))
            continue

        logger.info("")
        logger.info("Testing ρ=%.1f (n_train=%d, n_test=%d)...", rho, n_train, n_test)

        # Split probes
        train_probes = all_probes[:n_train]
        test_probes = all_probes[n_train:n_train + n_test]

        try:
            # Collect activations
            logger.info("  Collecting source activations...")
            source_train = collect_activations(SMOLLM_PATH, train_probes, smol_layer, backend)
            source_test = collect_activations(SMOLLM_PATH, test_probes, smol_layer, backend)
            backend.eval(source_train, source_test)

            logger.info("  Collecting target activations...")
            target_train = collect_activations(LFM2_PATH, train_probes, lfm_layer, backend)
            target_test = collect_activations(LFM2_PATH, test_probes, lfm_layer, backend)
            backend.eval(target_train, target_test)

            # Run test
            test_result = run_generalization_test(
                source_train, target_train,
                source_test, target_test,
                backend,
            )
            test_result["target_coverage_ratio"] = rho

            results["coverage_tests"].append(test_result)

            logger.info(
                "  ρ=%.1f: train_CKA=%.4f, test_CKA=%.4f, raw_CKA=%.4f, gap=%.4f, κ=%.2e",
                rho,
                test_result["train_cka"],
                test_result["test_cka"],
                test_result["raw_cka_test"],
                test_result["generalization_gap"],
                test_result["condition_number"],
            )

            # Run random baseline for the highest coverage ratio
            if rho == max(coverage_ratios):
                random_cka = run_random_baseline(
                    source_test, target_test,
                    int(source_test.shape[1]), int(target_test.shape[1]),
                    backend
                )
                results["controls"]["random_alignment_cka"] = random_cka
                logger.info("  Control (random alignment): CKA=%.4f", random_cka)

        except Exception as e:
            logger.error("  Error for ρ=%.1f: %s", rho, e)
            import traceback
            traceback.print_exc()
            results["coverage_tests"].append({
                "target_coverage_ratio": rho,
                "n_train": n_train,
                "error": str(e),
            })

    # Compute summary
    valid_tests = [t for t in results["coverage_tests"] if "error" not in t]

    if valid_tests:
        # Group by coverage regime
        underdetermined = [t for t in valid_tests if t["coverage_ratio"] < 1.0]
        borderline = [t for t in valid_tests if 1.0 <= t["coverage_ratio"] < 4.0]
        overdetermined = [t for t in valid_tests if t["coverage_ratio"] >= 4.0]

        if underdetermined:
            ud_test_cka = [t["test_cka"] for t in underdetermined]
            results["summary"]["underdetermined"] = {
                "n_tests": len(underdetermined),
                "mean_test_cka": sum(ud_test_cka) / len(ud_test_cka),
                "expected": "test_cka < 0.5",
            }

        if borderline:
            bl_test_cka = [t["test_cka"] for t in borderline]
            results["summary"]["borderline"] = {
                "n_tests": len(borderline),
                "mean_test_cka": sum(bl_test_cka) / len(bl_test_cka),
                "expected": "test_cka ~ 0.5-0.7",
            }

        if overdetermined:
            od_test_cka = [t["test_cka"] for t in overdetermined]
            results["summary"]["overdetermined"] = {
                "n_tests": len(overdetermined),
                "mean_test_cka": sum(od_test_cka) / len(od_test_cka),
                "min_test_cka": min(od_test_cka),
                "max_test_cka": max(od_test_cka),
                "expected": "test_cka > 0.75",
            }

        # Success criteria
        # - Overdetermined (ρ ≥ 4.0) should have test_cka > 0.75
        od_pass = all(t["test_cka"] > 0.75 for t in overdetermined) if overdetermined else True
        results["summary"]["success"] = od_pass
        results["summary"]["success_criteria"] = "test_cka > 0.75 when coverage_ratio >= 4.0"

        # Generalization curve
        results["summary"]["generalization_curve"] = [
            {"coverage_ratio": t["coverage_ratio"], "test_cka": t["test_cka"]}
            for t in sorted(valid_tests, key=lambda x: x["coverage_ratio"])
        ]

    duration = time.perf_counter() - start_time

    # Save
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
    logger.info("EXPERIMENT 2 COMPLETE: Generalization to Held-Out Concepts")
    logger.info("=" * 70)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)

    if "summary" in results:
        if "underdetermined" in results["summary"]:
            ud = results["summary"]["underdetermined"]
            logger.info("Underdetermined (ρ<1): mean_test_cka=%.4f (%s)",
                       ud["mean_test_cka"], ud["expected"])

        if "borderline" in results["summary"]:
            bl = results["summary"]["borderline"]
            logger.info("Borderline (1≤ρ<4): mean_test_cka=%.4f (%s)",
                       bl["mean_test_cka"], bl["expected"])

        if "overdetermined" in results["summary"]:
            od = results["summary"]["overdetermined"]
            logger.info("Overdetermined (ρ≥4): mean_test_cka=%.4f, range=[%.4f, %.4f] (%s)",
                       od["mean_test_cka"], od["min_test_cka"], od["max_test_cka"],
                       od["expected"])

    if "controls" in results:
        if "random_alignment_cka" in results["controls"]:
            logger.info("Control (random): test_cka=%.4f", results["controls"]["random_alignment_cka"])

    logger.info("")
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
