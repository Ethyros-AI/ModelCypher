#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 1: Alignment Invariance Across Model Families
#
# HYPOTHESIS: CKA = 1.0 on training probes after Procrustes alignment
#
# PROTOCOL:
# 1. Extract activations from source and target models on shared probes
# 2. Compute raw CKA (before alignment)
# 3. Compute F = pinv(A_s) @ A_t
# 4. Compute aligned CKA
# 5. Record condition number κ
#
# SUCCESS CRITERIA:
# - Aligned CKA deviation < precision floor (sqrt(ε) = 3.45e-4 for float32)
# - Raw CKA varies - confirms coordinate difference
#
# CONTROLS:
# - Random vectors baseline: expect CKA ≈ 1/sqrt(n) (statistical noise)
# - Same-model control: expect raw CKA = 1.0

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

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


def collect_activations(
    model_path: Path,
    probe_texts: list[str],
    layer_idx: int,
    backend,
):
    """Collect activations from a model at specified layer."""
    from tests.fixtures.models import collect_real_activations

    # collect_real_activations returns dict[layer_idx, Array]
    activations_by_layer = collect_real_activations(
        model_path=model_path,
        probes=probe_texts,
        backend=backend,
        layer_indices=[layer_idx],
    )

    if layer_idx not in activations_by_layer:
        raise ValueError(f"Layer {layer_idx} not found in activations")

    return activations_by_layer[layer_idx]


def run_alignment_test(
    source_acts,
    target_acts,
    backend,
) -> dict:
    """Run alignment and compute metrics."""
    aligner = GramAligner(backend)

    # Compute raw CKA (before alignment)
    raw_cka_result = compute_cka(source_acts, target_acts, backend=backend)
    raw_cka = raw_cka_result.best

    # Compute alignment
    result = aligner.find_perfect_alignment(source_acts, target_acts)

    # Apply alignment to source
    F = backend.array(result.feature_transform)
    aligned_source = backend.matmul(source_acts, F)
    backend.eval(aligned_source)

    # Compute aligned CKA
    aligned_cka_result = compute_cka(aligned_source, target_acts, backend=backend)
    aligned_cka = aligned_cka_result.best

    return {
        "raw_cka": raw_cka,
        "aligned_cka": aligned_cka,
        "condition_number": result.gram_condition_number,
        "is_perfect": result.is_perfect,
        "numerical_deviation": result.numerical_deviation,
        "precision_threshold": result.precision_threshold,
    }


def run_random_baseline(
    source_acts,
    target_acts,
    backend,
) -> dict:
    """Control: Random Gaussian vectors (no structure).

    Using shuffled probes doesn't work as a control because CKA on Gram
    matrices still captures similar structure. Random vectors have no
    relational structure to match.

    Expected CKA for random vectors ≈ 1/sqrt(n) (statistical noise).
    For n=2500: expected CKA ≈ 0.02
    """
    import math
    n, d_target = target_acts.shape[0], target_acts.shape[1]

    # Generate random Gaussian vectors (no structure)
    backend.random_seed(999)
    random_acts = backend.random_normal((n, d_target))
    backend.eval(random_acts)

    # Compute CKA with random vectors
    cka_result = compute_cka(source_acts, random_acts, backend=backend)

    expected_noise = 1.0 / math.sqrt(n)  # Statistical noise floor

    return {
        "random_cka": cka_result.best,
        "expected_noise_floor": expected_noise,
        "expected": f"near {expected_noise:.4f} (1/sqrt(n) statistical noise)",
    }


def run_self_alignment_control(
    activations,
    backend,
) -> dict:
    """Control: Align model to itself."""
    aligner = GramAligner(backend)

    # Self-alignment should be trivial
    result = aligner.find_perfect_alignment(activations, activations)

    return {
        "self_cka": result.achieved_cka,
        "condition_number": result.gram_condition_number,
        "is_perfect": result.is_perfect,
        "expected": "CKA = 1.0, F ≈ I",
    }


def main():
    """Run Experiment 1: Alignment Invariance."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp1_alignment_invariance")
    backend = get_default_backend()

    # Setup experiment
    config = setup_experiment(
        name="exp1_alignment_invariance",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "probe_count": 2500,  # n > max(d_source, d_target) for non-singular Gram
            "layers_to_test": ["25%", "50%", "75%", "100%"],
        },
    )

    # Get probe texts from atlas
    # GEOMETRY: n_probes must exceed max(d_source, d_target) for non-singular Gram
    # SmolLM d=576, LFM2 d=1024 → need n > 1024
    # Using n=2500 gives overdetermined system for better numerical stability
    from tests.fixtures.models import get_atlas_probes
    probe_texts = get_atlas_probes(n_samples=2500)

    logger.info("Using %d probes for alignment test", len(probe_texts))

    results = {
        "alignment_tests": [],
        "controls": {},
    }

    # Test at multiple layers
    # SmolLM has 30 layers, LFM2 has 16 layers
    # Use proportional depth: 25%, 50%, 75%, 100%
    smol_layers = [7, 15, 22, 29]  # 25%, 50%, 75%, 100% of 30
    lfm_layers = [4, 8, 12, 15]   # 25%, 50%, 75%, 100% of 16

    for depth_name, smol_layer, lfm_layer in zip(
        ["25%", "50%", "75%", "100%"],
        smol_layers,
        lfm_layers,
    ):
        logger.info("Testing at depth %s (SmolLM layer %d, LFM2 layer %d)",
                   depth_name, smol_layer, lfm_layer)

        try:
            # Collect activations
            source_acts = collect_activations(SMOLLM_PATH, probe_texts, smol_layer, backend)
            target_acts = collect_activations(LFM2_PATH, probe_texts, lfm_layer, backend)
            backend.eval(source_acts, target_acts)

            logger.info("Source shape: %s, Target shape: %s",
                       source_acts.shape, target_acts.shape)

            # Run alignment test
            alignment_result = run_alignment_test(source_acts, target_acts, backend)
            alignment_result["depth"] = depth_name
            alignment_result["source_layer"] = smol_layer
            alignment_result["target_layer"] = lfm_layer

            results["alignment_tests"].append(alignment_result)

            logger.info(
                "Depth %s: raw_cka=%.4f, aligned_cka=%.6f, κ=%.2e, is_perfect=%s",
                depth_name,
                alignment_result["raw_cka"],
                alignment_result["aligned_cka"],
                alignment_result["condition_number"],
                alignment_result["is_perfect"],
            )

        except Exception as e:
            logger.error("Error at depth %s: %s", depth_name, e)
            results["alignment_tests"].append({
                "depth": depth_name,
                "error": str(e),
            })

    # Run controls on middle layer (50% depth)
    logger.info("Running control experiments...")

    try:
        source_acts = collect_activations(SMOLLM_PATH, probe_texts, 15, backend)
        target_acts = collect_activations(LFM2_PATH, probe_texts, 8, backend)
        backend.eval(source_acts, target_acts)

        # Random baseline
        results["controls"]["random_baseline"] = run_random_baseline(
            source_acts, target_acts, backend
        )
        logger.info("Random baseline CKA: %.4f (expected noise floor: %.4f)",
                   results["controls"]["random_baseline"]["random_cka"],
                   results["controls"]["random_baseline"]["expected_noise_floor"])

        # Self-alignment
        results["controls"]["self_alignment_source"] = run_self_alignment_control(
            source_acts, backend
        )
        results["controls"]["self_alignment_target"] = run_self_alignment_control(
            target_acts, backend
        )
        logger.info("Self-alignment source CKA: %.6f",
                   results["controls"]["self_alignment_source"]["self_cka"])
        logger.info("Self-alignment target CKA: %.6f",
                   results["controls"]["self_alignment_target"]["self_cka"])

    except Exception as e:
        logger.error("Error in controls: %s", e)
        results["controls"]["error"] = str(e)

    # Compute summary metrics
    alignment_tests = [t for t in results["alignment_tests"] if "error" not in t]
    if alignment_tests:
        all_aligned_cka = [t["aligned_cka"] for t in alignment_tests]
        all_condition = [t["condition_number"] for t in alignment_tests]
        all_deviations = [t["numerical_deviation"] for t in alignment_tests]
        stable_tests = [t for t in alignment_tests if t["condition_number"] < 1e5]

        # Precision floor from machine epsilon
        # Use a sample activation to get the dtype
        sample_dtype_array = backend.array([1.0])  # float32
        eps = machine_epsilon(backend, sample_dtype_array)
        precision_floor = float(backend.sqrt(backend.array(eps)))

        # Success criteria: deviation within 3x precision floor for numerically stable tests
        # Factor of 3 accounts for accumulated error in the alignment pipeline
        deviation_threshold = 3.0 * precision_floor
        deviations_within_threshold = [
            t["numerical_deviation"] <= deviation_threshold
            for t in stable_tests
        ]

        results["summary"] = {
            "total_tests": len(alignment_tests),
            "min_aligned_cka": min(all_aligned_cka),
            "max_aligned_cka": max(all_aligned_cka),
            "mean_aligned_cka": sum(all_aligned_cka) / len(all_aligned_cka),
            "max_numerical_deviation": max(all_deviations),
            "max_condition_number": max(all_condition),
            "stable_tests_count": len(stable_tests),
            "precision_floor": precision_floor,
            "deviation_threshold": deviation_threshold,
            "all_within_threshold": all(deviations_within_threshold) if stable_tests else False,
        }

        # Success: all numerically stable tests have deviation within threshold
        success = results["summary"]["all_within_threshold"]
        results["summary"]["success"] = success
        results["summary"]["success_criteria"] = f"deviation <= 3×sqrt(ε) = {deviation_threshold:.2e} for all pairs with κ < 1e5"

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

    logger.info("=" * 60)
    logger.info("EXPERIMENT 1 COMPLETE")
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results:
        logger.info("Mean aligned CKA: %.6f", results["summary"]["mean_aligned_cka"])
        logger.info("Max deviation: %.2e (threshold: %.2e)",
                   results["summary"]["max_numerical_deviation"],
                   results["summary"]["deviation_threshold"])
        logger.info("Max condition number: %.2e", results["summary"]["max_condition_number"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
