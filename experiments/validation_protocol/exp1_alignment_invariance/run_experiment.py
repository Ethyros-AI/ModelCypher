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
# - Aligned CKA ≥ 0.9999 for all pairs with κ < 10^5
# - Raw CKA varies (0.3-0.9 expected) - confirms coordinate difference
#
# CONTROLS:
# - Random permutation baseline: expect CKA ≈ 0
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
    """Control: Random permutation of probe order."""
    n = source_acts.shape[0]

    # Shuffle indices
    backend.random_seed(999)  # Different seed for control
    perm = backend.randperm(n)
    shuffled_target = backend.take(target_acts, perm, axis=0)
    backend.eval(shuffled_target)

    # Compute CKA with shuffled
    cka_result = compute_cka(source_acts, shuffled_target, backend=backend)

    return {
        "shuffled_cka": cka_result.best,
        "expected": "near 0 (random baseline)",
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
            "probe_count": 2500,  # Need n >> d for stable alignment (ρ > 4)
            "layers_to_test": ["25%", "50%", "75%", "100%"],
        },
    )

    # Get probe texts from atlas
    # GEOMETRY: n_probes must exceed max(d_source, d_target) for stable alignment
    # SmolLM has d=576, so ρ=2500/576≈4.3 gives good overdetermination
    # Per Theorem 3: test CKA > 0.75 when n_train > 4 × max(d_s, d_t)
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
        logger.info("Random baseline CKA: %.4f",
                   results["controls"]["random_baseline"]["shuffled_cka"])

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
        stable_tests = [t for t in alignment_tests if t["condition_number"] < 1e5]

        results["summary"] = {
            "total_tests": len(alignment_tests),
            "min_aligned_cka": min(all_aligned_cka),
            "max_aligned_cka": max(all_aligned_cka),
            "mean_aligned_cka": sum(all_aligned_cka) / len(all_aligned_cka),
            "max_condition_number": max(all_condition),
            "stable_tests_count": len(stable_tests),
            "all_stable_perfect": all(t["is_perfect"] for t in stable_tests) if stable_tests else False,
        }

        # Success criteria check
        success = (
            results["summary"]["min_aligned_cka"] >= 0.9999
            and results["summary"]["all_stable_perfect"]
        )
        results["summary"]["success"] = success
        results["summary"]["success_criteria"] = "aligned_cka >= 0.9999 for all pairs with κ < 1e5"

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
        logger.info("Min aligned CKA: %.6f", results["summary"]["min_aligned_cka"])
        logger.info("Max condition number: %.2e", results["summary"]["max_condition_number"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
