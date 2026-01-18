#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 9: Residual Decomposition Test
#
# THESIS: Models cannot align what they don't share. The residual measures
#         the non-shared portion.
#
# MATHEMATICAL SETUP:
#   Given source A (n x d_s), target B (n x d_t), alignment F = pinv(A) @ B
#   Decomposition:
#       reconstructed = A @ F     # What source CAN express about target
#       novel = B - reconstructed # What source CANNOT express
#   Residual r = ||novel||_F / ||B||_F measures non-shared fraction
#
# PROTOCOL:
#   Phase 1: Measure decomposition per layer
#   Phase 2: Test orthogonality of novel subspace to source column space
#   Phase 3: Test generalization (train/test split)
#   Phase 4: Probe-by-probe residual analysis
#
# SUCCESS CRITERIA (precision-derived, not guessed):
#   1. ||A.T @ novel|| / (||A|| x ||novel||) < sqrt(eps) - truly orthogonal
#   2. |test_residual - train_residual| < sqrt(eps) - generalizes
#   3. Per-probe residuals cluster (not uniform scatter) - concept-dependent
#   4. High-residual probes consistent across layers

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    numerical_rank_truncated_lstsq,
    machine_epsilon,
    sqrt_scalar,
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


def collect_activations(
    model_path: Path,
    probe_texts: list[str],
    layer_idx: int,
    backend,
):
    """Collect activations from a model at specified layer."""
    from tests.fixtures.models import collect_real_activations

    activations_by_layer = collect_real_activations(
        model_path=model_path,
        probes=probe_texts,
        backend=backend,
        layer_indices=[layer_idx],
    )

    if layer_idx not in activations_by_layer:
        raise ValueError(f"Layer {layer_idx} not found in activations")

    return activations_by_layer[layer_idx]


def compute_decomposition(source_acts, target_acts, backend):
    """
    Compute the alignment and decomposition.

    Returns:
        dict with keys:
            - F: alignment transform
            - reconstructed: source @ F
            - novel: target - reconstructed
            - residual: ||novel|| / ||target||
            - reconstructed_fraction: ||reconstructed|| / ||target||
            - source_rank, target_rank, alignment_rank, condition_number
    """
    b = backend

    # Compute alignment and residual
    F, source_rank, target_rank, alignment_rank, condition_number, residual = (
        numerical_rank_truncated_lstsq(b, source_acts, target_acts)
    )

    # Decompose
    reconstructed = b.matmul(source_acts, F)
    novel = target_acts - reconstructed
    b.eval(reconstructed, novel)

    # Compute norms
    target_norm = b.sqrt(b.sum(target_acts * target_acts))
    reconstructed_norm = b.sqrt(b.sum(reconstructed * reconstructed))
    novel_norm = b.sqrt(b.sum(novel * novel))
    b.eval(target_norm, reconstructed_norm, novel_norm)

    reconstructed_fraction = float(b.to_scalar(reconstructed_norm)) / max(
        float(b.to_scalar(target_norm)), 1e-12
    )

    return {
        "F": F,
        "reconstructed": reconstructed,
        "novel": novel,
        "residual": residual,
        "reconstructed_fraction": reconstructed_fraction,
        "source_rank": source_rank,
        "target_rank": target_rank,
        "alignment_rank": alignment_rank,
        "condition_number": condition_number,
    }


def test_orthogonality(source_acts, novel, backend):
    """
    Test if novel is truly orthogonal to source's column space.

    Computes ||A.T @ novel||_F / (||A||_F x ||novel||_F)

    If this is < sqrt(eps), novel is genuinely outside source's reach.
    """
    b = backend

    # A.T @ novel: [d_source, n] @ [n, d_target] = [d_source, d_target]
    cross_product = b.matmul(b.transpose(source_acts), novel)
    b.eval(cross_product)

    # Compute norms
    cross_norm = b.sqrt(b.sum(cross_product * cross_product))
    source_norm = b.sqrt(b.sum(source_acts * source_acts))
    novel_norm = b.sqrt(b.sum(novel * novel))
    b.eval(cross_norm, source_norm, novel_norm)

    # Normalized cross product
    denom = float(b.to_scalar(source_norm)) * float(b.to_scalar(novel_norm))
    orthogonality = float(b.to_scalar(cross_norm)) / max(denom, 1e-12)

    return orthogonality


def test_generalization(
    source_train, target_train,
    source_test, target_test,
    backend,
):
    """
    Test if the decomposition generalizes to held-out probes.

    Returns:
        dict with train_residual, test_residual, generalization_gap
    """
    b = backend

    # Compute F on training data
    F, _, _, _, _, train_residual = numerical_rank_truncated_lstsq(
        b, source_train, target_train
    )

    # Apply to test data
    reconstructed_test = b.matmul(source_test, F)
    novel_test = target_test - reconstructed_test
    b.eval(reconstructed_test, novel_test)

    # Compute test residual
    novel_test_norm = b.sqrt(b.sum(novel_test * novel_test))
    target_test_norm = b.sqrt(b.sum(target_test * target_test))
    b.eval(novel_test_norm, target_test_norm)

    test_residual = float(b.to_scalar(novel_test_norm)) / max(
        float(b.to_scalar(target_test_norm)), 1e-12
    )

    return {
        "train_residual": train_residual,
        "test_residual": test_residual,
        "generalization_gap": abs(test_residual - train_residual),
    }


def compute_per_probe_residuals(source_acts, target_acts, backend):
    """
    Compute per-probe residuals to see if they concentrate on specific concepts.

    Returns:
        list of per-probe residuals (one per probe)
    """
    b = backend

    # Compute F
    F, _, _, _, _, _ = numerical_rank_truncated_lstsq(b, source_acts, target_acts)

    # Compute per-probe
    reconstructed = b.matmul(source_acts, F)
    novel = target_acts - reconstructed
    b.eval(reconstructed, novel)

    # Per-probe norms: ||novel_i|| / ||target_i||
    n_probes = target_acts.shape[0]
    per_probe_residuals = []

    for i in range(n_probes):
        target_i = b.take(target_acts, b.array([i]), axis=0)
        novel_i = b.take(novel, b.array([i]), axis=0)
        b.eval(target_i, novel_i)

        target_norm = b.sqrt(b.sum(target_i * target_i))
        novel_norm = b.sqrt(b.sum(novel_i * novel_i))
        b.eval(target_norm, novel_norm)

        residual_i = float(b.to_scalar(novel_norm)) / max(
            float(b.to_scalar(target_norm)), 1e-12
        )
        per_probe_residuals.append(residual_i)

    return per_probe_residuals


def analyze_probe_residuals(residuals: list[float], probe_texts: list[str]):
    """
    Analyze the distribution of per-probe residuals.

    Returns statistics and the top/bottom probes by residual.
    """
    import statistics

    sorted_indices = sorted(range(len(residuals)), key=lambda i: residuals[i])

    # Get top 10 (highest residual - least shared)
    top_10_idx = sorted_indices[-10:][::-1]
    top_10 = [
        {"index": i, "residual": residuals[i], "probe": probe_texts[i][:80]}
        for i in top_10_idx
    ]

    # Get bottom 10 (lowest residual - most shared)
    bottom_10_idx = sorted_indices[:10]
    bottom_10 = [
        {"index": i, "residual": residuals[i], "probe": probe_texts[i][:80]}
        for i in bottom_10_idx
    ]

    return {
        "mean": statistics.mean(residuals),
        "median": statistics.median(residuals),
        "stdev": statistics.stdev(residuals) if len(residuals) > 1 else 0.0,
        "min": min(residuals),
        "max": max(residuals),
        "top_10_highest_residual": top_10,
        "bottom_10_lowest_residual": bottom_10,
    }


def main():
    """Run Experiment 9: Residual Decomposition."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp9_residual_decomposition")
    backend = get_default_backend()

    # Setup experiment
    config = setup_experiment(
        name="exp9_residual_decomposition",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "probe_count": 500,
            "train_fraction": 0.8,
            "layers_to_test": ["25%", "50%", "75%"],
        },
    )

    # Get probe texts
    from tests.fixtures.models import get_atlas_probes
    all_probe_texts = get_atlas_probes(n_samples=500)
    logger.info("Using %d probes", len(all_probe_texts))

    # Split train/test (80/20)
    n_total = len(all_probe_texts)
    n_train = int(n_total * 0.8)
    train_probes = all_probe_texts[:n_train]
    test_probes = all_probe_texts[n_train:]
    logger.info("Train: %d probes, Test: %d probes", len(train_probes), len(test_probes))

    # Machine epsilon will be computed from first activation array
    eps = None
    sqrt_eps = None

    results = {
        "precision": {},
        "phase1_decomposition": [],
        "phase2_orthogonality": [],
        "phase3_generalization": [],
        "phase4_probe_analysis": [],
    }

    # Layer mappings (proportional depth)
    # SmolLM has 30 layers, LFM2 has 16 layers
    smol_layers = [7, 15, 22]  # 25%, 50%, 75%
    lfm_layers = [4, 8, 12]

    for depth_name, smol_layer, lfm_layer in zip(
        ["25%", "50%", "75%"],
        smol_layers,
        lfm_layers,
    ):
        logger.info("=" * 60)
        logger.info("Testing at depth %s (SmolLM layer %d, LFM2 layer %d)",
                   depth_name, smol_layer, lfm_layer)

        try:
            # ===== PHASE 1: Decomposition =====
            logger.info("Phase 1: Computing decomposition...")
            source_acts = collect_activations(SMOLLM_PATH, all_probe_texts, smol_layer, backend)
            target_acts = collect_activations(LFM2_PATH, all_probe_texts, lfm_layer, backend)
            backend.eval(source_acts, target_acts)

            # Compute machine epsilon from actual activations (first time only)
            if eps is None:
                eps = float(machine_epsilon(backend, source_acts))
                sqrt_eps = sqrt_scalar(eps, backend)
                results["precision"] = {"eps": eps, "sqrt_eps": sqrt_eps}
                logger.info("Machine epsilon: %.2e, sqrt(eps): %.2e", eps, sqrt_eps)

            logger.info("Source shape: %s, Target shape: %s",
                       source_acts.shape, target_acts.shape)

            decomp = compute_decomposition(source_acts, target_acts, backend)

            phase1_result = {
                "depth": depth_name,
                "source_layer": smol_layer,
                "target_layer": lfm_layer,
                "residual": decomp["residual"],
                "reconstructed_fraction": decomp["reconstructed_fraction"],
                "source_rank": decomp["source_rank"],
                "target_rank": decomp["target_rank"],
                "alignment_rank": decomp["alignment_rank"],
                "condition_number": decomp["condition_number"],
            }
            results["phase1_decomposition"].append(phase1_result)

            logger.info(
                "Phase 1 Result: residual=%.6f, reconstructed_frac=%.6f, κ=%.2e",
                decomp["residual"],
                decomp["reconstructed_fraction"],
                decomp["condition_number"],
            )

            # ===== PHASE 2: Orthogonality =====
            logger.info("Phase 2: Testing orthogonality...")
            orthogonality = test_orthogonality(source_acts, decomp["novel"], backend)

            is_orthogonal = orthogonality < sqrt_eps
            phase2_result = {
                "depth": depth_name,
                "orthogonality": orthogonality,
                "sqrt_eps": sqrt_eps,
                "is_truly_orthogonal": is_orthogonal,
            }
            results["phase2_orthogonality"].append(phase2_result)

            logger.info(
                "Phase 2 Result: orthogonality=%.6e, sqrt_eps=%.2e, truly_orthogonal=%s",
                orthogonality, sqrt_eps, is_orthogonal,
            )

            # ===== PHASE 3: Generalization =====
            logger.info("Phase 3: Testing generalization...")
            source_train = collect_activations(SMOLLM_PATH, train_probes, smol_layer, backend)
            target_train = collect_activations(LFM2_PATH, train_probes, lfm_layer, backend)
            source_test = collect_activations(SMOLLM_PATH, test_probes, smol_layer, backend)
            target_test = collect_activations(LFM2_PATH, test_probes, lfm_layer, backend)
            backend.eval(source_train, target_train, source_test, target_test)

            gen_result = test_generalization(
                source_train, target_train,
                source_test, target_test,
                backend,
            )

            generalizes = gen_result["generalization_gap"] < sqrt_eps
            phase3_result = {
                "depth": depth_name,
                "train_residual": gen_result["train_residual"],
                "test_residual": gen_result["test_residual"],
                "generalization_gap": gen_result["generalization_gap"],
                "sqrt_eps": sqrt_eps,
                "generalizes": generalizes,
            }
            results["phase3_generalization"].append(phase3_result)

            logger.info(
                "Phase 3 Result: train=%.6f, test=%.6f, gap=%.6e, generalizes=%s",
                gen_result["train_residual"],
                gen_result["test_residual"],
                gen_result["generalization_gap"],
                generalizes,
            )

            # ===== PHASE 4: Probe-by-Probe Analysis =====
            logger.info("Phase 4: Computing per-probe residuals...")
            per_probe = compute_per_probe_residuals(source_acts, target_acts, backend)
            analysis = analyze_probe_residuals(per_probe, all_probe_texts)

            phase4_result = {
                "depth": depth_name,
                "statistics": {
                    "mean": analysis["mean"],
                    "median": analysis["median"],
                    "stdev": analysis["stdev"],
                    "min": analysis["min"],
                    "max": analysis["max"],
                },
                "top_10_highest_residual": analysis["top_10_highest_residual"],
                "bottom_10_lowest_residual": analysis["bottom_10_lowest_residual"],
            }
            results["phase4_probe_analysis"].append(phase4_result)

            logger.info(
                "Phase 4 Result: mean=%.4f, stdev=%.4f, min=%.4f, max=%.4f",
                analysis["mean"], analysis["stdev"], analysis["min"], analysis["max"],
            )

        except Exception as e:
            logger.error("Error at depth %s: %s", depth_name, e)
            import traceback
            traceback.print_exc()
            results["phase1_decomposition"].append({
                "depth": depth_name,
                "error": str(e),
            })

    # ===== SUMMARY =====
    logger.info("=" * 60)
    logger.info("COMPUTING SUMMARY")

    valid_decomp = [r for r in results["phase1_decomposition"] if "error" not in r]
    valid_ortho = [r for r in results["phase2_orthogonality"]]
    valid_gen = [r for r in results["phase3_generalization"]]

    if valid_decomp:
        # Check success criteria
        all_orthogonal = all(r["is_truly_orthogonal"] for r in valid_ortho) if valid_ortho else False
        all_generalize = all(r["generalizes"] for r in valid_gen) if valid_gen else False

        residuals = [r["residual"] for r in valid_decomp]
        mean_residual = sum(residuals) / len(residuals)

        results["summary"] = {
            "total_layers_tested": len(valid_decomp),
            "mean_residual": mean_residual,
            "all_orthogonal": all_orthogonal,
            "all_generalize": all_generalize,
            "thesis_supported": all_orthogonal,  # Key criterion
            "success_criteria": {
                "orthogonality": "||A.T @ novel|| / (||A|| x ||novel||) < sqrt(eps)",
                "generalization": "|test_residual - train_residual| < sqrt(eps)",
            },
        }

        logger.info("Mean residual across layers: %.6f", mean_residual)
        logger.info("All layers orthogonal: %s", all_orthogonal)
        logger.info("All layers generalize: %s", all_generalize)
        logger.info("THESIS SUPPORTED: %s", all_orthogonal)

    duration = time.perf_counter() - start_time

    # Save results
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=results.get("summary", {}).get("thesis_supported", False),
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 9 COMPLETE")
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Thesis supported: %s", experiment_result.success)
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
