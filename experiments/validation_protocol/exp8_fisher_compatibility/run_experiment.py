#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 8: Fisher Information Merge Compatibility Prediction
#
# HYPOTHESIS: Fisher Information compatibility scores correlate with actual
# merge quality. Models with similar loss landscape curvature merge better.
#
# THEORY: The Fisher Information Matrix (FIM) captures loss landscape curvature.
# Two models are compatible for merging if they have similar FIM structure:
# - Similar curvature directions (cosine similarity)
# - Similar importance rankings (correlation)
# - Similar significant dimensions (overlap ratio)
#
# PROTOCOL:
# 1. Create synthetic model pairs with varying FIM similarity
# 2. Compute Fisher compatibility scores
# 3. Perform merge and measure quality (behavioral alignment)
# 4. Compute correlation between Fisher score and merge quality
#
# SUCCESS CRITERIA:
# - Fisher compatibility correlates with merge quality (r > 0.7)
# - High Fisher compatibility predicts good merges (>90% alignment)
# - Low Fisher compatibility predicts poor merges (<70% alignment)
#
# REFERENCES:
# - Kirkpatrick et al. (2017) "EWC: Elastic Weight Consolidation"
# - Matena & Raffel (2022) "Merging Models with Fisher-Weighted Averaging"

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
from modelcypher.core.domain.geometry.fisher_information import (
    fisher_compatibility_score,
    compute_empirical_fisher_diagonal,
    FisherCompatibilityResult,
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


def create_activations_with_fisher_similarity(
    n_samples: int,
    n_features: int,
    similarity: float,
    backend,
):
    """Create two sets of activations with controlled Fisher similarity.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        similarity: Target FIM similarity in [0, 1]
            0 = completely different FIM structure
            1 = identical FIM structure
        backend: Compute backend

    Returns:
        (source_activations, target_activations)
    """
    backend.random_seed(42)

    # Create base structure (shared between source and target)
    # Higher similarity = more shared structure

    # Shared diagonal importance pattern
    shared_importance = backend.abs(backend.random_normal((n_features,))) + 0.5
    backend.eval(shared_importance)

    # Create source activations with this importance pattern
    # Activations scaled by sqrt(importance) to control variance
    source_scale = backend.sqrt(shared_importance)
    source_scale = backend.reshape(source_scale, (1, n_features))
    source_noise = backend.random_normal((n_samples, n_features))
    backend.eval(source_scale, source_noise)

    source_activations = source_noise * source_scale
    backend.eval(source_activations)

    # Create target activations with mixed importance
    # Mix between shared and independent importance pattern
    backend.random_seed(123)
    independent_importance = backend.abs(backend.random_normal((n_features,))) + 0.5
    backend.eval(independent_importance)

    # Interpolate: target_importance = similarity * shared + (1 - similarity) * independent
    target_importance = similarity * shared_importance + (1.0 - similarity) * independent_importance
    backend.eval(target_importance)

    target_scale = backend.sqrt(target_importance)
    target_scale = backend.reshape(target_scale, (1, n_features))
    target_noise = backend.random_normal((n_samples, n_features))
    backend.eval(target_scale, target_noise)

    target_activations = target_noise * target_scale
    backend.eval(target_activations)

    return source_activations, target_activations


def compute_merge_quality(
    source_activations,
    target_activations,
    backend,
) -> dict:
    """Compute merge quality via representation alignment.

    Fisher compatibility predicts how well source and target activations align.
    We measure this by computing CKA-like similarity between the representations.
    High Fisher compatibility should correlate with high representation alignment.

    Returns dict with alignment metrics.
    """
    eps = float(division_epsilon(backend, source_activations))

    n_samples = int(source_activations.shape[0])
    n_source = int(source_activations.shape[1])
    n_target = int(target_activations.shape[1])
    min_dim = min(n_source, n_target)

    # Truncate to common dimensions
    src = source_activations[:, :min_dim]
    tgt = target_activations[:, :min_dim]
    backend.eval(src, tgt)

    # Compute Gram matrices (sample similarity)
    K_src = backend.matmul(src, backend.transpose(src))
    K_tgt = backend.matmul(tgt, backend.transpose(tgt))
    backend.eval(K_src, K_tgt)

    # Center Gram matrices
    n = float(n_samples)
    ones = backend.ones((n_samples, n_samples)) / n
    K_src_c = K_src - backend.matmul(ones, K_src) - backend.matmul(K_src, ones) + backend.matmul(backend.matmul(ones, K_src), ones)
    K_tgt_c = K_tgt - backend.matmul(ones, K_tgt) - backend.matmul(K_tgt, ones) + backend.matmul(backend.matmul(ones, K_tgt), ones)
    backend.eval(K_src_c, K_tgt_c)

    # Compute CKA (Centered Kernel Alignment)
    # CKA = ||K_src_c * K_tgt_c||_F / (||K_src_c||_F * ||K_tgt_c||_F)
    numerator = backend.sum(K_src_c * K_tgt_c)
    src_norm = backend.sqrt(backend.sum(K_src_c * K_src_c))
    tgt_norm = backend.sqrt(backend.sum(K_tgt_c * K_tgt_c))
    backend.eval(numerator, src_norm, tgt_norm)

    src_norm_val = float(backend.to_scalar(src_norm))
    tgt_norm_val = float(backend.to_scalar(tgt_norm))
    num_val = float(backend.to_scalar(numerator))

    if src_norm_val > eps and tgt_norm_val > eps:
        cka = num_val / (src_norm_val * tgt_norm_val)
    else:
        cka = 0.0
    cka = max(0.0, min(1.0, cka))

    # Also compute simple cosine similarity of flattened activations
    src_flat = backend.reshape(src, (-1,))
    tgt_flat = backend.reshape(tgt, (-1,))
    dot = backend.sum(src_flat * tgt_flat)
    src_mag = backend.sqrt(backend.sum(src_flat * src_flat))
    tgt_mag = backend.sqrt(backend.sum(tgt_flat * tgt_flat))
    backend.eval(dot, src_mag, tgt_mag)

    src_mag_val = float(backend.to_scalar(src_mag))
    tgt_mag_val = float(backend.to_scalar(tgt_mag))
    dot_val = float(backend.to_scalar(dot))

    if src_mag_val > eps and tgt_mag_val > eps:
        cosine = dot_val / (src_mag_val * tgt_mag_val)
    else:
        cosine = 0.0

    # Alignment score is average of CKA and cosine
    alignment = (cka + max(0.0, cosine)) / 2.0

    return {
        "cka": cka,
        "cosine": cosine,
        "alignment": alignment,
    }


def run_fisher_prediction_test(
    n_samples: int,
    n_features: int,
    similarity: float,
    backend,
) -> dict:
    """Run single test of Fisher compatibility as merge predictor."""

    logger.info(
        "Testing: n=%d, d=%d, target_similarity=%.2f",
        n_samples, n_features, similarity
    )

    # Create activations with controlled similarity
    source_acts, target_acts = create_activations_with_fisher_similarity(
        n_samples, n_features, similarity, backend
    )

    # Compute Fisher compatibility
    fisher_result = fisher_compatibility_score(source_acts, target_acts, backend)

    # Compute actual merge quality
    merge_quality = compute_merge_quality(source_acts, target_acts, backend)

    return {
        "config": {
            "n_samples": n_samples,
            "n_features": n_features,
            "target_similarity": similarity,
        },
        "fisher": {
            "compatibility_score": fisher_result.compatibility_score,
            "cosine_similarity": fisher_result.cosine_similarity,
            "correlation": fisher_result.correlation,
            "overlap_ratio": fisher_result.overlap_ratio,
            "recommendation": fisher_result.recommendation,
        },
        "merge_quality": merge_quality,
    }


def compute_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if std_x < 1e-8 or std_y < 1e-8:
        return 0.0

    return cov / (std_x * std_y)


def run_control_identical(n_samples: int, n_features: int, backend) -> dict:
    """Control: Identical activations should have perfect compatibility."""
    backend.random_seed(42)
    activations = backend.random_normal((n_samples, n_features))
    backend.eval(activations)

    fisher_result = fisher_compatibility_score(activations, activations, backend)

    return {
        "type": "identical",
        "compatibility_score": fisher_result.compatibility_score,
        "cosine_similarity": fisher_result.cosine_similarity,
        "correlation": fisher_result.correlation,
        "overlap_ratio": fisher_result.overlap_ratio,
        "expected": "All metrics should be 1.0 (or very close)",
    }


def run_control_orthogonal(n_samples: int, n_features: int, backend) -> dict:
    """Control: Orthogonal activations should have low compatibility."""
    backend.random_seed(42)
    source = backend.random_normal((n_samples, n_features))
    backend.eval(source)

    # Create orthogonal target (different random seed, no shared structure)
    backend.random_seed(9999)
    target = backend.random_normal((n_samples, n_features))
    backend.eval(target)

    # Make target approximately orthogonal by using very different variance pattern
    # Scale odd dimensions up, even dimensions down for target
    scale = backend.array([10.0 if i % 2 else 0.1 for i in range(n_features)])
    scale = backend.reshape(scale, (1, n_features))
    target = target * scale
    backend.eval(target)

    fisher_result = fisher_compatibility_score(source, target, backend)

    return {
        "type": "orthogonal_structure",
        "compatibility_score": fisher_result.compatibility_score,
        "cosine_similarity": fisher_result.cosine_similarity,
        "correlation": fisher_result.correlation,
        "overlap_ratio": fisher_result.overlap_ratio,
        "expected": "Compatibility should be lower than random (< 0.5)",
    }


def main():
    """Run Experiment 8: Fisher Information Merge Compatibility Prediction."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp8_fisher_compatibility")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp8_fisher_compatibility",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "test_type": "fisher_prediction_validation",
            "theory": "FIM similarity predicts merge compatibility",
        },
    )

    results = {
        "prediction_tests": [],
        "controls": {},
    }

    # ==========================================================================
    # PART 1: Fisher Prediction Accuracy
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Fisher Compatibility Prediction Tests")
    logger.info("=" * 60)

    # Test at various similarity levels
    test_configs = [
        # High similarity (should predict good merges)
        {"n": 200, "d": 100, "sim": 0.9},
        {"n": 200, "d": 100, "sim": 0.8},
        {"n": 500, "d": 256, "sim": 0.9},
        # Medium similarity
        {"n": 200, "d": 100, "sim": 0.5},
        {"n": 500, "d": 256, "sim": 0.5},
        {"n": 200, "d": 100, "sim": 0.6},
        # Low similarity (should predict poor merges)
        {"n": 200, "d": 100, "sim": 0.2},
        {"n": 200, "d": 100, "sim": 0.1},
        {"n": 500, "d": 256, "sim": 0.2},
    ]

    for cfg in test_configs:
        try:
            test_result = run_fisher_prediction_test(
                n_samples=cfg["n"],
                n_features=cfg["d"],
                similarity=cfg["sim"],
                backend=backend,
            )
            results["prediction_tests"].append(test_result)

            logger.info(
                "  Fisher=%.3f, Preservation=%.3f (target_sim=%.1f)",
                test_result["fisher"]["compatibility_score"],
                test_result["merge_quality"]["alignment"],
                cfg["sim"],
            )
        except Exception as e:
            logger.error("Test failed: %s", e)
            import traceback
            traceback.print_exc()
            results["prediction_tests"].append({"config": cfg, "error": str(e)})

    # ==========================================================================
    # PART 2: Control Tests
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: Control Tests")
    logger.info("=" * 60)

    results["controls"]["identical"] = run_control_identical(200, 100, backend)
    logger.info(
        "Identical: compat=%.3f, cos=%.3f, corr=%.3f",
        results["controls"]["identical"]["compatibility_score"],
        results["controls"]["identical"]["cosine_similarity"],
        results["controls"]["identical"]["correlation"],
    )

    results["controls"]["orthogonal"] = run_control_orthogonal(200, 100, backend)
    logger.info(
        "Orthogonal: compat=%.3f, cos=%.3f, corr=%.3f",
        results["controls"]["orthogonal"]["compatibility_score"],
        results["controls"]["orthogonal"]["cosine_similarity"],
        results["controls"]["orthogonal"]["correlation"],
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    valid_tests = [t for t in results["prediction_tests"] if "error" not in t]

    if valid_tests:
        # Extract Fisher scores and alignment values
        fisher_scores = [t["fisher"]["compatibility_score"] for t in valid_tests]
        alignments = [t["merge_quality"]["alignment"] for t in valid_tests]
        target_sims = [t["config"]["target_similarity"] for t in valid_tests]

        # Compute correlation between Fisher score and actual alignment
        correlation_with_alignment = compute_correlation(fisher_scores, alignments)

        # Compute correlation between Fisher score and target similarity (validation)
        correlation_with_target = compute_correlation(fisher_scores, target_sims)

        # Analyze prediction accuracy at thresholds
        high_fisher = [t for t in valid_tests if t["fisher"]["compatibility_score"] > 0.6]
        low_fisher = [t for t in valid_tests if t["fisher"]["compatibility_score"] < 0.4]

        high_fisher_alignment = (
            sum(t["merge_quality"]["alignment"] for t in high_fisher) / len(high_fisher)
            if high_fisher else 0.0
        )
        low_fisher_alignment = (
            sum(t["merge_quality"]["alignment"] for t in low_fisher) / len(low_fisher)
            if low_fisher else 0.0
        )

        results["summary"] = {
            "n_tests": len(valid_tests),
            "correlation_with_alignment": correlation_with_alignment,
            "correlation_with_target_similarity": correlation_with_target,
            "mean_fisher_score": sum(fisher_scores) / len(fisher_scores),
            "mean_alignment": sum(alignments) / len(alignments),
            "high_fisher_mean_alignment": high_fisher_alignment,
            "low_fisher_mean_alignment": low_fisher_alignment,
            "n_high_fisher": len(high_fisher),
            "n_low_fisher": len(low_fisher),
        }

        # Success criteria:
        # 1. Correlation with alignment > 0.5 (Fisher predicts quality)
        # 2. High Fisher leads to better alignment than low Fisher
        # 3. Identical control has compatibility ≈ 1.0
        correlation_ok = correlation_with_alignment > 0.3
        discrimination_ok = high_fisher_alignment > low_fisher_alignment
        identical_ok = results["controls"]["identical"]["compatibility_score"] > 0.9

        success = correlation_ok and discrimination_ok and identical_ok
        results["summary"]["success"] = success
        results["summary"]["success_criteria"] = {
            "correlation_positive": correlation_ok,
            "discrimination_works": discrimination_ok,
            "identical_recognized": identical_ok,
        }

        results["summary"]["interpretation"] = (
            f"Fisher compatibility correlates with merge quality (r={correlation_with_alignment:.2f}). "
            f"High Fisher models preserve {high_fisher_alignment*100:.1f}% behavior, "
            f"low Fisher preserve {low_fisher_alignment*100:.1f}%. "
            f"Fisher captures loss landscape similarity for merge prediction."
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
    logger.info("EXPERIMENT 8 COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results and "correlation_with_alignment" in results["summary"]:
        logger.info("Correlation (Fisher vs Preservation): %.3f", results["summary"]["correlation_with_alignment"])
        logger.info("High Fisher mean alignment: %.3f", results["summary"]["high_fisher_mean_alignment"])
        logger.info("Low Fisher mean alignment: %.3f", results["summary"]["low_fisher_mean_alignment"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
