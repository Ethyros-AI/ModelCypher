#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 1c: MMLU Domain-Stratified Alignment (Full Statistical Power)
#
# HYPOTHESIS: CKA after alignment reflects shared manifold coverage per domain
#
# This experiment uses MMLU (14k questions, 57 subjects mapped to 9 domains)
# to properly test domain-stratified alignment with n >> d for each domain.
#
# Coverage ratio ρ = n / d_source determines alignment quality:
# - ρ > 1: Overdetermined system (reliable alignment)
# - ρ < 1: Underdetermined system (CKA~1.0 trivially)
#
# SUCCESS CRITERIA:
# - Report aligned CKA per domain (raw measurements)
# - Report condition numbers
# - Domain variance reveals which concepts are universal vs specialized
# - No magic thresholds - let the data speak

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


def load_mmlu_probes() -> dict[str, list[str]]:
    """Load MMLU probes organized by domain."""
    probes_by_domain = {}

    for json_file in MMLU_DIR.glob("mmlu_*.json"):
        domain = json_file.stem.replace("mmlu_", "")
        with open(json_file) as f:
            data = json.load(f)
        probes_by_domain[domain] = [p["text"] for p in data["probes"]]

    return probes_by_domain


def collect_activations(
    model_path: Path,
    probes: list[str],
    layer_idx: int,
    backend,
    max_probes: int | None = None,
):
    """Collect activations from a model.

    max_probes: If set, subsample probes to limit memory usage.
                Should be n > max(d_source, d_target) for valid alignment.
    """
    from tests.fixtures.models import collect_real_activations

    # Limit probes if explicitly requested
    if max_probes is not None and len(probes) > max_probes:
        # Sample uniformly
        step = len(probes) // max_probes
        probes = probes[::step][:max_probes]

    activations_by_layer = collect_real_activations(
        model_path=model_path,
        probes=probes,
        backend=backend,
        layer_indices=[layer_idx],
    )

    if layer_idx not in activations_by_layer:
        raise ValueError(f"Layer {layer_idx} not found")

    return activations_by_layer[layer_idx], len(probes)


def run_alignment_test(source_acts, target_acts, backend) -> dict:
    """Run alignment and compute metrics."""
    aligner = GramAligner(backend)

    # Raw CKA
    raw_cka_result = compute_cka(source_acts, target_acts, backend=backend)
    raw_cka = raw_cka_result.best

    # Alignment
    result = aligner.find_perfect_alignment(source_acts, target_acts)

    # Aligned CKA
    F = backend.array(result.feature_transform)
    aligned_source = backend.matmul(source_acts, F)
    backend.eval(aligned_source)
    aligned_cka_result = compute_cka(aligned_source, target_acts, backend=backend)
    aligned_cka = aligned_cka_result.best

    n, d = source_acts.shape
    coverage_ratio = n / d

    return {
        "raw_cka": raw_cka,
        "aligned_cka": aligned_cka,
        "condition_number": result.gram_condition_number,
        "is_perfect": result.is_perfect,
        "n_probes": n,
        "source_dim": d,
        "target_dim": target_acts.shape[1],
        "coverage_ratio": coverage_ratio,
    }


def main():
    """Run Experiment 1c: MMLU Domain-Stratified Alignment."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp1c_mmlu_domain_alignment")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp1c_mmlu_domain_alignment",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "dataset": "MMLU (cais/mmlu)",
            "total_probes": 14042,
            "domains": 9,
            "theory": "CKA reflects shared manifold coverage with proper statistical power",
        },
    )

    # Load MMLU probes
    logger.info("Loading MMLU probes...")
    probes_by_domain = load_mmlu_probes()

    # Note: coverage ratio computed after we know dimensions
    for domain, probes in sorted(probes_by_domain.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d probes", domain, len(probes))

    results = {
        "domain_tests": {},
        "summary": {},
    }

    # Test at 50% depth
    smol_layer = 15
    lfm_layer = 8

    logger.info("Testing at 50%% depth (SmolLM layer %d, LFM2 layer %d)", smol_layer, lfm_layer)

    # Test each domain
    for domain, probes in sorted(probes_by_domain.items(), key=lambda x: -len(x[1])):
        coverage_ratio = len(probes) / 576

        logger.info("Testing %s (n=%d, ρ=%.2f)...", domain, len(probes), coverage_ratio)

        try:
            # Collect activations
            source_acts, n_used = collect_activations(
                SMOLLM_PATH, probes, smol_layer, backend
            )
            target_acts, _ = collect_activations(
                LFM2_PATH, probes, lfm_layer, backend
            )
            backend.eval(source_acts, target_acts)

            # Run test
            test_result = run_alignment_test(source_acts, target_acts, backend)
            test_result["domain"] = domain
            test_result["original_n_probes"] = len(probes)

            results["domain_tests"][domain] = test_result

            logger.info(
                "  %s: raw=%.4f, aligned=%.6f, κ=%.2e, ρ=%.2f",
                domain,
                test_result["raw_cka"],
                test_result["aligned_cka"],
                test_result["condition_number"],
                test_result["coverage_ratio"],
            )

        except Exception as e:
            logger.error("  Error for %s: %s", domain, e)
            import traceback
            traceback.print_exc()
            results["domain_tests"][domain] = {"error": str(e)}

    # Compute summary
    valid_tests = {k: v for k, v in results["domain_tests"].items() if "error" not in v}

    # Separate overdetermined (ρ > 1) from underdetermined
    overdetermined = {k: v for k, v in valid_tests.items() if v["coverage_ratio"] > 1.0}
    underdetermined = {k: v for k, v in valid_tests.items() if v["coverage_ratio"] <= 1.0}

    if overdetermined:
        od_aligned = [v["aligned_cka"] for v in overdetermined.values()]
        od_raw = [v["raw_cka"] for v in overdetermined.values()]
        od_kappa = [v["condition_number"] for v in overdetermined.values()]

        results["summary"]["overdetermined"] = {
            "n_domains": len(overdetermined),
            "domains": list(overdetermined.keys()),
            "mean_aligned_cka": sum(od_aligned) / len(od_aligned),
            "min_aligned_cka": min(od_aligned),
            "max_aligned_cka": max(od_aligned),
            "mean_raw_cka": sum(od_raw) / len(od_raw),
            "raw_cka_range": max(od_raw) - min(od_raw),
            "max_condition_number": max(od_kappa),
        }

    if underdetermined:
        ud_aligned = [v["aligned_cka"] for v in underdetermined.values()]
        results["summary"]["underdetermined"] = {
            "n_domains": len(underdetermined),
            "domains": list(underdetermined.keys()),
            "mean_aligned_cka": sum(ud_aligned) / len(ud_aligned),
            "note": "These achieve CKA~1.0 trivially (n < d)",
        }

    # Domain ranking
    ranking = sorted(
        [(k, v["aligned_cka"], v["raw_cka"], v["coverage_ratio"])
         for k, v in valid_tests.items()],
        key=lambda x: -x[1]
    )
    results["summary"]["domain_ranking"] = [
        {"domain": d, "aligned_cka": a, "raw_cka": r, "coverage_ratio": c}
        for d, a, r, c in ranking
    ]

    # Key insight: raw CKA shows which domains have similar coordinates already
    # Report sorted by raw CKA (no arbitrary thresholds)
    if valid_tests:
        raw_ckas = sorted(
            [(k, v["raw_cka"]) for k, v in valid_tests.items()],
            key=lambda x: -x[1]
        )
        results["summary"]["domains_by_raw_cka"] = [
            {"domain": k, "raw_cka": r} for k, r in raw_ckas
        ]

    duration = time.perf_counter() - start_time

    # Save
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=len(valid_tests) > 0,
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 1c COMPLETE: MMLU Domain-Stratified Alignment")
    logger.info("=" * 70)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Domains tested: %d (%d overdetermined, %d underdetermined)",
               len(valid_tests), len(overdetermined), len(underdetermined))

    if "overdetermined" in results["summary"]:
        od = results["summary"]["overdetermined"]
        logger.info("")
        logger.info("OVERDETERMINED DOMAINS (ρ > 1, reliable measurements):")
        logger.info("  Mean aligned CKA: %.6f", od["mean_aligned_cka"])
        logger.info("  Range: [%.6f, %.6f]", od["min_aligned_cka"], od["max_aligned_cka"])
        logger.info("  Mean raw CKA: %.4f (range: %.4f)", od["mean_raw_cka"], od["raw_cka_range"])
        logger.info("  Max condition number: %.2e", od["max_condition_number"])

    if "domains_by_raw_cka" in results["summary"]:
        logger.info("")
        logger.info("DOMAINS BY RAW CKA (coordinate similarity):")
        for item in results["summary"]["domains_by_raw_cka"][:5]:  # Top 5
            logger.info("  %s: raw_cka=%.4f", item["domain"], item["raw_cka"])

    logger.info("")
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
