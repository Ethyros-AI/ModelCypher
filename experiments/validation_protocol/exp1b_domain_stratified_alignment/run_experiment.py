#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 1b: Domain-Stratified Alignment Analysis
#
# REFINED HYPOTHESIS: CKA after alignment reflects shared manifold coverage
#
# THEORY:
# - Models learn different regions of the universal manifold based on training data
# - Alignment can only succeed where BOTH models have learned the concept space
# - CKA < 1.0 doesn't falsify the theory - it measures the INTERSECTION
# - Domain-specific CKA should correlate with both models' competence in that domain
#
# PROTOCOL:
# 1. Extract activations by domain (LINGUISTIC, MATHEMATICAL, etc.)
# 2. Compute per-domain CKA before and after alignment
# 3. Compare CKA across domains to see which are "shared manifold"
# 4. Predict: Universal domains (language, logic) → higher CKA
#            Specialized domains → variable CKA based on training overlap
#
# SUCCESS CRITERIA:
# - Report per-domain CKA variance (raw measurement)
# - Report domain ranking by aligned CKA
# - No thresholds - let the data speak

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.domains import AtlasDomain
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


def get_probes_by_domain() -> dict[AtlasDomain, list[str]]:
    """Get probe texts organized by domain."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    all_probes = UnifiedAtlasInventory.all_probes()

    probes_by_domain: dict[AtlasDomain, list[str]] = defaultdict(list)
    for probe in all_probes:
        # Get probe text
        if probe.support_texts:
            text = probe.support_texts[0]
        else:
            text = probe.name
        probes_by_domain[probe.domain].append(text)

    return dict(probes_by_domain)


def collect_domain_activations(
    model_path: Path,
    probes_by_domain: dict[AtlasDomain, list[str]],
    layer_idx: int,
    backend,
) -> dict[AtlasDomain, any]:
    """Collect activations per domain from a model."""
    from tests.fixtures.models import collect_real_activations

    results = {}
    for domain, probes in probes_by_domain.items():
        # Skip domains with insufficient probes for alignment
        # Need n > 1 for meaningful Gram matrix computation
        if len(probes) < 2:
            continue

        activations_by_layer = collect_real_activations(
            model_path=model_path,
            probes=probes,
            backend=backend,
            layer_indices=[layer_idx],
        )

        if layer_idx in activations_by_layer:
            results[domain] = activations_by_layer[layer_idx]

    return results


def run_domain_alignment_test(
    source_acts,
    target_acts,
    backend,
) -> dict:
    """Run alignment and compute metrics for a single domain."""
    aligner = GramAligner(backend)

    # Compute raw CKA
    raw_cka_result = compute_cka(source_acts, target_acts, backend=backend)
    raw_cka = raw_cka_result.best

    # Compute alignment
    result = aligner.find_perfect_alignment(source_acts, target_acts)

    # Apply alignment
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
        "n_probes": source_acts.shape[0],
        "source_dim": source_acts.shape[1],
        "target_dim": target_acts.shape[1],
    }


def main():
    """Run Experiment 1b: Domain-Stratified Alignment."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp1b_domain_stratified_alignment")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp1b_domain_stratified_alignment",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "analysis_type": "domain_stratified",
            "theory": "CKA reflects shared manifold coverage",
        },
    )

    # Get probes organized by domain
    logger.info("Organizing probes by domain...")
    probes_by_domain = get_probes_by_domain()

    for domain, probes in probes_by_domain.items():
        logger.info("  %s: %d probes", domain.value, len(probes))

    results = {
        "domain_tests": {},
        "summary": {},
    }

    # Test at 50% depth (middle layers - most abstract representations)
    smol_layer = 15  # 50% of 30
    lfm_layer = 8    # 50% of 16

    logger.info("Testing at 50%% depth (SmolLM layer %d, LFM2 layer %d)",
               smol_layer, lfm_layer)

    # Collect activations per domain
    logger.info("Collecting source activations by domain...")
    source_acts_by_domain = collect_domain_activations(
        SMOLLM_PATH, probes_by_domain, smol_layer, backend
    )

    logger.info("Collecting target activations by domain...")
    target_acts_by_domain = collect_domain_activations(
        LFM2_PATH, probes_by_domain, lfm_layer, backend
    )

    # Run alignment test per domain
    for domain in source_acts_by_domain:
        if domain not in target_acts_by_domain:
            continue

        source_acts = source_acts_by_domain[domain]
        target_acts = target_acts_by_domain[domain]
        backend.eval(source_acts, target_acts)

        logger.info("Testing domain %s (n=%d)...", domain.value, source_acts.shape[0])

        try:
            domain_result = run_domain_alignment_test(source_acts, target_acts, backend)
            results["domain_tests"][domain.value] = domain_result

            logger.info(
                "  %s: raw_cka=%.4f, aligned_cka=%.6f, κ=%.2e",
                domain.value,
                domain_result["raw_cka"],
                domain_result["aligned_cka"],
                domain_result["condition_number"],
            )

        except Exception as e:
            logger.error("  Error for %s: %s", domain.value, e)
            results["domain_tests"][domain.value] = {"error": str(e)}

    # Compute summary statistics
    valid_tests = {k: v for k, v in results["domain_tests"].items() if "error" not in v}
    if valid_tests:
        aligned_ckas = [v["aligned_cka"] for v in valid_tests.values()]
        raw_ckas = [v["raw_cka"] for v in valid_tests.values()]

        results["summary"] = {
            "n_domains_tested": len(valid_tests),
            "mean_raw_cka": sum(raw_ckas) / len(raw_ckas),
            "mean_aligned_cka": sum(aligned_ckas) / len(aligned_ckas),
            "min_aligned_cka": min(aligned_ckas),
            "max_aligned_cka": max(aligned_ckas),
            "aligned_cka_variance": sum((c - sum(aligned_ckas)/len(aligned_ckas))**2 for c in aligned_ckas) / len(aligned_ckas),
            "aligned_cka_range": max(aligned_ckas) - min(aligned_ckas),
        }

        # Rank domains by aligned CKA (raw data, no interpretation)
        ranked = sorted(valid_tests.items(), key=lambda x: x[1]["aligned_cka"], reverse=True)
        results["summary"]["domain_ranking"] = [
            {"domain": k, "aligned_cka": v["aligned_cka"], "raw_cka": v["raw_cka"], "n_probes": v["n_probes"]}
            for k, v in ranked
        ]

        # Report standard deviation for interpretability
        import math
        results["summary"]["aligned_cka_std"] = math.sqrt(results["summary"]["aligned_cka_variance"])

    duration = time.perf_counter() - start_time

    # Save results
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=len(valid_tests) > 0,
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 1b COMPLETE")
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Domains tested: %d", len(valid_tests))
    if "summary" in results and results["summary"]:
        logger.info("Mean aligned CKA: %.4f (std=%.4f)",
                   results["summary"]["mean_aligned_cka"],
                   results["summary"]["aligned_cka_std"])
        logger.info("CKA range: %.4f (min=%.4f, max=%.4f)",
                   results["summary"]["aligned_cka_range"],
                   results["summary"]["min_aligned_cka"],
                   results["summary"]["max_aligned_cka"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
