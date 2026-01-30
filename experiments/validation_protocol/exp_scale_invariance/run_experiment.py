#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment: Scale Invariance Within Model Family
#
# HYPOTHESIS: Geometric structure is preserved across model scales
#             within the same architecture family.
#
# TEST: LFM2 family (350M, 700M, 1.2B)
#       All pairwise CKA >= 0.90 after alignment
#
# RATIONALE: If geometric structure is truly invariant, it should be
#            preserved not just across architectures, but also across
#            scales within the same family.

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ScaleComparisonResult:
    """Results for comparing models at different scales."""
    source_model: str
    source_params: str
    target_model: str
    target_params: str
    raw_cka: float
    aligned_cka: float
    n_layers_tested: int
    condition_number: float


def run_scale_invariance_test(
    models: list[str],
    n_prompts: int = 50,
) -> dict[str, Any]:
    """Run scale invariance test across model family.

    Args:
        models: List of model paths (should be same family, different sizes)
        n_prompts: Number of prompts for activation extraction.

    Returns:
        Dict with results for all pairwise comparisons.
    """
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.domain.geometry.gram_aligner import GramAligner
    from tests.fixtures.models import collect_real_activations

    initialize_default_backend()
    backend = get_default_backend()
    backend.random_seed(42)

    # Simple test prompts
    prompts = [
        "What is the capital of France?",
        "How many days are in a week?",
        "What is 2 + 2?",
        "The quick brown fox jumps over the",
        "Once upon a time there was a",
        "What color is the sky?",
        "How does photosynthesis work?",
        "Explain the theory of evolution.",
        "What is machine learning?",
        "Define artificial intelligence.",
    ] * (n_prompts // 10 + 1)
    prompts = prompts[:n_prompts]

    results = []

    # Test all pairs
    for i, source_path in enumerate(models):
        for target_path in models[i+1:]:
            source_path_obj = Path(source_path)
            target_path_obj = Path(target_path)

            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {source_path_obj.name} vs {target_path_obj.name}")
            logger.info(f"{'='*60}")

            try:
                # Determine layer indices (test middle layer)
                config_path = source_path_obj / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    n_layers = config.get("num_hidden_layers", 16)
                    layer_indices = [n_layers // 2]
                else:
                    layer_indices = [8]

                # Collect activations
                source_acts_by_layer = collect_real_activations(
                    model_path=source_path_obj,
                    probes=prompts,
                    backend=backend,
                    layer_indices=layer_indices,
                )

                target_acts_by_layer = collect_real_activations(
                    model_path=target_path_obj,
                    probes=prompts,
                    backend=backend,
                    layer_indices=layer_indices,
                )

                # Compute alignment
                aligner = GramAligner(backend)
                aligned_ckas = []
                raw_ckas = []
                kappas = []

                for layer_idx in layer_indices:
                    if layer_idx not in source_acts_by_layer or layer_idx not in target_acts_by_layer:
                        continue

                    source_acts = source_acts_by_layer[layer_idx]
                    target_acts = target_acts_by_layer[layer_idx]

                    # Raw CKA
                    raw_cka_result = compute_cka(source_acts, target_acts, backend=backend)
                    raw_ckas.append(raw_cka_result.best)

                    # Aligned CKA
                    result = aligner.find_perfect_alignment(source_acts, target_acts)
                    F = backend.array(result.feature_transform)
                    aligned_source = backend.matmul(source_acts, F)
                    backend.eval(aligned_source)

                    aligned_cka_result = compute_cka(aligned_source, target_acts, backend=backend)
                    aligned_ckas.append(aligned_cka_result.best)
                    kappa = getattr(result, 'gram_condition_number', 1.0)
                    kappas.append(kappa)

                    logger.info(f"Layer {layer_idx}: raw={raw_cka_result.best:.4f}, aligned={aligned_cka_result.best:.4f}")

                import statistics
                mean_aligned = statistics.mean(aligned_ckas) if aligned_ckas else 0.0
                mean_raw = statistics.mean(raw_ckas) if raw_ckas else 0.0
                mean_kappa = statistics.mean(kappas) if kappas else 1.0

                result_data = ScaleComparisonResult(
                    source_model=source_path_obj.name,
                    source_params=source_path_obj.name,
                    target_model=target_path_obj.name,
                    target_params=target_path_obj.name,
                    raw_cka=mean_raw,
                    aligned_cka=mean_aligned,
                    n_layers_tested=len(aligned_ckas),
                    condition_number=mean_kappa,
                )
                results.append(result_data)

                status = "VALIDATED" if mean_aligned >= 0.90 else "NOT VALIDATED"
                logger.info(f"Result: {status} (aligned CKA = {mean_aligned:.4f})")

            except Exception as e:
                logger.error(f"Failed: {e}")
                results.append({
                    "source": source_path,
                    "target": target_path,
                    "error": str(e),
                })

    # Compute overall statistics
    valid_results = [r for r in results if isinstance(r, ScaleComparisonResult)]
    if valid_results:
        mean_aligned_cka = sum(r.aligned_cka for r in valid_results) / len(valid_results)
        min_aligned_cka = min(r.aligned_cka for r in valid_results)
        all_validated = all(r.aligned_cka >= 0.90 for r in valid_results)
    else:
        mean_aligned_cka = 0.0
        min_aligned_cka = 0.0
        all_validated = False

    logger.info(f"\n{'='*60}")
    logger.info("SCALE INVARIANCE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Comparisons: {len(valid_results)}")
    logger.info(f"Mean aligned CKA: {mean_aligned_cka:.4f}")
    logger.info(f"Min aligned CKA: {min_aligned_cka:.4f}")
    logger.info(f"All >= 0.90: {all_validated}")
    logger.info(f"Verdict: {'SCALE INVARIANCE VALIDATED' if all_validated else 'SCALE INVARIANCE NOT VALIDATED'}")

    return {
        "hypothesis": "Scale invariance within model family",
        "success_criterion": "All pairwise CKA >= 0.90 after alignment",
        "models_tested": [Path(m).name for m in models],
        "n_comparisons": len(valid_results),
        "mean_aligned_cka": mean_aligned_cka,
        "min_aligned_cka": min_aligned_cka,
        "all_validated": all_validated,
        "results": [asdict(r) if isinstance(r, ScaleComparisonResult) else r for r in results],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test scale invariance within model family"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to models of same family at different scales",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file",
    )

    args = parser.parse_args()

    results = run_scale_invariance_test(
        models=args.models,
        n_prompts=args.n_prompts,
    )

    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
