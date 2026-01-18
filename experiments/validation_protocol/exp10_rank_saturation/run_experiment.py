#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 10: Rank Saturation Curve
#
# QUESTION: When do probes stop spanning new dimensions?
#
# THESIS: The probe count isn't a magic number - it's emergent.
#         Stream probes until numerical rank saturates.
#         Saturation = the model tells us it's fully mapped.
#
# PROTOCOL:
#   1. Stream probes in batches (100 at a time)
#   2. After each batch, compute numerical_rank(A)
#   3. Track rank vs n_probes
#   4. Saturation = when delta_rank < 1 for consecutive batches
#
# SUCCESS CRITERIA:
#   - Clear saturation point exists (rank plateaus)
#   - Saturation point < hidden_dim (manifold is compressible)
#   - Different layers may saturate at different points

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


def compute_numerical_rank(activations, backend, eps):
    """
    Compute numerical rank of activation matrix.

    rank = count of singular values > sigma_max * sqrt(eps)
    """
    b = backend

    # Cast to float32 for SVD (bfloat16 not supported)
    # This is precision-increasing, not information-altering
    import mlx.core as mx
    acts_f32 = b.astype(activations, mx.float32)
    b.eval(acts_f32)

    # SVD of activations
    # A = U @ S @ Vh, shape [n_probes, hidden_dim]
    U, S, Vh = b.svd(acts_f32, full_matrices=False)
    b.eval(S)

    # Get singular values as list
    s_values = b.tolist(S)
    s_max = max(s_values) if s_values else 0.0

    # Threshold: sigma_max * sqrt(eps)
    threshold = s_max * sqrt_scalar(eps, b)

    # Count values above threshold
    rank = sum(1 for s in s_values if s > threshold)

    return rank, s_values


def find_saturation_point(rank_curve: list[dict]) -> dict:
    """
    Find the saturation point in the rank curve.

    Saturation = when delta_rank < 1 for 2 consecutive batches.
    """
    if len(rank_curve) < 3:
        return {"saturated": False, "reason": "insufficient data"}

    # Compute deltas
    for i in range(1, len(rank_curve)):
        rank_curve[i]["delta_rank"] = (
            rank_curve[i]["rank"] - rank_curve[i-1]["rank"]
        )

    # Find first point where delta < 1 for 2 consecutive batches
    for i in range(2, len(rank_curve)):
        if (rank_curve[i]["delta_rank"] < 1 and
            rank_curve[i-1]["delta_rank"] < 1):
            return {
                "saturated": True,
                "saturation_index": i - 1,
                "saturation_n_probes": rank_curve[i-1]["n_probes"],
                "saturation_rank": rank_curve[i-1]["rank"],
            }

    return {"saturated": False, "reason": "no saturation reached"}


def main():
    """Run Experiment 10: Rank Saturation Curve."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp10_rank_saturation")
    backend = get_default_backend()

    # Setup experiment
    config = setup_experiment(
        name="exp10_rank_saturation",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "batch_size": 100,
            "max_probes": 4596,  # Full atlas
            "layers_to_test": ["25%", "50%", "75%"],
        },
    )

    # Get ALL probe texts from atlas
    from tests.fixtures.models import get_atlas_probes
    all_probes = get_atlas_probes(n_samples=4596)
    logger.info("Total probes available: %d", len(all_probes))

    # Get machine epsilon
    test_acts = collect_activations(SMOLLM_PATH, all_probes[:10], 15, backend)
    eps = float(machine_epsilon(backend, test_acts))
    sqrt_eps = sqrt_scalar(eps, backend)
    logger.info("Machine epsilon: %.2e, sqrt(eps): %.2e", eps, sqrt_eps)

    results = {
        "precision": {"eps": eps, "sqrt_eps": sqrt_eps},
        "saturation_curves": {},
    }

    # Layer mappings
    smol_layers = [7, 15, 22]  # 25%, 50%, 75%
    lfm_layers = [4, 8, 12]
    depth_names = ["25%", "50%", "75%"]

    batch_size = 100

    for depth_name, smol_layer, lfm_layer in zip(depth_names, smol_layers, lfm_layers):
        logger.info("=" * 60)
        logger.info("Measuring saturation for depth %s", depth_name)

        source_curve = []
        target_curve = []

        # Stream probes in batches
        for batch_end in range(batch_size, len(all_probes) + 1, batch_size):
            probes_so_far = all_probes[:batch_end]

            try:
                # Collect activations for all probes so far
                source_acts = collect_activations(
                    SMOLLM_PATH, probes_so_far, smol_layer, backend
                )
                target_acts = collect_activations(
                    LFM2_PATH, probes_so_far, lfm_layer, backend
                )
                backend.eval(source_acts, target_acts)

                # Compute numerical rank
                source_rank, source_svs = compute_numerical_rank(source_acts, backend, eps)
                target_rank, target_svs = compute_numerical_rank(target_acts, backend, eps)

                source_curve.append({
                    "n_probes": batch_end,
                    "rank": source_rank,
                    "hidden_dim": source_acts.shape[1],
                    "top_5_svs": source_svs[:5] if len(source_svs) >= 5 else source_svs,
                })

                target_curve.append({
                    "n_probes": batch_end,
                    "rank": target_rank,
                    "hidden_dim": target_acts.shape[1],
                    "top_5_svs": target_svs[:5] if len(target_svs) >= 5 else target_svs,
                })

                logger.info(
                    "n=%d: source_rank=%d/%d, target_rank=%d/%d",
                    batch_end,
                    source_rank, source_acts.shape[1],
                    target_rank, target_acts.shape[1],
                )

                # Early stopping: if both have saturated for 3 consecutive batches
                if len(source_curve) >= 4:
                    recent_source_deltas = [
                        source_curve[i]["rank"] - source_curve[i-1]["rank"]
                        for i in range(-3, 0)
                    ]
                    recent_target_deltas = [
                        target_curve[i]["rank"] - target_curve[i-1]["rank"]
                        for i in range(-3, 0)
                    ]

                    if all(d < 1 for d in recent_source_deltas) and all(d < 1 for d in recent_target_deltas):
                        logger.info("Both source and target saturated. Stopping early.")
                        break

            except Exception as e:
                logger.error("Error at batch_end=%d: %s", batch_end, e)
                import traceback
                traceback.print_exc()
                break

        # Find saturation points
        source_saturation = find_saturation_point(source_curve)
        target_saturation = find_saturation_point(target_curve)

        results["saturation_curves"][depth_name] = {
            "source": {
                "model": "SmolLM-135M",
                "layer": smol_layer,
                "curve": source_curve,
                "saturation": source_saturation,
            },
            "target": {
                "model": "LFM2-350M",
                "layer": lfm_layer,
                "curve": target_curve,
                "saturation": target_saturation,
            },
        }

        logger.info("Source saturation: %s", source_saturation)
        logger.info("Target saturation: %s", target_saturation)

    # Summary
    logger.info("=" * 60)
    logger.info("COMPUTING SUMMARY")

    summary = {
        "layers_tested": len(depth_names),
        "source_model": "SmolLM-135M",
        "target_model": "LFM2-350M",
        "saturation_points": {},
    }

    for depth_name in depth_names:
        curve_data = results["saturation_curves"].get(depth_name, {})
        source_sat = curve_data.get("source", {}).get("saturation", {})
        target_sat = curve_data.get("target", {}).get("saturation", {})

        summary["saturation_points"][depth_name] = {
            "source_saturated": source_sat.get("saturated", False),
            "source_n_probes": source_sat.get("saturation_n_probes"),
            "source_rank": source_sat.get("saturation_rank"),
            "target_saturated": target_sat.get("saturated", False),
            "target_n_probes": target_sat.get("saturation_n_probes"),
            "target_rank": target_sat.get("saturation_rank"),
        }

        if source_sat.get("saturated"):
            logger.info(
                "Depth %s SOURCE: saturated at n=%d with rank=%d",
                depth_name,
                source_sat["saturation_n_probes"],
                source_sat["saturation_rank"],
            )
        if target_sat.get("saturated"):
            logger.info(
                "Depth %s TARGET: saturated at n=%d with rank=%d",
                depth_name,
                target_sat["saturation_n_probes"],
                target_sat["saturation_rank"],
            )

    results["summary"] = summary

    duration = time.perf_counter() - start_time

    # Save results
    experiment_result = ExperimentResult(
        config=config,
        metrics=summary,
        raw_data=results,
        duration_seconds=duration,
        success=True,  # Measurement experiment - always succeeds if it runs
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 10 COMPLETE")
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
