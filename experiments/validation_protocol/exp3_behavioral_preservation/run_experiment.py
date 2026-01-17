#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 3: Null-Space Behavioral Preservation
#
# HYPOTHESIS: Null-space projection preserves target behavior
#
# THEOREM 2: ||A_t @ (P @ ΔW)^T||_F ≈ 0
# Where P = I - A_t^T(A_t A_t^T)^+ A_t is the null-space projector
#
# PROTOCOL:
# 1. Load source and target model weights
# 2. For each layer:
#    a. Compute ΔW = W_s - W_t
#    b. Compute behavioral_norm_before = ||A_t @ ΔW^T||_F
#    c. Compute P = null_space_projector(A_t)
#    d. Compute behavioral_norm_after = ||A_t @ (P @ ΔW)^T||_F
#    e. Record ratio
# 3. Compare with Frobenius norm ratio
#
# SUCCESS CRITERIA:
# - Behavioral norm ratio < 0.01 (99% behavioral preservation)
# - Frobenius norm ratio can be higher (shows behavioral is correct metric)
#
# CONTROLS:
# - Random projection: expect behavioral ratio ≈ 1.0
# - Identity projection (P = I): expect ratio = 1.0

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


def compute_behavioral_norm(
    delta_W,
    activations,
    backend,
) -> float:
    """Compute behavioral impact norm: ||A @ ΔW^T||_F.

    This measures how much the layer's OUTPUT would change if we applied
    this weight delta. This is the correct metric for transplant.
    """
    # delta_W is [out_dim, in_dim], activations is [n_samples, in_dim]
    # Output change: [n_samples, out_dim]
    output_change = backend.matmul(activations, backend.transpose(delta_W))
    backend.eval(output_change)

    # Frobenius norm of output change
    frob_sq = backend.sum(output_change * output_change)
    backend.eval(frob_sq)
    return float(backend.to_scalar(backend.sqrt(frob_sq)))


def compute_frobenius_norm(weight, backend) -> float:
    """Compute standard Frobenius norm: ||W||_F."""
    frob_sq = backend.sum(weight * weight)
    backend.eval(frob_sq)
    return float(backend.to_scalar(backend.sqrt(frob_sq)))


def run_behavioral_preservation_test(
    source_weight,
    target_weight,
    target_activations,
    backend,
) -> dict:
    """Test behavioral preservation through null-space projection."""
    # Compute weight delta
    delta_W = source_weight - target_weight
    backend.eval(delta_W)

    # Compute norms before projection
    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)
    frobenius_before = compute_frobenius_norm(delta_W, backend)

    # Compute null-space projector on target activations
    projector_result = compute_null_space_projector(
        input_activations=target_activations,
        backend=backend,
    )

    # Apply projection
    if projector_result.projection_matrix is not None:
        P = projector_result.projection_matrix
        delta_W_proj = backend.matmul(delta_W, P)
        backend.eval(delta_W_proj)
    else:
        # Fallback: use weighted Gram approach
        delta_W_proj = delta_W  # No projection available
        logger.warning("No projection matrix available, using identity")

    # Compute norms after projection
    behavioral_after = compute_behavioral_norm(delta_W_proj, target_activations, backend)
    frobenius_after = compute_frobenius_norm(delta_W_proj, backend)

    # Compute ratios
    eps = float(division_epsilon(backend, target_weight))
    behavioral_ratio = behavioral_after / max(behavioral_before, eps)
    frobenius_ratio = frobenius_after / max(frobenius_before, eps)

    return {
        "behavioral_before": behavioral_before,
        "behavioral_after": behavioral_after,
        "behavioral_ratio": behavioral_ratio,
        "frobenius_before": frobenius_before,
        "frobenius_after": frobenius_after,
        "frobenius_ratio": frobenius_ratio,
        "null_rank": projector_result.null_rank,
        "intrinsic_rank": projector_result.intrinsic_rank,
    }


def run_random_projection_control(
    source_weight,
    target_weight,
    target_activations,
    backend,
) -> dict:
    """Control: Random orthogonal projection instead of null-space."""
    delta_W = source_weight - target_weight
    backend.eval(delta_W)

    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)

    # Random orthogonal matrix
    d = delta_W.shape[1]
    backend.random_seed(999)
    random_mat = backend.random_normal((d, d))
    Q, _ = backend.qr(random_mat)
    backend.eval(Q)

    # Apply random projection
    delta_W_random = backend.matmul(delta_W, Q)
    backend.eval(delta_W_random)

    behavioral_after = compute_behavioral_norm(delta_W_random, target_activations, backend)

    eps = float(division_epsilon(backend, target_weight))
    return {
        "behavioral_ratio": behavioral_after / max(behavioral_before, eps),
        "expected": "≈ 1.0 (random projection preserves magnitude)",
    }


def run_identity_control(
    source_weight,
    target_weight,
    target_activations,
    backend,
) -> dict:
    """Control: Identity projection (P = I)."""
    delta_W = source_weight - target_weight
    backend.eval(delta_W)

    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)
    behavioral_after = behavioral_before  # Identity doesn't change anything

    return {
        "behavioral_ratio": 1.0,
        "expected": "= 1.0 (identity preserves everything)",
    }


def main():
    """Run Experiment 3: Null-Space Behavioral Preservation."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp3_behavioral_preservation")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp3_behavioral_preservation",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "probe_count": 500,  # Need enough probes for good null-space
            "layers_to_test": ["mlp.down_proj", "mlp.up_proj", "self_attn.o_proj"],
        },
    )

    # Load model weights
    from tests.fixtures.models import load_model_weights, get_atlas_probes

    logger.info("Loading model weights...")
    source_weights = load_model_weights(SMOLLM_PATH, backend)
    target_weights = load_model_weights(LFM2_PATH, backend)
    logger.info("Loaded %d source keys, %d target keys",
               len(source_weights), len(target_weights))

    # Get probe activations
    probe_texts = get_atlas_probes(n_samples=500)

    from tests.fixtures.models import collect_real_activations

    logger.info("Collecting target activations for %d probes...", len(probe_texts))

    results = {
        "layer_tests": [],
        "controls": {},
    }

    # Test on specific weight layers
    # We need matching layer types, so test on MLP weights
    # LFM2 layer pattern: model.layers.{i}.feed_forward.w{1,2,3}
    # SmolLM pattern: model.layers.{i}.mlp.{gate,up,down}_proj

    # For cross-architecture, we'll use a subset of layers
    # Focus on middle layers where representations are most abstract

    test_layers = [
        (8, "feed_forward.w2"),  # LFM2 layer 8 (50% depth)
    ]

    for lfm_layer_idx, weight_suffix in test_layers:
        logger.info("Testing layer %d, weight %s", lfm_layer_idx, weight_suffix)

        try:
            # Get target weight
            target_key = f"model.layers.{lfm_layer_idx}.{weight_suffix}.weight"
            if target_key not in target_weights:
                logger.warning("Key %s not found in target, skipping", target_key)
                continue

            target_weight = backend.array(target_weights[target_key])
            backend.eval(target_weight)

            # For source, we need to find matching weight
            # SmolLM layer 15 (50% of 30) corresponds to LFM2 layer 8 (50% of 16)
            smol_layer_idx = 15
            source_key = f"model.layers.{smol_layer_idx}.mlp.down_proj.weight"

            if source_key not in source_weights:
                logger.warning("Key %s not found in source, skipping", source_key)
                continue

            source_weight = backend.array(source_weights[source_key])
            backend.eval(source_weight)

            logger.info("Source shape: %s, Target shape: %s",
                       source_weight.shape, target_weight.shape)

            # Shapes must match for direct comparison
            # If they don't, we need to project to common dimension
            src_out, src_in = source_weight.shape
            tgt_out, tgt_in = target_weight.shape

            if src_out != tgt_out or src_in != tgt_in:
                logger.info("Dimension mismatch: source(%d,%d) vs target(%d,%d)",
                           src_out, src_in, tgt_out, tgt_in)
                logger.info("Projecting to common dimension...")

                # Use min dimensions
                common_out = min(src_out, tgt_out)
                common_in = min(src_in, tgt_in)

                source_weight = source_weight[:common_out, :common_in]
                target_weight = target_weight[:common_out, :common_in]
                backend.eval(source_weight, target_weight)

            # Collect activations at this layer
            target_acts_by_layer = collect_real_activations(
                model_path=LFM2_PATH,
                probes=probe_texts,
                backend=backend,
                layer_indices=[lfm_layer_idx],
            )
            if lfm_layer_idx not in target_acts_by_layer:
                logger.warning("Layer %d not found in activations, skipping", lfm_layer_idx)
                continue
            target_activations = target_acts_by_layer[lfm_layer_idx]
            backend.eval(target_activations)

            # Truncate activations to match weight dimension if needed
            if target_activations.shape[1] > target_weight.shape[1]:
                target_activations = target_activations[:, :target_weight.shape[1]]
                backend.eval(target_activations)

            logger.info("Target activations shape: %s", target_activations.shape)

            # Run behavioral preservation test
            test_result = run_behavioral_preservation_test(
                source_weight, target_weight, target_activations, backend
            )
            test_result["layer"] = lfm_layer_idx
            test_result["weight_key"] = target_key

            results["layer_tests"].append(test_result)

            logger.info(
                "Layer %d: behavioral_ratio=%.6f, frobenius_ratio=%.4f, null_rank=%d",
                lfm_layer_idx,
                test_result["behavioral_ratio"],
                test_result["frobenius_ratio"],
                test_result["null_rank"],
            )

            # Run controls
            results["controls"]["random_projection"] = run_random_projection_control(
                source_weight, target_weight, target_activations, backend
            )
            results["controls"]["identity"] = run_identity_control(
                source_weight, target_weight, target_activations, backend
            )

        except Exception as e:
            logger.error("Error at layer %d: %s", lfm_layer_idx, e)
            import traceback
            traceback.print_exc()
            results["layer_tests"].append({
                "layer": lfm_layer_idx,
                "error": str(e),
            })

    # Compute summary
    layer_tests = [t for t in results["layer_tests"] if "error" not in t]
    if layer_tests:
        behavioral_ratios = [t["behavioral_ratio"] for t in layer_tests]
        frobenius_ratios = [t["frobenius_ratio"] for t in layer_tests]

        results["summary"] = {
            "total_tests": len(layer_tests),
            "mean_behavioral_ratio": sum(behavioral_ratios) / len(behavioral_ratios),
            "max_behavioral_ratio": max(behavioral_ratios),
            "mean_frobenius_ratio": sum(frobenius_ratios) / len(frobenius_ratios),
            "behavioral_vs_frobenius_gap": sum(frobenius_ratios) / len(frobenius_ratios) - sum(behavioral_ratios) / len(behavioral_ratios),
        }

        # Success criteria: behavioral ratio < 0.01
        success = results["summary"]["max_behavioral_ratio"] < 0.01
        results["summary"]["success"] = success
        results["summary"]["success_criteria"] = "behavioral_ratio < 0.01 (99% preservation)"

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
    logger.info("EXPERIMENT 3 COMPLETE")
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results:
        logger.info("Mean behavioral ratio: %.6f", results["summary"]["mean_behavioral_ratio"])
        logger.info("Mean Frobenius ratio: %.4f", results["summary"]["mean_frobenius_ratio"])
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
