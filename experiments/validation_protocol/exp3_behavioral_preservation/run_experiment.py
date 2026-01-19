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
# 1. Create weight delta ΔW
# 2. Create target activations A_t
# 3. Compute behavioral_norm_before = ||A_t @ ΔW^T||_F
# 4. Compute P = null_space_projector(A_t)
# 5. Compute behavioral_norm_after = ||A_t @ (P @ ΔW)^T||_F
# 6. Record ratio
#
# MEASUREMENTS:
# - behavioral_ratio: After/before ratio (closer to 0 = better preservation)
# - frobenius_ratio: Weight norm ratio (for comparison)
# - null_rank: Dimensions available for knowledge transfer
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


def compute_behavioral_norm(delta_W, activations, backend) -> float:
    """Compute behavioral impact norm: ||A @ ΔW^T||_F.

    This measures how much the layer's OUTPUT would change if we applied
    this weight delta. This is the correct metric for transplant.

    Args:
        delta_W: Weight delta of shape (out_dim, in_dim)
        activations: Target activations of shape (n_samples, in_dim)
    """
    # Output change: activations @ delta_W.T = (n, in_dim) @ (in_dim, out_dim) = (n, out_dim)
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


def apply_null_space_projection(delta_W, projector_result, backend):
    """Apply null-space projection using the projector components.

    The projection is: delta_W_proj = delta_W - (delta_W @ A^T) @ (AA^T)^+ @ A
    where A = weighted_activations and (AA^T)^+ = gram_inv.

    Args:
        delta_W: Weight delta of shape (out_dim, in_dim)
        projector_result: NullSpaceProjector with weighted_activations and gram_inv
        backend: Compute backend

    Returns:
        Projected delta_W of shape (out_dim, in_dim)
    """
    A_weighted = projector_result.weighted_activations
    gram_inv = projector_result.gram_inv
    backend.eval(A_weighted, gram_inv)

    # delta_W_proj = delta_W - (delta_W @ A.T) @ (A @ A.T)^+ @ A
    # Step 1: delta_W @ A.T  [out_dim, in_dim] @ [in_dim, n] = [out_dim, n]
    delta_row = backend.matmul(delta_W, backend.transpose(A_weighted))
    backend.eval(delta_row)

    # Step 2: [out_dim, n] @ [n, n] = [out_dim, n]
    correction = backend.matmul(delta_row, gram_inv)
    backend.eval(correction)

    # Step 3: [out_dim, n] @ [n, in_dim] = [out_dim, in_dim]
    correction = backend.matmul(correction, A_weighted)
    backend.eval(correction)

    # Step 4: Subtract correction
    delta_W_proj = delta_W - correction
    backend.eval(delta_W_proj)

    return delta_W_proj


def run_behavioral_preservation_test(
    delta_W,
    target_activations,
    backend,
) -> dict:
    """Test behavioral preservation through null-space projection.

    Args:
        delta_W: Weight delta of shape (out_dim, in_dim)
        target_activations: Activations of shape (n_samples, in_dim)
    """
    # Compute norms before projection
    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)
    frobenius_before = compute_frobenius_norm(delta_W, backend)

    # Compute null-space projector on target activations
    projector_result = compute_null_space_projector(
        input_activations=target_activations,
        backend=backend,
    )

    # Apply projection to delta_W using the projector components
    # The projection is: delta_W_proj = delta_W - (delta_W @ A^T) @ (AA^T)^+ @ A
    delta_W_proj = apply_null_space_projection(delta_W, projector_result, backend)

    # Compute norms after projection
    behavioral_after = compute_behavioral_norm(delta_W_proj, target_activations, backend)
    frobenius_after = compute_frobenius_norm(delta_W_proj, backend)

    # Compute ratios
    eps = float(division_epsilon(backend, delta_W))
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
        # Note: intrinsic_rank is not directly available in NullSpaceProjector
        # null_rank = in_dim - activation_rank, which serves a similar purpose
    }


def run_random_projection_control(delta_W, target_activations, backend) -> dict:
    """Control: Random orthogonal projection instead of null-space."""
    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)

    # Random orthogonal matrix via QR
    d = delta_W.shape[1]
    backend.random_seed(999)
    random_mat = backend.random_normal((d, d))
    Q, _ = backend.qr(random_mat)
    backend.eval(Q)

    # Apply random projection
    delta_W_random = backend.matmul(delta_W, Q)
    backend.eval(delta_W_random)

    behavioral_after = compute_behavioral_norm(delta_W_random, target_activations, backend)

    eps = float(division_epsilon(backend, delta_W))
    return {
        "behavioral_ratio": behavioral_after / max(behavioral_before, eps),
        "control_type": "random_orthogonal",
    }


def run_identity_control(delta_W, target_activations, backend) -> dict:
    """Control: Identity projection (P = I)."""
    behavioral_before = compute_behavioral_norm(delta_W, target_activations, backend)

    return {
        "behavioral_ratio": 1.0,
        "behavioral_before": behavioral_before,
        "control_type": "identity",
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
            "test_type": "mathematical_validation",
            "theorem": "||A_t @ (P @ ΔW)^T||_F ≈ 0 where P is null-space projector",
        },
    )

    results = {
        "mathematical_tests": [],
        "real_model_tests": [],
        "controls": {},
    }

    # ==========================================================================
    # PART 1: Mathematical Validation with Synthetic Data
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Mathematical Validation (Synthetic Data)")
    logger.info("=" * 60)

    # Test at multiple dimensions and coverage ratios
    test_configs = [
        {"n_samples": 200, "in_dim": 100, "out_dim": 50, "name": "small"},
        {"n_samples": 500, "in_dim": 256, "out_dim": 128, "name": "medium"},
        {"n_samples": 1000, "in_dim": 512, "out_dim": 256, "name": "large"},
    ]

    for cfg in test_configs:
        n = cfg["n_samples"]
        in_dim = cfg["in_dim"]
        out_dim = cfg["out_dim"]
        name = cfg["name"]

        logger.info("Testing %s: n=%d, in_dim=%d, out_dim=%d", name, n, in_dim, out_dim)

        backend.random_seed(42)

        # Create synthetic activations (target model activations)
        target_activations = backend.random_normal((n, in_dim))
        backend.eval(target_activations)

        # Create synthetic weight delta
        delta_W = backend.random_normal((out_dim, in_dim))
        backend.eval(delta_W)

        # Run test
        try:
            test_result = run_behavioral_preservation_test(
                delta_W, target_activations, backend
            )
            test_result["config"] = cfg

            results["mathematical_tests"].append(test_result)

            logger.info(
                "  %s: behavioral_ratio=%.6f, frobenius_ratio=%.4f, null_rank=%d",
                name,
                test_result["behavioral_ratio"],
                test_result["frobenius_ratio"],
                test_result["null_rank"],
            )

            # Run controls
            if name == "medium":  # Only need controls once
                results["controls"]["random_projection"] = run_random_projection_control(
                    delta_W, target_activations, backend
                )
                results["controls"]["identity"] = run_identity_control(
                    delta_W, target_activations, backend
                )
                logger.info("  Controls: random=%.4f, identity=%.4f",
                           results["controls"]["random_projection"]["behavioral_ratio"],
                           results["controls"]["identity"]["behavioral_ratio"])

        except Exception as e:
            logger.error("  Error in %s: %s", name, e)
            import traceback
            traceback.print_exc()
            results["mathematical_tests"].append({"config": cfg, "error": str(e)})

    # ==========================================================================
    # PART 2: Real Model Test (using hidden state dimension)
    # ==========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: Real Model Validation")
    logger.info("=" * 60)

    from tests.fixtures.models import load_model_weights, get_atlas_probes, collect_real_activations

    # Load weights
    source_weights = load_model_weights(SMOLLM_PATH, backend)
    target_weights = load_model_weights(LFM2_PATH, backend)

    # Get probe texts
    probe_texts = get_atlas_probes(n_samples=500)

    # Test using layer norm weights (in_dim = hidden_dim)
    # LFM2: model.layers.8.input_layernorm.weight has shape (1024,)
    # SmolLM: model.layers.15.input_layernorm.weight has shape (576,)
    # These are 1D, so let's use attention output projection instead

    # For a weight where in_dim = hidden_dim, use self_attn output projection
    # LFM2: self_attn.out_proj.weight has shape (hidden_dim, hidden_dim)
    # SmolLM: self_attn.o_proj.weight has shape (hidden_dim, hidden_dim)
    lfm_layer = 8
    smol_layer = 15

    try:
        # Get weights (output projection in attention)
        # Different models use different naming: out_proj vs o_proj
        target_key = f"model.layers.{lfm_layer}.self_attn.out_proj.weight"
        source_key = f"model.layers.{smol_layer}.self_attn.o_proj.weight"

        if target_key in target_weights and source_key in source_weights:
            target_weight = backend.array(target_weights[target_key])
            source_weight = backend.array(source_weights[source_key])
            backend.eval(target_weight, source_weight)

            logger.info("Target o_proj shape: %s", target_weight.shape)
            logger.info("Source o_proj shape: %s", source_weight.shape)

            # Truncate to common dimension
            tgt_out, tgt_in = target_weight.shape
            src_out, src_in = source_weight.shape
            common_out = min(tgt_out, src_out)
            common_in = min(tgt_in, src_in)

            target_weight = target_weight[:common_out, :common_in]
            source_weight = source_weight[:common_out, :common_in]
            backend.eval(target_weight, source_weight)

            delta_W = source_weight - target_weight
            backend.eval(delta_W)

            logger.info("Delta W shape: %s", delta_W.shape)

            # Collect activations
            target_acts_by_layer = collect_real_activations(
                model_path=LFM2_PATH,
                probes=probe_texts,
                backend=backend,
                layer_indices=[lfm_layer],
            )

            if lfm_layer in target_acts_by_layer:
                target_activations = target_acts_by_layer[lfm_layer]
                backend.eval(target_activations)

                # Truncate activations to match weight input dimension
                target_activations = target_activations[:, :common_in]
                backend.eval(target_activations)

                logger.info("Target activations shape: %s", target_activations.shape)

                # Run test
                test_result = run_behavioral_preservation_test(
                    delta_W, target_activations, backend
                )
                test_result["layer"] = lfm_layer
                test_result["weight"] = "self_attn.o_proj"

                results["real_model_tests"].append(test_result)

                logger.info(
                    "Real model: behavioral_ratio=%.6f, frobenius_ratio=%.4f, null_rank=%d",
                    test_result["behavioral_ratio"],
                    test_result["frobenius_ratio"],
                    test_result["null_rank"],
                )
        else:
            logger.warning("O_proj weights not found")

    except Exception as e:
        logger.error("Error in real model test: %s", e)
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # Summary
    # ==========================================================================
    math_tests = [t for t in results["mathematical_tests"] if "error" not in t]
    real_tests = [t for t in results["real_model_tests"] if "error" not in t]

    if math_tests:
        behavioral_ratios = [t["behavioral_ratio"] for t in math_tests]
        frobenius_ratios = [t["frobenius_ratio"] for t in math_tests]

        results["summary"] = {
            "mathematical": {
                "n_tests": len(math_tests),
                "mean_behavioral_ratio": sum(behavioral_ratios) / len(behavioral_ratios),
                "max_behavioral_ratio": max(behavioral_ratios),
                "mean_frobenius_ratio": sum(frobenius_ratios) / len(frobenius_ratios),
            },
        }

        if real_tests:
            real_behavioral = [t["behavioral_ratio"] for t in real_tests]
            results["summary"]["real_model"] = {
                "n_tests": len(real_tests),
                "mean_behavioral_ratio": sum(real_behavioral) / len(real_behavioral),
                "max_behavioral_ratio": max(real_behavioral),
            }

        # Report raw measurements - let the data speak
        synthetic_max = max(behavioral_ratios)
        synthetic_mean = sum(behavioral_ratios) / len(behavioral_ratios)

        real_max = max([t["behavioral_ratio"] for t in real_tests]) if real_tests else 0.0
        real_mean = sum(t["behavioral_ratio"] for t in real_tests) / len(real_tests) if real_tests else 0.0

        results["summary"]["synthetic_max_behavioral_ratio"] = synthetic_max
        results["summary"]["synthetic_mean_behavioral_ratio"] = synthetic_mean
        results["summary"]["real_max_behavioral_ratio"] = real_max
        results["summary"]["real_mean_behavioral_ratio"] = real_mean

        # Success: experiment ran and produced data
        results["summary"]["success"] = len(math_tests) > 0

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
    logger.info("EXPERIMENT 3 COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", experiment_result.success)
    if "summary" in results:
        if "mathematical" in results["summary"]:
            logger.info("Mathematical mean behavioral ratio: %.6f",
                       results["summary"]["mathematical"]["mean_behavioral_ratio"])
        if "real_model" in results["summary"]:
            logger.info("Real model mean behavioral ratio: %.6f",
                       results["summary"]["real_model"]["mean_behavioral_ratio"])
    if "controls" in results:
        logger.info("Control (random projection): %.4f",
                   results["controls"].get("random_projection", {}).get("behavioral_ratio", "N/A"))
        logger.info("Control (identity): %.4f",
                   results["controls"].get("identity", {}).get("behavioral_ratio", "N/A"))
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 60)

    return experiment_result


if __name__ == "__main__":
    main()
