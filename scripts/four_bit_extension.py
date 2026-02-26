#!/usr/bin/env python3
"""Experiment 5: 4-bit Extension.

Creates 4-bit quantized versions of the model pairs and runs the RMT gate
experiment to determine if the geometric story changes at 4-bit.

Predictions from the deep dive:
- Higher signal_rank in E_q (more systematic error due to coarser grid)
- More effective per-adapter correction (larger systematic component)
- More stacking rounds needed (more total error to correct)
- MASS spectral ceiling still robust (sigma_max and sigma_k at top of spectrum)

If the gate passes, Experiments 2-4 can be rerun with the 4-bit model paths.

Usage:
    poetry run python scripts/four_bit_extension.py

    # Skip quantization if 4-bit models already exist
    poetry run python scripts/four_bit_extension.py --skip-quantize

    # Custom model
    poetry run python scripts/four_bit_extension.py \
        --fp-model /path/to/fp16 --output-base results/four_bit_extension
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("four_bit_extension")

# Default FP models (same as other experiments)
DEFAULT_FP_MODELS = [
    "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16",
]

# 8-bit models for comparison
DEFAULT_8BIT_MODELS = [
    "results/feasibility_map/20260225T160732Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-8bit-g64-affine",
]

OUTPUT_DIR = Path("results/four_bit_extension")
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10000


def _clear_gpu_cache() -> None:
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the mean."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    observed_mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    alpha = 1.0 - ci
    lo = means[max(0, int(alpha / 2 * n_bootstrap))]
    hi = means[min(n_bootstrap - 1, int((1.0 - alpha / 2) * n_bootstrap))]
    return observed_mean, lo, hi


def _quantize_model(
    fp_path: str, output_dir: Path, bits: int, group_size: int,
) -> Path:
    """Quantize a model to the specified bit width."""
    from modelcypher.cli.composition import get_quantization_service

    model_name = Path(fp_path).name
    q_name = f"{model_name}-{bits}bit-g{group_size}-affine"
    q_path = output_dir / q_name

    if q_path.exists() and (q_path / "model.safetensors").exists():
        logger.info("  4-bit model already exists: %s", q_path)
        return q_path

    logger.info("  Quantizing %s → %s (%d-bit, g%d)", model_name, q_name, bits, group_size)
    service = get_quantization_service()
    result = service.quantize_model(
        model_path=fp_path,
        output_dir=q_path,
        bits=bits,
        group_size=group_size,
        mode="affine",
    )
    logger.info(
        "  Quantized: %d/%d 2D weights, output=%s",
        result.quantized_2d_weights, result.total_2d_weights, q_path,
    )
    return q_path


def _analyze_pair_rmt(
    fp_path: str, q_path: str, backend: Any, adapter: Any,
) -> dict[str, Any]:
    """Run RMT decomposition on a model pair (same as rmt_quantization_error.py)."""
    import mlx.core as mx
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )

    logger.info("  Loading FP model: %s", Path(fp_path).name)
    fp_model, _ = backend.load_model(fp_path)
    fp_weights = _extract_weights(fp_model, adapter)
    del fp_model
    gc.collect()
    _clear_gpu_cache()

    logger.info("  Loading Q model: %s", Path(q_path).name)
    q_model, _ = backend.load_model(q_path)
    q_weights = _extract_weights(q_model, adapter)
    del q_model
    gc.collect()
    _clear_gpu_cache()

    common_keys = sorted(set(fp_weights.keys()) & set(q_weights.keys()))
    logger.info("  Analyzing %d layers...", len(common_keys))

    per_layer: list[dict[str, Any]] = []
    n_svd_failures = 0

    for key in common_keys:
        E = mx.astype(fp_weights[key], mx.float32) - mx.astype(q_weights[key], mx.float32)
        mx.eval(E)
        m, n = int(E.shape[0]), int(E.shape[1])
        frob_norm = float(mx.sqrt(mx.sum(E * E)).item())

        signal_rank = 0
        noise_rank = min(m, n)
        signal_variance_fraction = 0.0
        mp_upper_edge = 0.0
        spectral_norm = 0.0
        svd_success = True

        try:
            S = mx.linalg.svd(E, compute_uv=False, stream=mx.cpu)
            mx.eval(S)
            spectral_norm = float(S[0].item()) if S.shape[0] > 0 else 0.0

            rmt = compute_signal_rank_from_singular_values(
                S, n_samples=m, n_features=n,
                backend=backend, center_correction=True,
            )
            signal_rank = rmt.signal_rank
            noise_rank = rmt.noise_rank
            signal_variance_fraction = rmt.signal_variance_fraction
            mp_upper_edge = rmt.mp_upper_edge
            del S
        except Exception as exc:
            logger.warning("  SVD failed for %s: %s", key, exc)
            svd_success = False
            n_svd_failures += 1

        del E
        gc.collect()

        per_layer.append({
            "layer_key": key,
            "shape": [m, n],
            "signal_rank": signal_rank,
            "noise_rank": noise_rank,
            "signal_variance_fraction": signal_variance_fraction,
            "mp_upper_edge": mp_upper_edge,
            "error_spectral_norm": spectral_norm,
            "error_frobenius_norm": frob_norm,
            "svd_success": svd_success,
        })

        logger.info(
            "    %s [%dx%d]: signal_rank=%d, sv_frac=%.4f, ||E||_2=%.6f",
            key.split(".")[-2], m, n,
            signal_rank, signal_variance_fraction, spectral_norm,
        )

    del fp_weights, q_weights
    gc.collect()
    _clear_gpu_cache()

    return {
        "fp_model": fp_path,
        "q_model": q_path,
        "n_layers": len(common_keys),
        "n_svd_failures": n_svd_failures,
        "per_layer": per_layer,
    }


def _extract_weights(model, adapter) -> dict[str, Any]:
    """Extract dequantized weights from model."""
    base = getattr(model, "model", model)
    weights: dict[str, Any] = {}
    for layer_idx, layer in enumerate(base.layers):
        for block_name, proj_names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("up_proj", "down_proj", "gate_proj")),
        ):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                weights[key] = adapter._dequantize_weight(proj)
    return weights


def main():
    args = _parse_args()

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    adapter = MLXTrainingAdapter(backend)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("4-bit Extension — run_id=%s", run_id)
    logger.info("Output: %s", output_dir)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "four_bit_extension",
        "config": {
            "bits": 4,
            "group_size": args.group_size,
            "fp_models": args.fp_models,
        },
    }

    # ── Phase 1: Create 4-bit models ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1: QUANTIZE TO 4-BIT")
    logger.info("=" * 60)

    four_bit_pairs: list[tuple[str, str]] = []
    quantize_dir = output_dir / "derived_models"

    for fp_path in args.fp_models:
        if args.skip_quantize:
            # Look for existing 4-bit model in standard locations
            model_name = Path(fp_path).name
            q_name = f"{model_name}-4bit-g{args.group_size}-affine"
            candidates = [
                Path("results/feasibility_map") / "*/derived_models" / q_name,
                quantize_dir / q_name,
            ]
            found = None
            for c in candidates:
                import glob
                matches = glob.glob(str(c))
                if matches:
                    found = matches[0]
                    break
            if found:
                four_bit_pairs.append((fp_path, found))
                logger.info("  Found existing: %s", found)
            else:
                logger.warning("  No existing 4-bit model for %s, will quantize", fp_path)
                q_path = _quantize_model(fp_path, quantize_dir, 4, args.group_size)
                four_bit_pairs.append((fp_path, str(q_path)))
        else:
            q_path = _quantize_model(fp_path, quantize_dir, 4, args.group_size)
            four_bit_pairs.append((fp_path, str(q_path)))

    results["model_pairs"] = [
        {"fp": fp, "q4": q4} for fp, q4 in four_bit_pairs
    ]

    # ── Phase 2: RMT gate on 4-bit models ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: RMT GATE (4-BIT)")
    logger.info("=" * 60)

    pair_results_4bit: list[dict[str, Any]] = []
    for fp_path, q_path in four_bit_pairs:
        logger.info("Analyzing 4-bit pair: %s", Path(q_path).name)
        pair_result = _analyze_pair_rmt(fp_path, q_path, backend, adapter)
        pair_results_4bit.append(pair_result)

    results["rmt_4bit"] = pair_results_4bit

    # ── Aggregate and gate ──
    all_layers = [
        l for r in pair_results_4bit
        for l in r.get("per_layer", [])
        if l.get("svd_success", True)
    ]

    signal_ranks = [float(l["signal_rank"]) for l in all_layers]
    sv_fracs = [l["signal_variance_fraction"] for l in all_layers]
    frob_norms = [l["error_frobenius_norm"] for l in all_layers]
    spec_norms = [l["error_spectral_norm"] for l in all_layers]

    sr_mean, sr_ci_lo, sr_ci_hi = _bootstrap_ci(signal_ranks)
    svf_mean, svf_ci_lo, svf_ci_hi = _bootstrap_ci(sv_fracs)

    gate_pass = sr_ci_lo > 0.0 and svf_ci_lo > 0.0

    aggregate_4bit = {
        "n_layers": len(all_layers),
        "n_layers_with_signal": sum(1 for r in signal_ranks if r > 0),
        "mean_signal_rank": sr_mean,
        "signal_rank_ci": [sr_ci_lo, sr_ci_hi],
        "mean_signal_variance_fraction": svf_mean,
        "sv_frac_ci": [svf_ci_lo, svf_ci_hi],
        "mean_frobenius_norm": statistics.mean(frob_norms) if frob_norms else 0.0,
        "mean_spectral_norm": statistics.mean(spec_norms) if spec_norms else 0.0,
        "gate_pass": gate_pass,
    }
    results["aggregate_4bit"] = aggregate_4bit

    # ── Phase 3: Compare 8-bit vs 4-bit (if 8-bit data available) ──
    eight_bit_ref = None
    for ref_path in [
        "results/rmt_quantization_error/20260226T001044Z/rmt_quantization_error.json",
    ]:
        if Path(ref_path).exists():
            with open(ref_path) as f:
                eight_bit_ref = json.load(f)
            break

    comparison = None
    if eight_bit_ref:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 3: 8-BIT vs 4-BIT COMPARISON")
        logger.info("=" * 60)

        agg_8bit = eight_bit_ref.get("aggregate", {})

        comparison = {
            "eight_bit": {
                "mean_signal_rank": agg_8bit.get("mean_signal_rank", 0),
                "mean_sv_frac": agg_8bit.get("mean_signal_variance_fraction", 0),
                "mean_frobenius_norm": agg_8bit.get("mean_error_frobenius_norm", 0),
            },
            "four_bit": {
                "mean_signal_rank": sr_mean,
                "mean_sv_frac": svf_mean,
                "mean_frobenius_norm": aggregate_4bit["mean_frobenius_norm"],
            },
        }

        # Ratios
        if agg_8bit.get("mean_signal_rank", 0) > 0:
            comparison["rank_ratio_4b_over_8b"] = sr_mean / agg_8bit["mean_signal_rank"]
        if agg_8bit.get("mean_error_frobenius_norm", 0) > 0:
            comparison["frob_ratio_4b_over_8b"] = (
                aggregate_4bit["mean_frobenius_norm"]
                / agg_8bit["mean_error_frobenius_norm"]
            )

        results["comparison_8bit_vs_4bit"] = comparison

        logger.info(
            "  8-bit: signal_rank=%.1f, sv_frac=%.4f, ||E||_F=%.6f",
            comparison["eight_bit"]["mean_signal_rank"],
            comparison["eight_bit"]["mean_sv_frac"],
            comparison["eight_bit"]["mean_frobenius_norm"],
        )
        logger.info(
            "  4-bit: signal_rank=%.1f, sv_frac=%.4f, ||E||_F=%.6f",
            sr_mean, svf_mean, aggregate_4bit["mean_frobenius_norm"],
        )

    # ── Verdict ──
    if gate_pass:
        verdict = (
            f"4-BIT GATE PASSES: {aggregate_4bit['n_layers_with_signal']}/"
            f"{aggregate_4bit['n_layers']} layers have signal. "
            f"mean_signal_rank={sr_mean:.1f} CI=[{sr_ci_lo:.1f}, {sr_ci_hi:.1f}], "
            f"mean_sv_frac={svf_mean:.4f}. "
            "Corrective LoRA is geometrically justified for 4-bit."
        )
        next_steps = (
            "Run Experiments 2-4 with 4-bit model paths:\n"
            + "\n".join(
                f"  --quantized-model {q4}" for _, q4 in four_bit_pairs
            )
        )
    else:
        verdict = (
            f"4-BIT GATE FAILS: signal_rank CI=[{sr_ci_lo:.1f}, {sr_ci_hi:.1f}], "
            f"sv_frac CI=[{svf_ci_lo:.4f}, {svf_ci_hi:.4f}]. "
            "4-bit quantization error may be too noisy for low-rank correction."
        )
        next_steps = "No further experiments warranted at 4-bit."

    results["verdict"] = verdict
    results["next_steps"] = next_steps

    # Write results
    output_path = output_dir / "four_bit_extension.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", output_path)

    # Pretty print
    print("\n" + "=" * 72)
    print("4-BIT EXTENSION — SUMMARY")
    print("=" * 72)
    print(f"  Layers analyzed:     {aggregate_4bit['n_layers']}")
    print(f"  Layers with signal:  {aggregate_4bit['n_layers_with_signal']}")
    print(f"  Mean signal rank:    {sr_mean:.1f} CI=[{sr_ci_lo:.1f}, {sr_ci_hi:.1f}]")
    print(f"  Mean sv_frac:        {svf_mean:.4f} CI=[{svf_ci_lo:.4f}, {svf_ci_hi:.4f}]")
    print(f"  Mean ||E||_F:        {aggregate_4bit['mean_frobenius_norm']:.6f}")
    print(f"  Mean ||E||_2:        {aggregate_4bit['mean_spectral_norm']:.6f}")
    print(f"  Gate:                {'PASS' if gate_pass else 'FAIL'}")
    print()
    if comparison:
        print("  8-bit vs 4-bit comparison:")
        print(f"    Signal rank ratio (4b/8b): {comparison.get('rank_ratio_4b_over_8b', 'N/A'):.2f}x")
        print(f"    Frobenius ratio (4b/8b):   {comparison.get('frob_ratio_4b_over_8b', 'N/A'):.2f}x")
        print()
    print(f"  VERDICT: {verdict}")
    print()
    print(f"  Next steps: {next_steps}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: 4-bit Extension",
    )
    parser.add_argument(
        "--fp-models",
        nargs="+",
        default=DEFAULT_FP_MODELS,
        help="Full-precision model paths",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Base output directory",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip quantization step (use existing 4-bit models)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
