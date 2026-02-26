#!/usr/bin/env python3
"""RMT decomposition of quantization error matrices.

GATE EXPERIMENT for quantization geometry deep dive.

For each layer, computes E_q = W_fp - W_q, takes SVD of E_q, and applies
Marchenko-Pastur signal/noise separation to determine whether the
quantization error has systematic (low-rank) structure or is effectively
random noise.

If E_q has signal components above the MP bulk edge, corrective LoRA
training is geometrically justified.  If all eigenvalues fall within
the MP bulk, the error is random and low-rank correction is hopeless.

Gate criterion (both must hold):
  - 95% bootstrap CI lower bound for mean(signal_rank) > 0
  - 95% bootstrap CI lower bound for mean(signal_variance_fraction) > 0

Usage:
    poetry run python scripts/rmt_quantization_error.py

    # Single model (faster)
    poetry run python scripts/rmt_quantization_error.py \\
        --pairs /path/to/fp16 /path/to/quantized

    # Custom bootstrap samples
    poetry run python scripts/rmt_quantization_error.py --bootstrap-samples 50000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.backends import initialize_default_backend
from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    compute_signal_rank_from_singular_values,
)

# Default pairs: same as Weyl validation script
DEFAULT_PAIRS = [
    (
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16",
        "results/feasibility_map/20260225T160732Z/derived_models/"
        "Qwen3-1.7B-MLX-bf16-8bit-g64-affine",
    ),
    (
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "results/feasibility_map/20260225T160732Z/derived_models/"
        "Qwen3-8B-bf16-8bit-g64-affine",
    ),
]

OUTPUT_DIR = Path("results/rmt_quantization_error")
DEFAULT_BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rmt_quant_error")


def _clear_gpu_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _extract_layer_weights_streaming(
    model: Any,
    adapter: MLXTrainingAdapter,
) -> dict[str, Any]:
    """Extract dequantized weight matrices one layer at a time."""
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute")

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
                w = adapter._dequantize_weight(proj)
                weights[key] = w

    return weights


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    observed_mean = sum(values) / n
    rng = random.Random(seed)
    means: list[float] = []

    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - ci
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1.0 - alpha / 2) * n_bootstrap))

    return observed_mean, means[lower_idx], means[upper_idx]


def _analyze_pair(
    fp_path: str,
    q_path: str,
    backend: Any,
    adapter: MLXTrainingAdapter,
) -> dict[str, Any]:
    """Compute RMT decomposition of quantization error for a model pair."""
    import mlx.core as mx

    logger.info("=== Analyzing pair ===")
    logger.info("  FP model:  %s", fp_path)
    logger.info("  Q model:   %s", q_path)

    # Load and extract weights from both models
    logger.info("Loading FP model...")
    fp_model, _ = backend.load_model(fp_path)
    logger.info("Extracting FP weights...")
    fp_weights = _extract_layer_weights_streaming(fp_model, adapter)
    del fp_model
    gc.collect()
    _clear_gpu_cache()

    logger.info("Loading quantized model...")
    q_model, _ = backend.load_model(q_path)
    logger.info("Extracting Q weights...")
    q_weights = _extract_layer_weights_streaming(q_model, adapter)
    del q_model
    gc.collect()
    _clear_gpu_cache()

    # Process each layer
    common_keys = sorted(set(fp_weights.keys()) & set(q_weights.keys()))
    logger.info("Analyzing %d layers...", len(common_keys))

    per_layer: list[dict[str, Any]] = []
    n_svd_failures = 0

    for key in common_keys:
        fp_w = fp_weights[key]
        q_w = q_weights[key]

        # Compute E_q = W_fp - W_q in float32
        E = backend.astype(fp_w, "float32") - backend.astype(q_w, "float32")
        backend.eval(E)

        m, n = int(E.shape[0]), int(E.shape[1])

        # Frobenius norm
        frob_norm = float(backend.to_scalar(backend.norm(E)))

        # SVD of E_q for singular values
        svd_success = True
        signal_rank = 0
        noise_rank = min(m, n)
        signal_variance_fraction = 0.0
        mp_upper_edge = 0.0
        top_svs: list[float] = []
        spectral_norm = 0.0

        try:
            # compute_uv=False returns only singular values — cheaper, safer
            S = mx.linalg.svd(E, compute_uv=False, stream=mx.cpu)
            mx.eval(S)

            n_sv = int(S.shape[0])
            spectral_norm = float(S[0].item()) if n_sv > 0 else 0.0
            top_svs = [float(S[i].item()) for i in range(min(5, n_sv))]

            # Apply RMT signal/noise separation
            rmt_result = compute_signal_rank_from_singular_values(
                S,
                n_samples=m,
                n_features=n,
                backend=backend,
                center_correction=True,
            )

            signal_rank = rmt_result.signal_rank
            noise_rank = rmt_result.noise_rank
            signal_variance_fraction = rmt_result.signal_variance_fraction
            mp_upper_edge = rmt_result.mp_upper_edge

            del S

        except Exception as exc:
            logger.warning("  SVD failed for %s: %s", key, exc)
            svd_success = False
            n_svd_failures += 1

        del E
        gc.collect()

        layer_result = {
            "layer_key": key,
            "shape": [m, n],
            "signal_rank": signal_rank,
            "noise_rank": noise_rank,
            "signal_variance_fraction": signal_variance_fraction,
            "mp_upper_edge": mp_upper_edge,
            "error_spectral_norm": spectral_norm,
            "error_frobenius_norm": frob_norm,
            "n_singular_values": min(m, n),
            "top_5_singular_values": top_svs,
            "svd_success": svd_success,
        }
        per_layer.append(layer_result)

        status = f"signal_rank={signal_rank}" if svd_success else "SVD_FAIL"
        logger.info(
            "  %s [%dx%d]: %s, sv_frac=%.4f, ||E||_2=%.6f, ||E||_F=%.6f",
            key.split(".")[-2],
            m,
            n,
            status,
            signal_variance_fraction,
            spectral_norm,
            frob_norm,
        )

    # Clean up
    del fp_weights, q_weights
    gc.collect()
    _clear_gpu_cache()

    result = {
        "fp_model": fp_path,
        "q_model": q_path,
        "fp_model_id": Path(fp_path).name,
        "q_model_id": Path(q_path).name,
        "n_layers": len(common_keys),
        "n_svd_failures": n_svd_failures,
        "per_layer": per_layer,
    }

    logger.info(
        "Pair complete: %d layers, %d SVD failures",
        len(common_keys),
        n_svd_failures,
    )

    return result


def _compute_aggregate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate RMT metrics across all model pairs."""
    all_layers = [
        layer
        for r in results
        for layer in r.get("per_layer", [])
        if layer.get("svd_success", True)
    ]

    if not all_layers:
        return {
            "n_layers": 0,
            "n_layers_with_signal": 0,
            "n_svd_failures": sum(r.get("n_svd_failures", 0) for r in results),
            "mean_signal_rank": 0.0,
            "median_signal_rank": 0.0,
            "max_signal_rank": 0,
            "mean_signal_variance_fraction": 0.0,
            "median_signal_variance_fraction": 0.0,
            "mean_mp_upper_edge": 0.0,
            "mean_error_spectral_norm": 0.0,
            "mean_error_frobenius_norm": 0.0,
        }

    signal_ranks = [layer["signal_rank"] for layer in all_layers]
    sv_fracs = [layer["signal_variance_fraction"] for layer in all_layers]
    mp_edges = [layer["mp_upper_edge"] for layer in all_layers]
    spec_norms = [layer["error_spectral_norm"] for layer in all_layers]
    frob_norms = [layer["error_frobenius_norm"] for layer in all_layers]

    return {
        "n_layers": len(all_layers),
        "n_layers_with_signal": sum(1 for r in signal_ranks if r > 0),
        "n_svd_failures": sum(r.get("n_svd_failures", 0) for r in results),
        "mean_signal_rank": statistics.mean(signal_ranks),
        "median_signal_rank": statistics.median(signal_ranks),
        "max_signal_rank": max(signal_ranks),
        "mean_signal_variance_fraction": statistics.mean(sv_fracs),
        "median_signal_variance_fraction": statistics.median(sv_fracs),
        "mean_mp_upper_edge": statistics.mean(mp_edges),
        "mean_error_spectral_norm": statistics.mean(spec_norms),
        "mean_error_frobenius_norm": statistics.mean(frob_norms),
    }


def _compute_gate(
    results: list[dict[str, Any]],
    n_bootstrap: int,
) -> dict[str, Any]:
    """Evaluate the gate criterion via bootstrap CI."""
    all_layers = [
        layer
        for r in results
        for layer in r.get("per_layer", [])
        if layer.get("svd_success", True)
    ]

    signal_ranks = [float(layer["signal_rank"]) for layer in all_layers]
    sv_fracs = [layer["signal_variance_fraction"] for layer in all_layers]

    sr_mean, sr_ci_lo, sr_ci_hi = _bootstrap_ci(signal_ranks, n_bootstrap)
    svf_mean, svf_ci_lo, svf_ci_hi = _bootstrap_ci(sv_fracs, n_bootstrap)

    gate_pass = sr_ci_lo > 0.0 and svf_ci_lo > 0.0

    if gate_pass:
        reason = (
            f"GATE PASSES: mean signal_rank={sr_mean:.2f} "
            f"CI=[{sr_ci_lo:.2f}, {sr_ci_hi:.2f}], "
            f"mean signal_var_frac={svf_mean:.4f} "
            f"CI=[{svf_ci_lo:.4f}, {svf_ci_hi:.4f}]"
        )
    else:
        reasons = []
        if sr_ci_lo <= 0.0:
            reasons.append(
                f"signal_rank CI lower={sr_ci_lo:.2f} includes 0"
            )
        if svf_ci_lo <= 0.0:
            reasons.append(
                f"signal_var_frac CI lower={svf_ci_lo:.6f} includes 0"
            )
        reason = "GATE FAILS: " + "; ".join(reasons)

    return {
        "gate_pass": gate_pass,
        "reason": reason,
        "signal_rank_mean": sr_mean,
        "signal_rank_ci_lower": sr_ci_lo,
        "signal_rank_ci_upper": sr_ci_hi,
        "signal_variance_fraction_mean": svf_mean,
        "signal_variance_fraction_ci_lower": svf_ci_lo,
        "signal_variance_fraction_ci_upper": svf_ci_hi,
        "n_layers_analyzed": len(all_layers),
        "n_bootstrap_samples": n_bootstrap,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RMT decomposition of quantization error matrices.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Model pairs as: fp1 q1 fp2 q2 ... (alternating fp/quantized paths).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples for CI computation (default 10000).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.pairs:
        if len(args.pairs) % 2 != 0:
            raise ValueError(
                "--pairs requires alternating fp/quantized paths (even count)"
            )
        pairs = [
            (args.pairs[i], args.pairs[i + 1])
            for i in range(0, len(args.pairs), 2)
        ]
    else:
        pairs = DEFAULT_PAIRS

    # Validate paths
    for fp_path, q_path in pairs:
        if not Path(fp_path).exists():
            raise FileNotFoundError(f"FP model not found: {fp_path}")
        if not Path(q_path).exists():
            raise FileNotFoundError(f"Quantized model not found: {q_path}")

    backend = initialize_default_backend()
    adapter = MLXTrainingAdapter(backend)

    results: list[dict[str, Any]] = []
    for fp_path, q_path in pairs:
        result = _analyze_pair(fp_path, q_path, backend, adapter)
        results.append(result)

    # Compute aggregate and gate
    aggregate = _compute_aggregate(results)
    gate_result = _compute_gate(results, args.bootstrap_samples)

    # Write output
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "rmt_quantization_error",
        "n_pairs": len(results),
        "gate_result": gate_result,
        "aggregate": aggregate,
        "pairs": results,
    }

    output_path = output_dir / "rmt_quantization_error.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("RMT QUANTIZATION ERROR DECOMPOSITION")
    print("=" * 80)

    for r in results:
        n_with_signal = sum(
            1
            for layer in r["per_layer"]
            if layer.get("svd_success", True) and layer["signal_rank"] > 0
        )
        print(
            f"\n{r['fp_model_id']} vs {r['q_model_id']}:"
            f"\n  Layers: {r['n_layers']}"
            f"\n  SVD failures: {r['n_svd_failures']}"
            f"\n  Layers with signal: {n_with_signal}/{r['n_layers']}"
        )

    print(f"\nAggregate ({aggregate['n_layers']} layers):")
    print(f"  Mean signal_rank: {aggregate['mean_signal_rank']:.2f}")
    print(f"  Median signal_rank: {aggregate['median_signal_rank']:.1f}")
    print(f"  Max signal_rank: {aggregate['max_signal_rank']}")
    print(
        f"  Layers with signal: "
        f"{aggregate['n_layers_with_signal']}/{aggregate['n_layers']}"
    )
    print(
        f"  Mean signal_var_frac: "
        f"{aggregate['mean_signal_variance_fraction']:.4f}"
    )
    print(
        f"  Mean ||E_q||_2: {aggregate['mean_error_spectral_norm']:.6f}"
    )
    print(
        f"  Mean ||E_q||_F: {aggregate['mean_error_frobenius_norm']:.6f}"
    )

    print(f"\n{'=' * 80}")
    print("GATE DECISION")
    print("=" * 80)
    print(f"  {gate_result['reason']}")
    if gate_result["gate_pass"]:
        print(
            "\n  PROCEED to corrective LoRA experiments "
            "(E_q has systematic structure)."
        )
    else:
        print(
            "\n  STOP — quantization error is random noise. "
            "Low-rank correction is geometrically unjustified."
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
