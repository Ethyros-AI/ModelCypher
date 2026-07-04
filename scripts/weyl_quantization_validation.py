#!/usr/bin/env python3
"""Weyl quantization validation on real model pairs.

For each layer, measures ||E_q||_2 (spectral norm of quantization error)
and checks the Weyl bound: ||E_q||_2 < spectral_gap / 2.

If the bound holds, Weyl's theorem (1912) guarantees no singular value
crossing — the quantized model's geometry is topologically identical
to full precision.  Training on quantized weights is geometrically safe.

Error norm ||E_q||_2 is computed via exact SVD in the shared domain module
(quantization_weyl_precheck).

Usage:
    poetry run python scripts/weyl_quantization_validation.py

    # Custom model pairs
    poetry run python scripts/weyl_quantization_validation.py \\
        --pairs /path/to/fp16 /path/to/quantized /path/to/fp16_2 /path/to/quant_2

    # Attach CKA artifacts (same order as --pairs)
    poetry run python scripts/weyl_quantization_validation.py \\
        --pairs /path/to/fp /path/to/q /path/to/fp2 /path/to/q2 \\
        --cka-artifacts /path/to/gates.json /path/to/closedform_correction.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from modelcypher.backends import initialize_default_backend
from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.training.quantization_weyl_precheck import (
    run_quantization_weyl_precheck,
)

# Default pairs: Codex's 8-bit derived models
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

OUTPUT_DIR = Path("results/weyl_quantization_validation")
DEFAULT_GEOMETRY_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("weyl_validation")


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
    """Extract raw weight matrices one layer at a time.

    Returns a dict mapping layer_key -> dequantized weight array.
    Weights are NOT released here — caller is responsible.
    """

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


def _validate_pair(
    fp_path: str,
    q_path: str,
    backend: Any,
    adapter: MLXTrainingAdapter,
    *,
    geometry_mode: str,
    geometry_seed: int,
) -> dict[str, Any]:
    """Compare geometry between full-precision and quantized model."""
    logger.info("=== Validating pair ===")
    logger.info("  FP model:  %s", fp_path)
    logger.info("  Q model:   %s", q_path)

    # 1. Analyze geometry on both models (for spectral_gap, sigma_k, etc.)
    logger.info("Loading FP model for geometry...")
    fp_model, _ = backend.load_model(fp_path)
    use_randomized = geometry_mode == "randomized"
    randomized_kwargs = {"seed": geometry_seed} if use_randomized else None
    fp_geoms = adapter.analyze_model_geometry_streaming(
        fp_model,
        use_randomized=use_randomized,
        randomized_kwargs=randomized_kwargs,
    )

    # Also extract raw weights from FP model for error computation
    logger.info("Extracting FP weights for error measurement...")
    fp_weights = _extract_layer_weights_streaming(fp_model, adapter)
    del fp_model
    gc.collect()
    _clear_gpu_cache()

    logger.info("Loading quantized model...")
    q_model, _ = backend.load_model(q_path)
    q_geoms = adapter.analyze_model_geometry_streaming(
        q_model,
        use_randomized=use_randomized,
        randomized_kwargs=randomized_kwargs,
    )

    # Extract raw weights from quantized model
    logger.info("Extracting quantized weights for error measurement...")
    q_weights = _extract_layer_weights_streaming(q_model, adapter)
    del q_model
    gc.collect()
    _clear_gpu_cache()

    # 2. Run shared domain precheck for Weyl crossing metrics
    precheck_payload = run_quantization_weyl_precheck(
        fp_weights=fp_weights,
        quantized_weights=q_weights,
        backend=backend,
    )
    precheck_by_layer = {
        row["layer_key"]: row for row in precheck_payload["per_layer"]
    }

    # 3. For each matching layer, merge geometry deltas + precheck
    common_keys = sorted(set(fp_geoms.keys()) & set(q_geoms.keys()))
    logger.info("Comparing %d layers...", len(common_keys))

    per_layer: list[dict[str, Any]] = []
    n_weyl_safe = 0
    n_tail_dims_match = 0
    max_error_norm = 0.0
    max_error_over_gap = 0.0

    for key in common_keys:
        fp_g = fp_geoms[key]
        q_g = q_geoms[key]
        precheck_layer = precheck_by_layer.get(key)
        if precheck_layer is None:
            continue
        error_norm = float(precheck_layer["error_norm_2"])
        spectral_gap = float(precheck_layer["spectral_gap"])
        weyl_threshold = float(precheck_layer["gap_half"])
        error_over_gap = float(precheck_layer["error_over_gap_half"])
        weyl_safe = not bool(precheck_layer["crossing"])

        tail_dims_match = fp_g.tail_dims == q_g.tail_dims

        if weyl_safe:
            n_weyl_safe += 1
        if tail_dims_match:
            n_tail_dims_match += 1
        if error_norm > max_error_norm:
            max_error_norm = error_norm
        if error_over_gap > max_error_over_gap and not math.isinf(error_over_gap):
            max_error_over_gap = error_over_gap

        layer_result = {
            "layer_key": key,
            "shape": list(fp_g.shape),
            "fp_sigma_max": fp_g.sigma_max,
            "q_sigma_max": q_g.sigma_max,
            "sigma_max_diff": abs(fp_g.sigma_max - q_g.sigma_max),
            "fp_sigma_k": fp_g.sigma_k,
            "q_sigma_k": q_g.sigma_k,
            "sigma_k_diff": abs(fp_g.sigma_k - q_g.sigma_k),
            "fp_tail_dims": fp_g.tail_dims,
            "q_tail_dims": q_g.tail_dims,
            "tail_dims_match": tail_dims_match,
            "fp_spectral_gap": spectral_gap,
            "error_norm": error_norm,
            "error_norm_mode": "exact_svd",
            "weyl_threshold": weyl_threshold,
            "error_over_gap_ratio": error_over_gap,
            "weyl_safe": weyl_safe,
        }
        per_layer.append(layer_result)

        status = "SAFE" if weyl_safe else "VIOLATION"
        logger.info(
            "  %s: ||E||_2=%.6f, gap/2=%.6f, ratio=%.4f [%s]%s",
            key.split(".")[-2],
            error_norm,
            weyl_threshold,
            error_over_gap if not math.isinf(error_over_gap) else float("nan"),
            status,
            "" if tail_dims_match else " (tail_dims DIFFER)",
        )

    # Clean up all weights
    del fp_weights, q_weights
    gc.collect()
    _clear_gpu_cache()

    all_safe = bool(precheck_payload["all_non_crossing"])

    result = {
        "fp_model": fp_path,
        "q_model": q_path,
        "fp_model_id": Path(fp_path).name,
        "q_model_id": Path(q_path).name,
        "n_layers": len(common_keys),
        "n_weyl_safe": n_weyl_safe,
        "n_tail_dims_match": n_tail_dims_match,
        "all_weyl_safe": all_safe,
        "max_error_norm": max_error_norm,
        "max_error_over_gap_ratio": max_error_over_gap,
        "verdict": "SAFE — quantized geometry topologically identical"
        if all_safe
        else f"VIOLATION — {len(common_keys) - n_weyl_safe} layers cross Weyl bound",
        "per_layer": per_layer,
    }

    logger.info(
        "Verdict: %s (%d/%d safe, %d/%d tail_dims match)",
        result["verdict"],
        n_weyl_safe,
        len(common_keys),
        n_tail_dims_match,
        len(common_keys),
    )

    return result


def _compute_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-layer metrics across all validated model pairs."""
    total_layers = sum(int(r["n_layers"]) for r in results)
    total_tail_match = sum(int(r["n_tail_dims_match"]) for r in results)
    tail_match_pct = (
        (100.0 * float(total_tail_match) / float(total_layers))
        if total_layers > 0
        else 0.0
    )

    all_layers = [layer for r in results for layer in r.get("per_layer", [])]
    if not all_layers:
        return {
            "n_pairs": len(results),
            "n_layers_total": total_layers,
            "n_tail_dims_match_total": total_tail_match,
            "tail_match_pct": tail_match_pct,
            "max_sigma_max_rel_pct": 0.0,
            "max_sigma_k_rel_pct": 0.0,
            "max_error_norm": 0.0,
            "max_error_over_gap_ratio": 0.0,
            "per_model": [],
        }

    def _max_finite(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return max(finite) if finite else 0.0

    max_sigma_max_rel_pct = _max_finite(
        [100.0 * float(layer["sigma_max_diff"]) / float(layer["fp_sigma_max"]) for layer in all_layers]
    )
    max_sigma_k_rel_pct = _max_finite(
        [100.0 * float(layer["sigma_k_diff"]) / float(layer["fp_sigma_k"]) for layer in all_layers]
    )
    max_error_norm = _max_finite([float(layer["error_norm"]) for layer in all_layers])
    max_error_over_gap_ratio = _max_finite(
        [float(layer["error_over_gap_ratio"]) for layer in all_layers]
    )

    per_model = []
    for result in results:
        per_layers = result.get("per_layer", [])
        model_sigma_max = _max_finite(
            [100.0 * float(layer["sigma_max_diff"]) / float(layer["fp_sigma_max"]) for layer in per_layers]
        )
        model_sigma_k = _max_finite(
            [100.0 * float(layer["sigma_k_diff"]) / float(layer["fp_sigma_k"]) for layer in per_layers]
        )
        per_model.append(
            {
                "fp_model_id": result["fp_model_id"],
                "q_model_id": result["q_model_id"],
                "n_layers": int(result["n_layers"]),
                "n_tail_dims_match": int(result["n_tail_dims_match"]),
                "tail_match_pct": (
                    100.0 * float(result["n_tail_dims_match"]) / float(result["n_layers"])
                    if int(result["n_layers"]) > 0
                    else 0.0
                ),
                "max_sigma_max_rel_pct": model_sigma_max,
                "max_sigma_k_rel_pct": model_sigma_k,
                "max_error_norm": float(result["max_error_norm"]),
                "max_error_over_gap_ratio": float(result["max_error_over_gap_ratio"]),
            },
        )

    return {
        "n_pairs": len(results),
        "n_layers_total": total_layers,
        "n_tail_dims_match_total": total_tail_match,
        "tail_match_pct": tail_match_pct,
        "max_sigma_max_rel_pct": max_sigma_max_rel_pct,
        "max_sigma_k_rel_pct": max_sigma_k_rel_pct,
        "max_error_norm": max_error_norm,
        "max_error_over_gap_ratio": max_error_over_gap_ratio,
        "per_model": per_model,
    }


def _extract_min_cka_from_artifact(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("min_cka") is not None:
            return float(diagnostics["min_cka"])

        if payload.get("min_cka") is not None:
            return float(payload["min_cka"])

        sweep = payload.get("sweep")
        if isinstance(sweep, list) and sweep:
            first = sweep[0]
            if isinstance(first, dict):
                baseline_cka = first.get("baseline_cka")
                if isinstance(baseline_cka, dict) and baseline_cka.get("min_cka") is not None:
                    return float(baseline_cka["min_cka"])
                post_cka = first.get("post_cka")
                if isinstance(post_cka, dict) and post_cka.get("min_cka") is not None:
                    return float(post_cka["min_cka"])

    return None


def _build_crossing_to_cka_map(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in results:
        min_cka = result.get("min_cka_artifact")
        if min_cka is None:
            continue
        n_layers = int(result.get("n_layers", 0))
        n_safe = int(result.get("n_weyl_safe", 0))
        non_crossing_fraction = (
            float(n_safe) / float(n_layers) if n_layers > 0 else 0.0
        )
        rows.append(
            {
                "fp_model_id": result.get("fp_model_id"),
                "q_model_id": result.get("q_model_id"),
                "non_crossing_layer_fraction": non_crossing_fraction,
                "max_error_over_gap_half": float(result.get("max_error_over_gap_ratio", 0.0)),
                "min_cka": float(min_cka),
                "cka_artifact": result.get("cka_artifact"),
            }
        )

    if len(rows) < 2:
        return {
            "n_rows": len(rows),
            "rows": rows,
            "spearman_non_crossing_fraction_vs_min_cka": None,
            "spearman_max_error_over_gap_half_vs_min_cka": None,
        }

    x_non_cross = [row["non_crossing_layer_fraction"] for row in rows]
    x_error = [row["max_error_over_gap_half"] for row in rows]
    y_cka = [row["min_cka"] for row in rows]
    rho_non_cross, p_non_cross = spearmanr(x_non_cross, y_cka)
    rho_error, p_error = spearmanr(x_error, y_cka)

    return {
        "n_rows": len(rows),
        "rows": rows,
        "spearman_non_crossing_fraction_vs_min_cka": {
            "rho": float(rho_non_cross) if rho_non_cross == rho_non_cross else None,
            "p_value": float(p_non_cross) if p_non_cross == p_non_cross else None,
        },
        "spearman_max_error_over_gap_half_vs_min_cka": {
            "rho": float(rho_error) if rho_error == rho_error else None,
            "p_value": float(p_error) if p_error == p_error else None,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weyl quantization validation on real model pairs.",
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
        "--geometry-mode",
        choices=("randomized", "exact"),
        default="randomized",
        help=(
            "Geometry analysis mode. "
            "'randomized' uses randomized SVD for faster throughput (default), "
            "'exact' uses full SVD."
        ),
    )
    parser.add_argument(
        "--geometry-seed",
        type=int,
        default=DEFAULT_GEOMETRY_SEED,
        help="Seed used when --geometry-mode=randomized.",
    )
    parser.add_argument(
        "--cka-artifacts",
        nargs="+",
        default=None,
        help=(
            "Optional CKA artifact JSON paths aligned to --pairs order. "
            "Use '-' for a pair without artifact."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.pairs:
        if len(args.pairs) % 2 != 0:
            raise ValueError("--pairs requires alternating fp/quantized paths (even count)")
        pairs = [
            (args.pairs[i], args.pairs[i + 1])
            for i in range(0, len(args.pairs), 2)
        ]
    else:
        pairs = DEFAULT_PAIRS

    cka_artifact_paths: list[Path | None] | None = None
    if args.cka_artifacts is not None:
        if len(args.cka_artifacts) != len(pairs):
            raise ValueError(
                "--cka-artifacts must have the same length as model pairs "
                f"(got {len(args.cka_artifacts)} artifacts for {len(pairs)} pairs)",
            )
        cka_artifact_paths = []
        for raw_path in args.cka_artifacts:
            if raw_path.strip() == "-":
                cka_artifact_paths.append(None)
                continue
            artifact_path = Path(raw_path).expanduser().resolve()
            if not artifact_path.exists():
                raise FileNotFoundError(f"CKA artifact not found: {artifact_path}")
            cka_artifact_paths.append(artifact_path)

    # Validate paths
    for fp_path, q_path in pairs:
        if not Path(fp_path).exists():
            raise FileNotFoundError(f"FP model not found: {fp_path}")
        if not Path(q_path).exists():
            raise FileNotFoundError(f"Quantized model not found: {q_path}")

    backend = initialize_default_backend()
    adapter = MLXTrainingAdapter(backend)

    results: list[dict[str, Any]] = []
    for pair_idx, (fp_path, q_path) in enumerate(pairs):
        result = _validate_pair(
            fp_path,
            q_path,
            backend,
            adapter,
            geometry_mode=args.geometry_mode,
            geometry_seed=args.geometry_seed,
        )
        if cka_artifact_paths is not None:
            cka_artifact = cka_artifact_paths[pair_idx]
            result["cka_artifact"] = str(cka_artifact) if cka_artifact is not None else None
            result["min_cka_artifact"] = (
                _extract_min_cka_from_artifact(cka_artifact)
                if cka_artifact is not None
                else None
            )
        results.append(result)

    # Write output
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "n_pairs": len(results),
        "all_safe": all(r["all_weyl_safe"] for r in results),
        "analysis_config": {
            "geometry_mode": args.geometry_mode,
            "geometry_seed": args.geometry_seed if args.geometry_mode == "randomized" else None,
            "error_norm_mode": "exact_svd",
        },
        "aggregate": _compute_aggregate(results),
        "crossing_to_cka_map": _build_crossing_to_cka_map(results),
        "pairs": results,
    }

    output_path = output_dir / "weyl_quantization_validation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("WEYL QUANTIZATION VALIDATION")
    print("=" * 80)
    for r in results:
        print(
            f"\n{r['fp_model_id']} vs {r['q_model_id']}:"
            f"\n  Verdict: {r['verdict']}"
            f"\n  Layers safe: {r['n_weyl_safe']}/{r['n_layers']}"
            f"\n  Tail dims match: {r['n_tail_dims_match']}/{r['n_layers']}"
            f"\n  Max ||E_q||_2: {r['max_error_norm']:.6f}"
            f"\n  Max error/gap ratio: {r['max_error_over_gap_ratio']:.4f}"
        )
    print("=" * 80)

    if payload["all_safe"]:
        print("\nALL PAIRS SAFE — quantized training is geometrically safe.")
    else:
        n_violations = sum(1 for r in results if not r["all_weyl_safe"])
        print(f"\n{n_violations} PAIR(S) HAVE WEYL VIOLATIONS.")


if __name__ == "__main__":
    main()
