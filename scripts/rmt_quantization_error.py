#!/usr/bin/env python3
"""RMT decomposition of quantization error matrices.

GATE EXPERIMENT for quantization geometry deep dive.

For each layer, computes E_q = W_fp - W_q, takes SVD of E_q, and applies
Marchenko-Pastur signal/noise separation to determine whether the
quantization error has systematic (low-rank) structure or is effectively
random noise.

Supports two modes:
  - raw:            SVD of E_q directly (weight-space structure)
  - input_weighted: SVD of E_q @ sqrt(Sigma_x) (activation-weighted,
                    measures functional error in directions the model uses)

    Activation weighting is RIGHT-SIDE: E_q @ sqrt(Sigma_x).
    Derived from E[||DeltaW x||^2] = tr(DeltaW Sigma_x DeltaW^T)
                                    = ||DeltaW Sigma_x^{1/2}||_F^2

Gate criterion (both must hold):
  - 95% bootstrap CI lower bound for mean(signal_rank) > 0
  - 95% bootstrap CI lower bound for mean(signal_variance_fraction) > 0

Usage:
    # Raw mode (default, original behavior)
    poetry run python scripts/rmt_quantization_error.py

    # Activation-weighted mode
    poetry run python scripts/rmt_quantization_error.py --mode input_weighted

    # Single model pair
    poetry run python scripts/rmt_quantization_error.py \\
        --pairs /path/to/fp16 /path/to/quantized

    # Custom calibration samples
    poetry run python scripts/rmt_quantization_error.py \\
        --mode input_weighted --n-calibration 64
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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

CALIBRATION_DATA_PATH = Path("data/training/benchmark_val.jsonl")
OUTPUT_DIR = Path("results/rmt_quantization_error")
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_N_CALIBRATION = 32
CALIBRATION_SEQ_LENGTH = 256
BOOTSTRAP_SEED = 42

# Mapping: projection name -> (covariance source, weighting quality)
# "attn" = use attention input covariance (input_layernorm output)
# "mlp"  = use MLP input covariance (post_attention_layernorm output)
# None   = no matching covariance (dimension mismatch), fall back to raw
PROJ_COV_MAP: dict[str, tuple[str | None, str]] = {
    "q_proj": ("attn", "exact"),
    "k_proj": ("attn", "exact"),
    "v_proj": ("attn", "exact"),
    "o_proj": ("attn", "approximate"),  # same dim, wrong distribution
    "up_proj": ("mlp", "exact"),
    "gate_proj": ("mlp", "exact"),
    "down_proj": (None, "raw_fallback"),  # intermediate_size != hidden_dim
}

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


def _load_calibration_texts(n_samples: int) -> list[str]:
    """Load calibration texts from benchmark_val.jsonl."""
    if not CALIBRATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {CALIBRATION_DATA_PATH}"
        )

    texts: list[str] = []
    with open(CALIBRATION_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry.get("text", "")
            if text:
                texts.append(text)

    logger.info("Loaded %d calibration texts from %s", len(texts), CALIBRATION_DATA_PATH)
    return texts


def _collect_layer_covariances(
    model: Any,
    tokenizer: Any,
    n_calibration: int,
    seq_length: int = CALIBRATION_SEQ_LENGTH,
) -> tuple[list[Any], list[Any]]:
    """Run calibration data through FP model, collect per-layer input covariances.

    Decomposes each layer's forward pass to collect two covariances:
      - attn_cov: covariance of input_layernorm(h) — exact for q/k/v projections
      - mlp_cov:  covariance of post_attention_layernorm(h) — exact for up/gate

    Returns:
        attn_sqrt_covs: list of sqrt(Sigma_x) for attention inputs [D, D]
        mlp_sqrt_covs:  list of sqrt(Sigma_x) for MLP inputs [D, D]
    """
    import mlx.core as mx

    texts = _load_calibration_texts(n_calibration)
    if not texts:
        raise ValueError("No calibration texts loaded")

    base = getattr(model, "model", model)
    embed_tokens = base.embed_tokens
    layers = base.layers
    n_layers = len(layers)

    # Accumulators: sum of x^T @ x for each layer
    attn_cov_sums: list[Any] = [None] * n_layers
    mlp_cov_sums: list[Any] = [None] * n_layers
    attn_n_tokens: list[int] = [0] * n_layers
    mlp_n_tokens: list[int] = [0] * n_layers

    logger.info(
        "Collecting covariances: %d samples, %d layers, seq_length=%d",
        len(texts), n_layers, seq_length,
    )

    for sample_idx, text in enumerate(texts):
        tokens = tokenizer.encode(text)[:seq_length]
        if len(tokens) < 2:
            continue

        input_ids = mx.array([tokens])
        h = embed_tokens(input_ids)  # [1, seq, D]
        mx.eval(h)

        # Causal mask — "causal" string is what Qwen3 layers expect
        mask = "causal"

        for layer_idx, layer in enumerate(layers):
            # --- Attention input covariance ---
            attn_in = layer.input_layernorm(h)  # [1, seq, D]
            mx.eval(attn_in)

            attn_2d = attn_in.reshape(-1, attn_in.shape[-1])  # [seq, D]
            attn_contrib = attn_2d.T @ attn_2d  # [D, D]
            mx.eval(attn_contrib)

            if attn_cov_sums[layer_idx] is None:
                attn_cov_sums[layer_idx] = attn_contrib
            else:
                attn_cov_sums[layer_idx] = attn_cov_sums[layer_idx] + attn_contrib
            attn_n_tokens[layer_idx] += int(attn_2d.shape[0])

            # --- Decomposed forward: attention + residual ---
            r = layer.self_attn(attn_in, mask)
            h_mid = h + r  # post-attention residual
            mx.eval(h_mid)

            # --- MLP input covariance ---
            mlp_in = layer.post_attention_layernorm(h_mid)  # [1, seq, D]
            mx.eval(mlp_in)

            mlp_2d = mlp_in.reshape(-1, mlp_in.shape[-1])  # [seq, D]
            mlp_contrib = mlp_2d.T @ mlp_2d  # [D, D]
            mx.eval(mlp_contrib)

            if mlp_cov_sums[layer_idx] is None:
                mlp_cov_sums[layer_idx] = mlp_contrib
            else:
                mlp_cov_sums[layer_idx] = mlp_cov_sums[layer_idx] + mlp_contrib
            mlp_n_tokens[layer_idx] += int(mlp_2d.shape[0])

            # --- Finish layer: MLP + residual ---
            r = layer.mlp(mlp_in)
            h = h_mid + r
            mx.eval(h)

            del attn_in, attn_2d, attn_contrib, r, h_mid, mlp_in, mlp_2d, mlp_contrib

        del input_ids, h
        gc.collect()

        if (sample_idx + 1) % 8 == 0:
            logger.info("  Calibration: %d/%d samples", sample_idx + 1, len(texts))

    logger.info("Calibration complete. Computing matrix square roots...")

    # Normalize and compute sqrt
    attn_sqrt_covs: list[Any] = []
    mlp_sqrt_covs: list[Any] = []

    for layer_idx in range(n_layers):
        # Attention covariance
        attn_cov = attn_cov_sums[layer_idx] / attn_n_tokens[layer_idx]
        mx.eval(attn_cov)
        attn_sqrt = _matrix_sqrt(attn_cov)
        attn_sqrt_covs.append(attn_sqrt)
        del attn_cov

        # MLP covariance
        mlp_cov = mlp_cov_sums[layer_idx] / mlp_n_tokens[layer_idx]
        mx.eval(mlp_cov)
        mlp_sqrt = _matrix_sqrt(mlp_cov)
        mlp_sqrt_covs.append(mlp_sqrt)
        del mlp_cov

        if (layer_idx + 1) % 7 == 0:
            logger.info(
                "  sqrt(Sigma_x): %d/%d layers", layer_idx + 1, n_layers,
            )

    del attn_cov_sums, mlp_cov_sums
    gc.collect()

    logger.info("Covariance collection done: %d layers", n_layers)
    return attn_sqrt_covs, mlp_sqrt_covs


def _matrix_sqrt(cov: Any) -> Any:
    """Compute sqrt(Sigma_x) via eigendecomposition. Sigma_x is symmetric PSD."""
    import mlx.core as mx

    # eigh requires float32/float64 — model activations may be bfloat16
    cov = cov.astype(mx.float32)
    eigenvalues, eigenvectors = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(eigenvalues, eigenvectors)

    # Clip negative eigenvalues (numerical safety for PSD)
    eigenvalues = mx.maximum(eigenvalues, mx.array(0.0))
    sqrt_eigenvalues = mx.sqrt(eigenvalues)

    # Reconstruct: V @ diag(sqrt(lambda)) @ V^T
    sqrt_cov = eigenvectors @ mx.diag(sqrt_eigenvalues) @ eigenvectors.T
    mx.eval(sqrt_cov)
    return sqrt_cov


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
    mode: str = "raw",
    n_calibration: int = DEFAULT_N_CALIBRATION,
) -> dict[str, Any]:
    """Compute RMT decomposition of quantization error for a model pair."""
    import mlx.core as mx

    logger.info("=== Analyzing pair (mode=%s) ===", mode)
    logger.info("  FP model:  %s", fp_path)
    logger.info("  Q model:   %s", q_path)

    attn_sqrt_covs: list[Any] | None = None
    mlp_sqrt_covs: list[Any] | None = None

    # For input_weighted mode: collect covariances from FP model before
    # extracting weights. This runs calibration data through the model.
    logger.info("Loading FP model...")
    fp_model, tokenizer = backend.load_model(fp_path)

    if mode == "input_weighted":
        logger.info("Collecting activation covariances (input_weighted mode)...")
        attn_sqrt_covs, mlp_sqrt_covs = _collect_layer_covariances(
            fp_model, tokenizer, n_calibration,
        )

    logger.info("Extracting FP weights...")
    fp_weights = _extract_layer_weights_streaming(fp_model, adapter)
    del fp_model, tokenizer
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

        # Frobenius norm of raw E_q (before weighting)
        frob_norm = float(backend.to_scalar(backend.norm(E)))

        # Determine weighting for this projection
        # Key format: model.layers.{idx}.{block}.{proj}.weight
        parts = key.split(".")
        proj_name = parts[-2]  # e.g. "q_proj"
        layer_idx = int(parts[2])  # layer index
        cov_source, weighting = PROJ_COV_MAP.get(proj_name, (None, "raw_fallback"))

        # Apply activation weighting if in input_weighted mode
        E_analysis = E  # what we run SVD on
        if mode == "input_weighted" and cov_source is not None:
            if cov_source == "attn" and attn_sqrt_covs is not None:
                sqrt_cov = attn_sqrt_covs[layer_idx]
            elif cov_source == "mlp" and mlp_sqrt_covs is not None:
                sqrt_cov = mlp_sqrt_covs[layer_idx]
            else:
                sqrt_cov = None

            if sqrt_cov is not None:
                # RIGHT-SIDE multiplication: E_q @ sqrt(Sigma_x)
                E_analysis = E @ sqrt_cov
                backend.eval(E_analysis)
            else:
                weighting = "raw_fallback"
        elif mode == "raw":
            weighting = "raw"
            cov_source = None

        m_a, n_a = int(E_analysis.shape[0]), int(E_analysis.shape[1])

        # SVD of E_analysis for singular values
        svd_success = True
        signal_rank = 0
        noise_rank = min(m_a, n_a)
        signal_variance_fraction = 0.0
        mp_upper_edge = 0.0
        top_svs: list[float] = []
        spectral_norm = 0.0

        try:
            S = mx.linalg.svd(E_analysis, compute_uv=False, stream=mx.cpu)
            mx.eval(S)

            n_sv = int(S.shape[0])
            spectral_norm = float(S[0].item()) if n_sv > 0 else 0.0
            top_svs = [float(S[i].item()) for i in range(min(5, n_sv))]

            # Apply RMT signal/noise separation
            rmt_result = compute_signal_rank_from_singular_values(
                S,
                n_samples=m_a,
                n_features=n_a,
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

        del E, E_analysis
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
            "n_singular_values": min(m_a, n_a),
            "top_5_singular_values": top_svs,
            "svd_success": svd_success,
            "weighting": weighting,
            "covariance_source": cov_source,
        }
        per_layer.append(layer_result)

        status = f"signal_rank={signal_rank}" if svd_success else "SVD_FAIL"
        logger.info(
            "  %s [%dx%d]: %s, sv_frac=%.4f, ||E||_2=%.6f, ||E||_F=%.6f [%s]",
            key.split(".")[-2],
            m,
            n,
            status,
            signal_variance_fraction,
            spectral_norm,
            frob_norm,
            weighting,
        )

    # Clean up
    del fp_weights, q_weights
    if attn_sqrt_covs is not None:
        del attn_sqrt_covs
    if mlp_sqrt_covs is not None:
        del mlp_sqrt_covs
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
        "--mode",
        choices=["raw", "input_weighted"],
        default="raw",
        help=(
            "Analysis mode. 'raw' = SVD of E_q directly. "
            "'input_weighted' = SVD of E_q @ sqrt(Sigma_x) "
            "(activation-weighted functional error). Default: raw."
        ),
    )
    parser.add_argument(
        "--n-calibration",
        type=int,
        default=DEFAULT_N_CALIBRATION,
        help=(
            "Number of calibration samples for covariance estimation "
            "(input_weighted mode only). Default: 32."
        ),
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
        result = _analyze_pair(
            fp_path, q_path, backend, adapter,
            mode=args.mode,
            n_calibration=args.n_calibration,
        )
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
        "mode": args.mode,
        "n_pairs": len(results),
        "gate_result": gate_result,
        "aggregate": aggregate,
        "pairs": results,
    }

    if args.mode == "input_weighted":
        payload["n_calibration_samples"] = args.n_calibration
        payload["calibration_data"] = str(CALIBRATION_DATA_PATH)

    output_path = output_dir / "rmt_quantization_error.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    # Print summary
    print("\n" + "=" * 80)
    print(f"RMT QUANTIZATION ERROR DECOMPOSITION (mode={args.mode})")
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

    if args.mode == "input_weighted":
        # Show weighting breakdown
        all_layers = [
            layer
            for r in results
            for layer in r.get("per_layer", [])
        ]
        n_exact = sum(1 for l in all_layers if l.get("weighting") == "exact")
        n_approx = sum(1 for l in all_layers if l.get("weighting") == "approximate")
        n_fallback = sum(1 for l in all_layers if l.get("weighting") == "raw_fallback")
        print(
            f"\n  Weighting: {n_exact} exact, "
            f"{n_approx} approximate, {n_fallback} raw_fallback"
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
