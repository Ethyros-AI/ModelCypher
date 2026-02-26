#!/usr/bin/env python3
"""Experiment: Closed-Form Sequential Layer Correction.

Computes analytical weight corrections per-layer, sequentially, using the
activation-space eigenbasis. Instead of training a corrective LoRA (which
oscillates due to stale activations across layers — the "seesaw"), this
projects the weight error E = W_fp - W_q onto the top-k eigenvectors of the
activation covariance X^T @ X, per layer, using fresh activations from the
already-corrected model.

Algorithm:
    For each layer l (0 → L-1), sequentially:
      1. Forward pass through layers 0..l-1 → activations X_l
      2. E_l = W_fp_l - W_quantized_l  (per-projection weight error)
      3. Eigendecompose X_l^T @ X_l → eigenvectors V, eigenvalues lambda
      4. Rank k = D_eff(lambda)  (participation ratio, derived from data)
      5. Delta_l = (E_l @ V_k) @ V_k^T  (project error onto activation subspace)
      6. W_corrected_l = W_quantized_l + Delta_l
      7. Continue to layer l+1 with corrected model

No optimizer. No oscillation. No seesaw. The projection is deterministic
and optimal for the given rank in the activation subspace.

Usage:
    poetry run python scripts/closedform_sequential_correction.py

    # Custom rank sweep
    poetry run python scripts/closedform_sequential_correction.py \
        --rank-multipliers 1,2,5,10,20

    # Custom model
    poetry run python scripts/closedform_sequential_correction.py \
        --quantized-model /path/to/4bit \
        --fp-model /path/to/bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("closedform_correction")

# Default paths
DEFAULT_QUANTIZED = (
    "results/four_bit_extension/20260226T023950Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-4bit-g64-affine"
)
DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"

PROJ_NAMES_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
PROJ_NAMES_MLP = ("up_proj", "down_proj", "gate_proj")
ALL_PROJ_NAMES = PROJ_NAMES_ATTN + PROJ_NAMES_MLP


# ── Utilities ────────────────────────────────────────────────────────────


def _clear_gpu_cache() -> None:
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _load_eval_texts(dataset_path: str, n_samples: int) -> list[str]:
    """Load text samples from JSONL dataset."""
    texts: list[str] = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
    return texts[:n_samples]


# ── CKA measurement (reused pattern from stacked_corrective_recovery) ───


def _collect_activations(
    model, tokenizer, texts, backend, n_samples=30,
) -> dict[int, list]:
    """Collect per-layer mean-pooled activations for CKA."""
    activations: dict[int, list] = {}
    for text in texts[:n_samples]:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx, act in acts.items():
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(pooled)
    return activations


def _compute_cka(acts_a, acts_b, backend) -> dict[str, Any]:
    """Linear CKA between two sets of per-layer activations."""
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )

    per_layer: dict[int, float] = {}
    common_layers = sorted(set(acts_a.keys()) & set(acts_b.keys()))

    for layer_idx in common_layers:
        mat_a = mx.stack(acts_a[layer_idx])
        mat_b = mx.stack(acts_b[layer_idx])
        mx.eval(mat_a, mat_b)
        cka = compute_linear_cka_from_activations(mat_a, mat_b, backend)
        per_layer[layer_idx] = float(cka)

    values = list(per_layer.values())
    return {
        "min_cka": min(values) if values else 0.0,
        "mean_cka": sum(values) / len(values) if values else 0.0,
        "per_layer_cka": per_layer,
        "n_layers": len(per_layer),
    }


# ── Weight extraction ────────────────────────────────────────────────────


def _extract_fp_weights(model, adapter) -> dict[str, mx.array]:
    """Extract all projection weights from the FP model as a flat dict."""
    base = getattr(model, "model", model)
    weights: dict[str, mx.array] = {}
    for layer_idx, layer in enumerate(base.layers):
        for block_name, proj_names in (
            ("self_attn", PROJ_NAMES_ATTN),
            ("mlp", PROJ_NAMES_MLP),
        ):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None:
                    continue
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                if hasattr(proj, "weight"):
                    weights[key] = proj.weight
                elif hasattr(proj, "scales") and adapter is not None:
                    # QuantizedLinear — dequantize
                    weights[key] = adapter._dequantize_weight(proj)
    return weights


def _dequantize_model(model, adapter) -> int:
    """Convert all QuantizedLinear modules to plain Linear.

    Returns number of layers dequantized.
    """
    base = getattr(model, "model", model)
    n_deq = 0

    for layer_idx, layer in enumerate(base.layers):
        for block_name in ("self_attn", "mlp"):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in ALL_PROJ_NAMES:
                proj = getattr(block, proj_name, None)
                if proj is None or not isinstance(proj, nn.QuantizedLinear):
                    continue

                w = adapter._dequantize_weight(proj)
                in_features = int(w.shape[1])
                out_features = int(w.shape[0])

                new_linear = nn.Linear(in_features, out_features, bias=False)
                new_linear.weight = w
                mx.eval(new_linear.weight)

                if hasattr(proj, "bias") and proj.bias is not None:
                    new_linear.bias = proj.bias

                setattr(block, proj_name, new_linear)
                n_deq += 1

    gc.collect()
    _clear_gpu_cache()
    return n_deq


# ── Closed-form correction ──────────────────────────────────────────────


def _correct_projection(
    proj,
    key: str,
    fp_weights: dict[str, mx.array],
    V_k: mx.array,
    k: int,
) -> dict[str, Any] | None:
    """Apply closed-form correction to a single projection.

    E = W_fp - W_q (weight error)
    Delta = (E @ V_k) @ V_k^T  (project error onto activation subspace)
    W_corrected = W_q + Delta

    Returns diagnostics dict or None if skipped.
    """
    fp_w = fp_weights.get(key)
    if fp_w is None:
        return None

    q_w = proj.weight.astype(mx.float32)
    fp_w_f32 = fp_w.astype(mx.float32)
    E = fp_w_f32 - q_w  # [out, in]
    mx.eval(E)

    E_frob_sq = float(mx.sum(E * E).item())
    if E_frob_sq <= 0:
        del E, q_w, fp_w_f32
        return None

    # Project: Delta = (E @ V_k) @ V_k^T
    # E is [out, in], V_k is [in, k]
    E_proj = E @ V_k  # [out, k]
    Delta = E_proj @ V_k.T  # [out, in]
    mx.eval(Delta)

    Delta_frob_sq = float(mx.sum(Delta * Delta).item())
    correction_fraction = Delta_frob_sq / E_frob_sq

    # Apply correction
    corrected = q_w + Delta
    mx.eval(corrected)
    proj.weight = corrected.astype(proj.weight.dtype)

    # Residual after correction
    residual = fp_w_f32 - corrected
    residual_frob = float(mx.sqrt(mx.sum(residual * residual)).item())

    result = {
        "layer_key": key,
        "error_frob": math.sqrt(E_frob_sq),
        "delta_frob": math.sqrt(Delta_frob_sq),
        "correction_fraction": correction_fraction,
        "residual_frob": residual_frob,
    }

    del E, E_proj, Delta, corrected, q_w, fp_w_f32, residual
    return result


def _correct_layer(
    layer,
    layer_idx: int,
    fp_weights: dict[str, mx.array],
    eigvecs: mx.array,
    eigvals: mx.array,
    k: int,
) -> list[dict[str, Any]]:
    """Apply closed-form correction to projections in a layer.

    The activation covariance eigenbasis is computed from the layer input h.
    Only projections whose input IS h (or layer_norm(h), which preserves the
    subspace) are corrected:
      - self_attn: q_proj, k_proj, v_proj (input = layer_norm(h))
      - mlp: up_proj, gate_proj (input = post_attention_norm(h))

    Skipped (input is a different space):
      - o_proj: input = attention output (different subspace)
      - down_proj: input = SiLU(up(x)) * gate(x) (MLP intermediate)

    Returns per-projection diagnostics.
    """
    V_k = eigvecs[:, :k]  # [D, k]
    stats: list[dict[str, Any]] = []

    # Projections whose input is h (or a norm of h — same subspace)
    h_input_projs = {
        "self_attn": ("q_proj", "k_proj", "v_proj"),
        "mlp": ("up_proj", "gate_proj"),
    }
    skipped_projs = {
        "self_attn": ("o_proj",),
        "mlp": ("down_proj",),
    }

    for block_name, proj_names in h_input_projs.items():
        block = getattr(layer, block_name, None)
        if block is None:
            continue
        for proj_name in proj_names:
            proj = getattr(block, proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue

            key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
            result = _correct_projection(proj, key, fp_weights, V_k, k)
            if result is not None:
                stats.append(result)

    # Log skipped projections
    for block_name, proj_names in skipped_projs.items():
        for proj_name in proj_names:
            key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
            if key in fp_weights:
                stats.append({
                    "layer_key": key,
                    "skipped": True,
                    "reason": "input_space_mismatch",
                })

    return stats


def _run_sequential_correction(
    q_model,
    fp_weights: dict[str, mx.array],
    tokenizer,
    eval_texts: list[str],
    rank_multiplier: float,
    n_samples: int,
    max_len: int,
) -> dict[str, Any]:
    """Run closed-form sequential correction at a given rank multiplier.

    rank_multiplier scales D_eff to determine k per layer:
        k = max(1, round(D_eff * rank_multiplier))

    Returns per-layer correction stats and final CKA.
    """
    base = getattr(q_model, "model", q_model)
    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute")

    n_layers = len(base.layers)
    logger.info(
        "Sequential correction: %d layers, rank_multiplier=%.1f, "
        "%d calibration samples",
        n_layers, rank_multiplier, n_samples,
    )

    # Tokenize calibration data
    all_tokens = []
    for text in eval_texts[:n_samples]:
        tokens = tokenizer.encode(text)
        all_tokens.append(mx.array(tokens[:max_len]))

    max_seq = max(t.shape[0] for t in all_tokens)
    padded = []
    for t in all_tokens:
        if t.shape[0] < max_seq:
            pad_len = max_seq - t.shape[0]
            t = mx.concatenate([t, mx.zeros(pad_len, dtype=t.dtype)])
        padded.append(t)
    batch = mx.stack(padded)

    # Run embedding
    h = base.embed_tokens(batch)
    mx.eval(h)

    per_layer_results: list[dict[str, Any]] = []
    total_corrected = 0

    for layer_idx, layer in enumerate(base.layers):
        layer_start = time.monotonic()

        # Flatten activations: [n_samples, seq_len, D] → [N, D]
        X = h.reshape(-1, h.shape[-1]).astype(mx.float32)
        N_tok, D = int(X.shape[0]), int(X.shape[1])
        mx.eval(X)

        # Activation covariance
        XtX = (X.T @ X) / N_tok
        mx.eval(XtX)

        try:
            eigvals, eigvecs = mx.linalg.eigh(XtX, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
        except Exception as exc:
            logger.warning(
                "  eigh failed for layer %d: %s, skipping", layer_idx, exc,
            )
            h = layer(h)
            mx.eval(h)
            del X, XtX
            gc.collect()
            continue

        # eigh returns ascending order; flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Effective dimensionality (participation ratio) — derived from data
        total_var = float(mx.sum(eigvals).item())
        sum_sq = float(mx.sum(eigvals * eigvals).item())
        D_eff = total_var ** 2 / sum_sq if sum_sq > 0 else float(D)

        # Rank from eigenspectrum
        k = max(1, int(round(D_eff * rank_multiplier)))
        k = min(k, D)  # Can't exceed dimensionality

        # Apply correction
        proj_stats = _correct_layer(
            layer, layer_idx, fp_weights, eigvecs, eigvals, k,
        )
        total_corrected += sum(1 for s in proj_stats if not s.get("skipped"))

        # Forward pass with corrected weights for next layer
        h = layer(h)
        mx.eval(h)

        layer_time = time.monotonic() - layer_start

        corrected_stats = [s for s in proj_stats if not s.get("skipped")]
        skipped_stats = [s for s in proj_stats if s.get("skipped")]

        per_layer_results.append({
            "layer_idx": layer_idx,
            "D": D,
            "D_eff": D_eff,
            "k": k,
            "rank_multiplier": rank_multiplier,
            "n_projections_corrected": len(corrected_stats),
            "n_projections_skipped": len(skipped_stats),
            "projection_stats": proj_stats,
            "mean_correction_fraction": (
                sum(s["correction_fraction"] for s in corrected_stats)
                / len(corrected_stats)
                if corrected_stats else 0.0
            ),
            "time_seconds": layer_time,
        })

        if layer_idx % 7 == 0 or layer_idx == n_layers - 1:
            mean_frac = per_layer_results[-1]["mean_correction_fraction"]
            n_corr = per_layer_results[-1]["n_projections_corrected"]
            n_skip = per_layer_results[-1]["n_projections_skipped"]
            logger.info(
                "  Layer %d/%d: D_eff=%.1f, k=%d, correction_frac=%.4f, "
                "corrected=%d, skipped=%d (%.1fs)",
                layer_idx, n_layers - 1, D_eff, k, mean_frac,
                n_corr, n_skip, layer_time,
            )

        del X, XtX, eigvals, eigvecs
        gc.collect()
        _clear_gpu_cache()

    return {
        "rank_multiplier": rank_multiplier,
        "n_layers": n_layers,
        "n_projections_corrected": total_corrected,
        "per_layer": per_layer_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-Form Sequential Layer Correction",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED,
        help="Path to quantized model",
    )
    parser.add_argument(
        "--fp-model",
        default=DEFAULT_FP,
        help="Path to full-precision (bf16) model",
    )
    parser.add_argument(
        "--eval-dataset",
        default=DEFAULT_EVAL,
        help="Path to evaluation dataset (JSONL) for CKA and calibration",
    )
    parser.add_argument(
        "--output-dir",
        default="results/closedform_sequential_correction",
        help="Base output directory",
    )
    parser.add_argument(
        "--rank-multipliers",
        default="1,2,5,10,20",
        help="Comma-separated D_eff multipliers for rank sweep",
    )
    parser.add_argument(
        "--n-calibration",
        type=int,
        default=30,
        help=(
            "Number of calibration samples for activation covariance. "
            "CLI-overridable, not a decision boundary. 30 is >10x "
            "oversampled for D_eff~3 (measured on Qwen3-1.7B)."
        ),
    )
    parser.add_argument(
        "--n-cka-samples",
        type=int,
        default=30,
        help=(
            "Number of samples for CKA measurement. "
            "CLI-overridable, not a decision boundary."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help=(
            "Maximum sequence length for calibration. "
            "Memory-compute tradeoff, CLI-overridable."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_multipliers = [float(x) for x in args.rank_multipliers.split(",")]

    logger.info("Closed-Form Sequential Correction — run_id=%s", run_id)
    logger.info("Quantized model: %s", args.quantized_model)
    logger.info("FP model: %s", args.fp_model)
    logger.info("Rank multipliers: %s", rank_multipliers)
    logger.info("Output: %s", output_dir)

    # Load evaluation texts
    eval_texts = _load_eval_texts(args.eval_dataset, max(args.n_calibration, args.n_cka_samples))
    logger.info("Loaded %d evaluation texts", len(eval_texts))

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "closedform_sequential_correction",
        "config": {
            "quantized_model": args.quantized_model,
            "fp_model": args.fp_model,
            "eval_dataset": args.eval_dataset,
            "rank_multipliers": rank_multipliers,
            "n_calibration": args.n_calibration,
            "n_cka_samples": args.n_cka_samples,
            "max_seq_len": args.max_seq_len,
        },
    }

    # ── Load FP model and extract reference weights ──
    logger.info("Loading FP model...")
    fp_model, fp_tokenizer = backend.load_model(args.fp_model)

    # Collect FP activations for CKA baseline
    logger.info("Collecting FP reference activations...")
    fp_acts = _collect_activations(
        fp_model, fp_tokenizer, eval_texts, backend, n_samples=args.n_cka_samples,
    )

    fp_weights = _extract_fp_weights(fp_model, None)
    logger.info("Extracted %d FP weight matrices", len(fp_weights))

    # Free FP model (keep weights and activations)
    del fp_model
    gc.collect()
    _clear_gpu_cache()

    # ── Rank sweep ──
    sweep_results: list[dict[str, Any]] = []

    for mult_idx, multiplier in enumerate(rank_multipliers):
        logger.info("\n" + "=" * 60)
        logger.info(
            "RANK SWEEP %d/%d: multiplier=%.1f",
            mult_idx + 1, len(rank_multipliers), multiplier,
        )
        logger.info("=" * 60)

        sweep_start = time.monotonic()

        # Load fresh quantized model for each sweep point
        logger.info("Loading quantized model (fresh copy)...")
        q_model, q_tokenizer = backend.load_model(args.quantized_model)

        # Collect quantized activations for CKA baseline
        logger.info("Collecting quantized baseline activations...")
        q_acts_before = _collect_activations(
            q_model, q_tokenizer, eval_texts, backend, n_samples=args.n_cka_samples,
        )
        baseline_cka = _compute_cka(fp_acts, q_acts_before, backend)
        logger.info(
            "Baseline CKA: mean=%.4f, min=%.4f",
            baseline_cka["mean_cka"], baseline_cka["min_cka"],
        )
        del q_acts_before

        # Dequantize model weights (so we can modify them)
        from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
        adapter = MLXTrainingAdapter(backend)

        n_deq = _dequantize_model(q_model, adapter)
        logger.info("Dequantized %d layers", n_deq)

        # Run sequential correction
        correction_result = _run_sequential_correction(
            q_model,
            fp_weights,
            q_tokenizer,
            eval_texts,
            rank_multiplier=multiplier,
            n_samples=args.n_calibration,
            max_len=args.max_seq_len,
        )

        # Measure post-correction CKA
        logger.info("Collecting post-correction activations...")
        q_acts_after = _collect_activations(
            q_model, q_tokenizer, eval_texts, backend, n_samples=args.n_cka_samples,
        )
        post_cka = _compute_cka(fp_acts, q_acts_after, backend)
        logger.info(
            "Post-correction CKA: mean=%.4f, min=%.4f",
            post_cka["mean_cka"], post_cka["min_cka"],
        )
        del q_acts_after

        sweep_time = time.monotonic() - sweep_start

        sweep_entry = {
            "rank_multiplier": multiplier,
            "baseline_cka": baseline_cka,
            "post_cka": post_cka,
            "cka_delta_mean": post_cka["mean_cka"] - baseline_cka["mean_cka"],
            "cka_delta_min": post_cka["min_cka"] - baseline_cka["min_cka"],
            "correction": correction_result,
            "wall_time_seconds": sweep_time,
        }
        sweep_results.append(sweep_entry)

        logger.info(
            "SWEEP %d result: mult=%.1f, CKA %.4f → %.4f (delta=%+.4f), "
            "%.1fs",
            mult_idx + 1,
            multiplier,
            baseline_cka["mean_cka"],
            post_cka["mean_cka"],
            post_cka["mean_cka"] - baseline_cka["mean_cka"],
            sweep_time,
        )

        # Free model
        del q_model, q_tokenizer, adapter
        gc.collect()
        _clear_gpu_cache()

    results["sweep"] = sweep_results

    # ── Summary ──
    logger.info("\n" + "=" * 72)
    logger.info("CLOSED-FORM SEQUENTIAL CORRECTION — SUMMARY")
    logger.info("=" * 72)

    print("\n" + "=" * 72)
    print("CLOSED-FORM SEQUENTIAL CORRECTION — CKA vs RANK")
    print("=" * 72)
    print(f"{'Multiplier':>12} {'k (avg)':>10} {'Baseline CKA':>14} "
          f"{'Post CKA':>10} {'Delta':>10} {'Time':>8}")
    print("-" * 72)

    for entry in sweep_results:
        layers = entry["correction"]["per_layer"]
        avg_k = sum(l["k"] for l in layers) / len(layers) if layers else 0
        print(
            f"{entry['rank_multiplier']:>12.1f} "
            f"{avg_k:>10.1f} "
            f"{entry['baseline_cka']['mean_cka']:>14.4f} "
            f"{entry['post_cka']['mean_cka']:>10.4f} "
            f"{entry['cka_delta_mean']:>+10.4f} "
            f"{entry['wall_time_seconds']:>7.1f}s"
        )

    print("=" * 72)
    # Source: results/stacked_corrective_recovery/20260226T134604Z/stacked_recovery.json
    print("\nReference: 5-round stacked corrective recovery CKA delta +0.023 (measured)")
    print()

    # ── Write results ──
    output_path = output_dir / "closedform_correction.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
