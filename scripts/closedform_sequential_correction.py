#!/usr/bin/env python3
"""Experiment: Closed-Form Sequential Layer Correction (Tikhonov).

Computes analytical weight corrections per-layer, sequentially, using
eigenvalue-weighted Tikhonov projection in the activation eigenbasis.

Algorithm:
    For each layer l (0 → L-1), sequentially:
      1. Forward pass through layers 0..l-1 → activations X_l
      2. E_l = W_fp_l - W_quantized_l  (per-projection weight error)
      3. Eigendecompose X_l^T @ X_l → eigenvectors V, eigenvalues λ
      4. Marchenko-Pastur noise edge: α = σ² × (1 + √(D/n))²
         where σ² = trace(C)/D (average eigenvalue), D/n = aspect ratio
      5. Tikhonov weights: w_i = λ_i / (λ_i + α)  (continuous, no integer rank)
      6. Delta_l = E_l @ V @ diag(w) @ V^T
      7. W_corrected_l = W_quantized_l + Delta_l
      8. Continue to layer l+1 with corrected model

Every number is derived from the data (eigenvalues) or from Marchenko-Pastur
theory (Marchenko & Pastur, 1967). No rank sweep. No integer rank. No re-noise.
Directions with small eigenvalues get w_i → 0, automatically preserving the
quantization residual in those directions.

Usage:
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.training.tikhonov_correction import (
    compute_mp_noise_edge,
    compute_tikhonov_weights,
    correct_projection_tikhonov,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("closedform_correction")

# Default paths
HISTORICAL_QUANTIZED_MODEL = (
    "results/four_bit_extension/20260226T023950Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-4bit-g64-affine"
)
DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"

PROJ_NAMES_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
PROJ_NAMES_MLP = ("up_proj", "down_proj", "gate_proj")
ALL_PROJ_NAMES = PROJ_NAMES_ATTN + PROJ_NAMES_MLP
TEST_PROMPTS = [
    "Explain what a prime number is.",
    "What causes the seasons on Earth?",
    "Describe how a binary search works.",
]


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


def _evaluate_ppl_inplace(
    model: Any, tokenizer: Any, texts: list[str], backend: Any,
) -> dict[str, float]:
    """Compute perplexity on an in-memory model (no disk load)."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens_arr = backend.array(tokens)
        input_arr = backend.reshape(tokens_arr, (1, -1))
        logits = model(input_arr)
        logits = logits[0, :-1, :]
        targets = tokens_arr[1:]
        log_scores = backend.log_softmax(logits, axis=-1)
        targets_expanded = backend.reshape(targets, (-1, 1))
        target_log_scores = backend.take_along_axis(
            log_scores, targets_expanded, axis=-1,
        )
        target_log_scores = backend.squeeze(target_log_scores, axis=-1)
        backend.eval(target_log_scores)
        mean_arr = backend.mean(target_log_scores)
        backend.eval(mean_arr)
        sample_loss = -float(backend.to_scalar(mean_arr))
        n_targets = int(targets.shape[0])
        total_loss += sample_loss * n_targets
        total_tokens += n_targets
    avg_loss = total_loss / max(total_tokens, 1)
    ppl_arr = backend.exp(backend.array([avg_loss]))
    backend.eval(ppl_arr)
    return {
        "average_loss": avg_loss,
        "perplexity": float(backend.to_scalar(ppl_arr)),
        "n_tokens": total_tokens,
    }


def _fourgram_repetition_rate(text: str) -> float:
    """Delegates to domain module — hardcoded n=4 for this experiment."""
    from modelcypher.core.domain.training.degeneration import ngram_repetition_rate

    return ngram_repetition_rate(text, 4)


def _generate_responses(
    model: Any, tokenizer: Any, prompts: list[str], backend: Any,
    max_tokens: int = 256,
) -> list[str]:
    """Generate responses from an in-memory model."""
    responses = []
    for prompt in prompts:
        resp = backend.generate(model, tokenizer, prompt, max_tokens)
        responses.append(resp)
    return responses


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
    eigvecs: mx.array,
    tikhonov_weights_arr: mx.array,
    backend: Any,
) -> dict[str, Any] | None:
    """Apply eigenvalue-weighted Tikhonov correction to a single projection.

    Delegates core math to domain module (tikhonov_correction.py).
    Handles MLX-specific weight mutation and memory cleanup.

    Returns diagnostics dict or None if skipped.
    """
    fp_w = fp_weights.get(key)
    if fp_w is None:
        return None

    corrected, layer_result = correct_projection_tikhonov(
        quantized_weight=proj.weight,
        fp_weight=fp_w,
        eigenvectors=eigvecs,
        tikhonov_weights=tikhonov_weights_arr,
        backend=backend,
        layer_key=key,
    )
    if layer_result is None:
        return None

    # Apply correction to model weight (MLX-specific mutation)
    mx.eval(corrected)
    proj.weight = corrected.astype(proj.weight.dtype)

    result = {
        "layer_key": key,
        "E_total_frob": layer_result.E_total_frob,
        "delta_frob": layer_result.delta_frob,
        "E_residual_frob": layer_result.E_residual_frob,
        "correction_fraction": layer_result.correction_fraction,
        "preserved_fraction": layer_result.preserved_fraction,
    }

    return result


def _correct_layer(
    layer,
    layer_idx: int,
    fp_weights: dict[str, mx.array],
    eigvecs: mx.array,
    tikhonov_weights: mx.array,
    backend: Any,
) -> list[dict[str, Any]]:
    """Apply Tikhonov-weighted correction to projections in a layer.

    Only projections whose input IS h (or layer_norm(h), which preserves the
    subspace) are corrected:
      - self_attn: q_proj, k_proj, v_proj (input = layer_norm(h))
      - mlp: up_proj, gate_proj (input = post_attention_norm(h))

    Skipped (input is a different space):
      - o_proj: input = attention output (different subspace)
      - down_proj: input = SiLU(up(x)) * gate(x) (MLP intermediate)

    Returns per-projection diagnostics.
    """
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
            result = _correct_projection(
                proj,
                key,
                fp_weights,
                eigvecs,
                tikhonov_weights,
                backend,
            )
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
    n_samples: int,
    max_len: int,
) -> dict[str, Any]:
    """Run eigenvalue-weighted Tikhonov sequential correction.

    Per layer:
      1. Compute activation covariance eigenbasis
      2. Derive Marchenko-Pastur noise edge α from the spectrum
      3. Compute Tikhonov weights w_i = λ_i / (λ_i + α)
      4. Apply weighted projection to each correctable weight matrix

    Returns per-layer correction stats including eigenspectrum and MP edge.
    """
    _backend = initialize_default_backend()

    base = getattr(q_model, "model", q_model)
    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute")

    n_layers = len(base.layers)
    logger.info(
        "Sequential Tikhonov correction: %d layers, %d calibration samples",
        n_layers, n_samples,
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
    total_e_sq = 0.0
    total_delta_sq = 0.0
    total_residual_sq = 0.0

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
            X = None
            XtX = None
            gc.collect()
            continue

        # eigh returns ascending order; flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Clamp negative eigenvalues to 0 (numerical noise from eigh)
        eigvals = mx.maximum(eigvals, mx.array(0.0, dtype=eigvals.dtype))
        mx.eval(eigvals)

        # Participation ratio (D_eff) — diagnostic only, not used for projection
        total_var = float(mx.sum(eigvals).item())
        sum_sq = float(mx.sum(eigvals * eigvals).item())
        D_eff = total_var ** 2 / sum_sq if sum_sq > 0 else float(D)

        # Marchenko-Pastur noise edge + Tikhonov weights (domain module)
        mp_edge = compute_mp_noise_edge(
            eigvals, n_tokens=N_tok, dimensionality=D, backend=_backend,
        )
        sigma_sq = total_var / D
        aspect = D / N_tok
        tikhonov_weights = compute_tikhonov_weights(eigvals, mp_edge, backend=_backend)

        # Effective rank from Tikhonov (sum of weights — diagnostic)
        effective_rank = float(mx.sum(tikhonov_weights).item())

        # Top eigenvalue weights for logging
        n_report = min(10, D)
        top_eigvals = [float(eigvals[i].item()) for i in range(n_report)]
        top_weights = [float(tikhonov_weights[i].item()) for i in range(n_report)]

        # Apply correction
        proj_stats = _correct_layer(
            layer, layer_idx, fp_weights, eigvecs, tikhonov_weights, _backend,
        )
        total_corrected += sum(1 for s in proj_stats if not s.get("skipped"))

        # Forward pass with corrected weights for next layer
        h = layer(h)
        mx.eval(h)

        layer_time = time.monotonic() - layer_start

        corrected_stats = [s for s in proj_stats if not s.get("skipped")]
        skipped_stats = [s for s in proj_stats if s.get("skipped")]
        layer_e_sq = sum(
            float(s.get("E_total_frob", 0.0)) ** 2 for s in corrected_stats
        )
        layer_delta_sq = sum(
            float(s.get("delta_frob", 0.0)) ** 2 for s in corrected_stats
        )
        layer_residual_sq = sum(
            float(s.get("E_residual_frob", 0.0)) ** 2 for s in corrected_stats
        )
        total_e_sq += layer_e_sq
        total_delta_sq += layer_delta_sq
        total_residual_sq += layer_residual_sq

        per_layer_results.append({
            "layer_idx": layer_idx,
            "D": D,
            "D_eff": D_eff,
            "mp_edge": mp_edge,
            "sigma_sq": sigma_sq,
            "aspect_ratio": aspect,
            "effective_rank": effective_rank,
            "top_eigenvalues": top_eigvals,
            "top_tikhonov_weights": top_weights,
            "n_projections_corrected": len(corrected_stats),
            "n_projections_skipped": len(skipped_stats),
            "projection_stats": proj_stats,
            "mean_correction_fraction": (
                sum(s["correction_fraction"] for s in corrected_stats)
                / len(corrected_stats)
                if corrected_stats else 0.0
            ),
            "E_total_frob": math.sqrt(layer_e_sq),
            "delta_frob": math.sqrt(layer_delta_sq),
            "E_residual_frob": math.sqrt(layer_residual_sq),
            "correction_fraction": (
                (layer_delta_sq / layer_e_sq) if layer_e_sq > 0.0 else 0.0
            ),
            "preserved_fraction": (
                (layer_residual_sq / layer_e_sq) if layer_e_sq > 0.0 else 0.0
            ),
            "time_seconds": layer_time,
        })

        if layer_idx % 7 == 0 or layer_idx == n_layers - 1:
            mean_frac = per_layer_results[-1]["mean_correction_fraction"]
            n_corr = per_layer_results[-1]["n_projections_corrected"]
            n_skip = per_layer_results[-1]["n_projections_skipped"]
            logger.info(
                "  Layer %d/%d: D_eff=%.1f, mp_edge=%.2e, eff_rank=%.1f, "
                "correction_frac=%.4f, corrected=%d, skipped=%d (%.1fs)",
                layer_idx, n_layers - 1, D_eff, mp_edge, effective_rank,
                mean_frac, n_corr, n_skip, layer_time,
            )

        del X, XtX, eigvals, eigvecs, tikhonov_weights
        gc.collect()
        _clear_gpu_cache()

    return {
        "n_layers": n_layers,
        "n_projections_corrected": total_corrected,
        "aggregate": {
            "E_total_frob": math.sqrt(total_e_sq),
            "delta_frob": math.sqrt(total_delta_sq),
            "E_residual_frob": math.sqrt(total_residual_sq),
            "correction_fraction": (
                (total_delta_sq / total_e_sq) if total_e_sq > 0.0 else 0.0
            ),
            "preserved_fraction": (
                (total_residual_sq / total_e_sq) if total_e_sq > 0.0 else 0.0
            ),
        },
        "per_layer": per_layer_results,
    }


def _measure_eval_bundle(
    *,
    model: Any,
    tokenizer: Any,
    backend: Any,
    eval_texts: list[str],
    fp_acts: dict[int, list],
    n_cka_samples: int,
) -> dict[str, Any]:
    acts = _collect_activations(
        model,
        tokenizer,
        eval_texts,
        backend,
        n_samples=n_cka_samples,
    )
    cka = _compute_cka(fp_acts, acts, backend)
    del acts
    ppl = _evaluate_ppl_inplace(model, tokenizer, eval_texts, backend)
    responses = _generate_responses(model, tokenizer, TEST_PROMPTS, backend, max_tokens=256)
    repeat_rates = [_fourgram_repetition_rate(r) for r in responses]
    max_repeat = max(repeat_rates) if repeat_rates else 0.0
    mean_repeat = sum(repeat_rates) / len(repeat_rates) if repeat_rates else 0.0
    return {
        "cka": cka,
        "ppl": ppl,
        "degeneration": {
            "max_4gram_repeat": max_repeat,
            "mean_4gram_repeat": mean_repeat,
            "responses": responses,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-Form Sequential Layer Correction (Tikhonov)",
    )
    parser.add_argument(
        "--quantized-model",
        required=True,
        help=(
            "Path to quantized model. Pass it explicitly; the historical "
            f"in-repo artifact {HISTORICAL_QUANTIZED_MODEL} is retained only "
            "as provenance in results, not as a live model directory."
        ),
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
        "--n-calibration",
        type=int,
        default=30,
        help=(
            "Number of calibration samples for activation covariance. "
            "CLI-overridable, not a decision boundary. 30 >> D_eff~3 "
            "(measured on Qwen3-1.7B). MP edge estimation requires "
            "n > D_eff (Marchenko & Pastur, 1967)."
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

    logger.info("Closed-Form Tikhonov Correction — run_id=%s", run_id)
    logger.info("Quantized model: %s", args.quantized_model)
    logger.info("FP model: %s", args.fp_model)
    logger.info("Projection: eigenvalue-weighted Tikhonov (Marchenko-Pastur)")
    logger.info("Output: %s", output_dir)

    # Load evaluation texts
    eval_texts = _load_eval_texts(
        args.eval_dataset, max(args.n_calibration, args.n_cka_samples),
    )
    logger.info("Loaded %d evaluation texts", len(eval_texts))

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "closedform_tikhonov_correction",
        "method": {
            "projection": "tikhonov",
            "regularization": "marchenko_pastur_noise_edge",
            "formula": "w_i = lambda_i / (lambda_i + alpha)",
            "alpha": "sigma_sq * (1 + sqrt(D/n))^2",
            "citation": "Marchenko & Pastur, 1967",
        },
        "config": {
            "quantized_model": args.quantized_model,
            "fp_model": args.fp_model,
            "eval_dataset": args.eval_dataset,
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

    # FP reference metrics
    logger.info("Computing FP reference PPL...")
    fp_ppl = _evaluate_ppl_inplace(fp_model, fp_tokenizer, eval_texts, backend)
    fp_responses = _generate_responses(
        fp_model, fp_tokenizer, TEST_PROMPTS, backend, max_tokens=256,
    )
    fp_repeat_rates = [_fourgram_repetition_rate(r) for r in fp_responses]
    fp_max_repeat = max(fp_repeat_rates) if fp_repeat_rates else 0.0

    fp_weights = _extract_fp_weights(fp_model, None)
    logger.info("Extracted %d FP weight matrices", len(fp_weights))

    # Free FP model (keep weights and activations)
    del fp_model
    gc.collect()
    _clear_gpu_cache()

    # ── Load quantized model ──
    logger.info("Loading quantized model...")
    q_model, q_tokenizer = backend.load_model(args.quantized_model)

    # Baseline metrics
    logger.info("Measuring quantized baseline metrics...")
    baseline_bundle = _measure_eval_bundle(
        model=q_model,
        tokenizer=q_tokenizer,
        backend=backend,
        eval_texts=eval_texts,
        fp_acts=fp_acts,
        n_cka_samples=args.n_cka_samples,
    )
    baseline_cka = baseline_bundle["cka"]
    baseline_ppl = baseline_bundle["ppl"]
    baseline_degeneration = baseline_bundle["degeneration"]
    logger.info(
        "Baseline CKA: mean=%.4f, min=%.4f",
        baseline_cka["mean_cka"], baseline_cka["min_cka"],
    )
    logger.info("Baseline PPL: %.4f", baseline_ppl["perplexity"])

    # Dequantize model weights (so we can modify them)
    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    adapter = MLXTrainingAdapter(backend)

    n_deq = _dequantize_model(q_model, adapter)
    logger.info("Dequantized %d layers", n_deq)

    # ── Single correction run (one formula, no sweep) ──
    run_start = time.monotonic()
    correction_result = _run_sequential_correction(
        q_model,
        fp_weights,
        q_tokenizer,
        eval_texts,
        n_samples=args.n_calibration,
        max_len=args.max_seq_len,
    )
    run_time = time.monotonic() - run_start

    # Post-correction metrics
    logger.info("Measuring post-correction metrics...")
    post_bundle = _measure_eval_bundle(
        model=q_model,
        tokenizer=q_tokenizer,
        backend=backend,
        eval_texts=eval_texts,
        fp_acts=fp_acts,
        n_cka_samples=args.n_cka_samples,
    )
    post_cka = post_bundle["cka"]
    post_ppl = post_bundle["ppl"]
    post_degeneration = post_bundle["degeneration"]
    logger.info(
        "Post-correction CKA: mean=%.4f, min=%.4f",
        post_cka["mean_cka"], post_cka["min_cka"],
    )
    logger.info("Post-correction PPL: %.4f", post_ppl["perplexity"])
    logger.info(
        # TODO(jk): 4-gram window n=4 not derived — diagnostic only,
        # not a decision boundary.
        "4-gram repetition (diagnostic): max=%.4f, mean=%.4f",
        post_degeneration["max_4gram_repeat"],
        post_degeneration["mean_4gram_repeat"],
    )

    # Free model
    del q_model, q_tokenizer, adapter
    gc.collect()
    _clear_gpu_cache()

    # ── Assemble results ──
    results["reference"] = {
        "fp_ppl": fp_ppl,
        "fp_max_4gram_repeat": fp_max_repeat,
        "fp_responses": fp_responses,
    }
    results["baseline"] = {
        "cka": baseline_cka,
        "ppl": baseline_ppl,
        "degeneration": baseline_degeneration,
    }
    results["correction"] = correction_result
    results["post_correction"] = {
        "cka": post_cka,
        "ppl": post_ppl,
        "degeneration": post_degeneration,
    }
    results["deltas"] = {
        "cka_mean": post_cka["mean_cka"] - baseline_cka["mean_cka"],
        "cka_min": post_cka["min_cka"] - baseline_cka["min_cka"],
        "ppl": post_ppl["perplexity"] - baseline_ppl["perplexity"],
        "max_4gram_repeat": (
            post_degeneration["max_4gram_repeat"]
            - baseline_degeneration["max_4gram_repeat"]
        ),
    }
    results["wall_time_seconds"] = run_time

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("CLOSED-FORM TIKHONOV CORRECTION — RESULTS")
    print("=" * 90)
    print(f"{'':>12} {'CKA mean':>10} {'CKA min':>10} "
          f"{'PPL':>8} {'4g-rep':>8}")
    print("-" * 90)
    print(f"{'FP ref':>12} {'1.0000':>10} {'1.0000':>10} "
          f"{fp_ppl['perplexity']:>8.2f} {fp_max_repeat:>8.4f}")
    print(f"{'Q baseline':>12} "
          f"{baseline_cka['mean_cka']:>10.4f} "
          f"{baseline_cka['min_cka']:>10.4f} "
          f"{baseline_ppl['perplexity']:>8.2f} "
          f"{baseline_degeneration['max_4gram_repeat']:>8.4f}")
    print(f"{'Tikhonov':>12} "
          f"{post_cka['mean_cka']:>10.4f} "
          f"{post_cka['min_cka']:>10.4f} "
          f"{post_ppl['perplexity']:>8.2f} "
          f"{post_degeneration['max_4gram_repeat']:>8.4f}")
    print("-" * 90)
    print(f"{'Delta':>12} "
          f"{results['deltas']['cka_mean']:>+10.4f} "
          f"{results['deltas']['cka_min']:>+10.4f} "
          f"{results['deltas']['ppl']:>+8.2f} "
          f"{results['deltas']['max_4gram_repeat']:>+8.4f}")
    print("=" * 90)

    # Per-layer eigenspectrum summary
    print("\nPer-layer Marchenko-Pastur profile:")
    print(f"{'Layer':>6} {'D_eff':>8} {'MP edge':>12} {'Eff rank':>10} "
          f"{'Corr frac':>10} {'Top w':>8}")
    print("-" * 70)
    for layer_result in correction_result["per_layer"]:
        top_w = layer_result["top_tikhonov_weights"][0] if layer_result["top_tikhonov_weights"] else 0.0
        print(
            f"{layer_result['layer_idx']:>6} "
            f"{layer_result['D_eff']:>8.1f} "
            f"{layer_result['mp_edge']:>12.2e} "
            f"{layer_result['effective_rank']:>10.1f} "
            f"{layer_result['correction_fraction']:>10.4f} "
            f"{top_w:>8.4f}"
        )
    print("-" * 70)

    # Source: results/stacked_corrective_recovery/20260226T134604Z/stacked_recovery.json
    print(f"\nWall time: {run_time:.1f}s")
    print("Reference: 5-round stacked corrective recovery CKA delta +0.023 (measured)")
    print()

    # ── Write results ──
    output_path = output_dir / "closedform_correction.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
