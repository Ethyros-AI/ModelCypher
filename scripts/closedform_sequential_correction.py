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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from scipy.stats import spearmanr

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
    """Fraction of 4-grams in text that are repeated."""
    # TODO(jk): derive n-gram window from measured trajectory geometry instead
    # of fixing n=4.
    words = text.split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)


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
    E_unused = E - Delta
    mx.eval(E_unused)

    Delta_frob_sq = float(mx.sum(Delta * Delta).item())
    E_unused_frob_sq = float(mx.sum(E_unused * E_unused).item())
    correction_fraction = Delta_frob_sq / E_frob_sq
    unused_fraction = E_unused_frob_sq / E_frob_sq

    # Apply correction
    corrected = q_w + Delta
    mx.eval(corrected)
    proj.weight = corrected.astype(proj.weight.dtype)

    # Residual after correction
    residual = fp_w_f32 - corrected
    residual_frob = float(mx.sqrt(mx.sum(residual * residual)).item())
    reconstruction = Delta + E_unused
    recon_error = E - reconstruction
    recon_error_frob = float(mx.sqrt(mx.sum(recon_error * recon_error)).item())

    result = {
        "layer_key": key,
        "error_frob": math.sqrt(E_frob_sq),
        "delta_frob": math.sqrt(Delta_frob_sq),
        "E_total_frob": math.sqrt(E_frob_sq),
        "E_used_frob": math.sqrt(Delta_frob_sq),
        "E_unused_frob": math.sqrt(E_unused_frob_sq),
        "correction_fraction": correction_fraction,
        "E_used_fraction": correction_fraction,
        "E_unused_fraction": unused_fraction,
        "decomposition_reconstruction_error_frob": recon_error_frob,
        "residual_frob": residual_frob,
    }

    del E, E_proj, Delta, E_unused, corrected, q_w, fp_w_f32, residual
    del reconstruction, recon_error
    return result


def _correct_layer(
    layer,
    layer_idx: int,
    fp_weights: dict[str, mx.array],
    eigvecs: mx.array,
    eigvals: mx.array,
    k: int,
    *,
    target_layer_idx: int | None = None,
    apply_renoise: bool = False,
    renoise_seed: int = 0,
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
    if target_layer_idx is not None and layer_idx != target_layer_idx:
        return stats

    D = int(eigvecs.shape[0])
    if apply_renoise:
        mx.random.seed(renoise_seed + layer_idx)
        eye = mx.eye(D, dtype=mx.float32)
        proj_used = V_k @ V_k.T
        proj_unused = eye - proj_used
        mx.eval(proj_unused)
    else:
        proj_unused = None

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
                if apply_renoise and proj_unused is not None:
                    w = proj.weight.astype(mx.float32)
                    if int(w.shape[1]) == D:
                        g = mx.random.normal(shape=w.shape).astype(mx.float32)
                        noise = g @ proj_unused
                        noise_norm_sq = float(mx.sum(noise * noise).item())
                        target_norm = float(result.get("E_unused_frob", 0.0))
                        if noise_norm_sq > 0.0 and target_norm > 0.0:
                            noise = noise * (target_norm / math.sqrt(noise_norm_sq))
                            w = w + noise
                            mx.eval(w)
                            proj.weight = w.astype(proj.weight.dtype)
                            result["renoise_frob"] = target_norm
                        else:
                            result["renoise_frob"] = 0.0
                    else:
                        result["renoise_frob"] = 0.0
                stats.append(result)

    # Log skipped projections
    if target_layer_idx is None:
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
    *,
    target_layer_idx: int | None = None,
    apply_renoise: bool = False,
    renoise_seed: int = 0,
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
    total_e_sq = 0.0
    total_used_sq = 0.0
    total_unused_sq = 0.0
    total_recon_error_sq = 0.0

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
            target_layer_idx=target_layer_idx,
            apply_renoise=apply_renoise,
            renoise_seed=renoise_seed,
        )
        total_corrected += sum(1 for s in proj_stats if not s.get("skipped"))

        # Forward pass with corrected weights for next layer
        h = layer(h)
        mx.eval(h)

        layer_time = time.monotonic() - layer_start

        corrected_stats = [s for s in proj_stats if not s.get("skipped")]
        skipped_stats = [s for s in proj_stats if s.get("skipped")]
        layer_e_sq = sum(float(s.get("E_total_frob", 0.0)) ** 2 for s in corrected_stats)
        layer_used_sq = sum(float(s.get("E_used_frob", 0.0)) ** 2 for s in corrected_stats)
        layer_unused_sq = sum(float(s.get("E_unused_frob", 0.0)) ** 2 for s in corrected_stats)
        layer_recon_error_sq = sum(
            float(s.get("decomposition_reconstruction_error_frob", 0.0)) ** 2
            for s in corrected_stats
        )
        total_e_sq += layer_e_sq
        total_used_sq += layer_used_sq
        total_unused_sq += layer_unused_sq
        total_recon_error_sq += layer_recon_error_sq

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
            "E_total_frob": math.sqrt(layer_e_sq),
            "E_used_frob": math.sqrt(layer_used_sq),
            "E_unused_frob": math.sqrt(layer_unused_sq),
            "E_used_fraction": (layer_used_sq / layer_e_sq) if layer_e_sq > 0.0 else 0.0,
            "E_unused_fraction": (
                (layer_unused_sq / layer_e_sq) if layer_e_sq > 0.0 else 0.0
            ),
            "decomposition_reconstruction_error_frob": math.sqrt(layer_recon_error_sq),
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
        "target_layer_idx": target_layer_idx,
        "apply_renoise": apply_renoise,
        "n_projections_corrected": total_corrected,
        "decomposition": {
            "E_total_frob": math.sqrt(total_e_sq),
            "E_used_frob": math.sqrt(total_used_sq),
            "E_unused_frob": math.sqrt(total_unused_sq),
            "E_used_fraction": (total_used_sq / total_e_sq) if total_e_sq > 0.0 else 0.0,
            "E_unused_fraction": (
                (total_unused_sq / total_e_sq) if total_e_sq > 0.0 else 0.0
            ),
            "decomposition_reconstruction_error_frob": math.sqrt(total_recon_error_sq),
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


def _compute_layer_repeat_correlation(
    *,
    decomposition_layers: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    intervention_by_layer = {
        int(item["layer_idx"]): item for item in interventions
    }
    x_vals: list[float] = []
    y_vals: list[float] = []
    used_layers: list[int] = []
    for layer in decomposition_layers:
        layer_idx = int(layer["layer_idx"])
        intervention = intervention_by_layer.get(layer_idx)
        if intervention is None:
            continue
        x_vals.append(float(layer.get("E_unused_frob", 0.0)))
        y_vals.append(float(intervention.get("delta_max_4gram_repeat", 0.0)))
        used_layers.append(layer_idx)
    if len(x_vals) < 2:
        return {
            "n_layers": len(x_vals),
            "spearman_rho": None,
            "p_value": None,
            "layers": used_layers,
        }
    rho, p_value = spearmanr(x_vals, y_vals)
    rho_val = float(rho) if rho == rho else None
    p_val = float(p_value) if p_value == p_value else None
    return {
        "n_layers": len(x_vals),
        "spearman_rho": rho_val,
        "p_value": p_val,
        "layers": used_layers,
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
            "4-gram repetition: max=%.4f, mean=%.4f",
            post_degeneration["max_4gram_repeat"],
            post_degeneration["mean_4gram_repeat"],
        )

        # Controlled re-noise in unused subspace.
        logger.info("Running controlled re-noise pass...")
        renoise_model, renoise_tokenizer = backend.load_model(args.quantized_model)
        renoise_adapter = MLXTrainingAdapter(backend)
        _dequantize_model(renoise_model, renoise_adapter)
        renoise_correction = _run_sequential_correction(
            renoise_model,
            fp_weights,
            renoise_tokenizer,
            eval_texts,
            rank_multiplier=multiplier,
            n_samples=args.n_calibration,
            max_len=args.max_seq_len,
            apply_renoise=True,
            renoise_seed=1337 + mult_idx,
        )
        renoise_bundle = _measure_eval_bundle(
            model=renoise_model,
            tokenizer=renoise_tokenizer,
            backend=backend,
            eval_texts=eval_texts,
            fp_acts=fp_acts,
            n_cka_samples=args.n_cka_samples,
        )
        del renoise_model, renoise_tokenizer, renoise_adapter
        gc.collect()
        _clear_gpu_cache()

        # Single-layer interventions (one corrected layer at a time).
        logger.info("Running single-layer intervention sweep...")
        single_layer_interventions: list[dict[str, Any]] = []
        for layer_idx in range(int(correction_result["n_layers"])):
            il_model, il_tokenizer = backend.load_model(args.quantized_model)
            il_adapter = MLXTrainingAdapter(backend)
            _dequantize_model(il_model, il_adapter)
            intervention_correction = _run_sequential_correction(
                il_model,
                fp_weights,
                il_tokenizer,
                eval_texts,
                rank_multiplier=multiplier,
                n_samples=args.n_calibration,
                max_len=args.max_seq_len,
                target_layer_idx=layer_idx,
            )
            intervention_bundle = _measure_eval_bundle(
                model=il_model,
                tokenizer=il_tokenizer,
                backend=backend,
                eval_texts=eval_texts,
                fp_acts=fp_acts,
                n_cka_samples=args.n_cka_samples,
            )
            single_layer_interventions.append(
                {
                    "layer_idx": layer_idx,
                    "correction": intervention_correction,
                    "metrics": intervention_bundle,
                    "delta_mean_cka": (
                        intervention_bundle["cka"]["mean_cka"]
                        - baseline_cka["mean_cka"]
                    ),
                    "delta_min_cka": (
                        intervention_bundle["cka"]["min_cka"]
                        - baseline_cka["min_cka"]
                    ),
                    "delta_ppl": (
                        intervention_bundle["ppl"]["perplexity"]
                        - baseline_ppl["perplexity"]
                    ),
                    "delta_max_4gram_repeat": (
                        intervention_bundle["degeneration"]["max_4gram_repeat"]
                        - baseline_degeneration["max_4gram_repeat"]
                    ),
                }
            )
            del il_model, il_tokenizer, il_adapter
            gc.collect()
            _clear_gpu_cache()

        correlations = {
            "unused_vs_repeat_delta": _compute_layer_repeat_correlation(
                decomposition_layers=correction_result["per_layer"],
                interventions=single_layer_interventions,
            ),
        }

        sweep_time = time.monotonic() - sweep_start

        sweep_entry = {
            "rank_multiplier": multiplier,
            "baseline_cka": baseline_cka,
            "post_cka": post_cka,
            "cka_delta_mean": post_cka["mean_cka"] - baseline_cka["mean_cka"],
            "cka_delta_min": post_cka["min_cka"] - baseline_cka["min_cka"],
            "post_ppl": post_ppl,
            "baseline_ppl": baseline_ppl,
            "degeneration": post_degeneration,
            "baseline_degeneration": baseline_degeneration,
            "correction": correction_result,
            "decomposition": correction_result["decomposition"],
            "single_layer_interventions": single_layer_interventions,
            "correlations": correlations,
            "renoise": {
                "correction": renoise_correction,
                "metrics": renoise_bundle,
                "delta_mean_cka": (
                    renoise_bundle["cka"]["mean_cka"] - baseline_cka["mean_cka"]
                ),
                "delta_min_cka": (
                    renoise_bundle["cka"]["min_cka"] - baseline_cka["min_cka"]
                ),
                "delta_ppl": (
                    renoise_bundle["ppl"]["perplexity"] - baseline_ppl["perplexity"]
                ),
                "delta_max_4gram_repeat": (
                    renoise_bundle["degeneration"]["max_4gram_repeat"]
                    - baseline_degeneration["max_4gram_repeat"]
                ),
            },
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
    results["decomposition"] = {
        str(entry["rank_multiplier"]): entry["decomposition"]
        for entry in sweep_results
    }
    results["single_layer_interventions"] = {
        str(entry["rank_multiplier"]): entry["single_layer_interventions"]
        for entry in sweep_results
    }
    results["correlations"] = {
        str(entry["rank_multiplier"]): entry["correlations"]
        for entry in sweep_results
    }
    results["renoise"] = {
        str(entry["rank_multiplier"]): entry["renoise"]
        for entry in sweep_results
    }

    # ── Summary ──
    logger.info("\n" + "=" * 72)
    logger.info("CLOSED-FORM SEQUENTIAL CORRECTION — SUMMARY")
    logger.info("=" * 72)

    # Also compute FP and baseline-quantized PPL for reference
    logger.info("Computing FP and baseline-quantized PPL for reference...")
    fp_model_ref, fp_tok_ref = backend.load_model(args.fp_model)
    fp_ppl = _evaluate_ppl_inplace(fp_model_ref, fp_tok_ref, eval_texts, backend)
    fp_responses = _generate_responses(
        fp_model_ref, fp_tok_ref,
        TEST_PROMPTS,
        backend, max_tokens=256,
    )
    fp_max_repeat = max(_fourgram_repetition_rate(r) for r in fp_responses)
    del fp_model_ref, fp_tok_ref
    gc.collect()
    _clear_gpu_cache()

    q_model_ref, q_tok_ref = backend.load_model(args.quantized_model)
    q_base_ppl = _evaluate_ppl_inplace(q_model_ref, q_tok_ref, eval_texts, backend)
    q_responses = _generate_responses(
        q_model_ref, q_tok_ref,
        TEST_PROMPTS,
        backend, max_tokens=256,
    )
    q_max_repeat = max(_fourgram_repetition_rate(r) for r in q_responses)
    del q_model_ref, q_tok_ref
    gc.collect()
    _clear_gpu_cache()

    results["reference_ppl"] = {
        "fp": fp_ppl,
        "quantized_baseline": q_base_ppl,
        "fp_max_4gram_repeat": fp_max_repeat,
        "quantized_max_4gram_repeat": q_max_repeat,
    }

    print("\n" + "=" * 90)
    print("CLOSED-FORM SEQUENTIAL CORRECTION — GATE TABLE")
    print("=" * 90)
    print(f"{'Mult':>6} {'k(avg)':>8} {'CKA mean':>10} {'CKA min':>10} "
          f"{'PPL':>8} {'4g-rep':>8} {'CKA Δ':>8} {'Time':>7}")
    print("-" * 90)
    print(f"{'FP ref':>6} {'':>8} {'1.0000':>10} {'1.0000':>10} "
          f"{fp_ppl['perplexity']:>8.2f} {fp_max_repeat:>8.4f} "
          f"{'':>8} {'':>7}")
    print(f"{'Q base':>6} {'':>8} "
          f"{sweep_results[0]['baseline_cka']['mean_cka']:>10.4f} "
          f"{sweep_results[0]['baseline_cka']['min_cka']:>10.4f} "
          f"{q_base_ppl['perplexity']:>8.2f} {q_max_repeat:>8.4f} "
          f"{'':>8} {'':>7}")
    print("-" * 90)

    for entry in sweep_results:
        layers = entry["correction"]["per_layer"]
        avg_k = sum(l["k"] for l in layers) / len(layers) if layers else 0
        print(
            f"{entry['rank_multiplier']:>6.1f} "
            f"{avg_k:>8.1f} "
            f"{entry['post_cka']['mean_cka']:>10.4f} "
            f"{entry['post_cka']['min_cka']:>10.4f} "
            f"{entry['post_ppl']['perplexity']:>8.2f} "
            f"{entry['degeneration']['max_4gram_repeat']:>8.4f} "
            f"{entry['cka_delta_mean']:>+8.4f} "
            f"{entry['wall_time_seconds']:>6.1f}s"
        )

    print("=" * 90)
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
