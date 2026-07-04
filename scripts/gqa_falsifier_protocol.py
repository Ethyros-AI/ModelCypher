#!/usr/bin/env python3
"""F-GQA-01: GQA Falsifier Protocol.

Pre-registered falsifier test for CR-EC-001: does GQA ratio modulate
entropy-curvature coupling?

Two observables:
    z_couple = atanh(corr(H_logit, H_attn))  — operator coupling strength
    c_cancel = |beta_num - beta_den|          — cancellation completeness

Three falsifiers:
    F1: b_g >= 0 with CI excluding zero → coupling-direction claim FALSIFIED
    F2: d_g <= 0 with CI excluding zero → cancellation-direction claim FALSIFIED
    F3: within-family contradiction under both Spearman and depth-controlled Pearson

Design matrix: z = a + b_g*log(GQA) + b_h*I(hybrid) + b_s*log(d)
    (Reduced model — no interaction term; 9 models, DOF=5.)

Usage:
    poetry run python scripts/gqa_falsifier_protocol.py
    poetry run python scripts/gqa_falsifier_protocol.py --smoke
    poetry run python scripts/gqa_falsifier_protocol.py --collect-missing
    poetry run python scripts/gqa_falsifier_protocol.py --include-smollm3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import validate_gqa_falsifier_artifacts as artifact_validator
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")
RESULTS_BASE = Path("results")
OPERATOR_SPLIT_DIR = RESULTS_BASE / "entropy_curvature_operator_split"
CURVATURE_ACCUM_DIR = RESULTS_BASE / "curvature_accumulation_operator_full_multifamily"
OUTPUT_BASE = RESULTS_BASE / "gqa_falsifier_protocol"


# ============================================================================
# Model registry (locked, pre-registered)
# ============================================================================

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024, "GQA": 2,
        "hybrid": True, "family": "lfm2", "quant": "bf16",
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16, "d": 1280, "GQA": 3,
        "hybrid": True, "family": "lfm2", "quant": "bf16",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072, "GQA": 3,
        "hybrid": False, "family": "llama", "quant": "bf16",
    },
    "Mistral-7B": {
        "path": f"{MODELS_BASE}/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "L": 32, "d": 4096, "GQA": 4,
        "hybrid": False, "family": "mistral", "quant": "4-bit",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "L": 36, "d": 4096, "GQA": 4,
        "hybrid": False, "family": "qwen3", "quant": "bf16",
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024, "GQA": 4,
        "hybrid": True, "family": "qwen3.5", "quant": "bf16",
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24, "d": 2048, "GQA": 4,
        "hybrid": True, "family": "qwen3.5", "quant": "bf16",
    },
    "Qwen3.5-4B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-bf16",
        "L": 32, "d": 2560, "GQA": 4,
        "hybrid": True, "family": "qwen3.5", "quant": "bf16",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048, "GQA": 8,
        "hybrid": False, "family": "qwen2.5", "quant": "bf16",
    },
}

SMOLLM3_ENTRY = {
    "SmolLM3-3B": {
        "path": f"{MODELS_BASE}/mlx-community/SmolLM3-3B-bf16",
        "L": 36, "d": 2048, "GQA": 4,
        "hybrid": False, "family": "smollm3", "quant": "bf16",
    },
}

# Curvature accumulation has cached c_cancel data for these models
CURVATURE_ACCUM_MODELS = {"LFM2-350M", "Llama-3.2-3B", "Qwen2.5-3B", "Qwen3-8B"}


# ============================================================================
# Section 1: Utilities (verbatim from entropy_curvature_operator_split.py:62-96)
# ============================================================================


def _residualize_ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing linear effect of x."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _effective_df(residuals: np.ndarray) -> tuple[float, float]:
    """Effective degrees of freedom under AR(1) autocorrelation.

    Bretherton et al. (1999) Eq. 31: n_eff = n * (1 - rho_1) / (1 + rho_1).
    """
    n = len(residuals)
    if n < 4:
        return float(n), 0.0
    r = np.asarray(residuals, dtype=float)
    r_centered = r - np.mean(r)
    denom = np.sum(r_centered ** 2)
    if abs(denom) < 1e-15:
        return float(n), 0.0
    rho_1 = float(np.sum(r_centered[:-1] * r_centered[1:]) / denom)
    rho_1_clamped = max(-0.99, min(0.99, rho_1))
    n_eff = n * (1 - rho_1_clamped) / (1 + rho_1_clamped)
    # Floor at 4 (minimum for partial correlation), ceiling at n
    n_eff = max(4.0, min(float(n), n_eff))
    return n_eff, rho_1


def _minimum_detectable_effect(n_eff: float, n_controls: int = 1) -> float:
    """Fisher-SE MDE: smallest |r| with SNR >= 1 at given effective df."""
    df = n_eff - n_controls - 2
    if df <= 0:
        return 1.0
    return float(np.tanh(1.0 / np.sqrt(df)))


# ============================================================================
# Section 2: z_couple computation
# ============================================================================


def _load_vocab_size(model_name: str) -> int | None:
    """Load vocab_size from model config.json.

    Handles standard config (vocab_size at top level) and Qwen3.5-style
    (vocab_size nested under text_config).
    """
    info = MODEL_REGISTRY.get(model_name) or SMOLLM3_ENTRY.get(model_name)
    if info is None:
        return None
    config_path = Path(info["path"]) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    v = cfg.get("vocab_size")
    if v is not None:
        return int(v)
    tc = cfg.get("text_config", {})
    v = tc.get("vocab_size")
    if v is not None:
        return int(v)
    return None


def compute_z_couple(model_name: str) -> dict:
    """Compute operator coupling z_couple from stored operator_split data.

    Loads operator_split.json, filters to attention layers, depth-residualizes
    H_logit and H_attn, computes Pearson r, Fisher-z transform, Bretherton
    n_eff, and MDE.

    Also computes commensurability gate: the depth-residualized H_logit range
    must exceed log(2) nats for the entropy operator to resolve at least one
    bit of posterior concentration variation across layers. This is a post-hoc
    measurement-operator validity criterion (not part of the pre-registered
    protocol). Below log(2), z_couple correlates precision-level fluctuations
    in a near-saturated entropy signal.
    """
    op_path = OPERATOR_SPLIT_DIR / model_name / "operator_split.json"
    if not op_path.exists():
        logger.warning("No operator_split.json for %s", model_name)
        return {"model": model_name, "z_couple": None, "error": "missing_data"}

    with open(op_path) as f:
        data = json.load(f)

    measurements = data["measurements"]

    # Filter to attention layers with valid H_logit and H_attn
    attn_layers = [
        m for m in measurements
        if m["is_attention_layer"] and m["H_attn"] is not None and m["H_logit"] is not None
    ]

    if len(attn_layers) < 4:
        logger.warning("%s: only %d attention layers, need >= 4", model_name, len(attn_layers))
        return {"model": model_name, "z_couple": None, "error": "insufficient_layers"}

    depth = np.array([m["depth_fraction"] for m in attn_layers])
    h_logit = np.array([m["H_logit"] for m in attn_layers])
    h_attn = np.array([m["H_attn"] for m in attn_layers])

    # Depth-residualize both operators
    h_logit_resid = _residualize_ols(h_logit, depth)
    h_attn_resid = _residualize_ols(h_attn, depth)

    # --- Commensurability gate ---
    # Applied to the SAME depth-residualized signal used for z_couple,
    # not the raw H_logit. This is a post-hoc measurement-operator validity
    # criterion, not part of the pre-registered protocol.
    #
    # Criterion: depth-residualized H_logit range must exceed log(2) nats
    # (one bit of posterior concentration variation). Below this, the
    # entropy operator doesn't resolve meaningful concentration differences
    # across layers — z_couple is correlating noise.
    h_logit_resid_range = float(h_logit_resid.max() - h_logit_resid.min())
    commensurable = h_logit_resid_range >= np.log(2)

    # H_logit saturation ratio: mean(H_logit) / log(V) across attention layers.
    # When s ≈ 1, D_KL(p||uniform) → 0 and k_eff ≈ V everywhere.
    vocab_size = _load_vocab_size(model_name)
    if vocab_size is not None and vocab_size > 1:
        h_logit_saturation = float(np.mean(h_logit) / np.log(vocab_size))
    else:
        h_logit_saturation = None

    # Pearson correlation on residuals
    r_val, p_val = sp_stats.pearsonr(h_logit_resid, h_attn_resid)

    # Fisher-z transform: z = atanh(r)
    r_clamped = max(-0.999, min(0.999, r_val))
    z_couple = float(np.arctanh(r_clamped))

    # Bretherton n_eff from combined residual
    combined_resid = h_logit_resid * h_attn_resid
    n_eff_z, rho_1 = _effective_df(combined_resid)

    # MDE
    mde = _minimum_detectable_effect(n_eff_z, n_controls=1)

    # Fisher-z SE = 1/sqrt(n_eff - 3)
    se_z = 1.0 / math.sqrt(max(1, n_eff_z - 3)) if n_eff_z > 3 else float("inf")

    return {
        "model": model_name,
        "n_attn_layers": len(attn_layers),
        "r_pearson": float(r_val),
        "p_pearson": float(p_val),
        "z_couple": z_couple,
        "n_eff_z": float(n_eff_z),
        "rho_1": float(rho_1),
        "se_z": se_z,
        "mde": mde,
        "above_mde": bool(abs(r_val) >= mde),
        # Commensurability gate (post-hoc measurement-operator validity)
        "commensurable": bool(commensurable),
        "h_logit_resid_range": h_logit_resid_range,
        "h_logit_saturation": h_logit_saturation,
    }


# ============================================================================
# Section 3: c_cancel computation
# ============================================================================


def compute_c_cancel_from_cached(model_name: str) -> dict | None:
    """Compute c_cancel from cached curvature_accumulation data.

    c_cancel = |beta_num - beta_den| where:
        beta_num = OLS slope of log||P_perp(h)delta||^2 on H_logit (depth-residualized)
        beta_den = OLS slope of log||h_in||^2 on H_logit (depth-residualized)

    Returns None if model not in cached results.
    """
    cache_path = CURVATURE_ACCUM_DIR / "curvature_accumulation_results.json"
    if not cache_path.exists():
        return None

    with open(cache_path) as f:
        data = json.load(f)

    model_data = None
    for m in data["models"]:
        if m["model_name"] == model_name:
            model_data = m
            break

    if model_data is None:
        return None

    return _compute_c_cancel_from_layers(model_name, model_data["measurements"])


def _compute_c_cancel_from_layers(model_name: str, layers: list[dict]) -> dict:
    """Compute c_cancel from layer measurements with log_perp_delta_sq and log_h_in_sq."""
    # Filter to layers with valid data AND H_logit
    valid = [
        l for l in layers
        if l.get("mean_log_perp_delta_sq") is not None
        and l.get("mean_log_h_in_sq") is not None
        and l.get("mean_h_logit") is not None
    ]

    if len(valid) < 4:
        return {
            "model": model_name,
            "c_cancel": None,
            "error": "insufficient_layers",
            "n_valid": len(valid),
        }

    depth = np.array([l["layer_idx"] / max(l["layer_idx"] for l in valid) for l in valid])
    log_perp = np.array([l["mean_log_perp_delta_sq"] for l in valid])
    log_hin = np.array([l["mean_log_h_in_sq"] for l in valid])
    h_logit = np.array([l["mean_h_logit"] for l in valid])

    # Depth-residualize all three
    log_perp_resid = _residualize_ols(log_perp, depth)
    log_hin_resid = _residualize_ols(log_hin, depth)
    h_logit_resid = _residualize_ols(h_logit, depth)

    # OLS slopes: regress each component on H_logit (depth-residualized)
    X_h = np.column_stack([np.ones(len(h_logit_resid)), h_logit_resid])
    beta_num = np.linalg.lstsq(X_h, log_perp_resid, rcond=None)[0][1]
    beta_den = np.linalg.lstsq(X_h, log_hin_resid, rcond=None)[0][1]

    c_cancel = abs(float(beta_num) - float(beta_den))

    # Bretherton n_eff for c_cancel (separate from z_couple, Correction #2)
    # Use the larger residual for autocorrelation estimate
    resid_num = log_perp_resid - X_h @ np.linalg.lstsq(X_h, log_perp_resid, rcond=None)[0]
    n_eff_c, rho_1_c = _effective_df(resid_num)

    return {
        "model": model_name,
        "beta_num": float(beta_num),
        "beta_den": float(beta_den),
        "c_cancel": c_cancel,
        "n_layers_valid": len(valid),
        "n_eff_c": float(n_eff_c),
        "rho_1_c": float(rho_1_c),
    }


# ============================================================================
# Section 4: GPU collection for missing c_cancel
# ============================================================================


def _resolve_backbone(model):
    """Resolve model backbone to the level with embed_tokens and layers.

    Handles standard (model.model), Qwen3.5 (model.model.language_model.model),
    and other nesting patterns.
    """
    base = getattr(model, "model", None)
    if base is not None:
        if getattr(base, "layers", None) is not None and getattr(base, "embed_tokens", None) is not None:
            return base
        lm = getattr(base, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if inner is not None and getattr(inner, "layers", None) is not None:
                return inner
            if getattr(lm, "layers", None) is not None:
                return lm
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            return inner
        if getattr(lm, "layers", None) is not None:
            return lm
    return model


def collect_c_cancel_gpu(model_name: str, model_info: dict) -> dict:
    """Collect log||P_perp(h)delta||^2 and log||h_in||^2 via GPU forward pass.

    Lightweight h_in/h_out loop ported from curvature_accumulation_analysis.py:620-699.
    Only collects the minimal data needed for c_cancel computation.
    """
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        return {"model": model_name, "c_cancel": None, "error": "mlx_not_available"}

    logger.info("GPU collection for %s ...", model_name)

    model_path = model_info["path"]
    if not Path(model_path).exists():
        return {"model": model_name, "c_cancel": None, "error": "model_not_found"}

    model, tokenizer = mlx_lm.load(model_path)

    # Resolve backbone (same logic as entropy_curvature_operator_split.py)
    base = _resolve_backbone(model)

    layers = base.layers
    num_layers = len(layers)
    embed = getattr(base, "embed_tokens", None)
    if embed is None:
        return {"model": model_name, "c_cancel": None, "error": "no_embed_tokens"}

    # Probes — same as curvature_accumulation
    probes = [
        "The capital of France is",
        "Who wrote Romeo and Juliet?",
        "The chemical symbol for water is",
        "The largest planet in our solar system is",
        "What is 347 + 528?",
        "What is 15 * 23?",
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Write a haiku about the ocean.",
        "Describe a sunset over the mountains in one vivid sentence.",
        "Once upon a time in a faraway kingdom, there lived a",
        "The old lighthouse keeper watched the storm approach from",
    ]

    # Per-layer accumulators
    layer_log_perp = [[] for _ in range(num_layers)]
    layer_log_hin = [[] for _ in range(num_layers)]
    layer_h_logit = [[] for _ in range(num_layers)]

    eps = 1e-12

    for probe_text in probes:
        tokens = tokenizer.encode(probe_text)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        mx.eval(hidden)

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden

            # Per-layer mask routing (LFM2 compatibility)
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = mx.array(
                    np.triu(np.full((hidden.shape[1], hidden.shape[1]), -1e9), k=1)
                ).astype(hidden.dtype)

            try:
                h_out = layer(h_in, mask=layer_mask)
            except (TypeError, ValueError):
                try:
                    h_out = layer(h_in, layer_mask)
                except (TypeError, ValueError):
                    h_out = layer(h_in)

            if isinstance(h_out, tuple):
                h_out = h_out[0]

            # Last-token representations
            h_in_last = h_in[:, -1, :].astype(mx.float32)
            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_in_last, h_out_last)

            h_vec = np.array(h_in_last[0].tolist(), dtype=np.float32)
            out_vec = np.array(h_out_last[0].tolist(), dtype=np.float32)
            delta_vec = out_vec - h_vec

            h_sq = float(np.dot(h_vec, h_vec))
            if h_sq > eps:
                proj_coeff = float(np.dot(delta_vec, h_vec) / h_sq)
                delta_perp = delta_vec - proj_coeff * h_vec
                perp_sq = float(np.dot(delta_perp, delta_perp))

                layer_log_perp[i].append(float(np.log(perp_sq + eps)))
                layer_log_hin[i].append(float(np.log(h_sq + eps)))

            # Logit entropy (project through unembedding)
            try:
                final_norm = getattr(base, "norm", None) or getattr(base, "embedding_norm", None)
                if final_norm is not None:
                    normed = final_norm(h_out_last.reshape(1, 1, -1))
                else:
                    normed = h_out_last.reshape(1, 1, -1)

                lm_head = getattr(model, "lm_head", None)
                if lm_head is not None:
                    logits = lm_head(normed)
                elif hasattr(base, "embed_tokens") and hasattr(base.embed_tokens, "as_linear"):
                    logits = base.embed_tokens.as_linear(normed)
                else:
                    logits = normed @ embed.weight.T

                mx.eval(logits)
                logits_np = np.array(logits[0, 0].tolist(), dtype=np.float32)
                logits_np = logits_np - np.max(logits_np)
                probs = np.exp(logits_np) / np.sum(np.exp(logits_np))
                h_logit = float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
                layer_h_logit[i].append(h_logit)
            except Exception:
                layer_h_logit[i].append(None)

            hidden = h_out

        mx.eval(hidden)

    # Build layer measurements
    layer_measurements = []
    for i in range(num_layers):
        valid_logit = [v for v in layer_h_logit[i] if v is not None]
        layer_measurements.append({
            "layer_idx": i,
            "mean_log_perp_delta_sq": float(np.mean(layer_log_perp[i])) if layer_log_perp[i] else None,
            "mean_log_h_in_sq": float(np.mean(layer_log_hin[i])) if layer_log_hin[i] else None,
            "mean_h_logit": float(np.mean(valid_logit)) if valid_logit else None,
        })

    # Clean up GPU memory
    del model, tokenizer
    import gc
    gc.collect()

    return _compute_c_cancel_from_layers(model_name, layer_measurements)


# ============================================================================
# Section 5: Design matrix assembly
# ============================================================================


def assemble_design_matrix(
    z_results: dict[str, dict],
    c_results: dict[str, dict],
    registry: dict[str, dict],
) -> list[dict]:
    """Assemble per-model records for regression."""
    records = []
    for name, info in registry.items():
        z = z_results.get(name, {})
        c = c_results.get(name, {})

        record = {
            "model": name,
            "GQA": info["GQA"],
            "log_GQA": float(np.log(info["GQA"])),
            "hybrid": info["hybrid"],
            "I_hybrid": 1 if info["hybrid"] else 0,
            "d_model": info["d"],
            "log_d": float(np.log(info["d"])),
            "family": info["family"],
            "quant": info["quant"],
            "L": info["L"],
            # z_couple
            "z_couple": z.get("z_couple"),
            "n_eff_z": z.get("n_eff_z"),
            "se_z": z.get("se_z"),
            "mde_z": z.get("mde"),
            "r_pearson": z.get("r_pearson"),
            "above_mde_z": z.get("above_mde"),
            # Commensurability (post-hoc measurement-operator validity)
            "commensurable": z.get("commensurable"),
            "h_logit_resid_range": z.get("h_logit_resid_range"),
            "h_logit_saturation": z.get("h_logit_saturation"),
            # c_cancel
            "c_cancel": c.get("c_cancel"),
            "beta_num": c.get("beta_num"),
            "beta_den": c.get("beta_den"),
            "n_eff_c": c.get("n_eff_c"),
        }
        records.append(record)

    return records


# ============================================================================
# Section 6: Design matrix diagnostics
# ============================================================================


def design_diagnostics(records: list[dict], response_key: str) -> dict:
    """Compute and report design matrix diagnostics BEFORE regression.

    Reports: cond(X'X), eigenspectrum, VIF per predictor, smallest eigenvector,
    predictor correlation matrix.
    """
    # Filter to records with valid response
    valid = [r for r in records if r.get(response_key) is not None]
    n = len(valid)

    if n < 4:
        return {"n": n, "error": "insufficient_data"}

    # Predictors: intercept, log(GQA), I(hybrid), log(d)
    pred_names = ["intercept", "log_GQA", "I_hybrid", "log_d"]
    X = np.column_stack([
        np.ones(n),
        np.array([r["log_GQA"] for r in valid]),
        np.array([r["I_hybrid"] for r in valid]),
        np.array([r["log_d"] for r in valid]),
    ])

    XtX = X.T @ X
    eigvals = np.linalg.eigvalsh(XtX)
    eigvals_sorted = np.sort(eigvals)[::-1]
    cond_XtX = float(eigvals_sorted[0] / max(eigvals_sorted[-1], 1e-15))

    # Full eigendecomposition for smallest eigenvector
    eigvals_full, eigvecs = np.linalg.eigh(XtX)
    smallest_idx = np.argmin(eigvals_full)
    smallest_eigvec = eigvecs[:, smallest_idx]

    # VIF per predictor (from auxiliary R^2)
    vifs = {}
    X_no_intercept = X[:, 1:]  # Drop intercept for VIF
    for j in range(X_no_intercept.shape[1]):
        others = np.delete(X_no_intercept, j, axis=1)
        X_aux = np.column_stack([np.ones(n), others])
        beta_aux = np.linalg.lstsq(X_aux, X_no_intercept[:, j], rcond=None)[0]
        fitted = X_aux @ beta_aux
        ss_res = np.sum((X_no_intercept[:, j] - fitted) ** 2)
        ss_tot = np.sum((X_no_intercept[:, j] - np.mean(X_no_intercept[:, j])) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)
        vifs[pred_names[j + 1]] = float(1.0 / max(1.0 - r_sq, 1e-15))

    # Predictor correlation matrix (no intercept)
    corr_matrix = np.corrcoef(X_no_intercept.T)

    return {
        "n": n,
        "predictors": pred_names,
        "cond_XtX": cond_XtX,
        "eigenspectrum": [float(e) for e in eigvals_sorted],
        "smallest_eigenvector": {
            pred_names[i]: float(smallest_eigvec[i])
            for i in range(len(pred_names))
        },
        "VIF": vifs,
        "predictor_correlation": {
            f"{pred_names[i+1]}_vs_{pred_names[j+1]}": float(corr_matrix[i, j])
            for i in range(corr_matrix.shape[0])
            for j in range(i + 1, corr_matrix.shape[1])
        },
    }


# ============================================================================
# Section 7: Regressions (weighted OLS)
# ============================================================================


def weighted_ols_regression(
    records: list[dict],
    response_key: str,
    weight_key: str,
    label: str,
) -> dict:
    """Weighted OLS: response = a + b_g*log(GQA) + b_h*I(hybrid) + b_s*log(d).

    Weights = n_eff (Bretherton). CIs via t-distribution.
    """
    valid = [r for r in records if r.get(response_key) is not None and r.get(weight_key) is not None]
    n = len(valid)

    if n < 5:
        return {
            "label": label,
            "n": n,
            "error": "insufficient_data",
            "note": f"Need >= 5 models for DOF > 0, have {n}",
        }

    y = np.array([r[response_key] for r in valid])
    w = np.array([r[weight_key] for r in valid])

    # Predictor matrix: [1, log(GQA), I(hybrid), log(d)]
    X = np.column_stack([
        np.ones(n),
        np.array([r["log_GQA"] for r in valid]),
        np.array([r["I_hybrid"] for r in valid]),
        np.array([r["log_d"] for r in valid]),
    ])
    pred_names = ["intercept", "b_g", "b_h", "b_s"]

    # Weighted OLS: beta = (X'WX)^{-1} X'Wy
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

    # Residuals and sigma^2
    residuals = y - X @ beta
    p = X.shape[1]
    dof = n - p
    if dof <= 0:
        return {
            "label": label, "n": n, "dof": dof,
            "error": "zero_dof",
            "coefficients": {pred_names[i]: float(beta[i]) for i in range(p)},
        }

    sigma_sq = float(np.sum(w * residuals ** 2) / dof)

    # Covariance matrix: sigma^2 * (X'WX)^{-1}
    try:
        cov_beta = sigma_sq * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov_beta = sigma_sq * np.linalg.pinv(XtWX)

    se = np.sqrt(np.diag(cov_beta))

    # t-statistics and CIs
    t_crit = float(sp_stats.t.ppf(0.975, dof))
    coefficients = {}
    for i, name in enumerate(pred_names):
        t_stat = float(beta[i] / se[i]) if se[i] > 0 else float("inf")
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), dof)))
        ci_lo = float(beta[i] - t_crit * se[i])
        ci_hi = float(beta[i] + t_crit * se[i])
        coefficients[name] = {
            "estimate": float(beta[i]),
            "se": float(se[i]),
            "t_stat": t_stat,
            "p_value": p_val,
            "ci_95": [ci_lo, ci_hi],
        }

    r_squared = 1.0 - float(np.sum(w * residuals ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2))

    return {
        "label": label,
        "n": n,
        "dof": dof,
        "sigma": float(np.sqrt(sigma_sq)),
        "r_squared": r_squared,
        "coefficients": coefficients,
        "models_used": [r["model"] for r in valid],
    }


# ============================================================================
# Section 8: Falsifier adjudication
# ============================================================================


def adjudicate_falsifiers(
    z_regression: dict,
    c_regression: dict,
    records: list[dict],
) -> dict:
    """Adjudicate F1, F2, F3 falsifiers.

    F1: b_g >= 0 with CI excluding zero → coupling claim FALSIFIED
    F2: d_g <= 0 with CI excluding zero → cancellation claim FALSIFIED
    F3: within-family contradiction (LFM2 only family with GQA variation)
    """
    outcomes = {}

    # F1: b_g in z_couple regression
    if "error" not in z_regression:
        bg = z_regression["coefficients"]["b_g"]
        ci = bg["ci_95"]
        if ci[0] > 0:
            # b_g > 0 with CI excluding zero → FALSIFIED
            outcomes["F1"] = {
                "status": "FALSIFIED",
                "reason": f"b_g = {bg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] excludes zero, sign positive",
            }
        elif ci[1] < 0:
            # b_g < 0 with CI excluding zero → SUPPORTED (predicted direction)
            outcomes["F1"] = {
                "status": "SUPPORTED",
                "reason": f"b_g = {bg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] excludes zero, sign negative (predicted)",
            }
        else:
            outcomes["F1"] = {
                "status": "INCONCLUSIVE",
                "reason": f"b_g = {bg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] crosses zero",
            }
    else:
        outcomes["F1"] = {
            "status": "UNDERPOWERED",
            "reason": z_regression.get("error", "regression_failed"),
        }

    # F2: d_g in c_cancel regression
    if "error" not in c_regression:
        dg = c_regression["coefficients"]["b_g"]  # b_g in c_cancel regression = d_g
        ci = dg["ci_95"]
        if ci[1] < 0:
            # d_g < 0 with CI excluding zero → FALSIFIED
            outcomes["F2"] = {
                "status": "FALSIFIED",
                "reason": f"d_g = {dg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] excludes zero, sign negative",
            }
        elif ci[0] > 0:
            # d_g > 0 with CI excluding zero → SUPPORTED (predicted direction)
            outcomes["F2"] = {
                "status": "SUPPORTED",
                "reason": f"d_g = {dg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] excludes zero, sign positive (predicted)",
            }
        else:
            outcomes["F2"] = {
                "status": "INCONCLUSIVE",
                "reason": f"d_g = {dg['estimate']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] crosses zero",
            }
    else:
        outcomes["F2"] = {
            "status": "UNDERPOWERED",
            "reason": c_regression.get("error", "regression_failed") + f" (n={c_regression.get('n', 0)})",
        }

    # F3: within-family (LFM2 is the only family with GQA variation: GQA=2 and GQA=3)
    lfm2 = [r for r in records if r["family"] == "lfm2" and r.get("z_couple") is not None]
    if len(lfm2) >= 2:
        # Sort by GQA
        lfm2_sorted = sorted(lfm2, key=lambda r: r["GQA"])

        # Check commensurability first: if both models have saturated H_logit
        # (depth-residualized range < log(2) nats), z_couple is correlating
        # precision-level fluctuations in a near-uniform distribution.
        # The observable is not the same quantity as z_couple for desaturated
        # models — comparison is mathematically invalid.
        all_incommensurable = all(
            r.get("commensurable") is False for r in lfm2_sorted
        )

        if all_incommensurable:
            outcomes["F3"] = {
                "status": "INCOMMENSURABLE",
                "reason": (
                    "Both LFM2 models have saturated H_logit "
                    "(depth-residualized range < log(2) nats). "
                    "z_couple correlates precision-level fluctuations, "
                    "not posterior concentration gradients. "
                    "Within-family comparison is mathematically invalid."
                ),
                "models": [
                    {
                        "model": r["model"], "GQA": r["GQA"],
                        "z_couple": r["z_couple"],
                        "h_logit_resid_range": r.get("h_logit_resid_range"),
                        "h_logit_saturation": r.get("h_logit_saturation"),
                        "commensurable": r.get("commensurable"),
                    }
                    for r in lfm2_sorted
                ],
            }
        # Check if both are below MDE
        elif all(not r.get("above_mde_z", False) for r in lfm2_sorted):
            outcomes["F3"] = {
                "status": "BELOW_MEASUREMENT_RESOLUTION",
                "reason": "Both LFM2 models below MDE — within-family trend not resolvable",
                "models": [
                    {"model": r["model"], "GQA": r["GQA"], "z_couple": r["z_couple"],
                     "mde": r.get("mde_z")}
                    for r in lfm2_sorted
                ],
            }
        else:
            # Check monotonicity: z_couple should decrease with GQA
            z_values = [r["z_couple"] for r in lfm2_sorted]
            gqa_values = [r["GQA"] for r in lfm2_sorted]

            if len(z_values) >= 3:
                spearman_r, _ = sp_stats.spearmanr(gqa_values, z_values)
            else:
                # Only 2 points: sign comparison
                spearman_r = -1.0 if z_values[1] < z_values[0] else 1.0

            # Depth-controlled Pearson: already computed as z_couple (which is depth-residualized)
            # Prediction: z_couple decreases with GQA → negative trend
            trend_sign = "negative" if z_values[-1] < z_values[0] else "positive"

            if trend_sign == "positive" and spearman_r > 0:
                outcomes["F3"] = {
                    "status": "FALSIFIED",
                    "reason": f"Within-family LFM2: z_couple increases with GQA (contradicts prediction). "
                              f"Spearman r={spearman_r:.3f}",
                    "models": [
                        {"model": r["model"], "GQA": r["GQA"], "z_couple": r["z_couple"]}
                        for r in lfm2_sorted
                    ],
                }
            elif trend_sign == "negative":
                outcomes["F3"] = {
                    "status": "CONSISTENT",
                    "reason": f"Within-family LFM2: z_couple decreases with GQA (consistent with prediction). "
                              f"Spearman r={spearman_r:.3f}",
                    "models": [
                        {"model": r["model"], "GQA": r["GQA"], "z_couple": r["z_couple"]}
                        for r in lfm2_sorted
                    ],
                }
            else:
                outcomes["F3"] = {
                    "status": "INCONCLUSIVE",
                    "reason": f"Within-family LFM2: mixed signals. Spearman r={spearman_r:.3f}",
                    "models": [
                        {"model": r["model"], "GQA": r["GQA"], "z_couple": r["z_couple"]}
                        for r in lfm2_sorted
                    ],
                }
    else:
        outcomes["F3"] = {
            "status": "INSUFFICIENT_DATA",
            "reason": f"Only {len(lfm2)} LFM2 model(s) with z_couple data",
        }

    return outcomes


# ============================================================================
# Section 9: Artifact emission
# ============================================================================


def emit_artifacts(
    run_id: str,
    records: list[dict],
    z_regression: dict,
    c_regression: dict,
    falsifier_outcomes: dict,
    z_diagnostics: dict,
    c_diagnostics: dict,
    z_regression_comm: dict | None = None,
) -> Path:
    """Write 4 JSON files per protocol spec."""
    out_dir = OUTPUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model_table.json
    model_table = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "protocol": "F-GQA-01",
        "n_models": len(records),
        "n_commensurable": sum(1 for r in records if r.get("commensurable") is True),
        "n_incommensurable": sum(1 for r in records if r.get("commensurable") is False),
        "commensurability_threshold": "log(2) nats on depth-residualized H_logit range",
        "models": records,
    }
    with open(out_dir / "model_table.json", "w") as f:
        json.dump(model_table, f, indent=2)
    logger.info("Wrote model_table.json (%d models)", len(records))

    # 2. regression_summary.json
    regression_summary = {
        "run_id": run_id,
        "z_couple_regression_full": z_regression,
        "z_couple_regression_commensurable": z_regression_comm,
        "c_cancel_regression": c_regression,
        "z_couple_diagnostics": z_diagnostics,
        "c_cancel_diagnostics": c_diagnostics,
        "commensurability_note": (
            "z_couple_regression_full includes incommensurable models "
            "(H_logit saturated, z_couple correlates noise). Use for exploratory "
            "analysis only. z_couple_regression_commensurable is the scientifically "
            "valid comparison but may be underpowered."
        ),
    }
    with open(out_dir / "regression_summary.json", "w") as f:
        json.dump(regression_summary, f, indent=2)
    logger.info("Wrote regression_summary.json")

    # 3. within_family_trends.json
    # Group by family
    families = {}
    for r in records:
        fam = r["family"]
        if fam not in families:
            families[fam] = []
        families[fam].append({
            "model": r["model"],
            "GQA": r["GQA"],
            "z_couple": r.get("z_couple"),
            "c_cancel": r.get("c_cancel"),
        })
    with open(out_dir / "within_family_trends.json", "w") as f:
        json.dump({"run_id": run_id, "families": families}, f, indent=2)
    logger.info("Wrote within_family_trends.json")

    # 4. falsifier_outcome.json
    outcome_doc = {
        "run_id": run_id,
        "protocol": "F-GQA-01",
        "timestamp": datetime.now().isoformat(),
        "falsifiers": falsifier_outcomes,
        "overall": _overall_verdict(falsifier_outcomes),
    }
    with open(out_dir / "falsifier_outcome.json", "w") as f:
        json.dump(outcome_doc, f, indent=2)
    logger.info("Wrote falsifier_outcome.json")

    return out_dir


def _overall_verdict(outcomes: dict) -> str:
    """Derive overall verdict from individual falsifier outcomes."""
    statuses = [v["status"] for v in outcomes.values()]
    if "FALSIFIED" in statuses:
        return "FALSIFIED"
    if all(s == "SUPPORTED" or s == "CONSISTENT" for s in statuses):
        return "SUPPORTED"
    non_adjudicating = (
        "UNDERPOWERED", "BELOW_MEASUREMENT_RESOLUTION",
        "INSUFFICIENT_DATA", "INCOMMENSURABLE",
    )
    if all(s in non_adjudicating for s in statuses):
        return "UNDERPOWERED"
    return "INCONCLUSIVE"


def _validate_emitted_artifacts_or_raise(out_dir: Path) -> dict:
    """Validate emitted artifacts and raise on integrity failure."""
    validation = artifact_validator.validate_run_dir(out_dir, schema_mode="v2")
    if not validation["ok"]:
        logger.error("Artifact integrity validation FAILED for %s", out_dir)
        for error in validation["errors"]:
            logger.error("  %s", error)
        raise RuntimeError(f"Artifact integrity validation failed for {out_dir}")

    for warning in validation["warnings"]:
        logger.warning("Artifact validator warning: %s", warning)

    logger.info(
        "Artifact integrity validation: PASS (schema=%s, files_checked=%d)",
        validation.get("detected_schema", "unknown"),
        len(validation.get("files_checked", [])),
    )
    return validation


# ============================================================================
# Section 10: CLI
# ============================================================================


def run_smoke() -> None:
    """Smoke test: offline, 2 models, validate z_couple against known stored values."""
    logger.info("=== SMOKE TEST (offline, 2 models) ===")
    smoke_models = {"LFM2-350M": MODEL_REGISTRY["LFM2-350M"], "Qwen2.5-3B": MODEL_REGISTRY["Qwen2.5-3B"]}

    z_results = {}
    for name in smoke_models:
        z = compute_z_couple(name)
        z_results[name] = z
        logger.info(
            "  %s: z_couple=%.4f, r=%.4f, n_eff=%.1f, mde=%.3f, above_mde=%s",
            name,
            z.get("z_couple", float("nan")),
            z.get("r_pearson", float("nan")),
            z.get("n_eff_z", float("nan")),
            z.get("mde", float("nan")),
            z.get("above_mde"),
        )

    # Validate: z_couple should be finite and non-zero for these models
    for name, z in z_results.items():
        assert z["z_couple"] is not None, f"{name}: z_couple is None"
        assert np.isfinite(z["z_couple"]), f"{name}: z_couple is not finite"
        logger.info("  %s: PASS (z_couple=%.4f)", name, z["z_couple"])

    # Check c_cancel from cached curvature accumulation
    for name in smoke_models:
        c = compute_c_cancel_from_cached(name)
        if c is not None:
            logger.info(
                "  %s: c_cancel=%.4f (beta_num=%.4f, beta_den=%.4f)",
                name, c.get("c_cancel", float("nan")),
                c.get("beta_num", float("nan")),
                c.get("beta_den", float("nan")),
            )
        else:
            logger.info("  %s: no cached c_cancel data", name)

    logger.info("=== SMOKE TEST PASSED ===")


def run_full(
    collect_missing: bool = False,
    include_smollm3: bool = False,
    run_id: str | None = None,
) -> None:
    """Full protocol run."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    registry = dict(MODEL_REGISTRY)
    if include_smollm3:
        registry.update(SMOLLM3_ENTRY)
        logger.info("Including SmolLM3-3B (requires prior GPU operator_split run)")

    logger.info("=== F-GQA-01: GQA Falsifier Protocol (run_id=%s) ===", run_id)
    logger.info("Models: %d", len(registry))

    # --- z_couple for all models ---
    logger.info("\n--- z_couple computation ---")
    z_results = {}
    for name in registry:
        z = compute_z_couple(name)
        z_results[name] = z
        if z.get("z_couple") is not None:
            comm_flag = "COMM" if z.get("commensurable") else "INCOMM"
            sat_val = z.get("h_logit_saturation")
            sat_str = f"{sat_val:.4f}" if sat_val is not None else "N/A"
            logger.info(
                "  %s: z=%.4f, r=%.4f, n_eff=%.1f, mde=%.3f, above=%s, "
                "h_logit_resid_range=%.4f, saturation=%s [%s]",
                name, z["z_couple"], z["r_pearson"], z["n_eff_z"],
                z["mde"], z["above_mde"],
                z.get("h_logit_resid_range", float("nan")),
                sat_str,
                comm_flag,
            )
        else:
            logger.info("  %s: SKIPPED (%s)", name, z.get("error", "unknown"))

    n_comm = sum(1 for z in z_results.values() if z.get("commensurable"))
    n_incomm = sum(1 for z in z_results.values() if z.get("commensurable") is False)
    logger.info("  Commensurability: %d commensurable, %d incommensurable (threshold=log(2)=%.3f nats)",
                n_comm, n_incomm, float(np.log(2)))

    # --- c_cancel ---
    logger.info("\n--- c_cancel computation ---")
    c_results = {}
    for name in registry:
        # Try cached data first
        c = compute_c_cancel_from_cached(name)
        if c is not None:
            c_results[name] = c
            logger.info(
                "  %s: c_cancel=%.4f (cached, beta_num=%.4f, beta_den=%.4f)",
                name, c["c_cancel"], c["beta_num"], c["beta_den"],
            )
        elif collect_missing:
            c = collect_c_cancel_gpu(name, registry[name])
            c_results[name] = c
            if c.get("c_cancel") is not None:
                logger.info(
                    "  %s: c_cancel=%.4f (GPU collected, beta_num=%.4f, beta_den=%.4f)",
                    name, c["c_cancel"], c["beta_num"], c["beta_den"],
                )
            else:
                logger.info("  %s: GPU collection failed (%s)", name, c.get("error", "unknown"))
        else:
            logger.info("  %s: no cached data (use --collect-missing for GPU collection)", name)

    # --- Design matrix assembly ---
    logger.info("\n--- Design matrix assembly ---")
    records = assemble_design_matrix(z_results, c_results, registry)

    n_z = sum(1 for r in records if r.get("z_couple") is not None)
    n_c = sum(1 for r in records if r.get("c_cancel") is not None)
    logger.info("  %d models with z_couple, %d with c_cancel", n_z, n_c)

    # --- Design matrix diagnostics (BEFORE regression) ---
    logger.info("\n--- Design matrix diagnostics ---")
    z_diag = design_diagnostics(records, "z_couple")
    c_diag = design_diagnostics(records, "c_cancel")

    if "error" not in z_diag:
        logger.info("  z_couple design (n=%d):", z_diag["n"])
        logger.info("    cond(X'X) = %.1f", z_diag["cond_XtX"])
        logger.info("    eigenspectrum = %s", [f"{e:.1f}" for e in z_diag["eigenspectrum"]])
        logger.info("    VIF: %s", {k: f"{v:.2f}" for k, v in z_diag["VIF"].items()})
        logger.info("    smallest eigenvector: %s",
                     {k: f"{v:.3f}" for k, v in z_diag["smallest_eigenvector"].items()})
        logger.info("    predictor correlations: %s",
                     {k: f"{v:.3f}" for k, v in z_diag["predictor_correlation"].items()})
    else:
        logger.info("  z_couple design: %s", z_diag.get("error"))

    if "error" not in c_diag:
        logger.info("  c_cancel design (n=%d):", c_diag["n"])
        logger.info("    cond(X'X) = %.1f", c_diag["cond_XtX"])
        logger.info("    VIF: %s", {k: f"{v:.2f}" for k, v in c_diag["VIF"].items()})
    else:
        logger.info("  c_cancel design: %s (n=%d)", c_diag.get("error", "unknown"), c_diag.get("n", 0))

    # --- Regressions ---
    logger.info("\n--- Regressions ---")

    # Full regression (all models — EXPLORATORY ONLY, includes incommensurable models)
    z_reg = weighted_ols_regression(
        records, "z_couple", "n_eff_z",
        "z_couple ~ log(GQA) + I(hybrid) + log(d) [EXPLORATORY: includes incommensurable models]",
    )
    c_reg = weighted_ols_regression(records, "c_cancel", "n_eff_c", "c_cancel ~ log(GQA) + I(hybrid) + log(d)")

    if "error" not in z_reg:
        logger.info("  z_couple FULL regression (n=%d, DOF=%d, R²=%.3f) [EXPLORATORY — includes incommensurable]:",
                     z_reg["n"], z_reg["dof"], z_reg["r_squared"])
        for name, coeff in z_reg["coefficients"].items():
            logger.info(
                "    %s: %.4f (SE=%.4f, t=%.2f, p=%.4f, CI=[%.4f, %.4f])",
                name, coeff["estimate"], coeff["se"], coeff["t_stat"],
                coeff["p_value"], coeff["ci_95"][0], coeff["ci_95"][1],
            )
    else:
        logger.info("  z_couple FULL regression: %s", z_reg.get("error"))
        if z_reg.get("note"):
            logger.info("    %s", z_reg["note"])

    # Commensurable-only z_couple regression
    comm_records = [r for r in records if r.get("commensurable") is True]
    z_reg_comm = weighted_ols_regression(
        comm_records, "z_couple", "n_eff_z",
        "z_couple ~ log(GQA) + I(hybrid) + log(d) [COMMENSURABLE ONLY]",
    )
    if "error" not in z_reg_comm:
        logger.info("  z_couple COMMENSURABLE regression (n=%d, DOF=%d, R²=%.3f):",
                     z_reg_comm["n"], z_reg_comm["dof"], z_reg_comm["r_squared"])
        for name, coeff in z_reg_comm["coefficients"].items():
            logger.info(
                "    %s: %.4f (SE=%.4f, t=%.2f, p=%.4f, CI=[%.4f, %.4f])",
                name, coeff["estimate"], coeff["se"], coeff["t_stat"],
                coeff["p_value"], coeff["ci_95"][0], coeff["ci_95"][1],
            )
    else:
        logger.info("  z_couple COMMENSURABLE regression: %s (n=%d)",
                     z_reg_comm.get("error", "unknown"), z_reg_comm.get("n", 0))
        if z_reg_comm.get("note"):
            logger.info("    %s", z_reg_comm["note"])

    if "error" not in c_reg:
        logger.info("  c_cancel regression (n=%d, DOF=%d, R²=%.3f):", c_reg["n"], c_reg["dof"], c_reg["r_squared"])
        for name, coeff in c_reg["coefficients"].items():
            logger.info(
                "    %s: %.4f (SE=%.4f, t=%.2f, p=%.4f, CI=[%.4f, %.4f])",
                name, coeff["estimate"], coeff["se"], coeff["t_stat"],
                coeff["p_value"], coeff["ci_95"][0], coeff["ci_95"][1],
            )
    else:
        logger.info("  c_cancel regression: %s (n=%d)", c_reg.get("error"), c_reg.get("n", 0))

    # --- Falsifier adjudication ---
    logger.info("\n--- Falsifier adjudication ---")
    falsifiers = adjudicate_falsifiers(z_reg, c_reg, records)
    for fname, outcome in falsifiers.items():
        logger.info("  %s: %s — %s", fname, outcome["status"], outcome["reason"])

    # --- Artifact emission ---
    logger.info("\n--- Artifact emission ---")
    out_dir = emit_artifacts(
        run_id, records, z_reg, c_reg, falsifiers, z_diag, c_diag,
        z_regression_comm=z_reg_comm,
    )
    _validate_emitted_artifacts_or_raise(out_dir)
    logger.info("Artifacts written to %s", out_dir)

    overall = _overall_verdict(falsifiers)
    logger.info("\n=== OVERALL VERDICT: %s ===", overall)


def main():
    parser = argparse.ArgumentParser(description="F-GQA-01: GQA Falsifier Protocol")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (offline, 2 models)")
    parser.add_argument("--collect-missing", action="store_true",
                        help="GPU collection for models without cached c_cancel data")
    parser.add_argument("--include-smollm3", action="store_true",
                        help="Include SmolLM3-3B (optional, requires prior GPU operator_split run)")
    parser.add_argument("--output", type=str, default=None,
                        help="Run ID for output directory (default: timestamp)")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
    else:
        run_full(
            collect_missing=args.collect_missing,
            include_smollm3=args.include_smollm3,
            run_id=args.output,
        )


if __name__ == "__main__":
    main()
