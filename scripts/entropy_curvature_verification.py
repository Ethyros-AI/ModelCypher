#!/usr/bin/env python3
"""Entropy → Curvature Verification.

Tests the mechanistic relationship between attention entropy and angular
curvature, decomposed into attention and MLP sublayer contributions.

Predictions tested:
  P1: Spearman(H_l, θ_attn_l) has OPPOSITE sign to Spearman(H_l, θ_mlp_l)
  P2: Attention fraction θ_attn/θ_total DECREASES with entropy
  P3: GQA ratio modulates the entropy-curvature correlation strength
  P4: MLP angular gain varies with attention entropy (not constant)
  P5: Value vector alignment with h_in predicts entropy-curvature sign
  P6: MLP gain variance partially explains residual variance in entropy-curvature

Measurements per attention layer:
  1. h_in, h_post_attn, h_out (sublayer activations)
  2. Attention weights (softmax(QK^T / √d_k))
  3. θ_attn, θ_mlp, θ_total (angular curvature decomposition)
  4. H_l (Shannon entropy of attention weights, mean across heads)
  5. G_mlp = θ_mlp / θ_attn (MLP angular gain)

Usage:
    poetry run python scripts/entropy_curvature_verification.py
    poetry run python scripts/entropy_curvature_verification.py --smoke
    poetry run python scripts/entropy_curvature_verification.py --models LFM2-700M Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")

MODEL_REGISTRY = {
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16, "d": 1280,
        "architecture": "lfm2",
        "gqa_ratio": 3,  # n_heads=24, n_kv_heads=8
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,  # n_heads=8, n_kv_heads=2
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
        "gqa_ratio": 2,
    },
}

# Diverse probes — 5 categories × 6 probes = 30 total
PROBES = [
    # Retrieval
    "The capital of France is",
    "Who wrote Romeo and Juliet?",
    "The chemical symbol for water is",
    "The largest planet in our solar system is",
    "The first president of the United States was",
    "The boiling point of water at sea level is",
    # Arithmetic
    "What is 347 + 528?",
    "What is 15 * 23?",
    "What is 1024 / 16?",
    "What is 99 - 37?",
    "What is 8 * 7 + 13?",
    "What is 256 + 384 - 100?",
    # Reasoning
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
    "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    # Creative
    "Write a haiku about the ocean.",
    "Describe a sunset over the mountains in one vivid sentence.",
    "Write a short poem about the passage of time.",
    "Describe the taste of your favorite food using only three words.",
    "Write a one-sentence story with a twist ending.",
    "Describe the sound of rain on a tin roof.",
    # Narrative
    "Once upon a time in a faraway kingdom, there lived a",
    "The old lighthouse keeper watched the storm approach from",
    "In the year 2150, humanity had finally achieved",
    "She opened the letter and read the first line:",
    "The forest was silent except for the sound of",
    "He had been walking for three days when he finally saw",
]


# =============================================================================
# Core measurements
# =============================================================================


def angular_change(v1: np.ndarray, v2: np.ndarray) -> float:
    """Geodesic distance on unit sphere: arccos(cosine_similarity)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_sim = np.dot(v1, v2) / (n1 * n2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(np.arccos(cos_sim))


def shannon_entropy(weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy H = -Σ p log(p), numerically stable."""
    eps = 1e-10
    w = np.clip(weights, eps, 1.0)
    return -np.sum(w * np.log(w), axis=axis)


# =============================================================================
# Attention entropy extraction
# =============================================================================


def _get_pre_norm(layer):
    """Find the pre-attention normalization layer."""
    for name in ("input_layernorm", "operator_norm", "attention_norm"):
        norm = getattr(layer, name, None)
        if norm is not None:
            return norm
    return None


def _is_full_attention_layer(layer) -> bool:
    """Check if a layer has standard softmax attention."""
    # LFM2: explicit flag
    if hasattr(layer, "is_attention_layer"):
        return layer.is_attention_layer
    # Check for self_attn with Q/K projections
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return False
    return hasattr(attn, "q_proj") and hasattr(attn, "k_proj")


def compute_attention_entropy(layer, h_in, seq_len: int):
    """Compute Shannon entropy of attention weights for one layer.

    Manually computes softmax(QK^T / sqrt(d_k)) since MLX fused kernels
    don't expose intermediate attention weights.

    Returns dict with layer_entropy, per_head_entropy, n_heads, n_kv_heads,
    or None if attention weights can't be extracted.
    """
    import mlx.core as mx

    if not _is_full_attention_layer(layer):
        return None

    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None

    pre_norm = _get_pre_norm(layer)
    if pre_norm is None:
        return None

    try:
        normed = pre_norm(h_in)
        q = attn.q_proj(normed)
        k = attn.k_proj(normed)

        n_heads = attn.n_heads
        n_kv_heads = getattr(attn, "n_kv_heads", n_heads)
        head_dim = q.shape[-1] // n_heads

        B, L, _ = q.shape

        # Reshape to [B, n_heads, L, head_dim]
        q = q.reshape(B, L, n_heads, head_dim)
        k = k.reshape(B, L, n_kv_heads, head_dim)

        # Post-projection layernorm (LFM2, some Qwen variants)
        for qn_name in ("q_layernorm", "q_norm"):
            qn = getattr(attn, qn_name, None)
            if qn is not None:
                q = qn(q)
                break
        for kn_name in ("k_layernorm", "k_norm"):
            kn = getattr(attn, kn_name, None)
            if kn is not None:
                k = kn(k)
                break

        q = q.transpose(0, 2, 1, 3)  # [B, nh, L, hd]
        k = k.transpose(0, 2, 1, 3)  # [B, nkv, L, hd]

        # RoPE
        rope_fn = getattr(attn, "rope", getattr(attn, "rotary_emb", None))
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        # GQA: expand K heads to match Q heads
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)

        # Attention scores = QK^T / sqrt(d_k)
        scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim))
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale

        # Causal mask
        causal = mx.triu(mx.full((L, L), -1e9, dtype=scores.dtype), k=1)
        scores = scores + causal

        # Softmax → attention weights [B, nh, L, L]
        weights = mx.softmax(scores, axis=-1)
        mx.eval(weights)

        # Convert to numpy for entropy computation
        weights_np = np.array(weights[0].tolist(), dtype=np.float32)  # [nh, L, L]

        # Shannon entropy per head per query position
        # H(l, h, q) = -Σ_k α(q,k) log α(q,k)
        per_head_per_pos = shannon_entropy(weights_np, axis=-1)  # [nh, L]

        # Mean across positions → per-head entropy
        per_head = np.mean(per_head_per_pos, axis=-1)  # [nh]

        # Mean across heads → layer entropy
        layer_entropy = float(np.mean(per_head))

        return {
            "layer_entropy": layer_entropy,
            "per_head_entropy": per_head.tolist(),
            "n_heads": int(n_heads),
            "n_kv_heads": int(n_kv_heads),
        }

    except Exception as exc:
        logger.debug("Attention entropy extraction failed for layer: %s", exc)
        return None


# =============================================================================
# Combined forward pass: sublayer activations + attention entropy
# =============================================================================


def collect_layer_data(
    model, tokenizer, prompts: list[str], num_layers: int, backend,
) -> list[dict]:
    """Collect sublayer activations and attention entropy per layer.

    For each layer, collects:
        h_in, h_post_attn, h_out: sublayer activations [N, d]
        attn_entropy: Shannon entropy of attention weights (or None)
    """
    import mlx.core as mx

    base = _resolve_backbone(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # Per-layer collectors
    layer_h_in = [[] for _ in range(num_layers)]
    layer_h_post_attn = [[] for _ in range(num_layers)]
    layer_h_out = [[] for _ in range(num_layers)]
    layer_entropy = [[] for _ in range(num_layers)]

    for pi, prompt in enumerate(prompts):
        if pi % 10 == 0:
            logger.info("  Probe %d/%d", pi + 1, len(prompts))

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden

            # Per-layer mask routing (LFM2 compatibility)
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            # Try sublayer decomposition
            h_post_attn = None
            input_norm = _get_pre_norm(layer)
            self_attn = getattr(layer, "self_attn", None)
            post_attn_norm = getattr(layer, "post_attention_layernorm", None)
            if post_attn_norm is None:
                post_attn_norm = getattr(layer, "ffn_norm", None)
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                mlp = getattr(layer, "feed_forward", None)

            if input_norm is not None and self_attn is not None and mlp is not None:
                try:
                    normed = input_norm(h_in)
                    # self_attn expects numeric mask, not "causal" string
                    attn_mask = numeric_mask
                    attn_out = self_attn(normed, mask=attn_mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_post_attn = h_in + attn_out

                    normed2 = post_attn_norm(h_post_attn) if post_attn_norm else h_post_attn
                    mlp_out = mlp(normed2)
                    h_out = h_post_attn + mlp_out
                except Exception:
                    h_post_attn = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            else:
                try:
                    h_out = layer(h_in, mask=layer_mask)
                except (TypeError, ValueError):
                    try:
                        h_out = layer(h_in, layer_mask)
                    except (TypeError, ValueError):
                        h_out = layer(h_in)

            # Last-token representation
            h_in_last = h_in[:, -1, :].astype(mx.float32)
            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_in_last, h_out_last)

            layer_h_in[i].append(np.array(h_in_last[0].tolist(), dtype=np.float32))
            layer_h_out[i].append(np.array(h_out_last[0].tolist(), dtype=np.float32))

            if h_post_attn is not None:
                h_pa_last = h_post_attn[:, -1, :].astype(mx.float32)
                mx.eval(h_pa_last)
                layer_h_post_attn[i].append(
                    np.array(h_pa_last[0].tolist(), dtype=np.float32)
                )
            else:
                layer_h_post_attn[i].append(None)

            # Attention entropy (only for first probe on first pass to get shape)
            ent_result = compute_attention_entropy(layer, h_in, seq_len)
            if ent_result is not None:
                layer_entropy[i].append(ent_result["layer_entropy"])
            else:
                layer_entropy[i].append(None)

            hidden = h_out

        mx.eval(hidden)

    # Build result
    result = []
    for i in range(num_layers):
        has_decomp = all(x is not None for x in layer_h_post_attn[i])
        has_entropy = all(x is not None for x in layer_entropy[i])

        result.append({
            "h_in": np.stack(layer_h_in[i]),
            "h_out": np.stack(layer_h_out[i]),
            "h_post_attn": np.stack(layer_h_post_attn[i]) if has_decomp else None,
            "has_decomposition": has_decomp,
            "attn_entropies": layer_entropy[i] if has_entropy else None,
            "has_entropy": has_entropy,
        })

    return result


# =============================================================================
# Analysis
# =============================================================================


def compute_layer_measurements(layer_data: list[dict]) -> list[dict]:
    """Compute per-layer curvature decomposition and entropy statistics."""
    measurements = []

    for i, data in enumerate(layer_data):
        h_in = data["h_in"]
        h_out = data["h_out"]
        n_probes = h_in.shape[0]

        # Total curvature
        total_angles = [angular_change(h_in[j], h_out[j]) for j in range(n_probes)]
        total_curvature = float(np.mean(total_angles))

        layer_result = {
            "layer_idx": i,
            "total_curvature": total_curvature,
            "attn_curvature": None,
            "mlp_curvature": None,
            "attn_fraction": None,
            "mlp_gain": None,
            "attn_entropy": None,
            "is_attention_layer": data["has_entropy"],
        }

        # Sublayer decomposition
        if data["has_decomposition"]:
            h_post_attn = data["h_post_attn"]
            attn_angles = [angular_change(h_in[j], h_post_attn[j]) for j in range(n_probes)]
            mlp_angles = [angular_change(h_post_attn[j], h_out[j]) for j in range(n_probes)]

            attn_curv = float(np.mean(attn_angles))
            mlp_curv = float(np.mean(mlp_angles))

            layer_result["attn_curvature"] = attn_curv
            layer_result["mlp_curvature"] = mlp_curv
            layer_result["attn_fraction"] = (
                attn_curv / (attn_curv + mlp_curv)
                if (attn_curv + mlp_curv) > 1e-10 else 0.5
            )
            layer_result["mlp_gain"] = (
                mlp_curv / attn_curv if attn_curv > 1e-10 else float("nan")
            )

        # Attention entropy (mean across probes)
        if data["has_entropy"] and data["attn_entropies"]:
            layer_result["attn_entropy"] = float(np.mean(data["attn_entropies"]))

        measurements.append(layer_result)

    return measurements


def test_predictions(measurements: list[dict], model_name: str) -> dict:
    """Test all 6 predictions on one model's measurements.

    Only uses attention layers (layers with both decomposition and entropy).
    """
    from scipy import stats

    # Filter to layers with full data
    full_layers = [
        m for m in measurements
        if m["attn_entropy"] is not None
        and m["attn_curvature"] is not None
        and m["mlp_curvature"] is not None
    ]

    n_full = len(full_layers)
    logger.info("  %s: %d layers with full entropy + curvature data", model_name, n_full)

    if n_full < 4:
        return {
            "model": model_name,
            "n_full_layers": n_full,
            "note": "Insufficient layers for correlation analysis (need >=4)",
            "predictions": {},
        }

    H = [m["attn_entropy"] for m in full_layers]
    theta_attn = [m["attn_curvature"] for m in full_layers]
    theta_mlp = [m["mlp_curvature"] for m in full_layers]
    theta_total = [m["total_curvature"] for m in full_layers]
    attn_frac = [m["attn_fraction"] for m in full_layers]
    mlp_gain = [m["mlp_gain"] for m in full_layers]

    def safe_spearman(x, y):
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 4:
            return float("nan"), float("nan")
        try:
            r, p = stats.spearmanr(x_arr[mask], y_arr[mask])
            return float(r), float(p)
        except Exception:
            return float("nan"), float("nan")

    # P1: Spearman(H, θ_attn) has OPPOSITE sign to Spearman(H, θ_mlp)
    r_h_attn, p_h_attn = safe_spearman(H, theta_attn)
    r_h_mlp, p_h_mlp = safe_spearman(H, theta_mlp)
    p1_opposite = (
        (r_h_attn > 0 and r_h_mlp < 0) or (r_h_attn < 0 and r_h_mlp > 0)
    ) if not (math.isnan(r_h_attn) or math.isnan(r_h_mlp)) else None

    # P2: Attention fraction DECREASES with entropy
    r_h_frac, p_h_frac = safe_spearman(H, attn_frac)
    p2_decreasing = r_h_frac < 0 if not math.isnan(r_h_frac) else None

    # P3: Tested across models (in cross-model analysis)
    r_h_total, p_h_total = safe_spearman(H, theta_total)

    # P4: MLP gain varies (CV > 0.1)
    mlp_gains_valid = [g for g in mlp_gain if not math.isnan(g)]
    if len(mlp_gains_valid) >= 2:
        mlp_gain_cv = float(np.std(mlp_gains_valid) / np.mean(mlp_gains_valid))
    else:
        mlp_gain_cv = float("nan")
    p4_varies = mlp_gain_cv > 0.1 if not math.isnan(mlp_gain_cv) else None

    # P5: Value alignment (requires V weights — deferred to separate analysis)
    # Mark as not tested in this script
    p5_result = None

    # P6: MLP gain variance explains residual variance
    r_h_gain, p_h_gain = safe_spearman(H, mlp_gain)

    predictions = {
        "P1_sign_opposition": {
            "r_H_theta_attn": r_h_attn,
            "p_H_theta_attn": p_h_attn,
            "r_H_theta_mlp": r_h_mlp,
            "p_H_theta_mlp": p_h_mlp,
            "opposite_sign": p1_opposite,
            "status": "PASS" if p1_opposite else ("FAIL" if p1_opposite is False else "INCONCLUSIVE"),
        },
        "P2_attn_fraction_decreases": {
            "r_H_attn_fraction": r_h_frac,
            "p_H_attn_fraction": p_h_frac,
            "decreasing": p2_decreasing,
            "status": "PASS" if p2_decreasing else ("FAIL" if p2_decreasing is False else "INCONCLUSIVE"),
        },
        "P3_gqa_modulation": {
            "r_H_theta_total": r_h_total,
            "p_H_theta_total": p_h_total,
            "note": "Compare r across models with different GQA ratios",
        },
        "P4_mlp_gain_varies": {
            "mlp_gain_cv": mlp_gain_cv,
            "mlp_gain_mean": float(np.mean(mlp_gains_valid)) if mlp_gains_valid else float("nan"),
            "mlp_gain_std": float(np.std(mlp_gains_valid)) if mlp_gains_valid else float("nan"),
            "varies": p4_varies,
            "status": "PASS" if p4_varies else ("FAIL" if p4_varies is False else "INCONCLUSIVE"),
        },
        "P5_value_alignment": {
            "note": "Requires V weight extraction — deferred to separate analysis",
            "status": "NOT_TESTED",
        },
        "P6_mlp_gain_explains_residual": {
            "r_H_mlp_gain": r_h_gain,
            "p_H_mlp_gain": p_h_gain,
            "status": (
                "PASS" if (not math.isnan(r_h_gain) and abs(r_h_gain) > 0.3)
                else ("FAIL" if not math.isnan(r_h_gain) else "INCONCLUSIVE")
            ),
        },
    }

    # Summary correlations
    summary = {
        "entropy_vs_total_curvature": {"r": r_h_total, "p": p_h_total},
        "entropy_vs_attn_curvature": {"r": r_h_attn, "p": p_h_attn},
        "entropy_vs_mlp_curvature": {"r": r_h_mlp, "p": p_h_mlp},
        "entropy_vs_attn_fraction": {"r": r_h_frac, "p": p_h_frac},
        "entropy_vs_mlp_gain": {"r": r_h_gain, "p": p_h_gain},
    }

    return {
        "model": model_name,
        "n_full_layers": n_full,
        "predictions": predictions,
        "correlations": summary,
    }


# =============================================================================
# Cross-model analysis
# =============================================================================


def cross_model_analysis(all_results: list[dict]) -> dict:
    """P3: Test whether GQA ratio modulates entropy-curvature correlation."""
    gqa_vs_r = []
    for r in all_results:
        gqa = r.get("gqa_ratio", 1)
        corr = r.get("analysis", {}).get("correlations", {})
        r_total = corr.get("entropy_vs_total_curvature", {}).get("r", float("nan"))
        if not math.isnan(r_total):
            gqa_vs_r.append((gqa, r_total))

    if len(gqa_vs_r) < 3:
        return {
            "P3_gqa_modulation": {
                "note": "Need >= 3 models with different GQA ratios",
                "status": "INCONCLUSIVE",
                "data": gqa_vs_r,
            }
        }

    from scipy import stats
    gqa_vals = [x[0] for x in gqa_vs_r]
    r_vals = [x[1] for x in gqa_vs_r]
    r_gqa, p_gqa = stats.spearmanr(gqa_vals, r_vals)

    return {
        "P3_gqa_modulation": {
            "gqa_vs_r": gqa_vs_r,
            "r_gqa_correlation": float(r_gqa) if not np.isnan(r_gqa) else float("nan"),
            "p_gqa_correlation": float(p_gqa) if not np.isnan(p_gqa) else float("nan"),
            "status": (
                "PASS" if (not np.isnan(r_gqa) and abs(r_gqa) > 0.5)
                else "FAIL" if not np.isnan(r_gqa) else "INCONCLUSIVE"
            ),
        }
    }


# =============================================================================
# Model runner
# =============================================================================


def run_single_model(
    model_name: str, model_info: dict, probes: list[str], backend,
) -> dict:
    """Run entropy-curvature analysis for one model."""
    logger.info("Loading model: %s from %s", model_name, model_info["path"])
    model, tokenizer = backend.load_model(model_info["path"])

    base = getattr(model, "model", model)
    if hasattr(base, "language_model"):
        base = base.language_model
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0
    d_model = model_info.get("d", 0)

    logger.info("Model loaded: %d layers, d=%d", num_layers, d_model)

    t0 = time.time()
    layer_data = collect_layer_data(model, tokenizer, probes, num_layers, backend)
    logger.info("Data collection: %.1fs", time.time() - t0)

    n_with_entropy = sum(1 for d in layer_data if d["has_entropy"])
    n_with_decomp = sum(1 for d in layer_data if d["has_decomposition"])
    logger.info("  Layers with attention entropy: %d/%d", n_with_entropy, num_layers)
    logger.info("  Layers with sublayer decomposition: %d/%d", n_with_decomp, num_layers)

    measurements = compute_layer_measurements(layer_data)
    analysis = test_predictions(measurements, model_name)

    # Log key results
    for pname, pred in analysis.get("predictions", {}).items():
        status = pred.get("status", "?")
        logger.info("  %s: %s", pname, status)

    del model, tokenizer, layer_data
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "gqa_ratio": model_info.get("gqa_ratio", 1),
        "num_layers": num_layers,
        "d_model": d_model,
        "n_probes": len(probes),
        "measurements": measurements,
        "analysis": analysis,
    }


# =============================================================================
# Main
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run entropy → curvature verification."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    if args.smoke:
        model_names = ["LFM2-700M"]
        probes = PROBES[:6]
    elif args.models:
        model_names = args.models
        probes = PROBES
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = PROBES

    logger.info("Experiment: %d models, %d probes", len(model_names), len(probes))

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model: %s, skipping", model_name)
            continue
        result = run_single_model(model_name, MODEL_REGISTRY[model_name], probes, backend)
        all_results.append(result)
        gc.collect()

    # Cross-model analysis
    cross = cross_model_analysis(all_results)

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("ENTROPY → CURVATURE VERIFICATION SUMMARY")
    logger.info("%s", "=" * 60)

    for r in all_results:
        logger.info("\n%s (GQA=%d):", r["model_name"], r["gqa_ratio"])
        for pname, pred in r["analysis"].get("predictions", {}).items():
            status = pred.get("status", "?")
            detail = ""
            if "r_H_theta_attn" in pred:
                detail = f" r(H,θ_attn)={pred['r_H_theta_attn']:.3f}, r(H,θ_mlp)={pred['r_H_theta_mlp']:.3f}"
            elif "r_H_attn_fraction" in pred:
                detail = f" r={pred['r_H_attn_fraction']:.3f}"
            elif "mlp_gain_cv" in pred:
                detail = f" CV={pred['mlp_gain_cv']:.3f}"
            elif "r_H_mlp_gain" in pred:
                detail = f" r={pred['r_H_mlp_gain']:.3f}"
            logger.info("  %s: %s%s", pname, status, detail)

    logger.info("\nCross-model P3 (GQA modulation):")
    p3 = cross.get("P3_gqa_modulation", {})
    logger.info("  Status: %s", p3.get("status", "?"))
    if "gqa_vs_r" in p3:
        for gqa, r_val in p3["gqa_vs_r"]:
            logger.info("    GQA=%d: r(H,θ)=%.3f", gqa, r_val)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "entropy_curvature_results.json"

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "entropy_curvature_verification",
        "n_models": len(all_results),
        "n_probes": len(probes),
        "models": all_results,
        "cross_model": cross,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info("\nResults saved to %s", output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Entropy → Curvature Verification"
    )
    parser.add_argument(
        "--output", default="results/entropy_curvature/",
        help="Output directory",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to test",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test (1 model, 6 probes)",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
