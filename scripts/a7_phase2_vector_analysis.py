#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""A7 Phase 2 — Vector Gradient Structure Analysis.

Phase 1 found per-head R² ≈ 0.71-0.76 using scalar summaries of w_t.
~25-30% of gradient variance is unexplained by scalar summaries.

The exact identity: ∂L/∂s_t / α_t = ⟨g, w_t⟩ + const (chain rule, exact).
The inner product decomposes: ⟨g, w_t⟩ = g_radial × r_t + ⟨g_⊥, w_t_⊥⟩.
The scalar model captures magnitudes but loses directional info about w_t_⊥.

Phase 2 recovers the actual gradient vector g (projected into span{w_t}),
decomposes it into radial/tangential components, and explains the scalar R² gap.

No decision thresholds. All outputs are raw measurements.

EXPERIMENTAL STATUS: scripts/ only, NOT CLI (AGENTS.md:695)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Reuse validated functions from A7 validator
from validate_a7_assumption import (
    MODELS,
    PROBE_TEXTS,
    _EPS_F32,
    _QUERY_POS,
    collect_baseline,
    collect_per_head_gradient_data,
    compute_radial_projections,
    compute_score_gradients,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Phase 2 uses only 2 smallest models (model size policy)
PHASE2_MODELS = {
    "LFM2-350M": MODELS["LFM2-350M"],
    "Qwen3.5-0.8B": MODELS["Qwen3.5-0.8B"],
}


# =====================================================================
# Analysis functions (pure numpy)
# =====================================================================


def vector_regression(
    w_t_vectors: np.ndarray, grad_over_alpha: np.ndarray
) -> dict:
    """Identity verification: recover g from ⟨g, w_t⟩ = ∂L/∂s_t / α_t - const.

    Build W = [w_1^T; ...; w_T^T] shape [T, hidden_dim], augment with intercept.
    Solve via np.linalg.lstsq (minimum-norm solution since T << d).
    R² should be ≈ 1.0 (identity is exact; deviation measures finite-diff noise).

    Returns: R², g_proj [hidden_dim], intercept, residual_norm.
    """
    T, d = w_t_vectors.shape
    # Augment: [W | 1] @ [g; c] = y
    W_aug = np.column_stack([w_t_vectors, np.ones(T)])  # [T, d+1]
    y = grad_over_alpha  # [T]

    beta, _, rank, _ = np.linalg.lstsq(W_aug, y, rcond=None)
    g_proj = beta[:d]  # [hidden_dim]
    intercept = float(beta[d])

    y_hat = W_aug @ beta
    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))

    if ss_tot < 1e-30:
        r_squared = 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    residual_norm = float(np.sqrt(ss_res))

    return {
        "r_squared": r_squared,
        "g_proj": g_proj,
        "intercept": intercept,
        "residual_norm": residual_norm,
        "rank": int(rank),
        "T": T,
        "d": d,
    }


def analyze_g_proj(g_proj: np.ndarray, h_hat: np.ndarray) -> dict:
    """Gradient direction characterization.

    - g_radial = ⟨g_proj, ĥ⟩
    - g_⊥ = g_proj - g_radial·ĥ
    - cos(g_proj, ĥ) — how radial is the downstream gradient?
    - radial_fraction = |g_radial| / ||g_proj||
    """
    g_norm = float(np.linalg.norm(g_proj))
    if g_norm < 1e-30:
        return {
            "g_radial": 0.0,
            "g_perp_norm": 0.0,
            "cos_g_hhat": 0.0,
            "radial_fraction": 0.0,
            "g_norm": 0.0,
        }

    g_radial = float(np.dot(g_proj, h_hat))
    g_perp = g_proj - g_radial * h_hat
    g_perp_norm = float(np.linalg.norm(g_perp))

    cos_g_hhat = g_radial / g_norm
    radial_fraction = abs(g_radial) / g_norm

    return {
        "g_radial": g_radial,
        "g_perp_norm": g_perp_norm,
        "cos_g_hhat": cos_g_hhat,
        "radial_fraction": radial_fraction,
        "g_norm": g_norm,
    }


def measure_tangential_direction_variance(
    w_t_vectors: np.ndarray, h_hat: np.ndarray
) -> dict:
    """Gap explanation: directional diversity of tangential components.

    For each token: û_t = w_t_⊥ / ||w_t_⊥|| (normalized tangential direction).
    Pairwise cosines between all û_t pairs.

    mean_pairwise_cos ≈ 1.0 → tangential directions aligned →
        scalar model should capture it → gap comes from something else
    mean_pairwise_cos ≈ 0 → diverse tangential directions →
        scalar model loses directional info → explains the R² ≈ 0.7 gap
    """
    T = w_t_vectors.shape[0]

    # Compute tangential components
    # w_t_⊥ = w_t - ⟨w_t, ĥ⟩ ĥ
    radial_projections = w_t_vectors @ h_hat  # [T]
    w_t_perp = w_t_vectors - np.outer(radial_projections, h_hat)  # [T, d]
    norms = np.linalg.norm(w_t_perp, axis=1)  # [T]

    # Filter tokens with negligible tangential component
    valid = norms > 1e-10
    n_valid = int(np.sum(valid))

    if n_valid < 2:
        return {
            "mean_pairwise_cos": float("nan"),
            "std_pairwise_cos": float("nan"),
            "min_pairwise_cos": float("nan"),
            "max_pairwise_cos": float("nan"),
            "n_valid_tokens": n_valid,
            "n_pairs": 0,
        }

    # Normalized tangential directions
    u_hat = w_t_perp[valid] / norms[valid, np.newaxis]  # [n_valid, d]

    # Pairwise cosine matrix
    cos_matrix = u_hat @ u_hat.T  # [n_valid, n_valid]

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_valid, k=1)
    pairwise_cos = cos_matrix[triu_indices]
    abs_pairwise_cos = np.abs(pairwise_cos)

    return {
        "mean_pairwise_cos": float(np.mean(abs_pairwise_cos)),
        "std_pairwise_cos": float(np.std(abs_pairwise_cos)),
        "min_pairwise_cos": float(np.min(abs_pairwise_cos)),
        "max_pairwise_cos": float(np.max(abs_pairwise_cos)),
        "n_valid_tokens": n_valid,
        "n_pairs": len(pairwise_cos),
        "mean_signed_cos": float(np.mean(pairwise_cos)),
    }


def decomposition_verification(
    w_t_vectors: np.ndarray,
    h_hat: np.ndarray,
    g_proj: np.ndarray,
    grad_over_alpha: np.ndarray,
    intercept: float,
) -> dict:
    """Decompose ⟨g, w_t⟩ into radial and tangential terms.

    Verify R²(radial) + R²(tangential) ≈ R²(both) ≈ vector R².

    - R²(radial only): y_hat = g_radial × r_t + c
    - R²(tangential only): y_hat = ⟨g_⊥, w_t_⊥⟩ + c
    - R²(both): should match vector regression R²
    """
    T = w_t_vectors.shape[0]
    y = grad_over_alpha
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean)**2))

    if ss_tot < 1e-30:
        return {
            "r_squared_radial_only": 0.0,
            "r_squared_tangential_only": 0.0,
            "r_squared_both": 0.0,
        }

    g_radial = float(np.dot(g_proj, h_hat))
    g_perp = g_proj - g_radial * h_hat

    # Per-token radial projections r_t = ⟨w_t, ĥ⟩
    r_t = w_t_vectors @ h_hat  # [T]
    # Per-token tangential components: w_t_⊥ = w_t - r_t ĥ
    w_t_perp = w_t_vectors - np.outer(r_t, h_hat)  # [T, d]
    # Tangential inner products: ⟨g_⊥, w_t_⊥⟩
    tang_inner = w_t_perp @ g_perp  # [T]

    # R²(radial only): y_hat = g_radial × r_t + c
    X_rad = np.column_stack([r_t, np.ones(T)])
    beta_rad, _, _, _ = np.linalg.lstsq(X_rad, y, rcond=None)
    y_hat_rad = X_rad @ beta_rad
    ss_res_rad = float(np.sum((y - y_hat_rad)**2))
    r_sq_radial = 1.0 - ss_res_rad / ss_tot

    # R²(tangential only): y_hat = ⟨g_⊥, w_t_⊥⟩ + c
    X_tang = np.column_stack([tang_inner, np.ones(T)])
    beta_tang, _, _, _ = np.linalg.lstsq(X_tang, y, rcond=None)
    y_hat_tang = X_tang @ beta_tang
    ss_res_tang = float(np.sum((y - y_hat_tang)**2))
    r_sq_tangential = 1.0 - ss_res_tang / ss_tot

    # R²(both): y_hat = g_radial × r_t + ⟨g_⊥, w_t_⊥⟩ + c
    X_both = np.column_stack([r_t, tang_inner, np.ones(T)])
    beta_both, _, _, _ = np.linalg.lstsq(X_both, y, rcond=None)
    y_hat_both = X_both @ beta_both
    ss_res_both = float(np.sum((y - y_hat_both)**2))
    r_sq_both = 1.0 - ss_res_both / ss_tot

    return {
        "r_squared_radial_only": r_sq_radial,
        "r_squared_tangential_only": r_sq_tangential,
        "r_squared_both": r_sq_both,
        "g_radial_scalar": g_radial,
        "g_perp_norm": float(np.linalg.norm(g_perp)),
        "beta_radial_fit": beta_rad.tolist(),
        "beta_tangential_fit": beta_tang.tolist(),
    }


def cross_head_g_similarity(
    g_projs: dict[tuple[int, int], np.ndarray],
) -> dict:
    """g direction similarity across heads (within-layer and across-layer).

    Pairwise cosine of g_proj across all heads.
    """
    heads = sorted(g_projs.keys())
    if len(heads) < 2:
        return {"within_layer": {}, "across_layer": {}, "n_heads": len(heads)}

    # Organize by layer
    by_layer: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (l, h), g in g_projs.items():
        by_layer.setdefault(l, []).append((h, g))

    within_layer = {}
    for layer_idx, head_gs in by_layer.items():
        if len(head_gs) < 2:
            continue
        cosines = []
        for i in range(len(head_gs)):
            for j in range(i + 1, len(head_gs)):
                g_i = head_gs[i][1]
                g_j = head_gs[j][1]
                n_i = float(np.linalg.norm(g_i))
                n_j = float(np.linalg.norm(g_j))
                if n_i > 1e-30 and n_j > 1e-30:
                    cosines.append(float(np.dot(g_i, g_j) / (n_i * n_j)))
        if cosines:
            within_layer[str(layer_idx)] = {
                "mean_cos": float(np.mean(cosines)),
                "std_cos": float(np.std(cosines)),
                "min_cos": float(np.min(cosines)),
                "max_cos": float(np.max(cosines)),
                "n_pairs": len(cosines),
            }

    # Across-layer: all pairs from different layers
    across_cosines = []
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            l_i, h_i = heads[i]
            l_j, h_j = heads[j]
            if l_i == l_j:
                continue
            g_i = g_projs[heads[i]]
            g_j = g_projs[heads[j]]
            n_i = float(np.linalg.norm(g_i))
            n_j = float(np.linalg.norm(g_j))
            if n_i > 1e-30 and n_j > 1e-30:
                across_cosines.append(float(np.dot(g_i, g_j) / (n_i * n_j)))

    across_layer = {}
    if across_cosines:
        across_layer = {
            "mean_cos": float(np.mean(across_cosines)),
            "std_cos": float(np.std(across_cosines)),
            "min_cos": float(np.min(across_cosines)),
            "max_cos": float(np.max(across_cosines)),
            "n_pairs": len(across_cosines),
        }

    return {
        "within_layer": within_layer,
        "across_layer": across_layer,
        "n_heads": len(heads),
    }


# =====================================================================
# Single model runner
# =====================================================================


def run_single_model(
    model_name: str, model_path: str, probes: list[str] | None = None
) -> dict:
    """Run Phase 2 vector analysis on one model across all probes."""
    import mlx.core as mx
    import mlx_lm

    from modelcypher.backends.mlx_backend import MLXBackend

    if probes is None:
        probes = PROBE_TEXTS

    logger.info("=" * 70)
    logger.info("Model: %s (%s)", model_name, model_path)
    logger.info("=" * 70)

    backend = MLXBackend()
    model, tokenizer = mlx_lm.load(model_path)

    probe_results = []

    for probe_idx, text in enumerate(probes):
        logger.info("--- Probe %d: %r ---", probe_idx + 1, text)

        # Reuse A7 pipeline
        logger.info("  Collecting baseline...")
        baseline = collect_baseline(model, tokenizer, text, backend, mx)
        seq_len = len(baseline["token_ids"])
        logger.info("  Baseline loss: %.6f, seq_len: %d", baseline["baseline_loss"], seq_len)

        logger.info("  Computing score gradients...")
        gradients = compute_score_gradients(model, tokenizer, text, baseline, backend, mx)

        logger.info("  Computing radial projections (with vectors)...")
        radial_data = compute_radial_projections(
            model, tokenizer, text, baseline, baseline["attn_weights"], backend, mx
        )

        logger.info("  Collecting per-head gradient data...")
        mono_data = collect_per_head_gradient_data(
            baseline["attn_weights"], gradients, baseline, mx
        )

        # Phase 2 analysis per head
        logger.info("  Running Phase 2 vector analysis...")
        per_head_results: dict[str, dict[str, dict]] = {}
        g_projs_for_cross: dict[tuple[int, int], np.ndarray] = {}

        for layer_idx in sorted(mono_data.keys()):
            l_key = str(layer_idx)
            per_head_results[l_key] = {}

            for head_idx in sorted(mono_data[layer_idx].keys()):
                h_key = str(head_idx)
                mono_entry = mono_data[layer_idx][head_idx]

                if mono_entry.get("excluded"):
                    per_head_results[l_key][h_key] = {
                        "excluded": True,
                        "reason": mono_entry.get("reason", ""),
                    }
                    continue

                radial_entry = radial_data.get(layer_idx, {}).get(head_idx)
                if radial_entry is None or "w_t_vectors" not in radial_entry:
                    per_head_results[l_key][h_key] = {
                        "excluded": True,
                        "reason": "no vector data",
                    }
                    continue

                w_t_vecs = np.array(radial_entry["w_t_vectors"], dtype=np.float64)  # [T, d]
                h_hat = np.array(radial_entry["h_hat"], dtype=np.float64)  # [d]
                goa = np.array(mono_entry["grad_over_alpha"], dtype=np.float64)  # [T]

                # (a) Vector regression → R², g_proj
                vreg = vector_regression(w_t_vecs, goa)

                # (b) Analyze g_proj → radial/tangential fractions
                g_analysis = analyze_g_proj(vreg["g_proj"], h_hat)

                # (c) Tangential direction variance
                tang_var = measure_tangential_direction_variance(w_t_vecs, h_hat)

                # (d) Decomposition verification
                decomp = decomposition_verification(
                    w_t_vecs, h_hat, vreg["g_proj"], goa, vreg["intercept"]
                )

                # Store g_proj for cross-head analysis
                g_projs_for_cross[(layer_idx, head_idx)] = vreg["g_proj"]

                per_head_results[l_key][h_key] = {
                    "excluded": False,
                    "vector_regression": {
                        "r_squared": vreg["r_squared"],
                        "intercept": vreg["intercept"],
                        "residual_norm": vreg["residual_norm"],
                        "rank": vreg["rank"],
                        "T": vreg["T"],
                        "d": vreg["d"],
                    },
                    "g_analysis": g_analysis,
                    "tangential_variance": tang_var,
                    "decomposition": decomp,
                }

                logger.info(
                    "    L%d H%d: vec_R²=%.4f, cos(g,ĥ)=%+.3f, rad_frac=%.3f, "
                    "tang_cos=%.3f, R²(rad)=%.4f, R²(tang)=%.4f",
                    layer_idx, head_idx,
                    vreg["r_squared"],
                    g_analysis["cos_g_hhat"],
                    g_analysis["radial_fraction"],
                    tang_var.get("mean_pairwise_cos", float("nan")),
                    decomp["r_squared_radial_only"],
                    decomp["r_squared_tangential_only"],
                )

        # Cross-head g similarity
        logger.info("  Computing cross-head g similarity...")
        cross_head = cross_head_g_similarity(g_projs_for_cross)

        # Aggregate statistics
        all_vec_r2 = []
        all_cos_g_hhat = []
        all_radial_frac = []
        all_tang_cos = []
        all_r2_rad = []
        all_r2_tang = []
        all_r2_both = []

        for l_key in per_head_results:
            for h_key in per_head_results[l_key]:
                entry = per_head_results[l_key][h_key]
                if entry.get("excluded"):
                    continue

                vreg = entry["vector_regression"]
                if not math.isnan(vreg["r_squared"]):
                    all_vec_r2.append(vreg["r_squared"])

                ga = entry["g_analysis"]
                if not math.isnan(ga["cos_g_hhat"]):
                    all_cos_g_hhat.append(ga["cos_g_hhat"])
                if not math.isnan(ga["radial_fraction"]):
                    all_radial_frac.append(ga["radial_fraction"])

                tv = entry["tangential_variance"]
                if not math.isnan(tv.get("mean_pairwise_cos", float("nan"))):
                    all_tang_cos.append(tv["mean_pairwise_cos"])

                dec = entry["decomposition"]
                all_r2_rad.append(dec["r_squared_radial_only"])
                all_r2_tang.append(dec["r_squared_tangential_only"])
                all_r2_both.append(dec["r_squared_both"])

        aggregate = {
            "vector_r_squared": _safe_stats(all_vec_r2),
            "cos_g_hhat": _safe_stats(all_cos_g_hhat),
            "radial_fraction": _safe_stats(all_radial_frac),
            "tangential_pairwise_cos": _safe_stats(all_tang_cos),
            "r_squared_radial_only": _safe_stats(all_r2_rad),
            "r_squared_tangential_only": _safe_stats(all_r2_tang),
            "r_squared_both": _safe_stats(all_r2_both),
        }

        logger.info(
            "  Probe %d aggregate: vec_R² mean=%.4f, cos(g,ĥ) mean=%+.3f, "
            "rad_frac mean=%.3f, tang_cos mean=%.3f",
            probe_idx + 1,
            aggregate["vector_r_squared"]["mean"] or 0,
            aggregate["cos_g_hhat"]["mean"] or 0,
            aggregate["radial_fraction"]["mean"] or 0,
            aggregate["tangential_pairwise_cos"]["mean"] or 0,
        )

        probe_results.append({
            "probe_text": text,
            "seq_len": seq_len,
            "baseline_loss": baseline["baseline_loss"],
            "per_head": per_head_results,
            "cross_head_g_similarity": cross_head,
            "aggregate": aggregate,
        })

    return {
        "model": model_name,
        "model_path": model_path,
        "probes": probe_results,
    }


def _safe_stats(vals: list[float]) -> dict:
    """Summary statistics for a list of floats."""
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    s = sorted(vals)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "mean": sum(s) / n,
        "median": median,
        "min": s[0],
        "max": s[-1],
        "n": n,
    }


# =====================================================================
# Text summary writer
# =====================================================================


def write_text_summary(txt_path: Path, run_doc: dict) -> None:
    """Write human-readable Phase 2 analysis summary."""
    lines = [
        "A7 Phase 2 — Vector Gradient Structure Analysis",
        f"Run ID: {run_doc['run_id']}",
        f"Timestamp: {run_doc['timestamp']}",
        "",
        "Phase 1 found per-head R² ≈ 0.71-0.76 using scalar summaries.",
        "Phase 2 recovers the gradient vector g and decomposes it.",
        "",
        "Key questions:",
        "  1. Vector R² ≈ 1.0?  → identity confirmed, finite-diff noise is small",
        "  2. cos(g, ĥ) distribution → how radial is g actually?",
        "  3. R²(rad) vs R²(tang) → fraction from each component",
        "  4. tangential pairwise cos → can scalar model capture it?",
        "",
    ]

    for model_doc in run_doc["models"]:
        lines.append("=" * 70)
        lines.append(f"Model: {model_doc['model']}")
        lines.append("=" * 70)

        for probe_idx, pr in enumerate(model_doc["probes"]):
            lines.append(f"\n--- Probe {probe_idx + 1}: {pr['probe_text']!r} ---")
            lines.append(f"  seq_len={pr['seq_len']}, loss={pr['baseline_loss']:.6f}")

            agg = pr["aggregate"]
            for key, label in [
                ("vector_r_squared", "Vector R²"),
                ("cos_g_hhat", "cos(g, ĥ)"),
                ("radial_fraction", "Radial fraction"),
                ("tangential_pairwise_cos", "Tangential pairwise |cos|"),
                ("r_squared_radial_only", "R²(radial only)"),
                ("r_squared_tangential_only", "R²(tangential only)"),
                ("r_squared_both", "R²(both)"),
            ]:
                st = agg[key]
                if st["mean"] is not None:
                    lines.append(
                        f"  {label:30s}: mean={st['mean']:+.4f}, "
                        f"range=[{st['min']:+.4f}, {st['max']:+.4f}] (n={st['n']})"
                    )
                else:
                    lines.append(f"  {label:30s}: No data")

            # Cross-head g similarity
            cross = pr.get("cross_head_g_similarity", {})
            within = cross.get("within_layer", {})
            across = cross.get("across_layer", {})
            lines.append(f"\n  Cross-head g similarity:")
            for l_key, stats in sorted(within.items()):
                lines.append(
                    f"    Layer {l_key} (within): mean_cos={stats['mean_cos']:+.4f}, "
                    f"range=[{stats['min_cos']:+.4f}, {stats['max_cos']:+.4f}] "
                    f"(n_pairs={stats['n_pairs']})"
                )
            if across:
                lines.append(
                    f"    Across layers: mean_cos={across['mean_cos']:+.4f}, "
                    f"range=[{across['min_cos']:+.4f}, {across['max_cos']:+.4f}] "
                    f"(n_pairs={across['n_pairs']})"
                )

            # Per-head detail
            lines.append(f"\n  Per-head detail:")
            for l_key in sorted(pr["per_head"].keys(), key=int):
                for h_key in sorted(pr["per_head"][l_key].keys(), key=int):
                    entry = pr["per_head"][l_key][h_key]
                    if entry.get("excluded"):
                        lines.append(f"    L{l_key} H{h_key}: EXCLUDED")
                        continue

                    vreg = entry["vector_regression"]
                    ga = entry["g_analysis"]
                    dec = entry["decomposition"]
                    tv = entry["tangential_variance"]
                    lines.append(
                        f"    L{l_key} H{h_key}: vec_R²={vreg['r_squared']:.4f}, "
                        f"cos(g,ĥ)={ga['cos_g_hhat']:+.3f}, "
                        f"rad_frac={ga['radial_fraction']:.3f}, "
                        f"R²(rad)={dec['r_squared_radial_only']:.4f}, "
                        f"R²(tang)={dec['r_squared_tangential_only']:.4f}, "
                        f"R²(both)={dec['r_squared_both']:.4f}, "
                        f"tang_cos={tv.get('mean_pairwise_cos', float('nan')):.3f}"
                    )

        lines.append("")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A7 Phase 2 — Vector Gradient Structure Analysis"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model keys from registry: {sorted(PHASE2_MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        default="results/a7_validation/phase2_vector",
        help="Output directory root (run_id subdir is created).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id for output subdirectory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 model, 1 probe.",
    )
    args = parser.parse_args()

    logger.info("A7 Phase 2 — Vector Gradient Structure Analysis")
    logger.info("Probes: %d", len(PROBE_TEXTS))

    if args.smoke:
        model_names = [args.models[0]] if args.models else ["LFM2-350M"]
        probes = PROBE_TEXTS[:1]
    else:
        model_names = args.models if args.models else list(PHASE2_MODELS.keys())
        probes = PROBE_TEXTS

    # Validate model paths
    for name in model_names:
        path = PHASE2_MODELS.get(name)
        if path is None:
            logger.error("Unknown model key: %s", name)
            sys.exit(1)
        if not Path(path).exists():
            logger.error("Model not found: %s", path)
            logger.error("Is the external volume mounted?")
            sys.exit(1)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    models_data = []
    for name in model_names:
        path = PHASE2_MODELS[name]
        t0 = time.time()
        model_doc = run_single_model(name, path, probes=probes)
        model_doc["elapsed_sec"] = time.time() - t0
        models_data.append(model_doc)

    run_doc = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_version": "a7_phase2_v1",
        "query_position": _QUERY_POS,
        "models": models_data,
    }

    json_path = out_dir / "a7_phase2_vector.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    txt_path = out_dir / "a7_phase2_vector.txt"
    write_text_summary(txt_path, run_doc)
    logger.info("Wrote %s", txt_path)


if __name__ == "__main__":
    main()
