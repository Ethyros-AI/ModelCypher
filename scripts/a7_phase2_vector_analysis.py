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

# Reuse validated functions from A7 validator
from validate_a7_assumption import (
    _QUERY_POS,
    MODELS,
    PROBE_TEXTS,
    collect_baseline,
    collect_per_head_gradient_data,
    compute_radial_projections,
    compute_score_gradients,
)

# IEEE 754 float64 constants for numerical guards
_EPS_F64 = math.ldexp(1.0, -52)  # ≈ 2.22e-16
# Geometric mean of 1 and eps: floor for directions whose unit vector is noise-dominated
_SQRT_EPS_F64 = math.sqrt(_EPS_F64)  # ≈ 1.49e-8

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
# Analysis functions (backend-native + Python scalars)
# =====================================================================


def _as_float_list(x: list[float] | tuple[float, ...] | float | int) -> list[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, tuple):
        return [float(v) for v in x]
    return [float(x)]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _mean(a: list[float]) -> float:
    if not a:
        return float("nan")
    return float(sum(a) / len(a))


def _std(a: list[float]) -> float:
    if not a:
        return float("nan")
    mu = _mean(a)
    return math.sqrt(sum((x - mu) ** 2 for x in a) / len(a))


def _r_squared(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = _mean(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    if ss_tot < _EPS_F64:
        return 0.0
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot


def _append_ones_column(backend, rows: list[list[float]]):
    if not rows:
        return backend.array([], dtype="float32")
    out = [row + [1.0] for row in rows]
    return backend.array(out, dtype="float32")


def _lstsq_via_pinv(backend, x_mat, y_vec):
    """Minimum-norm least squares via pseudoinverse: beta = pinv(X) @ y."""
    n_rows, n_cols = backend.shape(x_mat)
    y_col = backend.reshape(y_vec, (n_rows, 1))
    beta_col = backend.matmul(backend.pinv(x_mat), y_col)
    beta = backend.reshape(beta_col, (n_cols,))
    svals = backend.svd(x_mat, compute_uv=False)
    backend.eval(beta, svals)

    s_list = _as_float_list(backend.tolist(svals))
    if not s_list:
        rank = 0
    else:
        s_max = max(s_list)
        tol = _SQRT_EPS_F64 * s_max
        rank = sum(1 for s in s_list if s > tol)
    return beta, rank


def vector_regression(
    w_t_vectors, grad_over_alpha, backend
) -> dict:
    """Recover g from ⟨g, w_t⟩ = ∂L/∂s_t / α_t - const via held-out split.

    Since T << d (e.g. T=10, d=1024), full-data least squares can interpolate
    any y. The falsifiable metric is held-out R² from split-half prediction.
    """
    T, d = backend.shape(w_t_vectors)
    w_rows = backend.tolist(w_t_vectors)
    y_vals = _as_float_list(backend.tolist(grad_over_alpha))

    even_idx = [i for i in range(T) if i % 2 == 0]
    odd_idx = [i for i in range(T) if i % 2 == 1]

    held_out_r_squared = float("nan")
    if len(even_idx) >= 2 and len(odd_idx) >= 1:
        x_fit_rows = [w_rows[i] for i in even_idx]
        y_fit_vals = [y_vals[i] for i in even_idx]
        x_test_rows = [w_rows[i] for i in odd_idx]
        y_test_vals = [y_vals[i] for i in odd_idx]

        x_fit = _append_ones_column(backend, x_fit_rows)
        y_fit = backend.array(y_fit_vals, dtype="float32")
        beta_fit, _ = _lstsq_via_pinv(backend, x_fit, y_fit)

        x_test = _append_ones_column(backend, x_test_rows)
        y_pred = backend.matmul(x_test, backend.reshape(beta_fit, (d + 1, 1)))
        backend.eval(y_pred)
        y_pred_vals = _as_float_list(backend.tolist(backend.reshape(y_pred, (len(odd_idx),))))
        held_out_r_squared = _r_squared(y_test_vals, y_pred_vals)

    # Full-data fit for downstream decomposition (tautological metric retained for logging)
    x_full = _append_ones_column(backend, w_rows)
    y_full = backend.array(y_vals, dtype="float32")
    beta, rank = _lstsq_via_pinv(backend, x_full, y_full)
    beta_list = _as_float_list(backend.tolist(beta))
    g_proj = beta_list[:d]
    intercept = float(beta_list[d])

    y_hat = backend.matmul(x_full, backend.reshape(beta, (d + 1, 1)))
    backend.eval(y_hat)
    y_hat_vals = _as_float_list(backend.tolist(backend.reshape(y_hat, (T,))))
    full_r_squared = _r_squared(y_vals, y_hat_vals)
    residual_norm = math.sqrt(sum((y - y_hat_v) ** 2 for y, y_hat_v in zip(y_vals, y_hat_vals)))

    return {
        "held_out_r_squared": held_out_r_squared,
        "full_r_squared_tautological": full_r_squared,
        "g_proj": g_proj,
        "intercept": intercept,
        "residual_norm": residual_norm,
        "rank": int(rank),
        "T": T,
        "d": d,
        "n_fit": len(even_idx),
        "n_test": len(odd_idx),
    }


def analyze_g_proj(g_proj: list[float], h_hat: list[float]) -> dict:
    """Gradient direction characterization."""
    g_norm = _norm(g_proj)
    if g_norm < _EPS_F64:
        return {
            "g_radial": 0.0,
            "g_perp_norm": 0.0,
            "cos_g_hhat": 0.0,
            "radial_fraction": 0.0,
            "g_norm": 0.0,
        }

    g_radial = _dot(g_proj, h_hat)
    g_perp = [g - g_radial * h for g, h in zip(g_proj, h_hat)]
    g_perp_norm = _norm(g_perp)
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
    w_t_vectors: list[list[float]], h_hat: list[float]
) -> dict:
    """Directional diversity of tangential components."""
    u_hat: list[list[float]] = []
    for w in w_t_vectors:
        r = _dot(w, h_hat)
        w_perp = [w_i - r * h_i for w_i, h_i in zip(w, h_hat)]
        nrm = _norm(w_perp)
        if nrm > _SQRT_EPS_F64:
            u_hat.append([v / nrm for v in w_perp])

    n_valid = len(u_hat)
    if n_valid < 2:
        return {
            "mean_pairwise_cos": float("nan"),
            "std_pairwise_cos": float("nan"),
            "min_pairwise_cos": float("nan"),
            "max_pairwise_cos": float("nan"),
            "n_valid_tokens": n_valid,
            "n_pairs": 0,
        }

    pairwise_cos: list[float] = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            pairwise_cos.append(_dot(u_hat[i], u_hat[j]))

    abs_pairwise_cos = [abs(v) for v in pairwise_cos]
    return {
        "mean_pairwise_cos": _mean(abs_pairwise_cos),
        "std_pairwise_cos": _std(abs_pairwise_cos),
        "min_pairwise_cos": min(abs_pairwise_cos),
        "max_pairwise_cos": max(abs_pairwise_cos),
        "n_valid_tokens": n_valid,
        "n_pairs": len(pairwise_cos),
        "mean_signed_cos": _mean(pairwise_cos),
    }


def decomposition_verification(
    w_t_vectors: list[list[float]],
    h_hat: list[float],
    g_proj: list[float],
    grad_over_alpha: list[float],
    intercept: float,
) -> dict:
    """Fixed-coefficient decomposition verification (no re-fitting)."""
    y = [float(v) for v in grad_over_alpha]
    y_mean = _mean(y)
    ss_tot = sum((v - y_mean) ** 2 for v in y)

    if ss_tot < _EPS_F64:
        return {
            "r_squared_radial_only": 0.0,
            "r_squared_tangential_only": 0.0,
            "r_squared_both": 0.0,
            "g_radial_scalar": 0.0,
            "g_perp_norm": 0.0,
        }

    g_radial = _dot(g_proj, h_hat)
    g_perp = [g - g_radial * h for g, h in zip(g_proj, h_hat)]

    r_t: list[float] = []
    tang_inner: list[float] = []
    for w in w_t_vectors:
        r = _dot(w, h_hat)
        r_t.append(r)
        w_perp = [w_i - r * h_i for w_i, h_i in zip(w, h_hat)]
        tang_inner.append(_dot(w_perp, g_perp))

    y_hat_rad = [g_radial * r + intercept for r in r_t]
    y_hat_tang = [t + intercept for t in tang_inner]
    y_hat_both = [g_radial * r + t + intercept for r, t in zip(r_t, tang_inner)]

    r_sq_radial = _r_squared(y, y_hat_rad)
    r_sq_tangential = _r_squared(y, y_hat_tang)
    r_sq_both = _r_squared(y, y_hat_both)

    return {
        "r_squared_radial_only": r_sq_radial,
        "r_squared_tangential_only": r_sq_tangential,
        "r_squared_both": r_sq_both,
        "g_radial_scalar": g_radial,
        "g_perp_norm": _norm(g_perp),
    }


def cross_head_g_similarity(
    g_projs: dict[tuple[int, int], list[float]],
) -> dict:
    """g direction similarity across heads (within-layer and across-layer)."""
    heads = sorted(g_projs.keys())
    if len(heads) < 2:
        return {"within_layer": {}, "across_layer": {}, "n_heads": len(heads)}

    by_layer: dict[int, list[tuple[int, list[float]]]] = {}
    for (layer_idx, head_idx), g in g_projs.items():
        by_layer.setdefault(layer_idx, []).append((head_idx, g))

    within_layer = {}
    for layer_idx, head_gs in by_layer.items():
        if len(head_gs) < 2:
            continue
        cosines: list[float] = []
        for i in range(len(head_gs)):
            for j in range(i + 1, len(head_gs)):
                g_i = head_gs[i][1]
                g_j = head_gs[j][1]
                n_i = _norm(g_i)
                n_j = _norm(g_j)
                if n_i > _EPS_F64 and n_j > _EPS_F64:
                    cosines.append(_dot(g_i, g_j) / (n_i * n_j))
        if cosines:
            within_layer[str(layer_idx)] = {
                "mean_cos": _mean(cosines),
                "std_cos": _std(cosines),
                "min_cos": min(cosines),
                "max_cos": max(cosines),
                "n_pairs": len(cosines),
            }

    across_cosines: list[float] = []
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            l_i, _ = heads[i]
            l_j, _ = heads[j]
            if l_i == l_j:
                continue
            g_i = g_projs[heads[i]]
            g_j = g_projs[heads[j]]
            n_i = _norm(g_i)
            n_j = _norm(g_j)
            if n_i > _EPS_F64 and n_j > _EPS_F64:
                across_cosines.append(_dot(g_i, g_j) / (n_i * n_j))

    across_layer = {}
    if across_cosines:
        across_layer = {
            "mean_cos": _mean(across_cosines),
            "std_cos": _std(across_cosines),
            "min_cos": min(across_cosines),
            "max_cos": max(across_cosines),
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
            model, tokenizer, text, baseline, baseline["attn_weights"], backend, mx,
            include_vectors=True,
        )

        logger.info("  Collecting per-head gradient data...")
        mono_data = collect_per_head_gradient_data(
            baseline["attn_weights"], gradients, baseline, mx
        )

        # Phase 2 analysis per head
        logger.info("  Running Phase 2 vector analysis...")
        per_head_results: dict[str, dict[str, dict]] = {}
        g_projs_for_cross: dict[tuple[int, int], list[float]] = {}

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

                w_t_vecs = backend.array(radial_entry["w_t_vectors"], dtype="float32")  # [T, d]
                h_hat = [float(v) for v in radial_entry["h_hat"]]  # [d]
                goa = backend.array(mono_entry["grad_over_alpha"], dtype="float32")  # [T]

                # (a) Vector regression → R², g_proj
                vreg = vector_regression(w_t_vecs, goa, backend)

                # (b) Analyze g_proj → radial/tangential fractions
                g_analysis = analyze_g_proj(vreg["g_proj"], h_hat)

                # (c) Tangential direction variance
                tang_var = measure_tangential_direction_variance(radial_entry["w_t_vectors"], h_hat)

                # (d) Decomposition verification
                decomp = decomposition_verification(
                    radial_entry["w_t_vectors"],
                    h_hat,
                    vreg["g_proj"],
                    mono_entry["grad_over_alpha"],
                    vreg["intercept"],
                )

                # Store g_proj for cross-head analysis
                g_projs_for_cross[(layer_idx, head_idx)] = vreg["g_proj"]

                per_head_results[l_key][h_key] = {
                    "excluded": False,
                    "vector_regression": {
                        "held_out_r_squared": vreg["held_out_r_squared"],
                        "full_r_squared_tautological": vreg["full_r_squared_tautological"],
                        "intercept": vreg["intercept"],
                        "residual_norm": vreg["residual_norm"],
                        "rank": vreg["rank"],
                        "T": vreg["T"],
                        "d": vreg["d"],
                        "n_fit": vreg["n_fit"],
                        "n_test": vreg["n_test"],
                    },
                    "g_analysis": g_analysis,
                    "tangential_variance": tang_var,
                    "decomposition": decomp,
                }

                logger.info(
                    "    L%d H%d: held_out_R²=%.4f, cos(g,ĥ)=%+.3f, rad_frac=%.3f, "
                    "tang_cos=%.3f, R²(rad)=%.4f, R²(tang)=%.4f",
                    layer_idx, head_idx,
                    vreg["held_out_r_squared"],
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
        all_held_out_r2 = []
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
                if not math.isnan(vreg["held_out_r_squared"]):
                    all_held_out_r2.append(vreg["held_out_r_squared"])

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
            "held_out_r_squared": _safe_stats(all_held_out_r2),
            "cos_g_hhat": _safe_stats(all_cos_g_hhat),
            "radial_fraction": _safe_stats(all_radial_frac),
            "tangential_pairwise_cos": _safe_stats(all_tang_cos),
            "r_squared_radial_only": _safe_stats(all_r2_rad),
            "r_squared_tangential_only": _safe_stats(all_r2_tang),
            "r_squared_both": _safe_stats(all_r2_both),
        }

        logger.info(
            "  Probe %d aggregate: held_out_R² mean=%.4f, cos(g,ĥ) mean=%+.3f, "
            "rad_frac mean=%.3f, tang_cos mean=%.3f",
            probe_idx + 1,
            aggregate["held_out_r_squared"]["mean"] or 0,
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
        "  1. Held-out R² > 0?  → identity confirmed (falsifiable: T<<d split test)",
        "  2. cos(g, ĥ) distribution → how radial is g actually?",
        "  3. R²(rad) vs R²(tang) → fraction from each component (fixed coefficients)",
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
                ("held_out_r_squared", "Held-out R² (falsifiable)"),
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
            lines.append("\n  Cross-head g similarity:")
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
            lines.append("\n  Per-head detail:")
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
                        f"    L{l_key} H{h_key}: held_out_R²={vreg['held_out_r_squared']:.4f}, "
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
