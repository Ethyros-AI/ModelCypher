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

"""A7 Multivariate Gradient Structure Analysis.

A7 (radial-dominant gradient) is falsified. Univariate R²(radial) ≈ 0.16.
Best univariate correlate is token position (|ρ| ≈ 0.37). Each predictor
alone explains only ~30-37%.

The exact identity: ∂L/∂s_t / α_t = ⟨g, w_t⟩ + const, where
g = ∂L/∂δ (fixed per layer/head/input), w_t = W_O_head @ v_t.
ALL gradient structure comes from how ⟨g, w_t⟩ varies across tokens.

This script asks: can we decompose ⟨g, w_t⟩ into simple geometric
quantities of w_t? Design matrix per head:
    X = [1, position, α_t, r_t, |r_t - R|, ||w_t||, ||w_t^⊥||]

Pooled regression across all heads adds interaction terms.

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

# Diagnostic uses only 2 smallest models (model size policy)
DIAGNOSTIC_MODELS = {
    "LFM2-350M": MODELS["LFM2-350M"],
    "Qwen3.5-0.8B": MODELS["Qwen3.5-0.8B"],
}

# Predictor names for per-head regression (no intercept column name)
PER_HEAD_PREDICTORS = ["position", "alpha", "r_t", "abs_r_dev", "norm_w_t", "norm_w_t_perp"]
# Pooled adds interaction terms
POOLED_PREDICTORS = PER_HEAD_PREDICTORS + ["position_x_r_t", "position_x_alpha"]


# =====================================================================
# Statistical functions (pure numpy, no scipy)
# =====================================================================


def multivariate_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Full OLS regression via np.linalg.lstsq.

    X: [T, p+1] design matrix (column 0 = intercept = 1)
    y: [T] response

    Returns:
        beta: [p+1] coefficient vector
        r_squared: R²
        adj_r_squared: 1 - (1 - R²)(T - 1)/(T - p - 1)
        residuals: [T] residual vector
        std_beta: [p+1] standardized coefficients (β_j × std(x_j) / std(y))
    """
    T, p_plus_1 = X.shape
    p = p_plus_1 - 1  # number of predictors (excluding intercept)

    beta, residual_ss_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))

    if ss_tot < 1e-30:
        r_squared = 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    # Adjusted R²: penalizes for number of predictors
    dof_resid = T - p - 1
    if dof_resid > 0 and ss_tot > 1e-30:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (T - 1) / dof_resid
    else:
        adj_r_squared = float("nan")

    # Standardized coefficients: β_j × std(x_j) / std(y)
    std_y = float(np.std(y, ddof=1)) if T > 1 else 1.0
    std_beta = np.zeros(p_plus_1)
    for j in range(p_plus_1):
        std_xj = float(np.std(X[:, j], ddof=1)) if T > 1 else 1.0
        if std_y > 1e-30:
            std_beta[j] = beta[j] * std_xj / std_y
        else:
            std_beta[j] = 0.0

    return {
        "beta": beta.tolist(),
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "residuals": residuals.tolist(),
        "std_beta": std_beta.tolist(),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "rank": int(rank),
        "dof_resid": dof_resid,
    }


def compute_vif(X_no_intercept: np.ndarray, pred_names: list[str]) -> dict[str, float]:
    """Variance inflation factor per predictor via auxiliary R².

    VIF_j = 1 / (1 - R²_j) where R²_j is R² of regressing x_j on all
    other predictors. VIF > 10 = severe collinearity.
    """
    n_pred = X_no_intercept.shape[1]
    vif = {}

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for j in range(n_pred):
            y_j = X_no_intercept[:, j]
            # Other predictors + intercept
            others = np.delete(X_no_intercept, j, axis=1)
            X_aux = np.column_stack([np.ones(len(y_j)), others])

            beta_aux, _, _, _ = np.linalg.lstsq(X_aux, y_j, rcond=None)
            y_hat = X_aux @ beta_aux
            ss_res = float(np.sum((y_j - y_hat)**2))
            ss_tot = float(np.sum((y_j - np.mean(y_j))**2))

            if ss_tot < 1e-30:
                r_sq_aux = 0.0
            else:
                r_sq_aux = 1.0 - ss_res / ss_tot

            if r_sq_aux >= 1.0 - 1e-12:
                vif[pred_names[j]] = float("inf")
            else:
                vif[pred_names[j]] = 1.0 / (1.0 - r_sq_aux)

    return vif


def compute_partial_correlations(
    X_no_intercept: np.ndarray, y: np.ndarray, pred_names: list[str]
) -> dict[str, float]:
    """Partial correlation of each predictor with y after controlling for all others.

    Method: sequential residualization. For predictor j:
    1. Regress x_j on all other predictors → residual e_x
    2. Regress y on all other predictors → residual e_y
    3. partial_corr(x_j, y | rest) = Pearson(e_x, e_y)
    """
    n_pred = X_no_intercept.shape[1]
    partial_corrs = {}

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for j in range(n_pred):
            x_j = X_no_intercept[:, j]
            others = np.delete(X_no_intercept, j, axis=1)
            X_aux = np.column_stack([np.ones(len(x_j)), others])

            # Residualize x_j
            beta_x, _, _, _ = np.linalg.lstsq(X_aux, x_j, rcond=None)
            e_x = x_j - X_aux @ beta_x

            # Residualize y
            beta_y, _, _, _ = np.linalg.lstsq(X_aux, y, rcond=None)
            e_y = y - X_aux @ beta_y

            # Pearson correlation of residuals
            std_ex = float(np.std(e_x))
            std_ey = float(np.std(e_y))
            if std_ex < 1e-30 or std_ey < 1e-30:
                partial_corrs[pred_names[j]] = 0.0
            else:
                cov = float(np.mean(e_x * e_y) - np.mean(e_x) * np.mean(e_y))
                partial_corrs[pred_names[j]] = cov / (std_ex * std_ey)

    return partial_corrs


# =====================================================================
# Design matrix construction
# =====================================================================


def build_per_head_design_matrix(
    mono_entry: dict, radial_entry: dict, seq_len: int
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Build per-head design matrix X and response y.

    X = [1, position, α_t, r_t, |r_t - R|, ||w_t||, ||w_t^⊥||]
    y = ∂L/∂s_t / α_t (grad_over_alpha)

    Returns (X, y, predictor_names) or None if insufficient data.
    """
    if mono_entry.get("excluded") or radial_entry is None:
        return None

    grad_over_alpha = mono_entry["grad_over_alpha"]
    alpha = mono_entry["alpha"]
    r_t = radial_entry["r_t"]
    r_minus_R = radial_entry["r_minus_R"]
    norm_w_t = radial_entry.get("norm_w_t")
    norm_w_t_perp = radial_entry.get("norm_w_t_perp")

    if norm_w_t is None or norm_w_t_perp is None:
        return None

    T = len(grad_over_alpha)
    if T < 4:  # Need at least 4 observations for 7-column X
        return None

    positions = list(range(T))
    abs_r_dev = [abs(x) for x in r_minus_R]

    X = np.column_stack([
        np.ones(T),
        np.array(positions, dtype=np.float64),
        np.array(alpha, dtype=np.float64),
        np.array(r_t, dtype=np.float64),
        np.array(abs_r_dev, dtype=np.float64),
        np.array(norm_w_t, dtype=np.float64),
        np.array(norm_w_t_perp, dtype=np.float64),
    ])
    y = np.array(grad_over_alpha, dtype=np.float64)

    return X, y, PER_HEAD_PREDICTORS


def build_pooled_design_matrix(
    all_entries: list[tuple[dict, dict, int]],
    include_interactions: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Pool all tokens from all heads into a single design matrix.

    all_entries: list of (mono_entry, radial_entry, seq_len)
    """
    rows_X = []
    rows_y = []

    for mono_entry, radial_entry, seq_len in all_entries:
        result = build_per_head_design_matrix(mono_entry, radial_entry, seq_len)
        if result is None:
            continue
        X_head, y_head, _ = result
        rows_X.append(X_head)
        rows_y.append(y_head)

    if not rows_X:
        return None

    X_pooled = np.vstack(rows_X)  # [N_total, 7]
    y_pooled = np.concatenate(rows_y)  # [N_total]

    if include_interactions:
        # position × r_t (column indices: 1=position, 3=r_t)
        pos_x_rt = X_pooled[:, 1] * X_pooled[:, 3]
        # position × α_t (column indices: 1=position, 2=alpha)
        pos_x_alpha = X_pooled[:, 1] * X_pooled[:, 2]
        X_pooled = np.column_stack([X_pooled, pos_x_rt, pos_x_alpha])
        pred_names = POOLED_PREDICTORS
    else:
        pred_names = PER_HEAD_PREDICTORS

    return X_pooled, y_pooled, pred_names


# =====================================================================
# Single model runner
# =====================================================================


def run_single_model(
    model_name: str, model_path: str, probes: list[str] | None = None
) -> dict:
    """Run multivariate analysis on one model across all probes."""
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
    # Collect entries for model-level pooled regression
    all_model_entries: list[tuple[dict, dict, int]] = []

    for probe_idx, text in enumerate(probes):
        logger.info("--- Probe %d: %r ---", probe_idx + 1, text)

        # Reuse A7 pipeline
        logger.info("  Collecting baseline...")
        baseline = collect_baseline(model, tokenizer, text, backend, mx)
        seq_len = len(baseline["token_ids"])
        logger.info("  Baseline loss: %.6f, seq_len: %d", baseline["baseline_loss"], seq_len)

        logger.info("  Computing score gradients...")
        gradients = compute_score_gradients(model, tokenizer, text, baseline, backend, mx)

        logger.info("  Computing radial projections (with norms)...")
        radial_data = compute_radial_projections(
            model, tokenizer, text, baseline, baseline["attn_weights"], backend, mx
        )

        logger.info("  Collecting per-head gradient data...")
        mono_data = collect_per_head_gradient_data(
            baseline["attn_weights"], gradients, baseline, mx
        )

        # Per-head multivariate regression
        logger.info("  Running per-head OLS regressions...")
        per_head_results: dict[str, dict[str, dict]] = {}
        probe_entries: list[tuple[dict, dict, int]] = []

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
                if radial_entry is None:
                    per_head_results[l_key][h_key] = {
                        "excluded": True,
                        "reason": "no radial data",
                    }
                    continue

                result = build_per_head_design_matrix(mono_entry, radial_entry, seq_len)
                if result is None:
                    per_head_results[l_key][h_key] = {
                        "excluded": True,
                        "reason": "insufficient data for design matrix",
                    }
                    continue

                X, y, pred_names = result
                probe_entries.append((mono_entry, radial_entry, seq_len))

                # OLS
                ols = multivariate_ols(X, y)

                # VIF (on predictors only, no intercept column)
                X_no_int = X[:, 1:]
                vif = compute_vif(X_no_int, pred_names)

                # Partial correlations
                partial_corrs = compute_partial_correlations(X_no_int, y, pred_names)

                # Dominant predictor by |standardized β|
                # std_beta[0] is intercept, skip it
                abs_std_beta = [abs(b) for b in ols["std_beta"][1:]]
                dominant_idx = int(np.argmax(abs_std_beta))
                dominant_pred = pred_names[dominant_idx]

                per_head_results[l_key][h_key] = {
                    "excluded": False,
                    "r_squared": ols["r_squared"],
                    "adj_r_squared": ols["adj_r_squared"],
                    "beta": ols["beta"],
                    "std_beta": ols["std_beta"],
                    "vif": vif,
                    "partial_correlations": partial_corrs,
                    "dominant_predictor": dominant_pred,
                    "dominant_std_beta": abs_std_beta[dominant_idx],
                    "dof_resid": ols["dof_resid"],
                    "rank": ols["rank"],
                }

        all_model_entries.extend(probe_entries)

        # Per-probe aggregate
        all_r_sq = []
        all_adj_r_sq = []
        dominant_counts: dict[str, int] = {}
        for l_key in per_head_results:
            for h_key in per_head_results[l_key]:
                entry = per_head_results[l_key][h_key]
                if entry.get("excluded"):
                    continue
                r_sq = entry["r_squared"]
                if not math.isnan(r_sq):
                    all_r_sq.append(r_sq)
                adj = entry["adj_r_squared"]
                if not math.isnan(adj):
                    all_adj_r_sq.append(adj)
                dom = entry["dominant_predictor"]
                dominant_counts[dom] = dominant_counts.get(dom, 0) + 1

        probe_agg = {
            "r_squared": _safe_stats(all_r_sq),
            "adj_r_squared": _safe_stats(all_adj_r_sq),
            "dominant_predictor_counts": dominant_counts,
            "n_heads_included": len(all_r_sq),
        }

        logger.info(
            "  Probe %d per-head R²: mean=%.4f, median=%.4f (n=%d)",
            probe_idx + 1,
            probe_agg["r_squared"]["mean"] or 0,
            probe_agg["r_squared"]["median"] or 0,
            probe_agg["r_squared"]["n"],
        )
        logger.info("  Dominant predictor counts: %s", dominant_counts)

        probe_results.append({
            "probe_text": text,
            "seq_len": seq_len,
            "baseline_loss": baseline["baseline_loss"],
            "per_head": per_head_results,
            "aggregate": probe_agg,
        })

    # Model-level pooled regression
    logger.info("Running pooled regression (all probes, all heads)...")
    pooled_results = {}
    for label, include_interactions in [("base", False), ("interactions", True)]:
        result = build_pooled_design_matrix(all_model_entries, include_interactions)
        if result is None:
            pooled_results[label] = {"error": "no data for pooled regression"}
            continue

        X_pooled, y_pooled, pred_names = result
        T_pooled = len(y_pooled)

        ols = multivariate_ols(X_pooled, y_pooled)
        X_no_int = X_pooled[:, 1:]
        vif = compute_vif(X_no_int, pred_names)
        partial_corrs = compute_partial_correlations(X_no_int, y_pooled, pred_names)

        # Dominant predictor
        abs_std_beta = [abs(b) for b in ols["std_beta"][1:]]
        dominant_idx = int(np.argmax(abs_std_beta))
        dominant_pred = pred_names[dominant_idx]

        pooled_results[label] = {
            "n_observations": T_pooled,
            "r_squared": ols["r_squared"],
            "adj_r_squared": ols["adj_r_squared"],
            "beta": ols["beta"],
            "std_beta": ols["std_beta"],
            "predictor_names": ["intercept"] + pred_names,
            "vif": vif,
            "partial_correlations": partial_corrs,
            "dominant_predictor": dominant_pred,
            "dominant_std_beta": abs_std_beta[dominant_idx],
            "dof_resid": ols["dof_resid"],
            "rank": ols["rank"],
        }

        logger.info(
            "  Pooled [%s]: R²=%.4f, adj_R²=%.4f, n=%d, dominant=%s",
            label, ols["r_squared"], ols["adj_r_squared"], T_pooled, dominant_pred,
        )

    # Incremental R² from interactions
    if "base" in pooled_results and "interactions" in pooled_results:
        base_r2 = pooled_results["base"].get("r_squared", 0)
        inter_r2 = pooled_results["interactions"].get("r_squared", 0)
        if isinstance(base_r2, (int, float)) and isinstance(inter_r2, (int, float)):
            delta_r2 = inter_r2 - base_r2
            pooled_results["incremental_r_squared_from_interactions"] = delta_r2
            logger.info("  Incremental R² from interactions: %.4f", delta_r2)

    # VIF warnings
    vif_warnings = []
    for label in ["base", "interactions"]:
        if label not in pooled_results or "vif" not in pooled_results[label]:
            continue
        for pred, v in pooled_results[label]["vif"].items():
            if v > 10.0:
                vif_warnings.append(f"{label}/{pred}: VIF={v:.1f}")
    if vif_warnings:
        logger.warning("Collinearity warnings: %s", "; ".join(vif_warnings))

    return {
        "model": model_name,
        "model_path": model_path,
        "probes": probe_results,
        "pooled": pooled_results,
        "vif_warnings": vif_warnings,
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
    """Write human-readable multivariate analysis summary."""
    lines = [
        "A7 Multivariate Gradient Structure Analysis",
        f"Run ID: {run_doc['run_id']}",
        f"Timestamp: {run_doc['timestamp']}",
        "",
        "Central question: does pooled R² approach 1.0?",
        "If yes: simple geometric quantities collectively capture the gradient.",
        "If no: gradient lives in directions of w_t not captured by scalar summaries.",
        "",
        f"Per-head predictors: {', '.join(PER_HEAD_PREDICTORS)}",
        f"Pooled predictors (with interactions): {', '.join(POOLED_PREDICTORS)}",
        "",
    ]

    for model_doc in run_doc["models"]:
        lines.append("=" * 70)
        lines.append(f"Model: {model_doc['model']}")
        lines.append("=" * 70)

        # Pooled results
        pooled = model_doc.get("pooled", {})
        for label in ["base", "interactions"]:
            if label not in pooled or "error" in pooled[label]:
                lines.append(f"\nPooled [{label}]: NO DATA")
                continue

            p = pooled[label]
            lines.append(f"\nPooled [{label}] (n={p['n_observations']}):")
            lines.append(f"  R² = {p['r_squared']:.4f}")
            lines.append(f"  Adjusted R² = {p['adj_r_squared']:.4f}")
            lines.append(f"  Dominant predictor: {p['dominant_predictor']} "
                         f"(|std β| = {p['dominant_std_beta']:.4f})")
            lines.append(f"  DOF residual: {p['dof_resid']}")
            lines.append(f"  Matrix rank: {p['rank']}")

            # Coefficients table
            lines.append("  Coefficients (β, std_β):")
            for i, name in enumerate(p["predictor_names"]):
                lines.append(f"    {name:20s}: β={p['beta'][i]:+.6f}, "
                             f"std_β={p['std_beta'][i]:+.4f}")

            # Partial correlations
            lines.append("  Partial correlations:")
            for name, val in sorted(
                p["partial_correlations"].items(), key=lambda kv: -abs(kv[1])
            ):
                lines.append(f"    {name:20s}: {val:+.4f}")

            # VIF
            lines.append("  VIF (>10 = severe collinearity):")
            for name, val in sorted(p["vif"].items(), key=lambda kv: -kv[1]):
                flag = " *** COLLINEAR ***" if val > 10.0 else ""
                lines.append(f"    {name:20s}: {val:.2f}{flag}")

        # Incremental R²
        delta = pooled.get("incremental_r_squared_from_interactions")
        if delta is not None:
            lines.append(f"\nIncremental R² from interactions: {delta:.4f}")

        # VIF warnings
        warnings = model_doc.get("vif_warnings", [])
        if warnings:
            lines.append(f"\nCollinearity warnings: {'; '.join(warnings)}")

        # Per-probe summaries
        for probe_idx, pr in enumerate(model_doc["probes"]):
            lines.append(f"\n--- Probe {probe_idx + 1}: {pr['probe_text']!r} ---")
            lines.append(f"  seq_len={pr['seq_len']}, loss={pr['baseline_loss']:.6f}")

            agg = pr["aggregate"]
            r_sq = agg["r_squared"]
            adj = agg["adj_r_squared"]
            lines.append(
                f"  Per-head R²: mean={r_sq['mean']:.4f}, median={r_sq['median']:.4f}, "
                f"range=[{r_sq['min']:.4f}, {r_sq['max']:.4f}] (n={r_sq['n']})"
                if r_sq["mean"] is not None else "  Per-head R²: No data"
            )
            lines.append(
                f"  Per-head adj R²: mean={adj['mean']:.4f}, median={adj['median']:.4f}"
                if adj["mean"] is not None else "  Per-head adj R²: No data"
            )
            lines.append(f"  Dominant predictor counts: {agg['dominant_predictor_counts']}")

            # Per-head detail
            for l_key in sorted(pr["per_head"].keys(), key=int):
                for h_key in sorted(pr["per_head"][l_key].keys(), key=int):
                    entry = pr["per_head"][l_key][h_key]
                    if entry.get("excluded"):
                        lines.append(f"  L{l_key} H{h_key}: EXCLUDED")
                        continue

                    lines.append(
                        f"  L{l_key} H{h_key}: R²={entry['r_squared']:.4f}, "
                        f"adj_R²={entry['adj_r_squared']:.4f}, "
                        f"dom={entry['dominant_predictor']}"
                    )

        lines.append("")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A7 Multivariate Gradient Structure Analysis"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model keys from registry: {sorted(DIAGNOSTIC_MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        default="results/a7_validation/multivariate",
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

    logger.info("A7 Multivariate Gradient Structure Analysis")
    logger.info("Probes: %d", len(PROBE_TEXTS))

    if args.smoke:
        model_names = [args.models[0]] if args.models else ["LFM2-350M"]
        probes = PROBE_TEXTS[:1]
    else:
        model_names = args.models if args.models else list(DIAGNOSTIC_MODELS.keys())
        probes = PROBE_TEXTS

    # Validate model paths
    for name in model_names:
        path = DIAGNOSTIC_MODELS.get(name)
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
        path = DIAGNOSTIC_MODELS[name]
        t0 = time.time()
        model_doc = run_single_model(name, path, probes=probes)
        model_doc["elapsed_sec"] = time.time() - t0
        models_data.append(model_doc)

    run_doc = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_version": "a7_multivariate_v1",
        "query_position": _QUERY_POS,
        "per_head_predictors": PER_HEAD_PREDICTORS,
        "pooled_predictors": POOLED_PREDICTORS,
        "models": models_data,
    }

    json_path = out_dir / "a7_multivariate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    txt_path = out_dir / "a7_multivariate.txt"
    write_text_summary(txt_path, run_doc)
    logger.info("Wrote %s", txt_path)


if __name__ == "__main__":
    main()
