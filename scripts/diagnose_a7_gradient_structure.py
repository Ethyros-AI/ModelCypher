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

"""A7 Root Cause Diagnosis — Gradient Structure Analysis.

A7 is FALSIFIED across 5 models (0% pass rate). This script diagnoses:
1. R²(radial) — fraction of gradient variance explained by radial projection
2. Alternative correlates — what the gradient actually correlates with
3. Gradient concentration — effective support of the gradient signal

This is diagnostic, not a falsifier. All outputs are raw measurements.
No decision thresholds.

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
    MODELS,
    PROBE_TEXTS,
    _EPS_F32,
    _FLOAT32_MIN_NORMAL,
    _QUERY_POS,
    collect_baseline,
    collect_per_head_gradient_data,
    compute_radial_projections,
    compute_score_gradients,
    ols_slope,
    spearman_rank_correlation,
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


# =====================================================================
# Diagnostic functions
# =====================================================================


def pearson_r_squared(x: list[float], y: list[float]) -> float:
    """Fraction of variance in y explained by x (Pearson R²).

    R² = (Σ(x_i - x̄)(y_i - ȳ))² / (Σ(x_i - x̄)² × Σ(y_i - ȳ)²)
    """
    n = len(x)
    if n < 3:
        return float("nan")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    if var_x < _EPS_F32 or var_y < _EPS_F32:
        return 0.0

    r = cov_xy / math.sqrt(var_x * var_y)
    return r * r


def compute_gradient_concentration(dLds: list[float]) -> dict:
    """Effective support of gradient signal via Shannon entropy.

    k_eff = exp(H(|∂L/∂s_t|)) where H is entropy of the normalized
    absolute gradient distribution. k_eff/T near 1.0 = diffuse,
    near 1/T = concentrated on one token.
    """
    T = len(dLds)
    if T < 2:
        return {"k_eff": float("nan"), "k_eff_over_T": float("nan")}

    abs_grad = [abs(g) for g in dLds]
    total = sum(abs_grad)

    if total < _EPS_F32:
        # Zero gradient → maximally diffuse (no information)
        return {"k_eff": float(T), "k_eff_over_T": 1.0}

    # Normalized distribution
    p = [g / total for g in abs_grad]

    # Shannon entropy H = -Σ p_i log(p_i)
    H = 0.0
    for pi in p:
        if pi > _FLOAT32_MIN_NORMAL:
            H -= pi * math.log(pi)

    k_eff = math.exp(H)
    return {"k_eff": k_eff, "k_eff_over_T": k_eff / T}


def compute_alternative_correlates(
    mono_data: dict, radial_data: dict, seq_len: int
) -> dict[int, dict[int, dict]]:
    """Spearman ρ of ∂L/∂s_t against alternative predictors per (layer, head).

    Predictors:
    - position: token position index (positional bias under causal masking)
    - alpha: α_t itself (self-reinforcing attention)
    - r_t: raw radial projection
    - abs_r_dev: |r_t - R| (absolute radial deviation)
    """
    results: dict[int, dict[int, dict]] = {}

    positions = list(range(seq_len))

    for layer_idx in sorted(mono_data.keys()):
        results[layer_idx] = {}

        for head_idx in sorted(mono_data[layer_idx].keys()):
            mono = mono_data[layer_idx][head_idx]

            if mono.get("excluded", False):
                results[layer_idx][head_idx] = {"excluded": True}
                continue

            dLds = mono["dLds"]
            alpha = mono["alpha"]

            if layer_idx not in radial_data or head_idx not in radial_data[layer_idx]:
                results[layer_idx][head_idx] = {"excluded": True}
                continue

            rad = radial_data[layer_idx][head_idx]
            r_t = rad["r_t"]
            r_minus_R = rad["r_minus_R"]
            abs_r_dev = [abs(x) for x in r_minus_R]

            correlates = {
                "position": spearman_rank_correlation(dLds, positions),
                "alpha": spearman_rank_correlation(dLds, alpha),
                "r_t": spearman_rank_correlation(dLds, r_t),
                "abs_r_dev": spearman_rank_correlation(dLds, abs_r_dev),
            }

            # SE(ρ) = 1/√(n-1) under null (Kendall & Stuart)
            se_rho = 1.0 / math.sqrt(seq_len - 1) if seq_len > 1 else float("nan")

            # Rank by |ρ|
            ranked = sorted(correlates.items(), key=lambda kv: abs(kv[1]), reverse=True)

            results[layer_idx][head_idx] = {
                "excluded": False,
                "correlates": correlates,
                "se_rho_null": se_rho,
                "best_correlate": ranked[0][0],
                "best_correlate_rho": ranked[0][1],
                "ranking": [name for name, _ in ranked],
            }

    return results


def compute_radial_r_squared(
    mono_data: dict, radial_data: dict
) -> dict[int, dict[int, dict]]:
    """Pearson R² of (∂L/∂s_t / α_t) regressed on (r_t - R).

    Under A7: ∂L/∂s_t / α_t = -β(r_t - R) + noise
    R² measures how much gradient variance IS radial.
    """
    results: dict[int, dict[int, dict]] = {}

    for layer_idx in sorted(mono_data.keys()):
        results[layer_idx] = {}

        for head_idx in sorted(mono_data[layer_idx].keys()):
            mono = mono_data[layer_idx][head_idx]

            if mono.get("excluded", False):
                results[layer_idx][head_idx] = {"excluded": True}
                continue

            grad_over_alpha = mono["grad_over_alpha"]

            if layer_idx not in radial_data or head_idx not in radial_data[layer_idx]:
                results[layer_idx][head_idx] = {"excluded": True}
                continue

            r_minus_R = radial_data[layer_idx][head_idx]["r_minus_R"]

            r_sq = pearson_r_squared(r_minus_R, grad_over_alpha)
            slope = ols_slope(r_minus_R, grad_over_alpha)

            results[layer_idx][head_idx] = {
                "excluded": False,
                "r_squared": r_sq,
                "ols_slope": slope,
                "beta": -slope,
            }

    return results


# =====================================================================
# Single model runner
# =====================================================================


def run_single_model(
    model_name: str, model_path: str, probes: list[str] | None = None
) -> dict:
    """Run diagnostic on one model across all probes."""
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

        # Reuse A7 pipeline: baseline → gradients → radial → per-head data
        logger.info("  Collecting baseline...")
        baseline = collect_baseline(model, tokenizer, text, backend, mx)
        seq_len = len(baseline["token_ids"])
        logger.info("  Baseline loss: %.6f, seq_len: %d", baseline["baseline_loss"], seq_len)

        logger.info("  Computing score gradients...")
        gradients = compute_score_gradients(model, tokenizer, text, baseline, backend, mx)

        logger.info("  Computing radial projections...")
        radial_data = compute_radial_projections(
            model, tokenizer, text, baseline, baseline["attn_weights"], backend, mx
        )

        logger.info("  Collecting per-head gradient data...")
        mono_data = collect_per_head_gradient_data(
            baseline["attn_weights"], gradients, baseline, mx
        )

        # Diagnostic measurements
        logger.info("  Computing R²(radial)...")
        r_sq_data = compute_radial_r_squared(mono_data, radial_data)

        logger.info("  Computing alternative correlates...")
        alt_corr_data = compute_alternative_correlates(mono_data, radial_data, seq_len)

        logger.info("  Computing gradient concentration...")
        concentration_data: dict[int, dict[int, dict]] = {}
        for layer_idx in sorted(gradients.keys()):
            concentration_data[layer_idx] = {}
            for head_idx in sorted(gradients[layer_idx].keys()):
                dLds = gradients[layer_idx][head_idx]
                concentration_data[layer_idx][head_idx] = compute_gradient_concentration(dLds)

        # Collect per-head results
        per_head: dict[str, dict[str, dict]] = {}
        for layer_idx in sorted(mono_data.keys()):
            l_key = str(layer_idx)
            per_head[l_key] = {}
            for head_idx in sorted(mono_data[layer_idx].keys()):
                h_key = str(head_idx)
                mono = mono_data[layer_idx][head_idx]

                if mono.get("excluded", False):
                    per_head[l_key][h_key] = {
                        "excluded": True,
                        "reason": mono.get("reason", ""),
                    }
                    continue

                r_sq_entry = r_sq_data.get(layer_idx, {}).get(head_idx, {})
                alt_entry = alt_corr_data.get(layer_idx, {}).get(head_idx, {})
                conc_entry = concentration_data.get(layer_idx, {}).get(head_idx, {})

                if r_sq_entry.get("excluded") or alt_entry.get("excluded"):
                    per_head[l_key][h_key] = {"excluded": True, "reason": "no radial data"}
                    continue

                per_head[l_key][h_key] = {
                    "excluded": False,
                    "r_squared_radial": r_sq_entry.get("r_squared"),
                    "ols_slope": r_sq_entry.get("ols_slope"),
                    "beta": r_sq_entry.get("beta"),
                    "best_correlate": alt_entry.get("best_correlate"),
                    "best_correlate_rho": alt_entry.get("best_correlate_rho"),
                    "correlates": alt_entry.get("correlates"),
                    "correlate_ranking": alt_entry.get("ranking"),
                    "se_rho_null": alt_entry.get("se_rho_null"),
                    "k_eff": conc_entry.get("k_eff"),
                    "k_eff_over_T": conc_entry.get("k_eff_over_T"),
                }

        # Aggregate statistics across non-excluded heads
        all_r_sq = []
        best_correlate_counts: dict[str, int] = {}
        all_k_eff_over_T = []
        all_correlates: dict[str, list[float]] = {
            "position": [], "alpha": [], "r_t": [], "abs_r_dev": [],
        }

        for l_key in per_head:
            for h_key in per_head[l_key]:
                entry = per_head[l_key][h_key]
                if entry.get("excluded"):
                    continue

                r_sq_val = entry.get("r_squared_radial")
                if r_sq_val is not None and not math.isnan(r_sq_val):
                    all_r_sq.append(r_sq_val)

                best = entry.get("best_correlate")
                if best:
                    best_correlate_counts[best] = best_correlate_counts.get(best, 0) + 1

                k_val = entry.get("k_eff_over_T")
                if k_val is not None and not math.isnan(k_val):
                    all_k_eff_over_T.append(k_val)

                corrs = entry.get("correlates", {})
                for name in all_correlates:
                    val = corrs.get(name)
                    if val is not None and not math.isnan(val):
                        all_correlates[name].append(val)

        def _safe_stats(vals: list[float]) -> dict:
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

        aggregate = {
            "r_squared_radial": _safe_stats(all_r_sq),
            "k_eff_over_T": _safe_stats(all_k_eff_over_T),
            "best_correlate_counts": best_correlate_counts,
            "mean_abs_rho_by_correlate": {
                name: (sum(abs(v) for v in vals) / len(vals)) if vals else None
                for name, vals in all_correlates.items()
            },
        }

        # Log summary
        r_sq_stats = aggregate["r_squared_radial"]
        logger.info(
            "  Probe %d R²(radial): mean=%.4f, median=%.4f, range=[%.4f, %.4f] (n=%d)",
            probe_idx + 1,
            r_sq_stats["mean"] or 0, r_sq_stats["median"] or 0,
            r_sq_stats["min"] or 0, r_sq_stats["max"] or 0,
            r_sq_stats["n"],
        )
        logger.info("  Best correlate counts: %s", best_correlate_counts)

        probe_results.append({
            "probe_text": text,
            "seq_len": seq_len,
            "baseline_loss": baseline["baseline_loss"],
            "per_head": per_head,
            "aggregate": aggregate,
        })

    # Cross-probe aggregate
    all_r_sq_cross = []
    best_counts_cross: dict[str, int] = {}
    all_k_eff_cross = []
    all_corr_cross: dict[str, list[float]] = {
        "position": [], "alpha": [], "r_t": [], "abs_r_dev": [],
    }

    for pr in probe_results:
        agg = pr["aggregate"]
        # Collect raw values from per_head (not from stats)
        for l_key in pr["per_head"]:
            for h_key in pr["per_head"][l_key]:
                entry = pr["per_head"][l_key][h_key]
                if entry.get("excluded"):
                    continue
                r_sq_val = entry.get("r_squared_radial")
                if r_sq_val is not None and not math.isnan(r_sq_val):
                    all_r_sq_cross.append(r_sq_val)
                k_val = entry.get("k_eff_over_T")
                if k_val is not None and not math.isnan(k_val):
                    all_k_eff_cross.append(k_val)
                best = entry.get("best_correlate")
                if best:
                    best_counts_cross[best] = best_counts_cross.get(best, 0) + 1
                corrs = entry.get("correlates", {})
                for name in all_corr_cross:
                    val = corrs.get(name)
                    if val is not None and not math.isnan(val):
                        all_corr_cross[name].append(val)

    def _safe_stats_2(vals: list[float]) -> dict:
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

    cross_probe_aggregate = {
        "r_squared_radial": _safe_stats_2(all_r_sq_cross),
        "k_eff_over_T": _safe_stats_2(all_k_eff_cross),
        "best_correlate_counts": best_counts_cross,
        "mean_abs_rho_by_correlate": {
            name: (sum(abs(v) for v in vals) / len(vals)) if vals else None
            for name, vals in all_corr_cross.items()
        },
    }

    return {
        "model": model_name,
        "model_path": model_path,
        "probes": probe_results,
        "cross_probe_aggregate": cross_probe_aggregate,
    }


# =====================================================================
# Text summary writer
# =====================================================================


def write_text_summary(txt_path: Path, run_doc: dict) -> None:
    """Write human-readable diagnostic summary."""
    lines = [
        "A7 Gradient Structure Diagnostic",
        f"Run ID: {run_doc['run_id']}",
        f"Timestamp: {run_doc['timestamp']}",
        "",
        "Purpose: diagnose WHAT the CE gradient correlates with,",
        "now that A7 (radial dominance) is falsified.",
        "",
    ]

    for model_doc in run_doc["models"]:
        lines.append(f"{'='*70}")
        lines.append(f"Model: {model_doc['model']}")
        lines.append(f"{'='*70}")

        agg = model_doc["cross_probe_aggregate"]
        r_sq = agg["r_squared_radial"]
        lines.append("")
        lines.append(f"R²(radial) across all probes:")
        lines.append(
            f"  mean={r_sq['mean']:.4f}, median={r_sq['median']:.4f}, "
            f"range=[{r_sq['min']:.4f}, {r_sq['max']:.4f}] (n={r_sq['n']})"
            if r_sq["mean"] is not None else "  No data"
        )

        lines.append("")
        lines.append("Best correlate counts (which predictor wins most often):")
        for name, count in sorted(
            agg["best_correlate_counts"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"  {name}: {count}")

        lines.append("")
        lines.append("Mean |ρ| by correlate (average absolute Spearman):")
        for name, val in sorted(
            agg["mean_abs_rho_by_correlate"].items(),
            key=lambda kv: -(kv[1] or 0),
        ):
            lines.append(f"  {name}: {val:.4f}" if val is not None else f"  {name}: N/A")

        k_eff = agg["k_eff_over_T"]
        lines.append("")
        lines.append(f"Gradient concentration (k_eff/T, 1.0=diffuse, 1/T=concentrated):")
        lines.append(
            f"  mean={k_eff['mean']:.4f}, median={k_eff['median']:.4f}, "
            f"range=[{k_eff['min']:.4f}, {k_eff['max']:.4f}] (n={k_eff['n']})"
            if k_eff["mean"] is not None else "  No data"
        )

        # Per-probe detail
        for probe_idx, pr in enumerate(model_doc["probes"]):
            lines.append("")
            lines.append(f"--- Probe {probe_idx + 1}: {pr['probe_text']!r} ---")
            lines.append(f"  seq_len={pr['seq_len']}, loss={pr['baseline_loss']:.6f}")

            for l_key in sorted(pr["per_head"].keys(), key=int):
                for h_key in sorted(pr["per_head"][l_key].keys(), key=int):
                    entry = pr["per_head"][l_key][h_key]
                    if entry.get("excluded"):
                        lines.append(f"  L{l_key} H{h_key}: EXCLUDED")
                        continue

                    r_sq_val = entry.get("r_squared_radial", 0)
                    best = entry.get("best_correlate", "?")
                    best_rho = entry.get("best_correlate_rho", 0)
                    k_eff_val = entry.get("k_eff_over_T", 0)
                    lines.append(
                        f"  L{l_key} H{h_key}: R²={r_sq_val:.4f}, "
                        f"best={best}(ρ={best_rho:+.3f}), "
                        f"k_eff/T={k_eff_val:.3f}"
                    )

        lines.append("")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A7 Root Cause Diagnosis — Gradient Structure Analysis"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model keys from registry: {sorted(DIAGNOSTIC_MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        default="results/a7_validation/diagnostic",
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

    logger.info("A7 Gradient Structure Diagnostic")
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
        "diagnostic_version": "a7_diagnostic_v1",
        "query_position": _QUERY_POS,
        "models": models_data,
    }

    json_path = out_dir / "a7_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    txt_path = out_dir / "a7_diagnostic.txt"
    write_text_summary(txt_path, run_doc)
    logger.info("Wrote %s", txt_path)


if __name__ == "__main__":
    main()
