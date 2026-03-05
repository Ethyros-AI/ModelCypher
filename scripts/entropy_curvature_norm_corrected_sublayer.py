#!/usr/bin/env python3
"""B6.1: Norm-Corrected Sublayer Decomposition — Universal Centroid Test.

Tests whether the D3.1 sign reversal across architectures is a confound from
mixing three independent effects:

    1. Core operator centroid averaging (hypothesis: always negative)
    2. Norm-entropy coupling (architecture-dependent sign)
    3. MLP entropy-magnitude coupling (architecture-dependent)

The key decomposition: δ_total = δ_core + δ_mlp.
E_total = ||δ_total||². E_core = ||δ_core||². E_mlp = ||δ_mlp||².

By computing r² = ||δ||²/||h_in||² we factor out the norm contribution
and measure the GEOMETRIC effect only.

Computes per model:
    1. r(H_logit_norm, log(E_core/||h_in||²) | depth)  — norm-corrected core
    2. r(H_logit_norm, log(E_mlp/||h_in||²) | depth)   — norm-corrected MLP
    3. r(H_logit_norm, log(||h_in||²) | depth)          — norm-entropy coupling
    4. r(H_logit_norm, cos_alpha | depth)                — radial alignment

Falsifiers:
    F1: Universal centroid — norm-corrected core negative for ALL 10 models
    F2: Norm explains D3.1 — reversed models have positive norm-entropy
    F3: Sign reconstruction — three-way predicts raw D3.1 sign for 10/10
    F4: Radial alignment — cos_alpha same sign as norm-entropy for all

**Pure reanalysis script.** Reads existing operator_split.json files (no model
loading). Requires E_core, E_mlp, h_in_norm_sq, H_logit_norm, cos_alpha.

Usage:
    poetry run python scripts/entropy_curvature_norm_corrected_sublayer.py
    poetry run python scripts/entropy_curvature_norm_corrected_sublayer.py --models LFM2-700M Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_BASE = Path("results/entropy_curvature_operator_split")
OUTPUT_BASE = Path("results/entropy_curvature_norm_corrected_sublayer")

# IEEE 754 bf16 machine epsilon: 2^-7 = 0.0078125
EPS_BF16 = 2**-7

# Permutation count (matches existing convention across scripts)
N_PERMUTATIONS = 500


# ---------------------------------------------------------------------------
# Statistical utilities (from entropy_curvature_three_component.py)
# ---------------------------------------------------------------------------


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
    denom = np.sum(r_centered**2)
    if abs(denom) < 1e-15:
        return float(n), 0.0
    rho_1 = float(np.sum(r_centered[:-1] * r_centered[1:]) / denom)
    rho_1_clamped = max(-0.99, min(0.99, rho_1))
    n_eff = n * (1 - rho_1_clamped) / (1 + rho_1_clamped)
    n_eff = max(4.0, min(float(n), n_eff))
    return n_eff, rho_1


def _minimum_detectable_effect(n_eff: float, n_controls: int = 1) -> float:
    """Fisher-SE MDE: smallest |r| with SNR >= 1 at given effective df."""
    df = n_eff - n_controls - 2
    if df <= 0:
        return 1.0
    return float(np.tanh(1.0 / np.sqrt(df)))


def safe_spearman(x, y):
    """Spearman correlation with NaN handling."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    try:
        r, p = sp_stats.spearmanr(x_arr[mask], y_arr[mask])
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def _permutation_test(
    x_resid: np.ndarray,
    y_resid: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
) -> dict:
    """Depth-stratified permutation test for Spearman r.

    Permutes x_resid (breaking x-y association while preserving marginal
    distributions) and computes the distribution of |Spearman r| under the
    null. Reports exceedance fraction (empirical p-value).
    """
    n = len(x_resid)
    if n < 4:
        return {
            "observed_abs_r": float("nan"),
            "exceedance_fraction": float("nan"),
            "null_mean": float("nan"),
            "null_max": float("nan"),
            "n_permutations": n_permutations,
        }

    obs_r, _ = sp_stats.spearmanr(x_resid, y_resid)
    obs_abs_r = abs(float(obs_r))

    rng = np.random.default_rng(seed=seed)
    null_abs_r = np.empty(n_permutations)

    for i in range(n_permutations):
        x_perm = rng.permutation(x_resid)
        r_perm, _ = sp_stats.spearmanr(x_perm, y_resid)
        null_abs_r[i] = abs(float(r_perm))

    exceedance = float(np.mean(null_abs_r >= obs_abs_r))

    return {
        "observed_abs_r": obs_abs_r,
        "exceedance_fraction": exceedance,
        "null_mean": float(np.mean(null_abs_r)),
        "null_max": float(np.max(null_abs_r)),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


def load_model_data(model_dir: Path) -> dict | None:
    """Load operator_split.json and validate required fields."""
    json_path = model_dir / "operator_split.json"
    if not json_path.exists():
        logger.warning("No operator_split.json in %s", model_dir)
        return None

    with open(json_path) as f:
        data = json.load(f)

    measurements = data.get("measurements", [])
    if not measurements:
        logger.warning("No measurements in %s", model_dir.name)
        return None

    # Check required fields for norm-corrected sublayer analysis
    required = [
        "E_core", "E_mlp", "h_in_norm_sq", "H_logit_norm",
        "depth_fraction", "cos_alpha", "E_total",
    ]
    missing = [f for f in required if f not in measurements[0]]
    if missing:
        logger.warning(
            "%s: missing fields %s — needs re-collection", model_dir.name, missing
        )
        return None

    # Check for None values in required fields
    none_counts = {}
    for f in required:
        n_none = sum(1 for m in measurements if m.get(f) is None)
        if n_none > 0:
            none_counts[f] = n_none
    if none_counts:
        logger.warning(
            "%s: None values in required fields: %s", model_dir.name, none_counts
        )
        return None

    return data


# ---------------------------------------------------------------------------
# Norm-corrected sublayer analysis
# ---------------------------------------------------------------------------


def analyze_model(data: dict) -> dict:
    """Run norm-corrected sublayer decomposition for a single model.

    Decomposes raw D3.1 coupling into three independent effects:
        1. r(H, log(E_core/||h||²) | depth)  — geometric centroid (core)
        2. r(H, log(E_mlp/||h||²) | depth)   — geometric centroid (MLP)
        3. r(H, log(||h||²) | depth)          — norm-entropy coupling
    """
    model_name = data["model_name"]
    architecture = data.get("architecture", "unknown")
    measurements = data["measurements"]
    n_layers = len(measurements)

    # Extract per-layer arrays
    depth = np.array([m["depth_fraction"] for m in measurements])
    E_core = np.array([m["E_core"] for m in measurements])
    E_mlp = np.array([m["E_mlp"] for m in measurements])
    E_total = np.array([m["E_total"] for m in measurements])
    h_in_norm_sq = np.array([m["h_in_norm_sq"] for m in measurements])
    H_logit_norm = np.array([m["H_logit_norm"] for m in measurements])
    cos_alpha = np.array([m["cos_alpha"] for m in measurements])

    # Clip to prevent log(0) — use bf16 epsilon as floor
    E_core_clipped = np.maximum(E_core, 1e-30)
    E_mlp_clipped = np.maximum(E_mlp, 1e-30)
    E_total_clipped = np.maximum(E_total, 1e-30)
    h_in_clipped = np.maximum(h_in_norm_sq, 1e-30)

    # --- Norm-corrected sub-components (log space) ---
    # log(E_core / ||h||²) = log(||δ_core||² / ||h||²) = 2·log(||δ_core|| / ||h||)
    Y_core_normed = np.log(E_core_clipped) - np.log(h_in_clipped)
    # log(E_mlp / ||h||²) = log(||δ_mlp||² / ||h||²) = 2·log(||δ_mlp|| / ||h||)
    Y_mlp_normed = np.log(E_mlp_clipped) - np.log(h_in_clipped)
    # log(||h||²) — norm component
    Y_h = np.log(h_in_clipped)
    # Raw D3.1: log(E_total)
    Y_total = np.log(E_total_clipped)
    # cos_alpha (already in data, no transform needed)

    # --- Depth-residualize everything ---
    H_resid = _residualize_ols(H_logit_norm, depth)
    Y_core_normed_resid = _residualize_ols(Y_core_normed, depth)
    Y_mlp_normed_resid = _residualize_ols(Y_mlp_normed, depth)
    Y_h_resid = _residualize_ols(Y_h, depth)
    Y_total_resid = _residualize_ols(Y_total, depth)
    cos_alpha_resid = _residualize_ols(cos_alpha, depth)

    # --- Effective df (Bretherton AR(1)) ---
    n_eff_H, rho1_H = _effective_df(H_resid)
    mde = _minimum_detectable_effect(n_eff_H, n_controls=1)

    # --- Spearman correlations: r(H_logit_norm, Y_X | depth) ---
    components = {
        "core_normed": {
            "label": "log(E_core/||h||²)",
            "prediction": "F1: < 0 (universal centroid averaging)",
            "resid": Y_core_normed_resid,
        },
        "mlp_normed": {
            "label": "log(E_mlp/||h||²)",
            "prediction": "architecture-dependent",
            "resid": Y_mlp_normed_resid,
        },
        "norm": {
            "label": "log(||h_in||²)",
            "prediction": "F2: sign determines D3.1 reversal",
            "resid": Y_h_resid,
        },
        "cos_alpha": {
            "label": "cos(α)",
            "prediction": "F4: same sign as norm coupling",
            "resid": cos_alpha_resid,
        },
        "raw_total": {
            "label": "log(E_total) [raw D3.1]",
            "prediction": "reference — original D3.1 measurement",
            "resid": Y_total_resid,
        },
    }

    correlations = {}
    for comp_name, comp_data in components.items():
        r, p = safe_spearman(H_resid, comp_data["resid"])
        perm = _permutation_test(H_resid, comp_data["resid"])
        correlations[comp_name] = {
            "label": comp_data["label"],
            "prediction": comp_data["prediction"],
            "spearman_r": r,
            "spearman_p": p,
            "abs_r": abs(r) if not math.isnan(r) else float("nan"),
            "resolvable": abs(r) > mde if not math.isnan(r) else False,
            "sign": (
                "positive" if r > 0 else ("negative" if r < 0 else "zero")
                if not math.isnan(r) else "nan"
            ),
            "permutation": perm,
        }

    # --- Falsifier checks ---
    r_core_normed = correlations["core_normed"]["spearman_r"]
    r_mlp_normed = correlations["mlp_normed"]["spearman_r"]
    r_norm = correlations["norm"]["spearman_r"]
    r_cos_alpha = correlations["cos_alpha"]["spearman_r"]
    r_raw_total = correlations["raw_total"]["spearman_r"]

    # F1: Universal centroid — norm-corrected core negative
    f1_pass = r_core_normed < 0 if not math.isnan(r_core_normed) else None

    # F2: Norm explains D3.1 reversal
    # D3.1 is "reversed" when raw_total > 0 (should be negative by D3.1 prediction)
    raw_d3_1_sign = (
        "positive" if r_raw_total > 0 else "negative"
        if not math.isnan(r_raw_total) else "nan"
    )
    raw_d3_1_reversed = r_raw_total > 0 if not math.isnan(r_raw_total) else None

    # F2 check: if D3.1 is reversed, is norm coupling positive and dominant enough?
    if raw_d3_1_reversed is True:
        # Reversed model: norm coupling should be positive to explain the flip
        f2_pass = r_norm > 0 if not math.isnan(r_norm) else None
    elif raw_d3_1_reversed is False:
        # Non-reversed model: F2 is vacuously true (no reversal to explain)
        f2_pass = True
    else:
        f2_pass = None

    # F3: Sign reconstruction
    # Can the three-way decomposition predict the raw D3.1 sign?
    # E_total = E_core + E_mlp + E_mix (from operator split)
    # log(E_total) ≈ log(E_core) + log(1 + E_mlp/E_core + E_mix/E_core)
    # But more directly: r(H, log(E_total)) should have the same sign as
    # the weighted combination of the three effects.
    # Simple check: if core_normed < 0 and raw_total > 0, then norm or MLP
    # must be positive and large enough to overwhelm core.
    if not (math.isnan(r_core_normed) or math.isnan(r_raw_total)):
        if r_raw_total > 0:
            # Reversed: at least one of {norm, mlp_normed} must be positive
            has_positive_driver = (
                (not math.isnan(r_norm) and r_norm > 0)
                or (not math.isnan(r_mlp_normed) and r_mlp_normed > 0)
            )
            f3_pass = has_positive_driver
        else:
            # Not reversed: core_normed negative is consistent (core dominates)
            f3_pass = r_core_normed < 0
    else:
        f3_pass = None

    # F4: Radial alignment — cos_alpha same sign as norm coupling
    if not (math.isnan(r_cos_alpha) or math.isnan(r_norm)):
        f4_pass = (r_cos_alpha > 0) == (r_norm > 0)
    else:
        f4_pass = None

    falsifiers = {
        "F1_universal_centroid": {
            "prediction": "r(H, log(E_core/||h||²) | depth) < 0",
            "observed_r": r_core_normed,
            "pass": f1_pass,
        },
        "F2_norm_explains_reversal": {
            "prediction": "if D3.1 reversed, r(H, log(||h||²)) > 0",
            "raw_d3_1_sign": raw_d3_1_sign,
            "raw_d3_1_reversed": raw_d3_1_reversed,
            "norm_r": r_norm,
            "pass": f2_pass,
        },
        "F3_sign_reconstruction": {
            "prediction": "three-way predicts raw D3.1 sign",
            "raw_total_r": r_raw_total,
            "core_normed_r": r_core_normed,
            "norm_r": r_norm,
            "mlp_normed_r": r_mlp_normed,
            "pass": f3_pass,
        },
        "F4_radial_alignment": {
            "prediction": "r(H, cos_alpha) same sign as r(H, log(||h||²))",
            "cos_alpha_r": r_cos_alpha,
            "norm_r": r_norm,
            "pass": f4_pass,
        },
    }

    # --- Raw statistics ---
    raw_stats = {
        "E_core_range": [float(np.min(E_core)), float(np.max(E_core))],
        "E_mlp_range": [float(np.min(E_mlp)), float(np.max(E_mlp))],
        "E_total_range": [float(np.min(E_total)), float(np.max(E_total))],
        "h_in_norm_sq_range": [float(np.min(h_in_norm_sq)), float(np.max(h_in_norm_sq))],
        "H_logit_norm_range": [float(np.min(H_logit_norm)), float(np.max(H_logit_norm))],
        "cos_alpha_range": [float(np.min(cos_alpha)), float(np.max(cos_alpha))],
        "Y_core_normed_range": [float(np.min(Y_core_normed)), float(np.max(Y_core_normed))],
        "Y_mlp_normed_range": [float(np.min(Y_mlp_normed)), float(np.max(Y_mlp_normed))],
        "Y_h_range": [float(np.min(Y_h)), float(np.max(Y_h))],
    }

    result = {
        "model_name": model_name,
        "architecture": architecture,
        "num_layers": n_layers,
        "gqa_ratio": data.get("gqa_ratio"),
        "detection_floor": {
            "n_eff": n_eff_H,
            "rho1_H": rho1_H,
            "mde": mde,
            "n_layers": n_layers,
        },
        "correlations": correlations,
        "falsifiers": falsifiers,
        "raw_stats": raw_stats,
    }

    return result


# ---------------------------------------------------------------------------
# Cross-model summary and falsifier evaluation
# ---------------------------------------------------------------------------


def build_cross_model_summary(results: list[dict]) -> dict:
    """Build cross-model summary with falsifier evaluation."""
    summary_rows = []

    for r in results:
        row = {
            "model": r["model_name"],
            "architecture": r["architecture"],
            "n_layers": r["num_layers"],
            "gqa_ratio": r.get("gqa_ratio"),
            "mde": r["detection_floor"]["mde"],
            "n_eff": r["detection_floor"]["n_eff"],
        }
        for comp in ["core_normed", "mlp_normed", "norm", "cos_alpha", "raw_total"]:
            c = r["correlations"][comp]
            row[f"r_{comp}"] = c["spearman_r"]
            row[f"p_{comp}"] = c["spearman_p"]
            row[f"resolvable_{comp}"] = c["resolvable"]

        for f_name, f_data in r["falsifiers"].items():
            row[f"{f_name}_pass"] = f_data["pass"]

        summary_rows.append(row)

    # --- F1: Universal centroid (ALL models must be negative) ---
    f1_results = [r["falsifiers"]["F1_universal_centroid"]["pass"] for r in results]
    f1_testable = [p for p in f1_results if p is not None]
    f1_all_pass = all(f1_testable) if f1_testable else False
    f1_failing_models = [
        r["model_name"]
        for r in results
        if r["falsifiers"]["F1_universal_centroid"]["pass"] is False
    ]

    # --- F2: Norm explains reversal ---
    f2_results = [r["falsifiers"]["F2_norm_explains_reversal"]["pass"] for r in results]
    f2_testable = [p for p in f2_results if p is not None]
    f2_all_pass = all(f2_testable) if f2_testable else False

    # --- F3: Sign reconstruction ---
    f3_results = [r["falsifiers"]["F3_sign_reconstruction"]["pass"] for r in results]
    f3_testable = [p for p in f3_results if p is not None]
    f3_all_pass = all(f3_testable) if f3_testable else False

    # --- F4: Radial alignment ---
    f4_results = [r["falsifiers"]["F4_radial_alignment"]["pass"] for r in results]
    f4_testable = [p for p in f4_results if p is not None]
    f4_all_pass = all(f4_testable) if f4_testable else False

    # Count D3.1-reversed models
    reversed_models = [
        r["model_name"]
        for r in results
        if r["falsifiers"]["F2_norm_explains_reversal"].get("raw_d3_1_reversed") is True
    ]

    # Architecture grouping
    arch_groups = {}
    for r in results:
        arch = r["architecture"]
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append({
            "model": r["model_name"],
            "r_core_normed": r["correlations"]["core_normed"]["spearman_r"],
            "r_mlp_normed": r["correlations"]["mlp_normed"]["spearman_r"],
            "r_norm": r["correlations"]["norm"]["spearman_r"],
            "r_raw_total": r["correlations"]["raw_total"]["spearman_r"],
        })

    falsifier_summary = {
        "F1_universal_centroid": {
            "pass": f1_all_pass,
            "pass_count": sum(1 for p in f1_testable if p),
            "total": len(f1_testable),
            "failing_models": f1_failing_models,
            "verdict": (
                "PASS — centroid averaging is universal"
                if f1_all_pass
                else f"FAIL — {len(f1_failing_models)} model(s) have positive norm-corrected core coupling"
            ),
        },
        "F2_norm_explains_reversal": {
            "pass": f2_all_pass,
            "pass_count": sum(1 for p in f2_testable if p),
            "total": len(f2_testable),
            "reversed_models": reversed_models,
            "verdict": (
                "PASS — norm coupling explains all D3.1 reversals"
                if f2_all_pass
                else "FAIL — some reversed models lack positive norm coupling"
            ),
        },
        "F3_sign_reconstruction": {
            "pass": f3_all_pass,
            "pass_count": sum(1 for p in f3_testable if p),
            "total": len(f3_testable),
            "verdict": (
                "PASS — three-way decomposition predicts all D3.1 signs"
                if f3_all_pass
                else "FAIL — incomplete decomposition"
            ),
        },
        "F4_radial_alignment": {
            "pass": f4_all_pass,
            "pass_count": sum(1 for p in f4_testable if p),
            "total": len(f4_testable),
            "verdict": (
                "PASS — norm growth driven by radial projection"
                if f4_all_pass
                else "FAIL — norm growth not fully explained by radial projection"
            ),
        },
    }

    return {
        "n_models": len(results),
        "models": [r["model_name"] for r in results],
        "summary_rows": summary_rows,
        "reversed_models": reversed_models,
        "architecture_groups": arch_groups,
        "falsifier_summary": falsifier_summary,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], cross_summary: dict) -> None:
    """Print human-readable summary table."""
    logger.info("\n%s", "=" * 90)
    logger.info("B6.1: NORM-CORRECTED SUBLAYER DECOMPOSITION — UNIVERSAL CENTROID TEST")
    logger.info("%s", "=" * 90)

    # Per-model table
    header = (
        f"{'Model':<18} {'L':>3} {'MDE':>5} "
        f"{'r(core/h²)':>10} {'r(mlp/h²)':>10} {'r(h²)':>10} "
        f"{'r(cos_α)':>10} {'r(E_tot)':>10} "
        f"{'F1':>4}"
    )
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for r in results:
        c = r["correlations"]
        f1 = r["falsifiers"]["F1_universal_centroid"]["pass"]

        def _fmt_r(comp_name):
            val = c[comp_name]["spearman_r"]
            resolvable = c[comp_name]["resolvable"]
            if math.isnan(val):
                return "      nan"
            marker = "*" if resolvable else " "
            return f"{val:+.4f}{marker}"

        f1_str = "PASS" if f1 else ("FAIL" if f1 is False else "  - ")

        logger.info(
            "%-18s %3d %5.3f %10s %10s %10s %10s %10s %4s",
            r["model_name"],
            r["num_layers"],
            r["detection_floor"]["mde"],
            _fmt_r("core_normed"),
            _fmt_r("mlp_normed"),
            _fmt_r("norm"),
            _fmt_r("cos_alpha"),
            _fmt_r("raw_total"),
            f1_str,
        )

    logger.info("\n* = resolvable (|r| > MDE)")

    # D3.1 reversal analysis
    reversed_models = cross_summary["reversed_models"]
    logger.info(
        "\nD3.1-reversed models (%d): %s",
        len(reversed_models),
        reversed_models if reversed_models else "none",
    )

    # Falsifier summary
    logger.info("\n%s", "-" * 60)
    logger.info("FALSIFIER OUTCOMES:")
    logger.info("%s", "-" * 60)
    for f_name, f_data in cross_summary["falsifier_summary"].items():
        logger.info(
            "  %s: %d/%d pass — %s",
            f_name,
            f_data["pass_count"],
            f_data["total"],
            f_data["verdict"],
        )

    # Architecture-group summary
    logger.info("\n%s", "-" * 60)
    logger.info("ARCHITECTURE GROUPS:")
    logger.info("%s", "-" * 60)
    for arch, models in cross_summary["architecture_groups"].items():
        core_rs = [m["r_core_normed"] for m in models if not math.isnan(m["r_core_normed"])]
        norm_rs = [m["r_norm"] for m in models if not math.isnan(m["r_norm"])]
        logger.info(
            "  %s (%d models): core_normed=[%s], norm=[%s]",
            arch,
            len(models),
            ", ".join(f"{r:+.3f}" for r in core_rs),
            ", ".join(f"{r:+.3f}" for r in norm_rs),
        )

    # Permutation test details for F1-critical correlations
    logger.info("\n%s", "-" * 60)
    logger.info("PERMUTATION TEST DETAILS (core_normed — F1 critical):")
    logger.info("%s", "-" * 60)
    for r in results:
        perm = r["correlations"]["core_normed"]["permutation"]
        logger.info(
            "  %s: |r|=%.4f, exceedance=%.3f, null_max=%.4f",
            r["model_name"],
            perm["observed_abs_r"],
            perm["exceedance_fraction"],
            perm["null_max"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="B6.1: Norm-Corrected Sublayer Decomposition"
    )
    parser.add_argument(
        "--input", default=str(INPUT_BASE),
        help="Input directory with operator_split results",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_BASE),
        help="Output directory for norm-corrected sublayer results",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to analyze (default: all available)",
    )
    args = parser.parse_args()

    input_base = Path(args.input)
    output_base = Path(args.output)

    # Discover models
    if args.models:
        model_dirs = [input_base / m for m in args.models]
    else:
        model_dirs = sorted([
            d for d in input_base.iterdir()
            if d.is_dir() and (d / "operator_split.json").exists()
        ])

    logger.info(
        "B6.1 Norm-Corrected Sublayer Analysis: scanning %d model directories",
        len(model_dirs),
    )

    # Load and validate
    results = []
    skipped = []
    for model_dir in model_dirs:
        data = load_model_data(model_dir)
        if data is None:
            skipped.append(model_dir.name)
            continue
        result = analyze_model(data)
        results.append(result)

    if skipped:
        logger.warning("Skipped %d models (missing fields): %s", len(skipped), skipped)

    if not results:
        logger.error(
            "No models with complete data. "
            "Run entropy_curvature_operator_split.py first."
        )
        return

    # Cross-model summary
    cross_summary = build_cross_model_summary(results)

    # Print summary
    print_summary(results, cross_summary)

    # Save results
    output_base.mkdir(parents=True, exist_ok=True)
    for r in results:
        model_dir = output_base / r["model_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "norm_corrected_sublayer.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        logger.info("Saved: %s", model_dir / "norm_corrected_sublayer.json")

    with open(output_base / "cross_model_summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2, default=str)
    logger.info("Saved: %s", output_base / "cross_model_summary.json")


if __name__ == "__main__":
    main()
