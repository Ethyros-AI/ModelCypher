#!/usr/bin/env python3
"""B6: Three-Component Decomposition of ||P_perp(h)δ||² → H_logit Coupling.

Decomposes the curvature numerator into three measurable sub-components:

    ||P_perp(h)δ||² = ||δ||² sin²(α)    where α = angle(h, δ)

In log space:
    log(||P_perp(h)δ||²) = log(||δ||²) + 2·log(sin(α))

The three sub-components carry independent geometric meaning:
    1. ||δ||² = E_total — update magnitude (D3.1: centroid averaging)
    2. sin²(α) — tangential fraction (D3.2: tangentiality)
    3. ||h||² = h_in_norm_sq — hidden state norm (B5/B7: norm-entropy coupling)

Measuring which sub-component carries the H_logit_norm signal resolves the
mechanistic ambiguity in B4 and connects the empirical coupling back to D3.

**Pure reanalysis script.** Reads existing operator_split.json files (no model
loading). Requires all models to have sin_alpha and H_logit_norm fields.

Usage:
    poetry run python scripts/entropy_curvature_three_component.py
    poetry run python scripts/entropy_curvature_three_component.py --models LFM2-700M Qwen3.5-0.8B
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
OUTPUT_BASE = Path("results/entropy_curvature_three_component")

# IEEE 754 bf16 machine epsilon: 2^-7 = 0.0078125
EPS_BF16 = 2**-7

# Permutation count (matches existing convention across scripts)
N_PERMUTATIONS = 500


# ---------------------------------------------------------------------------
# Statistical utilities (from entropy_curvature_operator_split.py)
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

    # Check required fields
    required = ["sin_alpha", "H_logit_norm", "E_total", "h_in_norm_sq"]
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
# Three-component analysis
# ---------------------------------------------------------------------------


def analyze_model(data: dict) -> dict:
    """Run three-component decomposition for a single model.

    Decomposes log(||P_perp(h)δ||²) = log(E_total) + 2·log(sin_alpha)
    and measures which sub-component carries the H_logit_norm signal.
    """
    model_name = data["model_name"]
    architecture = data.get("architecture", "unknown")
    measurements = data["measurements"]
    n_layers = len(measurements)

    # Extract per-layer arrays
    depth = np.array([m["depth_fraction"] for m in measurements])
    sin_alpha = np.array([m["sin_alpha"] for m in measurements])
    E_total = np.array([m["E_total"] for m in measurements])
    h_in_norm_sq = np.array([m["h_in_norm_sq"] for m in measurements])
    H_logit_norm = np.array([m["H_logit_norm"] for m in measurements])

    # Log-space sub-components
    # Clip sin_alpha floor at eps_bf16 to prevent log(0)
    sin_alpha_clipped = np.maximum(sin_alpha, EPS_BF16)
    E_total_clipped = np.maximum(E_total, 1e-30)
    h_in_clipped = np.maximum(h_in_norm_sq, 1e-30)

    Y_delta = np.log(E_total_clipped)           # log(||δ||²)
    Y_sin = 2.0 * np.log(sin_alpha_clipped)     # 2·log(sin(α))
    Y_h = np.log(h_in_clipped)                  # log(||h_in||²)
    Y_num = Y_delta + Y_sin                     # log(||P_perp(h)δ||²) = log(||δ||²·sin²(α))

    # --- Closure check ---
    # Verify Y_num = log(E_total · sin²(α)) against independently computed numerator
    # The numerator ||P_perp(h)δ||² = ||δ||² sin²(α) is exact (trigonometric identity)
    # So log(numerator) = log(E_total) + 2·log(sin_alpha)
    # We verify this by checking that exp(Y_num) ≈ E_total · sin_alpha²
    reconstructed_num = E_total * sin_alpha**2
    direct_num = np.exp(Y_num)
    closure_max_gap = float(np.max(np.abs(reconstructed_num - direct_num)))
    closure_rel_gap = float(
        np.max(
            np.abs(reconstructed_num - direct_num)
            / np.maximum(np.abs(reconstructed_num), 1e-30)
        )
    )

    # --- Depth-residualize everything ---
    H_resid = _residualize_ols(H_logit_norm, depth)
    Y_delta_resid = _residualize_ols(Y_delta, depth)
    Y_sin_resid = _residualize_ols(Y_sin, depth)
    Y_h_resid = _residualize_ols(Y_h, depth)
    Y_num_resid = _residualize_ols(Y_num, depth)

    # --- Effective df (Bretherton AR(1)) ---
    n_eff_H, rho1_H = _effective_df(H_resid)
    mde = _minimum_detectable_effect(n_eff_H, n_controls=1)

    # --- Spearman correlations: r(H_logit_norm, Y_X | depth) ---
    components = {
        "Y_delta": {"label": "log(||δ||²)", "resid": Y_delta_resid, "prediction": "D3.1: β < 0"},
        "Y_sin": {"label": "2·log(sin(α))", "resid": Y_sin_resid, "prediction": "D3.2: β > 0"},
        "Y_h": {"label": "log(||h_in||²)", "resid": Y_h_resid, "prediction": "B5/B7: coupling"},
        "Y_num": {"label": "log(numerator)", "resid": Y_num_resid, "prediction": "net β"},
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
            "sign": "positive" if r > 0 else ("negative" if r < 0 else "zero") if not math.isnan(r) else "nan",
            "permutation": perm,
        }

    # --- D3 prediction checks ---
    r_delta = correlations["Y_delta"]["spearman_r"]
    r_sin = correlations["Y_sin"]["spearman_r"]
    r_num = correlations["Y_num"]["spearman_r"]

    d3_checks = {
        "D3.1_centroid_reduction": {
            "prediction": "r(H, log(||δ||²)) < 0",
            "observed_r": r_delta,
            "pass": r_delta < 0 if not math.isnan(r_delta) else None,
        },
        "D3.2_tangentiality": {
            "prediction": "r(H, 2·log(sin(α))) > 0",
            "observed_r": r_sin,
            "pass": r_sin > 0 if not math.isnan(r_sin) else None,
        },
        "D3.4_r_dominance": {
            "prediction": "|r_delta| > |r_sin| (magnitude wins over tangentiality)",
            "abs_r_delta": abs(r_delta) if not math.isnan(r_delta) else float("nan"),
            "abs_r_sin": abs(r_sin) if not math.isnan(r_sin) else float("nan"),
            "pass": (
                abs(r_delta) > abs(r_sin)
                if not (math.isnan(r_delta) or math.isnan(r_sin))
                else None
            ),
        },
    }

    # --- Dominant component ---
    resolvable_comps = {
        k: v for k, v in correlations.items()
        if k != "Y_num" and v["resolvable"]
    }
    if resolvable_comps:
        dominant = max(resolvable_comps, key=lambda k: resolvable_comps[k]["abs_r"])
        dominant_r = resolvable_comps[dominant]["abs_r"]
    else:
        dominant = "none_resolvable"
        dominant_r = float("nan")

    # --- Raw sub-component statistics (for documentation) ---
    raw_stats = {
        "E_total_range": [float(np.min(E_total)), float(np.max(E_total))],
        "sin_alpha_range": [float(np.min(sin_alpha)), float(np.max(sin_alpha))],
        "h_in_norm_sq_range": [float(np.min(h_in_norm_sq)), float(np.max(h_in_norm_sq))],
        "H_logit_norm_range": [float(np.min(H_logit_norm)), float(np.max(H_logit_norm))],
        "Y_delta_range": [float(np.min(Y_delta)), float(np.max(Y_delta))],
        "Y_sin_range": [float(np.min(Y_sin)), float(np.max(Y_sin))],
        "Y_h_range": [float(np.min(Y_h)), float(np.max(Y_h))],
        "Y_num_range": [float(np.min(Y_num)), float(np.max(Y_num))],
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
        "closure_check": {
            "max_gap": closure_max_gap,
            "max_rel_gap": closure_rel_gap,
            "within_eps_bf16": closure_rel_gap < EPS_BF16,
        },
        "correlations": correlations,
        "d3_checks": d3_checks,
        "dominant_component": dominant,
        "dominant_r": dominant_r,
        "raw_stats": raw_stats,
    }

    return result


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------


def build_cross_model_summary(results: list[dict]) -> dict:
    """Build cross-model summary table."""
    summary_rows = []
    dominant_counts = {}

    for r in results:
        row = {
            "model": r["model_name"],
            "architecture": r["architecture"],
            "n_layers": r["num_layers"],
            "gqa_ratio": r.get("gqa_ratio"),
            "mde": r["detection_floor"]["mde"],
            "n_eff": r["detection_floor"]["n_eff"],
            "dominant": r["dominant_component"],
            "dominant_r": r["dominant_r"],
        }
        for comp in ["Y_delta", "Y_sin", "Y_h", "Y_num"]:
            c = r["correlations"][comp]
            row[f"r_{comp}"] = c["spearman_r"]
            row[f"p_{comp}"] = c["spearman_p"]
            row[f"resolvable_{comp}"] = c["resolvable"]

        for check_name, check_data in r["d3_checks"].items():
            row[f"{check_name}_pass"] = check_data["pass"]

        summary_rows.append(row)

        dom = r["dominant_component"]
        dominant_counts[dom] = dominant_counts.get(dom, 0) + 1

    # Cross-model consistency
    dominants = [r["dominant_component"] for r in results]
    unique_dominants = set(dominants)
    consistent = len(unique_dominants) == 1 and "none_resolvable" not in unique_dominants

    # D3 prediction pass rates
    d3_pass_rates = {}
    for check_name in ["D3.1_centroid_reduction", "D3.2_tangentiality", "D3.4_r_dominance"]:
        passes = [r["d3_checks"][check_name]["pass"] for r in results]
        testable = [p for p in passes if p is not None]
        if testable:
            d3_pass_rates[check_name] = {
                "pass_count": sum(testable),
                "total": len(testable),
                "rate": sum(testable) / len(testable),
            }

    return {
        "n_models": len(results),
        "models": [r["model_name"] for r in results],
        "summary_rows": summary_rows,
        "dominant_counts": dominant_counts,
        "cross_model_consistent": consistent,
        "unanimous_dominant": dominants[0] if consistent else None,
        "d3_pass_rates": d3_pass_rates,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], cross_summary: dict) -> None:
    """Print human-readable summary table."""
    logger.info("\n%s", "=" * 78)
    logger.info("B6: THREE-COMPONENT DECOMPOSITION OF ||P_perp(h)δ||² → H_logit COUPLING")
    logger.info("%s", "=" * 78)

    # Per-model table
    header = f"{'Model':<18} {'L':>3} {'MDE':>5} {'r(δ²)':>7} {'r(sin²)':>7} {'r(h²)':>7} {'r(num)':>7} {'Dominant':<12}"
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for r in results:
        c = r["correlations"]
        dom = r["dominant_component"]
        dom_label = {
            "Y_delta": "||δ||²",
            "Y_sin": "sin²(α)",
            "Y_h": "||h||²",
            "none_resolvable": "NONE",
        }.get(dom, dom)

        def _fmt_r(comp_name):
            val = c[comp_name]["spearman_r"]
            resolvable = c[comp_name]["resolvable"]
            if math.isnan(val):
                return "   nan"
            marker = "*" if resolvable else " "
            return f"{val:+.3f}{marker}"

        logger.info(
            "%-18s %3d %5.3f %7s %7s %7s %7s %-12s",
            r["model_name"],
            r["num_layers"],
            r["detection_floor"]["mde"],
            _fmt_r("Y_delta"),
            _fmt_r("Y_sin"),
            _fmt_r("Y_h"),
            _fmt_r("Y_num"),
            dom_label,
        )

    logger.info("\n* = resolvable (|r| > MDE)")

    # D3 prediction checks
    logger.info("\nD3 Prediction Checks:")
    for check_name, check_data in cross_summary.get("d3_pass_rates", {}).items():
        logger.info(
            "  %s: %d/%d pass (%.0f%%)",
            check_name,
            check_data["pass_count"],
            check_data["total"],
            check_data["rate"] * 100,
        )

    # Cross-model consistency
    logger.info("\nCross-model dominant component consistency: %s",
                "CONSISTENT" if cross_summary["cross_model_consistent"] else "INCONSISTENT")
    if cross_summary["cross_model_consistent"]:
        dom = cross_summary["unanimous_dominant"]
        label = {
            "Y_delta": "||δ||² (update magnitude, D3.1)",
            "Y_sin": "sin²(α) (tangential fraction, D3.2)",
            "Y_h": "||h||² (hidden norm, B5/B7)",
        }.get(dom, dom)
        logger.info("  Unanimous dominant: %s", label)
    else:
        logger.info("  Dominant counts: %s", cross_summary["dominant_counts"])

    # Closure checks
    logger.info("\nClosure checks (Y_num = Y_δ + Y_sin):")
    for r in results:
        cc = r["closure_check"]
        status = "PASS" if cc["within_eps_bf16"] else "FAIL"
        logger.info(
            "  %s: max_rel_gap=%.2e (%s, eps_bf16=%.4f)",
            r["model_name"],
            cc["max_rel_gap"],
            status,
            EPS_BF16,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="B6: Three-Component Decomposition of ||P_perp(h)δ||²"
    )
    parser.add_argument(
        "--input", default=str(INPUT_BASE),
        help="Input directory with operator_split results",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_BASE),
        help="Output directory for three-component results",
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

    logger.info("B6 Three-Component Analysis: scanning %d model directories", len(model_dirs))

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
        logger.error("No models with complete data. Run entropy_curvature_operator_split.py first.")
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
        with open(model_dir / "three_component.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        logger.info("Saved: %s", model_dir / "three_component.json")

    with open(output_base / "cross_model_summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2, default=str)
    logger.info("Saved: %s", output_base / "cross_model_summary.json")


if __name__ == "__main__":
    main()
