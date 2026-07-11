#!/usr/bin/env python3
"""Multivariate analysis of isometry metrics from real LoRA adapter validation.

Reads results JSON from validate_isometry_real.py and performs:
  3a. Inter-metric correlation matrix (Pearson + Spearman)
  3b. Log-space correlations for nonlinear relationships
  3c. Composite score correlations (IS, Defect, AWP, SBF)
  3d. Layer-type stratification
  3e. Regime analysis (safe / moderate / extreme)
  3f. Multivariate OLS R²

Usage:
    poetry run python scripts/analyze_isometry_composites.py --input results/results_350m.json
    poetry run python scripts/analyze_isometry_composites.py --input results/results_350m_v2.json --output results/analysis_350m.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Pure-Python statistics (no Backend needed for post-hoc analysis) ─────────

def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _ranks(values: list[float]) -> list[float]:
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    return _pearson(_ranks(xs[:n]), _ranks(ys[:n]))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


# ── Composite scores (mirrors lora_isometry.py for offline computation) ──────

@dataclass
class CompositeRow:
    """Composite scores for one layer measurement."""
    isometry_score: float
    isometry_defect: float
    awp: float
    sbf: float | None


def compute_composite(rec: dict) -> CompositeRow:
    """Compute composites from a flat layer_metrics record."""
    spr = rec["spectral_preservation_ratio"]
    eta = rec["weyl_utilization"]
    d_g = rec["grassmann_distance"]
    theta = rec["spectral_angle"]
    kappa = rec["condition_ratio"]
    rfd = rec["relative_frobenius_deviation"]

    # IS = SPR × (1 - η) × cos(d_G)
    is_score = spr * (1.0 - eta) * math.cos(d_g)
    is_score = max(0.0, min(1.0, is_score))

    # Defect = √((1-SPR)² + (θ/90)² + (ln κ)² + η²)
    ln_kappa = math.log(max(kappa, 1e-30))
    defect = math.sqrt(
        (1.0 - spr) ** 2
        + (theta / 90.0) ** 2
        + ln_kappa ** 2
        + eta ** 2
    )

    # AWP = η × RFD
    awp = eta * rfd

    # SBF = η × (‖ΔW‖₂ / σ_k)
    sbf = None
    sigma_k = rec.get("sigma_k")
    if sigma_k is not None and sigma_k > 0:
        sbf = eta * (rec["delta_spectral_norm"] / sigma_k)

    return CompositeRow(
        isometry_score=is_score,
        isometry_defect=defect,
        awp=awp,
        sbf=sbf,
    )


# ── OLS via Gaussian elimination (pure Python, no numpy) ────────────────────

def _ols_r_squared(X: list[list[float]], y: list[float]) -> tuple[float, list[float]]:
    """OLS regression returning (R², coefficients).

    X: n×p design matrix (each row is a sample, each col a feature).
    y: n×1 target.

    Solves normal equations: (X^T X) β = X^T y via Gaussian elimination.
    """
    n = len(y)
    p = len(X[0]) if X else 0
    if n <= p or p == 0:
        return 0.0, []

    # X^T X (p×p)
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    # X^T y (p×1)
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]

    # Augmented matrix [XtX | Xty]
    aug = [XtX[a] + [Xty[a]] for a in range(p)]

    # Gaussian elimination with partial pivoting
    for col in range(p):
        # Find pivot
        max_row = col
        for row in range(col + 1, p):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-30:
            return 0.0, [0.0] * p

        for row in range(col + 1, p):
            factor = aug[row][col] / pivot
            for j in range(col, p + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    beta = [0.0] * p
    for row in range(p - 1, -1, -1):
        s = aug[row][p]
        for j in range(row + 1, p):
            s -= aug[row][j] * beta[j]
        if abs(aug[row][row]) < 1e-30:
            beta[row] = 0.0
        else:
            beta[row] = s / aug[row][row]

    # R²
    y_mean = _mean(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))

    if ss_tot < 1e-30:
        return 0.0, beta

    r_squared = 1.0 - ss_res / ss_tot
    return max(0.0, r_squared), beta


# ── Load and flatten data ───────────────────────────────────────────────────

def load_records(input_path: Path) -> list[dict]:
    """Load all layer-level records from results JSON."""
    data = json.loads(input_path.read_text())
    records = []
    for phase in data:
        for adapter in phase.get("adapters", []):
            for lm in adapter.get("layer_metrics", []):
                sr = lm.get("scale_ratio", float("inf"))
                if sr == float("inf") or sr != sr:
                    continue
                lm["adapter_name"] = adapter["adapter_name"]
                records.append(lm)
    return records


def extract_layer_type(layer_key: str) -> str:
    """Extract the layer type suffix from a layer key.

    Examples:
        model.layers.0.conv.in_proj -> conv.in_proj
        model.layers.5.feed_forward.w2 -> feed_forward.w2
    """
    parts = layer_key.split(".")
    # Find "layers" then skip the index
    for i, p in enumerate(parts):
        if p == "layers" and i + 2 < len(parts):
            return ".".join(parts[i + 2 :])
    return layer_key


# ── Analysis sections ───────────────────────────────────────────────────────

METRIC_NAMES = [
    "spectral_preservation_ratio",
    "spectral_angle",
    "condition_ratio",
    "grassmann_distance",
    "relative_frobenius_deviation",
    "weyl_utilization",
    "amplification_cv",
    "delta_spectral_norm",
]


def section_3a_correlation_matrix(records: list[dict]) -> dict:
    """3a. Inter-metric 8×8 Pearson + Spearman matrix."""
    vectors = {name: [r[name] for r in records] for name in METRIC_NAMES}
    n = len(records)

    pearson_matrix = {}
    spearman_matrix = {}
    for a in METRIC_NAMES:
        pearson_matrix[a] = {}
        spearman_matrix[a] = {}
        for b in METRIC_NAMES:
            pearson_matrix[a][b] = round(_pearson(vectors[a], vectors[b]), 4)
            spearman_matrix[a][b] = round(_spearman(vectors[a], vectors[b]), 4)

    # Find redundant pairs (|r| > 0.9)
    redundant = []
    for i, a in enumerate(METRIC_NAMES):
        for b in METRIC_NAMES[i + 1 :]:
            r = pearson_matrix[a][b]
            if abs(r) > 0.9:
                redundant.append({"pair": f"{a} ~ {b}", "pearson_r": r})

    return {
        "n_records": n,
        "pearson": pearson_matrix,
        "spearman": spearman_matrix,
        "redundant_pairs": redundant,
    }


def section_3b_log_space(records: list[dict]) -> dict:
    """3b. Log-space correlations where Spearman-Pearson gap suggests nonlinearity."""
    scale_ratios = [r["scale_ratio"] for r in records]

    results = {}
    for name in METRIC_NAMES:
        values = [r[name] for r in records]
        r_pearson = _pearson(scale_ratios, values)
        r_spearman = _spearman(scale_ratios, values)
        gap = abs(r_spearman) - abs(r_pearson)

        entry = {
            "pearson": round(r_pearson, 4),
            "spearman": round(r_spearman, 4),
            "gap": round(gap, 4),
        }

        if gap > 0.1:
            # Test log transforms
            log_transforms = {}

            # log(scale_ratio) vs metric
            log_sr = []
            vals_filt = []
            for sr, v in zip(scale_ratios, values):
                if sr > 0:
                    log_sr.append(math.log(sr))
                    vals_filt.append(v)
            if len(log_sr) >= 3:
                log_transforms["log_scale_ratio_vs_metric"] = round(
                    _pearson(log_sr, vals_filt), 4
                )

            # Handle specific log transforms based on metric
            if name == "spectral_preservation_ratio":
                # log(1 - SPR) vs log(scale_ratio)
                log_defect = []
                log_sr2 = []
                for sr, v in zip(scale_ratios, values):
                    if sr > 0 and (1.0 - v) > 0:
                        log_sr2.append(math.log(sr))
                        log_defect.append(math.log(1.0 - v))
                if len(log_sr2) >= 3:
                    log_transforms["log_1_minus_SPR_vs_log_scale"] = round(
                        _pearson(log_sr2, log_defect), 4
                    )
            elif name in ("relative_frobenius_deviation", "weyl_utilization"):
                log_metric = []
                log_sr2 = []
                for sr, v in zip(scale_ratios, values):
                    if sr > 0 and v > 0:
                        log_sr2.append(math.log(sr))
                        log_metric.append(math.log(v))
                if len(log_sr2) >= 3:
                    log_transforms[f"log_{name}_vs_log_scale"] = round(
                        _pearson(log_sr2, log_metric), 4
                    )

            entry["log_transforms"] = log_transforms

        results[name] = entry

    return results


def section_3c_composite_correlations(records: list[dict]) -> dict:
    """3c. Composite score correlations with scale_ratio."""
    scale_ratios = [r["scale_ratio"] for r in records]
    composites = [compute_composite(r) for r in records]

    is_vals = [c.isometry_score for c in composites]
    defect_vals = [c.isometry_defect for c in composites]
    awp_vals = [c.awp for c in composites]
    sbf_vals = [c.sbf for c in composites if c.sbf is not None]
    sbf_srs = [sr for sr, c in zip(scale_ratios, composites) if c.sbf is not None]

    # Log-space variants
    log_sr_full = []
    is_for_log = []
    defect_for_log = []
    awp_for_log = []
    for sr, c in zip(scale_ratios, composites):
        if sr > 0:
            log_sr_full.append(math.log(sr))
            is_for_log.append(c.isometry_score)
            defect_for_log.append(c.isometry_defect)
            awp_for_log.append(c.awp)

    results = {
        "isometry_score": {
            "pearson": round(_pearson(scale_ratios, is_vals), 4),
            "spearman": round(_spearman(scale_ratios, is_vals), 4),
            "log_pearson": round(_pearson(log_sr_full, is_for_log), 4) if len(log_sr_full) >= 3 else None,
            "mean": round(_mean(is_vals), 4),
            "std": round(_std(is_vals), 4),
        },
        "isometry_defect": {
            "pearson": round(_pearson(scale_ratios, defect_vals), 4),
            "spearman": round(_spearman(scale_ratios, defect_vals), 4),
            "log_pearson": round(_pearson(log_sr_full, defect_for_log), 4) if len(log_sr_full) >= 3 else None,
            "mean": round(_mean(defect_vals), 4),
            "std": round(_std(defect_vals), 4),
        },
        "alignment_weighted_perturbation": {
            "pearson": round(_pearson(scale_ratios, awp_vals), 4),
            "spearman": round(_spearman(scale_ratios, awp_vals), 4),
            "log_pearson": round(_pearson(log_sr_full, awp_for_log), 4) if len(log_sr_full) >= 3 else None,
            "mean": round(_mean(awp_vals), 4),
            "std": round(_std(awp_vals), 4),
        },
    }

    if sbf_vals:
        log_sbf_sr = []
        log_sbf_v = []
        for sr, c in zip(scale_ratios, composites):
            if c.sbf is not None and sr > 0:
                log_sbf_sr.append(math.log(sr))
                log_sbf_v.append(c.sbf)
        results["spectral_budget_fraction"] = {
            "pearson": round(_pearson(sbf_srs, sbf_vals), 4),
            "spearman": round(_spearman(sbf_srs, sbf_vals), 4),
            "log_pearson": round(_pearson(log_sbf_sr, log_sbf_v), 4) if len(log_sbf_sr) >= 3 else None,
            "mean": round(_mean(sbf_vals), 4),
            "std": round(_std(sbf_vals), 4),
            "n": len(sbf_vals),
        }
    else:
        results["spectral_budget_fraction"] = {
            "note": "sigma_k not available in v1 results; re-run validation for SBF",
        }

    # Compare best composite vs best individual
    best_individual = max(
        abs(_pearson(scale_ratios, [r[name] for r in records]))
        for name in METRIC_NAMES
    )
    best_composite = max(
        abs(results["isometry_score"]["pearson"]),
        abs(results["isometry_defect"]["pearson"]),
        abs(results["alignment_weighted_perturbation"]["pearson"]),
    )
    results["comparison"] = {
        "best_individual_abs_r": round(best_individual, 4),
        "best_composite_abs_r": round(best_composite, 4),
        "improvement": round(best_composite - best_individual, 4),
    }

    return results


def section_3d_layer_stratification(records: list[dict]) -> dict:
    """3d. Group by layer type, compute per-group statistics."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        lt = extract_layer_type(r["layer_key"])
        groups.setdefault(lt, []).append(r)

    composites_by_group = {}
    for lt, group_recs in sorted(groups.items()):
        scale_ratios = [r["scale_ratio"] for r in group_recs]
        comps = [compute_composite(r) for r in group_recs]

        is_vals = [c.isometry_score for c in comps]
        defect_vals = [c.isometry_defect for c in comps]
        awp_vals = [c.awp for c in comps]

        composites_by_group[lt] = {
            "n_layers": len(group_recs),
            "scale_ratio": {
                "mean": round(_mean(scale_ratios), 4),
                "std": round(_std(scale_ratios), 4),
            },
            "isometry_score": {
                "mean": round(_mean(is_vals), 4),
                "std": round(_std(is_vals), 4),
                "corr_with_scale_ratio": round(_pearson(scale_ratios, is_vals), 4),
            },
            "isometry_defect": {
                "mean": round(_mean(defect_vals), 4),
                "std": round(_std(defect_vals), 4),
                "corr_with_scale_ratio": round(_pearson(scale_ratios, defect_vals), 4),
            },
            "awp": {
                "mean": round(_mean(awp_vals), 4),
                "std": round(_std(awp_vals), 4),
                "corr_with_scale_ratio": round(_pearson(scale_ratios, awp_vals), 4),
            },
        }

    return composites_by_group


def section_3e_regime_analysis(records: list[dict]) -> dict:
    """3e. Split by scale_ratio into safe/moderate/extreme regimes."""
    regimes = {"safe": [], "moderate": [], "extreme": []}
    for r in records:
        sr = r["scale_ratio"]
        if sr < 1.0:
            regimes["safe"].append(r)
        elif sr <= 10.0:
            regimes["moderate"].append(r)
        else:
            regimes["extreme"].append(r)

    regime_stats = {}
    for name, group_recs in regimes.items():
        if not group_recs:
            regime_stats[name] = {"n": 0}
            continue

        comps = [compute_composite(r) for r in group_recs]
        regime_stats[name] = {
            "n": len(group_recs),
            "isometry_score": {
                "mean": round(_mean([c.isometry_score for c in comps]), 4),
                "std": round(_std([c.isometry_score for c in comps]), 4),
            },
            "isometry_defect": {
                "mean": round(_mean([c.isometry_defect for c in comps]), 4),
                "std": round(_std([c.isometry_defect for c in comps]), 4),
            },
            "awp": {
                "mean": round(_mean([c.awp for c in comps]), 4),
                "std": round(_std([c.awp for c in comps]), 4),
            },
        }

    # Permutation test: safe vs extreme on IS
    safe_is = [compute_composite(r).isometry_score for r in regimes["safe"]]
    extreme_is = [compute_composite(r).isometry_score for r in regimes["extreme"]]

    if len(safe_is) >= 3 and len(extreme_is) >= 3:
        # Simple permutation test (pure Python, no Backend dependency)
        import random

        observed_diff = _mean(safe_is) - _mean(extreme_is)
        combined = safe_is + extreme_is
        n_safe = len(safe_is)
        n_perms = min(1000, math.comb(len(combined), n_safe))

        random.seed(42)
        count_extreme = 0
        for _ in range(n_perms):
            random.shuffle(combined)
            perm_diff = _mean(combined[:n_safe]) - _mean(combined[n_safe:])
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        p_value = count_extreme / n_perms

        regime_stats["permutation_test_safe_vs_extreme"] = {
            "observed_diff_IS": round(observed_diff, 4),
            "p_value": round(p_value, 4),
            "n_permutations": n_perms,
        }

    return regime_stats


def section_3f_multivariate_r_squared(records: list[dict]) -> dict:
    """3f. OLS: log(scale_ratio) ~ metrics + layer depth + CV."""
    y = []
    X = []

    for r in records:
        sr = r["scale_ratio"]
        if sr <= 0:
            continue

        # Extract layer depth from key (e.g., "model.layers.5.conv.in_proj" -> 5)
        depth = 0
        parts = r["layer_key"].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    depth = int(parts[i + 1])
                except ValueError:
                    pass
                break

        kappa = r["condition_ratio"]
        ln_kappa = math.log(max(kappa, 1e-30))

        row = [
            1.0 - r["spectral_preservation_ratio"],  # SPR defect
            r["spectral_angle"] / 90.0,               # normalized angle
            ln_kappa,                                   # log condition ratio
            r["weyl_utilization"],                      # η
            r["relative_frobenius_deviation"],          # RFD
            math.log(depth + 1),                        # log(depth+1)
            r["amplification_cv"],                      # CV
        ]
        X.append(row)
        y.append(math.log(sr))

    if len(y) < 10:
        return {"error": "Too few records for OLS", "n": len(y)}

    r_squared, beta = _ols_r_squared(X, y)

    feature_names = [
        "1-SPR", "theta/90", "ln(kappa)", "eta",
        "RFD", "ln(depth+1)", "CV",
    ]

    # Compare to best individual r²
    scale_ratios = [r["scale_ratio"] for r in records]
    best_ind_r = max(
        abs(_pearson(scale_ratios, [r[name] for r in records]))
        for name in METRIC_NAMES
    )
    best_ind_r2 = best_ind_r ** 2

    return {
        "n": len(y),
        "n_features": len(feature_names),
        "R_squared": round(r_squared, 4),
        "best_individual_r_squared": round(best_ind_r2, 4),
        "improvement_ratio": round(r_squared / best_ind_r2, 2) if best_ind_r2 > 0 else None,
        "coefficients": {name: round(b, 4) for name, b in zip(feature_names, beta)},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multivariate analysis of isometry validation results",
    )
    parser.add_argument(
        "--input", required=True, type=str,
        help="Path to results JSON from validate_isometry_real.py",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for analysis results",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(input_path)
    if not records:
        print("ERROR: No valid records found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} layer measurements")
    print(f"Adapters: {len(set(r['adapter_name'] for r in records))}")
    print()

    # Check if v2 (has sigma_k)
    has_sigma_k = any(r.get("sigma_k") is not None for r in records)
    if has_sigma_k:
        print("Data format: v2 (with sigma_k -> SBF available)")
    else:
        print("Data format: v1 (no sigma_k -> SBF unavailable, re-run validation)")
    print()

    report = {}

    # 3a
    print("=" * 70)
    print("3a. Inter-Metric Correlation Matrix")
    print("=" * 70)
    corr = section_3a_correlation_matrix(records)
    report["correlation_matrix"] = corr
    print(f"  {corr['n_records']} records")
    if corr["redundant_pairs"]:
        print("  Redundant pairs (|r| > 0.9):")
        for pair in corr["redundant_pairs"]:
            print(f"    {pair['pair']}: r={pair['pearson_r']}")
    else:
        print("  No redundant pairs (|r| > 0.9)")

    # Print Pearson correlation with scale_ratio for each metric
    scale_ratios = [r["scale_ratio"] for r in records]
    print("\n  Pearson r with scale_ratio:")
    for name in METRIC_NAMES:
        vals = [r[name] for r in records]
        r_p = _pearson(scale_ratios, vals)
        r_s = _spearman(scale_ratios, vals)
        print(f"    {name:40s} Pearson={r_p:+.4f}  Spearman={r_s:+.4f}")
    print()

    # 3b
    print("=" * 70)
    print("3b. Log-Space Correlations")
    print("=" * 70)
    log_space = section_3b_log_space(records)
    report["log_space"] = log_space
    for name, entry in log_space.items():
        gap = entry["gap"]
        marker = " ** NONLINEAR" if gap > 0.1 else ""
        print(f"  {name:40s} gap={gap:+.4f}{marker}")
        if "log_transforms" in entry:
            for tname, r in entry["log_transforms"].items():
                print(f"    {tname}: r={r:+.4f}")
    print()

    # 3c
    print("=" * 70)
    print("3c. Composite Score Correlations")
    print("=" * 70)
    composite_corr = section_3c_composite_correlations(records)
    report["composite_correlations"] = composite_corr
    for cname in ["isometry_score", "isometry_defect", "alignment_weighted_perturbation", "spectral_budget_fraction"]:
        entry = composite_corr[cname]
        if "note" in entry:
            print(f"  {cname:40s} {entry['note']}")
        else:
            pearson = entry.get("pearson", "N/A")
            spearman = entry.get("spearman", "N/A")
            log_p = entry.get("log_pearson", "N/A")
            print(f"  {cname:40s} Pearson={pearson:+.4f}  Spearman={spearman:+.4f}  log_Pearson={log_p}")
    comp = composite_corr["comparison"]
    print(f"\n  Best individual |r|: {comp['best_individual_abs_r']}")
    print(f"  Best composite  |r|: {comp['best_composite_abs_r']}")
    print(f"  Improvement:         {comp['improvement']:+.4f}")
    print()

    # 3d
    print("=" * 70)
    print("3d. Layer-Type Stratification")
    print("=" * 70)
    strat = section_3d_layer_stratification(records)
    report["layer_stratification"] = strat
    for lt, stats in strat.items():
        n = stats["n_layers"]
        sr = stats["scale_ratio"]
        is_corr = stats["isometry_score"]["corr_with_scale_ratio"]
        print(f"  {lt:30s} n={n:4d}  scale_ratio={sr['mean']:8.2f}+/-{sr['std']:8.2f}  IS_corr={is_corr:+.4f}")
    print()

    # 3e
    print("=" * 70)
    print("3e. Regime Analysis")
    print("=" * 70)
    regime = section_3e_regime_analysis(records)
    report["regime_analysis"] = regime
    for rname in ["safe", "moderate", "extreme"]:
        rstats = regime[rname]
        n = rstats.get("n", 0)
        if n > 0:
            is_m = rstats["isometry_score"]["mean"]
            def_m = rstats["isometry_defect"]["mean"]
            print(f"  {rname:12s} n={n:4d}  IS={is_m:.4f}  Defect={def_m:.4f}")
        else:
            print(f"  {rname:12s} n=   0")
    if "permutation_test_safe_vs_extreme" in regime:
        pt = regime["permutation_test_safe_vs_extreme"]
        print(f"\n  Permutation test (safe vs extreme IS): diff={pt['observed_diff_IS']:+.4f}, p={pt['p_value']:.4f}")
    print()

    # 3f
    print("=" * 70)
    print("3f. Multivariate OLS R-squared")
    print("=" * 70)
    ols = section_3f_multivariate_r_squared(records)
    report["ols_regression"] = ols
    if "error" in ols:
        print(f"  {ols['error']}")
    else:
        print(f"  n={ols['n']}, features={ols['n_features']}")
        print(f"  R² = {ols['R_squared']:.4f}")
        print(f"  Best individual r² = {ols['best_individual_r_squared']:.4f}")
        if ols["improvement_ratio"] is not None:
            print(f"  Joint R² / best individual r² = {ols['improvement_ratio']:.2f}x")
        print("\n  Coefficients:")
        for name, b in ols["coefficients"].items():
            print(f"    {name:20s} β = {b:+.4f}")
    print()

    # Write JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Full report written to {output_path}")

    # Final summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_ind = comp["best_individual_abs_r"]
    best_comp = comp["best_composite_abs_r"]
    r2 = ols.get("R_squared", 0.0)
    best_ind_r2 = ols.get("best_individual_r_squared", 0.0)
    print(f"  Best individual metric |r|:  {best_ind:.4f} (r²={best_ind**2:.4f})")
    print(f"  Best composite metric |r|:   {best_comp:.4f} (r²={best_comp**2:.4f})")
    print(f"  Multivariate OLS R²:         {r2:.4f}")
    print(f"  Improvement over individual:  {r2 / best_ind_r2:.2f}x" if best_ind_r2 > 0 else "  N/A")


if __name__ == "__main__":
    main()
