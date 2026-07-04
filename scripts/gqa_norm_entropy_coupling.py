#!/usr/bin/env python3
"""GQA Norm-Entropy Coupling Analysis (Target B, Hypothesis B5)

Computes R²(H_logit → log||h||² | depth) per architecture family from
curvature_accumulation results, then tests the GQA monotone hypothesis:
higher GQA → weaker norm-entropy coupling.

Input: results/sign_split_investigation/curvature_accumulation_results.json
       (or --input path)

Output: per-family R² values, GQA regression, monotone test result.

Reference: docs/research/entropy-curvature-derivation.md, Hypothesis B5.
"""

import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_norm_entropy_coupling(model: dict) -> dict:
    """Compute R²(H_logit → log||h||² | depth) for a single model.

    Returns per-layer data and the depth-residualized R² value.
    """
    layers = model["measurements"]
    n = len(layers)

    h_logit = np.array([l["mean_h_logit"] for l in layers])
    log_h_sq = np.array([l["mean_log_h_in_sq"] for l in layers])
    log_perp_sq = np.array([l["mean_log_perp_delta_sq"] for l in layers])
    depth = np.array([l["layer_idx"] / (n - 1) for l in layers])

    # Depth-residualize both variables (remove linear depth trend via OLS)
    # H_logit residuals
    slope_h, intercept_h, _, _, _ = stats.linregress(depth, h_logit)
    h_resid = h_logit - (slope_h * depth + intercept_h)

    # log||h||² residuals
    slope_den, intercept_den, _, _, _ = stats.linregress(depth, log_h_sq)
    den_resid = log_h_sq - (slope_den * depth + intercept_den)

    # log||P_perp δ||² residuals
    slope_num, intercept_num, _, _, _ = stats.linregress(depth, log_perp_sq)
    num_resid = log_perp_sq - (slope_num * depth + intercept_num)

    # R²(H_logit → log||h||² | depth) from residualized correlation
    r_den, p_den = stats.pearsonr(h_resid, den_resid)
    r_sq_den = r_den ** 2

    # Also compute R²(H_logit → log||P_perp δ||² | depth)
    r_num, p_num = stats.pearsonr(h_resid, num_resid)
    r_sq_num = r_num ** 2

    # Spearman on residuals (non-parametric)
    rho_den, rho_den_p = stats.spearmanr(h_resid, den_resid)
    rho_num, rho_num_p = stats.spearmanr(h_resid, num_resid)

    # Beta ratio: |β_num| / |β_den| (cancellation indicator from Theorem B3)
    # Use OLS slope of residualized component on residualized H_logit
    if np.std(h_resid) > 0:
        beta_den = np.polyfit(h_resid, den_resid, 1)[0]
        beta_num = np.polyfit(h_resid, num_resid, 1)[0]
    else:
        beta_den = 0.0
        beta_num = 0.0

    beta_ratio = abs(beta_num) / abs(beta_den) if abs(beta_den) > 1e-12 else float("inf")

    return {
        "model_name": model["model_name"],
        "architecture": model["architecture"],
        "gqa_ratio": model.get("gqa_ratio"),
        "num_layers": model["num_layers"],
        # Norm-entropy coupling (denominator)
        "r_den": float(r_den),
        "r_sq_den": float(r_sq_den),
        "p_den": float(p_den),
        "rho_den": float(rho_den),
        "rho_den_p": float(rho_den_p),
        # Perp-entropy coupling (numerator)
        "r_num": float(r_num),
        "r_sq_num": float(r_sq_num),
        "p_num": float(p_num),
        "rho_num": float(rho_num),
        "rho_num_p": float(rho_num_p),
        # Cancellation
        "beta_num": float(beta_num),
        "beta_den": float(beta_den),
        "beta_ratio": float(beta_ratio),
    }


def gqa_monotone_test(results: list[dict]) -> dict:
    """Test B5: R²(H→||h||² | depth) is monotone decreasing with GQA.

    Uses only models with known integer GQA ratios (excludes LFM2 hybrid).
    """
    # Filter to models with known GQA
    gqa_models = [r for r in results if r["gqa_ratio"] is not None]

    if len(gqa_models) < 3:
        return {"status": "INSUFFICIENT_DATA", "n_families": len(gqa_models)}

    gqa_vals = np.array([r["gqa_ratio"] for r in gqa_models])
    r_sq_vals = np.array([r["r_sq_den"] for r in gqa_models])
    names = [r["model_name"] for r in gqa_models]

    # Spearman correlation (GQA vs R²)
    rho, p_val = stats.spearmanr(gqa_vals, r_sq_vals)

    # Log-linear regression: R² = a + b * log(GQA)
    log_gqa = np.log(gqa_vals)
    slope, intercept, r_value, p_reg, se = stats.linregress(log_gqa, r_sq_vals)

    # Fisher z-transform regression (same as existing GQA conditioning):
    # z = atanh(r_den) = a + b * log(GQA)
    z_vals = np.arctanh(np.clip(np.array([r["r_den"] for r in gqa_models]), -0.999, 0.999))
    z_slope, z_intercept, z_r, z_p, z_se = stats.linregress(log_gqa, z_vals)

    # Exact permutation test (all n! permutations)
    n = len(gqa_vals)
    n_perms = math.factorial(n)
    more_extreme = 0
    for perm in permutations(range(n)):
        perm_rho, _ = stats.spearmanr(gqa_vals, r_sq_vals[list(perm)])
        if perm_rho <= rho:  # Test for negative monotone
            more_extreme += 1
    perm_p = more_extreme / n_perms

    # Check monotone: is R² strictly decreasing with GQA?
    sorted_by_gqa = sorted(zip(gqa_vals, r_sq_vals, names), key=lambda x: x[0])
    is_monotone = all(
        sorted_by_gqa[i][1] >= sorted_by_gqa[i + 1][1]
        for i in range(len(sorted_by_gqa) - 1)
    )

    return {
        "status": "MONOTONE" if is_monotone else "NON_MONOTONE",
        "n_families": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "permutation_p": float(perm_p),
        "n_permutations": n_perms,
        "log_linear_slope": float(slope),
        "log_linear_intercept": float(intercept),
        "log_linear_r_sq": float(r_value ** 2),
        "fisher_z_slope": float(z_slope),
        "fisher_z_intercept": float(z_intercept),
        "fisher_z_r_sq": float(z_r ** 2),
        "sorted_table": [
            {"model": name, "gqa": int(gqa), "r_sq_den": float(rsq)}
            for gqa, rsq, name in sorted_by_gqa
        ],
        "prediction": "B5 predicts slope < 0 (higher GQA → lower R²)",
        "observed_slope_sign": "negative" if slope < 0 else "positive",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/sign_split_investigation/curvature_accumulation_results.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = load_results(args.input)
    models = data["models"]

    print(f"Loaded {len(models)} models from {args.input}\n")

    # Per-model coupling analysis
    results = []
    for model in models:
        r = compute_norm_entropy_coupling(model)
        results.append(r)
        gqa_str = f"GQA={r['gqa_ratio']}" if r["gqa_ratio"] else "GQA=N/A"
        print(
            f"{r['model_name']:20s} ({gqa_str:8s}): "
            f"R²(H→||h||²)={r['r_sq_den']:.3f} (p={r['p_den']:.3f}), "
            f"R²(H→||Pδ||²)={r['r_sq_num']:.3f} (p={r['p_num']:.3f}), "
            f"|β_num/β_den|={r['beta_ratio']:.2f}"
        )

    # GQA monotone test
    print("\n--- GQA Monotone Test (Hypothesis B5) ---\n")
    gqa_test = gqa_monotone_test(results)

    if gqa_test["status"] == "INSUFFICIENT_DATA":
        print(f"Insufficient data: {gqa_test['n_families']} families with known GQA")
    else:
        print(f"Status: {gqa_test['status']}")
        print(f"Families: {gqa_test['n_families']}")
        print(f"Spearman(GQA, R²): ρ={gqa_test['spearman_rho']:.3f}, p={gqa_test['spearman_p']:.3f}")
        print(f"Permutation p: {gqa_test['permutation_p']:.4f} ({gqa_test['n_permutations']} perms)")
        print(f"Log-linear slope: {gqa_test['log_linear_slope']:.4f} (R²={gqa_test['log_linear_r_sq']:.3f})")
        print(f"Fisher z slope: {gqa_test['fisher_z_slope']:.4f} (R²={gqa_test['fisher_z_r_sq']:.3f})")
        print("\nSorted by GQA:")
        for row in gqa_test["sorted_table"]:
            print(f"  GQA={row['gqa']:2d}  R²={row['r_sq_den']:.3f}  ({row['model']})")

    # Build output
    output = {
        "per_model": results,
        "gqa_monotone_test": gqa_test,
        "source": str(args.input),
        "derivation_ref": "docs/research/entropy-curvature-derivation.md, Hypothesis B5",
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Print JSON to stdout
        print("\n--- Full Results (JSON) ---\n")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
