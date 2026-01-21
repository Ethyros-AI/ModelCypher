#!/usr/bin/env python3
"""Experiment 12: Residual Structure Analysis.

PC1 = SNR, PC2 ≈ DM. But is there MORE structure beyond distance × brightness?

If we REMOVE the SNR and DM effects, what remains?
- If residuals are noise → FRBs only encode basic physics
- If residuals have structure → FRBs encode ADDITIONAL information

This could reveal:
- Emission mechanism differences
- Source type signatures
- Propagation effects
- Something we don't understand yet

Usage:
    poetry run python experiments/astronomy/exp12_residual_structure.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def remove_linear_effects(features: np.ndarray, predictors: np.ndarray):
    """Remove linear effects of predictors from features.

    For each feature dimension, fit: feature = a*predictor1 + b*predictor2 + ... + residual
    Return the residuals.
    """
    n_samples, n_features = features.shape
    n_predictors = predictors.shape[1]

    # Add intercept
    X = np.column_stack([np.ones(n_samples), predictors])

    residuals = np.zeros_like(features)
    r_squared = []

    for i in range(n_features):
        y = features[:, i]

        # Skip constant features
        if np.std(y) < 1e-10:
            residuals[:, i] = 0
            r_squared.append(0)
            continue

        # Fit linear model: y = X @ beta
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            residuals[:, i] = y - y_pred

            # Compute R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r_squared.append(r2)
        except:
            residuals[:, i] = y
            r_squared.append(0)

    return residuals, np.array(r_squared)


def analyze_residual_structure(residuals: np.ndarray, names: list, backend):
    """Analyze what structure remains in the residuals."""
    results = {}

    # 1. Variance analysis
    total_var = np.var(residuals)
    per_feature_var = np.var(residuals, axis=0)
    results["total_variance"] = float(total_var)
    results["per_feature_variance"] = per_feature_var.tolist()

    # Find features with most residual variance
    var_ranking = sorted(enumerate(per_feature_var), key=lambda x: x[1], reverse=True)
    results["top_residual_features"] = [
        {"index": idx, "variance": float(var)} for idx, var in var_ranking[:5]
    ]

    # 2. Clustering on residuals
    if np.std(residuals) > 1e-10:
        Z = linkage(residuals, method='ward')
        labels_2 = fcluster(Z, 2, criterion='maxclust')
        labels_3 = fcluster(Z, 3, criterion='maxclust')
        results["residual_clusters_2"] = labels_2.tolist()
        results["residual_clusters_3"] = labels_3.tolist()
    else:
        results["residual_clusters_2"] = [1] * len(residuals)
        results["residual_clusters_3"] = [1] * len(residuals)

    # 3. Intrinsic dimension of residuals
    try:
        id_estimator = IntrinsicDimension(backend)
        # Filter out near-zero variance features
        active_features = [i for i, v in enumerate(per_feature_var) if v > 1e-8]
        if len(active_features) > 1:
            residuals_active = residuals[:, active_features]
            id_result = id_estimator.compute(backend.array(residuals_active), with_ci=True)
            results["residual_intrinsic_dimension"] = float(id_result.intrinsic_dimension)
            if id_result.ci:
                results["residual_id_ci"] = [float(id_result.ci.lower), float(id_result.ci.upper)]
        else:
            results["residual_intrinsic_dimension"] = 0
    except Exception as e:
        results["residual_intrinsic_dimension"] = None
        results["id_error"] = str(e)

    # 4. Pairwise correlations in residuals
    n_features = residuals.shape[1]
    correlations = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.std(residuals[:, i]) > 1e-10 and np.std(residuals[:, j]) > 1e-10:
                r, p = stats.pearsonr(residuals[:, i], residuals[:, j])
                if not np.isnan(r):
                    correlations.append({
                        "feature_i": i,
                        "feature_j": j,
                        "r": float(r),
                        "p": float(p)
                    })

    # Strong residual correlations
    strong_corr = [c for c in correlations if abs(c["r"]) > 0.5]
    results["strong_residual_correlations"] = strong_corr

    return results


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 12: Residual Structure Analysis")
    print("=" * 60)
    print("\nQuestion: What structure remains after removing DM and SNR effects?")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    # Extract features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Normalize DM and SNR for regression
    dm_norm = (dms - np.mean(dms)) / np.std(dms)
    snr_norm = (snrs - np.mean(snrs)) / np.std(snrs)
    predictors = np.column_stack([dm_norm, snr_norm])

    print("\n" + "=" * 40)
    print("PART 1: REMOVING DM AND SNR EFFECTS")
    print("=" * 40)

    residuals, r_squared = remove_linear_effects(frb_np, predictors)

    print(f"\nR² of linear model (DM + SNR) per feature:")
    feature_names = [
        "band_0_mean", "band_0_std", "band_1_mean", "band_1_std",
        "band_2_mean", "band_2_std", "band_3_mean", "band_3_std",
        "band_4_mean", "band_4_std", "band_5_mean", "band_5_std",
        "band_6_mean", "band_6_std", "band_7_mean", "band_7_std",
        "ts_mean", "ts_std", "ts_max", "ts_peak_loc",
        "spec_entropy", "dm", "width", "fluence", "total_intensity", "snr_proxy"
    ]

    # Show R² for each feature
    r2_items = [(feature_names[i] if i < len(feature_names) else f"f{i}", r_squared[i])
                for i in range(len(r_squared))]
    r2_items.sort(key=lambda x: x[1], reverse=True)

    print("\nMost explained by DM+SNR:")
    for name, r2 in r2_items[:8]:
        print(f"  {name}: R²={r2:.3f} ({r2*100:.1f}% explained)")

    print("\nLeast explained by DM+SNR:")
    for name, r2 in r2_items[-5:]:
        print(f"  {name}: R²={r2:.3f} ({r2*100:.1f}% explained)")

    total_variance_explained = np.mean(r_squared)
    print(f"\nOverall: DM+SNR explains {total_variance_explained*100:.1f}% of feature variance")

    print("\n" + "=" * 40)
    print("PART 2: RESIDUAL STRUCTURE")
    print("=" * 40)

    residual_analysis = analyze_residual_structure(residuals, names, backend)

    print(f"\nTotal residual variance: {residual_analysis['total_variance']:.6f}")

    if residual_analysis.get("residual_intrinsic_dimension"):
        print(f"Residual intrinsic dimension: {residual_analysis['residual_intrinsic_dimension']:.2f}D")
        if residual_analysis.get("residual_id_ci"):
            ci = residual_analysis["residual_id_ci"]
            print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\nFeatures with most residual variance:")
    for item in residual_analysis["top_residual_features"][:5]:
        idx = item["index"]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {name}: var={item['variance']:.6f}")

    # Clustering on residuals
    labels_2 = residual_analysis["residual_clusters_2"]
    labels_3 = residual_analysis["residual_clusters_3"]

    print("\n" + "=" * 40)
    print("PART 3: RESIDUAL CLUSTERS")
    print("=" * 40)
    print("\nDo residual clusters reveal hidden structure?")

    for n_clust, labels in [(2, labels_2), (3, labels_3)]:
        print(f"\n--- {n_clust} Residual Clusters ---")
        for cid in range(1, n_clust + 1):
            mask = np.array(labels) == cid
            n_members = np.sum(mask)
            dm_mean = np.mean(dms[mask])
            dm_std = np.std(dms[mask])
            snr_mean = np.mean(snrs[mask])
            snr_std = np.std(snrs[mask])

            # Get FRB names in this cluster
            cluster_names = [names[i] for i, m in enumerate(mask) if m]

            print(f"  Cluster {cid} (n={n_members}): DM={dm_mean:.0f}±{dm_std:.0f}, SNR={snr_mean:.1f}±{snr_std:.1f}")

    # Check if residual clusters are different from DM/SNR clusters
    print("\n" + "=" * 40)
    print("PART 4: RESIDUAL CORRELATIONS")
    print("=" * 40)

    strong_corr = residual_analysis.get("strong_residual_correlations", [])
    if strong_corr:
        print(f"\nStrong correlations remaining after removing DM+SNR:")
        for c in strong_corr[:10]:
            i, j = c["feature_i"], c["feature_j"]
            name_i = feature_names[i] if i < len(feature_names) else f"f{i}"
            name_j = feature_names[j] if j < len(feature_names) else f"f{j}"
            print(f"  {name_i} ↔ {name_j}: r={c['r']:.3f}")
    else:
        print("\nNo strong correlations remain after removing DM+SNR effects.")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    unexplained = 1 - total_variance_explained
    print(f"\n{unexplained*100:.1f}% of feature variance is NOT explained by DM and SNR.")

    if residual_analysis.get("residual_intrinsic_dimension", 0) > 1:
        print(f"Residuals have {residual_analysis['residual_intrinsic_dimension']:.1f}D structure.")
        print("→ FRBs encode information BEYOND just distance and brightness.")
    else:
        print("Residuals have minimal structure.")
        print("→ FRBs may only encode distance and brightness.")

    if strong_corr:
        print(f"\n{len(strong_corr)} strong correlations remain after removing physics.")
        print("→ These correlations may reveal emission mechanism or propagation effects.")

    results = {
        "experiment": "exp12_residual_structure",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "physics_explained": {
            "total_r_squared": float(total_variance_explained),
            "per_feature_r_squared": r_squared.tolist(),
        },
        "residual_analysis": residual_analysis,
        "frb_names": names,
    }

    output_path = results_dir / "exp12_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
