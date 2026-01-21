#!/usr/bin/env python3
"""Experiment 17: Null Space Analysis.

Key question: What dimensions are FRBs NOT using?

The null space = directions with near-zero variance in the data.
These are the "unused capacity" of the feature space.

Why this matters:
- If FRBs only use 5 dimensions of 26, there's 21D of unused space
- This unused space could contain additional information
- Or it could be noise/measurement artifacts

Analysis:
1. SVD to find the rank and null space
2. What features contribute to the null space?
3. Do different FRB populations use different dimensions?

Usage:
    poetry run python experiments/astronomy/exp17_null_space.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_svd_analysis(features: np.ndarray, threshold_ratio: float = 0.01):
    """Compute SVD and analyze rank structure.

    Args:
        features: [N, D] feature matrix
        threshold_ratio: fraction of max singular value below which we consider null

    Returns:
        Dictionary with SVD analysis
    """
    # Center the data
    centered = features - np.mean(features, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Determine effective rank
    threshold = threshold_ratio * S[0]
    effective_rank = np.sum(S > threshold)

    # Null space dimensions
    null_rank = len(S) - effective_rank

    # Explained variance by each singular value
    total_var = np.sum(S ** 2)
    explained_var = (S ** 2) / total_var

    # Cumulative explained variance
    cumulative_var = np.cumsum(explained_var)

    # Find dimensions for 90%, 95%, 99% variance
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    return {
        "singular_values": S.tolist(),
        "explained_variance": explained_var.tolist(),
        "cumulative_variance": cumulative_var.tolist(),
        "effective_rank": int(effective_rank),
        "null_rank": int(null_rank),
        "total_dimensions": len(S),
        "n_for_90_percent": int(n_90),
        "n_for_95_percent": int(n_95),
        "n_for_99_percent": int(n_99),
        "right_singular_vectors": Vt,  # For feature analysis
        "left_singular_vectors": U,  # For sample analysis
    }


def analyze_feature_contributions(Vt: np.ndarray, feature_names: list):
    """Analyze which features contribute to which singular directions."""
    n_components, n_features = Vt.shape

    contributions = []
    for i in range(min(n_components, 10)):  # Top 10 components
        component_contrib = []
        for j in range(n_features):
            component_contrib.append({
                "feature": feature_names[j] if j < len(feature_names) else f"f{j}",
                "weight": float(Vt[i, j]),
                "abs_weight": float(abs(Vt[i, j])),
            })
        # Sort by absolute weight
        component_contrib.sort(key=lambda x: x["abs_weight"], reverse=True)
        contributions.append({
            "component": i + 1,
            "top_features": component_contrib[:5],
        })

    return contributions


def analyze_population_subspaces(features: np.ndarray, labels: np.ndarray, backend):
    """Analyze if different populations use different subspaces."""
    unique_labels = np.unique(labels)
    subspace_info = []

    for label in unique_labels:
        mask = labels == label
        subset = features[mask]

        if len(subset) >= 3:  # Need at least 3 samples
            svd_result = compute_svd_analysis(subset)
            subspace_info.append({
                "label": int(label),
                "n_samples": int(np.sum(mask)),
                "effective_rank": svd_result["effective_rank"],
                "n_for_90_percent": svd_result["n_for_90_percent"],
            })

    return subspace_info


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 17: Null Space Analysis")
    print("=" * 60)
    print("\nQuestion: What dimensions are FRBs NOT using?")
    print("(Null space = unused capacity)")

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

    n_features = frb_np.shape[1]
    print(f"Feature space: {n_features} dimensions")

    feature_names = [
        "band_0_mean", "band_0_std", "band_1_mean", "band_1_std",
        "band_2_mean", "band_2_std", "band_3_mean", "band_3_std",
        "band_4_mean", "band_4_std", "band_5_mean", "band_5_std",
        "band_6_mean", "band_6_std", "band_7_mean", "band_7_std",
        "ts_mean", "ts_std", "ts_max", "ts_peak_loc",
        "spec_entropy", "dm", "width", "fluence", "total_intensity", "snr_proxy"
    ]

    print("\n" + "=" * 40)
    print("PART 1: SVD ANALYSIS")
    print("=" * 40)

    svd_result = compute_svd_analysis(frb_np)

    print(f"\nSingular value spectrum:")
    for i in range(min(10, len(svd_result["singular_values"]))):
        sv = svd_result["singular_values"][i]
        ev = svd_result["explained_variance"][i] * 100
        cv = svd_result["cumulative_variance"][i] * 100
        print(f"  SV{i+1}: {sv:.4f} ({ev:.1f}% var, {cv:.1f}% cumulative)")

    print(f"\nEffective rank: {svd_result['effective_rank']} / {svd_result['total_dimensions']}")
    print(f"Null space dimensions: {svd_result['null_rank']}")
    print(f"\nDimensions needed for:")
    print(f"  90% variance: {svd_result['n_for_90_percent']}")
    print(f"  95% variance: {svd_result['n_for_95_percent']}")
    print(f"  99% variance: {svd_result['n_for_99_percent']}")

    print("\n" + "=" * 40)
    print("PART 2: FEATURE CONTRIBUTIONS")
    print("=" * 40)

    contributions = analyze_feature_contributions(
        svd_result["right_singular_vectors"],
        feature_names
    )

    print("\nTop features contributing to each principal direction:")
    for comp in contributions[:5]:  # First 5 components
        top_3 = comp["top_features"][:3]
        features_str = ", ".join([f"{f['feature']}({f['weight']:.2f})" for f in top_3])
        print(f"  PC{comp['component']}: {features_str}")

    print("\n" + "=" * 40)
    print("PART 3: NULL SPACE STRUCTURE")
    print("=" * 40)

    # Which features are in the null space?
    # These are features with low weights in the significant singular vectors
    # but high weights in the near-zero singular vectors

    n_significant = svd_result["effective_rank"]
    Vt = svd_result["right_singular_vectors"]

    # Compute "significance" of each feature
    # = sum of squared weights in significant directions
    feature_significance = np.sum(Vt[:n_significant, :] ** 2, axis=0)

    # Compute "null-ness" of each feature
    # = sum of squared weights in null directions
    if n_significant < len(Vt):
        feature_nullness = np.sum(Vt[n_significant:, :] ** 2, axis=0)
    else:
        feature_nullness = np.zeros(n_features)

    print("\nFeature significance (in active subspace):")
    sig_ranking = sorted(zip(feature_names, feature_significance), key=lambda x: x[1], reverse=True)
    for name, sig in sig_ranking[:5]:
        print(f"  {name}: {sig:.4f}")

    print("\nFeatures mostly in NULL space:")
    null_ranking = sorted(zip(feature_names, feature_nullness), key=lambda x: x[1], reverse=True)
    for name, null in null_ranking[:5]:
        print(f"  {name}: {null:.4f}")

    print("\n" + "=" * 40)
    print("PART 4: POPULATION SUBSPACES")
    print("=" * 40)

    # Cluster FRBs by DM
    dm_quartiles = np.percentile(dms, [33, 66])
    dm_labels = np.digitize(dms, dm_quartiles)

    print("\nAnalyzing subspaces by DM group...")
    dm_subspaces = analyze_population_subspaces(frb_np, dm_labels, backend)

    for info in dm_subspaces:
        dm_group = ["low DM", "medium DM", "high DM"][info["label"]]
        print(f"  {dm_group} (n={info['n_samples']}): rank={info['effective_rank']}, 90% var in {info['n_for_90_percent']}D")

    # Cluster by SNR
    snr_quartiles = np.percentile(snrs, [33, 66])
    snr_labels = np.digitize(snrs, snr_quartiles)

    print("\nAnalyzing subspaces by SNR group...")
    snr_subspaces = analyze_population_subspaces(frb_np, snr_labels, backend)

    for info in snr_subspaces:
        snr_group = ["low SNR", "medium SNR", "high SNR"][info["label"]]
        print(f"  {snr_group} (n={info['n_samples']}): rank={info['effective_rank']}, 90% var in {info['n_for_90_percent']}D")

    print("\n" + "=" * 40)
    print("PART 5: INFORMATION CAPACITY")
    print("=" * 40)

    # How much "room" is there for additional information?
    active_dims = svd_result["n_for_95_percent"]
    total_dims = svd_result["total_dimensions"]
    unused_dims = total_dims - active_dims

    print(f"\nActive dimensions (95% var): {active_dims}")
    print(f"Unused dimensions: {unused_dims}")
    print(f"Capacity utilization: {100 * active_dims / total_dims:.0f}%")

    # Spectral gap
    if len(svd_result["singular_values"]) > active_dims:
        sv_active = svd_result["singular_values"][active_dims - 1]
        sv_null = svd_result["singular_values"][active_dims]
        spectral_gap = sv_active / (sv_null + 1e-10)
        print(f"\nSpectral gap (SV{active_dims}/SV{active_dims+1}): {spectral_gap:.1f}x")
        print("(Large gap = clear separation between signal and null space)")
    else:
        spectral_gap = None

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    capacity_used = active_dims / total_dims

    if capacity_used < 0.3:
        print(f"\n** FRBs use only {100*capacity_used:.0f}% of feature space capacity **")
        print("→ Most dimensions are UNUSED (null space)")
        print("→ Information is COMPRESSED into a low-dimensional subspace")
        print("→ This is consistent with structured, not random, information")
    elif capacity_used < 0.6:
        print(f"\n** FRBs use {100*capacity_used:.0f}% of feature space **")
        print("→ Moderate compression")
    else:
        print(f"\n** FRBs use {100*capacity_used:.0f}% of feature space **")
        print("→ Little compression, data spans most dimensions")
        print("→ Could indicate noise or high-complexity signal")

    results = {
        "experiment": "exp17_null_space",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "n_features": n_features,
        "svd_analysis": {
            "singular_values": svd_result["singular_values"],
            "explained_variance": svd_result["explained_variance"],
            "cumulative_variance": svd_result["cumulative_variance"],
            "effective_rank": svd_result["effective_rank"],
            "null_rank": svd_result["null_rank"],
            "n_for_90_percent": svd_result["n_for_90_percent"],
            "n_for_95_percent": svd_result["n_for_95_percent"],
            "n_for_99_percent": svd_result["n_for_99_percent"],
        },
        "feature_contributions": contributions,
        "feature_significance": dict(zip(feature_names, feature_significance.tolist())),
        "feature_nullness": dict(zip(feature_names, feature_nullness.tolist())),
        "population_subspaces": {
            "by_dm": dm_subspaces,
            "by_snr": snr_subspaces,
        },
        "capacity": {
            "active_dims": active_dims,
            "unused_dims": unused_dims,
            "utilization": float(capacity_used),
            "spectral_gap": float(spectral_gap) if spectral_gap else None,
        },
        "frb_names": names,
    }

    output_path = results_dir / "exp17_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
