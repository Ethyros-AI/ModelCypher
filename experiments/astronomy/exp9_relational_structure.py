#!/usr/bin/env python3
"""Experiment 9: Relational Structure Analysis.

The Arrival insight: Look at the WHOLE structure at once, not sequential ordering.

Key test: If FRB features encode physical information (distance, intensity),
then FRBs with SIMILAR physical properties should have SIMILAR features.

Method:
1. Compute pairwise physical property differences (DM-DM, SNR-SNR)
2. Compute pairwise feature distances
3. Test correlation: Do similar physics → similar features?

If FRBs encode physics: Strong correlation
If FRBs are noise: No correlation
If noise is noise: No correlation

This is the relational test - the "grammar" of the data.

Usage:
    poetry run python experiments/astronomy/exp9_relational_structure.py
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
from shared.feature_extraction import batch_extract_features, extract_frb_features


def compute_pairwise_distances(arr: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    n = arr.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(arr[i] - arr[j])
            distances[i, j] = d
            distances[j, i] = d
    return distances


def compute_pairwise_differences(values: np.ndarray) -> np.ndarray:
    """Compute pairwise absolute differences for 1D array."""
    n = len(values)
    differences = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(values[i] - values[j])
            differences[i, j] = d
            differences[j, i] = d
    return differences


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle (excluding diagonal) as flat array."""
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from white noise."""
    rng = np.random.default_rng(seed)
    features = []

    for i in range(n_samples):
        n_freq, n_time = 256, 1024
        waterfall = rng.standard_normal((n_freq, n_time)).astype(np.float32)
        waterfall = backend.array(waterfall)
        time_series = backend.array(rng.standard_normal(n_time).astype(np.float32))
        spectrum = backend.array(rng.standard_normal(n_freq).astype(np.float32))

        frb_feat = extract_frb_features(
            waterfall, time_series, spectrum, backend,
            tns_name=f"noise_{i}"
        )
        features.append(backend.tolist(frb_feat.features))

    return np.array(features)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 9: Relational Structure Analysis")
    print("=" * 60)
    print("\nArrival insight: The 'grammar' is in the RELATIONAL structure.")
    print("Test: Do FRBs with similar physics have similar features?")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])

    # Extract features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Generate noise with fake "physical properties" (random values)
    noise_np = generate_noise_features(n_frbs, backend)
    rng = np.random.default_rng(42)
    fake_dms = rng.uniform(dms.min(), dms.max(), n_frbs)
    fake_snrs = rng.uniform(snrs.min(), snrs.max(), n_frbs)

    print("\n" + "=" * 40)
    print("PART 1: PAIRWISE STRUCTURE")
    print("=" * 40)

    # Compute pairwise distances/differences
    frb_feature_dists = compute_pairwise_distances(frb_np)
    noise_feature_dists = compute_pairwise_distances(noise_np)

    dm_diffs = compute_pairwise_differences(dms)
    snr_diffs = compute_pairwise_differences(snrs)
    fake_dm_diffs = compute_pairwise_differences(fake_dms)

    # Extract upper triangles for correlation
    frb_dists_flat = upper_triangle_values(frb_feature_dists)
    noise_dists_flat = upper_triangle_values(noise_feature_dists)
    dm_flat = upper_triangle_values(dm_diffs)
    snr_flat = upper_triangle_values(snr_diffs)
    fake_dm_flat = upper_triangle_values(fake_dm_diffs)

    print(f"\nNumber of pairwise comparisons: {len(frb_dists_flat)}")

    print("\n" + "=" * 40)
    print("PART 2: FRB RELATIONAL STRUCTURE")
    print("=" * 40)

    # FRB features vs DM differences
    frb_dm_corr, frb_dm_pval = stats.pearsonr(frb_dists_flat, dm_flat)
    print(f"\nFRB feature distance vs DM difference:")
    print(f"  Pearson r = {frb_dm_corr:.4f} (p = {frb_dm_pval:.2e})")

    # FRB features vs SNR differences
    frb_snr_corr, frb_snr_pval = stats.pearsonr(frb_dists_flat, snr_flat)
    print(f"\nFRB feature distance vs SNR difference:")
    print(f"  Pearson r = {frb_snr_corr:.4f} (p = {frb_snr_pval:.2e})")

    # Also compute Spearman (rank correlation - more robust)
    frb_dm_spearman, _ = stats.spearmanr(frb_dists_flat, dm_flat)
    frb_snr_spearman, _ = stats.spearmanr(frb_dists_flat, snr_flat)
    print(f"\nSpearman correlations (rank-based):")
    print(f"  FRB features vs DM: ρ = {frb_dm_spearman:.4f}")
    print(f"  FRB features vs SNR: ρ = {frb_snr_spearman:.4f}")

    print("\n" + "=" * 40)
    print("PART 3: NOISE CONTROL")
    print("=" * 40)
    print("\nNoise should show NO correlation with any physical property")

    # Noise features vs real DM differences (meaningless pairing)
    noise_dm_corr, noise_dm_pval = stats.pearsonr(noise_dists_flat, dm_flat)
    print(f"\nNoise feature distance vs DM difference:")
    print(f"  Pearson r = {noise_dm_corr:.4f} (p = {noise_dm_pval:.2e})")

    # Noise features vs fake DM differences
    noise_fake_corr, noise_fake_pval = stats.pearsonr(noise_dists_flat, fake_dm_flat)
    print(f"\nNoise feature distance vs fake DM difference:")
    print(f"  Pearson r = {noise_fake_corr:.4f} (p = {noise_fake_pval:.2e})")

    print("\n" + "=" * 40)
    print("PART 4: FEATURE-BY-FEATURE ANALYSIS")
    print("=" * 40)
    print("\nWhich specific features correlate with physical properties?")

    # Test each feature dimension
    feature_names = [
        "band_0_mean", "band_0_std", "band_1_mean", "band_1_std",
        "band_2_mean", "band_2_std", "band_3_mean", "band_3_std",
        "band_4_mean", "band_4_std", "band_5_mean", "band_5_std",
        "band_6_mean", "band_6_std", "band_7_mean", "band_7_std",
        "ts_mean", "ts_std", "ts_max", "ts_peak_loc",
        "spec_entropy", "dm", "width", "fluence", "total_intensity", "snr_proxy"
    ]

    dm_correlations = []
    snr_correlations = []

    for i, name in enumerate(feature_names):
        if i < frb_np.shape[1]:
            feature_diffs = compute_pairwise_differences(frb_np[:, i])
            feature_flat = upper_triangle_values(feature_diffs)

            dm_corr, dm_p = stats.pearsonr(feature_flat, dm_flat)
            snr_corr, snr_p = stats.pearsonr(feature_flat, snr_flat)

            dm_correlations.append({"feature": name, "r": dm_corr, "p": dm_p})
            snr_correlations.append({"feature": name, "r": snr_corr, "p": snr_p})

    # Sort by absolute correlation
    dm_correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
    snr_correlations.sort(key=lambda x: abs(x["r"]), reverse=True)

    print("\nTop features correlated with DM:")
    for item in dm_correlations[:5]:
        sig = "*" if item["p"] < 0.05 else ""
        print(f"  {item['feature']}: r = {item['r']:.4f} {sig}")

    print("\nTop features correlated with SNR:")
    for item in snr_correlations[:5]:
        sig = "*" if item["p"] < 0.05 else ""
        print(f"  {item['feature']}: r = {item['r']:.4f} {sig}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print("\nThe 'grammar' of FRB features:")
    print(f"  - DM encodes ~{abs(frb_dm_corr)*100:.1f}% of feature variation")
    print(f"  - SNR encodes ~{abs(frb_snr_corr)*100:.1f}% of feature variation")

    if frb_dm_pval < 0.05 or frb_snr_pval < 0.05:
        print("\n** SIGNIFICANT RELATIONAL STRUCTURE FOUND **")
        if frb_dm_pval < 0.05:
            print(f"  FRB features encode DM (distance) information (p < 0.05)")
        if frb_snr_pval < 0.05:
            print(f"  FRB features encode SNR (intensity) information (p < 0.05)")
    else:
        print("\n** No significant relational structure between features and physics **")

    # Check if correlation is positive (similar physics → similar features)
    # or negative (similar physics → different features)
    if frb_dm_corr > 0:
        print("\nPositive DM correlation: Similar distances → similar features")
    else:
        print("\nNegative DM correlation: Similar distances → DIFFERENT features (!)")

    results = {
        "experiment": "exp9_relational_structure",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "n_pairwise": int(len(frb_dists_flat)),
        "frb_correlations": {
            "dm_pearson_r": float(frb_dm_corr),
            "dm_pearson_p": float(frb_dm_pval),
            "dm_spearman_rho": float(frb_dm_spearman),
            "snr_pearson_r": float(frb_snr_corr),
            "snr_pearson_p": float(frb_snr_pval),
            "snr_spearman_rho": float(frb_snr_spearman),
        },
        "noise_correlations": {
            "dm_pearson_r": float(noise_dm_corr),
            "dm_pearson_p": float(noise_dm_pval),
            "fake_dm_pearson_r": float(noise_fake_corr),
            "fake_dm_pearson_p": float(noise_fake_pval),
        },
        "feature_dm_correlations": dm_correlations[:10],
        "feature_snr_correlations": snr_correlations[:10],
    }

    output_path = results_dir / "exp9_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
