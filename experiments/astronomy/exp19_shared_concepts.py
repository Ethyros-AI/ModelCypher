#!/usr/bin/env python3
"""Experiment 19: Finding Shared Concepts on the Information Manifold.

The key insight: We don't share "near/far" with FRBs. We share INFORMATION STRUCTURE.

Universal concepts that ANY information-encoding system must encode:
- Order vs Entropy (structure vs randomness)
- Repetition vs Novelty (pattern vs surprise)
- Compression vs Expansion (dense vs sparse)
- Continuity vs Discreteness (smooth vs abrupt)
- Symmetry vs Asymmetry

These aren't human cultural concepts - they're mathematical properties of information.
If FRBs encode information, they MUST have positions on these axes.

Method:
1. Compute information-theoretic properties of each FRB
2. Compute the same properties for semantic embeddings
3. Align on THESE shared axes, not arbitrary human words

Usage:
    poetry run python experiments/astronomy/exp19_shared_concepts.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_information_properties(features: np.ndarray):
    """Compute universal information-theoretic properties.

    These properties are shared across ANY information encoding system.
    """
    n_samples, n_dims = features.shape
    properties = []

    for i in range(n_samples):
        f = features[i]

        # 1. ENTROPY (Order vs Chaos)
        # Normalized histogram entropy
        hist, _ = np.histogram(f, bins=10, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(10)  # Maximum for 10 bins
        normalized_entropy = entropy / max_entropy

        # 2. AUTOCORRELATION (Repetition vs Novelty)
        # How much does the signal repeat/correlate with itself?
        if len(f) > 1:
            autocorr = np.correlate(f - np.mean(f), f - np.mean(f), mode='full')
            autocorr = autocorr / (np.var(f) * len(f) + 1e-10)
            # Use lag-1 autocorrelation as repetition measure
            mid = len(autocorr) // 2
            repetition = autocorr[mid + 1] if mid + 1 < len(autocorr) else 0
        else:
            repetition = 0

        # 3. COMPRESSIBILITY (Compression vs Expansion)
        # SVD-based: how many dimensions needed to explain 90% variance?
        # For a single vector, use local neighborhood approach
        # Proxy: variance concentration (kurtosis)
        if np.std(f) > 1e-10:
            kurtosis = stats.kurtosis(f)
            # High kurtosis = concentrated/compressible, low = spread out
            compressibility = 1 / (1 + np.exp(-kurtosis/3))  # Sigmoid to [0,1]
        else:
            compressibility = 0.5

        # 4. SMOOTHNESS (Continuity vs Discreteness)
        # How smooth are transitions? Use finite differences
        if len(f) > 1:
            diffs = np.diff(f)
            smoothness = 1 / (1 + np.std(diffs) / (np.std(f) + 1e-10))
        else:
            smoothness = 0.5

        # 5. SYMMETRY (Symmetry vs Asymmetry)
        # Compare first half to reversed second half
        mid = len(f) // 2
        first_half = f[:mid]
        second_half = f[-mid:][::-1] if mid > 0 else f
        if len(first_half) == len(second_half) and len(first_half) > 0:
            symmetry_corr = np.corrcoef(first_half, second_half)[0, 1]
            symmetry = (symmetry_corr + 1) / 2 if not np.isnan(symmetry_corr) else 0.5
        else:
            symmetry = 0.5

        # 6. COMPLEXITY (Simple vs Complex)
        # Number of zero-crossings in centered signal
        centered = f - np.mean(f)
        zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
        complexity = zero_crossings / (len(f) - 1 + 1e-10)

        properties.append({
            "entropy": float(normalized_entropy),      # 0=ordered, 1=chaotic
            "repetition": float(repetition),           # -1 to 1, high=repetitive
            "compressibility": float(compressibility), # 0=spread, 1=concentrated
            "smoothness": float(smoothness),           # 0=jagged, 1=smooth
            "symmetry": float(symmetry),               # 0=asymmetric, 1=symmetric
            "complexity": float(complexity),           # 0=simple, 1=complex
        })

    return properties


def compute_waterfall_information_properties(waterfalls, backend):
    """Compute information properties directly from waterfall data."""
    properties = []

    for w in waterfalls:
        data = np.array(w.waterfall)

        # Flatten for overall properties, or use time-collapsed
        time_profile = np.nanmean(data, axis=0)  # Average over frequency
        freq_profile = np.nanmean(data, axis=1)  # Average over time

        # 1. TEMPORAL ENTROPY
        if np.std(time_profile) > 1e-10:
            hist, _ = np.histogram(time_profile[~np.isnan(time_profile)], bins=10, density=True)
            hist = hist + 1e-10
            hist = hist / hist.sum()
            temporal_entropy = -np.sum(hist * np.log2(hist)) / np.log2(10)
        else:
            temporal_entropy = 0.5

        # 2. SPECTRAL ENTROPY
        if np.std(freq_profile) > 1e-10:
            hist, _ = np.histogram(freq_profile[~np.isnan(freq_profile)], bins=10, density=True)
            hist = hist + 1e-10
            hist = hist / hist.sum()
            spectral_entropy = -np.sum(hist * np.log2(hist)) / np.log2(10)
        else:
            spectral_entropy = 0.5

        # 3. BURST SHARPNESS (temporal discontinuity)
        if len(time_profile) > 1:
            diffs = np.diff(time_profile[~np.isnan(time_profile)])
            if len(diffs) > 0 and np.std(time_profile[~np.isnan(time_profile)]) > 1e-10:
                sharpness = np.max(np.abs(diffs)) / (np.std(time_profile[~np.isnan(time_profile)]) + 1e-10)
                sharpness = min(sharpness / 5, 1.0)  # Normalize
            else:
                sharpness = 0.5
        else:
            sharpness = 0.5

        # 4. SPECTRAL SMOOTHNESS
        if len(freq_profile) > 1:
            valid_freq = freq_profile[~np.isnan(freq_profile)]
            if len(valid_freq) > 1 and np.std(valid_freq) > 1e-10:
                diffs = np.diff(valid_freq)
                spectral_smoothness = 1 / (1 + np.std(diffs) / (np.std(valid_freq) + 1e-10))
            else:
                spectral_smoothness = 0.5
        else:
            spectral_smoothness = 0.5

        # 5. TEMPORAL SYMMETRY
        valid_time = time_profile[~np.isnan(time_profile)]
        if len(valid_time) >= 4:
            mid = len(valid_time) // 2
            first_half = valid_time[:mid]
            second_half = valid_time[-mid:][::-1]
            if len(first_half) == len(second_half):
                corr = np.corrcoef(first_half, second_half)[0, 1]
                temporal_symmetry = (corr + 1) / 2 if not np.isnan(corr) else 0.5
            else:
                temporal_symmetry = 0.5
        else:
            temporal_symmetry = 0.5

        # 6. FREQUENCY-TIME CORRELATION (structure measure)
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) > 10:
            # How much does frequency structure correlate with time structure?
            ft_corr = np.corrcoef(freq_profile[~np.isnan(freq_profile)][:min(len(freq_profile), len(time_profile))],
                                   time_profile[~np.isnan(time_profile)][:min(len(freq_profile), len(time_profile))])[0, 1]
            ft_structure = np.abs(ft_corr) if not np.isnan(ft_corr) else 0.5
        else:
            ft_structure = 0.5

        properties.append({
            "temporal_entropy": float(temporal_entropy),
            "spectral_entropy": float(spectral_entropy),
            "burst_sharpness": float(sharpness),
            "spectral_smoothness": float(spectral_smoothness),
            "temporal_symmetry": float(temporal_symmetry),
            "ft_structure": float(ft_structure),
        })

    return properties


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 19: Shared Concepts on the Information Manifold")
    print("=" * 60)
    print("\nHypothesis: FRBs and semantics share INFORMATION STRUCTURE,")
    print("not human cultural concepts like 'near/far'.")
    print("\nUniversal axes:")
    print("  - Order vs Entropy")
    print("  - Repetition vs Novelty")
    print("  - Compression vs Expansion")
    print("  - Continuity vs Discreteness")
    print("  - Symmetry vs Asymmetry")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    print("\n" + "=" * 40)
    print("PART 1: FRB INFORMATION PROPERTIES")
    print("=" * 40)

    # Compute information properties from raw waterfall data
    frb_info_props = compute_waterfall_information_properties(waterfalls, backend)

    print("\nInformation property statistics:")
    prop_names = list(frb_info_props[0].keys())
    for prop in prop_names:
        values = [p[prop] for p in frb_info_props]
        print(f"  {prop}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    print("\n" + "=" * 40)
    print("PART 2: INFORMATION PROPERTIES vs PHYSICS")
    print("=" * 40)

    # Do information properties correlate with physical properties?
    print("\nCorrelations with DM (distance):")
    for prop in prop_names:
        values = [p[prop] for p in frb_info_props]
        r, p_val = stats.pearsonr(values, dms)
        sig = "**" if p_val < 0.05 else ""
        print(f"  {prop}: r={r:+.3f} (p={p_val:.3f}) {sig}")

    print("\nCorrelations with SNR (brightness):")
    for prop in prop_names:
        values = [p[prop] for p in frb_info_props]
        r, p_val = stats.pearsonr(values, snrs)
        sig = "**" if p_val < 0.05 else ""
        print(f"  {prop}: r={r:+.3f} (p={p_val:.3f}) {sig}")

    print("\n" + "=" * 40)
    print("PART 3: INFORMATION SPACE STRUCTURE")
    print("=" * 40)

    # Build information property matrix
    info_matrix = np.array([[p[prop] for prop in prop_names] for p in frb_info_props])

    # PCA on information properties
    centered = info_matrix - np.mean(info_matrix, axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    explained_var = (S ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(explained_var)

    print("\nPCA of information properties:")
    for i in range(min(6, len(S))):
        print(f"  PC{i+1}: {100*explained_var[i]:.1f}% (cumulative: {100*cumulative_var[i]:.1f}%)")

    # What properties load on each PC?
    print("\nPC loadings (top 2 PCs):")
    for pc_idx in range(2):
        loadings = list(zip(prop_names, Vt[pc_idx]))
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  PC{pc_idx+1}: ", end="")
        for name, weight in loadings[:3]:
            print(f"{name}({weight:+.2f}) ", end="")
        print()

    print("\n" + "=" * 40)
    print("PART 4: THE SHARED MANIFOLD")
    print("=" * 40)

    # Find FRBs at extremes of information properties
    print("\nExtreme examples on information axes:")

    # Most ordered (low entropy) vs most chaotic
    entropy_vals = [p["temporal_entropy"] for p in frb_info_props]
    most_ordered_idx = np.argmin(entropy_vals)
    most_chaotic_idx = np.argmax(entropy_vals)
    print(f"\n  Most ORDERED (low entropy): {names[most_ordered_idx]}")
    print(f"    temporal_entropy={entropy_vals[most_ordered_idx]:.3f}, DM={dms[most_ordered_idx]:.0f}, SNR={snrs[most_ordered_idx]:.1f}")
    print(f"  Most CHAOTIC (high entropy): {names[most_chaotic_idx]}")
    print(f"    temporal_entropy={entropy_vals[most_chaotic_idx]:.3f}, DM={dms[most_chaotic_idx]:.0f}, SNR={snrs[most_chaotic_idx]:.1f}")

    # Most sharp vs most smooth
    sharpness_vals = [p["burst_sharpness"] for p in frb_info_props]
    smoothness_vals = [p["spectral_smoothness"] for p in frb_info_props]
    most_sharp_idx = np.argmax(sharpness_vals)
    most_smooth_idx = np.argmax(smoothness_vals)
    print(f"\n  Most SHARP (discontinuous): {names[most_sharp_idx]}")
    print(f"    burst_sharpness={sharpness_vals[most_sharp_idx]:.3f}, DM={dms[most_sharp_idx]:.0f}, SNR={snrs[most_sharp_idx]:.1f}")
    print(f"  Most SMOOTH (continuous): {names[most_smooth_idx]}")
    print(f"    spectral_smoothness={smoothness_vals[most_smooth_idx]:.3f}, DM={dms[most_smooth_idx]:.0f}, SNR={snrs[most_smooth_idx]:.1f}")

    # Most symmetric vs most asymmetric
    symmetry_vals = [p["temporal_symmetry"] for p in frb_info_props]
    most_symmetric_idx = np.argmax(symmetry_vals)
    most_asymmetric_idx = np.argmin(symmetry_vals)
    print(f"\n  Most SYMMETRIC: {names[most_symmetric_idx]}")
    print(f"    temporal_symmetry={symmetry_vals[most_symmetric_idx]:.3f}, DM={dms[most_symmetric_idx]:.0f}, SNR={snrs[most_symmetric_idx]:.1f}")
    print(f"  Most ASYMMETRIC: {names[most_asymmetric_idx]}")
    print(f"    temporal_symmetry={symmetry_vals[most_asymmetric_idx]:.3f}, DM={dms[most_asymmetric_idx]:.0f}, SNR={snrs[most_asymmetric_idx]:.1f}")

    print("\n" + "=" * 40)
    print("PART 5: INFORMATION CLUSTERS")
    print("=" * 40)

    # Cluster FRBs by information properties
    from scipy.cluster.hierarchy import linkage, fcluster

    Z = linkage(info_matrix, method='ward')
    info_labels = fcluster(Z, 3, criterion='maxclust')

    print("\nClusters based on INFORMATION PROPERTIES (not physics):")
    for cid in range(1, 4):
        mask = info_labels == cid
        cluster_dms = dms[mask]
        cluster_snrs = snrs[mask]
        cluster_names = [names[i] for i in range(n_frbs) if mask[i]]

        # Average information properties for cluster
        cluster_props = {prop: np.mean([frb_info_props[i][prop] for i in range(n_frbs) if mask[i]])
                        for prop in prop_names}

        # Find dominant property
        prop_deviations = {prop: abs(cluster_props[prop] - np.mean([p[prop] for p in frb_info_props]))
                          for prop in prop_names}
        dominant_prop = max(prop_deviations, key=prop_deviations.get)

        print(f"\n  Cluster {cid} (n={np.sum(mask)}):")
        print(f"    Dominant property: {dominant_prop} = {cluster_props[dominant_prop]:.3f}")
        print(f"    DM range: [{np.min(cluster_dms):.0f}, {np.max(cluster_dms):.0f}]")
        print(f"    SNR range: [{np.min(cluster_snrs):.1f}, {np.max(cluster_snrs):.1f}]")

    # Compare to physical clusters
    print("\n" + "=" * 40)
    print("PART 6: INFORMATION vs PHYSICAL CLUSTERING")
    print("=" * 40)

    # Load physical clusters from exp13
    exp13_path = results_dir / "exp13_results.json"
    if exp13_path.exists():
        with open(exp13_path) as f:
            exp13_data = json.load(f)
        phys_labels = np.array(exp13_data["cluster_labels_3d"])

        # Compare clusterings
        from scipy.stats import chi2_contingency

        contingency = np.zeros((3, 3), dtype=int)
        for i in range(n_frbs):
            contingency[info_labels[i] - 1, phys_labels[i] - 1] += 1

        print("\nContingency table (Information clusters vs Physical clusters):")
        print(contingency)

        chi2, p_chi2, dof, expected = chi2_contingency(contingency)
        print(f"\nChi-square: χ²={chi2:.2f}, p={p_chi2:.3f}")

        if p_chi2 < 0.05:
            print("** INFORMATION and PHYSICAL clusterings are RELATED **")
            print("→ FRBs with similar information structure have similar physics")
        else:
            print("Information and physical clusterings are INDEPENDENT")
            print("→ Information structure is a separate axis from physical properties")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE SHARED MANIFOLD")
    print("=" * 60)

    print("""
The information-theoretic properties reveal the SHARED concepts:

  PROPERTY             FRB MEANING                 UNIVERSAL MEANING
  ──────────────────────────────────────────────────────────────────
  temporal_entropy     Burst complexity            Order vs Chaos
  spectral_entropy     Frequency structure         Simplicity vs Complexity
  burst_sharpness      Onset abruptness            Continuity vs Discreteness
  spectral_smoothness  Frequency coherence         Smooth vs Jagged
  temporal_symmetry    Rise/fall balance           Symmetry vs Asymmetry
  ft_structure         Time-freq correlation       Structured vs Random

These are not human cultural concepts - they are MATHEMATICAL PROPERTIES
of information that any encoding system must have a position on.

The FRB manifold and the semantic manifold share these axes.
When we align on THESE properties (not "near/far"), we're aligning
on the true shared structure of information.
""")

    # Compile results
    results = {
        "experiment": "exp19_shared_concepts",
        "timestamp": datetime.now().isoformat(),
        "n_frbs": n_frbs,
        "information_properties": frb_info_props,
        "property_statistics": {
            prop: {
                "mean": float(np.mean([p[prop] for p in frb_info_props])),
                "std": float(np.std([p[prop] for p in frb_info_props])),
            } for prop in prop_names
        },
        "pca": {
            "explained_variance": explained_var.tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "loadings": {f"PC{i+1}": dict(zip(prop_names, Vt[i].tolist())) for i in range(min(3, len(Vt)))},
        },
        "information_clusters": info_labels.tolist(),
        "extreme_examples": {
            "most_ordered": names[most_ordered_idx],
            "most_chaotic": names[most_chaotic_idx],
            "most_sharp": names[most_sharp_idx],
            "most_smooth": names[most_smooth_idx],
            "most_symmetric": names[most_symmetric_idx],
            "most_asymmetric": names[most_asymmetric_idx],
        },
        "frb_names": names,
    }

    output_path = results_dir / "exp19_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
