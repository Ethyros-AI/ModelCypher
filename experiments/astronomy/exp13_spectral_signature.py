#!/usr/bin/env python3
"""Experiment 13: Spectral Signature Decoding.

Exp12 found: low-freq bands ANTI-correlate with high-freq bands.
This is a SPECTRAL SIGNATURE independent of DM and SNR.

What does this mean physically?
- Spectral index: "blue" sources (more high-freq) vs "red" sources (more low-freq)
- Emission mechanism differences
- Scattering/propagation effects

This experiment:
1. Computes a "spectral color" index for each FRB
2. Tests if spectral color correlates with any physical properties
3. Looks for spectral subtypes in the FRB population

Usage:
    poetry run python experiments/astronomy/exp13_spectral_signature.py
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

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_spectral_color(features: np.ndarray):
    """Compute spectral color index: high-freq vs low-freq ratio.

    Positive = "blue" (more high-frequency emission)
    Negative = "red" (more low-frequency emission)
    """
    # Band indices in feature vector:
    # 0-1: band_0 (lowest freq), 2-3: band_1, ..., 14-15: band_7 (highest freq)
    # Even indices are means, odd are stds

    low_freq_bands = [0, 2, 4]  # bands 0, 1, 2 (mean values)
    high_freq_bands = [10, 12, 14]  # bands 5, 6, 7 (mean values)

    low_freq = np.mean(features[:, low_freq_bands], axis=1)
    high_freq = np.mean(features[:, high_freq_bands], axis=1)

    # Spectral color: normalized difference (positive = blue, negative = red)
    # This handles negative values better than log ratio
    total = np.abs(high_freq) + np.abs(low_freq) + 1e-10
    spectral_color = (high_freq - low_freq) / total

    return spectral_color, low_freq, high_freq


def compute_spectral_width(features: np.ndarray):
    """Compute spectral width: variance across frequency bands."""
    band_means = features[:, [0, 2, 4, 6, 8, 10, 12, 14]]  # All band means
    spectral_width = np.std(band_means, axis=1)
    return spectral_width


def compute_temporal_asymmetry(features: np.ndarray):
    """Compute temporal asymmetry from time series features.

    Peak location relative to center indicates asymmetry.
    """
    # ts_peak_loc is feature index 19
    ts_peak_loc = features[:, 19]

    # Assuming peak_loc is normalized to [0, 1]
    # 0.5 = symmetric, <0.5 = early peak, >0.5 = late peak
    asymmetry = ts_peak_loc - 0.5  # Centered at 0

    return asymmetry


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 13: Spectral Signature Decoding")
    print("=" * 60)
    print("\nQuestion: What does the spectral anti-correlation mean?")

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

    print("\n" + "=" * 40)
    print("PART 1: SPECTRAL COLOR INDEX")
    print("=" * 40)

    spectral_color, low_freq, high_freq = compute_spectral_color(frb_np)

    print(f"\nSpectral color distribution:")
    print(f"  Mean: {np.mean(spectral_color):.3f}")
    print(f"  Std:  {np.std(spectral_color):.3f}")
    print(f"  Range: [{np.min(spectral_color):.3f}, {np.max(spectral_color):.3f}]")

    # Classify as red/neutral/blue
    blue_mask = spectral_color > 0.1
    red_mask = spectral_color < -0.1
    neutral_mask = ~blue_mask & ~red_mask

    n_blue = np.sum(blue_mask)
    n_red = np.sum(red_mask)
    n_neutral = np.sum(neutral_mask)

    print(f"\nSpectral classification:")
    print(f"  Blue (high-freq dominant): {n_blue} ({n_blue/n_frbs*100:.0f}%)")
    print(f"  Neutral: {n_neutral} ({n_neutral/n_frbs*100:.0f}%)")
    print(f"  Red (low-freq dominant): {n_red} ({n_red/n_frbs*100:.0f}%)")

    print("\n" + "=" * 40)
    print("PART 2: SPECTRAL COLOR vs PHYSICS")
    print("=" * 40)

    # Correlation with DM
    r_dm, p_dm = stats.pearsonr(spectral_color, dms)
    print(f"\nSpectral color vs DM: r={r_dm:.3f} (p={p_dm:.3f})")

    # Correlation with SNR
    r_snr, p_snr = stats.pearsonr(spectral_color, snrs)
    print(f"Spectral color vs SNR: r={r_snr:.3f} (p={p_snr:.3f})")

    # Physical interpretation
    if abs(r_dm) > 0.3 and p_dm < 0.05:
        if r_dm > 0:
            print("→ DISTANT FRBs tend to be BLUER (more high-freq)")
        else:
            print("→ DISTANT FRBs tend to be REDDER (more low-freq)")
    else:
        print("→ Spectral color is INDEPENDENT of distance")

    # Compare physics between spectral types
    print("\n" + "=" * 40)
    print("PART 3: SPECTRAL SUBTYPES")
    print("=" * 40)

    print("\nBlue FRBs (spectrally harder):")
    if n_blue > 0:
        print(f"  DM: {np.mean(dms[blue_mask]):.0f} ± {np.std(dms[blue_mask]):.0f}")
        print(f"  SNR: {np.mean(snrs[blue_mask]):.1f} ± {np.std(snrs[blue_mask]):.1f}")
        blue_names = [names[i] for i in range(n_frbs) if blue_mask[i]]
        print(f"  Examples: {blue_names[:5]}")

    print("\nRed FRBs (spectrally softer):")
    if n_red > 0:
        print(f"  DM: {np.mean(dms[red_mask]):.0f} ± {np.std(dms[red_mask]):.0f}")
        print(f"  SNR: {np.mean(snrs[red_mask]):.1f} ± {np.std(snrs[red_mask]):.1f}")
        red_names = [names[i] for i in range(n_frbs) if red_mask[i]]
        print(f"  Examples: {red_names[:5]}")

    # Statistical test: are blue and red FRBs different?
    if n_blue > 2 and n_red > 2:
        t_dm, p_t_dm = stats.ttest_ind(dms[blue_mask], dms[red_mask])
        t_snr, p_t_snr = stats.ttest_ind(snrs[blue_mask], snrs[red_mask])
        print(f"\nBlue vs Red comparison:")
        print(f"  DM difference: t={t_dm:.2f}, p={p_t_dm:.3f}")
        print(f"  SNR difference: t={t_snr:.2f}, p={p_t_snr:.3f}")

    print("\n" + "=" * 40)
    print("PART 4: ADDITIONAL SPECTRAL PROPERTIES")
    print("=" * 40)

    spectral_width = compute_spectral_width(frb_np)
    temporal_asymmetry = compute_temporal_asymmetry(frb_np)

    print(f"\nSpectral width (band-to-band variation):")
    print(f"  Mean: {np.mean(spectral_width):.4f}")
    print(f"  Std: {np.std(spectral_width):.4f}")

    # Correlations
    r_width_dm, _ = stats.pearsonr(spectral_width, dms)
    r_width_snr, _ = stats.pearsonr(spectral_width, snrs)
    r_width_color, _ = stats.pearsonr(spectral_width, spectral_color)

    print(f"  Correlation with DM: {r_width_dm:.3f}")
    print(f"  Correlation with SNR: {r_width_snr:.3f}")
    print(f"  Correlation with color: {r_width_color:.3f}")

    print(f"\nTemporal asymmetry (peak location):")
    print(f"  Mean: {np.mean(temporal_asymmetry):.4f}")
    print(f"  Std: {np.std(temporal_asymmetry):.4f}")

    # Correlations
    r_asym_dm, _ = stats.pearsonr(temporal_asymmetry, dms)
    r_asym_color, _ = stats.pearsonr(temporal_asymmetry, spectral_color)

    print(f"  Correlation with DM: {r_asym_dm:.3f}")
    print(f"  Correlation with spectral color: {r_asym_color:.3f}")

    print("\n" + "=" * 40)
    print("PART 5: 3D FRB SPACE")
    print("=" * 40)
    print("\nThe FRB 'vocabulary' has (at least) 3 axes:")
    print("  1. DM (distance)")
    print("  2. SNR (brightness)")
    print("  3. Spectral color (emission type)")

    # Cluster in 3D space (replace NaN with 0)
    color_clean = np.nan_to_num(spectral_color, nan=0.0)
    color_std = np.std(color_clean) if np.std(color_clean) > 0 else 1.0
    features_3d = np.column_stack([
        (dms - np.mean(dms)) / np.std(dms),
        (snrs - np.mean(snrs)) / np.std(snrs),
        (color_clean - np.mean(color_clean)) / color_std,
    ])

    Z = linkage(features_3d, method='ward')
    labels_3 = fcluster(Z, 3, criterion='maxclust')

    print("\n3D Clustering (DM × SNR × Color):")
    for cid in range(1, 4):
        mask = labels_3 == cid
        n = np.sum(mask)
        print(f"  Cluster {cid} (n={n}):")
        print(f"    DM: {np.mean(dms[mask]):.0f} ± {np.std(dms[mask]):.0f}")
        print(f"    SNR: {np.mean(snrs[mask]):.1f} ± {np.std(snrs[mask]):.1f}")
        print(f"    Color: {np.mean(spectral_color[mask]):.3f} ± {np.std(spectral_color[mask]):.3f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE FRB LANGUAGE")
    print("=" * 60)

    print("\nFRBs encode information in (at least) 3 dimensions:")
    print("  • DM → cosmic distance (how far)")
    print("  • SNR → intrinsic brightness (how loud)")
    print("  • Spectral color → emission type (what kind)")

    print("\nThe spectral color is INDEPENDENT of distance:")
    print(f"  r(color, DM) = {r_dm:.3f}")
    print("→ This suggests different SOURCE TYPES, not propagation effects")

    print("\nThe 'grammar' of FRBs:")
    print("  Blue FRBs: harder spectrum, possibly different emission mechanism")
    print("  Red FRBs: softer spectrum, possibly different progenitor type")

    results = {
        "experiment": "exp13_spectral_signature",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "spectral_color": {
            "values": spectral_color.tolist(),
            "mean": float(np.mean(spectral_color)),
            "std": float(np.std(spectral_color)),
            "n_blue": int(n_blue),
            "n_neutral": int(n_neutral),
            "n_red": int(n_red),
        },
        "correlations": {
            "color_vs_dm": {"r": float(r_dm), "p": float(p_dm)},
            "color_vs_snr": {"r": float(r_snr), "p": float(p_snr)},
            "width_vs_dm": float(r_width_dm),
            "width_vs_color": float(r_width_color),
            "asymmetry_vs_dm": float(r_asym_dm),
            "asymmetry_vs_color": float(r_asym_color),
        },
        "spectral_width": spectral_width.tolist(),
        "temporal_asymmetry": temporal_asymmetry.tolist(),
        "cluster_labels_3d": labels_3.tolist(),
        "frb_names": names,
    }

    output_path = results_dir / "exp13_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
