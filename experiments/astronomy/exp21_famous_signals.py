#!/usr/bin/env python3
"""Experiment 21: Analyzing Famous Space Signals.

Apply our information structure detector to historically significant signals:
1. The Wow! Signal (1977) - most famous SETI candidate
2. Comparison with our FRB baseline

The question: Does the Wow! signal have information structure that
distinguishes it from noise? How does it compare to FRBs?

Usage:
    poetry run python experiments/astronomy/exp21_famous_signals.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_information_score(data_2d: np.ndarray) -> dict:
    """Compute information structure score for a 2D signal.

    Same algorithm as exp20, adapted for general 2D data.
    """
    # Handle the data shape
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(1, -1)

    # Time profile (collapse over frequency/channels)
    time_profile = np.nanmean(data_2d, axis=0)
    # Frequency profile (collapse over time)
    freq_profile = np.nanmean(data_2d, axis=1)

    # Remove NaN/inf
    time_profile = time_profile[np.isfinite(time_profile)]
    freq_profile = freq_profile[np.isfinite(freq_profile)]

    if len(time_profile) < 3 or len(freq_profile) < 3:
        return {"score": 0.5, "interpretation": "insufficient_data", "components": {}}

    # 1. TEMPORAL ENTROPY
    if np.std(time_profile) > 1e-10:
        hist, _ = np.histogram(time_profile, bins=min(10, len(time_profile)//2), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        temporal_entropy = -np.sum(hist * np.log2(hist)) / np.log2(len(hist))
    else:
        temporal_entropy = 1.0
    entropy_score = 1 - temporal_entropy

    # 2. BURST SHARPNESS
    if len(time_profile) > 1 and np.std(time_profile) > 1e-10:
        diffs = np.diff(time_profile)
        sharpness = np.max(np.abs(diffs)) / (np.std(time_profile) + 1e-10)
        sharpness_score = min(sharpness / 5, 1.0)
    else:
        sharpness_score = 0.0

    # 3. SPECTRAL SMOOTHNESS
    if len(freq_profile) > 1 and np.std(freq_profile) > 1e-10:
        diffs = np.diff(freq_profile)
        smoothness = 1 / (1 + np.std(diffs) / (np.std(freq_profile) + 1e-10))
        smoothness_score = smoothness
    else:
        smoothness_score = 0.5

    # 4. SIGNAL-TO-NOISE PROXY
    if np.std(time_profile) > 1e-10:
        snr_proxy = (np.max(time_profile) - np.mean(time_profile)) / np.std(time_profile)
        snr_score = min(snr_proxy / 10, 1.0)
    else:
        snr_score = 0.0

    # 5. PEAK CONCENTRATION (how localized is the signal?)
    if np.max(time_profile) > 0:
        normalized = time_profile / np.max(time_profile)
        concentration = 1 - (np.sum(normalized > 0.5) / len(normalized))
        concentration_score = concentration
    else:
        concentration_score = 0.5

    # Combine with weights
    weights = {
        "entropy": 0.25,
        "sharpness": 0.20,
        "smoothness": 0.20,
        "snr_proxy": 0.20,
        "concentration": 0.15,
    }

    combined_score = (
        weights["entropy"] * entropy_score +
        weights["sharpness"] * sharpness_score +
        weights["smoothness"] * smoothness_score +
        weights["snr_proxy"] * snr_score +
        weights["concentration"] * concentration_score
    )

    if combined_score > 0.7:
        interpretation = "highly_structured"
    elif combined_score > 0.5:
        interpretation = "moderately_structured"
    elif combined_score > 0.3:
        interpretation = "weakly_structured"
    else:
        interpretation = "noise_like"

    return {
        "score": float(combined_score),
        "interpretation": interpretation,
        "components": {
            "entropy_score": float(entropy_score),
            "sharpness_score": float(sharpness_score),
            "smoothness_score": float(smoothness_score),
            "snr_proxy_score": float(snr_score),
            "concentration_score": float(concentration_score),
        },
        "raw_values": {
            "temporal_entropy": float(temporal_entropy),
            "peak_snr": float(snr_proxy) if np.std(time_profile) > 1e-10 else 0,
        }
    }


def load_wow_signal():
    """Load the Wow! signal data from IDL save file."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"

    if not data_path.exists():
        return None, "File not found"

    # Load IDL save file
    wow_raw = readsav(str(data_path))

    # Data is nested inside 'oseti' record
    oseti = wow_raw['oseti'][0]

    # Extract the SNR matrix - shape is [82, 50] = [time, channels]
    snr_matrix = np.array(oseti['snr'])

    # Extract flux estimates
    flux_matrix = np.array(oseti['flux']) if 'flux' in oseti.dtype.names else None

    # Get frequency information
    freq_chan = np.array(oseti['freq_chan']) if 'freq_chan' in oseti.dtype.names else None

    return {
        'snr': snr_matrix,
        'flux': flux_matrix,
        'freq_chan': freq_chan,
        'oseti': oseti,
    }, None


def find_wow_peak(snr_matrix):
    """Find the location of the Wow! signal peak in the data."""
    # The Wow! signal is the famous "6EQUJ5" sequence
    # Find the maximum SNR value
    max_val = np.nanmax(snr_matrix)
    max_loc = np.unravel_index(np.nanargmax(snr_matrix), snr_matrix.shape)

    return max_loc, max_val


def run_experiment():
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 21: Famous Space Signals Analysis")
    print("=" * 60)
    print("\nApplying information structure detector to historical signals.")

    # Load the Wow! signal
    print("\n" + "=" * 40)
    print("THE WOW! SIGNAL (August 15, 1977)")
    print("=" * 40)

    wow_result, error = load_wow_signal()

    if error:
        print(f"Error loading Wow! signal: {error}")
        return

    snr_matrix = wow_result['snr']  # Shape is [82, 50] = [time, channels]
    print(f"\nData shape: {snr_matrix.shape} (time samples × channels)")
    print(f"SNR range: [{np.nanmin(snr_matrix):.1f}, {np.nanmax(snr_matrix):.1f}]")

    # Find the peak
    peak_loc, peak_val = find_wow_peak(snr_matrix)
    peak_time, peak_chan = peak_loc
    print(f"\nPeak location: time sample {peak_time}, channel {peak_chan}")
    print(f"Peak SNR value: {peak_val:.1f}")

    # Show the famous "6EQUJ5" sequence
    print("\nThe '6EQUJ5' sequence (base-36 SNR values):")
    print("  6=6, E=14, Q=26, U=30, J=19, 5=5")

    # Show actual values around the peak
    print("\nActual values around peak (peak channel time series):")
    wow_time_series = snr_matrix[:, peak_chan]
    for t in range(max(0, peak_time-3), min(len(wow_time_series), peak_time+4)):
        val = wow_time_series[t]
        if val < 10:
            char = str(int(val))
        else:
            char = chr(int(val) - 10 + ord('A'))
        marker = ' <-- PEAK ("U")' if t == peak_time else ''
        print(f"  t={t}: {val:5.1f} = \"{char}\"{marker}")

    # Extract the Wow! signal region (around the peak)
    time_window = 10  # samples around peak
    chan_window = 5   # channels around peak

    t_start = max(0, peak_time - time_window)
    t_end = min(snr_matrix.shape[0], peak_time + time_window + 1)
    c_start = max(0, peak_chan - chan_window)
    c_end = min(snr_matrix.shape[1], peak_chan + chan_window + 1)

    wow_region = snr_matrix[t_start:t_end, c_start:c_end]
    print(f"\nExtracted region shape: {wow_region.shape}")

    # Compute information score for the Wow! signal region
    print("\n" + "=" * 40)
    print("INFORMATION STRUCTURE ANALYSIS")
    print("=" * 40)

    wow_score = compute_information_score(wow_region)

    print(f"\nWow! Signal Information Score: {wow_score['score']:.3f}")
    print(f"Interpretation: {wow_score['interpretation']}")
    print("\nComponent scores:")
    for comp, val in wow_score['components'].items():
        print(f"  {comp}: {val:.3f}")

    # Compare to full observation (background)
    print("\n" + "=" * 40)
    print("BACKGROUND COMPARISON")
    print("=" * 40)

    # Exclude the Wow! region and analyze the rest
    background_mask = np.ones_like(snr_matrix, dtype=bool)
    background_mask[t_start:t_end, c_start:c_end] = False

    # Sample random regions from background
    n_background_samples = 20
    background_scores = []

    for _ in range(n_background_samples):
        # Random time and channel windows
        rand_t = np.random.randint(0, snr_matrix.shape[0] - time_window*2)
        rand_c = np.random.randint(0, snr_matrix.shape[1] - chan_window*2)

        bg_region = snr_matrix[rand_t:rand_t+time_window*2, rand_c:rand_c+chan_window*2]
        bg_score = compute_information_score(bg_region)
        background_scores.append(bg_score['score'])

    bg_mean = np.mean(background_scores)
    bg_std = np.std(background_scores)

    print(f"\nBackground regions (n={n_background_samples}):")
    print(f"  Mean score: {bg_mean:.3f} ± {bg_std:.3f}")
    print(f"  Range: [{np.min(background_scores):.3f}, {np.max(background_scores):.3f}]")

    # Z-score of Wow! signal vs background
    z_score = (wow_score['score'] - bg_mean) / (bg_std + 1e-10)
    print(f"\nWow! signal z-score vs background: {z_score:.2f}")

    # Statistical test
    t_stat, p_val = stats.ttest_1samp(background_scores, wow_score['score'])
    print(f"t-test (Wow! vs background): t={t_stat:.2f}, p={p_val:.4f}")

    print("\n" + "=" * 40)
    print("COMPARISON WITH FRB BASELINE")
    print("=" * 40)

    # Load FRB detector results for comparison
    frb_results_path = results_dir / "exp20_results.json"
    if frb_results_path.exists():
        with open(frb_results_path) as f:
            frb_data = json.load(f)

        frb_scores = [s['score'] for s in frb_data['frb_scores']]
        frb_mean = np.mean(frb_scores)
        frb_std = np.std(frb_scores)

        print(f"\nFRB baseline (n={len(frb_scores)}):")
        print(f"  Mean score: {frb_mean:.3f} ± {frb_std:.3f}")

        print(f"\nWow! signal score: {wow_score['score']:.3f}")

        # Where does Wow! fall in FRB distribution?
        wow_percentile = np.sum(np.array(frb_scores) < wow_score['score']) / len(frb_scores) * 100
        print(f"Wow! signal percentile in FRB distribution: {wow_percentile:.0f}%")

        # Gaussian noise baseline
        noise_mean = frb_data['noise_results']['gaussian']['mean']
        noise_std = frb_data['noise_results']['gaussian']['std']
        print(f"\nGaussian noise baseline: {noise_mean:.3f} ± {noise_std:.3f}")

        wow_vs_noise_z = (wow_score['score'] - noise_mean) / (noise_std + 1e-10)
        print(f"Wow! z-score vs Gaussian noise: {wow_vs_noise_z:.2f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(f"""
THE WOW! SIGNAL ANALYSIS:

Information Score: {wow_score['score']:.3f} ({wow_score['interpretation']})

Key findings:
- Peak SNR: {peak_val:.0f} (the famous '{int(peak_val)}' = 'U' in base-36)
- Z-score vs local background: {z_score:.1f}σ
- Interpretation: {wow_score['interpretation']}

The Wow! signal shows {'SIGNIFICANT' if z_score > 2 else 'MODERATE' if z_score > 1 else 'WEAK'}
information structure compared to surrounding noise.

{'This is consistent with a genuine signal (natural or artificial).' if z_score > 2 else
 'The signal stands out but structure is not conclusively different from noise.'}
""")

    # Save results
    results = {
        "experiment": "exp21_famous_signals",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "data_shape": [int(x) for x in snr_matrix.shape],
            "peak_location": [int(x) for x in peak_loc],
            "peak_snr": float(peak_val),
            "information_score": wow_score,
            "background_comparison": {
                "mean": float(bg_mean),
                "std": float(bg_std),
                "z_score": float(z_score),
            },
        },
    }

    output_path = results_dir / "exp21_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
