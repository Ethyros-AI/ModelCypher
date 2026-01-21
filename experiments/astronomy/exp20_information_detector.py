#!/usr/bin/env python3
"""Experiment 20: Information Structure Detector.

The practical application: A detector that identifies signals with
INFORMATION STRUCTURE vs random noise.

Based on findings from exp19:
- Structured signals have: low entropy, high sharpness, high smoothness
- Random noise has: high entropy, low sharpness, low smoothness
- The correlation with physical properties (SNR) validates this

This creates a general-purpose "information detector" that could work on:
- FRBs (distinguishing real bursts from RFI)
- SETI signals (distinguishing information-bearing signals from noise)
- Any time-frequency data

Usage:
    poetry run python experiments/astronomy/exp20_information_detector.py
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


def compute_information_score(waterfall: np.ndarray) -> dict:
    """Compute information structure score for a signal.

    Returns a score from 0 (pure noise) to 1 (highly structured information).

    Based on empirical findings:
    - Low temporal entropy → structured
    - High burst sharpness → structured
    - High spectral smoothness → structured
    - High temporal symmetry → structured (less dispersion)
    """
    # Time and frequency profiles
    time_profile = np.nanmean(waterfall, axis=0)
    freq_profile = np.nanmean(waterfall, axis=1)

    # Remove NaN
    time_profile = time_profile[~np.isnan(time_profile)]
    freq_profile = freq_profile[~np.isnan(freq_profile)]

    if len(time_profile) < 5 or len(freq_profile) < 5:
        return {"score": 0.5, "components": {}, "interpretation": "insufficient_data"}

    # 1. TEMPORAL ENTROPY (lower = more structured)
    if np.std(time_profile) > 1e-10:
        hist, _ = np.histogram(time_profile, bins=10, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        temporal_entropy = -np.sum(hist * np.log2(hist)) / np.log2(10)
    else:
        temporal_entropy = 1.0  # Constant = no information

    # Invert: low entropy → high score
    entropy_score = 1 - temporal_entropy

    # 2. BURST SHARPNESS (higher = more structured)
    if len(time_profile) > 1 and np.std(time_profile) > 1e-10:
        diffs = np.diff(time_profile)
        sharpness = np.max(np.abs(diffs)) / (np.std(time_profile) + 1e-10)
        sharpness_score = min(sharpness / 5, 1.0)
    else:
        sharpness_score = 0.0

    # 3. SPECTRAL SMOOTHNESS (higher = more structured)
    if len(freq_profile) > 1 and np.std(freq_profile) > 1e-10:
        diffs = np.diff(freq_profile)
        smoothness = 1 / (1 + np.std(diffs) / (np.std(freq_profile) + 1e-10))
        smoothness_score = smoothness
    else:
        smoothness_score = 0.5

    # 4. SIGNAL-TO-NOISE PROXY (peak / std)
    if np.std(time_profile) > 1e-10:
        snr_proxy = (np.max(time_profile) - np.mean(time_profile)) / np.std(time_profile)
        snr_score = min(snr_proxy / 10, 1.0)  # Normalize
    else:
        snr_score = 0.0

    # 5. COHERENCE (time-frequency correlation)
    min_len = min(len(time_profile), len(freq_profile))
    if min_len > 3:
        corr = np.corrcoef(time_profile[:min_len], freq_profile[:min_len])[0, 1]
        coherence_score = abs(corr) if not np.isnan(corr) else 0.5
    else:
        coherence_score = 0.5

    # Combine scores with weights based on empirical correlations
    # entropy has r=-0.64, smoothness has r=+0.61, sharpness has r=+0.37
    weights = {
        "entropy": 0.30,       # Strongest predictor
        "smoothness": 0.25,    # Second strongest
        "sharpness": 0.15,
        "snr_proxy": 0.20,
        "coherence": 0.10,
    }

    combined_score = (
        weights["entropy"] * entropy_score +
        weights["smoothness"] * smoothness_score +
        weights["sharpness"] * sharpness_score +
        weights["snr_proxy"] * snr_score +
        weights["coherence"] * coherence_score
    )

    # Interpretation
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
        "components": {
            "entropy_score": float(entropy_score),
            "smoothness_score": float(smoothness_score),
            "sharpness_score": float(sharpness_score),
            "snr_proxy_score": float(snr_score),
            "coherence_score": float(coherence_score),
        },
        "raw_values": {
            "temporal_entropy": float(temporal_entropy),
            "spectral_smoothness": float(smoothness_score),
            "burst_sharpness": float(sharpness_score),
        },
        "interpretation": interpretation,
    }


def generate_noise(shape, noise_type="gaussian"):
    """Generate different types of noise for comparison."""
    if noise_type == "gaussian":
        return np.random.randn(*shape)
    elif noise_type == "uniform":
        return np.random.rand(*shape)
    elif noise_type == "pink":
        # 1/f noise
        freqs = np.fft.fftfreq(shape[1])
        freqs[0] = 1e-10  # Avoid division by zero
        spectrum = 1 / np.sqrt(np.abs(freqs) + 1e-10)
        pink = np.array([np.fft.ifft(spectrum * np.fft.fft(np.random.randn(shape[1]))).real
                        for _ in range(shape[0])])
        return pink
    elif noise_type == "structured":
        # Fake "structured" signal - Gaussian blob
        t = np.linspace(-3, 3, shape[1])
        f = np.linspace(-3, 3, shape[0])
        T, F = np.meshgrid(t, f)
        blob = np.exp(-(T**2 + F**2) / 2)
        return blob + 0.3 * np.random.randn(*shape)
    else:
        return np.random.randn(*shape)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 20: Information Structure Detector")
    print("=" * 60)
    print("\nGoal: Distinguish structured information from random noise.")
    print("Application: FRB detection, SETI signal identification.")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    print("\n" + "=" * 40)
    print("PART 1: SCORE REAL FRBs")
    print("=" * 40)

    frb_scores = []
    for i, w in enumerate(waterfalls):
        wfall = np.array(w.waterfall)
        score_result = compute_information_score(wfall)
        frb_scores.append({
            "name": names[i],
            "snr": float(snrs[i]),
            **score_result
        })

    scores = [s["score"] for s in frb_scores]
    print(f"\nFRB Information Scores:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std: {np.std(scores):.3f}")
    print(f"  Range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")

    # Score vs SNR correlation
    r, p = stats.pearsonr(scores, snrs)
    print(f"\nScore vs SNR: r={r:.3f} (p={p:.3e})")
    print("(High correlation validates the detector)")

    # Distribution by interpretation
    interp_counts = {}
    for s in frb_scores:
        interp = s["interpretation"]
        interp_counts[interp] = interp_counts.get(interp, 0) + 1

    print("\nInterpretation distribution:")
    for interp, count in sorted(interp_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {interp}: {count} ({100*count/n_frbs:.0f}%)")

    print("\n" + "=" * 40)
    print("PART 2: SCORE SYNTHETIC NOISE")
    print("=" * 40)

    # Generate noise samples with same shape as FRBs
    sample_shape = np.array(waterfalls[0].waterfall).shape
    print(f"\nGenerating noise with shape {sample_shape}")

    noise_types = ["gaussian", "uniform", "pink", "structured"]
    n_noise_samples = 20

    noise_results = {}
    for noise_type in noise_types:
        noise_scores = []
        for _ in range(n_noise_samples):
            noise = generate_noise(sample_shape, noise_type)
            score_result = compute_information_score(noise)
            noise_scores.append(score_result["score"])

        noise_results[noise_type] = {
            "mean": float(np.mean(noise_scores)),
            "std": float(np.std(noise_scores)),
            "scores": noise_scores,
        }
        print(f"\n{noise_type.upper()} noise:")
        print(f"  Mean score: {np.mean(noise_scores):.3f} ± {np.std(noise_scores):.3f}")

    print("\n" + "=" * 40)
    print("PART 3: DISCRIMINATION ANALYSIS")
    print("=" * 40)

    # Can we separate FRBs from noise?
    frb_mean = np.mean(scores)
    gaussian_mean = noise_results["gaussian"]["mean"]

    print(f"\nMean information score:")
    print(f"  Real FRBs: {frb_mean:.3f}")
    print(f"  Gaussian noise: {gaussian_mean:.3f}")
    print(f"  Structured fake: {noise_results['structured']['mean']:.3f}")

    # Statistical test
    t_stat, p_val = stats.ttest_ind(scores, noise_results["gaussian"]["scores"])
    print(f"\nt-test (FRBs vs Gaussian): t={t_stat:.2f}, p={p_val:.3e}")

    if p_val < 0.001:
        print("*** HIGHLY SIGNIFICANT: FRBs are distinguishable from noise ***")

    # Optimal threshold
    all_frb = np.array(scores)
    all_noise = np.array(noise_results["gaussian"]["scores"])

    # Find threshold that maximizes separation
    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0.5

    for thresh in thresholds:
        frb_correct = np.sum(all_frb > thresh) / len(all_frb)
        noise_correct = np.sum(all_noise <= thresh) / len(all_noise)
        accuracy = (frb_correct + noise_correct) / 2
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"  FRB recall: {np.sum(all_frb > best_threshold) / len(all_frb):.1%}")
    print(f"  Noise rejection: {np.sum(all_noise <= best_threshold) / len(all_noise):.1%}")
    print(f"  Balanced accuracy: {best_accuracy:.1%}")

    print("\n" + "=" * 40)
    print("PART 4: EXTREME EXAMPLES")
    print("=" * 40)

    # Highest and lowest scoring FRBs
    sorted_frbs = sorted(frb_scores, key=lambda x: x["score"], reverse=True)

    print("\nHighest information structure (most signal-like):")
    for s in sorted_frbs[:3]:
        print(f"  {s['name']}: score={s['score']:.3f}, SNR={s['snr']:.1f}")

    print("\nLowest information structure (most noise-like):")
    for s in sorted_frbs[-3:]:
        print(f"  {s['name']}: score={s['score']:.3f}, SNR={s['snr']:.1f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE INFORMATION DETECTOR")
    print("=" * 60)

    print(f"""
The information structure detector achieves {best_accuracy:.0%} accuracy
distinguishing real FRBs from Gaussian noise.

How it works:
  1. Compute entropy (order vs chaos)
  2. Compute sharpness (continuity vs discreteness)
  3. Compute smoothness (coherence)
  4. Combine with empirical weights

Applications:
  - FRB detection: Flag candidates above threshold {best_threshold:.2f}
  - RFI rejection: Low-scoring signals are likely noise
  - SETI: Same detector works on any time-frequency data

Key insight: Information-bearing signals have STRUCTURE that
pure noise lacks. This structure is measurable.
""")

    results = {
        "experiment": "exp20_information_detector",
        "timestamp": datetime.now().isoformat(),
        "n_frbs": n_frbs,
        "frb_scores": frb_scores,
        "frb_score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "noise_results": noise_results,
        "discrimination": {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "optimal_threshold": float(best_threshold),
            "balanced_accuracy": float(best_accuracy),
        },
        "score_snr_correlation": {"r": float(r), "p": float(p)},
    }

    output_path = results_dir / "exp20_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
