#!/usr/bin/env python3
"""Experiment 29: Baseline Comparison.

Exp28 found prime number patterns in the Wow! signal's modes.
But would RANDOM noise show similar patterns?

We need a baseline:
1. Generate random noise with same shape
2. Extract principal components
3. Run same pattern searches
4. Compare: Is Wow! signal unusual?

Usage:
    poetry run python experiments/astronomy/exp29_baseline_comparison.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav
from scipy.linalg import svd
from scipy.signal import find_peaks

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def extract_mode_patterns(matrix: np.ndarray, n_modes: int = 10) -> list:
    """Extract the principal component patterns."""
    matrix = np.nan_to_num(matrix, nan=0.0)
    if np.std(matrix) > 1e-10:
        matrix_norm = (matrix - np.mean(matrix)) / np.std(matrix)
    else:
        return None

    U, s, Vh = svd(matrix_norm, full_matrices=False)

    modes = []
    for i in range(min(n_modes, len(s))):
        modes.append({
            "time": U[:, i],
            "freq": Vh[i, :],
            "energy": (s[i]**2) / (np.sum(s**2)),
        })

    return modes


def count_prime_spacings(pattern: np.ndarray) -> float:
    """Count what fraction of peak spacings are prime numbers."""
    p = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
    peaks, _ = find_peaks(p, height=0.3)

    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

    if len(peaks) >= 2:
        spacings = np.diff(peaks)
        prime_matches = sum(1 for s in spacings if s in primes)
        return prime_matches / len(spacings)
    return 0.0


def count_constant_ratios(pattern: np.ndarray) -> int:
    """Count ratios matching mathematical constants."""
    constants = [3.14159, 2.71828, 1.61803, 1.41421, 1.73205]
    count = 0

    for i in range(len(pattern) - 1):
        if abs(pattern[i]) > 1e-10:
            ratio = abs(pattern[i+1] / pattern[i])
            for c in constants:
                if abs(ratio - c) < 0.1 or abs(ratio - 1/c) < 0.1:
                    count += 1
                    break
    return count


def analyze_signal(matrix: np.ndarray) -> dict:
    """Analyze a signal for pattern metrics."""
    modes = extract_mode_patterns(matrix, n_modes=10)
    if modes is None:
        return None

    total_prime_score = 0
    total_constant_count = 0

    for mode in modes:
        # Time pattern
        total_prime_score += count_prime_spacings(mode["time"])
        total_constant_count += count_constant_ratios(mode["time"])

        # Freq pattern
        total_prime_score += count_prime_spacings(mode["freq"])
        total_constant_count += count_constant_ratios(mode["freq"])

    avg_prime_score = total_prime_score / (2 * len(modes))

    return {
        "avg_prime_spacing_match": avg_prime_score,
        "total_constant_ratios": total_constant_count,
    }


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 29: Baseline Comparison")
    print("=" * 60)
    print("\nComparing Wow! signal patterns to random noise baseline.")

    # Load the Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    # Analyze Wow! signal
    print("\n" + "=" * 40)
    print("WOW! SIGNAL ANALYSIS")
    print("=" * 40)

    wow_metrics = analyze_signal(snr_matrix)
    print(f"\n  Prime spacing match: {wow_metrics['avg_prime_spacing_match']:.1%}")
    print(f"  Constant ratio matches: {wow_metrics['total_constant_ratios']}")

    # Generate random baselines
    print("\n" + "=" * 40)
    print("RANDOM NOISE BASELINE")
    print("=" * 40)

    n_baselines = 100
    print(f"\nGenerating {n_baselines} random noise samples...")

    noise_prime_scores = []
    noise_constant_counts = []

    for i in range(n_baselines):
        noise = np.random.randn(*snr_matrix.shape)
        metrics = analyze_signal(noise)
        if metrics:
            noise_prime_scores.append(metrics["avg_prime_spacing_match"])
            noise_constant_counts.append(metrics["total_constant_ratios"])

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{n_baselines}...")

    noise_prime_mean = np.mean(noise_prime_scores)
    noise_prime_std = np.std(noise_prime_scores)
    noise_const_mean = np.mean(noise_constant_counts)
    noise_const_std = np.std(noise_constant_counts)

    print(f"\nRandom noise baseline (n={len(noise_prime_scores)}):")
    print(f"  Prime spacing match: {noise_prime_mean:.1%} ± {noise_prime_std:.1%}")
    print(f"  Constant ratio matches: {noise_const_mean:.0f} ± {noise_const_std:.0f}")

    # Statistical comparison
    print("\n" + "=" * 40)
    print("STATISTICAL COMPARISON")
    print("=" * 40)

    z_prime = (wow_metrics["avg_prime_spacing_match"] - noise_prime_mean) / (noise_prime_std + 1e-10)
    z_const = (wow_metrics["total_constant_ratios"] - noise_const_mean) / (noise_const_std + 1e-10)

    percentile_prime = np.sum(np.array(noise_prime_scores) < wow_metrics["avg_prime_spacing_match"]) / len(noise_prime_scores) * 100
    percentile_const = np.sum(np.array(noise_constant_counts) < wow_metrics["total_constant_ratios"]) / len(noise_constant_counts) * 100

    print(f"\nPRIME SPACING PATTERNS:")
    print(f"  Wow!: {wow_metrics['avg_prime_spacing_match']:.1%}")
    print(f"  Noise: {noise_prime_mean:.1%} ± {noise_prime_std:.1%}")
    print(f"  Z-score: {z_prime:.2f}σ")
    print(f"  Percentile: {percentile_prime:.0f}%")

    print(f"\nMATHEMATICAL CONSTANT RATIOS:")
    print(f"  Wow!: {wow_metrics['total_constant_ratios']}")
    print(f"  Noise: {noise_const_mean:.0f} ± {noise_const_std:.0f}")
    print(f"  Z-score: {z_const:.2f}σ")
    print(f"  Percentile: {percentile_const:.0f}%")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    is_significant_prime = z_prime > 2 or z_prime < -2
    is_significant_const = z_const > 2 or z_const < -2

    print(f"""
COMPARISON TO RANDOM BASELINE:

Prime Number Spacings:
  {'✓ SIGNIFICANT' if is_significant_prime else '✗ NOT SIGNIFICANT'} at {z_prime:.1f}σ
  Wow! is in the {percentile_prime:.0f}th percentile

Mathematical Constant Ratios:
  {'✓ SIGNIFICANT' if is_significant_const else '✗ NOT SIGNIFICANT'} at {z_const:.1f}σ
  Wow! is in the {percentile_const:.0f}th percentile
""")

    if is_significant_prime or is_significant_const:
        print("""
THE PATTERNS ARE STATISTICALLY UNUSUAL.

The Wow! signal's principal components contain mathematical structure
that differs significantly from random noise. This could indicate:

1. Genuine encoding (intentional or natural)
2. Systematic effects from the telescope/recording system
3. Properties of the signal source

The anomaly is GEOMETRIC - independent of any specific interpretation.
""")
    else:
        print("""
THE PATTERNS ARE NOT STATISTICALLY UNUSUAL.

The mathematical patterns found in the Wow! signal's modes are
comparable to what we'd expect from random noise. The search
space is large enough that coincidental matches are expected.

However, the COMPRESSION (low effective rank) remains significant.
The structure exists; it just doesn't show clear mathematical encoding.
""")

    # Save results
    results = {
        "experiment": "exp29_baseline_comparison",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "prime_spacing_match": float(wow_metrics["avg_prime_spacing_match"]),
            "constant_ratio_matches": int(wow_metrics["total_constant_ratios"]),
        },
        "noise_baseline": {
            "n_samples": len(noise_prime_scores),
            "prime_spacing": {
                "mean": float(noise_prime_mean),
                "std": float(noise_prime_std),
            },
            "constant_ratios": {
                "mean": float(noise_const_mean),
                "std": float(noise_const_std),
            },
        },
        "comparison": {
            "z_score_prime": float(z_prime),
            "z_score_const": float(z_const),
            "percentile_prime": float(percentile_prime),
            "percentile_const": float(percentile_const),
            "significant_prime": bool(is_significant_prime),
            "significant_const": bool(is_significant_const),
        },
    }

    output_path = results_dir / "exp29_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
