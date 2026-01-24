#!/usr/bin/env python3
"""Experiment 1: Vrillon Spectrogram Manifold Analysis

Apply the same manifold analysis pipeline used on Wow! signal to the
Vrillon broadcast spectrogram. Test hypothesis that both signals encode
the same geometric constants (π, e) at similar precision.

Usage:
    poetry run python experiments/astronomy/vrillon_manifold_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from geometry_utils import (
    load_vrillon_spectrogram,
    load_wow_signal,
    compute_all_metrics,
    print_metrics_report,
    compare_metrics,
    find_closest_constant,
    PI, E, PHI, SQRT2,
)


def analyze_vrillon_manifold():
    """Run full manifold analysis on Vrillon spectrogram."""
    print("=" * 70)
    print("EXPERIMENT 1: VRILLON SPECTROGRAM MANIFOLD ANALYSIS")
    print("=" * 70)
    print()
    print("Hypothesis: Vrillon spectrogram will show same geometric constants")
    print("           (π, e) at similar precision as Wow! signal.")
    print()

    # Load Vrillon spectrogram (message portion only)
    print("Loading Vrillon broadcast spectrogram...")
    print("  - Message portion: 10.5s to 345.0s")

    vrillon_matrix, sample_rate = load_vrillon_spectrogram()
    print(f"  - Spectrogram shape: {vrillon_matrix.shape}")
    print(f"    ({vrillon_matrix.shape[0]} time bins × {vrillon_matrix.shape[1]} frequency bins)")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Non-zero entries: {np.count_nonzero(vrillon_matrix)}")
    print(f"  - Value range: [{vrillon_matrix.min():.4f}, {vrillon_matrix.max():.4f}]")

    # Load Wow! for comparison
    print("\nLoading Wow! signal for comparison...")
    wow_matrix = load_wow_signal()
    print(f"  - Wow! shape: {wow_matrix.shape}")

    # Compute metrics for Vrillon
    print("\nComputing Vrillon manifold metrics...")
    vrillon_metrics = compute_all_metrics(vrillon_matrix)
    print_metrics_report(vrillon_metrics, "VRILLON BROADCAST MANIFOLD METRICS")

    # Compute metrics for Wow! (for comparison)
    print("\nComputing Wow! manifold metrics for comparison...")
    wow_metrics = compute_all_metrics(wow_matrix)

    # Compare the two
    compare_metrics(wow_metrics, vrillon_metrics, "Wow!", "Vrillon")

    # Detailed constant analysis
    print("\n" + "=" * 70)
    print("GEOMETRIC CONSTANT ENCODING ANALYSIS")
    print("=" * 70)

    findings = []

    # Check Renyi rank
    print(f"\n  Renyi Effective Rank:")
    print(f"    Wow!:    {wow_metrics.renyi_rank:.6f} → {wow_metrics.renyi_match.name} ({wow_metrics.renyi_match.error_percent:.4f}% error)")
    print(f"    Vrillon: {vrillon_metrics.renyi_rank:.6f} → {vrillon_metrics.renyi_match.name} ({vrillon_metrics.renyi_match.error_percent:.4f}% error)")

    if vrillon_metrics.renyi_match.is_significant:
        findings.append(("Vrillon Renyi rank", vrillon_metrics.renyi_rank, vrillon_metrics.renyi_match))

    # Check spectral PR
    print(f"\n  Spectral Participation Ratio:")
    print(f"    Wow!:    {wow_metrics.spectral_pr:.6f} → {wow_metrics.spectral_pr_match.name} ({wow_metrics.spectral_pr_match.error_percent:.4f}% error)")
    print(f"    Vrillon: {vrillon_metrics.spectral_pr:.6f} → {vrillon_metrics.spectral_pr_match.name} ({vrillon_metrics.spectral_pr_match.error_percent:.4f}% error)")

    if vrillon_metrics.spectral_pr_match.is_significant:
        findings.append(("Vrillon Spectral PR", vrillon_metrics.spectral_pr, vrillon_metrics.spectral_pr_match))

    # Check specific constants with different thresholds
    print(f"\n  Direct Constant Comparisons:")

    for name, value in [
        ("Vrillon Renyi", vrillon_metrics.renyi_rank),
        ("Vrillon Spectral PR", vrillon_metrics.spectral_pr),
        ("Vrillon ID (time)", vrillon_metrics.intrinsic_dim_time),
        ("Vrillon ID (freq)", vrillon_metrics.intrinsic_dim_freq),
        ("Vrillon Geo/Euc", vrillon_metrics.mean_geo_euc_ratio),
    ]:
        if value is not None:
            for const_name, const_val in [("π", PI), ("e", E), ("φ", PHI), ("√2", SQRT2)]:
                error = abs(value - const_val) / const_val * 100
                if error < 10:  # Within 10%
                    marker = "✓✓" if error < 3 else "✓" if error < 5 else ""
                    print(f"    {name}: {value:.6f} vs {const_name} = {const_val:.6f} → {error:.4f}% error {marker}")

    # SV ratio analysis
    print(f"\n  Singular Value Ratio Comparison:")
    print(f"    {'Ratio':<15} {'Wow!':<15} {'Vrillon':<15} {'Same const?':<15}")
    print(f"    {'-' * 60}")

    for i, (w_ratio, v_ratio) in enumerate(zip(wow_metrics.sv_ratios[:5], vrillon_metrics.sv_ratios[:5])):
        w_match = find_closest_constant(w_ratio)
        v_match = find_closest_constant(v_ratio)
        same = "YES ✓" if w_match.name == v_match.name else "no"
        print(f"    S[{i}]/S[{i+1}]        {w_ratio:<15.4f} {v_ratio:<15.4f} {same:<15}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 70)

    # Count significant matches
    significant_matches = []
    if vrillon_metrics.renyi_match.is_significant:
        significant_matches.append(f"Renyi rank ≈ {vrillon_metrics.renyi_match.name}")
    if vrillon_metrics.spectral_pr_match.is_significant:
        significant_matches.append(f"Spectral PR ≈ {vrillon_metrics.spectral_pr_match.name}")

    if significant_matches:
        print(f"\n  ✓ SIGNIFICANT FINDINGS:")
        for match in significant_matches:
            print(f"    - {match}")
    else:
        print(f"\n  No significant matches (< 5% error) found in primary metrics.")
        print(f"  Closest matches:")
        print(f"    - Renyi: {vrillon_metrics.renyi_match.name} ({vrillon_metrics.renyi_match.error_percent:.2f}% error)")
        print(f"    - Spectral PR: {vrillon_metrics.spectral_pr_match.name} ({vrillon_metrics.spectral_pr_match.error_percent:.2f}% error)")

    # Cross-signal pattern
    print(f"\n  Cross-signal pattern check:")
    # Both encode e?
    wow_e_error = abs(wow_metrics.renyi_rank - E) / E * 100
    vrillon_e_error = abs(vrillon_metrics.renyi_rank - E) / E * 100
    print(f"    Wow! Renyi vs e: {wow_e_error:.4f}% error")
    print(f"    Vrillon Renyi vs e: {vrillon_e_error:.4f}% error")

    if wow_e_error < 5 and vrillon_e_error < 5:
        print(f"\n    ✓✓ BOTH SIGNALS ENCODE e WITH < 5% ERROR")

    # Both encode π in spectral PR?
    wow_pi_error = abs(wow_metrics.spectral_pr - PI) / PI * 100
    vrillon_pi_error = abs(vrillon_metrics.spectral_pr - PI) / PI * 100
    print(f"\n    Wow! Spectral PR vs π: {wow_pi_error:.4f}% error")
    print(f"    Vrillon Spectral PR vs π: {vrillon_pi_error:.4f}% error")

    if wow_pi_error < 5 and vrillon_pi_error < 5:
        print(f"\n    ✓✓ BOTH SIGNALS ENCODE π IN SPECTRAL PR WITH < 5% ERROR")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return vrillon_metrics, wow_metrics


if __name__ == "__main__":
    analyze_vrillon_manifold()
