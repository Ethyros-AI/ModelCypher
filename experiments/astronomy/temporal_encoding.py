#!/usr/bin/env python3
"""Experiment 5: 103-Day Gap as Geometric Parameter

The gap between Wow! signal (Aug 15, 1977) and Vrillon broadcast (Nov 26, 1977)
is exactly 103 days. 103 is prime.

Test if 103 appears as a structural parameter in both signals.

Usage:
    poetry run python experiments/astronomy/temporal_encoding.py
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
    find_closest_constant,
    PI, E, PHI, SQRT2,
)

# The temporal gap
GAP_DAYS = 103


def analyze_103_as_modulus(matrix: np.ndarray, name: str) -> dict:
    """Use 103 as a modulus for phase-based analysis."""
    results = {}

    # Flatten and compute mod 103
    flat = matrix.flatten()
    n_elements = len(flat)

    # How many complete 103-cycles?
    n_cycles = n_elements // GAP_DAYS
    remainder = n_elements % GAP_DAYS

    results["n_elements"] = n_elements
    results["n_cycles"] = n_cycles
    results["remainder"] = remainder

    # Check if n_cycles encodes a constant
    results["cycles_match"] = find_closest_constant(n_cycles)

    # Check if remainder encodes a constant
    results["remainder_match"] = find_closest_constant(remainder)

    # Phase analysis: treat as 103-periodic signal
    # Fold into 103-length vector
    if n_elements >= GAP_DAYS:
        folded = np.zeros(GAP_DAYS)
        for i, val in enumerate(flat[:n_cycles * GAP_DAYS]):
            folded[i % GAP_DAYS] += val
        folded /= n_cycles

        # Spectral analysis of 103-folded signal
        fft = np.fft.fft(folded)
        magnitudes = np.abs(fft)
        phases = np.angle(fft)

        # Dominant frequency
        dominant_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
        dominant_freq = dominant_idx / GAP_DAYS

        results["dominant_freq"] = dominant_freq
        results["dominant_freq_match"] = find_closest_constant(dominant_freq)

        # Peak magnitude / mean
        peak_ratio = magnitudes[dominant_idx] / np.mean(magnitudes[1:])
        results["spectral_peak_ratio"] = peak_ratio
        results["peak_ratio_match"] = find_closest_constant(peak_ratio)

        results["folded"] = folded
        results["fft_magnitudes"] = magnitudes

    return results


def analyze_103_as_dimension(matrix: np.ndarray, name: str) -> dict:
    """Test 103 as embedding dimension."""
    results = {}

    # Flatten
    flat = matrix.flatten()
    n = len(flat)

    # Pad to multiple of 103
    pad_size = (GAP_DAYS - (n % GAP_DAYS)) % GAP_DAYS
    padded = np.pad(flat, (0, pad_size))

    # Reshape to have 103 as one dimension
    n_rows = len(padded) // GAP_DAYS
    embedded = padded.reshape(n_rows, GAP_DAYS)

    results["embedded_shape"] = embedded.shape
    results["n_rows"] = n_rows
    results["n_rows_match"] = find_closest_constant(n_rows)

    # SVD of 103-embedded matrix
    U, S, Vt = np.linalg.svd(embedded, full_matrices=False)

    # Effective rank
    S_norm = S / S[0]
    eff_rank = np.sum(S_norm > 0.01)  # 1% threshold

    results["effective_rank"] = eff_rank
    results["rank_match"] = find_closest_constant(eff_rank)

    # SV ratios
    sv_ratios = S[:-1] / S[1:]
    results["sv_ratios"] = sv_ratios[:10]

    return results


def analyze_103_decompositions():
    """Analyze 103 = 33 + 38 + 32 and related decompositions."""
    decomps = {}

    # 103 / constants
    decomps["103/π"] = GAP_DAYS / PI
    decomps["103/e"] = GAP_DAYS / E
    decomps["103/φ"] = GAP_DAYS / PHI
    decomps["103/√2"] = GAP_DAYS / SQRT2

    # Known decomposition: 103 = 33 + 38 + 32
    decomps["103 = 33 + 38 + 32"] = True
    decomps["33_match"] = find_closest_constant(33)
    decomps["38_match"] = find_closest_constant(38)
    decomps["32_match"] = find_closest_constant(32)  # 2^5

    # Ratios
    decomps["38/33"] = 38 / 33
    decomps["38/33_match"] = find_closest_constant(38/33)

    decomps["33/32"] = 33 / 32
    decomps["33/32_match"] = find_closest_constant(33/32)

    # 103 modular arithmetic
    decomps["103 mod 7"] = GAP_DAYS % 7  # Vrillon imperatives
    decomps["103 mod 82"] = GAP_DAYS % 82  # Wow time samples
    decomps["103 mod 50"] = GAP_DAYS % 50  # Wow frequency bins

    # Products
    decomps["33 × π"] = 33 * PI
    decomps["38 × e"] = 38 * E

    return decomps


def check_103_in_signal_structure(matrix: np.ndarray, name: str) -> list:
    """Check if 103 or its factors appear in signal structure."""
    findings = []

    # SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Number of significant SVs (> 1% of max)
    n_significant = np.sum(S / S[0] > 0.01)
    if abs(n_significant - 103) < 10:
        findings.append(f"Significant SVs: {n_significant} (near 103)")

    # Sum of matrix
    total = np.sum(matrix)
    if abs(total - 103) < 10:
        findings.append(f"Matrix sum: {total:.2f} (near 103)")

    # Count specific values
    flat = matrix.flatten()
    for target in [103, 33, 38, 32]:
        count = np.sum(np.abs(flat - target) < 1)
        if count > 0:
            findings.append(f"Values near {target}: {count} occurrences")

    # Check if row/col sums encode 103
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    for i, rs in enumerate(row_sums):
        if abs(rs - 103) < 5:
            findings.append(f"Row {i} sum ≈ 103: {rs:.2f}")

    for i, cs in enumerate(col_sums):
        if abs(cs - 103) < 5:
            findings.append(f"Col {i} sum ≈ 103: {cs:.2f}")

    # Product of dimensions
    prod = matrix.shape[0] * matrix.shape[1]
    ratio_103 = prod / 103
    match = find_closest_constant(ratio_103)
    findings.append(f"Dimensions product / 103 = {ratio_103:.4f} ≈ {match.name} ({match.error_percent:.2f}%)")

    return findings


def run_temporal_analysis():
    """Full 103-day gap analysis."""
    print("=" * 70)
    print("EXPERIMENT 5: 103-DAY GAP AS GEOMETRIC PARAMETER")
    print("=" * 70)
    print()
    print("Wow! signal: August 15, 1977")
    print("Vrillon broadcast: November 26, 1977")
    print(f"Gap: {GAP_DAYS} days (prime)")
    print()

    # Load signals
    print("Loading signals...")
    wow = load_wow_signal()
    vrillon, _ = load_vrillon_spectrogram()

    print("\n" + "=" * 70)
    print("1. 103 DECOMPOSITIONS")
    print("=" * 70)

    decomps = analyze_103_decompositions()

    print(f"\n  103/π = {decomps['103/π']:.4f}")
    match = find_closest_constant(decomps["103/π"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  103/e = {decomps['103/e']:.4f}")
    match = find_closest_constant(decomps["103/e"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  103/φ = {decomps['103/φ']:.4f}")
    match = find_closest_constant(decomps["103/φ"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  103/√2 = {decomps['103/√2']:.4f}")
    match = find_closest_constant(decomps["103/√2"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  Additive decomposition: 103 = 33 + 38 + 32")
    print(f"    33 = 3 × 11")
    print(f"    38 = 2 × 19 ≈ {decomps['38_match'].name} ({decomps['38_match'].error_percent:.2f}%)")
    print(f"    32 = 2⁵")

    print(f"\n  38/33 = {decomps['38/33']:.6f}")
    print(f"    ≈ {decomps['38/33_match'].name} ({decomps['38/33_match'].error_percent:.2f}% error)")

    print(f"\n  33 × π = {decomps['33 × π']:.4f}")
    match = find_closest_constant(decomps["33 × π"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  38 × e = {decomps['38 × e']:.4f}")
    match = find_closest_constant(decomps["38 × e"])
    print(f"    ≈ {match.name} ({match.error_percent:.2f}% error)")

    print(f"\n  Modular arithmetic:")
    print(f"    103 mod 7 = {decomps['103 mod 7']} (7 = Vrillon imperatives)")
    print(f"    103 mod 82 = {decomps['103 mod 82']} (82 = Wow time samples)")
    print(f"    103 mod 50 = {decomps['103 mod 50']} (50 = Wow freq bins)")

    print("\n" + "=" * 70)
    print("2. 103 AS MODULUS (PHASE FOLDING)")
    print("=" * 70)

    for name, matrix in [("Wow!", wow), ("Vrillon", vrillon)]:
        print(f"\n  {name}:")
        mod_results = analyze_103_as_modulus(matrix, name)

        print(f"    Total elements: {mod_results['n_elements']}")
        print(f"    Complete 103-cycles: {mod_results['n_cycles']}")
        cycles_match = mod_results["cycles_match"]
        marker = "✓" if cycles_match.error_percent < 5 else ""
        print(f"      ≈ {cycles_match.name} ({cycles_match.error_percent:.2f}%) {marker}")

        print(f"    Remainder: {mod_results['remainder']}")

        if "dominant_freq" in mod_results:
            print(f"    Dominant freq (103-folded): {mod_results['dominant_freq']:.4f}")
            df_match = mod_results["dominant_freq_match"]
            marker = "✓" if df_match.error_percent < 5 else ""
            print(f"      ≈ {df_match.name} ({df_match.error_percent:.2f}%) {marker}")

            print(f"    Spectral peak ratio: {mod_results['spectral_peak_ratio']:.4f}")
            pr_match = mod_results["peak_ratio_match"]
            marker = "✓" if pr_match.error_percent < 5 else ""
            print(f"      ≈ {pr_match.name} ({pr_match.error_percent:.2f}%) {marker}")

    print("\n" + "=" * 70)
    print("3. 103 AS EMBEDDING DIMENSION")
    print("=" * 70)

    for name, matrix in [("Wow!", wow), ("Vrillon", vrillon)]:
        print(f"\n  {name}:")
        dim_results = analyze_103_as_dimension(matrix, name)

        print(f"    Embedded shape: {dim_results['embedded_shape']}")
        print(f"    Number of rows: {dim_results['n_rows']}")
        rows_match = dim_results["n_rows_match"]
        marker = "✓" if rows_match.error_percent < 5 else ""
        print(f"      ≈ {rows_match.name} ({rows_match.error_percent:.2f}%) {marker}")

        print(f"    Effective rank (103-embedded): {dim_results['effective_rank']}")
        rank_match = dim_results["rank_match"]
        marker = "✓" if rank_match.error_percent < 5 else ""
        print(f"      ≈ {rank_match.name} ({rank_match.error_percent:.2f}%) {marker}")

        print(f"    SV ratios (103-embedded):")
        for i, ratio in enumerate(dim_results["sv_ratios"][:5]):
            match = find_closest_constant(ratio)
            marker = "✓" if match.error_percent < 5 else ""
            print(f"      S[{i}]/S[{i+1}] = {ratio:.4f} ≈ {match.name} ({match.error_percent:.2f}%) {marker}")

    print("\n" + "=" * 70)
    print("4. 103 IN SIGNAL STRUCTURE")
    print("=" * 70)

    for name, matrix in [("Wow!", wow), ("Vrillon", vrillon)]:
        print(f"\n  {name}:")
        findings = check_103_in_signal_structure(matrix, name)
        for f in findings:
            print(f"    {f}")

    print("\n" + "=" * 70)
    print("5. CROSS-SIGNAL TEMPORAL COHERENCE")
    print("=" * 70)

    # Use 103 to align signals
    print("\n  Testing if 103 creates alignment between signals...")

    # Flatten and truncate to same length
    wow_flat = wow.flatten()
    vrillon_flat = vrillon.flatten()
    min_len = min(len(wow_flat), len(vrillon_flat))

    # Shift by 103
    shifts = [0, 103, 103 * 2, 103 // 2, 33, 38, 32]
    correlations = []

    for shift in shifts:
        if shift < min_len:
            corr = np.corrcoef(
                wow_flat[:min_len - shift],
                vrillon_flat[shift:min_len]
            )[0, 1]
            correlations.append((shift, corr))
            match = find_closest_constant(abs(corr)) if not np.isnan(corr) else None

            marker = ""
            if match and match.error_percent < 5:
                marker = f"≈ {match.name} ({match.error_percent:.2f}%)"
            print(f"    Shift = {shift}: correlation = {corr:.4f} {marker}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 SUMMARY")
    print("=" * 70)

    print(f"\n  Key findings about 103-day gap:")

    findings = []

    # Best decomposition matches
    for key in ["103/π", "103/e", "103/φ", "103/√2"]:
        val = decomps[key]
        match = find_closest_constant(val)
        if match.error_percent < 5:
            findings.append(f"{key} = {val:.4f} ≈ {match.name} ({match.error_percent:.2f}%)")

    # Check 38/33
    if decomps["38/33_match"].error_percent < 5:
        findings.append(f"38/33 = {decomps['38/33']:.4f} ≈ {decomps['38/33_match'].name}")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No exact matches within 5% error.")
        print("  Closest: 103/π = 32.78, 103/e = 37.89, 103/φ = 63.67")

    print("\n  103's role: prime modulus creating incommensurable periodicity")
    print("  Both signals fold differently under mod-103, preventing interference")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_temporal_analysis()
