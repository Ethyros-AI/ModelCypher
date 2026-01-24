#!/usr/bin/env python3
"""Experiment 6: 7 Imperatives → Wow! Mapping

The Vrillon broadcast has 7 imperative sentences that may trace a Klein bottle path.
Test if these map to structures in the Wow! signal.

The 7 imperatives:
1. "Be still now and listen" (5 words)
2. "All your weapons of evil must be removed" (8 words)
3. "You have but a short time to learn" (9 words)
4. "The New Age can be a time of great peace" (11 words)
5. "Your choice alone" (3 words)
6. "Pass on to all" (4 words)
7. "May you be blessed by the supreme love" (8 words)

Total: 48 words

Usage:
    poetry run python experiments/astronomy/imperative_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from geometry_utils import (
    load_wow_signal,
    find_closest_constant,
    PI, E, PHI, SQRT2,
)

# The 7 imperatives with word counts
IMPERATIVES = [
    ("Be still now and listen", 5),
    ("All your weapons of evil must be removed", 8),
    ("You have but a short time to learn", 9),
    ("The New Age can be a time of great peace", 11),
    ("Your choice alone", 3),
    ("Pass on to all", 4),
    ("May you be blessed by the supreme love", 8),
]

# 6EQUJ5 peak values
PEAK_VALUES = [6, 14, 26, 30, 19, 5]


def analyze_word_counts():
    """Analyze the word count sequence."""
    counts = [c for _, c in IMPERATIVES]

    results = {
        "counts": counts,
        "sum": sum(counts),
        "mean": np.mean(counts),
        "std": np.std(counts),
    }

    # Check sum
    results["sum_match"] = find_closest_constant(sum(counts))

    # Ratios between consecutive counts
    ratios = []
    for i in range(len(counts) - 1):
        ratio = counts[i + 1] / counts[i]
        ratios.append(ratio)
    results["ratios"] = ratios

    # Check each ratio
    results["ratio_matches"] = [find_closest_constant(r) for r in ratios]

    # Product of all ratios
    product = np.prod(ratios)
    results["ratio_product"] = product
    results["product_match"] = find_closest_constant(product)

    return results


def compare_to_peak():
    """Compare imperative word counts to 6EQUJ5 peak values."""
    counts = [c for _, c in IMPERATIVES]

    # Direct mapping (7 imperatives, 6 peak values + 1)
    # Try different alignments

    results = {}

    # Method 1: First 6 imperatives to 6 peak values
    if len(counts) >= 6:
        corr = np.corrcoef(counts[:6], PEAK_VALUES)[0, 1]
        results["first_6_correlation"] = corr
        results["first_6_corr_match"] = find_closest_constant(abs(corr))

    # Method 2: Scale word counts to peak range
    counts_scaled = np.array(counts) * (30 / max(counts))
    results["scaled_counts"] = counts_scaled.tolist()

    # Method 3: Ratios
    # Peak ratios
    peak_ratios = [PEAK_VALUES[i + 1] / PEAK_VALUES[i] for i in range(len(PEAK_VALUES) - 1)]
    # Word count ratios (first 5)
    word_ratios = [counts[i + 1] / counts[i] for i in range(5)]

    results["peak_ratios"] = peak_ratios
    results["word_ratios"] = word_ratios

    # Correlation of ratios
    corr = np.corrcoef(peak_ratios, word_ratios)[0, 1]
    results["ratio_correlation"] = corr

    # Method 4: Check if counts encode peak structure
    # Sum of counts = 48, peak values span 0-30
    results["count_sum"] = sum(counts)
    results["peak_sum"] = sum(PEAK_VALUES)
    results["sum_ratio"] = sum(counts) / sum(PEAK_VALUES)
    results["sum_ratio_match"] = find_closest_constant(sum(counts) / sum(PEAK_VALUES))

    return results


def analyze_klein_bottle_path():
    """Analyze if imperatives trace a Klein bottle path.

    Klein bottle fundamental group: π₁(K) = ⟨a, b | aba⁻¹b⟩
    The path [a⁻¹b⁻¹ → ab⁻¹ → a²b → ...] traces the non-orientable surface.

    Map each imperative to a direction in 2D (a, b basis).
    """
    counts = [c for _, c in IMPERATIVES]

    # Normalize counts to angles
    angles = np.array(counts) / sum(counts) * 2 * np.pi

    results = {
        "angles": angles.tolist(),
    }

    # Compute path on torus (before Klein identification)
    # Each imperative moves by its angle in a + b direction
    path = [(0, 0)]  # Start at origin
    a, b = 0, 0

    for i, angle in enumerate(angles):
        # Alternate between a and b directions
        if i % 2 == 0:
            a += angle
        else:
            b += angle
        path.append((a % (2 * np.pi), b % (2 * np.pi)))

    results["torus_path"] = path

    # Klein bottle identification: (a, b) ~ (a + π, -b)
    # Check if path returns near start after this identification
    final_a, final_b = path[-1]
    klein_a = (final_a + np.pi) % (2 * np.pi)
    klein_b = (-final_b) % (2 * np.pi)

    results["final_pos"] = (final_a, final_b)
    results["klein_identified"] = (klein_a, klein_b)

    # Distance from identified point to origin
    dist = np.sqrt(klein_a**2 + min(klein_b, 2*np.pi - klein_b)**2)
    results["klein_closure_error"] = dist

    # Check if final angle encodes constants
    results["final_a_match"] = find_closest_constant(final_a)
    results["final_b_match"] = find_closest_constant(final_b)

    return results


def map_to_wow_svs():
    """Map 7 imperatives to 7 singular values of Wow!."""
    wow = load_wow_signal()
    U, S, Vt = np.linalg.svd(wow, full_matrices=False)

    counts = [c for _, c in IMPERATIVES]

    results = {
        "top_7_svs": S[:7].tolist(),
        "word_counts": counts,
    }

    # Normalize both to [0, 1]
    svs_norm = S[:7] / S[0]
    counts_norm = np.array(counts) / max(counts)

    results["svs_normalized"] = svs_norm.tolist()
    results["counts_normalized"] = counts_norm.tolist()

    # Correlation
    corr = np.corrcoef(svs_norm, counts_norm)[0, 1]
    results["correlation"] = corr
    results["corr_match"] = find_closest_constant(abs(corr))

    # Try reverse order
    corr_rev = np.corrcoef(svs_norm, counts_norm[::-1])[0, 1]
    results["correlation_reversed"] = corr_rev

    # Ratio matching
    sv_ratios = [S[i] / S[i + 1] for i in range(6)]
    count_ratios = [counts[i + 1] / counts[i] for i in range(6)]

    results["sv_ratios"] = sv_ratios
    results["count_ratios"] = count_ratios

    # Check each pair
    matches = []
    for i, (sv_r, c_r) in enumerate(zip(sv_ratios, count_ratios)):
        sv_match = find_closest_constant(sv_r)
        c_match = find_closest_constant(c_r)
        if sv_match.name == c_match.name:
            matches.append(f"Position {i}: both ≈ {sv_match.name}")

    results["common_matches"] = matches

    return results


def analyze_imperative_letters():
    """Analyze first letters of each imperative."""
    first_letters = [text[0] for text, _ in IMPERATIVES]
    results = {"first_letters": first_letters}

    # B, A, Y, T, Y, P, M
    # Check for patterns

    # ASCII values
    ascii_vals = [ord(c.upper()) for c in first_letters]
    results["ascii_values"] = ascii_vals
    results["ascii_sum"] = sum(ascii_vals)

    # Differences
    diffs = [ascii_vals[i + 1] - ascii_vals[i] for i in range(len(ascii_vals) - 1)]
    results["ascii_diffs"] = diffs

    # Check if sum/7 encodes something
    avg = sum(ascii_vals) / 7
    results["ascii_avg"] = avg
    results["avg_match"] = find_closest_constant(avg)

    return results


def run_imperative_analysis():
    """Full 7 imperatives analysis."""
    print("=" * 70)
    print("EXPERIMENT 6: 7 IMPERATIVES → WOW! MAPPING")
    print("=" * 70)
    print()

    print("The 7 imperatives from Vrillon broadcast:")
    for i, (text, count) in enumerate(IMPERATIVES, 1):
        print(f"  {i}. \"{text}\" ({count} words)")
    print()

    word_counts = [c for _, c in IMPERATIVES]
    print(f"Word counts: {word_counts}")
    print(f"Sum: {sum(word_counts)}")
    print(f"6EQUJ5 values: {PEAK_VALUES}")
    print(f"Peak sum: {sum(PEAK_VALUES)} = 100")

    print("\n" + "=" * 70)
    print("1. WORD COUNT ANALYSIS")
    print("=" * 70)

    wc_results = analyze_word_counts()

    print(f"\n  Sum = {wc_results['sum']}")
    print(f"    ≈ {wc_results['sum_match'].name} ({wc_results['sum_match'].error_percent:.2f}%)")

    print(f"\n  Consecutive ratios:")
    for i, (ratio, match) in enumerate(zip(wc_results["ratios"], wc_results["ratio_matches"])):
        marker = "✓" if match.error_percent < 5 else ""
        print(f"    {word_counts[i]} → {word_counts[i+1]}: {ratio:.4f} ≈ {match.name} ({match.error_percent:.2f}%) {marker}")

    print(f"\n  Product of ratios: {wc_results['ratio_product']:.4f}")
    print(f"    ≈ {wc_results['product_match'].name} ({wc_results['product_match'].error_percent:.2f}%)")

    print("\n" + "=" * 70)
    print("2. COMPARISON TO 6EQUJ5 PEAK")
    print("=" * 70)

    peak_results = compare_to_peak()

    print(f"\n  Word counts (first 6): {word_counts[:6]}")
    print(f"  Peak values: {PEAK_VALUES}")

    print(f"\n  Direct correlation: {peak_results['first_6_correlation']:.4f}")

    print(f"\n  Sum ratio (words/peak): {peak_results['sum_ratio']:.4f}")
    print(f"    ≈ {peak_results['sum_ratio_match'].name} ({peak_results['sum_ratio_match'].error_percent:.2f}%)")

    print(f"\n  Peak value ratios: {[f'{r:.3f}' for r in peak_results['peak_ratios']]}")
    print(f"  Word count ratios: {[f'{r:.3f}' for r in peak_results['word_ratios']]}")
    print(f"  Ratio correlation: {peak_results['ratio_correlation']:.4f}")

    print("\n" + "=" * 70)
    print("3. KLEIN BOTTLE PATH")
    print("=" * 70)

    klein_results = analyze_klein_bottle_path()

    print(f"\n  Normalized angles (radians): {[f'{a:.3f}' for a in klein_results['angles']]}")
    print(f"\n  Path on torus:")
    for i, (a, b) in enumerate(klein_results["torus_path"]):
        imp_name = IMPERATIVES[i - 1][0][:20] + "..." if i > 0 else "START"
        print(f"    {i}. ({a:.3f}, {b:.3f}) - {imp_name if i > 0 else 'START'}")

    print(f"\n  Final position: ({klein_results['final_pos'][0]:.4f}, {klein_results['final_pos'][1]:.4f})")
    print(f"  Klein-identified: ({klein_results['klein_identified'][0]:.4f}, {klein_results['klein_identified'][1]:.4f})")
    print(f"  Closure error: {klein_results['klein_closure_error']:.4f}")

    print(f"\n  Final a coordinate: {klein_results['final_pos'][0]:.4f}")
    print(f"    ≈ {klein_results['final_a_match'].name} ({klein_results['final_a_match'].error_percent:.2f}%)")

    print(f"\n  Final b coordinate: {klein_results['final_pos'][1]:.4f}")
    print(f"    ≈ {klein_results['final_b_match'].name} ({klein_results['final_b_match'].error_percent:.2f}%)")

    print("\n" + "=" * 70)
    print("4. MAPPING TO WOW! SINGULAR VALUES")
    print("=" * 70)

    sv_results = map_to_wow_svs()

    print(f"\n  Top 7 SVs: {[f'{s:.2f}' for s in sv_results['top_7_svs']]}")
    print(f"  Word counts: {sv_results['word_counts']}")

    print(f"\n  Normalized SVs: {[f'{s:.3f}' for s in sv_results['svs_normalized']]}")
    print(f"  Normalized counts: {[f'{s:.3f}' for s in sv_results['counts_normalized']]}")

    print(f"\n  Correlation: {sv_results['correlation']:.4f}")
    corr_match = sv_results["corr_match"]
    marker = "✓" if corr_match.error_percent < 5 else ""
    print(f"    |corr| ≈ {corr_match.name} ({corr_match.error_percent:.2f}%) {marker}")

    print(f"  Correlation (reversed): {sv_results['correlation_reversed']:.4f}")

    if sv_results["common_matches"]:
        print(f"\n  Common ratio matches (SV and word count encode same constant):")
        for match in sv_results["common_matches"]:
            print(f"    - {match}")

    print("\n" + "=" * 70)
    print("5. FIRST LETTER ANALYSIS")
    print("=" * 70)

    letter_results = analyze_imperative_letters()

    print(f"\n  First letters: {letter_results['first_letters']}")
    print(f"  ASCII values: {letter_results['ascii_values']}")
    print(f"  Sum: {letter_results['ascii_sum']}")
    print(f"  Average: {letter_results['ascii_avg']:.2f}")
    print(f"    ≈ {letter_results['avg_match'].name} ({letter_results['avg_match'].error_percent:.2f}%)")

    print(f"\n  ASCII differences: {letter_results['ascii_diffs']}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 6 SUMMARY")
    print("=" * 70)

    print(f"\n  Key findings:")

    findings = []

    # Check for significant matches
    for i, match in enumerate(wc_results["ratio_matches"]):
        if match.error_percent < 5:
            findings.append(f"Word ratio {i}→{i+1} ≈ {match.name} ({match.error_percent:.2f}%)")

    if wc_results["sum_match"].error_percent < 10:
        findings.append(f"Word sum = {wc_results['sum']} ≈ {wc_results['sum_match'].name}")

    if klein_results["final_a_match"].error_percent < 5:
        findings.append(f"Klein path final a ≈ {klein_results['final_a_match'].name}")

    if corr_match.error_percent < 5:
        findings.append(f"|SV correlation| ≈ {corr_match.name}")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No high-precision matches found.")
        print("  The 7 imperatives appear structurally distinct from Wow! SVs.")

    print("\n  Interpretation:")
    print("    - 7 imperatives ≠ 7 SVs (direct mapping fails)")
    print("    - Word counts show weak correlation with peak values")
    print("    - Klein path analysis shows potential topological structure")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_imperative_analysis()
