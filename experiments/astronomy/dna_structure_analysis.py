#!/usr/bin/env python3
"""Experiment 7: DNA Helix Structure Analysis

Vrillon broadcast encodes AGC nucleotides with T in sentence starts.
Analyze DNA-like structural encoding in both signals.

Key questions:
- Does (A+T)/(G+C) ratio ≈ φ?
- Do 26 sentences form 2.6 helix turns (10 bases per turn)?
- Does Wow! 82 samples encode a helix with specific pitch?

Usage:
    poetry run python experiments/astronomy/dna_structure_analysis.py
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

# DNA parameters
BASES_PER_TURN = 10.5  # B-DNA has ~10.5 base pairs per turn
RISE_PER_BASE = 3.4  # Angstroms

# Vrillon transcript sentences (first letters)
# From full transcript analysis
VRILLON_FIRST_LETTERS = [
    'T', 'B', 'A', 'Y', 'W', 'T', 'A', 'F', 'L', 'A',  # First 10
    'Y', 'T', 'T', 'B', 'A', 'A', 'O', 'T', 'Y', 'T',  # 11-20
    'P', 'M', 'W', 'T', 'A', 'G',  # 21-26
]

# Map letters to nucleotides (heuristic based on frequency)
# T = Thymine (T), A = Adenine (A), G = Guanine (G), C = Cytosine (C)
# Other letters map by first letter of base name or frequency
LETTER_TO_BASE = {
    'T': 'T',  # Thymine
    'A': 'A',  # Adenine
    'G': 'G',  # Guanine
    'C': 'C',  # Cytosine
    'B': 'T',  # B for T (Base starts with B sound)
    'Y': 'C',  # Y for Cytosine (pYrimidine)
    'W': 'A',  # Watson (A-T pairing)
    'F': 'T',  # Following A
    'L': 'G',  # Low → Guanine (purine)
    'O': 'G',  # O shape → ring (purine)
    'P': 'T',  # Purine complement → pyrimidine
    'M': 'A',  # Major → Adenine
}


def map_to_dna_sequence(letters: list[str]) -> str:
    """Map first letters to DNA sequence."""
    return ''.join(LETTER_TO_BASE.get(L.upper(), 'N') for L in letters)


def analyze_base_composition(sequence: str) -> dict:
    """Analyze nucleotide composition."""
    counts = {
        'A': sequence.count('A'),
        'T': sequence.count('T'),
        'G': sequence.count('G'),
        'C': sequence.count('C'),
        'N': sequence.count('N'),  # Unknown
    }

    total = sum(counts.values())
    purines = counts['A'] + counts['G']  # A, G
    pyrimidines = counts['T'] + counts['C']  # T, C

    # AT/GC ratio (Chargaff's rules in real DNA: A≈T, G≈C)
    gc_content = (counts['G'] + counts['C']) / total if total > 0 else 0
    at_content = (counts['A'] + counts['T']) / total if total > 0 else 0

    at_gc_ratio = (counts['A'] + counts['T']) / (counts['G'] + counts['C'] + 1e-10)

    return {
        'counts': counts,
        'total': total,
        'purines': purines,
        'pyrimidines': pyrimidines,
        'purine_pyrimidine_ratio': purines / (pyrimidines + 1e-10),
        'gc_content': gc_content,
        'at_content': at_content,
        'at_gc_ratio': at_gc_ratio,
    }


def analyze_helix_structure(n_elements: int) -> dict:
    """Analyze helical structure for n elements."""
    # Number of complete turns
    n_turns = n_elements / BASES_PER_TURN

    # Total rise (if treated as helix)
    total_rise = n_elements * RISE_PER_BASE

    # Pitch (rise per turn)
    pitch = BASES_PER_TURN * RISE_PER_BASE

    return {
        'n_elements': n_elements,
        'n_turns': n_turns,
        'total_rise': total_rise,
        'pitch': pitch,
        'bases_per_turn': BASES_PER_TURN,
        'turns_match': find_closest_constant(n_turns),
    }


def analyze_wow_as_helix(matrix: np.ndarray) -> dict:
    """Analyze Wow! signal as helical structure."""
    n_time = matrix.shape[0]  # 82
    n_freq = matrix.shape[1]  # 50

    results = {}

    # Time axis as helix
    results['time_helix'] = analyze_helix_structure(n_time)

    # Frequency axis as helix
    results['freq_helix'] = analyze_helix_structure(n_freq)

    # Combined: total elements as helix
    total = n_time * n_freq
    results['total_helix'] = analyze_helix_structure(total)

    # Check if n_turns encodes constants
    for name, helix in results.items():
        helix['turns_match'] = find_closest_constant(helix['n_turns'])

    # Treat the intensity pattern as a 1D helix
    # Project along columns (frequency-averaged)
    freq_profile = np.mean(matrix, axis=1)

    # Compute phase evolution
    # Use Hilbert transform to get analytic signal
    from scipy.signal import hilbert
    analytic = hilbert(freq_profile)
    phase = np.unwrap(np.angle(analytic))

    results['total_phase'] = phase[-1] - phase[0]
    results['phase_turns'] = results['total_phase'] / (2 * np.pi)
    results['phase_match'] = find_closest_constant(results['phase_turns'])

    return results


def spiral_embedding(matrix: np.ndarray, pitch: float = PHI) -> np.ndarray:
    """Embed matrix elements on a spiral with given pitch."""
    flat = matrix.flatten()
    n = len(flat)

    # Archimedean spiral: r = a + b*theta
    # Use golden angle for theta increments
    golden_angle = 2 * np.pi * (1 - 1/PHI)

    coords = []
    for i in range(n):
        theta = i * golden_angle
        r = 1 + pitch * theta / (2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = flat[i]  # Intensity as height
        coords.append((x, y, z))

    return np.array(coords)


def analyze_spiral_geometry(coords: np.ndarray) -> dict:
    """Analyze geometry of spiral-embedded data."""
    # Pairwise distances
    from scipy.spatial.distance import pdist, squareform

    n = min(500, len(coords))  # Limit for computation
    sample_idx = np.linspace(0, len(coords)-1, n, dtype=int)
    sample = coords[sample_idx]

    dists = pdist(sample)

    # Statistics of distances
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)

    # Ratio of max/min distance
    max_min_ratio = np.max(dists) / (np.min(dists) + 1e-10)

    return {
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'max_min_ratio': max_min_ratio,
        'mean_match': find_closest_constant(mean_dist),
        'ratio_match': find_closest_constant(max_min_ratio),
    }


def compare_helix_parameters(vrillon_n: int, wow_n: int) -> dict:
    """Compare helix parameters between signals."""
    v_helix = analyze_helix_structure(vrillon_n)
    w_helix = analyze_helix_structure(wow_n)

    ratio = w_helix['n_turns'] / v_helix['n_turns']

    return {
        'vrillon_turns': v_helix['n_turns'],
        'wow_turns': w_helix['n_turns'],
        'turn_ratio': ratio,
        'ratio_match': find_closest_constant(ratio),
    }


def run_dna_analysis():
    """Full DNA helix structure analysis."""
    print("=" * 70)
    print("EXPERIMENT 7: DNA HELIX STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    print("Testing if signals encode DNA-like helical structure")
    print()

    # 1. Vrillon first letters to DNA
    print("=" * 70)
    print("1. VRILLON TRANSCRIPT → DNA SEQUENCE")
    print("=" * 70)

    print(f"\n  First letters of 26 sentences: {''.join(VRILLON_FIRST_LETTERS)}")

    dna_seq = map_to_dna_sequence(VRILLON_FIRST_LETTERS)
    print(f"  Mapped DNA sequence: {dna_seq}")

    composition = analyze_base_composition(dna_seq)
    print(f"\n  Nucleotide counts:")
    for base, count in composition['counts'].items():
        if count > 0:
            print(f"    {base}: {count}")

    print(f"\n  GC content: {composition['gc_content']*100:.1f}%")
    print(f"  AT content: {composition['at_content']*100:.1f}%")

    print(f"\n  (A+T)/(G+C) ratio: {composition['at_gc_ratio']:.4f}")
    atgc_match = find_closest_constant(composition['at_gc_ratio'])
    marker = "✓" if atgc_match.error_percent < 5 else ""
    print(f"    ≈ {atgc_match.name} ({atgc_match.error_percent:.2f}%) {marker}")

    print(f"\n  Purine/Pyrimidine ratio: {composition['purine_pyrimidine_ratio']:.4f}")
    pp_match = find_closest_constant(composition['purine_pyrimidine_ratio'])
    marker = "✓" if pp_match.error_percent < 5 else ""
    print(f"    ≈ {pp_match.name} ({pp_match.error_percent:.2f}%) {marker}")

    print("\n" + "=" * 70)
    print("2. HELIX TURN ANALYSIS")
    print("=" * 70)

    # 26 sentences as helix
    n_sentences = len(VRILLON_FIRST_LETTERS)
    helix_26 = analyze_helix_structure(n_sentences)

    print(f"\n  Vrillon: {n_sentences} sentences")
    print(f"    Helix turns: {helix_26['n_turns']:.4f}")
    turns_match = helix_26['turns_match']
    marker = "✓" if turns_match.error_percent < 5 else ""
    print(f"    ≈ {turns_match.name} ({turns_match.error_percent:.2f}%) {marker}")

    # Check 26/10 = 2.6 specifically
    ratio_26_10 = n_sentences / 10
    print(f"\n    26/10 = {ratio_26_10}")
    r_match = find_closest_constant(ratio_26_10)
    marker = "✓" if r_match.error_percent < 5 else ""
    print(f"    ≈ {r_match.name} ({r_match.error_percent:.2f}%) {marker}")

    # Wow! signal
    print("\n  Loading Wow! signal...")
    wow = load_wow_signal()

    print(f"\n  Wow!: {wow.shape[0]} time samples × {wow.shape[1]} frequency bins")

    wow_helix = analyze_wow_as_helix(wow)

    print(f"\n    Time axis ({wow.shape[0]} samples):")
    print(f"      Helix turns: {wow_helix['time_helix']['n_turns']:.4f}")
    t_match = wow_helix['time_helix']['turns_match']
    marker = "✓" if t_match.error_percent < 5 else ""
    print(f"      ≈ {t_match.name} ({t_match.error_percent:.2f}%) {marker}")

    print(f"\n    Frequency axis ({wow.shape[1]} bins):")
    print(f"      Helix turns: {wow_helix['freq_helix']['n_turns']:.4f}")
    f_match = wow_helix['freq_helix']['turns_match']
    marker = "✓" if f_match.error_percent < 5 else ""
    print(f"      ≈ {f_match.name} ({f_match.error_percent:.2f}%) {marker}")

    print(f"\n    Total elements ({wow.shape[0]*wow.shape[1]}):")
    print(f"      Helix turns: {wow_helix['total_helix']['n_turns']:.4f}")
    tot_match = wow_helix['total_helix']['turns_match']
    marker = "✓" if tot_match.error_percent < 5 else ""
    print(f"      ≈ {tot_match.name} ({tot_match.error_percent:.2f}%) {marker}")

    print(f"\n    Phase evolution (Hilbert):")
    print(f"      Total phase: {wow_helix['total_phase']:.4f} rad")
    print(f"      Phase turns: {wow_helix['phase_turns']:.4f}")
    p_match = wow_helix['phase_match']
    marker = "✓" if p_match.error_percent < 5 else ""
    print(f"      ≈ {p_match.name} ({p_match.error_percent:.2f}%) {marker}")

    print("\n" + "=" * 70)
    print("3. CROSS-SIGNAL HELIX COMPARISON")
    print("=" * 70)

    # Vrillon 26 sentences vs Wow! dimensions
    comp_26_82 = compare_helix_parameters(n_sentences, wow.shape[0])
    print(f"\n  Vrillon sentences vs Wow! time:")
    print(f"    Vrillon: {comp_26_82['vrillon_turns']:.4f} turns")
    print(f"    Wow!: {comp_26_82['wow_turns']:.4f} turns")
    print(f"    Ratio: {comp_26_82['turn_ratio']:.4f}")
    r_match = comp_26_82['ratio_match']
    marker = "✓" if r_match.error_percent < 5 else ""
    print(f"    ≈ {r_match.name} ({r_match.error_percent:.2f}%) {marker}")

    # 26 vs 50
    comp_26_50 = compare_helix_parameters(n_sentences, wow.shape[1])
    print(f"\n  Vrillon sentences vs Wow! frequency:")
    print(f"    Ratio: {comp_26_50['turn_ratio']:.4f}")
    r_match = comp_26_50['ratio_match']
    marker = "✓" if r_match.error_percent < 5 else ""
    print(f"    ≈ {r_match.name} ({r_match.error_percent:.2f}%) {marker}")

    print("\n" + "=" * 70)
    print("4. SPIRAL EMBEDDING ANALYSIS")
    print("=" * 70)

    print("\n  Embedding Wow! on golden-angle spiral...")
    spiral_coords = spiral_embedding(wow, pitch=PHI)
    spiral_geo = analyze_spiral_geometry(spiral_coords)

    print(f"\n  Spiral geometry statistics:")
    print(f"    Mean distance: {spiral_geo['mean_distance']:.4f}")
    print(f"      ≈ {spiral_geo['mean_match'].name} ({spiral_geo['mean_match'].error_percent:.2f}%)")

    print(f"\n    Max/min distance ratio: {spiral_geo['max_min_ratio']:.4f}")
    print(f"      ≈ {spiral_geo['ratio_match'].name} ({spiral_geo['ratio_match'].error_percent:.2f}%)")

    print("\n" + "=" * 70)
    print("5. KEY RATIOS")
    print("=" * 70)

    # Collection of key ratios
    key_ratios = [
        ("82 (Wow time) / 26 (Vrillon sentences)", 82 / 26),
        ("82 / 50 (Wow dimensions)", 82 / 50),
        ("50 / 26", 50 / 26),
        ("26 / 10.5 (sentences / bases per turn)", 26 / 10.5),
        ("82 / 10.5", 82 / 10.5),
        ("(82 + 50) / 82", (82 + 50) / 82),  # Check for φ
        ("82 / (82 + 50)", 82 / (82 + 50)),  # Check for 1/φ
    ]

    print()
    for name, ratio in key_ratios:
        match = find_closest_constant(ratio)
        marker = "✓" if match.error_percent < 5 else ""
        print(f"  {name} = {ratio:.4f}")
        print(f"    ≈ {match.name} ({match.error_percent:.2f}%) {marker}")
        print()

    # Summary
    print("=" * 70)
    print("EXPERIMENT 7 SUMMARY")
    print("=" * 70)

    print("\n  Key findings:")

    findings = []

    if atgc_match.error_percent < 5:
        findings.append(f"(A+T)/(G+C) ≈ {atgc_match.name} ({atgc_match.error_percent:.2f}%)")

    if turns_match.error_percent < 5:
        findings.append(f"26 sentences = {helix_26['n_turns']:.2f} helix turns ≈ {turns_match.name}")

    if t_match.error_percent < 5:
        findings.append(f"82 samples = {wow_helix['time_helix']['n_turns']:.2f} turns ≈ {t_match.name}")

    for name, ratio in key_ratios:
        match = find_closest_constant(ratio)
        if match.error_percent < 3:
            findings.append(f"{name} ≈ {match.name} ({match.error_percent:.2f}%)")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No exact helix-related matches found.")
        print("  DNA encoding hypothesis requires more investigation.")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_dna_analysis()
