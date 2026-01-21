"""
Experiment 52: Direct Search for Mathematical Constants

The semantic highway experiments showed alignment with pi, e, phi.
But is there DIRECT evidence of these constants in the signal?

This experiment searches for:
1. Ratios between eigenvalues matching pi, e, phi
2. Ratios between peak values
3. Spacings that correspond to these constants
4. Frequency domain patterns

If we find direct numerical signatures, it strengthens the case
that the signal encodes mathematical structure.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.fft import fft2, fftfreq

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Mathematical constants to search for
CONSTANTS = {
    "pi": np.pi,               # 3.14159...
    "e": np.e,                 # 2.71828...
    "phi": (1 + np.sqrt(5))/2, # 1.61803... (golden ratio)
    "sqrt2": np.sqrt(2),       # 1.41421...
    "sqrt3": np.sqrt(3),       # 1.73205...
    "tau": 2 * np.pi,          # 6.28318...
    "ln2": np.log(2),          # 0.69314...
    "1/pi": 1/np.pi,           # 0.31831...
    "1/e": 1/np.e,             # 0.36787...
    "1/phi": 2/(1+np.sqrt(5)), # 0.61803...
    "pi/e": np.pi/np.e,        # 1.15572...
    "e/pi": np.e/np.pi,        # 0.86525...
    "pi*phi": np.pi * (1+np.sqrt(5))/2,  # 5.08320...
}


def find_ratio_matches(values, constants, tolerance=0.01):
    """Find ratios between consecutive values that match constants."""
    matches = []

    for i in range(len(values) - 1):
        if values[i+1] != 0 and values[i] != 0:
            ratio = values[i] / values[i+1]
            inv_ratio = values[i+1] / values[i]

            for name, const in constants.items():
                # Check ratio
                if abs(ratio - const) / const < tolerance:
                    matches.append({
                        "type": "ratio",
                        "indices": (i, i+1),
                        "ratio": float(ratio),
                        "constant": name,
                        "expected": float(const),
                        "error": float(abs(ratio - const) / const),
                    })
                # Check inverse ratio
                if abs(inv_ratio - const) / const < tolerance:
                    matches.append({
                        "type": "inv_ratio",
                        "indices": (i, i+1),
                        "ratio": float(inv_ratio),
                        "constant": name,
                        "expected": float(const),
                        "error": float(abs(inv_ratio - const) / const),
                    })

    return matches


def find_all_pair_ratios(values, constants, tolerance=0.02):
    """Find ratios between ALL pairs of values that match constants."""
    matches = []
    n = len(values)

    for i in range(n):
        for j in range(i+1, n):
            if values[i] != 0 and values[j] != 0:
                ratio = values[i] / values[j]

                for name, const in constants.items():
                    if abs(ratio - const) / const < tolerance:
                        matches.append({
                            "indices": (i, j),
                            "ratio": float(ratio),
                            "constant": name,
                            "expected": float(const),
                            "error": float(abs(ratio - const) / const),
                        })

    return matches


def analyze_eigenvalues(signal, constants):
    """Analyze eigenvalue ratios for mathematical constants."""
    # SVD of the signal
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    # Normalize eigenvalues
    S_norm = S / S[0]

    # Find consecutive ratio matches
    consecutive_matches = find_ratio_matches(S, constants, tolerance=0.02)

    # Find all-pair matches in top 10 eigenvalues
    pair_matches = find_all_pair_ratios(S[:10], constants, tolerance=0.02)

    # Check if eigenvalue ratios encode specific constants
    results = {
        "eigenvalues": [float(s) for s in S[:20]],
        "eigenvalues_normalized": [float(s) for s in S_norm[:20]],
        "consecutive_matches": consecutive_matches,
        "pair_matches": pair_matches,
        "participation_ratio": float((S.sum()**2) / (S**4).sum()),
    }

    # Look for specific patterns
    # e.g., is S[0]/S[1] close to pi?
    key_ratios = {}
    for i in range(min(5, len(S))):
        for j in range(i+1, min(5, len(S))):
            if S[j] > 0:
                ratio = S[i] / S[j]
                key_ratios[f"S{i}/S{j}"] = float(ratio)

                # Check against constants
                for name, const in constants.items():
                    if abs(ratio - const) / const < 0.05:  # 5% tolerance
                        key_ratios[f"S{i}/S{j}_match"] = name

    results["key_ratios"] = key_ratios

    return results


def analyze_peak_values(signal, constants):
    """Analyze ratios between peak values."""
    # Find peaks in each row (time slice)
    row_maxes = np.max(signal, axis=1)
    row_maxes_sorted = np.sort(row_maxes)[::-1]

    # Find peaks in each column (frequency bin)
    col_maxes = np.max(signal, axis=0)
    col_maxes_sorted = np.sort(col_maxes)[::-1]

    # Find global peaks
    flat = signal.flatten()
    top_values = np.sort(flat)[::-1][:20]

    results = {
        "row_max_ratios": find_ratio_matches(row_maxes_sorted[:10], constants),
        "col_max_ratios": find_ratio_matches(col_maxes_sorted[:10], constants),
        "top_value_ratios": find_ratio_matches(top_values, constants),
        "top_values": [float(v) for v in top_values],
    }

    return results


def analyze_frequency_domain(signal, constants):
    """Analyze the 2D FFT for constant signatures."""
    # 2D FFT
    F = fft2(signal)
    magnitude = np.abs(F)

    # Get dominant frequencies
    mag_flat = magnitude.flatten()
    top_indices = np.argsort(mag_flat)[::-1][:20]

    top_magnitudes = mag_flat[top_indices]
    top_magnitudes_norm = top_magnitudes / top_magnitudes[0]

    # Check ratios
    ratio_matches = find_ratio_matches(top_magnitudes, constants)

    # Analyze frequency spacing
    rows, cols = signal.shape
    freq_rows = fftfreq(rows)
    freq_cols = fftfreq(cols)

    results = {
        "top_magnitudes": [float(m) for m in top_magnitudes],
        "top_magnitudes_normalized": [float(m) for m in top_magnitudes_norm],
        "frequency_ratio_matches": ratio_matches,
    }

    return results


def analyze_spacing_patterns(signal, constants):
    """Analyze spacing between significant features."""
    # Find the "6EQUJ5" peak - the famous intensity pattern
    row_energies = np.sum(signal**2, axis=1)
    peak_row = np.argmax(row_energies)

    # Find significant time points (rows with high energy)
    threshold = np.mean(row_energies) + np.std(row_energies)
    significant_rows = np.where(row_energies > threshold)[0]

    # Analyze spacing between significant rows
    if len(significant_rows) > 1:
        spacings = np.diff(significant_rows)
        spacing_ratios = find_ratio_matches(spacings, constants) if len(spacings) > 1 else []
    else:
        spacings = []
        spacing_ratios = []

    # Find significant frequency bins
    col_energies = np.sum(signal**2, axis=0)
    threshold_col = np.mean(col_energies) + np.std(col_energies)
    significant_cols = np.where(col_energies > threshold_col)[0]

    if len(significant_cols) > 1:
        col_spacings = np.diff(significant_cols)
        col_spacing_ratios = find_ratio_matches(col_spacings, constants) if len(col_spacings) > 1 else []
    else:
        col_spacings = []
        col_spacing_ratios = []

    results = {
        "peak_row": int(peak_row),
        "significant_rows": [int(r) for r in significant_rows],
        "row_spacings": [int(s) for s in spacings],
        "row_spacing_ratio_matches": spacing_ratios,
        "significant_cols": [int(c) for c in significant_cols],
        "col_spacings": [int(s) for s in col_spacings],
        "col_spacing_ratio_matches": col_spacing_ratios,
    }

    return results


def compute_baseline_matches(shape, constants, n_trials=100):
    """Compute how many matches random matrices produce."""
    all_eigen_matches = []
    all_peak_matches = []

    for _ in range(n_trials):
        rand = np.random.randn(*shape)

        # Eigenvalue matches
        U, S, Vh = linalg.svd(rand, full_matrices=False)
        eigen_matches = len(find_ratio_matches(S, constants, tolerance=0.02))
        all_eigen_matches.append(eigen_matches)

        # Peak matches
        top_vals = np.sort(rand.flatten())[::-1][:20]
        peak_matches = len(find_ratio_matches(top_vals, constants, tolerance=0.02))
        all_peak_matches.append(peak_matches)

    return {
        "eigen_matches_mean": float(np.mean(all_eigen_matches)),
        "eigen_matches_std": float(np.std(all_eigen_matches)),
        "peak_matches_mean": float(np.mean(all_peak_matches)),
        "peak_matches_std": float(np.std(all_peak_matches)),
    }


def main():
    print("=" * 60)
    print("Experiment 52: Direct Search for Mathematical Constants")
    print("=" * 60)
    print("\nQuestion: Are pi, e, phi DIRECTLY encoded in the signal?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape}")

    # Compute baseline
    print("\n2. Computing random baseline (100 trials)...")
    baseline = compute_baseline_matches(signal.shape, CONSTANTS)
    print(f"   Random eigenvalue ratio matches: {baseline['eigen_matches_mean']:.1f} +/- {baseline['eigen_matches_std']:.1f}")
    print(f"   Random peak value ratio matches: {baseline['peak_matches_mean']:.1f} +/- {baseline['peak_matches_std']:.1f}")

    # Analyze eigenvalues
    print("\n3. Analyzing eigenvalue ratios...")
    eigen_results = analyze_eigenvalues(signal, CONSTANTS)
    n_eigen_matches = len(eigen_results["consecutive_matches"])
    print(f"   Found {n_eigen_matches} consecutive ratio matches")

    # Z-score
    z_eigen = (n_eigen_matches - baseline["eigen_matches_mean"]) / (baseline["eigen_matches_std"] + 1e-8)
    print(f"   Z-score vs random: {z_eigen:+.2f}")

    if eigen_results["consecutive_matches"]:
        print("\n   Matches found:")
        for match in eigen_results["consecutive_matches"][:10]:
            print(f"      S[{match['indices'][0]}]/S[{match['indices'][1]}] = {match['ratio']:.4f} ≈ {match['constant']} ({match['expected']:.4f}), error={match['error']*100:.1f}%")

    # Key ratios
    print("\n   Key eigenvalue ratios:")
    for key, val in eigen_results["key_ratios"].items():
        if "_match" not in key:
            match = eigen_results["key_ratios"].get(f"{key}_match", "no match")
            print(f"      {key}: {val:.4f} -> {match}")

    # Analyze peak values
    print("\n4. Analyzing peak value ratios...")
    peak_results = analyze_peak_values(signal, CONSTANTS)
    n_top_matches = len(peak_results["top_value_ratios"])
    z_peak = (n_top_matches - baseline["peak_matches_mean"]) / (baseline["peak_matches_std"] + 1e-8)
    print(f"   Found {n_top_matches} top value ratio matches")
    print(f"   Z-score vs random: {z_peak:+.2f}")

    if peak_results["top_value_ratios"]:
        print("\n   Top value ratio matches:")
        for match in peak_results["top_value_ratios"][:5]:
            print(f"      vals[{match['indices'][0]}]/vals[{match['indices'][1]}] = {match['ratio']:.4f} ≈ {match['constant']}")

    # Analyze frequency domain
    print("\n5. Analyzing frequency domain...")
    freq_results = analyze_frequency_domain(signal, CONSTANTS)
    n_freq_matches = len(freq_results["frequency_ratio_matches"])
    print(f"   Found {n_freq_matches} frequency magnitude ratio matches")

    if freq_results["frequency_ratio_matches"]:
        print("\n   Frequency matches:")
        for match in freq_results["frequency_ratio_matches"][:5]:
            print(f"      F[{match['indices'][0]}]/F[{match['indices'][1]}] = {match['ratio']:.4f} ≈ {match['constant']}")

    # Analyze spacing patterns
    print("\n6. Analyzing spacing patterns...")
    spacing_results = analyze_spacing_patterns(signal, CONSTANTS)
    print(f"   Peak row: {spacing_results['peak_row']}")
    print(f"   Significant rows: {spacing_results['significant_rows']}")
    print(f"   Row spacings: {spacing_results['row_spacings']}")

    if spacing_results["row_spacing_ratio_matches"]:
        print("\n   Row spacing ratio matches:")
        for match in spacing_results["row_spacing_ratio_matches"]:
            print(f"      spacing[{match['indices'][0]}]/spacing[{match['indices'][1]}] ≈ {match['constant']}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    total_matches = n_eigen_matches + n_top_matches + n_freq_matches
    print(f"\nTotal ratio matches found: {total_matches}")
    print(f"   Eigenvalue ratios: {n_eigen_matches} (z={z_eigen:+.2f})")
    print(f"   Peak value ratios: {n_top_matches} (z={z_peak:+.2f})")
    print(f"   Frequency ratios: {n_freq_matches}")

    # Which constants appear most?
    all_matches = (
        eigen_results["consecutive_matches"] +
        peak_results["top_value_ratios"] +
        freq_results["frequency_ratio_matches"]
    )

    constant_counts = {}
    for match in all_matches:
        const = match["constant"]
        constant_counts[const] = constant_counts.get(const, 0) + 1

    if constant_counts:
        print("\n   Most frequent constants found:")
        for const, count in sorted(constant_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {const}: {count} matches")

    # Interpretation
    print("\n   INTERPRETATION:")
    if z_eigen > 2 or z_peak > 2:
        print("   --> Signal has SIGNIFICANTLY more constant ratios than random")
        print("      This suggests direct encoding of mathematical relationships")
        significant = True
    else:
        print("   --> Signal does NOT have significantly more constant ratios than random")
        print("      The semantic alignment may not be due to direct numerical encoding")
        significant = False

    # Save results
    results = {
        "experiment": "exp52_constant_search",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "constants_searched": list(CONSTANTS.keys()),
        "baseline": baseline,
        "eigenvalue_analysis": eigen_results,
        "peak_analysis": peak_results,
        "frequency_analysis": freq_results,
        "spacing_analysis": spacing_results,
        "summary": {
            "n_eigen_matches": n_eigen_matches,
            "n_peak_matches": n_top_matches,
            "n_freq_matches": n_freq_matches,
            "z_eigen": float(z_eigen),
            "z_peak": float(z_peak),
            "constant_counts": constant_counts,
            "significant_vs_random": significant,
        },
    }

    output_path = RESULTS_DIR / "exp52_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n7. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
