"""
Experiment 62: Temporal Analysis

Why 72 seconds?

The Wow! signal lasted exactly 72 seconds - the maximum time a signal could
be observed by the Big Ear radio telescope as Earth rotated the beam across
the sky. But is this coincidence, or does the signal encode structure in
its temporal dimension?

72 = 8 × 9 = 2³ × 3²

Questions:
1. Is there repeating structure within the 72-second window?
2. Are there boundary markers (beginning/end signatures)?
3. Does the signal have temporal symmetry?
4. Do different time slices encode different information?
5. Is 72 itself meaningful? (factors, relationships to constants)

Key insight: The signal was recorded as intensity over time/frequency.
The temporal dimension IS one axis of our 2D matrix.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.signal import find_peaks
from scipy.ndimage import zoom
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FRB_DIR = Path(__file__).parent / "data" / "raw"

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e


def analyze_72():
    """Analyze the number 72 itself."""
    results = {
        "value": 72,
        "factorization": "2³ × 3² = 8 × 9",
        "factors": [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72],
        "n_divisors": 12,
        "relationships": {
            "72/phi": 72 / PHI,
            "72/pi": 72 / PI,
            "72/e": 72 / E,
            "72/(phi*pi)": 72 / (PHI * PI),
            "72/8": 9,  # Clean division
            "72/9": 8,  # Clean division
            "72/12": 6,  # Clean division
            "sqrt(72)": np.sqrt(72),  # 8.485...
            "72 mod 10": 72 % 10,  # 2
        },
        "notes": [
            "72 = 8 × 9 (consecutive integers product)",
            "72° = 360°/5 (pentagonal angle, related to phi)",
            "72 is a highly composite number",
            "72 = sum of twin primes 5+67, 11+61, 13+59, 17+43, 19+53, 29+43, 31+41",
        ]
    }

    # Check if 72/phi is close to anything special
    r = 72 / PHI  # ≈ 44.50
    results["72/phi_nearest_int"] = round(r)
    results["72/phi_error"] = abs(r - round(r)) / round(r)

    # 72/pi ≈ 22.92 (close to 23)
    r = 72 / PI
    results["72/pi_nearest_int"] = round(r)
    results["72/pi_error"] = abs(r - round(r)) / round(r)

    return results


def analyze_temporal_structure(signal):
    """Analyze the temporal dimension of the signal."""
    # Signal shape: (frequency, time) or (time, frequency)
    # Need to determine which axis is time

    # Assuming time is the second axis (columns)
    n_freq, n_time = signal.shape

    results = {
        "shape": {"n_freq": n_freq, "n_time": n_time},
    }

    # 1. Compute intensity profile over time (sum across frequencies)
    time_profile = np.sum(signal, axis=0)
    time_profile_normalized = time_profile / np.max(np.abs(time_profile) + 1e-10)

    results["time_profile"] = {
        "mean": float(np.mean(time_profile)),
        "std": float(np.std(time_profile)),
        "max": float(np.max(time_profile)),
        "min": float(np.min(time_profile)),
        "peak_location": int(np.argmax(time_profile)),
        "peak_location_fraction": float(np.argmax(time_profile) / n_time),
    }

    # 2. Check for temporal symmetry
    # Compare first half to reversed second half
    half = n_time // 2
    first_half = time_profile[:half]
    second_half = time_profile[half:2*half][::-1]  # Reverse second half

    if len(first_half) == len(second_half):
        symmetry_corr = np.corrcoef(first_half, second_half)[0, 1]
        results["temporal_symmetry"] = {
            "correlation": float(symmetry_corr) if not np.isnan(symmetry_corr) else 0.0,
            "is_symmetric": bool(abs(symmetry_corr) > 0.7),
        }
    else:
        results["temporal_symmetry"] = {"correlation": 0.0, "is_symmetric": False}

    # 3. Look for repeating patterns (autocorrelation)
    autocorr = np.correlate(time_profile_normalized, time_profile_normalized, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Find peaks in autocorrelation (repeating periods)
    peaks, properties = find_peaks(autocorr[1:], height=0.3, distance=3)
    peaks = peaks + 1  # Adjust for the slice

    results["autocorrelation"] = {
        "n_significant_peaks": len(peaks),
        "peak_lags": [int(p) for p in peaks[:5]],  # First 5 peaks
        "peak_heights": [float(autocorr[p]) for p in peaks[:5]] if len(peaks) > 0 else [],
    }

    # 4. Slice into segments and compare eigenvalue structure
    n_slices = 8  # 72/8 = 9
    slice_size = n_time // n_slices

    slice_eigenratios = []
    for i in range(n_slices):
        start = i * slice_size
        end = start + slice_size
        slice_data = signal[:, start:end]

        if slice_data.size > 0 and slice_data.shape[1] > 1:
            try:
                _, S, _ = linalg.svd(slice_data, full_matrices=False)
                if len(S) > 1 and S[1] > 1e-10:
                    ratio = S[0] / S[1]
                    slice_eigenratios.append(float(ratio))
                else:
                    slice_eigenratios.append(None)
            except Exception:
                slice_eigenratios.append(None)
        else:
            slice_eigenratios.append(None)

    results["slice_analysis"] = {
        "n_slices": n_slices,
        "slice_size": slice_size,
        "s0_s1_ratios": slice_eigenratios,
    }

    # Check if any slice ratio matches phi
    phi_matches = []
    for i, ratio in enumerate(slice_eigenratios):
        if ratio is not None:
            error = abs(ratio - PHI) / PHI
            if error < 0.15:
                phi_matches.append({"slice": i, "ratio": ratio, "error": error})

    results["slice_analysis"]["phi_matches"] = phi_matches

    # 5. Boundary analysis - are edges different from middle?
    edge_width = n_time // 8

    left_edge = signal[:, :edge_width]
    right_edge = signal[:, -edge_width:]
    middle = signal[:, edge_width:-edge_width]

    def compute_spectral_entropy(data):
        _, S, _ = linalg.svd(data, full_matrices=False)
        S_norm = S / (np.sum(S) + 1e-10)
        S_norm = S_norm[S_norm > 1e-10]
        return float(-np.sum(S_norm * np.log(S_norm)))

    try:
        results["boundary_analysis"] = {
            "left_edge_entropy": compute_spectral_entropy(left_edge),
            "right_edge_entropy": compute_spectral_entropy(right_edge),
            "middle_entropy": compute_spectral_entropy(middle),
        }

        # Check if edges are different from middle
        edge_avg = (results["boundary_analysis"]["left_edge_entropy"] +
                   results["boundary_analysis"]["right_edge_entropy"]) / 2
        middle_entropy = results["boundary_analysis"]["middle_entropy"]

        results["boundary_analysis"]["edge_vs_middle_diff"] = abs(edge_avg - middle_entropy)
        results["boundary_analysis"]["edges_are_different"] = bool(
            abs(edge_avg - middle_entropy) > 0.1 * middle_entropy
        )
    except Exception:
        results["boundary_analysis"] = {"error": "Could not compute"}

    # 6. Check for 8-fold or 9-fold structure
    for n_divisions in [8, 9, 12]:
        div_size = n_time // n_divisions
        division_energies = []

        for i in range(n_divisions):
            start = i * div_size
            end = start + div_size
            division_data = signal[:, start:end]
            energy = float(np.sum(division_data ** 2))
            division_energies.append(energy)

        # Check for pattern in division energies
        division_energies = np.array(division_energies)
        if np.std(division_energies) > 0:
            division_normalized = division_energies / np.mean(division_energies)

            results[f"division_{n_divisions}"] = {
                "energies": [float(e) for e in division_normalized],
                "std": float(np.std(division_normalized)),
                "max_min_ratio": float(np.max(division_normalized) / (np.min(division_normalized) + 1e-10)),
            }

    return results


def analyze_freq_structure(signal):
    """Analyze the frequency dimension."""
    n_freq, n_time = signal.shape

    # Frequency profile (sum across time)
    freq_profile = np.sum(signal, axis=1)

    results = {
        "n_freq": n_freq,
        "freq_profile_std": float(np.std(freq_profile)),
        "freq_profile_mean": float(np.mean(freq_profile)),
    }

    # Find frequency peaks
    freq_normalized = freq_profile / (np.max(np.abs(freq_profile)) + 1e-10)
    peaks, _ = find_peaks(freq_normalized, height=0.3, distance=2)

    results["n_freq_peaks"] = len(peaks)
    results["peak_frequencies"] = [int(p) for p in peaks[:10]]

    # Check frequency ratios between peaks
    if len(peaks) >= 2:
        peak_ratios = []
        for i in range(len(peaks) - 1):
            if peaks[i] > 0:
                ratio = peaks[i + 1] / peaks[i]
                peak_ratios.append(float(ratio))
        results["peak_frequency_ratios"] = peak_ratios[:5]

    return results


def compare_to_frbs(wow_temporal, n_frbs=20):
    """Compare temporal structure to FRBs."""
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:n_frbs]

    frb_symmetries = []
    frb_autocorr_peaks = []

    for frb_file in frb_files:
        try:
            with h5py.File(frb_file, "r") as f:
                if "frb" in f and "wfall" in f["frb"]:
                    data = f["frb"]["wfall"][:].astype(np.float64)
                elif "frb" in f and "calibrated_wfall" in f["frb"]:
                    data = f["frb"]["calibrated_wfall"][:].astype(np.float64)
                else:
                    continue

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            if data.shape[0] > data.shape[1]:
                data = data.T

            frb_temporal = analyze_temporal_structure(data)

            if "temporal_symmetry" in frb_temporal:
                frb_symmetries.append(frb_temporal["temporal_symmetry"]["correlation"])
            if "autocorrelation" in frb_temporal:
                frb_autocorr_peaks.append(frb_temporal["autocorrelation"]["n_significant_peaks"])

        except Exception:
            continue

    results = {
        "n_frbs_analyzed": len(frb_symmetries),
    }

    if len(frb_symmetries) > 3:
        wow_sym = wow_temporal["temporal_symmetry"]["correlation"]
        frb_mean = np.mean(frb_symmetries)
        frb_std = np.std(frb_symmetries) + 1e-10
        z_symmetry = (wow_sym - frb_mean) / frb_std

        results["symmetry_comparison"] = {
            "wow": float(wow_sym),
            "frb_mean": float(frb_mean),
            "frb_std": float(frb_std),
            "z_score": float(z_symmetry),
        }

    if len(frb_autocorr_peaks) > 3:
        wow_peaks = wow_temporal["autocorrelation"]["n_significant_peaks"]
        frb_mean = np.mean(frb_autocorr_peaks)
        frb_std = np.std(frb_autocorr_peaks) + 1e-10
        z_peaks = (wow_peaks - frb_mean) / frb_std

        results["autocorr_comparison"] = {
            "wow": int(wow_peaks),
            "frb_mean": float(frb_mean),
            "frb_std": float(frb_std),
            "z_score": float(z_peaks),
        }

    return results


def main():
    print("=" * 60)
    print("Experiment 62: Temporal Analysis")
    print("=" * 60)
    print("\nQuestion: Why 72 seconds? Is there structure in time?")

    # 1. Analyze the number 72
    print("\n1. Analyzing the number 72...")
    n72 = analyze_72()

    print(f"\n   72 = {n72['factorization']}")
    print(f"   Divisors: {n72['factors']}")
    print(f"   72/phi = {n72['relationships']['72/phi']:.3f} ≈ {n72['72/phi_nearest_int']} ({n72['72/phi_error']*100:.1f}% error)")
    print(f"   72/pi = {n72['relationships']['72/pi']:.3f} ≈ {n72['72/pi_nearest_int']} ({n72['72/pi_error']*100:.1f}% error)")
    print(f"   Note: 72° = 360°/5 (pentagonal angle, phi-related)")

    # 2. Load Wow! signal
    print("\n2. Loading Wow! signal...")
    wow = load_wow_signal()
    print(f"   Shape: {wow.shape}")

    # 3. Temporal structure analysis
    print("\n3. Analyzing temporal structure...")
    wow_temporal = analyze_temporal_structure(wow)

    print(f"\n   Time profile:")
    print(f"      Peak at position {wow_temporal['time_profile']['peak_location']} "
          f"({wow_temporal['time_profile']['peak_location_fraction']*100:.1f}% through signal)")

    print(f"\n   Temporal symmetry:")
    print(f"      First-half/Second-half correlation: {wow_temporal['temporal_symmetry']['correlation']:.3f}")
    print(f"      Is symmetric: {wow_temporal['temporal_symmetry']['is_symmetric']}")

    print(f"\n   Autocorrelation (repeating patterns):")
    print(f"      Significant peaks: {wow_temporal['autocorrelation']['n_significant_peaks']}")
    if wow_temporal['autocorrelation']['peak_lags']:
        print(f"      Peak lags: {wow_temporal['autocorrelation']['peak_lags']}")

    print(f"\n   8-fold slice analysis (72/8 = 9):")
    print(f"      S0/S1 ratios per slice: ", end="")
    for i, r in enumerate(wow_temporal['slice_analysis']['s0_s1_ratios']):
        if r is not None:
            phi_mark = " *" if abs(r - PHI) / PHI < 0.15 else ""
            print(f"{r:.2f}{phi_mark}", end=" ")
        else:
            print("N/A", end=" ")
    print()

    if wow_temporal['slice_analysis']['phi_matches']:
        print(f"      Phi matches in slices: {wow_temporal['slice_analysis']['phi_matches']}")

    print(f"\n   Boundary analysis:")
    if "error" not in wow_temporal.get("boundary_analysis", {"error": True}):
        ba = wow_temporal['boundary_analysis']
        print(f"      Left edge entropy: {ba['left_edge_entropy']:.3f}")
        print(f"      Middle entropy: {ba['middle_entropy']:.3f}")
        print(f"      Right edge entropy: {ba['right_edge_entropy']:.3f}")
        print(f"      Edges different from middle: {ba['edges_are_different']}")

    # 4. Frequency structure
    print("\n4. Analyzing frequency structure...")
    freq_structure = analyze_freq_structure(wow)
    print(f"   Frequency peaks: {freq_structure['n_freq_peaks']}")
    if freq_structure.get('peak_frequency_ratios'):
        print(f"   Peak frequency ratios: {freq_structure['peak_frequency_ratios']}")

    # 5. Compare to FRBs
    print("\n5. Comparing temporal structure to FRBs...")
    frb_comparison = compare_to_frbs(wow_temporal)

    if "symmetry_comparison" in frb_comparison:
        sc = frb_comparison["symmetry_comparison"]
        print(f"   Symmetry: Wow!={sc['wow']:.3f}, FRBs={sc['frb_mean']:.3f}±{sc['frb_std']:.3f}, z={sc['z_score']:+.1f}")

    if "autocorr_comparison" in frb_comparison:
        ac = frb_comparison["autocorr_comparison"]
        print(f"   Autocorr peaks: Wow!={ac['wow']}, FRBs={ac['frb_mean']:.1f}±{ac['frb_std']:.1f}, z={ac['z_score']:+.1f}")

    # 6. Division energy patterns
    print("\n6. Energy distribution by divisions...")
    for n_div in [8, 9, 12]:
        key = f"division_{n_div}"
        if key in wow_temporal:
            d = wow_temporal[key]
            print(f"   {n_div}-fold: std={d['std']:.3f}, max/min={d['max_min_ratio']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    findings = []

    # Check symmetry
    if wow_temporal['temporal_symmetry']['is_symmetric']:
        findings.append(f"Signal has temporal SYMMETRY (r={wow_temporal['temporal_symmetry']['correlation']:.2f})")
    else:
        findings.append(f"Signal is NOT temporally symmetric (r={wow_temporal['temporal_symmetry']['correlation']:.2f})")

    # Check repeating patterns
    n_peaks = wow_temporal['autocorrelation']['n_significant_peaks']
    if n_peaks > 0:
        findings.append(f"Found {n_peaks} repeating pattern(s) in autocorrelation")
    else:
        findings.append("No clear repeating patterns detected")

    # Check phi in slices
    if wow_temporal['slice_analysis']['phi_matches']:
        findings.append(f"Phi appears in slice(s): {[m['slice'] for m in wow_temporal['slice_analysis']['phi_matches']]}")

    # Check boundary structure
    if "boundary_analysis" in wow_temporal and "edges_are_different" in wow_temporal["boundary_analysis"]:
        if wow_temporal["boundary_analysis"]["edges_are_different"]:
            findings.append("Edges have DIFFERENT structure from middle (boundary markers?)")
        else:
            findings.append("Edges similar to middle (no boundary markers)")

    # Check 72 relationships
    if n72['72/phi_error'] < 0.05:
        findings.append(f"72/phi ≈ {n72['72/phi_nearest_int']} ({n72['72/phi_error']*100:.1f}% error)")

    print("\n" + "\n".join(f"   {i+1}. {f}" for i, f in enumerate(findings)))

    print(f"\n   INTERPRETATION:")
    print(f"   72 = 8 × 9 has rich mathematical structure.")
    print(f"   72° is the internal angle of a regular pentagon (phi-related).")

    if wow_temporal['temporal_symmetry']['correlation'] > 0.5:
        print(f"   The signal shows temporal symmetry - possibly palindromic structure.")

    if n_peaks > 2:
        print(f"   Multiple autocorrelation peaks suggest a repeating motif.")

    # Save results
    results = {
        "experiment": "exp62_temporal_analysis",
        "timestamp": datetime.now().isoformat(),
        "analysis_72": n72,
        "temporal_structure": wow_temporal,
        "frequency_structure": freq_structure,
        "frb_comparison": frb_comparison,
        "findings": findings,
    }

    output_path = RESULTS_DIR / "exp62_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n7. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
