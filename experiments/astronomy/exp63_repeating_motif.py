"""
Experiment 63: Repeating Motif Analysis

exp62 found autocorrelation peaks at lags 9 and 12.
This means there's a repeating pattern at these intervals.

Questions:
1. What is the repeating motif?
2. How similar are adjacent segments?
3. Does the motif itself encode phi/pi/e?
4. Is there variation between repetitions (message content)?
5. What does stacking/averaging reveal?

The repetition is the first layer of meaning:
- It differentiates from noise (noise doesn't repeat cleanly)
- It provides internal validation (checksum-like)
- The motif is the "word" that repeats
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.signal import correlate
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FRB_DIR = Path(__file__).parent / "data" / "raw"

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e


def extract_segments(signal, period):
    """Extract segments of given period from the signal."""
    n_freq, n_time = signal.shape
    n_segments = n_time // period

    segments = []
    for i in range(n_segments):
        start = i * period
        end = start + period
        segment = signal[:, start:end]
        segments.append(segment)

    # Handle remainder
    remainder_start = n_segments * period
    if remainder_start < n_time:
        remainder = signal[:, remainder_start:]
    else:
        remainder = None

    return segments, remainder


def compute_segment_similarity(segments):
    """Compute pairwise similarity between segments using CKA-like metric."""
    n_segments = len(segments)

    # Flatten each segment for comparison
    flat_segments = [s.flatten() for s in segments]

    # Pairwise correlations
    similarities = np.zeros((n_segments, n_segments))
    for i in range(n_segments):
        for j in range(n_segments):
            corr = np.corrcoef(flat_segments[i], flat_segments[j])[0, 1]
            similarities[i, j] = corr if not np.isnan(corr) else 0

    return similarities


def analyze_motif(segments):
    """Analyze the average motif and its properties."""
    if not segments:
        return None

    # Stack and average
    stacked = np.stack(segments, axis=0)  # (n_segments, n_freq, period)
    mean_motif = np.mean(stacked, axis=0)
    std_motif = np.std(stacked, axis=0)

    # Compute variance across repetitions (where does it vary?)
    variance_map = np.var(stacked, axis=0)

    # SVD of the mean motif
    U, S, Vh = linalg.svd(mean_motif, full_matrices=False)

    # Check for constants in motif eigenvalues
    ratios = {}
    if len(S) > 1 and S[1] > 1e-10:
        ratios["S0/S1"] = float(S[0] / S[1])
    if len(S) > 2 and S[2] > 1e-10:
        ratios["S1/S2"] = float(S[1] / S[2])
        ratios["S0/S2"] = float(S[0] / S[2])

    # Check matches to constants
    constant_matches = {}
    for name, ratio in ratios.items():
        phi_err = abs(ratio - PHI) / PHI
        pi_err = abs(ratio - PI) / PI
        e_err = abs(ratio - E) / E

        best_match = "phi" if phi_err < pi_err and phi_err < e_err else (
            "pi" if pi_err < e_err else "e"
        )
        best_err = min(phi_err, pi_err, e_err)

        constant_matches[name] = {
            "value": ratio,
            "best_match": best_match,
            "error": float(best_err),
        }

    return {
        "mean_motif_shape": mean_motif.shape,
        "eigenvalues": [float(s) for s in S[:10]],
        "ratios": ratios,
        "constant_matches": constant_matches,
        "variance_mean": float(np.mean(variance_map)),
        "variance_max": float(np.max(variance_map)),
        "snr": float(np.mean(mean_motif**2) / (np.mean(std_motif**2) + 1e-10)),
    }


def analyze_segment_differences(segments):
    """Analyze what varies between segment repetitions."""
    if len(segments) < 2:
        return None

    # Compute differences between consecutive segments
    diffs = []
    for i in range(len(segments) - 1):
        diff = segments[i+1] - segments[i]
        diffs.append(diff)

    # Stack differences
    diff_stack = np.stack(diffs, axis=0)
    mean_diff = np.mean(diff_stack, axis=0)

    # Is the difference structured or noise-like?
    U, S, Vh = linalg.svd(mean_diff, full_matrices=False)

    # Participation ratio of difference
    S_norm = S / (np.sum(S) + 1e-10)
    pr = 1.0 / (np.sum(S_norm**2) + 1e-10)

    return {
        "n_diffs": len(diffs),
        "mean_diff_norm": float(np.linalg.norm(mean_diff)),
        "diff_eigenvalues": [float(s) for s in S[:5]],
        "diff_participation_ratio": float(pr),
        "diff_is_structured": bool(pr < 10),  # Low PR = structured
    }


def compare_periods(signal, periods=[9, 12]):
    """Compare different period segmentations."""
    results = {}

    for period in periods:
        segments, remainder = extract_segments(signal, period)

        if len(segments) < 2:
            continue

        similarities = compute_segment_similarity(segments)

        # Average similarity (excluding diagonal)
        mask = ~np.eye(len(segments), dtype=bool)
        avg_similarity = float(np.mean(similarities[mask]))

        # Adjacent similarity
        adjacent_sims = [similarities[i, i+1] for i in range(len(segments)-1)]
        avg_adjacent = float(np.mean(adjacent_sims))

        # Motif analysis
        motif_analysis = analyze_motif(segments)

        # Difference analysis
        diff_analysis = analyze_segment_differences(segments)

        results[f"period_{period}"] = {
            "n_segments": len(segments),
            "segment_shape": segments[0].shape,
            "avg_similarity": avg_similarity,
            "avg_adjacent_similarity": avg_adjacent,
            "similarity_matrix": similarities.tolist(),
            "motif": motif_analysis,
            "differences": diff_analysis,
            "remainder_size": remainder.shape[1] if remainder is not None else 0,
        }

    return results


def load_frb(filepath):
    """Load an FRB from H5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            if "frb" in f:
                frb_group = f["frb"]
                if "wfall" in frb_group:
                    data = frb_group["wfall"][:]
                elif "calibrated_wfall" in frb_group:
                    data = frb_group["calibrated_wfall"][:]
                else:
                    return None
            else:
                return None

            data = data.astype(np.float64)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)

            if data.shape[0] > data.shape[1]:
                data = data.T

            return data
    except Exception:
        return None


def compare_to_frbs(wow_results, n_frbs=20):
    """Compare Wow! motif structure to FRBs."""
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:n_frbs]

    frb_similarities_9 = []
    frb_similarities_12 = []

    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None or data.shape[1] < 15:
            continue

        try:
            frb_results = compare_periods(data, periods=[9, 12])

            if "period_9" in frb_results:
                frb_similarities_9.append(frb_results["period_9"]["avg_similarity"])
            if "period_12" in frb_results:
                frb_similarities_12.append(frb_results["period_12"]["avg_similarity"])
        except Exception:
            continue

    comparison = {}

    if frb_similarities_9 and "period_9" in wow_results:
        wow_sim = wow_results["period_9"]["avg_similarity"]
        frb_mean = np.mean(frb_similarities_9)
        frb_std = np.std(frb_similarities_9) + 1e-10
        z = (wow_sim - frb_mean) / frb_std
        comparison["period_9"] = {
            "wow": float(wow_sim),
            "frb_mean": float(frb_mean),
            "frb_std": float(frb_std),
            "z_score": float(z),
        }

    if frb_similarities_12 and "period_12" in wow_results:
        wow_sim = wow_results["period_12"]["avg_similarity"]
        frb_mean = np.mean(frb_similarities_12)
        frb_std = np.std(frb_similarities_12) + 1e-10
        z = (wow_sim - frb_mean) / frb_std
        comparison["period_12"] = {
            "wow": float(wow_sim),
            "frb_mean": float(frb_mean),
            "frb_std": float(frb_std),
            "z_score": float(z),
        }

    return comparison


def main():
    print("=" * 60)
    print("Experiment 63: Repeating Motif Analysis")
    print("=" * 60)
    print("\nQuestion: What is the repeating pattern? What does it encode?")

    # Load signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    print(f"   Shape: {wow.shape}")
    n_time = wow.shape[1]
    print(f"   Temporal samples: {n_time}")

    # Compare periods
    print("\n2. Extracting segments at periods 9 and 12...")
    results = compare_periods(wow, periods=[9, 12])

    for period_key, period_data in results.items():
        period = int(period_key.split("_")[1])
        print(f"\n   === Period {period} ===")
        print(f"   Segments: {period_data['n_segments']}")
        print(f"   Segment shape: {period_data['segment_shape']}")
        print(f"   Remainder: {period_data['remainder_size']} samples")

        print(f"\n   Segment Similarity:")
        print(f"      Average (all pairs): {period_data['avg_similarity']:.3f}")
        print(f"      Average (adjacent): {period_data['avg_adjacent_similarity']:.3f}")

        if period_data['motif']:
            print(f"\n   Motif Eigenstructure:")
            motif = period_data['motif']
            print(f"      Top eigenvalues: {motif['eigenvalues'][:5]}")

            if motif['constant_matches']:
                print(f"      Constant matches:")
                for ratio_name, match in motif['constant_matches'].items():
                    mark = " <--" if match['error'] < 0.10 else ""
                    print(f"         {ratio_name} = {match['value']:.3f} ≈ {match['best_match']} ({match['error']*100:.1f}%){mark}")

            print(f"      SNR (mean/variance): {motif['snr']:.2f}")

        if period_data['differences']:
            diff = period_data['differences']
            print(f"\n   Segment Differences:")
            print(f"      Mean diff norm: {diff['mean_diff_norm']:.4f}")
            print(f"      Diff PR: {diff['diff_participation_ratio']:.2f}")
            print(f"      Diff is structured: {diff['diff_is_structured']}")

    # Compare to FRBs
    print("\n3. Comparing segment similarity to FRBs...")
    frb_comparison = compare_to_frbs(results)

    for period_key, comp in frb_comparison.items():
        print(f"   {period_key}: Wow!={comp['wow']:.3f}, FRBs={comp['frb_mean']:.3f}±{comp['frb_std']:.3f}, z={comp['z_score']:+.1f}")

    # Analyze the similarity matrix pattern
    print("\n4. Analyzing similarity matrix structure...")
    for period_key, period_data in results.items():
        sim_matrix = np.array(period_data['similarity_matrix'])

        # Is there a pattern? Check diagonal bands
        n = len(sim_matrix)
        if n > 2:
            # Main diagonal (always 1.0)
            # Off-diagonal patterns
            off_diag_1 = [sim_matrix[i, i+1] for i in range(n-1)]
            off_diag_2 = [sim_matrix[i, i+2] for i in range(n-2)] if n > 2 else []

            print(f"\n   {period_key} similarity bands:")
            print(f"      Adjacent (k=1): {np.mean(off_diag_1):.3f} ± {np.std(off_diag_1):.3f}")
            if off_diag_2:
                print(f"      Skip-one (k=2): {np.mean(off_diag_2):.3f} ± {np.std(off_diag_2):.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    findings = []

    # Check if segments are similar
    for period_key, period_data in results.items():
        period = int(period_key.split("_")[1])
        sim = period_data['avg_similarity']

        if sim > 0.5:
            findings.append(f"Period {period}: HIGH segment similarity ({sim:.2f}) - clear repetition")
        elif sim > 0.3:
            findings.append(f"Period {period}: MODERATE segment similarity ({sim:.2f}) - partial repetition")
        else:
            findings.append(f"Period {period}: LOW segment similarity ({sim:.2f}) - weak repetition")

        # Check motif constants
        if period_data['motif'] and period_data['motif']['constant_matches']:
            for ratio_name, match in period_data['motif']['constant_matches'].items():
                if match['error'] < 0.10:
                    findings.append(f"   Motif {ratio_name} ≈ {match['best_match']} ({match['error']*100:.1f}%)")

        # Check if differences are structured
        if period_data['differences'] and period_data['differences']['diff_is_structured']:
            findings.append(f"   Segment differences are STRUCTURED (not noise)")

    # FRB comparison
    for period_key, comp in frb_comparison.items():
        if abs(comp['z_score']) > 2:
            direction = "MORE" if comp['z_score'] > 0 else "LESS"
            findings.append(f"{period_key}: Wow! is {direction} repetitive than FRBs (z={comp['z_score']:+.1f})")

    print("\n" + "\n".join(f"   {i+1}. {f}" for i, f in enumerate(findings)))

    print("\n   INTERPRETATION:")

    # Determine if there's a clear repeating motif
    best_period = None
    best_sim = 0
    for period_key, period_data in results.items():
        if period_data['avg_similarity'] > best_sim:
            best_sim = period_data['avg_similarity']
            best_period = int(period_key.split("_")[1])

    if best_sim > 0.5:
        print(f"   The signal has a clear repeating motif at period {best_period}.")
        print(f"   Segment correlation: {best_sim:.2f}")
        print(f"   This confirms: the repetition is real, not autocorrelation artifact.")
    elif best_sim > 0.3:
        print(f"   The signal has partial repetition at period {best_period}.")
        print(f"   Segments are related but not identical.")
        print(f"   This could indicate: repeated structure with varying content.")
    else:
        print(f"   Segment similarity is low ({best_sim:.2f}).")
        print(f"   The autocorrelation may indicate local patterns, not global repetition.")

    # Save results
    all_results = {
        "experiment": "exp63_repeating_motif",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": wow.shape,
        "period_analysis": results,
        "frb_comparison": frb_comparison,
        "findings": findings,
        "best_period": best_period,
        "best_similarity": float(best_sim),
    }

    output_path = RESULTS_DIR / "exp63_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n5. Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
