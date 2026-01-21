"""
Experiment 61: Phase Structure Analysis

We've analyzed amplitude (singular values S). Now analyze phase (U, Vh matrices).

The SVD decomposition: signal = U @ diag(S) @ Vh

- S = amplitudes (we found phi, pi, e here)
- U = left singular vectors (row space orientation)
- Vh = right singular vectors (column space orientation)

Question: Is there additional structure encoded in the phase/orientation?

Approaches:
1. Angles between singular vectors
2. Correlation patterns in U and Vh
3. Constants in angular relationships
4. Comparison to random and FRBs
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
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
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)

CONSTANTS = {
    "phi": PHI,
    "pi": PI,
    "e": E,
    "sqrt2": SQRT2,
    "sqrt3": SQRT3,
    "tau": 2 * PI,
    "phi^2": PHI ** 2,
    "1/phi": 1 / PHI,
    "pi/2": PI / 2,
    "pi/4": PI / 4,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
}


def analyze_phase_structure(signal):
    """Comprehensive phase analysis of signal's SVD components."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    results = {}

    # 1. Analyze angles between consecutive U vectors
    u_angles = []
    for i in range(min(20, U.shape[1] - 1)):
        # Angle between u_i and u_{i+1}
        dot = np.clip(np.dot(U[:, i], U[:, i+1]), -1, 1)
        angle = np.arccos(np.abs(dot))  # Absolute because sign is arbitrary
        u_angles.append(float(angle))
    results["u_angles"] = u_angles

    # 2. Analyze angles between consecutive Vh vectors
    vh_angles = []
    for i in range(min(20, Vh.shape[0] - 1)):
        dot = np.clip(np.dot(Vh[i, :], Vh[i+1, :]), -1, 1)
        angle = np.arccos(np.abs(dot))
        vh_angles.append(float(angle))
    results["vh_angles"] = vh_angles

    # 3. Check if angles match constants (as fractions of pi)
    u_angle_ratios = []
    for i, angle in enumerate(u_angles[:10]):
        ratio = angle / PI
        u_angle_ratios.append({
            "index": i,
            "angle": float(angle),
            "ratio_to_pi": float(ratio),
        })
    results["u_angle_ratios"] = u_angle_ratios

    vh_angle_ratios = []
    for i, angle in enumerate(vh_angles[:10]):
        ratio = angle / PI
        vh_angle_ratios.append({
            "index": i,
            "angle": float(angle),
            "ratio_to_pi": float(ratio),
        })
    results["vh_angle_ratios"] = vh_angle_ratios

    # 4. Correlation structure in U
    # Look at how U columns correlate with each other (should be 0 for orthogonal)
    # But look at the MAGNITUDE pattern
    u_norms = np.linalg.norm(U, axis=0)
    results["u_norms"] = [float(n) for n in u_norms[:20]]

    # 5. Look for structure in the first few U vectors
    # Are there patterns or periodicities?
    u0_fft = np.abs(np.fft.fft(U[:, 0]))
    u1_fft = np.abs(np.fft.fft(U[:, 1]))
    u2_fft = np.abs(np.fft.fft(U[:, 2]))

    # Find dominant frequencies
    def find_peaks(fft_result, n_peaks=5):
        # Skip DC component
        fft_no_dc = fft_result[1:len(fft_result)//2]
        peak_indices = np.argsort(fft_no_dc)[-n_peaks:][::-1]
        return [(int(idx + 1), float(fft_no_dc[idx])) for idx in peak_indices]

    results["u0_fft_peaks"] = find_peaks(u0_fft)
    results["u1_fft_peaks"] = find_peaks(u1_fft)
    results["u2_fft_peaks"] = find_peaks(u2_fft)

    # 6. Analyze the "phase" of the payload components (S3-S6, S8+)
    # These are the non-header components
    payload_indices = [i for i in range(min(20, len(S))) if i not in [0, 1, 2, 7]]

    payload_u_entropy = []
    for idx in payload_indices[:10]:
        if idx < U.shape[1]:
            # Entropy of the U vector (discretized)
            u_vec = U[:, idx]
            hist, _ = np.histogram(u_vec, bins=20, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            payload_u_entropy.append(float(entropy))
    results["payload_u_entropy"] = payload_u_entropy

    # 7. Check for golden ratio in angle relationships
    # If angle[i] / angle[i+1] ≈ phi, that's interesting
    if len(u_angles) > 1:
        u_angle_consecutive_ratios = []
        for i in range(len(u_angles) - 1):
            if u_angles[i+1] > 1e-6:
                ratio = u_angles[i] / u_angles[i+1]
                u_angle_consecutive_ratios.append({
                    "i": i,
                    "ratio": float(ratio),
                })
        results["u_angle_consecutive_ratios"] = u_angle_consecutive_ratios

    # 8. Cross-correlation between U and Vh structure
    # Do the left and right singular vectors have related structure?
    if U.shape[1] >= 3 and Vh.shape[0] >= 3:
        # Compare the "shape" of U[:,0] to Vh[0,:]
        # Resample to same length for comparison
        u0_resampled = np.interp(
            np.linspace(0, 1, 50),
            np.linspace(0, 1, len(U[:, 0])),
            U[:, 0]
        )
        vh0_resampled = np.interp(
            np.linspace(0, 1, 50),
            np.linspace(0, 1, len(Vh[0, :])),
            Vh[0, :]
        )
        u_vh_corr = float(np.corrcoef(u0_resampled, vh0_resampled)[0, 1])
        results["u0_vh0_correlation"] = u_vh_corr if not np.isnan(u_vh_corr) else 0.0

    # 9. Check for constants in cumulative angle sums
    cumsum_u_angles = np.cumsum(u_angles)
    results["cumsum_u_angles"] = [float(x) for x in cumsum_u_angles[:10]]

    # Check if cumulative sums match constants × pi
    cumsum_matches = []
    for i, cumsum in enumerate(cumsum_u_angles[:10]):
        ratio = cumsum / PI
        # Find closest constant
        best_match = None
        best_error = float('inf')
        for name, val in CONSTANTS.items():
            error = abs(ratio - val) / (val + 1e-8)
            if error < best_error:
                best_error = error
                best_match = name
        if best_error < 0.15:  # Within 15%
            cumsum_matches.append({
                "index": i,
                "cumsum": float(cumsum),
                "ratio_to_pi": float(ratio),
                "match": best_match,
                "error": float(best_error),
            })
    results["cumsum_constant_matches"] = cumsum_matches

    return results, U, S, Vh


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


def resize_to_match(data, target_shape):
    """Resize data to match target shape."""
    if data is None or data.size == 0:
        return None
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    try:
        return zoom(data, zoom_factors, order=1)
    except Exception:
        return None


def main():
    print("=" * 60)
    print("Experiment 61: Phase Structure Analysis")
    print("=" * 60)
    print("\nQuestion: Is there structure encoded in the phase (U, Vh)?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Analyze phase structure
    print("\n2. Analyzing phase structure...")
    wow_phase, U, S, Vh = analyze_phase_structure(wow)

    print(f"\n   Angles between consecutive U vectors (first 10):")
    for i, angle in enumerate(wow_phase["u_angles"][:10]):
        ratio = angle / PI
        print(f"   U{i}→U{i+1}: {angle:.4f} rad = {ratio:.4f}π")

    print(f"\n   Angles between consecutive Vh vectors (first 10):")
    for i, angle in enumerate(wow_phase["vh_angles"][:10]):
        ratio = angle / PI
        print(f"   Vh{i}→Vh{i+1}: {angle:.4f} rad = {ratio:.4f}π")

    # Check for constants in angle ratios
    print(f"\n3. Checking for mathematical constants in angles...")

    if wow_phase.get("cumsum_constant_matches"):
        print(f"\n   Cumulative angle sums matching constants (<15% error):")
        for match in wow_phase["cumsum_constant_matches"]:
            print(f"   Sum(angles 0→{match['index']}): {match['cumsum']:.4f} = {match['ratio_to_pi']:.4f}π ≈ {match['match']} ({match['error']*100:.1f}%)")
    else:
        print("   No strong constant matches in cumulative angles")

    # Check consecutive angle ratios
    print(f"\n   Consecutive angle ratios (looking for phi, etc.):")
    if wow_phase.get("u_angle_consecutive_ratios"):
        for item in wow_phase["u_angle_consecutive_ratios"][:5]:
            ratio = item["ratio"]
            # Check against constants
            for name, val in CONSTANTS.items():
                if abs(ratio - val) / val < 0.10:
                    print(f"   angle[{item['i']}]/angle[{item['i']+1}] = {ratio:.4f} ≈ {name} ({abs(ratio-val)/val*100:.1f}%)")
                    break

    # FFT analysis
    print(f"\n4. Frequency analysis of U vectors...")
    print(f"   U0 dominant frequencies: {wow_phase['u0_fft_peaks'][:3]}")
    print(f"   U1 dominant frequencies: {wow_phase['u1_fft_peaks'][:3]}")
    print(f"   U2 dominant frequencies: {wow_phase['u2_fft_peaks'][:3]}")

    # Generate random baseline
    print("\n5. Generating random baseline...")
    n_random = 100
    random_u_angles_all = []
    random_vh_angles_all = []

    for _ in range(n_random):
        random_signal = np.random.randn(*wow_shape)
        random_phase, _, _, _ = analyze_phase_structure(random_signal)
        random_u_angles_all.append(random_phase["u_angles"])
        random_vh_angles_all.append(random_phase["vh_angles"])

    # Compute statistics
    random_u_angles_mean = np.mean([a[0] for a in random_u_angles_all if len(a) > 0])
    random_u_angles_std = np.std([a[0] for a in random_u_angles_all if len(a) > 0])

    wow_u_angle_0 = wow_phase["u_angles"][0] if wow_phase["u_angles"] else 0
    z_u_angle = (wow_u_angle_0 - random_u_angles_mean) / (random_u_angles_std + 1e-8)

    print(f"\n   First U angle: Wow! = {wow_u_angle_0:.4f}, Random = {random_u_angles_mean:.4f}±{random_u_angles_std:.4f}")
    print(f"   Z-score: {z_u_angle:+.2f}")

    # Compare to FRBs
    print("\n6. Comparing to FRBs...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:30]

    frb_u_angles_all = []
    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None:
            continue
        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue
        frb_phase, _, _, _ = analyze_phase_structure(data_resized)
        frb_u_angles_all.append(frb_phase["u_angles"])

    n_frbs = len(frb_u_angles_all)
    print(f"   Analyzed {n_frbs} FRBs")

    if n_frbs > 5:
        frb_u_angle_0_mean = np.mean([a[0] for a in frb_u_angles_all if len(a) > 0])
        frb_u_angle_0_std = np.std([a[0] for a in frb_u_angles_all if len(a) > 0])
        z_u_angle_frb = (wow_u_angle_0 - frb_u_angle_0_mean) / (frb_u_angle_0_std + 1e-8)

        print(f"   First U angle: Wow! = {wow_u_angle_0:.4f}, FRBs = {frb_u_angle_0_mean:.4f}±{frb_u_angle_0_std:.4f}")
        print(f"   Z-score vs FRBs: {z_u_angle_frb:+.2f}")

    # Special analysis: look for phi in phase relationships
    print("\n7. Special analysis: searching for phi in phase structure...")

    # Check if any angle is close to pi/phi or phi*something
    phi_matches = []
    for i, angle in enumerate(wow_phase["u_angles"][:15]):
        # Check various phi-related values
        targets = {
            "pi/phi": PI / PHI,
            "phi": PHI,
            "1/phi": 1 / PHI,
            "pi*phi/10": PI * PHI / 10,
        }
        for name, target in targets.items():
            if abs(angle - target) / target < 0.10:
                phi_matches.append({
                    "index": i,
                    "angle": float(angle),
                    "match": name,
                    "target": float(target),
                    "error": float(abs(angle - target) / target),
                })

    if phi_matches:
        print(f"\n   Phi-related matches in U angles (<10% error):")
        for match in phi_matches:
            print(f"   angle[{match['index']}] = {match['angle']:.4f} ≈ {match['match']} = {match['target']:.4f} ({match['error']*100:.1f}%)")
    else:
        print("   No direct phi matches in angles")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. PHASE STRUCTURE:")
    print(f"   - U angles range: {min(wow_phase['u_angles']):.4f} to {max(wow_phase['u_angles']):.4f}")
    print(f"   - Vh angles range: {min(wow_phase['vh_angles']):.4f} to {max(wow_phase['vh_angles']):.4f}")

    print(f"\n2. VS RANDOM:")
    if abs(z_u_angle) > 2:
        print(f"   - First U angle is DIFFERENT from random (z = {z_u_angle:+.2f})")
        phase_unique = True
    else:
        print(f"   - First U angle is SIMILAR to random (z = {z_u_angle:+.2f})")
        phase_unique = False

    print(f"\n3. VS FRBs:")
    if n_frbs > 5 and abs(z_u_angle_frb) > 2:
        print(f"   - First U angle is DIFFERENT from FRBs (z = {z_u_angle_frb:+.2f})")
        frb_different = True
    else:
        print(f"   - First U angle is SIMILAR to FRBs (z = {z_u_angle_frb:+.2f})")
        frb_different = False

    print(f"\n4. CONSTANT ENCODING:")
    if wow_phase.get("cumsum_constant_matches"):
        print(f"   - Found {len(wow_phase['cumsum_constant_matches'])} cumulative angle matches")
        print(f"   - Structure MAY exist in phase")
    else:
        print(f"   - No clear constants in phase angles")
        print(f"   - Phase may be 'carrier' for amplitude-encoded information")

    # Save results
    results = {
        "experiment": "exp61_phase_structure",
        "timestamp": datetime.now().isoformat(),
        "wow_phase_analysis": wow_phase,
        "z_scores": {
            "u_angle_vs_random": float(z_u_angle),
            "u_angle_vs_frb": float(z_u_angle_frb) if n_frbs > 5 else None,
        },
        "phi_matches": phi_matches,
        "conclusion": {
            "phase_unique_vs_random": phase_unique,
            "phase_different_vs_frb": frb_different if n_frbs > 5 else None,
        }
    }

    output_path = RESULTS_DIR / "exp61_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n8. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
