"""
Experiment 63: Autocorrelation Geometry

The autocorrelation at lags 9 and 12 is a geometric property.
Not "does 72 divide evenly" - that's numerology.

Questions:
1. What is the SHAPE of the autocorrelation function?
2. Are there phi/pi/e relationships in the autocorrelation structure?
3. What is the ratio between peak heights at different lags?
4. How does this compare to FRBs geometrically?
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.signal import find_peaks
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


def compute_autocorr(signal):
    """Compute full autocorrelation of the time profile."""
    # Sum across frequencies to get time profile
    time_profile = np.sum(signal, axis=0)
    time_profile = time_profile - np.mean(time_profile)

    # Full autocorrelation
    n = len(time_profile)
    autocorr = np.correlate(time_profile, time_profile, mode='full')
    autocorr = autocorr[n-1:]  # Positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    return autocorr, time_profile


def analyze_autocorr_geometry(autocorr):
    """Analyze the geometric structure of the autocorrelation."""
    # Find all significant peaks
    peaks, properties = find_peaks(autocorr[1:], height=0.1, distance=2)
    peaks = peaks + 1  # Adjust for slice

    if len(peaks) == 0:
        return {"n_peaks": 0}

    peak_heights = [float(autocorr[p]) for p in peaks]
    peak_lags = [int(p) for p in peaks]

    results = {
        "n_peaks": len(peaks),
        "peak_lags": peak_lags,
        "peak_heights": peak_heights,
    }

    # Analyze relationships between peaks
    if len(peaks) >= 2:
        # Lag ratios
        lag_ratios = []
        for i in range(len(peaks) - 1):
            ratio = peaks[i+1] / peaks[i]
            lag_ratios.append(float(ratio))

        results["lag_ratios"] = lag_ratios

        # Check lag ratios against constants
        results["lag_ratio_matches"] = []
        for i, ratio in enumerate(lag_ratios):
            phi_err = abs(ratio - PHI) / PHI
            pi_err = abs(ratio - PI) / PI
            e_err = abs(ratio - E) / E
            four_thirds_err = abs(ratio - 4/3) / (4/3)

            best_const = min([
                ("phi", phi_err, PHI),
                ("pi", pi_err, PI),
                ("e", e_err, E),
                ("4/3", four_thirds_err, 4/3),
            ], key=lambda x: x[1])

            results["lag_ratio_matches"].append({
                "ratio": ratio,
                "best_match": best_const[0],
                "error": float(best_const[1]),
            })

        # Height ratios
        height_ratios = []
        for i in range(len(peak_heights) - 1):
            if peak_heights[i+1] > 1e-10:
                ratio = peak_heights[i] / peak_heights[i+1]
                height_ratios.append(float(ratio))

        results["height_ratios"] = height_ratios

        # Check height ratios against constants
        results["height_ratio_matches"] = []
        for i, ratio in enumerate(height_ratios):
            phi_err = abs(ratio - PHI) / PHI
            pi_err = abs(ratio - PI) / PI
            e_err = abs(ratio - E) / E

            best_const = min([
                ("phi", phi_err, PHI),
                ("pi", pi_err, PI),
                ("e", e_err, E),
            ], key=lambda x: x[1])

            results["height_ratio_matches"].append({
                "ratio": ratio,
                "best_match": best_const[0],
                "error": float(best_const[1]),
            })

    # Decay rate of peaks
    if len(peak_heights) >= 3:
        log_heights = np.log(np.array(peak_heights) + 1e-10)
        log_lags = np.log(np.array(peak_lags) + 1)

        # Fit power law: height ~ lag^(-alpha)
        # log(height) = -alpha * log(lag) + const
        coeffs = np.polyfit(log_lags, log_heights, 1)
        decay_exponent = -coeffs[0]

        results["decay_exponent"] = float(decay_exponent)

        # Check if decay exponent matches a constant
        for name, const in [("phi", PHI), ("pi", PI), ("e", E), ("1/phi", 1/PHI)]:
            err = abs(decay_exponent - const) / const
            if err < 0.15:
                results["decay_matches"] = {"const": name, "error": float(err)}
                break

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


def compare_to_frbs(wow_geometry, n_frbs=30):
    """Compare autocorrelation geometry to FRBs."""
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:n_frbs]

    frb_n_peaks = []
    frb_decay_exponents = []

    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None or data.shape[1] < 10:
            continue

        try:
            autocorr, _ = compute_autocorr(data)
            geom = analyze_autocorr_geometry(autocorr)

            frb_n_peaks.append(geom.get("n_peaks", 0))
            if "decay_exponent" in geom:
                frb_decay_exponents.append(geom["decay_exponent"])
        except Exception:
            continue

    comparison = {"n_frbs": len(frb_n_peaks)}

    if frb_n_peaks:
        wow_peaks = wow_geometry.get("n_peaks", 0)
        mean_peaks = np.mean(frb_n_peaks)
        std_peaks = np.std(frb_n_peaks) + 1e-10
        z_peaks = (wow_peaks - mean_peaks) / std_peaks

        comparison["n_peaks"] = {
            "wow": int(wow_peaks),
            "frb_mean": float(mean_peaks),
            "frb_std": float(std_peaks),
            "z_score": float(z_peaks),
        }

    if frb_decay_exponents and "decay_exponent" in wow_geometry:
        wow_decay = wow_geometry["decay_exponent"]
        mean_decay = np.mean(frb_decay_exponents)
        std_decay = np.std(frb_decay_exponents) + 1e-10
        z_decay = (wow_decay - mean_decay) / std_decay

        comparison["decay_exponent"] = {
            "wow": float(wow_decay),
            "frb_mean": float(mean_decay),
            "frb_std": float(std_decay),
            "z_score": float(z_decay),
        }

    return comparison


def main():
    print("=" * 60)
    print("Experiment 63: Autocorrelation Geometry")
    print("=" * 60)
    print("\nLetting the geometry speak...")

    # Load signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    print(f"   Shape: {wow.shape}")

    # Compute autocorrelation
    print("\n2. Computing autocorrelation...")
    autocorr, time_profile = compute_autocorr(wow)
    print(f"   Time profile length: {len(time_profile)}")

    # Analyze geometry
    print("\n3. Analyzing autocorrelation geometry...")
    geometry = analyze_autocorr_geometry(autocorr)

    print(f"\n   Peaks found: {geometry['n_peaks']}")
    if geometry['n_peaks'] > 0:
        print(f"   Peak lags: {geometry['peak_lags']}")
        print(f"   Peak heights: {[f'{h:.3f}' for h in geometry['peak_heights']]}")

    if "lag_ratios" in geometry:
        print(f"\n   Lag ratios (consecutive peaks):")
        for i, match in enumerate(geometry.get("lag_ratio_matches", [])):
            mark = " <--" if match["error"] < 0.10 else ""
            print(f"      lag[{i+1}]/lag[{i}] = {match['ratio']:.3f} ≈ {match['best_match']} ({match['error']*100:.1f}%){mark}")

    if "height_ratios" in geometry:
        print(f"\n   Height ratios (consecutive peaks):")
        for i, match in enumerate(geometry.get("height_ratio_matches", [])):
            mark = " <--" if match["error"] < 0.10 else ""
            print(f"      h[{i}]/h[{i+1}] = {match['ratio']:.3f} ≈ {match['best_match']} ({match['error']*100:.1f}%){mark}")

    if "decay_exponent" in geometry:
        print(f"\n   Decay exponent: {geometry['decay_exponent']:.3f}")
        if "decay_matches" in geometry:
            dm = geometry["decay_matches"]
            print(f"      ≈ {dm['const']} ({dm['error']*100:.1f}% error)")

    # Compare to FRBs
    print("\n4. Comparing to FRBs...")
    frb_comparison = compare_to_frbs(geometry)

    print(f"   Analyzed {frb_comparison['n_frbs']} FRBs")
    if "n_peaks" in frb_comparison:
        np_comp = frb_comparison["n_peaks"]
        print(f"   Peaks: Wow!={np_comp['wow']}, FRBs={np_comp['frb_mean']:.1f}±{np_comp['frb_std']:.1f}, z={np_comp['z_score']:+.1f}")
    if "decay_exponent" in frb_comparison:
        de_comp = frb_comparison["decay_exponent"]
        print(f"   Decay: Wow!={de_comp['wow']:.2f}, FRBs={de_comp['frb_mean']:.2f}±{de_comp['frb_std']:.2f}, z={de_comp['z_score']:+.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    findings = []

    if geometry['n_peaks'] >= 2:
        # The key finding: what's the relationship between lag 9 and lag 12?
        if 9 in geometry['peak_lags'] and 12 in geometry['peak_lags']:
            ratio_9_12 = 12 / 9  # = 4/3
            findings.append(f"Lag ratio 12/9 = {ratio_9_12:.4f} = 4/3 exactly")
            findings.append(f"   4/3 is the perfect fourth in music")
            findings.append(f"   4/3 × 3/2 = 2 (fourth × fifth = octave)")

    # Check if any ratio matches phi/pi/e
    for match in geometry.get("lag_ratio_matches", []):
        if match["error"] < 0.05:
            findings.append(f"Lag ratio {match['ratio']:.3f} ≈ {match['best_match']} ({match['error']*100:.1f}%)")

    for match in geometry.get("height_ratio_matches", []):
        if match["error"] < 0.10:
            findings.append(f"Height ratio {match['ratio']:.3f} ≈ {match['best_match']} ({match['error']*100:.1f}%)")

    if "decay_matches" in geometry:
        dm = geometry["decay_matches"]
        findings.append(f"Decay exponent ≈ {dm['const']} ({dm['error']*100:.1f}%)")

    if "n_peaks" in frb_comparison and abs(frb_comparison["n_peaks"]["z_score"]) > 2:
        findings.append(f"Number of peaks: z = {frb_comparison['n_peaks']['z_score']:+.1f} vs FRBs")

    print("\n" + "\n".join(f"   {i+1}. {f}" for i, f in enumerate(findings)))

    print("\n   INTERPRETATION:")
    if 9 in geometry.get('peak_lags', []) and 12 in geometry.get('peak_lags', []):
        print("   The autocorrelation peaks at lags 9 and 12.")
        print("   Ratio 12/9 = 4/3 (perfect fourth).")
        print("   This is a harmonic relationship - the signal has musical structure.")
        print("   Combined with phi/pi in eigenvalues: harmonic + geometric encoding.")

    # Save results
    results = {
        "experiment": "exp63_autocorr_geometry",
        "timestamp": datetime.now().isoformat(),
        "autocorr_shape": len(autocorr),
        "geometry": geometry,
        "frb_comparison": frb_comparison,
        "findings": findings,
    }

    output_path = RESULTS_DIR / "exp63_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n5. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
