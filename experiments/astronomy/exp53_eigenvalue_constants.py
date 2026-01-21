"""
Experiment 53: Eigenvalue Constant Analysis

exp52 found that Wow! eigenvalue ratios match mathematical constants:
- S0/S1 ≈ phi (1.563 vs 1.618)
- S1/S2 ≈ pi (3.294 vs 3.142)
- S0/S2 ≈ pi*phi (5.149 vs 5.083)

Questions:
1. Are these ratios unique to Wow! or common in FRBs?
2. How close are they to the exact constants?
3. Is this statistically significant?
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

# Key constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618...
PI = np.pi                   # 3.14159...
E = np.e                     # 2.71828...
TAU = 2 * np.pi              # 6.28318...
PI_PHI = PI * PHI            # 5.083...


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

    except Exception as e:
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


def compute_eigenvalue_ratios(signal):
    """Compute key eigenvalue ratios."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    if len(S) < 5:
        return None

    return {
        "S0": float(S[0]),
        "S1": float(S[1]),
        "S2": float(S[2]),
        "S3": float(S[3]),
        "S4": float(S[4]),
        "S0_S1": float(S[0] / S[1]) if S[1] > 0 else None,  # Should be ≈ phi?
        "S1_S2": float(S[1] / S[2]) if S[2] > 0 else None,  # Should be ≈ pi?
        "S0_S2": float(S[0] / S[2]) if S[2] > 0 else None,  # Should be ≈ pi*phi?
        "S0_S4": float(S[0] / S[4]) if S[4] > 0 else None,  # Should be ≈ tau?
    }


def compute_constant_errors(ratios):
    """Compute how far each ratio is from the expected constant."""
    if ratios is None:
        return None

    return {
        "S0_S1_vs_phi": abs(ratios["S0_S1"] - PHI) / PHI if ratios["S0_S1"] else None,
        "S1_S2_vs_pi": abs(ratios["S1_S2"] - PI) / PI if ratios["S1_S2"] else None,
        "S0_S2_vs_pi_phi": abs(ratios["S0_S2"] - PI_PHI) / PI_PHI if ratios["S0_S2"] else None,
        "S0_S4_vs_tau": abs(ratios["S0_S4"] - TAU) / TAU if ratios["S0_S4"] else None,
    }


def main():
    print("=" * 60)
    print("Experiment 53: Eigenvalue Constant Analysis")
    print("=" * 60)
    print("\nQuestion: Are Wow!'s eigenvalue ratios (phi, pi) unique?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Compute Wow! ratios
    print("\n2. Computing Wow! eigenvalue ratios...")
    wow_ratios = compute_eigenvalue_ratios(wow)
    wow_errors = compute_constant_errors(wow_ratios)

    print(f"\n   Wow! eigenvalue ratios:")
    print(f"      S0/S1 = {wow_ratios['S0_S1']:.4f} (phi = {PHI:.4f}, error = {wow_errors['S0_S1_vs_phi']*100:.2f}%)")
    print(f"      S1/S2 = {wow_ratios['S1_S2']:.4f} (pi  = {PI:.4f}, error = {wow_errors['S1_S2_vs_pi']*100:.2f}%)")
    print(f"      S0/S2 = {wow_ratios['S0_S2']:.4f} (pi*phi = {PI_PHI:.4f}, error = {wow_errors['S0_S2_vs_pi_phi']*100:.2f}%)")
    print(f"      S0/S4 = {wow_ratios['S0_S4']:.4f} (tau = {TAU:.4f}, error = {wow_errors['S0_S4_vs_tau']*100:.2f}%)")

    # Load and analyze FRBs
    print("\n3. Analyzing FRBs...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))
    print(f"   Found {len(frb_files)} FRB files")

    frb_ratios_list = []
    frb_errors_list = []

    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None:
            continue

        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue

        ratios = compute_eigenvalue_ratios(data_resized)
        if ratios is None:
            continue

        errors = compute_constant_errors(ratios)
        if errors is None:
            continue

        frb_ratios_list.append(ratios)
        frb_errors_list.append(errors)

    n_frbs = len(frb_ratios_list)
    print(f"   Successfully analyzed {n_frbs} FRBs")

    # Compute statistics
    print("\n4. Computing statistics...")

    if n_frbs > 0:
        # Collect FRB ratio values
        frb_s0_s1 = [r["S0_S1"] for r in frb_ratios_list if r["S0_S1"]]
        frb_s1_s2 = [r["S1_S2"] for r in frb_ratios_list if r["S1_S2"]]
        frb_s0_s2 = [r["S0_S2"] for r in frb_ratios_list if r["S0_S2"]]

        # Collect FRB errors
        frb_phi_errors = [e["S0_S1_vs_phi"] for e in frb_errors_list if e["S0_S1_vs_phi"]]
        frb_pi_errors = [e["S1_S2_vs_pi"] for e in frb_errors_list if e["S1_S2_vs_pi"]]
        frb_pi_phi_errors = [e["S0_S2_vs_pi_phi"] for e in frb_errors_list if e["S0_S2_vs_pi_phi"]]

        # Statistics
        stats = {
            "S0_S1": {
                "wow": wow_ratios["S0_S1"],
                "frb_mean": float(np.mean(frb_s0_s1)),
                "frb_std": float(np.std(frb_s0_s1)),
                "target": PHI,
                "wow_error": wow_errors["S0_S1_vs_phi"],
                "frb_error_mean": float(np.mean(frb_phi_errors)),
                "frb_error_std": float(np.std(frb_phi_errors)),
            },
            "S1_S2": {
                "wow": wow_ratios["S1_S2"],
                "frb_mean": float(np.mean(frb_s1_s2)),
                "frb_std": float(np.std(frb_s1_s2)),
                "target": PI,
                "wow_error": wow_errors["S1_S2_vs_pi"],
                "frb_error_mean": float(np.mean(frb_pi_errors)),
                "frb_error_std": float(np.std(frb_pi_errors)),
            },
            "S0_S2": {
                "wow": wow_ratios["S0_S2"],
                "frb_mean": float(np.mean(frb_s0_s2)),
                "frb_std": float(np.std(frb_s0_s2)),
                "target": PI_PHI,
                "wow_error": wow_errors["S0_S2_vs_pi_phi"],
                "frb_error_mean": float(np.mean(frb_pi_phi_errors)),
                "frb_error_std": float(np.std(frb_pi_phi_errors)),
            },
        }

        # Compute z-scores
        for key in stats:
            s = stats[key]
            if s["frb_std"] > 0:
                # Z-score of Wow!'s ratio vs FRB distribution
                s["z_ratio"] = (s["wow"] - s["frb_mean"]) / s["frb_std"]
                # Z-score of Wow!'s error vs FRB error distribution
                s["z_error"] = (s["wow_error"] - s["frb_error_mean"]) / (s["frb_error_std"] + 1e-8)
            else:
                s["z_ratio"] = 0
                s["z_error"] = 0

        print("\n   Comparison: Wow! vs FRBs")
        print("   " + "-" * 70)
        print(f"   {'Ratio':<10} {'Wow!':<10} {'FRBs':<20} {'Target':<10} {'Z-score':<10}")
        print("   " + "-" * 70)

        for key in ["S0_S1", "S1_S2", "S0_S2"]:
            s = stats[key]
            target_name = {"S0_S1": "phi", "S1_S2": "pi", "S0_S2": "pi*phi"}[key]
            print(f"   {key:<10} {s['wow']:<10.4f} {s['frb_mean']:.4f} +/- {s['frb_std']:.4f}   {target_name}={s['target']:.4f}  z={s['z_ratio']:+.2f}")

        print("\n   Error comparison (closeness to constant):")
        print("   " + "-" * 70)
        for key in ["S0_S1", "S1_S2", "S0_S2"]:
            s = stats[key]
            target_name = {"S0_S1": "phi", "S1_S2": "pi", "S0_S2": "pi*phi"}[key]
            print(f"   {key} vs {target_name:<7}: Wow!={s['wow_error']*100:.2f}%, FRBs={s['frb_error_mean']*100:.2f}% +/- {s['frb_error_std']*100:.2f}%  z={s['z_error']:+.2f}")

        # Check how many FRBs are closer to constants than Wow!
        print("\n5. How many FRBs are CLOSER to constants than Wow!?")

        n_closer_phi = sum(1 for e in frb_phi_errors if e < wow_errors["S0_S1_vs_phi"])
        n_closer_pi = sum(1 for e in frb_pi_errors if e < wow_errors["S1_S2_vs_pi"])
        n_closer_pi_phi = sum(1 for e in frb_pi_phi_errors if e < wow_errors["S0_S2_vs_pi_phi"])

        print(f"   S0/S1 vs phi: {n_closer_phi}/{n_frbs} FRBs are closer ({n_closer_phi/n_frbs*100:.1f}%)")
        print(f"   S1/S2 vs pi:  {n_closer_pi}/{n_frbs} FRBs are closer ({n_closer_pi/n_frbs*100:.1f}%)")
        print(f"   S0/S2 vs pi*phi: {n_closer_pi_phi}/{n_frbs} FRBs are closer ({n_closer_pi_phi/n_frbs*100:.1f}%)")

        # Combined score
        wow_combined_error = wow_errors["S0_S1_vs_phi"] + wow_errors["S1_S2_vs_pi"] + wow_errors["S0_S2_vs_pi_phi"]
        frb_combined_errors = [e["S0_S1_vs_phi"] + e["S1_S2_vs_pi"] + e["S0_S2_vs_pi_phi"]
                               for e in frb_errors_list
                               if e["S0_S1_vs_phi"] and e["S1_S2_vs_pi"] and e["S0_S2_vs_pi_phi"]]

        n_closer_combined = sum(1 for e in frb_combined_errors if e < wow_combined_error)
        percentile = (1 - n_closer_combined / len(frb_combined_errors)) * 100 if frb_combined_errors else 0

        print(f"\n   Combined error (all three ratios):")
        print(f"      Wow! = {wow_combined_error*100:.2f}%")
        print(f"      FRBs = {np.mean(frb_combined_errors)*100:.2f}% +/- {np.std(frb_combined_errors)*100:.2f}%")
        print(f"      Wow! is at {percentile:.1f}th percentile (lower = closer to constants)")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nWow! eigenvalue ratios vs mathematical constants:")
    print(f"   S0/S1 = {wow_ratios['S0_S1']:.4f} ≈ phi ({PHI:.4f}) - error {wow_errors['S0_S1_vs_phi']*100:.1f}%")
    print(f"   S1/S2 = {wow_ratios['S1_S2']:.4f} ≈ pi  ({PI:.4f}) - error {wow_errors['S1_S2_vs_pi']*100:.1f}%")
    print(f"   S0/S2 = {wow_ratios['S0_S2']:.4f} ≈ pi*phi ({PI_PHI:.4f}) - error {wow_errors['S0_S2_vs_pi_phi']*100:.1f}%")

    if n_frbs > 0:
        print(f"\n   vs FRBs:")
        print(f"      Wow! is at {percentile:.1f}th percentile for closeness to constants")
        if percentile > 80:
            print(f"      --> Wow! is UNUSUALLY close to phi/pi/tau ratios compared to FRBs!")
            unique = True
        elif percentile > 50:
            print(f"      --> Wow! is ABOVE AVERAGE for closeness to constants")
            unique = False
        else:
            print(f"      --> Wow! is NOT unusually close to constants (FRBs are similar or closer)")
            unique = False

    # Save results
    results = {
        "experiment": "exp53_eigenvalue_constants",
        "timestamp": datetime.now().isoformat(),
        "wow_ratios": wow_ratios,
        "wow_errors": wow_errors,
        "constants": {
            "phi": float(PHI),
            "pi": float(PI),
            "pi_phi": float(PI_PHI),
            "tau": float(TAU),
        },
        "frb_analysis": {
            "n_frbs": n_frbs,
            "stats": stats if n_frbs > 0 else None,
            "percentile": float(percentile) if n_frbs > 0 else None,
            "n_closer_phi": n_closer_phi if n_frbs > 0 else None,
            "n_closer_pi": n_closer_pi if n_frbs > 0 else None,
            "n_closer_pi_phi": n_closer_pi_phi if n_frbs > 0 else None,
        },
    }

    output_path = RESULTS_DIR / "exp53_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
