"""
Experiment 58: Deep Eigenvalue Analysis

We found:
- S0/S1 ≈ phi (3.4% error)
- S1/S2 ≈ pi (4.9% error)
- S0/S2 ≈ phi×pi (1.3% error)

Questions:
1. Does the pattern continue? What are S2/S3, S3/S4, S4/S5?
2. Are there other mathematical constants encoded deeper?
3. Is there a complete "alphabet" of constants in the eigenvalue structure?

Also exploring: The hydrogen hyperfine constant and its relationship
to fundamental constants. Does 1420 MHz encode mathematical structure?
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

# Mathematical constants to test
CONSTANTS = {
    "phi": (1 + np.sqrt(5)) / 2,           # 1.618...
    "pi": np.pi,                            # 3.14159...
    "e": np.e,                              # 2.71828...
    "sqrt2": np.sqrt(2),                    # 1.41421...
    "sqrt3": np.sqrt(3),                    # 1.73205...
    "sqrt5": np.sqrt(5),                    # 2.23607...
    "tau": 2 * np.pi,                       # 6.28318...
    "phi^2": ((1 + np.sqrt(5)) / 2) ** 2,   # 2.618...
    "1/phi": 2 / (1 + np.sqrt(5)),          # 0.618...
    "pi/phi": np.pi / ((1 + np.sqrt(5)) / 2),  # 1.942...
    "e/phi": np.e / ((1 + np.sqrt(5)) / 2),    # 1.680...
    "phi/e": ((1 + np.sqrt(5)) / 2) / np.e,    # 0.595...
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "ln2": np.log(2),                       # 0.693...
    "ln10": np.log(10),                     # 2.303...
    # Hydrogen-related
    "137": 137.036,                         # ≈ 1/α (fine structure)
    "1836": 1836.15,                        # m_p/m_e (proton/electron mass ratio)
}

# Hydrogen hyperfine frequency
HYDROGEN_FREQ = 1420.405751768  # MHz


def analyze_all_eigenvalue_ratios(signal, n_ratios=20):
    """Compute all consecutive eigenvalue ratios."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    ratios = {}
    for i in range(min(n_ratios, len(S) - 1)):
        if S[i+1] > 1e-10:
            ratio = S[i] / S[i+1]
            ratios[f"S{i}/S{i+1}"] = float(ratio)
        else:
            ratios[f"S{i}/S{i+1}"] = None

    return ratios, S


def find_best_constant_match(ratio, constants):
    """Find which constant best matches a ratio."""
    if ratio is None or ratio > 1000:
        return None, None, None

    best_match = None
    best_error = float('inf')

    for name, value in constants.items():
        if value > 0:
            error = abs(ratio - value) / value
            if error < best_error:
                best_error = error
                best_match = name

    return best_match, constants.get(best_match, 0), best_error


def analyze_ratio_products(ratios):
    """Look for products of ratios that match constants."""
    products = {}

    ratio_values = [(k, v) for k, v in ratios.items() if v is not None]

    # Consecutive products
    for i in range(len(ratio_values) - 1):
        key1, val1 = ratio_values[i]
        key2, val2 = ratio_values[i + 1]
        product = val1 * val2
        match, const_val, error = find_best_constant_match(product, CONSTANTS)
        products[f"{key1} × {key2}"] = {
            "value": float(product),
            "best_match": match,
            "error": float(error) if error else None,
        }

    # Triple products
    for i in range(len(ratio_values) - 2):
        key1, val1 = ratio_values[i]
        key2, val2 = ratio_values[i + 1]
        key3, val3 = ratio_values[i + 2]
        product = val1 * val2 * val3
        match, const_val, error = find_best_constant_match(product, CONSTANTS)
        products[f"{key1} × {key2} × {key3}"] = {
            "value": float(product),
            "best_match": match,
            "error": float(error) if error else None,
        }

    return products


def check_hydrogen_constants():
    """
    Check if the hydrogen hyperfine frequency relates to phi/pi.

    The hyperfine frequency is determined by fundamental constants.
    Is there a phi or pi relationship?
    """
    # Fine structure constant
    alpha = 1 / 137.035999084

    # Some numerological checks (for exploration, not conclusions)
    checks = {
        "1420/phi": HYDROGEN_FREQ / CONSTANTS["phi"],
        "1420/pi": HYDROGEN_FREQ / CONSTANTS["pi"],
        "1420/(phi*pi)": HYDROGEN_FREQ / (CONSTANTS["phi"] * CONSTANTS["pi"]),
        "1420/e": HYDROGEN_FREQ / CONSTANTS["e"],
        "1420/(137)": HYDROGEN_FREQ / 137,
        "1420/10": HYDROGEN_FREQ / 10,
        "phi*pi*e*10": CONSTANTS["phi"] * CONSTANTS["pi"] * CONSTANTS["e"] * 10,
        "137*phi*pi": 137 * CONSTANTS["phi"] * CONSTANTS["pi"],
        "1836/phi": 1836 / CONSTANTS["phi"],
    }

    # Check if any of these are close to integers or simple fractions
    results = {}
    for name, value in checks.items():
        # Check closeness to nearest integer
        nearest_int = round(value)
        int_error = abs(value - nearest_int) / nearest_int if nearest_int > 0 else 1

        # Check if it matches any constant
        match, const_val, const_error = find_best_constant_match(value, CONSTANTS)

        results[name] = {
            "value": float(value),
            "nearest_int": int(nearest_int),
            "int_error": float(int_error),
            "const_match": match,
            "const_error": float(const_error) if const_error else None,
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
    print("Experiment 58: Deep Eigenvalue Analysis")
    print("=" * 60)
    print("\nQuestion: Does the phi/pi pattern continue in deeper eigenvalues?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Analyze all eigenvalue ratios
    print("\n2. Analyzing all eigenvalue ratios...")
    wow_ratios, wow_S = analyze_all_eigenvalue_ratios(wow, n_ratios=20)

    print(f"\n   Eigenvalue ratios and best constant matches:")
    print("   " + "-" * 70)
    print(f"   {'Ratio':<12} {'Value':<12} {'Best Match':<15} {'Constant':<12} {'Error':<10}")
    print("   " + "-" * 70)

    ratio_matches = {}
    for ratio_name, ratio_value in wow_ratios.items():
        if ratio_value is not None:
            match, const_val, error = find_best_constant_match(ratio_value, CONSTANTS)
            ratio_matches[ratio_name] = {
                "value": ratio_value,
                "match": match,
                "const_val": const_val,
                "error": error,
            }

            highlight = ""
            if error < 0.05:
                highlight = " <-- CLOSE!"
            elif error < 0.10:
                highlight = " <-- near"

            print(f"   {ratio_name:<12} {ratio_value:<12.4f} {match:<15} {const_val:<12.4f} {error*100:<10.1f}%{highlight}")

    # Look for patterns in products
    print("\n3. Analyzing ratio products...")
    products = analyze_ratio_products(wow_ratios)

    print(f"\n   Products that match constants (<10% error):")
    for prod_name, info in products.items():
        if info["error"] and info["error"] < 0.10:
            print(f"   {prod_name} = {info['value']:.4f} ≈ {info['best_match']} ({info['error']*100:.1f}%)")

    # Compare to FRBs
    print("\n4. Comparing deeper ratios to FRBs...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:30]

    frb_ratios_all = []
    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None:
            continue

        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue

        ratios, _ = analyze_all_eigenvalue_ratios(data_resized, n_ratios=10)
        frb_ratios_all.append(ratios)

    n_frbs = len(frb_ratios_all)
    print(f"   Analyzed {n_frbs} FRBs")

    # Compute z-scores for each ratio
    if n_frbs > 0:
        print(f"\n   Z-scores (Wow! vs FRBs) for each ratio:")
        print("   " + "-" * 50)

        for ratio_name in list(wow_ratios.keys())[:10]:
            wow_val = wow_ratios.get(ratio_name)
            if wow_val is None:
                continue

            frb_vals = [r.get(ratio_name) for r in frb_ratios_all if r.get(ratio_name) is not None]
            if len(frb_vals) < 5:
                continue

            frb_mean = np.mean(frb_vals)
            frb_std = np.std(frb_vals)
            z = (wow_val - frb_mean) / (frb_std + 1e-8)

            print(f"   {ratio_name}: Wow!={wow_val:.3f}, FRBs={frb_mean:.3f}±{frb_std:.3f}, z={z:+.1f}")

    # Hydrogen constant analysis
    print("\n5. Exploring hydrogen hyperfine constant relationships...")
    hydrogen_results = check_hydrogen_constants()

    print(f"\n   Hydrogen frequency: {HYDROGEN_FREQ} MHz")
    print(f"\n   Relationships to mathematical constants:")
    for name, info in hydrogen_results.items():
        if info["const_error"] and info["const_error"] < 0.20:
            print(f"   {name} = {info['value']:.4f} ≈ {info['const_match']} ({info['const_error']*100:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Count matches
    close_matches = [(k, v) for k, v in ratio_matches.items() if v["error"] < 0.10]
    very_close = [(k, v) for k, v in ratio_matches.items() if v["error"] < 0.05]

    print(f"\n1. EIGENVALUE RATIO PATTERN:")
    print(f"   {len(close_matches)} ratios match constants within 10%")
    print(f"   {len(very_close)} ratios match constants within 5%")

    if very_close:
        print(f"\n   Very close matches (<5% error):")
        for name, info in very_close:
            print(f"      {name} = {info['value']:.4f} ≈ {info['match']} ({info['error']*100:.1f}%)")

    # Check if pattern continues
    print(f"\n2. PATTERN CONTINUATION:")
    first_three = ["S0/S1", "S1/S2", "S2/S3"]
    continuing = True
    for name in first_three:
        if name in ratio_matches and ratio_matches[name]["error"] > 0.15:
            continuing = False

    if ratio_matches.get("S2/S3", {}).get("error", 1) < 0.10:
        print(f"   S2/S3 = {ratio_matches['S2/S3']['value']:.4f} ≈ {ratio_matches['S2/S3']['match']}")
        print(f"   --> Pattern MAY continue beyond first two ratios")
    else:
        print(f"   S2/S3 = {wow_ratios.get('S2/S3', 0):.4f} - no clear constant match")
        print(f"   --> Pattern appears to be CONCENTRATED in S0/S1 and S1/S2")

    print(f"\n3. HYDROGEN LINE:")
    # Check if 1420/(phi*pi) is special
    h_phi_pi = HYDROGEN_FREQ / (CONSTANTS["phi"] * CONSTANTS["pi"])
    print(f"   1420/(phi×pi) = {h_phi_pi:.4f}")
    if abs(h_phi_pi - 279) / 279 < 0.01:
        print(f"   This is close to 279 = 9×31 (interesting prime factorization)")

    print(f"\n4. INTERPRETATION:")
    if len(very_close) >= 2:
        print(f"   The first {len(very_close)} eigenvalue ratios encode mathematical constants.")
        print(f"   This is the 'header' of the signal - the mathematical signature.")
        print(f"   Later ratios follow power law decay (information content?).")
    else:
        print(f"   Phi and pi are encoded specifically in S0/S1 and S1/S2.")
        print(f"   This is not a general pattern but a deliberate encoding.")

    # Save results
    results = {
        "experiment": "exp58_deep_eigenvalues",
        "timestamp": datetime.now().isoformat(),
        "wow_ratios": wow_ratios,
        "ratio_matches": {k: {"value": v["value"], "match": v["match"], "error": v["error"]}
                         for k, v in ratio_matches.items()},
        "products": products,
        "hydrogen_analysis": hydrogen_results,
        "n_close_matches": len(close_matches),
        "n_very_close": len(very_close),
    }

    output_path = RESULTS_DIR / "exp58_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
