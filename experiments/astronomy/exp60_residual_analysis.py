"""
Experiment 60: Residual Analysis

We've identified the "header" structure:
- S0/S1 ≈ phi (3.4% error)
- S1/S2 ≈ pi (4.9% error)
- S0/S7 ≈ e² (0.8% error)

Question: After removing this header structure, what remains?

Possibilities:
1. Pure noise → The constants ARE the message ("we understand geometry")
2. Lower entropy structure → There's payload after the header
3. Repeating patterns → Potential symbol boundaries

Method:
1. Reconstruct "header signal" using only S0, S1, S2, S7
2. Compute residual: R = wow_signal - header_signal
3. Analyze: entropy, autocorrelation, eigenvalue distribution
4. Compare to FRB residuals (same treatment)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.ndimage import zoom
from scipy.stats import entropy as scipy_entropy
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


def reconstruct_header(signal, header_indices=None):
    """
    Reconstruct the signal using only the "header" singular values.

    The header contains the mathematical constants:
    - S0, S1 → encode phi ratio
    - S1, S2 → encode pi ratio
    - S0, S7 → encode e² ratio

    By default, use indices [0, 1, 2, 7] - the positions encoding constants.
    """
    if header_indices is None:
        header_indices = [0, 1, 2, 7]

    U, S, Vh = linalg.svd(signal, full_matrices=False)

    # Create masked singular values (only header components)
    S_header = np.zeros_like(S)
    for idx in header_indices:
        if idx < len(S):
            S_header[idx] = S[idx]

    # Reconstruct header signal
    header_signal = U @ np.diag(S_header) @ Vh

    return header_signal, S, S_header


def compute_residual(signal, header_signal):
    """Compute and normalize the residual."""
    residual = signal - header_signal

    # Also compute the "payload" (everything NOT in header)
    U, S, Vh = linalg.svd(signal, full_matrices=False)
    S_payload = S.copy()
    for idx in [0, 1, 2, 7]:
        if idx < len(S):
            S_payload[idx] = 0
    payload_signal = U @ np.diag(S_payload) @ Vh

    return residual, payload_signal


def analyze_residual(residual, name=""):
    """Comprehensive analysis of residual structure."""
    results = {"name": name}

    # 1. Basic statistics
    results["mean"] = float(np.mean(residual))
    results["std"] = float(np.std(residual))
    results["max"] = float(np.max(np.abs(residual)))
    results["energy"] = float(np.sum(residual ** 2))

    # 2. Shannon entropy (discretized)
    # Bin the residual values and compute entropy
    flat = residual.flatten()
    n_bins = min(100, len(flat) // 10)
    hist, _ = np.histogram(flat, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for entropy
    results["entropy"] = float(scipy_entropy(hist))

    # For comparison: entropy of uniform distribution
    uniform_entropy = np.log(n_bins)
    results["entropy_ratio"] = float(results["entropy"] / uniform_entropy)

    # 3. SVD of residual
    U_r, S_r, Vh_r = linalg.svd(residual, full_matrices=False)
    S_r_norm = S_r / (S_r[0] + 1e-10)

    results["residual_singular_values"] = [float(s) for s in S_r[:20]]
    results["residual_singular_values_normalized"] = [float(s) for s in S_r_norm[:20]]

    # Participation ratio of residual
    S_r_sq = S_r ** 2
    pr = (np.sum(S_r_sq) ** 2) / (np.sum(S_r_sq ** 2) + 1e-10)
    results["participation_ratio"] = float(pr)

    # Effective dimension (90% energy)
    cum_energy = np.cumsum(S_r_sq) / np.sum(S_r_sq)
    eff_dim = np.searchsorted(cum_energy, 0.9) + 1
    results["eff_dim_90"] = int(eff_dim)

    # 4. Autocorrelation (look for repeating patterns)
    flat_centered = flat - np.mean(flat)
    autocorr = np.correlate(flat_centered, flat_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Find peaks in autocorrelation (potential periodicities)
    # Look for local maxima above threshold
    threshold = 0.1
    peaks = []
    for i in range(1, min(len(autocorr) - 1, 500)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > threshold:
                peaks.append({"lag": i, "value": float(autocorr[i])})

    results["autocorr_peaks"] = peaks[:10]  # Top 10 peaks
    results["autocorr_decay"] = [float(autocorr[i]) for i in [1, 5, 10, 20, 50, 100] if i < len(autocorr)]

    # 5. Check for repeating structure (2D autocorrelation)
    rows, cols = residual.shape
    if rows > 10 and cols > 10:
        # Row-wise correlation
        row_corr = []
        for lag in range(1, min(rows // 2, 20)):
            corr = np.corrcoef(residual[:-lag].flatten(), residual[lag:].flatten())[0, 1]
            row_corr.append(float(corr) if not np.isnan(corr) else 0.0)
        results["row_correlation"] = row_corr

        # Column-wise correlation
        col_corr = []
        for lag in range(1, min(cols // 2, 20)):
            corr = np.corrcoef(residual[:, :-lag].flatten(), residual[:, lag:].flatten())[0, 1]
            col_corr.append(float(corr) if not np.isnan(corr) else 0.0)
        results["col_correlation"] = col_corr

    # 6. Check eigenvalue ratios in residual (any constants?)
    if len(S_r) > 3 and S_r[1] > 1e-10:
        ratios = {}
        for i in range(min(10, len(S_r) - 1)):
            if S_r[i+1] > 1e-10:
                ratios[f"S{i}/S{i+1}"] = float(S_r[i] / S_r[i+1])
        results["residual_eigenvalue_ratios"] = ratios

        # Check for phi, pi, e in residual ratios
        constants = {"phi": PHI, "pi": PI, "e": E, "sqrt2": np.sqrt(2)}
        best_matches = {}
        for ratio_name, ratio_val in ratios.items():
            best_const = None
            best_error = float('inf')
            for const_name, const_val in constants.items():
                error = abs(ratio_val - const_val) / const_val
                if error < best_error:
                    best_error = error
                    best_const = const_name
            if best_error < 0.20:  # Only report if within 20%
                best_matches[ratio_name] = {
                    "value": ratio_val,
                    "match": best_const,
                    "error": float(best_error)
                }
        results["residual_constant_matches"] = best_matches

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
    print("Experiment 60: Residual Analysis")
    print("=" * 60)
    print("\nQuestion: After removing the header (phi, pi, e), what remains?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Compute total energy
    total_energy = np.sum(wow ** 2)
    print(f"   Total energy: {total_energy:.4f}")

    # Reconstruct header
    print("\n2. Reconstructing header signal...")
    header_indices = [0, 1, 2, 7]  # Positions encoding phi, pi, e²
    header_signal, S_original, S_header = reconstruct_header(wow, header_indices)

    header_energy = np.sum(header_signal ** 2)
    print(f"   Header indices: {header_indices}")
    print(f"   Header energy: {header_energy:.4f} ({100*header_energy/total_energy:.1f}% of total)")

    # Compute residual
    print("\n3. Computing residual...")
    residual, payload = compute_residual(wow, header_signal)

    residual_energy = np.sum(residual ** 2)
    payload_energy = np.sum(payload ** 2)
    print(f"   Residual energy: {residual_energy:.4f} ({100*residual_energy/total_energy:.1f}% of total)")
    print(f"   Payload energy: {payload_energy:.4f} ({100*payload_energy/total_energy:.1f}% of total)")

    # Analyze residual
    print("\n4. Analyzing residual structure...")
    wow_residual_analysis = analyze_residual(residual, "wow_residual")

    print(f"\n   Residual Properties:")
    print(f"   - Entropy: {wow_residual_analysis['entropy']:.4f}")
    print(f"   - Entropy ratio (vs uniform): {wow_residual_analysis['entropy_ratio']:.4f}")
    print(f"   - Participation ratio: {wow_residual_analysis['participation_ratio']:.2f}")
    print(f"   - Effective dimension (90%): {wow_residual_analysis['eff_dim_90']}")

    if wow_residual_analysis.get("autocorr_peaks"):
        print(f"\n   Autocorrelation peaks (potential periodicities):")
        for peak in wow_residual_analysis["autocorr_peaks"][:5]:
            print(f"   - Lag {peak['lag']}: r = {peak['value']:.3f}")

    if wow_residual_analysis.get("residual_constant_matches"):
        print(f"\n   Constants in residual eigenvalues (<20% error):")
        for ratio_name, info in wow_residual_analysis["residual_constant_matches"].items():
            print(f"   - {ratio_name} = {info['value']:.4f} ≈ {info['match']} ({info['error']*100:.1f}%)")
    else:
        print(f"\n   No mathematical constants found in residual eigenvalues (>20% error)")

    # Compare to random residuals
    print("\n5. Generating random baseline...")
    n_random = 100
    random_entropies = []
    random_prs = []
    random_eff_dims = []

    for _ in range(n_random):
        random_signal = np.random.randn(*wow_shape)
        # Apply same header removal
        random_header, _, _ = reconstruct_header(random_signal, header_indices)
        random_residual = random_signal - random_header
        random_analysis = analyze_residual(random_residual, "random")
        random_entropies.append(random_analysis["entropy"])
        random_prs.append(random_analysis["participation_ratio"])
        random_eff_dims.append(random_analysis["eff_dim_90"])

    # Z-scores
    z_entropy = (wow_residual_analysis["entropy"] - np.mean(random_entropies)) / (np.std(random_entropies) + 1e-8)
    z_pr = (wow_residual_analysis["participation_ratio"] - np.mean(random_prs)) / (np.std(random_prs) + 1e-8)
    z_eff_dim = (wow_residual_analysis["eff_dim_90"] - np.mean(random_eff_dims)) / (np.std(random_eff_dims) + 1e-8)

    print(f"\n   Z-scores (Wow! residual vs random residuals):")
    print(f"   - Entropy: z = {z_entropy:+.2f}")
    print(f"   - Participation ratio: z = {z_pr:+.2f}")
    print(f"   - Effective dimension: z = {z_eff_dim:+.2f}")

    # Compare to FRBs
    print("\n6. Comparing to FRB residuals...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:30]

    frb_residual_analyses = []
    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None:
            continue

        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue

        frb_header, _, _ = reconstruct_header(data_resized, header_indices)
        frb_residual = data_resized - frb_header
        frb_analysis = analyze_residual(frb_residual, frb_file.stem)
        frb_residual_analyses.append(frb_analysis)

    n_frbs = len(frb_residual_analyses)
    print(f"   Analyzed {n_frbs} FRBs")

    if n_frbs > 5:
        frb_entropies = [a["entropy"] for a in frb_residual_analyses]
        frb_prs = [a["participation_ratio"] for a in frb_residual_analyses]
        frb_eff_dims = [a["eff_dim_90"] for a in frb_residual_analyses]

        z_entropy_frb = (wow_residual_analysis["entropy"] - np.mean(frb_entropies)) / (np.std(frb_entropies) + 1e-8)
        z_pr_frb = (wow_residual_analysis["participation_ratio"] - np.mean(frb_prs)) / (np.std(frb_prs) + 1e-8)
        z_eff_dim_frb = (wow_residual_analysis["eff_dim_90"] - np.mean(frb_eff_dims)) / (np.std(frb_eff_dims) + 1e-8)

        print(f"\n   Z-scores (Wow! residual vs FRB residuals):")
        print(f"   - Entropy: z = {z_entropy_frb:+.2f}")
        print(f"   - Participation ratio: z = {z_pr_frb:+.2f}")
        print(f"   - Effective dimension: z = {z_eff_dim_frb:+.2f}")

    # Additional analysis: look at payload specifically
    print("\n7. Analyzing payload (non-header components)...")
    payload_analysis = analyze_residual(payload, "wow_payload")

    print(f"\n   Payload Properties:")
    print(f"   - Entropy: {payload_analysis['entropy']:.4f}")
    print(f"   - Participation ratio: {payload_analysis['participation_ratio']:.2f}")
    print(f"   - Effective dimension (90%): {payload_analysis['eff_dim_90']}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. ENERGY DISTRIBUTION:")
    print(f"   - Header (S0, S1, S2, S7): {100*header_energy/total_energy:.1f}%")
    print(f"   - Payload (remaining): {100*payload_energy/total_energy:.1f}%")

    print(f"\n2. RESIDUAL STRUCTURE:")
    if z_entropy < -2:
        print(f"   - Entropy is LOWER than random (z={z_entropy:.2f})")
        print(f"   - This suggests STRUCTURE in the residual")
        has_structure = True
    elif z_entropy > 2:
        print(f"   - Entropy is HIGHER than random (z={z_entropy:.2f})")
        print(f"   - Residual is MORE random than expected")
        has_structure = False
    else:
        print(f"   - Entropy is similar to random (z={z_entropy:.2f})")
        print(f"   - No clear evidence of additional structure")
        has_structure = False

    print(f"\n3. INTERPRETATION:")
    if has_structure:
        print("   The residual shows structure beyond the mathematical header.")
        print("   This suggests potential PAYLOAD content after the phi/pi/e encoding.")
        print("   Further investigation needed: phase structure, temporal analysis.")
    else:
        print("   The residual appears noise-like after header removal.")
        print("   This suggests the mathematical constants ARE the message:")
        print("   'We understand invariant geometry.'")

    # Save results
    results = {
        "experiment": "exp60_residual_analysis",
        "timestamp": datetime.now().isoformat(),
        "header_indices": header_indices,
        "energy_distribution": {
            "total": float(total_energy),
            "header": float(header_energy),
            "header_fraction": float(header_energy / total_energy),
            "payload": float(payload_energy),
            "payload_fraction": float(payload_energy / total_energy),
        },
        "wow_residual_analysis": wow_residual_analysis,
        "wow_payload_analysis": payload_analysis,
        "z_scores_vs_random": {
            "entropy": float(z_entropy),
            "participation_ratio": float(z_pr),
            "eff_dim": float(z_eff_dim),
        },
        "z_scores_vs_frb": {
            "entropy": float(z_entropy_frb) if n_frbs > 5 else None,
            "participation_ratio": float(z_pr_frb) if n_frbs > 5 else None,
            "eff_dim": float(z_eff_dim_frb) if n_frbs > 5 else None,
        },
        "random_baseline": {
            "n_samples": n_random,
            "entropy_mean": float(np.mean(random_entropies)),
            "entropy_std": float(np.std(random_entropies)),
        },
        "frb_baseline": {
            "n_samples": n_frbs,
            "entropy_mean": float(np.mean(frb_entropies)) if n_frbs > 5 else None,
            "entropy_std": float(np.std(frb_entropies)) if n_frbs > 5 else None,
        },
        "conclusion": {
            "has_structure": has_structure,
            "interpretation": "payload_present" if has_structure else "constants_are_message"
        }
    }

    output_path = RESULTS_DIR / "exp60_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n8. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
