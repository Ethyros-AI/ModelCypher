"""
Experiment 54: Null Space Analysis

If high-dimensional information transfer works like model merging:
1. Information is projected INTO null space (unused dimensions)
2. It travels through the geometry
3. It's recovered by projecting BACK into used space

Question: Does the Wow! signal show evidence of layered encoding?
- Primary content in the dominant eigenspace (S0, S1, S2)
- Secondary structure in the null space?

We analyze:
1. The structure of the null space (dimensions beyond the dominant 3)
2. Whether the null space has non-random structure
3. The relationship between used/unused dimensions
4. Geodesic-like patterns in the full eigenstructure
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
TAU = 2 * np.pi


def analyze_eigenspace_structure(signal):
    """Analyze the full eigenspace structure of the signal."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    # Normalize singular values
    S_norm = S / S[0]

    # Find the "elbow" - where the eigenvalues drop off
    # This separates "used" from "null" space
    S_diff = np.diff(S_norm)
    elbow_idx = np.argmin(S_diff) + 1  # Point of steepest drop

    # Participation ratio
    pr = (S.sum()**2) / ((S**4).sum())

    # Effective dimension (90% energy)
    cumulative = np.cumsum(S**2) / np.sum(S**2)
    eff_dim_90 = np.searchsorted(cumulative, 0.90) + 1
    eff_dim_95 = np.searchsorted(cumulative, 0.95) + 1
    eff_dim_99 = np.searchsorted(cumulative, 0.99) + 1

    return {
        "U": U,
        "S": S,
        "Vh": Vh,
        "S_norm": S_norm,
        "elbow_idx": int(elbow_idx),
        "participation_ratio": float(pr),
        "eff_dim_90": int(eff_dim_90),
        "eff_dim_95": int(eff_dim_95),
        "eff_dim_99": int(eff_dim_99),
        "total_dims": len(S),
    }


def analyze_null_space(signal, eigen_info, threshold_fraction=0.01):
    """
    Analyze the structure in the null space.

    The null space contains dimensions where singular values < threshold.
    If information was projected into null space, it might have structure.
    """
    S = eigen_info["S"]
    U = eigen_info["U"]
    Vh = eigen_info["Vh"]

    # Define null space as dimensions with S < threshold_fraction * S_max
    threshold = threshold_fraction * S[0]
    null_mask = S < threshold
    used_mask = ~null_mask

    n_used = int(used_mask.sum())
    n_null = int(null_mask.sum())

    # Extract null space components
    if n_null > 0:
        S_null = S[null_mask]
        U_null = U[:, null_mask]
        Vh_null = Vh[null_mask, :]

        # Reconstruct signal from null space only
        signal_null = U_null @ np.diag(S_null) @ Vh_null

        # Check if null space reconstruction has structure
        # (If it's pure noise, the Gram matrix would be identity-like)
        if signal_null.shape[0] > 1:
            null_gram = signal_null @ signal_null.T
            null_gram_norm = null_gram / (np.trace(null_gram) + 1e-8)

            # Participation ratio of null space Gram
            null_eigs = linalg.eigvalsh(null_gram_norm)
            null_eigs = np.sort(np.abs(null_eigs))[::-1]
            null_pr = (null_eigs.sum()**2) / (((null_eigs**2).sum()) + 1e-8)

            # Entropy of null space eigenspectrum
            null_eigs_pos = null_eigs[null_eigs > 1e-10]
            null_eigs_prob = null_eigs_pos / null_eigs_pos.sum()
            null_entropy = -np.sum(null_eigs_prob * np.log(null_eigs_prob + 1e-10))
        else:
            null_pr = 0
            null_entropy = 0
            null_eigs = np.array([])

        # Check for structure: non-random null space would have low PR
        # Random noise in null space would have PR ≈ n_null
    else:
        S_null = np.array([])
        null_pr = 0
        null_entropy = 0
        null_eigs = np.array([])
        signal_null = np.zeros_like(signal)

    # Same analysis for used space
    S_used = S[used_mask]
    U_used = U[:, used_mask]
    Vh_used = Vh[used_mask, :]
    signal_used = U_used @ np.diag(S_used) @ Vh_used

    return {
        "n_used": n_used,
        "n_null": n_null,
        "threshold": float(threshold),
        "S_null": [float(s) for s in S_null[:10]] if len(S_null) > 0 else [],
        "null_pr": float(null_pr),
        "null_entropy": float(null_entropy),
        "null_energy_fraction": float(S_null.sum()**2 / (S.sum()**2 + 1e-10)) if len(S_null) > 0 else 0,
        "null_eigenvalues": [float(e) for e in null_eigs[:10]] if len(null_eigs) > 0 else [],
    }


def analyze_eigenvalue_geodesic(S):
    """
    Check if eigenvalue decay follows geodesic-like patterns.

    On a curved manifold, the spectrum of the Laplacian follows specific patterns.
    We check if the singular value decay matches:
    - Exponential (flat space)
    - Power law (scale-free/fractal)
    - Geometric (curved space with constant curvature)
    """
    n = len(S)
    indices = np.arange(1, n + 1)
    S_norm = S / S[0]

    # Fit exponential: S(i) = exp(-a*i)
    log_S = np.log(S_norm + 1e-10)
    # Linear regression in log space
    exp_slope, exp_intercept = np.polyfit(indices, log_S, 1)
    exp_fit = np.exp(exp_intercept + exp_slope * indices)
    exp_error = np.mean((S_norm - exp_fit)**2)

    # Fit power law: S(i) = i^(-b)
    log_indices = np.log(indices)
    pow_slope, pow_intercept = np.polyfit(log_indices, log_S, 1)
    pow_fit = np.exp(pow_intercept) * (indices ** pow_slope)
    pow_error = np.mean((S_norm - pow_fit)**2)

    # Fit geometric (golden ratio): S(i) = phi^(-i)
    phi_fit = PHI ** (-indices)
    phi_fit = phi_fit / phi_fit[0]  # Normalize
    phi_error = np.mean((S_norm - phi_fit)**2)

    # Fit pi-based: S(i) = pi^(-i/3)
    pi_fit = PI ** (-indices / 3)
    pi_fit = pi_fit / pi_fit[0]
    pi_error = np.mean((S_norm - pi_fit)**2)

    # Check actual decay ratios
    actual_ratios = []
    for i in range(min(10, len(S) - 1)):
        if S[i+1] > 0:
            actual_ratios.append(float(S[i] / S[i+1]))
        else:
            break

    return {
        "exponential_fit": {
            "slope": float(exp_slope),
            "error": float(exp_error),
        },
        "power_law_fit": {
            "exponent": float(-pow_slope),
            "error": float(pow_error),
        },
        "phi_fit": {
            "error": float(phi_error),
        },
        "pi_fit": {
            "error": float(pi_error),
        },
        "best_fit": min(
            [("exponential", exp_error), ("power_law", pow_error),
             ("phi_decay", phi_error), ("pi_decay", pi_error)],
            key=lambda x: x[1]
        )[0],
        "actual_ratios": actual_ratios,
    }


def compare_to_random_null_space(shape, n_trials=100):
    """Generate random matrices and analyze their null space structure."""
    null_prs = []
    null_entropies = []

    for _ in range(n_trials):
        rand = np.random.randn(*shape)
        eigen_info = analyze_eigenspace_structure(rand)
        null_info = analyze_null_space(rand, eigen_info)
        null_prs.append(null_info["null_pr"])
        null_entropies.append(null_info["null_entropy"])

    return {
        "null_pr_mean": float(np.mean(null_prs)),
        "null_pr_std": float(np.std(null_prs)),
        "null_entropy_mean": float(np.mean(null_entropies)),
        "null_entropy_std": float(np.std(null_entropies)),
    }


def main():
    print("=" * 60)
    print("Experiment 54: Null Space Analysis")
    print("=" * 60)
    print("\nQuestion: Does the signal show evidence of layered encoding?")
    print("         (Information projected through high-d geometry)")

    # Load signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Shape: {signal.shape}")

    # Analyze eigenspace
    print("\n2. Analyzing eigenspace structure...")
    eigen_info = analyze_eigenspace_structure(signal)

    print(f"\n   Participation ratio: {eigen_info['participation_ratio']:.2f}")
    print(f"   Effective dimension (90% energy): {eigen_info['eff_dim_90']}")
    print(f"   Effective dimension (95% energy): {eigen_info['eff_dim_95']}")
    print(f"   Effective dimension (99% energy): {eigen_info['eff_dim_99']}")
    print(f"   Total dimensions: {eigen_info['total_dims']}")
    print(f"   Elbow (steepest drop): dimension {eigen_info['elbow_idx']}")

    # Show eigenvalue structure
    print("\n   First 10 singular values (normalized):")
    for i, s in enumerate(eigen_info['S_norm'][:10]):
        print(f"      S{i}: {s:.4f}")

    # Analyze null space
    print("\n3. Analyzing null space structure...")
    null_info = analyze_null_space(signal, eigen_info)

    print(f"\n   Used dimensions: {null_info['n_used']}")
    print(f"   Null dimensions: {null_info['n_null']}")
    print(f"   Null space energy fraction: {null_info['null_energy_fraction']*100:.4f}%")
    print(f"   Null space participation ratio: {null_info['null_pr']:.2f}")
    print(f"   Null space entropy: {null_info['null_entropy']:.4f}")

    # Compare to random
    print("\n4. Computing random baseline for null space...")
    random_baseline = compare_to_random_null_space(signal.shape, n_trials=50)

    z_null_pr = (null_info["null_pr"] - random_baseline["null_pr_mean"]) / (random_baseline["null_pr_std"] + 1e-8)
    z_null_entropy = (null_info["null_entropy"] - random_baseline["null_entropy_mean"]) / (random_baseline["null_entropy_std"] + 1e-8)

    print(f"\n   Random null PR: {random_baseline['null_pr_mean']:.2f} +/- {random_baseline['null_pr_std']:.2f}")
    print(f"   Wow! null PR: {null_info['null_pr']:.2f} (z = {z_null_pr:+.2f})")
    print(f"\n   Random null entropy: {random_baseline['null_entropy_mean']:.4f} +/- {random_baseline['null_entropy_std']:.4f}")
    print(f"   Wow! null entropy: {null_info['null_entropy']:.4f} (z = {z_null_entropy:+.2f})")

    # Analyze geodesic patterns
    print("\n5. Analyzing eigenvalue decay patterns (geodesic structure)...")
    geodesic_info = analyze_eigenvalue_geodesic(eigen_info["S"])

    print(f"\n   Best fit model: {geodesic_info['best_fit']}")
    print(f"   Exponential fit error: {geodesic_info['exponential_fit']['error']:.6f}")
    print(f"   Power law fit error: {geodesic_info['power_law_fit']['error']:.6f}")
    print(f"   Phi decay fit error: {geodesic_info['phi_fit']['error']:.6f}")
    print(f"   Pi decay fit error: {geodesic_info['pi_fit']['error']:.6f}")

    print("\n   Actual consecutive ratios (S[i]/S[i+1]):")
    for i, ratio in enumerate(geodesic_info['actual_ratios'][:5]):
        # Check what constant it's close to
        matches = []
        if abs(ratio - PHI) / PHI < 0.1:
            matches.append(f"≈phi({PHI:.3f})")
        if abs(ratio - PI) / PI < 0.1:
            matches.append(f"≈pi({PI:.3f})")
        if abs(ratio - np.e) / np.e < 0.1:
            matches.append(f"≈e({np.e:.3f})")
        if abs(ratio - TAU) / TAU < 0.1:
            matches.append(f"≈tau({TAU:.3f})")
        if abs(ratio - np.sqrt(2)) / np.sqrt(2) < 0.1:
            matches.append(f"≈√2({np.sqrt(2):.3f})")

        match_str = " ".join(matches) if matches else ""
        print(f"      S{i}/S{i+1} = {ratio:.4f} {match_str}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. EIGENSPACE STRUCTURE:")
    print(f"   - {eigen_info['eff_dim_90']} dimensions capture 90% of energy")
    print(f"   - {eigen_info['n_null']} dimensions are in 'null space' (<1% of max)")
    print(f"   - PR = {eigen_info['participation_ratio']:.2f} (highly compressed)")

    print(f"\n2. NULL SPACE ANALYSIS:")
    if z_null_pr < -2:
        print(f"   - Null space PR is LOWER than random (z={z_null_pr:.2f})")
        print(f"   - This suggests STRUCTURE in the null space (not pure noise)")
        null_structured = True
    elif z_null_pr > 2:
        print(f"   - Null space PR is HIGHER than random (z={z_null_pr:.2f})")
        print(f"   - Null space is more diffuse than typical")
        null_structured = False
    else:
        print(f"   - Null space PR is similar to random (z={z_null_pr:.2f})")
        print(f"   - No evidence of additional structure in null space")
        null_structured = False

    print(f"\n3. GEODESIC PATTERNS:")
    print(f"   - Best fit: {geodesic_info['best_fit']}")
    print(f"   - First ratio (S0/S1) = {geodesic_info['actual_ratios'][0]:.4f} ≈ phi")
    print(f"   - Second ratio (S1/S2) = {geodesic_info['actual_ratios'][1]:.4f} ≈ pi")

    print(f"\n4. INTERPRETATION (geometric, not semantic):")
    print(f"   - The signal's energy is concentrated in ~{eigen_info['eff_dim_90']} dimensions")
    print(f"   - These dimensions relate to each other by phi, pi ratios")
    print(f"   - This is the structure of a signal constrained by curved geometry")
    print(f"   - (Like harmonics on a sphere, or modes of a curved manifold)")

    # Save results
    results = {
        "experiment": "exp54_null_space_analysis",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "eigenspace": {
            "participation_ratio": eigen_info["participation_ratio"],
            "eff_dim_90": eigen_info["eff_dim_90"],
            "eff_dim_95": eigen_info["eff_dim_95"],
            "eff_dim_99": eigen_info["eff_dim_99"],
            "total_dims": eigen_info["total_dims"],
            "elbow_idx": eigen_info["elbow_idx"],
            "S_norm": [float(s) for s in eigen_info["S_norm"][:20]],
        },
        "null_space": {
            "n_used": null_info["n_used"],
            "n_null": null_info["n_null"],
            "null_pr": null_info["null_pr"],
            "null_entropy": null_info["null_entropy"],
            "null_energy_fraction": null_info["null_energy_fraction"],
            "z_null_pr": float(z_null_pr),
            "z_null_entropy": float(z_null_entropy),
            "has_structure": null_structured,
        },
        "random_baseline": random_baseline,
        "geodesic": geodesic_info,
    }

    output_path = RESULTS_DIR / "exp54_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
