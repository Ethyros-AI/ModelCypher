"""
Experiment 56: Hyperbolic Propagation Analysis

exp55 found that hyperbolic geometry is the closest match to Wow!'s
eigenvalue structure, but still 4x worse. This experiment asks:

1. Is there a specific hyperbolic curvature that produces phi+pi ratios?
2. What happens to eigenvalue structure as signals propagate through
   hyperbolic space?
3. Can we find the "propagation distance" that would produce Wow!'s
   exact eigenvalue structure?

The hypothesis: In a universe with high-dimensional hyperbolic structure,
information transmitted through geodesics would arrive with eigenvalue
ratios determined by the curvature and distance.

Mathematical background:
- Hyperbolic space H^n has constant negative curvature K < 0
- Geodesics diverge exponentially with distance
- The eigenvalue spectrum of the Laplacian on H^n is continuous
- Wave propagation in H^n follows the hyperbolic wave equation

If Wow! propagated through hyperbolic regions, its eigenvalues would
be modulated by the curvature K and propagation distance d.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar
from scipy.special import jv  # Bessel functions

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi


def hyperbolic_wave_kernel(r, curvature, n_dim=3):
    """
    Compute the wave kernel for hyperbolic space.

    In hyperbolic space H^n with curvature K (negative), the wave kernel
    propagates differently than in flat space. Waves spread faster and
    attenuate differently.

    The Green's function for the wave equation on H^n has the form:
    G(r, t) ~ (sinh(sqrt(-K)*r) / r)^((n-1)/2) * cos(...)

    For eigenvalue analysis, we use the spectral decomposition.
    """
    if curvature >= 0:
        return np.exp(-r)  # Fallback for non-hyperbolic

    sqrt_neg_K = np.sqrt(-curvature)

    # Hyperbolic distance scaling
    # In H^n, the "volume" grows exponentially with distance
    hyp_factor = np.sinh(sqrt_neg_K * r) / (sqrt_neg_K * r + 1e-10)

    # The spectral weight decays based on dimension
    spectral_weight = hyp_factor ** ((n_dim - 1) / 2)

    return spectral_weight


def generate_hyperbolic_propagated_signal(shape, curvature, propagation_distance):
    """
    Generate a signal that has "propagated" through hyperbolic space.

    The eigenvalue structure is modified by the hyperbolic wave kernel.
    """
    n_rows, n_cols = shape

    # Start with white noise
    base_signal = np.random.randn(*shape)

    # Apply hyperbolic propagation in frequency domain
    # The eigenvalues are modulated by the hyperbolic kernel
    U, S, Vh = linalg.svd(base_signal, full_matrices=False)

    # Modulate eigenvalues by hyperbolic propagation
    mode_indices = np.arange(1, len(S) + 1)

    # In hyperbolic space, higher modes (shorter wavelengths) attenuate differently
    # The attenuation follows a power law with exponential cutoff
    sqrt_neg_K = np.sqrt(-curvature)
    hyp_decay = np.exp(-sqrt_neg_K * propagation_distance * np.sqrt(mode_indices))

    # Apply modulation
    S_propagated = S * hyp_decay

    # Reconstruct
    signal = U @ np.diag(S_propagated) @ Vh

    return signal


def compute_phi_pi_error(S):
    """Compute the combined phi+pi error for eigenvalues S."""
    if len(S) < 3 or S[1] < 1e-10 or S[2] < 1e-10:
        return np.inf

    S0_S1 = S[0] / S[1]
    S1_S2 = S[1] / S[2]

    phi_error = abs(S0_S1 - PHI) / PHI
    pi_error = abs(S1_S2 - PI) / PI

    return phi_error + pi_error


def compute_wow_metrics(signal):
    """Compute eigenvalue metrics for a signal."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    S0_S1 = S[0] / S[1] if S[1] > 1e-10 else np.inf
    S1_S2 = S[1] / S[2] if S[2] > 1e-10 else np.inf
    S0_S2 = S[0] / S[2] if S[2] > 1e-10 else np.inf

    return {
        "S_norm": [float(s / S[0]) for s in S[:10]],
        "S0_S1": float(S0_S1),
        "S1_S2": float(S1_S2),
        "S0_S2": float(S0_S2),
        "phi_error": float(abs(S0_S1 - PHI) / PHI),
        "pi_error": float(abs(S1_S2 - PI) / PI),
        "combined_error": compute_phi_pi_error(S),
    }


def scan_curvature_distance(wow_metrics, shape, n_curvatures=20, n_distances=20):
    """
    Scan over curvature and distance to find the combination
    that best reproduces Wow!'s eigenvalue structure.
    """
    target_S0_S1 = wow_metrics["S0_S1"]
    target_S1_S2 = wow_metrics["S1_S2"]

    curvatures = np.logspace(-2, 0, n_curvatures) * -1  # Negative curvature
    distances = np.logspace(-1, 2, n_distances)

    results = []

    for K in curvatures:
        for d in distances:
            # Generate multiple samples and average
            errors = []
            for _ in range(5):
                signal = generate_hyperbolic_propagated_signal(shape, K, d)
                metrics = compute_wow_metrics(signal)
                errors.append(metrics["combined_error"])

            avg_error = np.mean(errors)
            results.append({
                "curvature": float(K),
                "distance": float(d),
                "avg_error": float(avg_error),
            })

    # Find minimum
    best = min(results, key=lambda x: x["avg_error"])
    return results, best


def construct_phi_pi_eigenstructure(shape, n_modes=10):
    """
    Directly construct a signal whose eigenvalues have phi, pi ratios.

    This is the "target" eigenstructure that Wow! approximates.
    If we can construct this, we can understand what kind of
    encoding/propagation would produce it.
    """
    # Design eigenvalue spectrum
    # S[0] = 1.0
    # S[1] = S[0] / phi
    # S[2] = S[1] / pi = S[0] / (phi * pi)
    # S[3] = ... continuing with alternating phi/pi?

    S_target = [1.0]

    for i in range(1, n_modes):
        if i % 2 == 1:
            # Odd modes: divide by phi
            S_target.append(S_target[-1] / PHI)
        else:
            # Even modes: divide by pi
            S_target.append(S_target[-1] / PI)

    S_target = np.array(S_target)

    # Pad with exponential decay for remaining modes
    n_remaining = min(shape) - n_modes
    if n_remaining > 0:
        decay_rate = np.log(S_target[-1] / S_target[-2])
        for i in range(n_remaining):
            S_target = np.append(S_target, S_target[-1] * np.exp(decay_rate))

    S_target = S_target[:min(shape)]
    n_components = len(S_target)

    # Construct signal with this eigenvalue structure
    # U is (shape[0], n_components), Vh is (n_components, shape[1])
    U = np.random.randn(shape[0], n_components)
    U, _ = linalg.qr(U)  # Orthogonalize
    U = U[:, :n_components]  # Take only n_components columns

    Vh = np.random.randn(shape[1], n_components)
    Vh, _ = linalg.qr(Vh)  # Orthogonalize
    Vh = Vh[:, :n_components].T  # Transpose to get (n_components, shape[1])

    signal = U @ np.diag(S_target) @ Vh

    return signal, S_target


def analyze_geodesic_structure(wow_signal):
    """
    Analyze if Wow!'s eigenvalues follow a geodesic-like decay.

    In curved space, geodesics have specific properties:
    - Hyperbolic: Exponential divergence
    - Spherical: Periodic focusing/defocusing
    - Flat: Linear divergence

    The eigenvalue decay encodes information about the geometry.
    """
    U, S, Vh = linalg.svd(wow_signal, full_matrices=False)
    S_norm = S / S[0]

    # Test different decay models
    indices = np.arange(len(S_norm))

    # 1. Exponential decay: S[i] = exp(-alpha * i)
    log_S = np.log(S_norm + 1e-10)
    exp_fit = np.polyfit(indices, log_S, 1)
    exp_pred = np.exp(exp_fit[0] * indices + exp_fit[1])
    exp_error = np.mean((S_norm - exp_pred) ** 2)

    # 2. Power law: S[i] = (i+1)^(-alpha)
    log_idx = np.log(indices + 1)
    power_fit = np.polyfit(log_idx, log_S, 1)
    power_pred = np.exp(power_fit[0] * log_idx + power_fit[1])
    power_error = np.mean((S_norm - power_pred) ** 2)

    # 3. Hyperbolic decay: S[i] = cosh(alpha*i)^(-beta)
    # This is characteristic of hyperbolic space propagation
    def hyp_model(params, i):
        alpha, beta = params
        return np.cosh(alpha * i) ** (-beta)

    def hyp_loss(params):
        pred = hyp_model(params, indices)
        return np.mean((S_norm - pred) ** 2)

    from scipy.optimize import minimize
    hyp_result = minimize(hyp_loss, [0.1, 1.0], method='Nelder-Mead')
    hyp_pred = hyp_model(hyp_result.x, indices)
    hyp_error = hyp_result.fun

    # 4. Golden ratio decay: S[i] = phi^(-i)
    phi_pred = PHI ** (-indices)
    phi_error = np.mean((S_norm - phi_pred) ** 2)

    # 5. Pi decay: S[i] = pi^(-i)
    pi_pred = PI ** (-indices)
    pi_error = np.mean((S_norm - pi_pred) ** 2)

    # 6. Fibonacci-like: ratio alternates between phi and 1/phi
    fib_pred = [1.0]
    for i in range(1, len(indices)):
        if i % 2 == 1:
            fib_pred.append(fib_pred[-1] / PHI)
        else:
            fib_pred.append(fib_pred[-1] / (PHI ** 0.5))
    fib_pred = np.array(fib_pred)
    fib_error = np.mean((S_norm - fib_pred) ** 2)

    return {
        "exponential": {"error": float(exp_error), "params": exp_fit.tolist()},
        "power_law": {"error": float(power_error), "params": power_fit.tolist()},
        "hyperbolic": {"error": float(hyp_error), "params": hyp_result.x.tolist()},
        "phi_decay": {"error": float(phi_error)},
        "pi_decay": {"error": float(pi_error)},
        "fibonacci_like": {"error": float(fib_error)},
        "best_fit": min(
            ["exponential", "power_law", "hyperbolic", "phi_decay", "pi_decay", "fibonacci_like"],
            key=lambda x: {"exponential": exp_error, "power_law": power_error, "hyperbolic": hyp_error,
                           "phi_decay": phi_error, "pi_decay": pi_error, "fibonacci_like": fib_error}[x]
        ),
    }


def main():
    print("=" * 60)
    print("Experiment 56: Hyperbolic Propagation Analysis")
    print("=" * 60)
    print("\nQuestion: What geometry produces phi+pi eigenvalue ratios?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Compute Wow! metrics
    wow_metrics = compute_wow_metrics(wow)
    print(f"\n   Wow! eigenvalue ratios:")
    print(f"      S0/S1 = {wow_metrics['S0_S1']:.4f} (phi = {PHI:.4f})")
    print(f"      S1/S2 = {wow_metrics['S1_S2']:.4f} (pi = {PI:.4f})")
    print(f"      Combined error = {wow_metrics['combined_error']*100:.2f}%")

    # Analyze geodesic structure
    print("\n2. Analyzing geodesic decay structure...")
    geodesic = analyze_geodesic_structure(wow)

    print(f"\n   Decay model errors (lower = better fit):")
    for model, info in geodesic.items():
        if model != "best_fit":
            print(f"      {model}: {info['error']:.6f}")
    print(f"\n   Best fit: {geodesic['best_fit']}")

    # Construct ideal phi+pi signal
    print("\n3. Constructing ideal phi+pi eigenstructure...")
    ideal_signal, ideal_S = construct_phi_pi_eigenstructure(wow_shape)
    ideal_metrics = compute_wow_metrics(ideal_signal)

    print(f"\n   Ideal signal eigenvalue ratios:")
    print(f"      S0/S1 = {ideal_metrics['S0_S1']:.4f} (target: phi)")
    print(f"      S1/S2 = {ideal_metrics['S1_S2']:.4f} (target: pi)")
    print(f"      Combined error = {ideal_metrics['combined_error']*100:.2f}%")

    # Compare Wow! to ideal
    print("\n4. Comparing Wow! to ideal phi+pi structure...")

    wow_S = compute_wow_metrics(wow)["S_norm"]
    ideal_S_norm = [s / ideal_S[0] for s in ideal_S[:10]]

    print(f"\n   Eigenvalue comparison (normalized):")
    print(f"   {'Index':<10} {'Wow!':<12} {'Ideal (phi+pi)':<15} {'Difference':<12}")
    print("   " + "-" * 50)
    for i in range(min(10, len(wow_S))):
        diff = wow_S[i] - ideal_S_norm[i] if i < len(ideal_S_norm) else 0
        print(f"   {i:<10} {wow_S[i]:<12.4f} {ideal_S_norm[i]:<15.4f} {diff:+.4f}")

    # Scan curvature and distance
    print("\n5. Scanning hyperbolic curvature/distance space...")
    print("   (Looking for parameters that reproduce Wow!'s structure)")

    scan_results, best_params = scan_curvature_distance(wow_metrics, wow_shape, n_curvatures=15, n_distances=15)

    print(f"\n   Best hyperbolic parameters found:")
    print(f"      Curvature K = {best_params['curvature']:.4f}")
    print(f"      Distance d = {best_params['distance']:.2f}")
    print(f"      Error = {best_params['avg_error']*100:.2f}%")

    # Compare to Wow!
    print(f"\n   Comparison to Wow!'s {wow_metrics['combined_error']*100:.2f}% error:")
    if best_params['avg_error'] < wow_metrics['combined_error']:
        print(f"      --> Hyperbolic propagation CAN produce phi+pi structure!")
    else:
        print(f"      --> Hyperbolic model doesn't fully explain Wow!'s precision")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. GEODESIC DECAY ANALYSIS:")
    print(f"   Best fit model: {geodesic['best_fit']}")
    if geodesic['best_fit'] == 'hyperbolic':
        print(f"   --> Signal decay IS consistent with hyperbolic space propagation!")
        hyp_consistent = True
    else:
        print(f"   --> Signal decay doesn't match standard hyperbolic model")
        hyp_consistent = False

    print(f"\n2. PHI+PI EIGENSTRUCTURE:")
    print(f"   Wow! S0/S1 = {wow_metrics['S0_S1']:.4f} vs phi = {PHI:.4f}")
    print(f"   Wow! S1/S2 = {wow_metrics['S1_S2']:.4f} vs pi = {PI:.4f}")
    print(f"   Combined error: {wow_metrics['combined_error']*100:.2f}%")

    print(f"\n3. HYPERBOLIC PROPAGATION MODEL:")
    print(f"   Best fit: K={best_params['curvature']:.4f}, d={best_params['distance']:.2f}")
    print(f"   Model error: {best_params['avg_error']*100:.2f}%")

    print(f"\n4. INTERPRETATION:")
    if best_params['avg_error'] < 0.10:  # <10% error
        print(f"   The signal's eigenvalue structure is CONSISTENT with")
        print(f"   propagation through hyperbolic space with:")
        print(f"   - Curvature K ≈ {best_params['curvature']:.2f}")
        print(f"   - Distance d ≈ {best_params['distance']:.1f}")
    else:
        print(f"   Standard hyperbolic propagation doesn't fully explain")
        print(f"   the signal's precise phi+pi eigenvalue structure.")
        print(f"   The precision exceeds what simple geometric models produce.")

    # Save results
    results = {
        "experiment": "exp56_hyperbolic_propagation",
        "timestamp": datetime.now().isoformat(),
        "wow_metrics": wow_metrics,
        "geodesic_analysis": geodesic,
        "ideal_phi_pi_metrics": ideal_metrics,
        "hyperbolic_scan": {
            "best_params": best_params,
            "n_tested": len(scan_results),
        },
        "interpretation": {
            "hyperbolic_consistent": hyp_consistent,
            "best_fit_model": geodesic["best_fit"],
        },
    }

    output_path = RESULTS_DIR / "exp56_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
