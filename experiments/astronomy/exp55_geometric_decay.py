"""
Experiment 55: Geometric Decay Analysis

exp54 found that Wow!'s eigenvalue decay follows a power law with
phi and pi in the consecutive ratios. But what GEOMETRY produces this?

This experiment compares Wow!'s eigenvalue decay to:
1. Random matrices (baseline)
2. FRBs (natural radio transients)
3. Signals constrained by different geometries:
   - Spherical harmonics (signal on a sphere)
   - Hyperbolic geometry (Poincare disk)
   - Toroidal geometry (signal on a torus)
   - Fractal/self-similar (power law by construction)

If Wow! matches a specific geometry, that constrains hypotheses about
its origin - either it was transmitted through that geometry, or
constructed to encode that geometry.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.ndimage import zoom
from scipy.special import sph_harm
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FRB_DIR = Path(__file__).parent / "data" / "raw"

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi


def generate_spherical_harmonic_signal(shape, l_max=10):
    """
    Generate a signal from spherical harmonics.

    If Wow! was transmitted omnidirectionally from a spherical source,
    it might show spherical harmonic structure.
    """
    n_rows, n_cols = shape

    # Create angular coordinates
    theta = np.linspace(0, np.pi, n_rows)
    phi = np.linspace(0, 2*np.pi, n_cols)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Sum up spherical harmonics with decreasing weights
    signal = np.zeros(shape)
    for l in range(l_max):
        for m in range(-l, l+1):
            # Weight by 1/l to create natural spectrum
            weight = 1.0 / (l + 1)
            Y = sph_harm(m, l, phi_grid, theta_grid)
            signal += weight * np.real(Y)

    return signal


def generate_hyperbolic_signal(shape):
    """
    Generate a signal with hyperbolic (Poincare disk) geometry.

    Hyperbolic space has constant negative curvature.
    If the universe has hyperbolic structure at large scales,
    signals would propagate with this geometry.
    """
    n_rows, n_cols = shape

    # Create Poincare disk coordinates (within unit circle)
    x = np.linspace(-0.9, 0.9, n_cols)
    y = np.linspace(-0.9, 0.9, n_rows)
    X, Y = np.meshgrid(x, y)

    # Hyperbolic distance from center
    r_euclid = np.sqrt(X**2 + Y**2)
    r_euclid = np.clip(r_euclid, 0, 0.99)  # Avoid singularity at boundary

    # Hyperbolic metric: distances grow exponentially near boundary
    r_hyp = 2 * np.arctanh(r_euclid)

    # Create signal with hyperbolic structure
    # Multiple "modes" with hyperbolic decay
    signal = np.zeros(shape)
    for k in range(1, 10):
        signal += np.exp(-k * r_hyp) * np.cos(k * np.arctan2(Y, X))

    return signal


def generate_toroidal_signal(shape):
    """
    Generate a signal on a torus.

    A torus has different topological structure than a sphere.
    If spacetime has toroidal topology locally, this matters.
    """
    n_rows, n_cols = shape

    # Angular coordinates on torus
    u = np.linspace(0, 2*np.pi, n_rows)
    v = np.linspace(0, 2*np.pi, n_cols)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Toroidal harmonics
    R = 2  # Major radius
    r = 1  # Minor radius

    signal = np.zeros(shape)
    for n in range(1, 6):
        for m in range(1, 6):
            weight = 1.0 / (n + m)
            signal += weight * np.cos(n * U) * np.cos(m * V)

    return signal


def generate_fractal_signal(shape, exponent=1.5):
    """
    Generate a signal with 1/f^alpha (pink/brown noise) spectrum.

    Many natural phenomena follow power law spectra.
    This is the "generic" case for self-similar structures.
    """
    n_rows, n_cols = shape

    # Generate in frequency domain with power law
    freq_rows = np.fft.fftfreq(n_rows)
    freq_cols = np.fft.fftfreq(n_cols)
    freq_grid = np.sqrt(freq_rows[:, np.newaxis]**2 + freq_cols[np.newaxis, :]**2)
    freq_grid[0, 0] = 1e-10  # Avoid division by zero

    # Power law spectrum
    amplitude = 1.0 / (freq_grid ** exponent)

    # Random phases
    phases = np.random.uniform(0, 2*np.pi, shape)

    # Inverse FFT to get signal
    spectrum = amplitude * np.exp(1j * phases)
    signal = np.real(np.fft.ifft2(spectrum))

    return signal


def generate_phi_structured_signal(shape):
    """
    Generate a signal explicitly structured by golden ratio.

    If someone wanted to encode phi, what would that look like?
    """
    n_rows, n_cols = shape

    # Fibonacci-like indices
    fib_indices = [1, 2, 3, 5, 8, 13, 21, 34, 55]

    # Create signal with phi-structured components
    signal = np.zeros(shape)
    for i, idx in enumerate(fib_indices):
        if idx < min(n_rows, n_cols):
            # Each component scaled by phi^-i
            weight = PHI ** (-i)
            # Spatial frequency related to Fibonacci number
            k_row = idx % n_rows
            k_col = idx % n_cols

            x = np.linspace(0, 2*np.pi * idx, n_rows)
            y = np.linspace(0, 2*np.pi * idx, n_cols)
            X, Y = np.meshgrid(x, y, indexing='ij')
            signal += weight * np.sin(X + Y)

    return signal


def generate_pi_structured_signal(shape):
    """
    Generate a signal explicitly structured by pi.

    If someone wanted to encode pi, what would that look like?
    """
    n_rows, n_cols = shape

    # Components at pi-related frequencies
    signal = np.zeros(shape)
    for k in range(1, 10):
        weight = 1.0 / (PI ** k)  # Decay by powers of pi
        x = np.linspace(0, 2*np.pi * k, n_rows)
        y = np.linspace(0, 2*np.pi * k, n_cols)
        X, Y = np.meshgrid(x, y, indexing='ij')
        signal += weight * np.sin(PI * X + Y)

    return signal


def compute_eigenvalue_metrics(signal):
    """Compute eigenvalue decay metrics."""
    U, S, Vh = linalg.svd(signal, full_matrices=False)
    S_norm = S / S[0]

    # Consecutive ratios
    ratios = []
    for i in range(min(10, len(S)-1)):
        if S[i+1] > 1e-10:
            ratios.append(S[i] / S[i+1])
        else:
            ratios.append(np.inf)

    # Fit power law: S[i] ≈ S[0] * i^(-alpha)
    indices = np.arange(1, len(S_norm) + 1)
    log_S = np.log(S_norm + 1e-10)
    log_i = np.log(indices)

    # Linear regression in log space
    valid = ~np.isinf(log_S) & ~np.isnan(log_S)
    if valid.sum() > 2:
        coeffs = np.polyfit(log_i[valid], log_S[valid], 1)
        power_alpha = -coeffs[0]
    else:
        power_alpha = 0

    # Participation ratio
    pr = (S.sum()**2) / ((S**4).sum() + 1e-10)

    # Effective dimension (90% energy)
    cumulative = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
    eff_dim = np.searchsorted(cumulative, 0.90) + 1

    # How close are first two ratios to phi and pi?
    phi_error = abs(ratios[0] - PHI) / PHI if len(ratios) > 0 and ratios[0] < np.inf else 1.0
    pi_error = abs(ratios[1] - PI) / PI if len(ratios) > 1 and ratios[1] < np.inf else 1.0
    combined_error = phi_error + pi_error

    return {
        "S_norm": [float(s) for s in S_norm[:10]],
        "ratios": [float(r) if r < np.inf else None for r in ratios],
        "power_alpha": float(power_alpha),
        "participation_ratio": float(pr),
        "eff_dim_90": int(eff_dim),
        "phi_error": float(phi_error),
        "pi_error": float(pi_error),
        "combined_phi_pi_error": float(combined_error),
    }


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
    print("Experiment 55: Geometric Decay Analysis")
    print("=" * 60)
    print("\nQuestion: Which GEOMETRY produces Wow!'s eigenvalue decay?")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Compute Wow! metrics
    print("\n2. Computing Wow! eigenvalue metrics...")
    wow_metrics = compute_eigenvalue_metrics(wow)

    print(f"   Power law exponent: {wow_metrics['power_alpha']:.4f}")
    print(f"   S0/S1 = {wow_metrics['ratios'][0]:.4f} (phi = {PHI:.4f})")
    print(f"   S1/S2 = {wow_metrics['ratios'][1]:.4f} (pi = {PI:.4f})")
    print(f"   Combined phi+pi error: {wow_metrics['combined_phi_pi_error']*100:.2f}%")

    # Generate geometric signals
    print("\n3. Generating geometric signals...")

    geometric_signals = {
        "spherical_harmonics": generate_spherical_harmonic_signal(wow_shape),
        "hyperbolic": generate_hyperbolic_signal(wow_shape),
        "toroidal": generate_toroidal_signal(wow_shape),
        "fractal_1.5": generate_fractal_signal(wow_shape, exponent=1.5),
        "fractal_2.0": generate_fractal_signal(wow_shape, exponent=2.0),
        "phi_structured": generate_phi_structured_signal(wow_shape),
        "pi_structured": generate_pi_structured_signal(wow_shape),
    }

    # Compute metrics for each
    print("\n4. Computing metrics for geometric signals...")

    all_metrics = {"wow": wow_metrics}

    for name, signal in geometric_signals.items():
        metrics = compute_eigenvalue_metrics(signal)
        all_metrics[name] = metrics
        print(f"\n   {name}:")
        print(f"      Power law: alpha = {metrics['power_alpha']:.4f}")
        print(f"      S0/S1 = {metrics['ratios'][0]:.4f}, S1/S2 = {metrics['ratios'][1]:.4f}")
        print(f"      Phi+pi error: {metrics['combined_phi_pi_error']*100:.2f}%")

    # Generate random baselines
    print("\n5. Computing random baseline...")
    random_metrics = []
    for _ in range(100):
        rand = np.random.randn(*wow_shape)
        m = compute_eigenvalue_metrics(rand)
        random_metrics.append(m)

    random_phi_pi_errors = [m["combined_phi_pi_error"] for m in random_metrics]
    random_mean = float(np.mean(random_phi_pi_errors))
    random_std = float(np.std(random_phi_pi_errors))

    print(f"   Random phi+pi error: {random_mean*100:.2f}% +/- {random_std*100:.2f}%")

    # Load FRBs
    print("\n6. Analyzing FRBs...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))
    frb_metrics = []

    for frb_file in frb_files[:50]:  # Limit for speed
        data = load_frb(frb_file)
        if data is None:
            continue

        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue

        m = compute_eigenvalue_metrics(data_resized)
        frb_metrics.append(m)

    n_frbs = len(frb_metrics)
    print(f"   Analyzed {n_frbs} FRBs")

    if n_frbs > 0:
        frb_phi_pi_errors = [m["combined_phi_pi_error"] for m in frb_metrics]
        frb_mean = float(np.mean(frb_phi_pi_errors))
        frb_std = float(np.std(frb_phi_pi_errors))
        print(f"   FRB phi+pi error: {frb_mean*100:.2f}% +/- {frb_std*100:.2f}%")

    # Ranking
    print("\n" + "=" * 60)
    print("RANKING: Closest to Wow!'s eigenvalue structure")
    print("=" * 60)

    print(f"\nWow!'s phi+pi error: {wow_metrics['combined_phi_pi_error']*100:.2f}%")
    print("\nComparison:")
    print("-" * 60)

    # Rank all sources by how close they are to Wow!'s phi+pi pattern
    rankings = []

    for name, metrics in all_metrics.items():
        if name != "wow":
            rankings.append((name, metrics["combined_phi_pi_error"]))

    # Add random baseline
    rankings.append(("random_baseline", random_mean))

    # Add FRB
    if n_frbs > 0:
        # Best FRB
        best_frb_idx = np.argmin(frb_phi_pi_errors)
        rankings.append(("best_FRB", frb_phi_pi_errors[best_frb_idx]))
        rankings.append(("FRB_average", frb_mean))

    # Sort by error (lower = closer to phi+pi)
    rankings.sort(key=lambda x: x[1])

    for rank, (name, error) in enumerate(rankings, 1):
        match_quality = ""
        if error < 0.10:
            match_quality = " <-- VERY CLOSE to phi+pi"
        elif error < 0.20:
            match_quality = " <-- close to phi+pi"
        print(f"   {rank}. {name}: {error*100:.2f}%{match_quality}")

    # Z-score of Wow! vs random
    z_vs_random = (wow_metrics["combined_phi_pi_error"] - random_mean) / (random_std + 1e-8)

    # Z-score vs FRBs
    if n_frbs > 0:
        z_vs_frbs = (wow_metrics["combined_phi_pi_error"] - frb_mean) / (frb_std + 1e-8)
    else:
        z_vs_frbs = 0

    print(f"\nZ-scores:")
    print(f"   Wow! vs random: z = {z_vs_random:+.2f}")
    if n_frbs > 0:
        print(f"   Wow! vs FRBs:   z = {z_vs_frbs:+.2f}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find which geometry is closest
    closest_name, closest_error = rankings[0]

    print(f"\n1. CLOSEST GEOMETRY TO WOW!:")
    print(f"   {closest_name} with phi+pi error = {closest_error*100:.2f}%")

    print(f"\n2. WOW!'S EIGENVALUE STRUCTURE:")
    print(f"   S0/S1 = {wow_metrics['ratios'][0]:.4f} (phi = {PHI:.4f}, err = {wow_metrics['phi_error']*100:.1f}%)")
    print(f"   S1/S2 = {wow_metrics['ratios'][1]:.4f} (pi = {PI:.4f}, err = {wow_metrics['pi_error']*100:.1f}%)")
    print(f"   Power law exponent: alpha = {wow_metrics['power_alpha']:.4f}")

    print(f"\n3. STATISTICAL SIGNIFICANCE:")
    print(f"   vs random: z = {z_vs_random:+.2f}")
    if abs(z_vs_random) > 2:
        print(f"   --> Wow! is SIGNIFICANTLY different from random noise")
    if n_frbs > 0:
        print(f"   vs FRBs: z = {z_vs_frbs:+.2f}")
        if z_vs_frbs < -2:
            print(f"   --> Wow! is SIGNIFICANTLY closer to phi/pi than FRBs")

    print(f"\n4. GEOMETRIC INTERPRETATION:")

    # Interpret based on findings
    if closest_name == "phi_structured":
        print(f"   Wow! most resembles a signal explicitly structured by golden ratio")
    elif closest_name == "pi_structured":
        print(f"   Wow! most resembles a signal explicitly structured by pi")
    elif closest_name == "spherical_harmonics":
        print(f"   Wow! most resembles spherical harmonic decomposition")
        print(f"   This is consistent with omnidirectional transmission from a point source")
    elif closest_name == "hyperbolic":
        print(f"   Wow! most resembles hyperbolic geometry")
        print(f"   This suggests propagation through negatively curved space")
    elif closest_name == "toroidal":
        print(f"   Wow! most resembles toroidal geometry")
    elif "fractal" in closest_name:
        print(f"   Wow! most resembles fractal/self-similar structure")
        print(f"   BUT with phi+pi ratios imposed on top")

    # Count how many are closer than Wow! to phi/pi
    n_closer = sum(1 for m in frb_metrics if m["combined_phi_pi_error"] < wow_metrics["combined_phi_pi_error"])

    if n_frbs > 0:
        print(f"\n5. WOW! VS NATURAL SOURCES:")
        print(f"   {n_closer}/{n_frbs} FRBs are closer to phi/pi than Wow!")
        if n_closer == 0:
            print(f"   --> Wow! is the MOST phi/pi-structured signal in our sample")

    # Save results
    results = {
        "experiment": "exp55_geometric_decay",
        "timestamp": datetime.now().isoformat(),
        "wow_metrics": wow_metrics,
        "geometric_metrics": {name: m for name, m in all_metrics.items() if name != "wow"},
        "random_baseline": {
            "mean_phi_pi_error": random_mean,
            "std_phi_pi_error": random_std,
        },
        "frb_analysis": {
            "n_frbs": n_frbs,
            "mean_phi_pi_error": frb_mean if n_frbs > 0 else None,
            "std_phi_pi_error": frb_std if n_frbs > 0 else None,
            "n_closer_than_wow": n_closer if n_frbs > 0 else None,
        },
        "z_scores": {
            "vs_random": float(z_vs_random),
            "vs_frbs": float(z_vs_frbs) if n_frbs > 0 else None,
        },
        "rankings": [(name, float(err)) for name, err in rankings],
        "closest_geometry": closest_name,
    }

    output_path = RESULTS_DIR / "exp55_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n7. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
