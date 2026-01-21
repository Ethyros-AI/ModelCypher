#!/usr/bin/env python3
"""Experiment 35: 2D Pattern Alignment.

The 1D alignment was weak because we collapsed the 2D structure.
The Wow! signal is 82×50 (time × frequency) - a 2D image.

If information is encoded, it might be in:
1. The 2D SHAPE of the signal (Arecibo-like pictorial message)
2. The PRINCIPAL COMPONENTS (each mode is a semantic axis)
3. The MODULATION across time-frequency (FSK/PSK-like)

Let's try aligning the 2D structure to known patterns:
- Binary image patterns
- Mathematical 2D structures
- Known signal templates

Usage:
    poetry run python experiments/astronomy/exp35_2d_pattern_alignment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd
from scipy.signal import correlate2d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_2d_templates(shape: tuple) -> dict:
    """Create 2D templates for pattern matching.

    These are 2D structures that might be used for communication.
    """
    n_time, n_freq = shape
    templates = {}

    # 1. CROSSHAIR: Universal "here" marker
    crosshair = np.zeros(shape)
    crosshair[n_time//2, :] = 1.0  # Horizontal line
    crosshair[:, n_freq//2] = 1.0  # Vertical line
    templates["crosshair"] = {
        "pattern": crosshair,
        "description": "Crosshair (universal marker)",
    }

    # 2. DIAGONAL: Direction/arrow
    diagonal = np.zeros(shape)
    for i in range(min(n_time, n_freq)):
        t = int(i * n_time / min(n_time, n_freq))
        f = int(i * n_freq / min(n_time, n_freq))
        if t < n_time and f < n_freq:
            diagonal[t, f] = 1.0
    templates["diagonal"] = {
        "pattern": diagonal,
        "description": "Diagonal line (direction)",
    }

    # 3. BINARY GRID: Checkerboard (digital pattern)
    binary_grid = np.zeros(shape)
    for i in range(n_time):
        for j in range(n_freq):
            binary_grid[i, j] = (i + j) % 2
    templates["checkerboard"] = {
        "pattern": binary_grid,
        "description": "Checkerboard (binary pattern)",
    }

    # 4. FREQUENCY SWEEP: Chirp (radar/communication)
    chirp = np.zeros(shape)
    for t in range(n_time):
        freq_idx = int((t / n_time) * n_freq * 0.8)
        if freq_idx < n_freq:
            chirp[t, freq_idx] = 1.0
    templates["chirp"] = {
        "pattern": chirp,
        "description": "Frequency sweep (chirp)",
    }

    # 5. GAUSSIAN SPOT: Point source
    center_t, center_f = n_time // 2, n_freq // 2
    gaussian = np.zeros(shape)
    for t in range(n_time):
        for f in range(n_freq):
            dist2 = ((t - center_t) / (n_time/4))**2 + ((f - center_f) / (n_freq/4))**2
            gaussian[t, f] = np.exp(-dist2)
    templates["gaussian_spot"] = {
        "pattern": gaussian,
        "description": "Gaussian spot (point source)",
    }

    # 6. VERTICAL STRIPE: Narrowband (continuous)
    stripe = np.zeros(shape)
    stripe[:, n_freq//2-2:n_freq//2+3] = 1.0
    templates["vertical_stripe"] = {
        "pattern": stripe,
        "description": "Vertical stripe (narrowband carrier)",
    }

    # 7. HORIZONTAL PULSE: Burst (the Wow! pattern)
    pulse = np.zeros(shape)
    center = n_time // 2
    for t in range(n_time):
        amplitude = np.exp(-((t - center) / (n_time/10))**2)
        pulse[t, :] = amplitude
    templates["horizontal_pulse"] = {
        "pattern": pulse,
        "description": "Horizontal pulse (burst envelope)",
    }

    # 8. PRIME GRID: Dots at prime positions
    prime_grid = np.zeros(shape)
    primes_t = [p for p in range(2, n_time) if all(p % i != 0 for i in range(2, int(np.sqrt(p)) + 1))]
    primes_f = [p for p in range(2, n_freq) if all(p % i != 0 for i in range(2, int(np.sqrt(p)) + 1))]
    for pt in primes_t[:20]:
        for pf in primes_f[:10]:
            if pt < n_time and pf < n_freq:
                prime_grid[pt, pf] = 1.0
    templates["prime_grid"] = {
        "pattern": prime_grid,
        "description": "Prime number grid positions",
    }

    # 9. PI PATTERN: Encode pi digits as intensity
    pi_str = "314159265358979323846264338327950288"
    pi_pattern = np.zeros(shape)
    for i, digit in enumerate(pi_str):
        if i >= n_time:
            break
        intensity = int(digit) / 9.0
        # Spread across frequency band
        pi_pattern[i, :] = intensity * np.exp(-(np.arange(n_freq) - n_freq//2)**2 / (n_freq/4)**2)
    templates["pi_encoding"] = {
        "pattern": pi_pattern,
        "description": "Pi digits encoded as intensity",
    }

    # 10. FIBONACCI SPIRAL (simplified 2D)
    fib_pattern = np.zeros(shape)
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for i, f in enumerate(fib):
        if f < n_time and i < n_freq:
            fib_pattern[f, i * (n_freq // len(fib))] = 1.0
    templates["fibonacci"] = {
        "pattern": fib_pattern,
        "description": "Fibonacci positions",
    }

    return templates


def compute_2d_correlation(signal: np.ndarray, template: np.ndarray) -> dict:
    """Compute 2D cross-correlation between signal and template."""
    # Normalize both
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)

    # Full 2D cross-correlation
    corr = correlate2d(signal_norm, template_norm, mode='full')

    # Find peak
    peak_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    peak_val = corr[peak_idx]

    # Normalized correlation at center (direct overlap)
    center_t = corr.shape[0] // 2
    center_f = corr.shape[1] // 2
    center_corr = corr[center_t, center_f] / (signal.size)

    # CKA-like measure
    K_sig = signal_norm @ signal_norm.T
    K_tpl = template_norm @ template_norm.T

    # Safely compute traces
    try:
        hsic = np.trace(K_sig @ K_tpl)
        hsic_ss = np.trace(K_sig @ K_sig)
        hsic_tt = np.trace(K_tpl @ K_tpl)
        if hsic_ss > 0 and hsic_tt > 0:
            cka = hsic / np.sqrt(hsic_ss * hsic_tt)
        else:
            cka = 0.0
    except:
        cka = 0.0

    return {
        "peak_correlation": float(peak_val),
        "peak_location": [int(peak_idx[0]), int(peak_idx[1])],
        "center_correlation": float(center_corr),
        "cka": float(cka) if not np.isnan(cka) else 0.0,
    }


def analyze_principal_component_patterns(snr_matrix: np.ndarray) -> dict:
    """Analyze the 2D patterns formed by principal components.

    Each principal component is a rank-1 outer product: u_i @ v_i.T
    These 2D patterns ARE the signal's "vocabulary".
    """
    # Normalize
    matrix = snr_matrix.astype(np.float64)
    matrix_norm = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-10)

    # SVD
    U, s, Vh = svd(matrix_norm, full_matrices=False)

    # Extract 2D modes
    modes = []
    for i in range(min(10, len(s))):
        # The 2D pattern for this mode
        pattern_2d = s[i] * np.outer(U[:, i], Vh[i, :])

        # Analyze the pattern
        # Is it symmetric?
        if pattern_2d.shape[0] == pattern_2d.shape[1]:
            symmetry = np.corrcoef(pattern_2d.ravel(), pattern_2d.T.ravel())[0, 1]
        else:
            symmetry = 0.0

        # Is it localized?
        energy = pattern_2d ** 2
        total_energy = np.sum(energy)
        sorted_energy = np.sort(energy.ravel())[::-1]
        cumsum = np.cumsum(sorted_energy) / (total_energy + 1e-10)
        n_for_90 = np.searchsorted(cumsum, 0.90) + 1
        localization = 1 - (n_for_90 / pattern_2d.size)

        # Peak location
        peak_idx = np.unravel_index(np.argmax(np.abs(pattern_2d)), pattern_2d.shape)

        modes.append({
            "component": i + 1,
            "energy_fraction": float(s[i]**2 / np.sum(s**2)),
            "pattern_2d": pattern_2d,
            "symmetry": float(symmetry) if not np.isnan(symmetry) else 0.0,
            "localization": float(localization),
            "peak_location": [int(peak_idx[0]), int(peak_idx[1])],
        })

    return {"modes": modes}


def match_modes_to_templates(modes: list, templates: dict) -> dict:
    """Match each principal component mode to the best template."""
    matches = []

    for mode in modes:
        pattern = mode["pattern_2d"]
        best_match = None
        best_score = -1

        for name, template_data in templates.items():
            template = template_data["pattern"]

            # Resize template if needed
            if template.shape != pattern.shape:
                from scipy.ndimage import zoom
                zoom_factors = (pattern.shape[0] / template.shape[0],
                              pattern.shape[1] / template.shape[1])
                template_resized = zoom(template, zoom_factors, order=1)
            else:
                template_resized = template

            # Compute correlation
            corr_result = compute_2d_correlation(pattern, template_resized)

            if abs(corr_result["cka"]) > best_score:
                best_score = abs(corr_result["cka"])
                best_match = {
                    "template_name": name,
                    "template_description": template_data["description"],
                    "cka": corr_result["cka"],
                    "correlation": corr_result["peak_correlation"],
                }

        matches.append({
            "mode": mode["component"],
            "energy": mode["energy_fraction"],
            "best_match": best_match,
        })

    return {"matches": matches}


def visualize_mode_patterns(modes: list, output_path: Path):
    """Visualize the 2D patterns of each principal component."""
    n_modes = min(6, len(modes))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(n_modes):
        ax = axes[i // 3, i % 3]
        mode = modes[i]
        pattern = mode["pattern_2d"]

        im = ax.imshow(pattern.T, aspect='auto', cmap='RdBu_r', origin='lower',
                      vmin=-np.max(np.abs(pattern)), vmax=np.max(np.abs(pattern)))
        ax.set_title(f"PC{mode['component']} ({mode['energy_fraction']:.1%})\n"
                    f"loc={mode['localization']:.2f}, sym={mode['symmetry']:.2f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")


def run_experiment():
    """Run the 2D pattern alignment experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 35: 2D Pattern Alignment")
    print("=" * 60)
    print("\nAnalyzing the 2D structure of the Wow! signal.")

    # Load Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    # Create 2D templates
    print("\n" + "=" * 40)
    print("PART 1: CREATE 2D TEMPLATES")
    print("=" * 40)

    templates = create_2d_templates(snr_matrix.shape)
    print(f"\nCreated {len(templates)} 2D templates:")
    for name, data in templates.items():
        print(f"  {name}: {data['description']}")

    # Direct template matching on full signal
    print("\n" + "=" * 40)
    print("PART 2: DIRECT TEMPLATE MATCHING")
    print("=" * 40)

    direct_matches = {}
    for name, template_data in templates.items():
        corr = compute_2d_correlation(snr_matrix, template_data["pattern"])
        direct_matches[name] = {
            "description": template_data["description"],
            **corr,
        }

    # Rank by CKA
    ranked_direct = sorted(direct_matches.items(), key=lambda x: abs(x[1]["cka"]), reverse=True)

    print("\nDirect template matching (by CKA):")
    for i, (name, data) in enumerate(ranked_direct[:5]):
        print(f"  {i+1}. {name}: CKA = {data['cka']:.4f}")

    # Analyze principal component patterns
    print("\n" + "=" * 40)
    print("PART 3: PRINCIPAL COMPONENT 2D PATTERNS")
    print("=" * 40)

    pc_analysis = analyze_principal_component_patterns(snr_matrix)

    print("\nPrincipal component 2D patterns:")
    for mode in pc_analysis["modes"][:5]:
        print(f"\n  PC{mode['component']} ({mode['energy_fraction']:.1%}):")
        print(f"    Localization: {mode['localization']:.3f}")
        print(f"    Peak at: time={mode['peak_location'][0]}, freq={mode['peak_location'][1]}")

    # Match modes to templates
    print("\n" + "=" * 40)
    print("PART 4: MODE-TO-TEMPLATE MATCHING")
    print("=" * 40)

    mode_matches = match_modes_to_templates(pc_analysis["modes"], templates)

    print("\nEach mode's best template match:")
    for match in mode_matches["matches"][:5]:
        best = match["best_match"]
        print(f"\n  PC{match['mode']} ({match['energy']:.1%}):")
        print(f"    Best match: {best['template_name']}")
        print(f"    ({best['template_description']})")
        print(f"    CKA: {best['cka']:.4f}")

    # Visualization
    print("\n" + "=" * 40)
    print("PART 5: VISUALIZATION")
    print("=" * 40)

    viz_path = results_dir / "exp35_2d_patterns.png"
    visualize_mode_patterns(pc_analysis["modes"], viz_path)

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    best_direct = ranked_direct[0]
    best_mode_match = mode_matches["matches"][0]["best_match"]

    print(f"""
2D PATTERN ANALYSIS:

DIRECT TEMPLATE MATCHING:
  Best match: {best_direct[0]} (CKA = {best_direct[1]['cka']:.4f})
  Description: {best_direct[1]['description']}

PRIMARY MODE (PC1) MATCHES:
  Best template: {best_mode_match['template_name']}
  Description: {best_mode_match['template_description']}
  CKA: {best_mode_match['cka']:.4f}

WHAT THE 2D STRUCTURE TELLS US:
""")

    # Check what patterns dominate
    pulse_cka = direct_matches.get("horizontal_pulse", {}).get("cka", 0)
    stripe_cka = direct_matches.get("vertical_stripe", {}).get("cka", 0)
    chirp_cka = direct_matches.get("chirp", {}).get("cka", 0)
    prime_cka = direct_matches.get("prime_grid", {}).get("cka", 0)
    pi_cka = direct_matches.get("pi_encoding", {}).get("cka", 0)

    print(f"""
  Pattern strengths:
  - Horizontal pulse (burst): CKA = {pulse_cka:.4f}
  - Vertical stripe (carrier): CKA = {stripe_cka:.4f}
  - Chirp (sweep): CKA = {chirp_cka:.4f}
  - Prime grid: CKA = {prime_cka:.4f}
  - Pi encoding: CKA = {pi_cka:.4f}
""")

    if pulse_cka > 0.5 and stripe_cka > 0.3:
        print("""
  INTERPRETATION: The signal shows BOTH:
  1. Burst envelope (localized in time)
  2. Narrowband carrier (localized in frequency)

  This is characteristic of a PULSED NARROWBAND TRANSMISSION.
  Natural or artificial? The combination suggests deliberate design.
""")
    elif pulse_cka > 0.5:
        print("""
  INTERPRETATION: Dominated by BURST pattern.
  The signal is a time-localized event.
  Consistent with natural transient (FRB, maser, etc.)
""")
    elif stripe_cka > 0.5:
        print("""
  INTERPRETATION: Dominated by CARRIER pattern.
  The signal is narrowband and continuous.
  Consistent with artificial transmission.
""")
    else:
        print("""
  INTERPRETATION: Complex 2D structure.
  Not dominated by simple patterns.
  May contain modulated information.
""")

    # Save results
    results = {
        "experiment": "exp35_2d_pattern_alignment",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(snr_matrix.shape),
        "direct_template_matching": {
            name: {k: v for k, v in data.items() if k != "description"}
            for name, data in direct_matches.items()
        },
        "direct_ranking": [(name, data["cka"]) for name, data in ranked_direct],
        "principal_component_analysis": {
            "n_modes": len(pc_analysis["modes"]),
            "modes": [{
                "component": m["component"],
                "energy_fraction": m["energy_fraction"],
                "localization": m["localization"],
                "peak_location": m["peak_location"],
            } for m in pc_analysis["modes"]],
        },
        "mode_template_matches": mode_matches["matches"],
    }

    output_path = results_dir / "exp35_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
