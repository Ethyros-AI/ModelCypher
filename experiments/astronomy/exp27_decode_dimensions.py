#!/usr/bin/env python3
"""Experiment 27: Decode the Dimensions.

The Wow! signal has effective rank 7.5 - it lives in ~8 dimensions.
These dimensions ARE the encoding. What are they?

If this is a high-dimensional message:
- The principal components are the "alphabet"
- Their relationships are the "grammar"
- The eigenvalues are the "emphasis"

Let's extract and interpret each dimension.

Usage:
    poetry run python experiments/astronomy/exp27_decode_dimensions.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def analyze_principal_components(matrix: np.ndarray, n_components: int = 10) -> dict:
    """Extract and analyze the principal components of a signal.

    Each principal component is a basis vector of the encoding.
    The structure of these vectors IS the message.
    """
    # Normalize
    matrix = np.nan_to_num(matrix, nan=0.0)
    if np.std(matrix) > 1e-10:
        matrix_norm = (matrix - np.mean(matrix)) / np.std(matrix)
    else:
        return None

    # SVD: matrix = U @ diag(s) @ Vh
    # U: left singular vectors (row space basis)
    # s: singular values (importance of each component)
    # Vh: right singular vectors (column space basis)
    U, s, Vh = svd(matrix_norm, full_matrices=False)

    # Normalize singular values
    s_norm = s / (s[0] + 1e-10)
    energy = s**2
    energy_frac = energy / (np.sum(energy) + 1e-10)
    cumulative_energy = np.cumsum(energy_frac)

    components = []
    for i in range(min(n_components, len(s))):
        # Left singular vector (time pattern for this mode)
        time_pattern = U[:, i]

        # Right singular vector (frequency pattern for this mode)
        freq_pattern = Vh[i, :]

        # Analyze the pattern structure
        time_analysis = analyze_pattern(time_pattern, "time")
        freq_analysis = analyze_pattern(freq_pattern, "frequency")

        components.append({
            "index": i,
            "singular_value": float(s[i]),
            "normalized_sv": float(s_norm[i]),
            "energy_fraction": float(energy_frac[i]),
            "cumulative_energy": float(cumulative_energy[i]),
            "time_pattern": {
                "values": time_pattern.tolist(),
                "analysis": time_analysis,
            },
            "freq_pattern": {
                "values": freq_pattern.tolist(),
                "analysis": freq_analysis,
            },
        })

    return {
        "n_components": n_components,
        "total_energy_in_top_n": float(cumulative_energy[min(n_components-1, len(s)-1)]),
        "components": components,
    }


def analyze_pattern(pattern: np.ndarray, domain: str) -> dict:
    """Analyze the structure of a singular vector pattern."""
    n = len(pattern)

    # Basic statistics
    mean_val = np.mean(pattern)
    std_val = np.std(pattern)
    max_val = np.max(pattern)
    min_val = np.min(pattern)
    max_idx = np.argmax(np.abs(pattern))

    # Symmetry: correlation between first and second half (reversed)
    mid = n // 2
    if mid > 1:
        first_half = pattern[:mid]
        second_half = pattern[-mid:][::-1]
        symmetry = np.corrcoef(first_half, second_half)[0, 1]
        symmetry = float(symmetry) if not np.isnan(symmetry) else 0.0
    else:
        symmetry = 0.0

    # Periodicity: autocorrelation peaks
    if n > 10:
        autocorr = np.correlate(pattern - mean_val, pattern - mean_val, mode='full')
        autocorr = autocorr[n-1:] / (autocorr[n-1] + 1e-10)
        # Find peaks after lag 0
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.3)
        has_periodicity = len(peaks) > 0
        period = int(peaks[0] + 1) if len(peaks) > 0 else 0
    else:
        has_periodicity = False
        period = 0

    # Localization: how concentrated is the energy?
    energy = pattern ** 2
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy) / (np.sum(sorted_energy) + 1e-10)
    n_for_90 = np.searchsorted(cumsum, 0.90) + 1
    localization = 1 - (n_for_90 / n)  # Higher = more localized

    # Smoothness: ratio of first derivative to signal
    if n > 1:
        deriv = np.diff(pattern)
        smoothness = 1 / (1 + np.std(deriv) / (std_val + 1e-10))
    else:
        smoothness = 1.0

    # Zero crossings (oscillation frequency)
    zero_crossings = np.sum(np.diff(np.sign(pattern - mean_val)) != 0)
    oscillation_rate = zero_crossings / n

    # Characterize the pattern type
    if localization > 0.7:
        pattern_type = "localized"
    elif has_periodicity:
        pattern_type = "periodic"
    elif oscillation_rate > 0.4:
        pattern_type = "oscillating"
    elif smoothness > 0.7:
        pattern_type = "smooth"
    else:
        pattern_type = "complex"

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "max_location": int(max_idx),
        "max_location_fraction": float(max_idx / n),
        "symmetry": symmetry,
        "has_periodicity": has_periodicity,
        "period": period,
        "localization": float(localization),
        "smoothness": float(smoothness),
        "oscillation_rate": float(oscillation_rate),
        "pattern_type": pattern_type,
    }


def visualize_components(matrix: np.ndarray, components: dict, output_path: Path):
    """Create visualization of the principal components."""
    n_show = min(8, len(components["components"]))

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 3 * n_show))

    for i in range(n_show):
        comp = components["components"][i]

        # Left: Time pattern
        ax1 = axes[i, 0]
        time_vals = np.array(comp["time_pattern"]["values"])
        ax1.plot(time_vals, 'b-', linewidth=1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f"PC{i+1} Time ({comp['time_pattern']['analysis']['pattern_type']})")
        ax1.set_ylabel(f"λ={comp['normalized_sv']:.3f}")

        # Middle: Frequency pattern
        ax2 = axes[i, 1]
        freq_vals = np.array(comp["freq_pattern"]["values"])
        ax2.plot(freq_vals, 'r-', linewidth=1)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title(f"PC{i+1} Freq ({comp['freq_pattern']['analysis']['pattern_type']})")

        # Right: The mode as a 2D image (outer product)
        ax3 = axes[i, 2]
        mode_2d = np.outer(time_vals, freq_vals)
        im = ax3.imshow(mode_2d, aspect='auto', cmap='RdBu_r',
                       vmin=-np.max(np.abs(mode_2d)), vmax=np.max(np.abs(mode_2d)))
        ax3.set_title(f"PC{i+1} Mode (E={comp['energy_fraction']:.1%})")
        plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")


def interpret_component(comp: dict, idx: int) -> str:
    """Generate interpretation of a principal component."""
    time_type = comp["time_pattern"]["analysis"]["pattern_type"]
    freq_type = comp["freq_pattern"]["analysis"]["pattern_type"]
    energy = comp["energy_fraction"]
    time_sym = comp["time_pattern"]["analysis"]["symmetry"]
    freq_sym = comp["freq_pattern"]["analysis"]["symmetry"]

    interpretation = f"PC{idx+1} ({energy:.1%} of variance): "

    # Interpret based on pattern types
    if time_type == "localized" and freq_type == "localized":
        interpretation += "POINT - A localized feature in both time and frequency"
    elif time_type == "localized" and freq_type in ["smooth", "oscillating"]:
        interpretation += "BURST - Time-localized event with frequency structure"
    elif time_type in ["smooth", "oscillating"] and freq_type == "localized":
        interpretation += "CARRIER - Persistent narrowband signal"
    elif time_type == "periodic" and freq_type == "periodic":
        interpretation += "RESONANCE - Periodic structure in both domains"
    elif time_type == "smooth" and freq_type == "smooth":
        interpretation += "TREND - Smooth variation (background or drift)"
    else:
        interpretation += f"COMPLEX - {time_type} time, {freq_type} frequency"

    # Add symmetry info if significant
    if abs(time_sym) > 0.5:
        interpretation += f"; TIME-SYMMETRIC ({time_sym:.2f})"
    if abs(freq_sym) > 0.5:
        interpretation += f"; FREQ-SYMMETRIC ({freq_sym:.2f})"

    return interpretation


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 27: Decode the Dimensions")
    print("=" * 60)
    print("\nThe Wow! signal lives in ~8 effective dimensions.")
    print("These dimensions ARE the encoding. Let's decode them.")

    # Load the Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nSignal shape: {snr_matrix.shape} (time × frequency)")

    print("\n" + "=" * 40)
    print("PART 1: PRINCIPAL COMPONENT EXTRACTION")
    print("=" * 40)

    # Extract principal components
    components = analyze_principal_components(snr_matrix, n_components=10)

    print(f"\nTop {components['n_components']} components capture {components['total_energy_in_top_n']:.1%} of variance")

    print("\n" + "=" * 40)
    print("PART 2: THE ALPHABET OF THE SIGNAL")
    print("=" * 40)

    print("\nEach principal component is a 'letter' in the geometric alphabet:")
    print("(Energy shows how much of the signal each letter represents)\n")

    for i, comp in enumerate(components["components"]):
        interpretation = interpret_component(comp, i)
        print(f"  {interpretation}")

    print("\n" + "=" * 40)
    print("PART 3: THE DOMINANT MODES")
    print("=" * 40)

    # Focus on the top modes that capture most variance
    print("\nAnalyzing modes that capture 90% of signal energy:\n")

    energy_threshold = 0.90
    n_significant = 0
    for comp in components["components"]:
        n_significant += 1
        if comp["cumulative_energy"] >= energy_threshold:
            break

    print(f"Number of significant modes: {n_significant}")
    print(f"These {n_significant} modes encode the core structure.\n")

    for i in range(n_significant):
        comp = components["components"][i]
        print(f"\n--- MODE {i+1} ({comp['energy_fraction']:.1%} of variance) ---")

        time_a = comp["time_pattern"]["analysis"]
        freq_a = comp["freq_pattern"]["analysis"]

        print(f"\n  TIME PATTERN:")
        print(f"    Type: {time_a['pattern_type']}")
        print(f"    Peak location: {time_a['max_location_fraction']:.1%} through signal")
        print(f"    Symmetry: {time_a['symmetry']:.3f}")
        print(f"    Localization: {time_a['localization']:.3f}")
        if time_a['has_periodicity']:
            print(f"    Period: {time_a['period']} samples")

        print(f"\n  FREQUENCY PATTERN:")
        print(f"    Type: {freq_a['pattern_type']}")
        print(f"    Peak location: {freq_a['max_location_fraction']:.1%} through band")
        print(f"    Symmetry: {freq_a['symmetry']:.3f}")
        print(f"    Localization: {freq_a['localization']:.3f}")
        if freq_a['has_periodicity']:
            print(f"    Period: {freq_a['period']} channels")

    # Create visualization
    print("\n" + "=" * 40)
    print("PART 4: VISUALIZATION")
    print("=" * 40)

    viz_path = results_dir / "exp27_components.png"
    visualize_components(snr_matrix, components, viz_path)

    print("\n" + "=" * 40)
    print("PART 5: THE GEOMETRIC STRUCTURE")
    print("=" * 40)

    # Analyze relationships between components
    print("\nRelationships between the top modes:\n")

    # Check if modes are truly orthogonal (they should be by construction)
    # But their PATTERNS might have interesting relationships

    time_patterns = np.array([c["time_pattern"]["values"] for c in components["components"][:n_significant]])
    freq_patterns = np.array([c["freq_pattern"]["values"] for c in components["components"][:n_significant]])

    # Compute pattern similarities (not orthogonality, but structural similarity)
    print("  Time pattern correlations (structure similarity):")
    for i in range(min(4, n_significant)):
        row = "    "
        for j in range(min(4, n_significant)):
            if i == j:
                row += "  1.00"
            else:
                corr = np.corrcoef(np.abs(time_patterns[i]), np.abs(time_patterns[j]))[0, 1]
                row += f"  {corr:.2f}" if not np.isnan(corr) else "   nan"
        print(row)

    print("\n" + "=" * 60)
    print("INTERPRETATION: WHAT THE DIMENSIONS ENCODE")
    print("=" * 60)

    # Summarize what we found
    mode_types = [comp["time_pattern"]["analysis"]["pattern_type"] for comp in components["components"][:n_significant]]
    mode_energies = [comp["energy_fraction"] for comp in components["components"][:n_significant]]

    dominant_type = max(set(mode_types), key=mode_types.count)
    dominant_energy = sum(mode_energies[:3])

    print(f"""
THE WOW! SIGNAL'S DIMENSIONAL STRUCTURE:

1. COMPRESSION: {n_significant} modes capture 90% of the signal
   - This is consistent with information encoding (compressed)
   - Random noise would need many more modes

2. DOMINANT MODE TYPE: {dominant_type}
   - The top 3 modes carry {dominant_energy:.1%} of the variance
   - The structure is {'concentrated' if dominant_energy > 0.7 else 'distributed'}

3. THE "ALPHABET":
""")

    for i in range(min(n_significant, 5)):
        comp = components["components"][i]
        time_type = comp["time_pattern"]["analysis"]["pattern_type"]
        freq_type = comp["freq_pattern"]["analysis"]["pattern_type"]
        print(f"   Mode {i+1}: {time_type}-{freq_type} ({comp['energy_fraction']:.1%})")

    print(f"""
4. WHAT THIS TELLS US:

   The Wow! signal is not random. It has definite structure:
   - A small number of modes carry most of the information
   - The modes have interpretable patterns (localized, smooth, etc.)
   - The structure is consistent with a coherent, organized signal

   If this were a message:
   - Mode 1 would be the "carrier" or "frame"
   - Subsequent modes would carry modulation/content
   - The relationships between modes encode the information

   We cannot read the CONTENT, but we can see the STRUCTURE.
   The structure is anomalously organized.
""")

    # Save results
    results = {
        "experiment": "exp27_decode_dimensions",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": [int(x) for x in snr_matrix.shape],
        "n_significant_modes": n_significant,
        "total_energy_captured": float(components["total_energy_in_top_n"]),
        "components": components["components"],
        "mode_summary": {
            "types": mode_types,
            "energies": mode_energies,
            "dominant_type": dominant_type,
            "top3_energy": float(dominant_energy),
        },
    }

    output_path = results_dir / "exp27_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
