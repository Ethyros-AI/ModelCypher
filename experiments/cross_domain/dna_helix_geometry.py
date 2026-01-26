#!/usr/bin/env python3
"""
DNA Helix Geometry Analysis

Tests the hypothesis that fundamental constants (π/e, φ, √2) appear in DNA
helix parameters, similar to their appearance in neural network SVD ratios
and the Wow! signal.

DNA parameters that are famously non-integer:
- B-DNA: 10.5 base pairs per turn (= 21/2, where 21 = hydrogen wavelength in cm)
- 34.3° twist per base pair
- 3.4 Å rise per base pair

If dimension = π (not exactly 3), these should encode π-related ratios.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Constants (same as geometric_experiments.py)
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

CONSTANTS = {
    "pi/e": PI / E,
    "e/pi": E / PI,
    "phi": PHI,
    "1/phi": 1 / PHI,
    "sqrt2": SQRT2,
    "1/sqrt2": 1 / SQRT2,
    "sqrt3": SQRT3,
    "e": E,
    "pi": PI,
    # Additional constants relevant to helix geometry
    "2pi": 2 * PI,
    "pi/2": PI / 2,
    "pi/3": PI / 3,  # 60 degrees - hexagonal
    "21": 21.0,      # Hydrogen wavelength in cm
    "21/2": 10.5,    # B-DNA bp/turn
}

MATCH_THRESHOLD = 0.05  # 5% relative error


@dataclass
class DNAForm:
    """Parameters for a DNA structural form."""
    name: str
    bp_per_turn: float      # Base pairs per helical turn
    twist_per_bp: float     # Degrees rotation per base pair
    rise_per_bp: float      # Angstroms vertical rise per base pair
    diameter: float         # Angstroms
    major_groove: float     # Angstroms
    minor_groove: float     # Angstroms
    pitch: float            # Angstroms (= rise × bp_per_turn)
    inclination: float      # Degrees - base pair tilt


# Canonical DNA forms (from crystallography/NMR)
# Sources: Saenger (1984), Dickerson et al. (various)
B_DNA = DNAForm(
    name="B-DNA",
    bp_per_turn=10.5,
    twist_per_bp=34.3,      # 360/10.5 = 34.29
    rise_per_bp=3.4,
    diameter=20.0,
    major_groove=22.0,
    minor_groove=12.0,
    pitch=35.7,             # 3.4 × 10.5
    inclination=-6.0,
)

A_DNA = DNAForm(
    name="A-DNA",
    bp_per_turn=11.0,
    twist_per_bp=32.7,      # 360/11 = 32.73
    rise_per_bp=2.6,
    diameter=23.0,
    major_groove=27.0,      # Shallow and wide
    minor_groove=11.0,
    pitch=28.6,             # 2.6 × 11
    inclination=20.0,
)

Z_DNA = DNAForm(
    name="Z-DNA",
    bp_per_turn=12.0,
    twist_per_bp=-30.0,     # Left-handed! 360/12 = 30
    rise_per_bp=3.7,
    diameter=18.0,
    major_groove=8.5,       # Very narrow
    minor_groove=2.0,       # Almost none
    pitch=44.4,             # 3.7 × 12
    inclination=-7.0,
)


def count_constant_matches(S: np.ndarray, bidirectional: bool = True) -> Dict[str, int]:
    """Count matches for each constant in singular value ratios.

    Args:
        S: Singular values (sorted descending)
        bidirectional: If True, check both σᵢ/σⱼ and σⱼ/σᵢ
    """
    matches = {name: 0 for name in CONSTANTS}

    for i in range(min(len(S) - 1, 20)):
        for j in range(i + 1, min(len(S), i + 6)):
            if S[j] > 1e-10:
                ratio1 = S[i] / S[j]
                ratio2 = S[j] / S[i] if bidirectional else None

                for const_name, const_val in CONSTANTS.items():
                    error1 = abs(ratio1 - const_val) / const_val
                    if error1 < MATCH_THRESHOLD:
                        matches[const_name] += 1

                    if bidirectional and ratio2 is not None:
                        error2 = abs(ratio2 - const_val) / const_val
                        if error2 < MATCH_THRESHOLD:
                            matches[const_name] += 1

    return matches


def find_all_ratio_matches(values: np.ndarray) -> List[Dict]:
    """Find all pairwise ratios that match constants."""
    matches = []
    n = len(values)

    for i in range(n):
        for j in range(n):
            if i != j and values[j] > 1e-10:
                ratio = values[i] / values[j]
                for const_name, const_val in CONSTANTS.items():
                    error = abs(ratio - const_val) / const_val
                    if error < MATCH_THRESHOLD:
                        matches.append({
                            "i": i,
                            "j": j,
                            "ratio": float(ratio),
                            "constant": const_name,
                            "target": float(const_val),
                            "error_pct": float(error * 100),
                        })

    return matches


def dna_form_to_vector(form: DNAForm) -> np.ndarray:
    """Convert DNA form to parameter vector."""
    return np.array([
        form.bp_per_turn,
        abs(form.twist_per_bp),  # Use absolute value for Z-DNA
        form.rise_per_bp,
        form.diameter,
        form.major_groove,
        form.minor_groove,
        form.pitch,
        abs(form.inclination),
    ])


def build_ratio_matrix(params: np.ndarray) -> np.ndarray:
    """Build pairwise ratio matrix from parameters."""
    n = len(params)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if params[j] > 1e-10:
                matrix[i, j] = params[i] / params[j]
    return matrix


def generate_helix_coordinates(form: DNAForm, n_bp: int = 100) -> np.ndarray:
    """Generate 3D coordinates for a DNA helix.

    Returns array of shape [n_bp, 3] with (x, y, z) coordinates.
    """
    coords = []
    radius = form.diameter / 2
    twist_rad = math.radians(form.twist_per_bp)

    for i in range(n_bp):
        theta = i * twist_rad
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = i * form.rise_per_bp
        coords.append([x, y, z])

    return np.array(coords)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix from coordinates."""
    from scipy.spatial.distance import cdist
    return cdist(coords, coords)


def analyze_dna_form(form: DNAForm) -> Dict:
    """Complete geometric analysis of a DNA form."""
    print(f"\n{'='*60}")
    print(f"Analyzing {form.name}")
    print(f"{'='*60}")

    results = {
        "name": form.name,
        "parameters": {
            "bp_per_turn": form.bp_per_turn,
            "twist_per_bp": form.twist_per_bp,
            "rise_per_bp": form.rise_per_bp,
            "diameter": form.diameter,
            "major_groove": form.major_groove,
            "minor_groove": form.minor_groove,
            "pitch": form.pitch,
            "inclination": form.inclination,
        },
    }

    # Analysis 1: Direct parameter ratios
    print("\n--- Direct Parameter Ratios ---")
    params = dna_form_to_vector(form)
    param_names = ["bp_per_turn", "twist", "rise", "diameter", "major", "minor", "pitch", "incl"]

    direct_matches = find_all_ratio_matches(params)
    results["direct_ratio_matches"] = direct_matches

    print(f"Found {len(direct_matches)} ratio matches:")
    for m in direct_matches:
        print(f"  {param_names[m['i']]}/{param_names[m['j']]} = {m['ratio']:.4f} "
              f"≈ {m['constant']} ({m['target']:.4f}, {m['error_pct']:.2f}% error)")

    # Analysis 2: SVD of ratio matrix
    print("\n--- SVD of Ratio Matrix ---")
    ratio_matrix = build_ratio_matrix(params)
    U, S, Vt = np.linalg.svd(ratio_matrix)

    results["ratio_matrix_svd"] = {
        "singular_values": list(S),
        "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
    }

    print(f"Singular values: {S[:5]}")
    print(f"Condition number: {results['ratio_matrix_svd']['condition_number']:.2f}")

    svd_matches = count_constant_matches(S, bidirectional=True)
    results["ratio_matrix_svd"]["constant_matches"] = svd_matches
    total_svd = sum(svd_matches.values())
    print(f"SVD constant matches: {total_svd}")
    for name, count in svd_matches.items():
        if count > 0:
            print(f"  {name}: {count}")

    # Analysis 3: Participation ratio
    S_sq = S ** 2
    pr = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)
    results["participation_ratio"] = float(pr)

    # Check if PR matches any constant
    for const_name, const_val in CONSTANTS.items():
        error = abs(pr - const_val) / const_val
        if error < MATCH_THRESHOLD:
            print(f"Participation ratio {pr:.4f} ≈ {const_name} ({const_val:.4f}, {error*100:.2f}% error)")
            results["pr_matches_constant"] = const_name
            break

    # Analysis 4: 3D helix distance matrix
    print("\n--- 3D Helix Distance Matrix SVD ---")
    coords = generate_helix_coordinates(form, n_bp=50)
    dist_matrix = compute_distance_matrix(coords)

    # Normalize distances
    dist_matrix_norm = dist_matrix / np.max(dist_matrix)

    _, S_dist, _ = np.linalg.svd(dist_matrix_norm)
    results["distance_matrix_svd"] = {
        "singular_values": list(S_dist[:20]),
    }

    dist_matches = count_constant_matches(S_dist, bidirectional=True)
    results["distance_matrix_svd"]["constant_matches"] = dist_matches
    total_dist = sum(dist_matches.values())
    print(f"Distance matrix SVD constant matches: {total_dist}")
    for name, count in dist_matches.items():
        if count > 0:
            print(f"  {name}: {count}")

    # Analysis 5: Key derived ratios
    print("\n--- Derived Ratios of Interest ---")
    derived = {
        "major/minor": form.major_groove / form.minor_groove if form.minor_groove > 0 else float('inf'),
        "diameter/rise": form.diameter / form.rise_per_bp,
        "pitch/diameter": form.pitch / form.diameter,
        "360/twist": 360 / abs(form.twist_per_bp),  # Should equal bp_per_turn
        "bp_per_turn/pi": form.bp_per_turn / PI,
        "twist/30": abs(form.twist_per_bp) / 30,  # Z-DNA has 30 exactly
        "rise*bp/pi": (form.rise_per_bp * form.bp_per_turn) / PI,
    }
    results["derived_ratios"] = derived

    for name, value in derived.items():
        # Check against constants
        for const_name, const_val in CONSTANTS.items():
            error = abs(value - const_val) / const_val
            if error < MATCH_THRESHOLD:
                print(f"  {name} = {value:.4f} ≈ {const_name} ({const_val:.4f}, {error*100:.2f}% error)")
                break
        else:
            print(f"  {name} = {value:.4f}")

    return results


def null_hypothesis_test(n_samples: int = 1000) -> Dict:
    """Test whether DNA parameters are statistically different from random.

    Generates random "DNA-like" parameter sets and compares constant matches.
    """
    print("\n" + "="*60)
    print("NULL HYPOTHESIS TEST")
    print("="*60)

    # Get baseline from B-DNA
    b_params = dna_form_to_vector(B_DNA)
    b_ratio_matrix = build_ratio_matrix(b_params)
    _, b_S, _ = np.linalg.svd(b_ratio_matrix)
    b_matches = count_constant_matches(b_S, bidirectional=True)
    b_total = sum(b_matches.values())

    print(f"B-DNA SVD matches: {b_total}")

    # Generate random parameter sets
    # Use same magnitude distribution but random values
    magnitudes = np.log10(b_params)
    random_totals = []

    for _ in range(n_samples):
        # Random parameters with similar magnitude distribution
        random_params = 10 ** (np.random.normal(magnitudes, 0.3))
        random_matrix = build_ratio_matrix(random_params)
        _, random_S, _ = np.linalg.svd(random_matrix)
        random_matches = count_constant_matches(random_S, bidirectional=True)
        random_totals.append(sum(random_matches.values()))

    random_mean = np.mean(random_totals)
    random_std = np.std(random_totals)

    if random_std > 0:
        z_score = (b_total - random_mean) / random_std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = float('inf') if b_total > random_mean else 0
        p_value = 0.0

    print(f"Random samples: mean={random_mean:.2f}, std={random_std:.2f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant (p < 0.01): {p_value < 0.01}")

    return {
        "b_dna_matches": b_total,
        "random_mean": float(random_mean),
        "random_std": float(random_std),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "significant": p_value < 0.01,
        "n_samples": n_samples,
    }


def cross_form_comparison() -> Dict:
    """Compare constant matches across DNA forms."""
    print("\n" + "="*60)
    print("CROSS-FORM COMPARISON")
    print("="*60)

    forms = [B_DNA, A_DNA, Z_DNA]
    comparison = {}

    for form in forms:
        params = dna_form_to_vector(form)
        ratio_matrix = build_ratio_matrix(params)
        _, S, _ = np.linalg.svd(ratio_matrix)
        matches = count_constant_matches(S, bidirectional=True)
        total = sum(matches.values())

        comparison[form.name] = {
            "total_matches": total,
            "matches_by_constant": matches,
        }

        print(f"{form.name}: {total} total matches")

    # Which form has most matches?
    best_form = max(comparison.keys(), key=lambda k: comparison[k]["total_matches"])
    print(f"\nForm with most matches: {best_form}")

    return comparison


def main():
    """Run complete DNA helix geometry analysis."""
    print("DNA HELIX GEOMETRY ANALYSIS")
    print("Testing π-dimensional hypothesis")
    print("="*60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "DNA helix parameters encode fundamental constants (π/e, φ, √2)",
        "analysis": {},
    }

    # Analyze each form
    for form in [B_DNA, A_DNA, Z_DNA]:
        results["analysis"][form.name] = analyze_dna_form(form)

    # Null hypothesis test
    results["null_hypothesis"] = null_hypothesis_test(n_samples=1000)

    # Cross-form comparison
    results["cross_form"] = cross_form_comparison()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    b_analysis = results["analysis"]["B-DNA"]
    print(f"\nB-DNA (canonical):")
    print(f"  Direct ratio matches: {len(b_analysis['direct_ratio_matches'])}")
    print(f"  SVD constant matches: {sum(b_analysis['ratio_matrix_svd']['constant_matches'].values())}")
    print(f"  Distance matrix matches: {sum(b_analysis['distance_matrix_svd']['constant_matches'].values())}")
    print(f"  Participation ratio: {b_analysis['participation_ratio']:.4f}")

    null = results["null_hypothesis"]
    print(f"\nNull hypothesis test:")
    print(f"  B-DNA matches vs random: {null['b_dna_matches']} vs {null['random_mean']:.1f}±{null['random_std']:.1f}")
    print(f"  Z-score: {null['z_score']:.2f}, p-value: {null['p_value']:.6f}")
    print(f"  Statistically significant: {null['significant']}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"dna_helix_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
