#!/usr/bin/env python3
"""
Information Processing vs Pure Geometry

HYPOTHESIS:
- π/e is the signature of information processing
- φ/√3 is the signature of physical geometry

TEST:
- Protein structures (fold to perform functions) → should show π/e
- Crystal lattices (atoms pack by physics) → should show φ/√3

METHODOLOGY:
- Same constants, same threshold (5%), same null hypothesis framework
- Fetch real structures from PDB (proteins) and crystallographic databases
- Report ALL results including null findings
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from scipy import stats
from scipy.spatial.distance import cdist

# Constants
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
}

MATCH_THRESHOLD = 0.05


def count_constant_matches(S: np.ndarray, bidirectional: bool = True) -> Dict[str, int]:
    """Count matches for each constant in singular value ratios."""
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


# ============================================================================
# PROTEIN STRUCTURES - Information Processing
# ============================================================================

def fetch_pdb_coordinates(pdb_id: str, atom_filter: str = "CA") -> np.ndarray:
    """Fetch alpha-carbon coordinates from PDB.

    CA (alpha carbon) traces the protein backbone -
    the structural scaffold for information processing.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {pdb_id}: {e}")
        return np.array([]).reshape(0, 3)

    coords = []
    for line in response.text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == atom_filter:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue

    return np.array(coords) if coords else np.array([]).reshape(0, 3)


# Proteins that PERFORM functions (information processing)
FUNCTIONAL_PROTEINS = {
    # Enzymes - catalyze reactions (active information processing)
    "1HHP": "HIV-1 Protease (enzyme)",
    "1LYZ": "Lysozyme (enzyme)",
    "4HHB": "Hemoglobin (oxygen transport)",
    "1TIM": "Triose Phosphate Isomerase (enzyme)",
    "1UBQ": "Ubiquitin (signaling)",
    # Transcription factors - read DNA (information processing)
    "1YSA": "Arc Repressor (DNA binding)",
}


# ============================================================================
# CRYSTAL LATTICES - Pure Geometry
# ============================================================================

def generate_crystal_lattice(
    lattice_type: str,
    n_cells: int = 5,
) -> np.ndarray:
    """Generate atomic coordinates for common crystal structures.

    These are pure geometric packings - no information processing,
    just energy minimization through spatial arrangement.
    """
    coords = []

    if lattice_type == "fcc":
        # Face-centered cubic (gold, aluminum, copper)
        # Close-packed structure - maximum geometric efficiency
        a = 1.0  # Lattice constant
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ]) * a

        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    for b in basis:
                        coords.append(b + np.array([i, j, k]) * a)

    elif lattice_type == "bcc":
        # Body-centered cubic (iron, chromium)
        a = 1.0
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ]) * a

        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    for b in basis:
                        coords.append(b + np.array([i, j, k]) * a)

    elif lattice_type == "hcp":
        # Hexagonal close-packed (zinc, titanium, magnesium)
        # Maximum packing efficiency
        a = 1.0
        c = a * math.sqrt(8/3)  # Ideal c/a ratio
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5/math.sqrt(3), c/2],
        ])

        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    offset = np.array([i*a + (j%2)*a/2, j*a*math.sqrt(3)/2, k*c])
                    for b in basis:
                        coords.append(b + offset)

    elif lattice_type == "diamond":
        # Diamond cubic (carbon, silicon)
        # Tetrahedral bonding geometry
        a = 1.0
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]) * a

        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    for b in basis:
                        coords.append(b + np.array([i, j, k]) * a)

    elif lattice_type == "simple_cubic":
        # Simple cubic (rare - polonium)
        a = 1.0
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    coords.append([i*a, j*a, k*a])

    return np.array(coords)


CRYSTAL_LATTICES = {
    "fcc": "Face-Centered Cubic (gold, copper)",
    "bcc": "Body-Centered Cubic (iron)",
    "hcp": "Hexagonal Close-Packed (zinc)",
    "diamond": "Diamond Cubic (carbon, silicon)",
    "simple_cubic": "Simple Cubic (polonium)",
}


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_structure(coords: np.ndarray, name: str) -> Dict:
    """Analyze a structure's distance matrix for constant matches."""

    if len(coords) < 10:
        return {"name": name, "error": "Too few atoms", "n_atoms": len(coords)}

    # Compute distance matrix
    dist_matrix = cdist(coords, coords)

    # Normalize
    dist_norm = dist_matrix / np.max(dist_matrix)

    # SVD
    U, S, Vt = np.linalg.svd(dist_norm, full_matrices=False)

    # Count matches
    matches = count_constant_matches(S, bidirectional=True)
    total = sum(matches.values())

    # Compute profile
    pi_e_total = matches["pi/e"] + matches["e/pi"]
    phi_sqrt3_total = matches["phi"] + matches["1/phi"] + matches["sqrt3"]

    return {
        "name": name,
        "n_atoms": len(coords),
        "matches": matches,
        "total_matches": total,
        "pi_e_matches": pi_e_total,
        "phi_sqrt3_matches": phi_sqrt3_total,
        "pi_e_fraction": pi_e_total / total if total > 0 else 0,
        "phi_sqrt3_fraction": phi_sqrt3_total / total if total > 0 else 0,
        "top_singular_values": list(S[:10]),
    }


def main():
    """Run the information vs geometry comparison."""

    print("=" * 70)
    print("INFORMATION PROCESSING vs PURE GEOMETRY")
    print("Testing: π/e → information, φ/√3 → geometry")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "π/e signatures information processing, φ/√3 signatures pure geometry",
        "proteins": {},
        "crystals": {},
    }

    # ========================================================================
    # PROTEINS (Information Processing)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PROTEINS - Information Processing Systems")
    print("=" * 70)

    protein_pi_e = []
    protein_phi_sqrt3 = []
    protein_totals = []

    for pdb_id, description in FUNCTIONAL_PROTEINS.items():
        print(f"\nAnalyzing {pdb_id}: {description}")
        coords = fetch_pdb_coordinates(pdb_id)

        if len(coords) < 10:
            print(f"  Skipped: only {len(coords)} atoms")
            continue

        result = analyze_structure(coords, f"{pdb_id} - {description}")
        results["proteins"][pdb_id] = result

        protein_pi_e.append(result["pi_e_fraction"])
        protein_phi_sqrt3.append(result["phi_sqrt3_fraction"])
        protein_totals.append(result["total_matches"])

        print(f"  Atoms: {result['n_atoms']}")
        print(f"  Total matches: {result['total_matches']}")
        print(f"  π/e fraction: {result['pi_e_fraction']*100:.1f}%")
        print(f"  φ/√3 fraction: {result['phi_sqrt3_fraction']*100:.1f}%")

    # ========================================================================
    # CRYSTALS (Pure Geometry)
    # ========================================================================
    print("\n" + "=" * 70)
    print("CRYSTALS - Pure Geometric Packing")
    print("=" * 70)

    crystal_pi_e = []
    crystal_phi_sqrt3 = []
    crystal_totals = []

    for lattice_type, description in CRYSTAL_LATTICES.items():
        print(f"\nAnalyzing {lattice_type}: {description}")
        coords = generate_crystal_lattice(lattice_type, n_cells=5)

        result = analyze_structure(coords, f"{lattice_type} - {description}")
        results["crystals"][lattice_type] = result

        crystal_pi_e.append(result["pi_e_fraction"])
        crystal_phi_sqrt3.append(result["phi_sqrt3_fraction"])
        crystal_totals.append(result["total_matches"])

        print(f"  Atoms: {result['n_atoms']}")
        print(f"  Total matches: {result['total_matches']}")
        print(f"  π/e fraction: {result['pi_e_fraction']*100:.1f}%")
        print(f"  φ/√3 fraction: {result['phi_sqrt3_fraction']*100:.1f}%")

    # ========================================================================
    # STATISTICAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    # π/e comparison
    protein_pi_e_mean = np.mean(protein_pi_e)
    crystal_pi_e_mean = np.mean(crystal_pi_e)

    # φ/√3 comparison
    protein_phi_mean = np.mean(protein_phi_sqrt3)
    crystal_phi_mean = np.mean(crystal_phi_sqrt3)

    print(f"\nπ/e Fraction:")
    print(f"  Proteins (info processing): {protein_pi_e_mean*100:.1f}%")
    print(f"  Crystals (pure geometry):   {crystal_pi_e_mean*100:.1f}%")

    if len(protein_pi_e) >= 2 and len(crystal_pi_e) >= 2:
        t_stat_pi_e, p_value_pi_e = stats.ttest_ind(protein_pi_e, crystal_pi_e)
        print(f"  T-test: t={t_stat_pi_e:.2f}, p={p_value_pi_e:.4f}")
        results["pi_e_comparison"] = {
            "protein_mean": float(protein_pi_e_mean),
            "crystal_mean": float(crystal_pi_e_mean),
            "t_statistic": float(t_stat_pi_e),
            "p_value": float(p_value_pi_e),
            "significant": bool(p_value_pi_e < 0.05),
        }

    print(f"\nφ/√3 Fraction:")
    print(f"  Proteins (info processing): {protein_phi_mean*100:.1f}%")
    print(f"  Crystals (pure geometry):   {crystal_phi_mean*100:.1f}%")

    if len(protein_phi_sqrt3) >= 2 and len(crystal_phi_sqrt3) >= 2:
        t_stat_phi, p_value_phi = stats.ttest_ind(protein_phi_sqrt3, crystal_phi_sqrt3)
        print(f"  T-test: t={t_stat_phi:.2f}, p={p_value_phi:.4f}")
        results["phi_sqrt3_comparison"] = {
            "protein_mean": float(protein_phi_mean),
            "crystal_mean": float(crystal_phi_mean),
            "t_statistic": float(t_stat_phi),
            "p_value": float(p_value_phi),
            "significant": bool(p_value_phi < 0.05),
        }

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    hypothesis_supported = (
        protein_pi_e_mean > crystal_pi_e_mean and
        crystal_phi_mean > protein_phi_mean
    )

    if hypothesis_supported:
        print("\n✓ HYPOTHESIS SUPPORTED:")
        print(f"  - Proteins show MORE π/e ({protein_pi_e_mean*100:.1f}% vs {crystal_pi_e_mean*100:.1f}%)")
        print(f"  - Crystals show MORE φ/√3 ({crystal_phi_mean*100:.1f}% vs {protein_phi_mean*100:.1f}%)")
        print("\n  π/e appears to signature INFORMATION PROCESSING")
        print("  φ/√3 appears to signature PURE GEOMETRY")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED BY THIS DATA:")
        print(f"  - Proteins π/e: {protein_pi_e_mean*100:.1f}%")
        print(f"  - Crystals π/e: {crystal_pi_e_mean*100:.1f}%")
        print(f"  - Proteins φ/√3: {protein_phi_mean*100:.1f}%")
        print(f"  - Crystals φ/√3: {crystal_phi_mean*100:.1f}%")

    results["verdict"] = {
        "hypothesis_supported": hypothesis_supported,
        "protein_pi_e_mean": float(protein_pi_e_mean),
        "crystal_pi_e_mean": float(crystal_pi_e_mean),
        "protein_phi_mean": float(protein_phi_mean),
        "crystal_phi_mean": float(crystal_phi_mean),
    }

    # ========================================================================
    # RAW DATA TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("RAW DATA")
    print("=" * 70)

    print("\n{:<25} {:>10} {:>10} {:>10}".format(
        "Structure", "Total", "π/e %", "φ/√3 %"
    ))
    print("-" * 60)

    print("\nPROTEINS:")
    for pdb_id, r in results["proteins"].items():
        if "error" not in r:
            print("{:<25} {:>10} {:>10.1f} {:>10.1f}".format(
                pdb_id, r["total_matches"], r["pi_e_fraction"]*100, r["phi_sqrt3_fraction"]*100
            ))

    print("\nCRYSTALS:")
    for lattice, r in results["crystals"].items():
        if "error" not in r:
            print("{:<25} {:>10} {:>10.1f} {:>10.1f}".format(
                lattice, r["total_matches"], r["pi_e_fraction"]*100, r["phi_sqrt3_fraction"]*100
            ))

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"info_vs_geometry_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
