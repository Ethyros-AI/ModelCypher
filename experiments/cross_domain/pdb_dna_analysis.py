#!/usr/bin/env python3
"""
Fetch and analyze real DNA structures from the Protein Data Bank.

Validates that the fundamental constants found in canonical parameters
also appear in actual crystallographic/NMR measurements.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

import numpy as np
import requests

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
    "2pi": 2 * PI,
    "pi/2": PI / 2,
    "pi/3": PI / 3,
    "21": 21.0,
    "21/2": 10.5,
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


@dataclass
class PDBStructure:
    """Parsed PDB structure with coordinates."""
    pdb_id: str
    title: str
    resolution: Optional[float]
    atoms: List[Dict]  # List of atom records


def fetch_pdb(pdb_id: str) -> str:
    """Fetch PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_pdb(pdb_text: str, pdb_id: str) -> PDBStructure:
    """Parse PDB format text into structure."""
    atoms = []
    title = ""
    resolution = None

    for line in pdb_text.split("\n"):
        if line.startswith("TITLE"):
            title += line[10:].strip() + " "
        elif line.startswith("REMARK   2 RESOLUTION"):
            try:
                res_str = line.split()[3]
                resolution = float(res_str)
            except (IndexError, ValueError):
                pass
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                atom = {
                    "serial": int(line[6:11].strip()),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resseq": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                continue

    return PDBStructure(
        pdb_id=pdb_id,
        title=title.strip(),
        resolution=resolution,
        atoms=atoms,
    )


def extract_backbone_atoms(structure: PDBStructure, atom_names: List[str] = None) -> np.ndarray:
    """Extract backbone atom coordinates.

    For DNA, we use P (phosphate) atoms to trace the backbone.
    """
    if atom_names is None:
        atom_names = ["P"]  # Phosphate backbone

    coords = []
    for atom in structure.atoms:
        if atom["name"] in atom_names:
            coords.append([atom["x"], atom["y"], atom["z"]])

    return np.array(coords) if coords else np.array([]).reshape(0, 3)


def extract_base_atoms(structure: PDBStructure) -> Dict[str, np.ndarray]:
    """Extract coordinates grouped by base type."""
    bases = {"A": [], "T": [], "G": [], "C": [], "U": []}

    # Map residue names to single letter codes
    res_map = {
        "DA": "A", "DT": "T", "DG": "G", "DC": "C",
        "A": "A", "T": "T", "G": "G", "C": "C", "U": "U",
        "ADE": "A", "THY": "T", "GUA": "G", "CYT": "C", "URA": "U",
    }

    for atom in structure.atoms:
        base = res_map.get(atom["resname"])
        if base and atom["name"] in ["C1'", "N1", "N9"]:  # Key base atoms
            bases[base].append([atom["x"], atom["y"], atom["z"]])

    return {k: np.array(v) if v else np.array([]).reshape(0, 3) for k, v in bases.items()}


def compute_helix_parameters(coords: np.ndarray) -> Dict[str, float]:
    """Compute helix parameters from backbone coordinates.

    Uses SVD to fit helix axis and compute geometric parameters.
    """
    if len(coords) < 10:
        return {}

    # Center coordinates
    center = coords.mean(axis=0)
    centered = coords - center

    # SVD to find principal axis (helix axis)
    U, S, Vt = np.linalg.svd(centered)
    axis = Vt[0]  # First principal component = helix axis

    # Project onto plane perpendicular to axis
    projections = centered - np.outer(centered @ axis, axis)
    radii = np.linalg.norm(projections, axis=1)

    # Compute rise per residue (projection onto axis)
    z_coords = centered @ axis
    rises = np.diff(z_coords)

    # Compute angles in the perpendicular plane
    angles = np.arctan2(projections[:, 1], projections[:, 0])
    angle_diffs = np.diff(angles)
    # Unwrap angles
    angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
    angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)

    # Convert to degrees
    twist_per_residue = np.abs(np.mean(angle_diffs)) * 180 / np.pi

    params = {
        "mean_radius": float(np.mean(radii)),
        "std_radius": float(np.std(radii)),
        "mean_rise": float(np.mean(np.abs(rises))),
        "std_rise": float(np.std(rises)),
        "twist_per_residue": float(twist_per_residue),
        "residues_per_turn": float(360 / twist_per_residue) if twist_per_residue > 0 else float('inf'),
        "pitch": float(np.mean(np.abs(rises)) * 360 / twist_per_residue) if twist_per_residue > 0 else 0,
        "n_residues": len(coords),
    }

    return params


def analyze_pdb_structure(pdb_id: str) -> Dict:
    """Complete analysis of a PDB structure."""
    print(f"\n{'='*60}")
    print(f"Analyzing {pdb_id}")
    print(f"{'='*60}")

    try:
        pdb_text = fetch_pdb(pdb_id)
        structure = parse_pdb(pdb_text, pdb_id)
    except Exception as e:
        print(f"Error fetching {pdb_id}: {e}")
        return {"error": str(e), "pdb_id": pdb_id}

    print(f"Title: {structure.title[:60]}...")
    print(f"Resolution: {structure.resolution} Å" if structure.resolution else "Resolution: N/A")
    print(f"Total atoms: {len(structure.atoms)}")

    results = {
        "pdb_id": pdb_id,
        "title": structure.title,
        "resolution": structure.resolution,
        "n_atoms": len(structure.atoms),
    }

    # Extract backbone coordinates
    backbone = extract_backbone_atoms(structure, ["P"])
    print(f"Backbone atoms (P): {len(backbone)}")

    if len(backbone) >= 10:
        # Compute helix parameters
        params = compute_helix_parameters(backbone)
        results["helix_parameters"] = params

        print(f"\nHelix Parameters:")
        print(f"  Radius: {params['mean_radius']:.2f} ± {params['std_radius']:.2f} Å")
        print(f"  Rise/residue: {params['mean_rise']:.2f} ± {params['std_rise']:.2f} Å")
        print(f"  Twist/residue: {params['twist_per_residue']:.2f}°")
        print(f"  Residues/turn: {params['residues_per_turn']:.2f}")
        print(f"  Pitch: {params['pitch']:.2f} Å")

        # Check ratios against constants
        print(f"\n--- Ratio Analysis ---")
        param_values = np.array([
            params['mean_radius'] * 2,  # diameter
            params['mean_rise'],
            params['twist_per_residue'],
            params['residues_per_turn'],
            params['pitch'],
        ])

        # Find matches
        matches = []
        param_names = ["diameter", "rise", "twist", "res/turn", "pitch"]
        for i, (ni, vi) in enumerate(zip(param_names, param_values)):
            for j, (nj, vj) in enumerate(zip(param_names, param_values)):
                if i != j and vj > 1e-10:
                    ratio = vi / vj
                    for const_name, const_val in CONSTANTS.items():
                        error = abs(ratio - const_val) / const_val
                        if error < MATCH_THRESHOLD:
                            matches.append({
                                "ratio": f"{ni}/{nj}",
                                "value": float(ratio),
                                "constant": const_name,
                                "error_pct": float(error * 100),
                            })
                            print(f"  {ni}/{nj} = {ratio:.4f} ≈ {const_name} ({const_val:.4f}, {error*100:.2f}% error)")

        results["ratio_matches"] = matches

        # Distance matrix SVD
        print(f"\n--- Distance Matrix SVD ---")
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(backbone, backbone)
        dist_norm = dist_matrix / np.max(dist_matrix)
        _, S, _ = np.linalg.svd(dist_norm)

        svd_matches = count_constant_matches(S, bidirectional=True)
        total = sum(svd_matches.values())
        results["distance_matrix_matches"] = {
            "total": total,
            "by_constant": svd_matches,
        }
        print(f"Total SVD matches: {total}")
        for name, count in svd_matches.items():
            if count > 0:
                print(f"  {name}: {count}")

    # Base composition analysis
    bases = extract_base_atoms(structure)
    base_counts = {k: len(v) for k, v in bases.items() if len(v) > 0}
    if base_counts:
        results["base_counts"] = base_counts
        print(f"\nBase counts: {base_counts}")

        # GC content and ratios
        total_bases = sum(base_counts.values())
        if total_bases > 0:
            gc = base_counts.get("G", 0) + base_counts.get("C", 0)
            at = base_counts.get("A", 0) + base_counts.get("T", 0)
            if at > 0:
                gc_at_ratio = gc / at
                print(f"GC/AT ratio: {gc_at_ratio:.4f}")
                for const_name, const_val in CONSTANTS.items():
                    error = abs(gc_at_ratio - const_val) / const_val
                    if error < MATCH_THRESHOLD:
                        print(f"  GC/AT ≈ {const_name} ({error*100:.2f}% error)")

    return results


def analyze_multiple_structures():
    """Analyze multiple DNA structures from PDB."""
    # Classic DNA structures
    structures = [
        # B-DNA
        "1BNA",  # Dickerson dodecamer - THE classic B-DNA structure
        "1D98",  # B-DNA decamer
        "3BSE",  # Z-DNA
        "440D",  # A-DNA
        # Additional high-resolution DNA structures
        "1ZF1",  # B-DNA (high resolution)
        "1EHV",  # B-DNA
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "structures": {},
    }

    for pdb_id in structures:
        try:
            result = analyze_pdb_structure(pdb_id)
            results["structures"][pdb_id] = result
        except Exception as e:
            print(f"Error analyzing {pdb_id}: {e}")
            results["structures"][pdb_id] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS STRUCTURES")
    print("="*60)

    all_matches = []
    for pdb_id, data in results["structures"].items():
        if "ratio_matches" in data:
            for m in data["ratio_matches"]:
                m["pdb_id"] = pdb_id
                all_matches.append(m)

    # Count matches by constant
    const_counts = {}
    for m in all_matches:
        const = m["constant"]
        const_counts[const] = const_counts.get(const, 0) + 1

    print("\nRatio matches across all structures:")
    for const, count in sorted(const_counts.items(), key=lambda x: -x[1]):
        print(f"  {const}: {count}")

    # Distance matrix totals
    total_dist = 0
    for pdb_id, data in results["structures"].items():
        if "distance_matrix_matches" in data:
            total_dist += data["distance_matrix_matches"]["total"]
    print(f"\nTotal distance matrix SVD matches: {total_dist}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"pdb_dna_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = analyze_multiple_structures()
