#!/usr/bin/env python3
"""
Codon Usage Analysis

Analyze the structure of the universal genetic code for fundamental constants.

The genetic code maps 64 codons to 21 outputs (20 amino acids + stop).
This is a universal encoding system that evolved once and is shared by
virtually all life on Earth.

Key questions:
- Does the codon usage table encode π/e, φ, √2?
- Is there geometric structure in the amino acid assignments?
- Why 64 → 21? (64 = 4³, but why 21?)

Note: 21 again appears - same as hydrogen wavelength and DNA bp/turn.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

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
    "64/21": 64/21,  # Codon to amino acid ratio
}

MATCH_THRESHOLD = 0.05

# The Standard Genetic Code
# Codons map to amino acids (single letter codes)
# * = stop codon
GENETIC_CODE = {
    # First position U
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    # First position C
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    # First position A
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    # First position G
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Amino acid properties (for structural analysis)
AMINO_ACID_PROPERTIES = {
    "A": {"name": "Alanine", "hydropathy": 1.8, "volume": 88.6, "charge": 0, "polar": False},
    "R": {"name": "Arginine", "hydropathy": -4.5, "volume": 173.4, "charge": 1, "polar": True},
    "N": {"name": "Asparagine", "hydropathy": -3.5, "volume": 114.1, "charge": 0, "polar": True},
    "D": {"name": "Aspartate", "hydropathy": -3.5, "volume": 111.1, "charge": -1, "polar": True},
    "C": {"name": "Cysteine", "hydropathy": 2.5, "volume": 108.5, "charge": 0, "polar": False},
    "Q": {"name": "Glutamine", "hydropathy": -3.5, "volume": 143.8, "charge": 0, "polar": True},
    "E": {"name": "Glutamate", "hydropathy": -3.5, "volume": 138.4, "charge": -1, "polar": True},
    "G": {"name": "Glycine", "hydropathy": -0.4, "volume": 60.1, "charge": 0, "polar": False},
    "H": {"name": "Histidine", "hydropathy": -3.2, "volume": 153.2, "charge": 0.5, "polar": True},
    "I": {"name": "Isoleucine", "hydropathy": 4.5, "volume": 166.7, "charge": 0, "polar": False},
    "L": {"name": "Leucine", "hydropathy": 3.8, "volume": 166.7, "charge": 0, "polar": False},
    "K": {"name": "Lysine", "hydropathy": -3.9, "volume": 168.6, "charge": 1, "polar": True},
    "M": {"name": "Methionine", "hydropathy": 1.9, "volume": 162.9, "charge": 0, "polar": False},
    "F": {"name": "Phenylalanine", "hydropathy": 2.8, "volume": 189.9, "charge": 0, "polar": False},
    "P": {"name": "Proline", "hydropathy": -1.6, "volume": 112.7, "charge": 0, "polar": False},
    "S": {"name": "Serine", "hydropathy": -0.8, "volume": 89.0, "charge": 0, "polar": True},
    "T": {"name": "Threonine", "hydropathy": -0.7, "volume": 116.1, "charge": 0, "polar": True},
    "W": {"name": "Tryptophan", "hydropathy": -0.9, "volume": 227.8, "charge": 0, "polar": False},
    "Y": {"name": "Tyrosine", "hydropathy": -1.3, "volume": 193.6, "charge": 0, "polar": True},
    "V": {"name": "Valine", "hydropathy": 4.2, "volume": 140.0, "charge": 0, "polar": False},
    "*": {"name": "Stop", "hydropathy": 0, "volume": 0, "charge": 0, "polar": False},
}


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


def build_codon_matrix() -> np.ndarray:
    """Build a 64x4 matrix encoding codon structure.

    Each row is a codon, columns are:
    - Position 1 base (U=0, C=1, A=2, G=3)
    - Position 2 base
    - Position 3 base
    - Amino acid index (0-20)
    """
    base_map = {"U": 0, "C": 1, "A": 2, "G": 3}
    aa_list = sorted(set(GENETIC_CODE.values()))
    aa_map = {aa: i for i, aa in enumerate(aa_list)}

    matrix = []
    for codon, aa in sorted(GENETIC_CODE.items()):
        row = [
            base_map[codon[0]],
            base_map[codon[1]],
            base_map[codon[2]],
            aa_map[aa],
        ]
        matrix.append(row)

    return np.array(matrix, dtype=np.float64)


def build_degeneracy_matrix() -> np.ndarray:
    """Build a 21x64 matrix showing which codons map to which amino acids.

    Rows are amino acids, columns are codons.
    Entry is 1 if codon maps to amino acid, 0 otherwise.
    """
    aa_list = sorted(set(GENETIC_CODE.values()))
    codon_list = sorted(GENETIC_CODE.keys())

    matrix = np.zeros((len(aa_list), len(codon_list)))

    for j, codon in enumerate(codon_list):
        aa = GENETIC_CODE[codon]
        i = aa_list.index(aa)
        matrix[i, j] = 1

    return matrix


def build_property_matrix() -> np.ndarray:
    """Build matrix of amino acid properties.

    Shape: 20 x 4 (excluding stop codon)
    Columns: hydropathy, volume, charge, polar (0/1)
    """
    aa_list = [aa for aa in sorted(set(GENETIC_CODE.values())) if aa != "*"]

    matrix = []
    for aa in aa_list:
        props = AMINO_ACID_PROPERTIES[aa]
        row = [
            props["hydropathy"],
            props["volume"],
            props["charge"],
            1.0 if props["polar"] else 0.0,
        ]
        matrix.append(row)

    return np.array(matrix)


def analyze_genetic_code_structure():
    """Complete analysis of genetic code geometry."""
    print("GENETIC CODE STRUCTURE ANALYSIS")
    print("="*60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "The genetic code (64→21) encodes fundamental constants",
    }

    # Basic statistics
    n_codons = len(GENETIC_CODE)
    aa_set = set(GENETIC_CODE.values())
    n_aa = len(aa_set)  # 21 including stop

    print(f"\nBasic structure:")
    print(f"  Codons: {n_codons}")
    print(f"  Amino acids + stop: {n_aa}")
    print(f"  Degeneracy ratio: {n_codons}/{n_aa} = {n_codons/n_aa:.4f}")

    results["basic"] = {
        "n_codons": n_codons,
        "n_amino_acids": n_aa,
        "degeneracy_ratio": n_codons / n_aa,
    }

    # Check degeneracy ratio against constants
    print(f"\n--- Degeneracy Ratio Analysis ---")
    deg_ratio = n_codons / n_aa
    for const_name, const_val in CONSTANTS.items():
        error = abs(deg_ratio - const_val) / const_val
        if error < MATCH_THRESHOLD:
            print(f"  64/21 = {deg_ratio:.4f} ≈ {const_name} ({const_val:.4f}, {error*100:.2f}% error)")

    # The fact that it's 21 is interesting on its own
    print(f"\n  Note: 21 = hydrogen wavelength (cm) = B-DNA bp/turn × 2")

    # Codon degeneracy (how many codons per amino acid)
    degeneracy = {}
    for codon, aa in GENETIC_CODE.items():
        degeneracy[aa] = degeneracy.get(aa, 0) + 1

    print(f"\n--- Degeneracy per Amino Acid ---")
    deg_values = sorted(degeneracy.values())
    print(f"  Values: {deg_values}")
    print(f"  Unique: {sorted(set(deg_values))}")

    # Check ratios between degeneracy levels
    deg_unique = sorted(set(deg_values))
    for i, d1 in enumerate(deg_unique):
        for d2 in deg_unique[i+1:]:
            ratio = d2 / d1
            for const_name, const_val in CONSTANTS.items():
                error = abs(ratio - const_val) / const_val
                if error < MATCH_THRESHOLD:
                    print(f"  {d2}/{d1} = {ratio:.4f} ≈ {const_name} ({error*100:.2f}% error)")

    results["degeneracy"] = degeneracy

    # Codon matrix SVD
    print(f"\n--- Codon Matrix SVD ---")
    codon_matrix = build_codon_matrix()
    print(f"  Shape: {codon_matrix.shape}")

    U, S, Vt = np.linalg.svd(codon_matrix, full_matrices=False)
    print(f"  Singular values: {S}")

    codon_matches = count_constant_matches(S, bidirectional=True)
    total = sum(codon_matches.values())
    results["codon_matrix_svd"] = {
        "singular_values": list(S),
        "matches": codon_matches,
        "total_matches": total,
    }
    print(f"  Total matches: {total}")
    for name, count in codon_matches.items():
        if count > 0:
            print(f"    {name}: {count}")

    # Degeneracy matrix SVD (21 x 64)
    print(f"\n--- Degeneracy Matrix SVD (21×64) ---")
    deg_matrix = build_degeneracy_matrix()
    print(f"  Shape: {deg_matrix.shape}")

    U_deg, S_deg, Vt_deg = np.linalg.svd(deg_matrix, full_matrices=False)
    print(f"  Top singular values: {S_deg[:10]}")

    deg_matches = count_constant_matches(S_deg, bidirectional=True)
    total_deg = sum(deg_matches.values())
    results["degeneracy_matrix_svd"] = {
        "singular_values": list(S_deg),
        "matches": deg_matches,
        "total_matches": total_deg,
    }
    print(f"  Total matches: {total_deg}")
    for name, count in deg_matches.items():
        if count > 0:
            print(f"    {name}: {count}")

    # Amino acid property matrix
    print(f"\n--- Amino Acid Property Matrix SVD ---")
    prop_matrix = build_property_matrix()
    print(f"  Shape: {prop_matrix.shape}")

    # Normalize columns
    prop_norm = (prop_matrix - prop_matrix.mean(axis=0)) / (prop_matrix.std(axis=0) + 1e-10)

    U_prop, S_prop, Vt_prop = np.linalg.svd(prop_norm, full_matrices=False)
    print(f"  Singular values: {S_prop}")

    prop_matches = count_constant_matches(S_prop, bidirectional=True)
    total_prop = sum(prop_matches.values())
    results["property_matrix_svd"] = {
        "singular_values": list(S_prop),
        "matches": prop_matches,
        "total_matches": total_prop,
    }
    print(f"  Total matches: {total_prop}")
    for name, count in prop_matches.items():
        if count > 0:
            print(f"    {name}: {count}")

    # Analyze codon usage bias across species
    # Using average mammalian codon usage as an example
    print(f"\n--- Codon Position Bias ---")

    # Count bases at each position
    pos1 = {"U": 0, "C": 0, "A": 0, "G": 0}
    pos2 = {"U": 0, "C": 0, "A": 0, "G": 0}
    pos3 = {"U": 0, "C": 0, "A": 0, "G": 0}

    for codon in GENETIC_CODE.keys():
        pos1[codon[0]] += 1
        pos2[codon[1]] += 1
        pos3[codon[2]] += 1

    print(f"  Position 1: {pos1}")
    print(f"  Position 2: {pos2}")
    print(f"  Position 3: {pos3}")

    # Each position has 16 of each base (uniform in standard code)
    # But the MAPPING is structured

    # Analyze which amino acids are encoded by which first base
    first_base_aa = {base: set() for base in "UCAG"}
    for codon, aa in GENETIC_CODE.items():
        first_base_aa[codon[0]].add(aa)

    print(f"\n  First base → amino acids:")
    for base, aas in first_base_aa.items():
        print(f"    {base}: {len(aas)} amino acids: {sorted(aas)}")

    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nKey findings:")
    print(f"  1. Degeneracy ratio 64/21 ≈ {deg_ratio:.4f} (≈ 64/21 = {64/21:.4f})")
    print(f"  2. The number 21 appears again (cf. hydrogen, DNA)")
    print(f"  3. Codon matrix SVD matches: {total}")
    print(f"  4. Degeneracy matrix SVD matches: {total_deg}")
    print(f"  5. Property matrix SVD matches: {total_prop}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"codon_usage_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def analyze_21_connection():
    """Deep dive into why 21 keeps appearing."""
    print("\n" + "="*60)
    print("THE 21 CONNECTION")
    print("="*60)

    appearances = {
        "Hydrogen wavelength": "21.1 cm",
        "DNA bp/turn × 2": "10.5 × 2 = 21",
        "Genetic code outputs": "21 (20 AA + stop)",
        "Wow! angular velocity": "360°/21 ≈ 17.14°",
        "Triangular number T(6)": "1+2+3+4+5+6 = 21",
        "Fibonacci F(8)": "21",
        "C(7,2)": "21",
    }

    print("\n21 appears in:")
    for context, value in appearances.items():
        print(f"  {context}: {value}")

    # Mathematical properties of 21
    print(f"\nMathematical properties of 21:")
    print(f"  Prime factorization: 3 × 7")
    print(f"  21 = 3! + 15 = 6 + 15")
    print(f"  21 = T(6) (6th triangular number)")
    print(f"  21 = F(8) (8th Fibonacci number)")
    print(f"  21/π = {21/PI:.4f}")
    print(f"  21/e = {21/E:.4f}")
    print(f"  21/φ = {21/PHI:.4f}")

    # Check if 21 ratios match constants
    print(f"\nRatios involving 21:")
    for const_name, const_val in CONSTANTS.items():
        ratio = 21 / const_val
        # Check if this ratio is close to a simple number
        for simple in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20]:
            if abs(ratio - simple) / simple < 0.05:
                print(f"  21/{const_name} ≈ {simple} ({ratio:.4f})")


if __name__ == "__main__":
    results = analyze_genetic_code_structure()
    analyze_21_connection()
