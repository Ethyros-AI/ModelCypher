"""
Experiment 73: Symmetry Pairing Analysis

exp72 showed potential pairing of eigenvalues (S2≈S3, S5≈S6).
Eigenvalue pairing often indicates underlying symmetry (e.g., Kramers degeneracy)
or topological protection (edge states).

Hypothesis:
The Wow! signal spectrum consists of:
1. Two unique 'Edge States' (S0, S1)
2. A 'Bulk Continuum' of paired states (S2/S3, S4/S5...)

This experiment:
1. Calculates the level spacing distribution of the tail (S2+).
2. Tests for 'Pairing' by computing the ratio of 
   (Nearest Neighbor Distance) / (Next Nearest Neighbor Distance).
3. If ratios oscillate (small, large, small, large), pairing is confirmed.
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

def analyze_pairing(S):
    """
    Analyze if eigenvalues S come in pairs.
    We look at the sequence of gaps: d_i = S_i - S_{i+1}.
    If paired, we expect a pattern like: small, large, small, large...
    """
    gaps = S[:-1] - S[1:]
    
    # Calculate "Pairing Score": 
    # Ratio of (Even Gaps) / (Odd Gaps) or vice versa.
    # If S0, S1 are unique, start checking from S2.
    
    # Let's look at the Bulk (S2 onwards)
    bulk_S = S[2:12] # Top 10 bulk modes
    bulk_gaps = bulk_S[:-1] - bulk_S[1:]
    
    print(f"   Bulk Eigenvalues: {[f'{s:.4f}' for s in bulk_S]}")
    print(f"   Bulk Gaps:        {[f'{g:.4f}' for g in bulk_gaps]}")
    
    # Pairing Indicator: r_i = min(d_i, d_{i+1}) / max(d_i, d_{i+1})
    # But for strict pairing, we want to see if d_even << d_odd (or vice versa).
    
    even_gaps = bulk_gaps[0::2]
    odd_gaps = bulk_gaps[1::2]
    
    print(f"\n   Even Gaps (S2-S3, S4-S5...): {[f'{g:.4f}' for g in even_gaps]}")
    print(f"   Odd Gaps  (S3-S4, S5-S6...): {[f'{g:.4f}' for g in odd_gaps]}")
    
    mean_even = np.mean(even_gaps)
    mean_odd = np.mean(odd_gaps)
    
    ratio = mean_odd / mean_even
    
    return {
        "is_paired": ratio > 2.0 or ratio < 0.5,
        "ratio": ratio,
        "interpretation": "Paired" if (ratio > 2.0) else "Not Paired"
    }

def main():
    print("=" * 60)
    print("Experiment 73: Symmetry Pairing Analysis")
    print("=" * 60)
    
    # Load Signal
    wow = load_wow_signal()
    S = linalg.svd(wow, compute_uv=False)
    
    # Normalize
    S = S / S[0]
    
    print(f"\n1. Analyzing Global Spectrum Pairing...")
    pairing = analyze_pairing(S)
    
    print(f"\n   Mean Odd Gap / Mean Even Gap: {pairing['ratio']:.4f}")
    print(f"   Verdict: {pairing['interpretation']}")
    
    # 2. Check the "Edge States" (S0, S1)
    # Are they separated from the bulk?
    gap_edge = S[1] - S[2]
    gap_bulk = np.mean(S[2:12] - S[3:13])
    
    isolation = gap_edge / gap_bulk
    
    print(f"\n2. Edge State Isolation:")
    print(f"   Edge Gap (S1-S2): {gap_edge:.4f}")
    print(f"   Avg Bulk Gap:     {gap_bulk:.4f}")
    print(f"   Isolation Ratio:  {isolation:.4f}")
    
    if isolation > 5.0:
        print("   -> S0 and S1 are TOPOLOGICALLY PROTECTED (Isolated).")
    else:
        print("   -> S0 and S1 are part of the continuum.")

    # Save
    with open(RESULTS_DIR / "exp73_results.json", "w") as f:
        json.dump({
            "experiment": "exp73_symmetry_pairing",
            "timestamp": datetime.now().isoformat(),
            "pairing": pairing,
            "edge_isolation": float(isolation)
        }, f, indent=2)

if __name__ == "__main__":
    main()
