"""
Experiment 64: Period 12 Payload Structure

exp63 findings:
- Period 12 segments have essentially ZERO raw correlation (-0.002).
- BUT their eigenvalues encode Euler's number (e):
  S0/S1 = 2.645 ≈ e (2.7% error)
  S1/S2 = 2.537 ≈ e (6.7% error)

Hypothesis:
The segments are NOT copies (which would correlate).
They are ISOMETRIC VARIANTS. They share the same internal geometry (eigenvalues),
but are rotated in state space.

This experiment:
1. Extracts Period 12 segments.
2. Computes their eigenvalues individually.
3. Aligns them using Procrustes analysis (finding the optimal rotation).
4. Checks if they align AFTER rotation (structural similarity).
5. Verifies the "double e" signature in the stacked motif.
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

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e

def analyze_eigenstructure(matrix, label=""):
    """Compute and analyze eigenvalues of a matrix."""
    # SVD
    U, S, Vh = linalg.svd(matrix, full_matrices=False)
    
    results = {
        "eigenvalues": [float(s) for s in S[:10]],
        "ratios": {},
        "matches": {}
    }
    
    # Calculate ratios
    for i in range(min(4, len(S)-1)):
        if S[i+1] > 1e-10:
            ratio = S[i] / S[i+1]
            results["ratios"][f"S{i}/S{i+1}"] = float(ratio)
            
            # Check constants
            best_match = None
            min_err = 1.0
            
            for name, val in [("phi", PHI), ("pi", PI), ("e", E), ("sqrt2", np.sqrt(2)), ("sqrt3", np.sqrt(3))]:
                err = abs(ratio - val) / val
                if err < min_err:
                    min_err = err
                    best_match = name
            
            if min_err < 0.10: # 10% tolerance
                results["matches"][f"S{i}/S{i+1}"] = {
                    "value": float(ratio),
                    "match": best_match,
                    "error": float(min_err)
                }

    return results

def procrustes_alignment(segments):
    """
    Test if segments are rotations of each other.
    
    For each pair of segments A and B:
    1. Solve for orthogonal matrix R that minimizes ||A - B@R||
    2. Compute aligned similarity
    """
    n_segments = len(segments)
    raw_sims = []
    aligned_sims = []
    
    print(f"   Aligning {n_segments} segments (Period 12)...")
    
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            A = segments[i]
            B = segments[j]
            
            # Raw correlation (flattened)
            raw_corr = np.corrcoef(A.flatten(), B.flatten())[0, 1]
            raw_sims.append(raw_corr)
            
            # Procrustes Alignment (Orthogonal Procrustes problem)
            # Find R to map B to A
            M = B.T @ A
            U, S, Vh = linalg.svd(M)
            R = U @ Vh
            
            B_aligned = B @ R
            
            # Aligned correlation
            aligned_corr = np.corrcoef(A.flatten(), B_aligned.flatten())[0, 1]
            aligned_sims.append(aligned_corr)
            
    return {
        "mean_raw_sim": float(np.mean(raw_sims)),
        "mean_aligned_sim": float(np.mean(aligned_sims)),
        "max_aligned_sim": float(np.max(aligned_sims)),
        "improvement": float(np.mean(aligned_sims) - np.mean(raw_sims))
    }

def main():
    print("=" * 60)
    print("Experiment 64: Period 12 Payload Structure")
    print("=" * 60)
    
    # Load signal
    wow = load_wow_signal()
    n_freq, n_time = wow.shape
    print(f"   Signal shape: {n_freq} x {n_time}")
    
    # Extract Period 12 segments
    period = 12
    n_segments = n_time // period
    segments = []
    
    print(f"\n1. Extracting {n_segments} segments of length {period}...")
    for i in range(n_segments):
        start = i * period
        end = start + period
        seg = wow[:, start:end]
        segments.append(seg)
        
    # Stack and analyze motif
    print("\n2. Analyzing Stacked Motif (The 'Average' Symbol)...")
    stacked = np.stack(segments, axis=0)
    mean_motif = np.mean(stacked, axis=0)
    
    motif_structure = analyze_eigenstructure(mean_motif, "Mean Motif")
    print(f"   Top Eigenvalues: {[f'{x:.2f}' for x in motif_structure['eigenvalues'][:5]]}")
    
    print("   Checking for constants in eigenvalues:")
    if not motif_structure['matches']:
        print("   No obvious constants found.")
    for ratio, data in motif_structure['matches'].items():
        print(f"   {ratio} = {data['value']:.3f} ≈ {data['match']} (error: {data['error']*100:.2f}%)")
        
    # Analyze individual segments
    print("\n3. Analyzing Individual Segment Geometry...")
    segment_matches = {"phi": 0, "pi": 0, "e": 0}
    total_ratios = 0
    
    for i, seg in enumerate(segments):
        struct = analyze_eigenstructure(seg)
        # Check first ratio only (S0/S1)
        if "S0/S1" in struct["matches"]:
            match = struct["matches"]["S0/S1"]["match"]
            if match in segment_matches:
                segment_matches[match] += 1
        
    print("   Dominant constant in S0/S1 across segments:")
    for const, count in segment_matches.items():
        print(f"   {const}: {count}/{len(segments)} segments")
        
    # Structural Alignment
    print("\n4. Testing Structural Alignment (Are they rotations?)...")
    alignment_stats = procrustes_alignment(segments)
    
    print(f"   Mean Raw Correlation:     {alignment_stats['mean_raw_sim']:.4f} (Should be near 0)")
    print(f"   Mean Aligned Correlation: {alignment_stats['mean_aligned_sim']:.4f}")
    print(f"   Alignment Improvement:    +{alignment_stats['improvement']:.4f}")
    
    # Findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    findings = []
    
    # Check the "double e" hypothesis
    r1 = motif_structure['ratios'].get('S0/S1', 0)
    r2 = motif_structure['ratios'].get('S1/S2', 0)
    
    e_err1 = abs(r1 - E) / E
    e_err2 = abs(r2 - E) / E
    
    if e_err1 < 0.05:
        findings.append(f"Motif S0/S1 encodes e (error {e_err1*100:.2f}%)")
    if e_err2 < 0.10:
        findings.append(f"Motif S1/S2 encodes e (error {e_err2*100:.2f}%)")
        
    if alignment_stats['mean_aligned_sim'] > 0.5:
        findings.append(f"Segments are STRUCTURALLY ALIGNED (sim {alignment_stats['mean_aligned_sim']:.2f})")
        findings.append("The message is encoded in the GEOMETRY, not the waveform.")
    elif alignment_stats['mean_aligned_sim'] > 0.2:
         findings.append(f"Weak structural alignment (sim {alignment_stats['mean_aligned_sim']:.2f})")
    else:
         findings.append("No structural alignment found.")

    for f in findings:
        print(f"   - {f}")
        
    # Save
    output = {
        "experiment": "exp64_period12_payload",
        "timestamp": datetime.now().isoformat(),
        "motif_structure": motif_structure,
        "alignment_stats": alignment_stats,
        "segment_matches": segment_matches
    }
    
    with open(RESULTS_DIR / "exp64_results.json", "w") as f:
        json.dump(output, f, indent=2)
        
if __name__ == "__main__":
    main()
