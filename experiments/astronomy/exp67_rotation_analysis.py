"""
Experiment 67: Rotation Analysis of Period 12 Payload

exp64 verified:
1. Segments are structurally related (Procrustes > 0.3).
exp66 verified:
1. Semantic projection is an artifact (random noise also maps to "Excitement").

Hypothesis:
The message is in the SEQUENCE of rotations.
The payload is 4 segments. Each is a rotated version of the motif.
The information is encoded in the rotation angles between segments.

This experiment:
1. Extracts the 4 Period 12 segments.
2. Computes the optimal rotation matrix R between consecutive segments.
3. Computes the rotation angle theta from trace(R).
4. Checks if the angles match constants (phi, pi, e, etc.).
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

def compute_rotation_angle(R):
    """
    Compute rotation angle from orthogonal matrix R.
    For N-dimensions, this is complex, but we can look at the
    eigenvalues of R, which are exp(±i*theta).
    The trace is also related: trace(R) = sum(cos(theta_i)).
    
    We'll assume a principal rotation angle.
    """
    # Force orthogonality
    U, _, Vh = linalg.svd(R)
    R_ortho = U @ Vh
    
    # Eigenvalues of R are complex on unit circle
    eigvals = linalg.eigvals(R_ortho)
    angles = np.angle(eigvals)
    
    # Filter positive angles (conjugate pairs exist)
    pos_angles = np.sort(angles[angles > 1e-5])
    
    return pos_angles

def main():
    print("=" * 60)
    print("Experiment 67: Rotation Analysis of Period 12 Payload")
    print("=" * 60)
    
    # 1. Extract Segments
    wow = load_wow_signal()
    period = 12
    n_time = wow.shape[1]
    n_segments = n_time // period # Should be 4
    segments = []
    
    print(f"\n1. Extracting {n_segments} segments...")
    for i in range(n_segments):
        start = i * period
        end = start + period
        seg = wow[:, start:end]
        # Normalize
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)
        segments.append(seg)
        
    # 2. Compute Rotations
    print("\n2. Computing Rotations between consecutive segments...")
    
    transitions = []
    
    for i in range(n_segments - 1):
        A = segments[i]
        B = segments[i+1]
        
        # Procrustes: Find R such that ||B - AR|| is minimized
        # M = A.T @ B
        # U, S, Vh = svd(M)
        # R = U @ Vh
        
        # NOTE: A is (82, 12). Procrustes usually aligns "rows" or "cols".
        # We want to map the 12-step trajectory of segment A to segment B.
        # So we treat them as (82, 12) matrices.
        # But rotation must be square.
        
        # Let's assume the rotation happens in the 12-dimensional time embedding space?
        # Or the 82-dimensional frequency space?
        
        # Hypothesis: The frequency vector rotates. (82x82 rotation)
        # M = B @ A.T (Aligns frequency dimensions)
        
        M_freq = B @ A.T
        U, _, Vh = linalg.svd(M_freq)
        R_freq = U @ Vh
        
        angles = compute_rotation_angle(R_freq)
        primary_angle = angles[-1] if len(angles) > 0 else 0
        
        transitions.append({
            "from": i,
            "to": i+1,
            "angle": float(primary_angle),
            "angle_deg": float(np.degrees(primary_angle)),
            "all_angles": [float(a) for a in angles]
        })
        
        print(f"   Transition {i}->{i+1}: Principal Angle = {primary_angle:.4f} rad ({np.degrees(primary_angle):.1f} deg)")
        
    # 3. Analyze Angles
    print("\n3. Checking Constants...")
    constants = {
        "phi": PHI,
        "pi": PI,
        "e": E,
        "pi/2": PI/2,
        "pi/3": PI/3,
        "pi/4": PI/4,
        "pi/5": PI/5, # 36 deg (pentagon)
        "2pi/5": 2*PI/5 # 72 deg
    }
    
    matches = []
    for t in transitions:
        angle = t["angle"]
        best_match = None
        min_err = 1.0
        
        for name, val in constants.items():
            err = abs(angle - val) / val
            if err < min_err:
                min_err = err
                best_match = name
                
        if min_err < 0.10:
            matches.append(f"Transition {t['from']}->{t['to']}: {angle:.4f} ≈ {best_match} ({min_err*100:.1f}%)")
            
    if not matches:
        print("   No obvious constants found in rotation angles.")
    else:
        for m in matches:
            print(f"   {m}")
            
    # Save
    with open(RESULTS_DIR / "exp67_results.json", "w") as f:
        json.dump({
            "experiment": "exp67_rotation_analysis",
            "timestamp": datetime.now().isoformat(),
            "transitions": transitions,
            "matches": matches
        }, f, indent=2)

if __name__ == "__main__":
    main()
