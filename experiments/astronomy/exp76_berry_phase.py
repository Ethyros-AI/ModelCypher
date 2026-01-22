"""
Experiment 76: Berry Phase Integration

Objective:
Calculate the Geometric (Berry) Phase accumulated by the signal's 
eigenstate as it evolves over time.

Hypothesis:
If the signal is a topological object (Instanton/Soliton), it will 
	have a non-zero Topological Charge (Winding Number).
This charge is measured by the total Berry Phase / 2pi.

Method:
1. Slice the signal into small time steps dt.
2. For each step, compute the instantaneous eigenvector |u(t)>.
3. Calculate the phase change: exp(i * gamma) = <u(t) | u(t+dt)>.
4. Sum the phases: Gamma = Sum( arg(<u(t)|u(t+dt)>) ).
5. Check if Gamma is a multiple of pi or 2pi.
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

def compute_berry_phase(signal, window_size=4, step=1):
    """
    Compute the discrete Berry phase for the dominant eigenmode.
    """
    n_freq, n_time = signal.shape
    
    # We track the dominant spatial mode (U vector) evolution
    # We need to compute eigenvectors for sliding windows
    
    phases = []
    overlaps = []
    
    # Initial state
    start = 0
    end = start + window_size
    seg = signal[:, start:end]
    U, _, _ = linalg.svd(seg, full_matrices=False)
    u_prev = U[:, 0] # Top mode
    
    for t in range(step, n_time - window_size, step):
        start = t
        end = start + window_size
        seg = signal[:, start:end]
        
        U, _, _ = linalg.svd(seg, full_matrices=False)
        u_curr = U[:, 0]
        
        # Compute overlap <u_prev | u_curr>
        overlap = np.dot(u_prev.conj(), u_curr)
        
        # Fix gauge freedom (SVD sign ambiguity)
        # We assume continuity: overlap should be positive real if possible
        # But if there is a real phase change, we capture it.
        # For real vectors, phase is 0 or pi.
        # But we can treat them as potentially rotating in a plane if we had 2 modes.
        
        # Let's track the phase of the overlap relative to a reference?
        # Actually, for 1D real subspace, Berry phase is 0 or pi (Zak phase).
        
        phases.append(np.angle(overlap)) # Will be 0 or pi
        overlaps.append(overlap)
        
        # Update (with gauge alignment to maximize smoothness)
        if overlap < 0:
            u_prev = -u_curr
        else:
            u_prev = u_curr
            
    # Accumulate 'Geometric Phase' in 2D subspace?
    # To see a continuous phase, we need at least 2 dimensions (U[:, 0] and U[:, 1])
    # Let's project the evolution onto the U0-U1 plane.
    
    return overlaps

def compute_2d_berry_phase(signal, window_size=4, step=1):
    """
    Compute Berry phase in the 2D subspace of the first two eigenmodes.
    Gamma = Integral ( u0 * du1 - u1 * du0 )
    """
    n_freq, n_time = signal.shape
    accumulated_phase = 0.0
    
    # Initial state
    seg = signal[:, 0:window_size]
    U, _, _ = linalg.svd(seg, full_matrices=False)
    # The subspace is spanned by u0, u1
    P_prev = U[:, :2] # 82x2 projector
    
    for t in range(step, n_time - window_size, step):
        seg = signal[:, t:t+window_size]
        U, _, _ = linalg.svd(seg, full_matrices=False)
        P_curr = U[:, :2]
        
        # Compute unitary connection between subspaces
        # M = P_prev.T @ P_curr (2x2 matrix)
        M = np.dot(P_prev.T, P_curr)
        
        # The phase is the angle of the determinant (for U(1) bundle)
        # det(M) ~ exp(-i * delta_gamma)
        det_M = linalg.det(M)
        d_gamma = np.angle(det_M)
        
        accumulated_phase += d_gamma
        
        P_prev = P_curr
        
    return accumulated_phase

def main():
    print("=" * 60)
    print("Experiment 76: Berry Phase Integration")
    print("=" * 60)
    
    wow = load_wow_signal()
    
    # 1. Compute 2D Berry Phase
    print("\n1. Integrating Geometric Phase (Subspace U0-U1)...")
    gamma = compute_2d_berry_phase(wow, window_size=6, step=1)
    
    print(f"   Total Accumulated Phase (Gamma): {gamma:.4f} rad")
    print(f"   Gamma / Pi: {gamma / np.pi:.4f}")
    print(f"   Gamma / 2Pi: {gamma / (2*np.pi):.4f}")
    
    # 2. Check for Quantization
    # Topological charges are integers (N * 2pi) or half-integers (N * pi)
    
    winding_number = gamma / (2 * np.pi)
    remainder = abs(winding_number - round(winding_number))
    
    print(f"\n2. Topological Charge Analysis:")
    print(f"   Winding Number (W): {winding_number:.4f}")
    print(f"   Deviation from Integer: {remainder:.4f}")
    
    if remainder < 0.1:
        print(f"   -> QUANTIZED! W = {round(winding_number)}")
        print("   The signal has a non-trivial topological winding.")
    elif abs(abs(gamma) - np.pi) < 0.2:
        print("   -> PI PHASE (Zak Phase).")
        print("   The signal has a symmetry-protected crossing.")
    else:
        print("   -> Not quantized. Continuous evolution.")
        
    # Save
    with open(RESULTS_DIR / "exp76_results.json", "w") as f:
        json.dump({
            "experiment": "exp76_berry_phase",
            "timestamp": datetime.now().isoformat(),
            "gamma": float(gamma),
            "winding_number": float(winding_number)
        }, f, indent=2)

if __name__ == "__main__":
    main()
