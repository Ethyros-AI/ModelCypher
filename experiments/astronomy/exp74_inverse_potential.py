"""
Experiment 74: Inverse Hamiltonian Reconstruction (Fixed)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def reconstruct_jacobi_matrix(eigenvalues):
    # Ensure eigenvalues is array
    lambda_tgt = np.sort(np.array(eigenvalues))
    N = len(lambda_tgt)
    
    # Precompute kinetic matrix T
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 2.0
        if i < N - 1:
            T[i, i+1] = -1.0
            T[i+1, i] = -1.0
            
    # Define loss function with closure over T and lambda_tgt
    def loss(V):
        H = T + np.diag(V)
        w = linalg.eigvalsh(H)
        w_sorted = np.sort(w)
        return np.sum((w_sorted - lambda_tgt)**2)
    
    # Optimization
    V0 = np.zeros(N)
    
    try:
        res = minimize(loss, V0, method='L-BFGS-B', tol=1e-6)
        return res.x, res.fun
    except Exception as e:
        print(f"Optimization error: {e}")
        return V0, 999.0

def analyze_potential_shape(V):
    V_flip = V[::-1]
    symmetry_err = np.mean((V - V_flip)**2)
    smoothness = np.sum((V[1:] - V[:-1])**2)
    x = np.linspace(-1, 1, len(V))
    
    # Harmonic
    p_harm = np.polyfit(x, V, 2)
    fit_harm = np.polyval(p_harm, x)
    # R2
    ss_tot = np.sum((V - np.mean(V))**2) + 1e-10
    ss_res_harm = np.sum((V - fit_harm)**2)
    r2_harm = 1 - ss_res_harm/ss_tot
    
    # Double Well (x^4)
    p_well = np.polyfit(x, V, 4)
    fit_well = np.polyval(p_well, x)
    ss_res_well = np.sum((V - fit_well)**2)
    r2_well = 1 - ss_res_well/ss_tot
    
    return {
        "symmetry_error": float(symmetry_err),
        "smoothness": float(smoothness),
        "harmonic_r2": float(r2_harm),
        "double_well_r2": float(r2_well),
        "poly_coeffs": p_well.tolist()
    }

def main():
    print("=" * 60)
    print("Experiment 74: Inverse Hamiltonian Reconstruction")
    print("=" * 60)
    
    wow = load_wow_signal()
    S = linalg.svd(wow, compute_uv=False)
    
    # Use top 20 modes
    S_top = S[:20]
    
    print("\n1. Mapping Spectrum to Energy Levels...")
    # Map probability amplitude to Energy: E ~ -log(S)
    E_levels = -np.log(S_top / S_top[0] + 1e-10)
    
    print(f"   Energy Levels: {[f'{e:.2f}' for e in E_levels[:5]]}...")
    
    print("\n2. Reconstructing Potential V(x)...")
    V_recon, error = reconstruct_jacobi_matrix(E_levels)
    
    print(f"   Reconstruction Error: {error:.6f}")
    print(f"   Potential V(x):")
    print("   [" + ", ".join([f"{v:.2f}" for v in V_recon]) + "]")
    
    print("\n3. Analyzing Shape...")
    stats = analyze_potential_shape(V_recon)
    
    print(f"   Harmonic Fit (R2): {stats['harmonic_r2']:.4f}")
    print(f"   Double Well Fit (R2): {stats['double_well_r2']:.4f}")
    print(f"   Symmetry Error: {stats['symmetry_error']:.4f}")
    
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if stats['harmonic_r2'] > 0.9:
        print("POTENTIAL IS A HARMONIC OSCILLATOR.")
        print("This implies a bound system with linear restoring force.")
    elif stats['double_well_r2'] > 0.9:
        print("POTENTIAL IS A DOUBLE WELL.")
        print("This implies a bistable system (tunneling).")
    else:
        print("POTENTIAL IS COMPLEX/UNSTRUCTURED.")
        
    # Check edges
    edge_avg = (V_recon[0] + V_recon[-1]) / 2
    mid_avg = np.mean(V_recon[8:12])
    print(f"   Edge Height: {edge_avg:.2f}")
    print(f"   Center Height: {mid_avg:.2f}")
    
    if edge_avg < mid_avg - 1.0:
        print("   -> Potential has DEEP EDGE WELLS (Surface States).")
    elif mid_avg < edge_avg - 1.0:
        print("   -> Potential is CONFINING (Particle in a Box).")

    with open(RESULTS_DIR / "exp74_results.json", "w") as f:
        json.dump({
            "experiment": "exp74_inverse_potential",
            "potential": V_recon.tolist(),
            "stats": stats
        }, f, indent=2)

if __name__ == "__main__":
    main()
