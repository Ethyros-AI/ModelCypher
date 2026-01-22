"""
Audit Part 3: What's special about the S[1]/S[2] ratio?
This is the actual anomaly (z=3.86 vs FRBs).
"""

import sys
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.io import readsav
import h5py

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"


def main():
    print("=" * 60)
    print("Wow! S[1]/S[2] Anomaly Analysis")
    print("=" * 60)

    # Load Wow! data
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)

    U, S, Vt = linalg.svd(signal, full_matrices=False)

    print("\n1. WOW! SIGNAL SVD")
    print("-" * 40)
    print(f"S[0] = {S[0]:.4f}")
    print(f"S[1] = {S[1]:.4f}")
    print(f"S[2] = {S[2]:.4f}")
    print(f"S[3] = {S[3]:.4f}")
    print(f"\nS[1]/S[2] = {S[1]/S[2]:.4f} (this is 3.86σ above FRB mean)")

    # What do the singular vectors look like?
    print("\n2. WHAT DO THE MODES REPRESENT?")
    print("-" * 40)

    # First right singular vector (frequency pattern for mode 0)
    print("V[0] (mode 0, frequency pattern):")
    print(f"  Max at channel: {np.argmax(np.abs(Vt[0]))}")
    print(f"  Max value: {Vt[0].max():.4f}")

    print("V[1] (mode 1, frequency pattern):")
    print(f"  Max at channel: {np.argmax(np.abs(Vt[1]))}")
    print(f"  Max value: {Vt[1].max():.4f}")

    print("V[2] (mode 2, frequency pattern):")
    print(f"  Max at channel: {np.argmax(np.abs(Vt[2]))}")
    print(f"  Max value: {Vt[2].max():.4f}")

    # Reconstruct with just first 2 modes vs first 3 modes
    print("\n3. RECONSTRUCTION ANALYSIS")
    print("-" * 40)

    recon_1 = S[0] * np.outer(U[:, 0], Vt[0])
    recon_2 = recon_1 + S[1] * np.outer(U[:, 1], Vt[1])
    recon_3 = recon_2 + S[2] * np.outer(U[:, 2], Vt[2])

    err_1 = np.linalg.norm(signal - recon_1, 'fro') / np.linalg.norm(signal, 'fro')
    err_2 = np.linalg.norm(signal - recon_2, 'fro') / np.linalg.norm(signal, 'fro')
    err_3 = np.linalg.norm(signal - recon_3, 'fro') / np.linalg.norm(signal, 'fro')

    print(f"Reconstruction error with 1 mode: {err_1*100:.2f}%")
    print(f"Reconstruction error with 2 modes: {err_2*100:.2f}%")
    print(f"Reconstruction error with 3 modes: {err_3*100:.2f}%")

    # Variance explained
    total_var = np.sum(S**2)
    var_1 = S[0]**2 / total_var
    var_2 = (S[0]**2 + S[1]**2) / total_var
    var_3 = (S[0]**2 + S[1]**2 + S[2]**2) / total_var

    print(f"\nVariance explained by 1 mode: {var_1*100:.2f}%")
    print(f"Variance explained by 2 modes: {var_2*100:.2f}%")
    print(f"Variance explained by 3 modes: {var_3*100:.2f}%")

    # 4. Compare to FRBs in detail
    print("\n4. FRB DETAILED COMPARISON")
    print("-" * 40)

    frb_dir = Path(__file__).parent / "data" / "raw"
    frb_files = sorted(frb_dir.glob("FRB*.h5"))[:45]

    frb_data = []
    for frb_path in frb_files:
        try:
            with h5py.File(frb_path, 'r') as f:
                wfall = f['frb']['wfall'][:].astype(np.float64)
                wfall = np.nan_to_num(wfall, nan=0.0, posinf=0.0, neginf=0.0)
                if wfall.size > 0:
                    _, S_frb, _ = linalg.svd(wfall, full_matrices=False)
                    if len(S_frb) >= 3:
                        r1 = S_frb[0] / S_frb[1]
                        r2 = S_frb[1] / S_frb[2]
                        frb_data.append({
                            'name': frb_path.stem,
                            'shape': wfall.shape,
                            'r1': r1,
                            'r2': r2
                        })
        except:
            pass

    print(f"Loaded {len(frb_data)} FRBs")

    # Sort by r2 to see distribution
    frb_data.sort(key=lambda x: x['r2'])

    print(f"\nFRB S[1]/S[2] distribution:")
    r2_values = [f['r2'] for f in frb_data]
    print(f"  Min: {min(r2_values):.4f}")
    print(f"  Max: {max(r2_values):.4f}")
    print(f"  Mean: {np.mean(r2_values):.4f}")
    print(f"  Std: {np.std(r2_values):.4f}")
    print(f"  Wow!: {S[1]/S[2]:.4f}")

    # How many FRBs have r2 > Wow!?
    higher = sum(1 for r2 in r2_values if r2 > S[1]/S[2])
    print(f"\n  FRBs with S[1]/S[2] > Wow!: {higher}/{len(r2_values)}")

    # Is there any FRB with r2 close to π?
    pi = np.pi
    closest_to_pi = min(frb_data, key=lambda x: abs(x['r2'] - pi))
    print(f"\n  Closest FRB to π (3.14): {closest_to_pi['name']} with r2={closest_to_pi['r2']:.4f}")

    # 5. What would cause high S[1]/S[2]?
    print("\n5. INTERPRETATION")
    print("-" * 40)
    print("High S[1]/S[2] means mode 1 is much stronger than mode 2.")
    print("In other words: the signal has exactly TWO dominant directions,")
    print("with a large gap before the third direction.")
    print("\nThis suggests the Wow! signal has a very clean 2D structure,")
    print("unlike typical FRBs which have more distributed spectral energy.")

    # 6. Is proximity to π meaningful?
    print("\n6. IS PROXIMITY TO π MEANINGFUL?")
    print("-" * 40)
    print(f"Wow! S[1]/S[2] = {S[1]/S[2]:.4f}")
    print(f"π = {np.pi:.4f}")
    print(f"Error: {abs(S[1]/S[2] - np.pi)/np.pi*100:.2f}%")
    print("\nTo test if this is meaningful, we need to ask:")
    print("  1. Is there a physical reason for π to appear here?")
    print("  2. How often do random 2D structures produce ratios near π?")

    # Test: generate random 2D structures and check their r2
    print("\nGenerating random rank-2 dominated matrices...")
    n_trials = 10000
    pi_hits = 0
    for _ in range(n_trials):
        # Create a matrix with 2 dominant modes
        U_rand = np.random.randn(82, 2)
        S_rand = np.array([10.0, 3.0])  # Arbitrary 2 dominant singular values
        V_rand = np.random.randn(2, 50)
        base = U_rand @ np.diag(S_rand) @ V_rand
        # Add some noise
        noise = np.random.randn(82, 50) * 0.5
        mat = base + noise
        _, S_test, _ = linalg.svd(mat, full_matrices=False)
        r2 = S_test[1] / S_test[2]
        if abs(r2 - np.pi) / np.pi < 0.05:  # Within 5% of π
            pi_hits += 1

    print(f"Random rank-2 matrices with S[1]/S[2] within 5% of π: {pi_hits}/{n_trials} ({pi_hits/n_trials*100:.2f}%)")


if __name__ == "__main__":
    main()
