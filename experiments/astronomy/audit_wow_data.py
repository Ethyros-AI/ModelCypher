"""
Audit the Wow! signal data: what exactly are we analyzing?
No interpretation, just facts about the data.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.io import readsav

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"


def main():
    print("=" * 60)
    print("Wow! Signal Data Audit")
    print("=" * 60)

    # 1. Load raw data
    print("\n1. RAW DATA")
    print("-" * 40)

    if not DATA_PATH.exists():
        print(f"ERROR: File not found at {DATA_PATH}")
        return

    data = readsav(str(DATA_PATH))
    print(f"Keys in .sav file: {list(data.keys())}")

    oseti = data['oseti'][0]
    print(f"Fields in oseti record: {oseti.dtype.names}")

    snr = oseti['SNR']
    print(f"\nSNR matrix:")
    print(f"  Shape: {snr.shape}")
    print(f"  Dtype: {snr.dtype}")
    print(f"  Min: {np.nanmin(snr):.4f}")
    print(f"  Max: {np.nanmax(snr):.4f}")
    print(f"  Mean: {np.nanmean(snr):.4f}")
    print(f"  NaN count: {np.sum(np.isnan(snr))}")
    print(f"  Inf count: {np.sum(np.isinf(snr))}")

    # 2. Clean data (as experiments do)
    print("\n2. CLEANED DATA")
    print("-" * 40)
    signal = snr.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Shape: {signal.shape}")
    print(f"  Min: {signal.min():.4f}")
    print(f"  Max: {signal.max():.4f}")
    print(f"  Mean: {signal.mean():.4f}")
    print(f"  Std: {signal.std():.4f}")

    # 3. SVD analysis (what experiments compute)
    print("\n3. SVD ANALYSIS")
    print("-" * 40)
    U, S, Vt = linalg.svd(signal, full_matrices=False)
    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")

    print(f"\n  Top 10 singular values (raw):")
    for i, s in enumerate(S[:10]):
        print(f"    S[{i}] = {s:.4f}")

    print(f"\n  Top 10 singular values (normalized to S[0]=1):")
    S_norm = S / S[0]
    for i, s in enumerate(S_norm[:10]):
        print(f"    S[{i}] = {s:.4f}")

    # 4. Ratios that Gemini claimed
    print("\n4. EIGENVALUE RATIOS")
    print("-" * 40)
    print(f"  S[0]/S[1] = {S[0]/S[1]:.4f}  (Gemini claimed ≈ φ = 1.618)")
    print(f"  S[1]/S[2] = {S[1]/S[2]:.4f}  (Gemini claimed ≈ π = 3.142)")
    print(f"  S[2]/S[3] = {S[2]/S[3]:.4f}")
    print(f"  S[3]/S[4] = {S[3]/S[4]:.4f}")

    # 5. Compare to mathematical constants
    print("\n5. COMPARISON TO CONSTANTS")
    print("-" * 40)
    phi = (1 + np.sqrt(5)) / 2
    pi = np.pi
    e = np.e

    r1 = S[0]/S[1]
    r2 = S[1]/S[2]

    print(f"  φ (golden ratio) = {phi:.4f}")
    print(f"  π = {pi:.4f}")
    print(f"  e = {e:.4f}")
    print(f"\n  S[0]/S[1] = {r1:.4f}, error from φ: {abs(r1-phi)/phi*100:.2f}%")
    print(f"  S[1]/S[2] = {r2:.4f}, error from π: {abs(r2-pi)/pi*100:.2f}%")

    # 6. Is this unusual? Generate random matrices and compare
    print("\n6. NULL HYPOTHESIS TEST")
    print("-" * 40)
    print("  Generating 1000 random matrices with same shape and similar statistics...")

    n_trials = 1000
    ratios_r1 = []
    ratios_r2 = []

    for _ in range(n_trials):
        # Random matrix with same shape and similar mean/std
        rand_matrix = np.random.randn(82, 50) * signal.std() + signal.mean()
        _, S_rand, _ = linalg.svd(rand_matrix, full_matrices=False)
        ratios_r1.append(S_rand[0] / S_rand[1])
        ratios_r2.append(S_rand[1] / S_rand[2])

    ratios_r1 = np.array(ratios_r1)
    ratios_r2 = np.array(ratios_r2)

    print(f"\n  Random S[0]/S[1]: mean={ratios_r1.mean():.4f}, std={ratios_r1.std():.4f}")
    print(f"  Wow! S[0]/S[1] = {r1:.4f}")
    z1 = (r1 - ratios_r1.mean()) / ratios_r1.std()
    print(f"  z-score: {z1:.2f}")

    print(f"\n  Random S[1]/S[2]: mean={ratios_r2.mean():.4f}, std={ratios_r2.std():.4f}")
    print(f"  Wow! S[1]/S[2] = {r2:.4f}")
    z2 = (r2 - ratios_r2.mean()) / ratios_r2.std()
    print(f"  z-score: {z2:.2f}")

    # 7. How often does random hit φ or π?
    print("\n7. HOW OFTEN DO RANDOM MATRICES HIT φ OR π?")
    print("-" * 40)
    phi_tolerance = 0.05  # 5%
    pi_tolerance = 0.05

    hits_phi = np.sum(np.abs(ratios_r1 - phi) / phi < phi_tolerance)
    hits_pi = np.sum(np.abs(ratios_r2 - pi) / pi < pi_tolerance)

    print(f"  Random matrices with S[0]/S[1] within 5% of φ: {hits_phi}/{n_trials} ({hits_phi/n_trials*100:.1f}%)")
    print(f"  Random matrices with S[1]/S[2] within 5% of π: {hits_pi}/{n_trials} ({hits_pi/n_trials*100:.1f}%)")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Data: 82×50 SNR matrix from {DATA_PATH.name}")
    print(f"S[0]/S[1] = {r1:.4f} (z={z1:.1f} vs random, {abs(r1-phi)/phi*100:.1f}% from φ)")
    print(f"S[1]/S[2] = {r2:.4f} (z={z2:.1f} vs random, {abs(r2-pi)/pi*100:.1f}% from π)")


if __name__ == "__main__":
    main()
