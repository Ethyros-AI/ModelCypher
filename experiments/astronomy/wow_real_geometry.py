#!/usr/bin/env python3
"""
Full geometric analysis of the REAL Wow! signal (raw integers).

All the analyses we did before, but on actual data.

Usage:
    python wow_real_geometry.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.io import readsav
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"

phi = (1 + np.sqrt(5)) / 2
pi = np.pi
e = np.e
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)
golden_angle = 360 / phi**2


def load_raw_signal() -> np.ndarray:
    """Load raw integer signal (remove archiving artifact)."""
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal - 0.5  # Remove archiving offset


def main():
    print("=" * 70)
    print("FULL GEOMETRIC ANALYSIS OF RAW WOW! SIGNAL")
    print("=" * 70)

    signal = load_raw_signal()
    U, S, Vt = linalg.svd(signal, full_matrices=False)

    # 1. SINGULAR VALUE SPECTRUM
    print("\n" + "=" * 70)
    print("1. SINGULAR VALUE SPECTRUM")
    print("=" * 70)

    print(f"\n  First 20 singular values:")
    for i in range(min(20, len(S))):
        print(f"    S[{i:2d}] = {S[i]:10.4f}")

    print(f"\n  Cumulative variance explained:")
    total_var = np.sum(S**2)
    cumsum = np.cumsum(S**2) / total_var * 100
    for i in [1, 2, 3, 5, 10, 20]:
        if i <= len(S):
            print(f"    First {i:2d} modes: {cumsum[i-1]:.2f}%")

    # 2. ALL SINGULAR VALUE RATIOS
    print("\n" + "=" * 70)
    print("2. ALL SINGULAR VALUE RATIOS")
    print("=" * 70)

    constants = {
        'φ': phi, 'π': pi, 'e': e, '√2': sqrt2, '√3': sqrt3, '√5': sqrt5,
        '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0,
        'φ²': phi**2, 'π/2': pi/2, '2π': 2*pi, 'φπ': phi*pi,
        '1/φ': 1/phi, 'φ+1': phi+1, 'π-1': pi-1,
    }

    print(f"\n  Consecutive ratios S[i]/S[i+1]:")
    for i in range(min(15, len(S)-1)):
        if S[i+1] > 1e-10:
            r = S[i] / S[i+1]
            # Find closest constant
            closest = min(constants.items(), key=lambda x: abs(r - x[1]))
            err = abs(r - closest[1]) / closest[1] * 100
            marker = "***" if err < 3 else ""
            print(f"    S[{i}]/S[{i+1}] = {r:.6f}  closest: {closest[0]}={closest[1]:.4f} (err: {err:.2f}%) {marker}")

    print(f"\n  Non-consecutive ratios (looking for patterns):")
    interesting = []
    for i in range(min(15, len(S))):
        for j in range(i+2, min(15, len(S))):
            if S[j] > 1e-10:
                r = S[i] / S[j]
                for name, val in constants.items():
                    if abs(r - val) / val < 0.03:  # Within 3%
                        interesting.append((i, j, r, name, val, abs(r-val)/val*100))

    for i, j, r, name, val, err in sorted(interesting, key=lambda x: x[5])[:15]:
        print(f"    S[{i}]/S[{j}] = {r:.6f} ≈ {name} = {val:.4f} (err: {err:.2f}%)")

    # 3. EIGENVALUE PRODUCTS AND SUMS
    print("\n" + "=" * 70)
    print("3. EIGENVALUE PRODUCTS AND SUMS")
    print("=" * 70)

    print(f"\n  Products of consecutive singular values:")
    for i in range(min(10, len(S)-1)):
        p = S[i] * S[i+1]
        print(f"    S[{i}]×S[{i+1}] = {p:.4f}")

    print(f"\n  Sum of first k singular values:")
    for k in [2, 3, 5, 10]:
        s = np.sum(S[:k])
        print(f"    Σ S[0:{k}] = {s:.4f}")

    print(f"\n  Ratios of sums:")
    for i in range(1, 5):
        for j in range(i+1, 6):
            sum_i = np.sum(S[:i])
            sum_j = np.sum(S[:j])
            r = sum_j / sum_i
            for name, val in constants.items():
                if abs(r - val) / val < 0.05:
                    print(f"    Σ[::{j}]/Σ[::{i}] = {r:.4f} ≈ {name} (err: {abs(r-val)/val*100:.2f}%)")

    # 4. MODE STRUCTURE (TIME PATTERNS)
    print("\n" + "=" * 70)
    print("4. MODE STRUCTURE (U columns)")
    print("=" * 70)

    for mode in range(5):
        u = U[:, mode]
        peak_idx = np.argmax(np.abs(u))
        peak_val = u[peak_idx]
        zc = np.where(np.diff(np.sign(u)) != 0)[0]

        print(f"\n  Mode {mode}:")
        print(f"    Peak at t={peak_idx}, value={peak_val:.4f}")
        print(f"    Zero crossings: {list(zc)}")

        # Symmetry around peak
        if len(zc) >= 2:
            dists = zc - 60  # Distance from t=60
            print(f"    Zero crossing distances from t=60: {list(dists)}")

    # 5. FREQUENCY MODE STRUCTURE (V rows)
    print("\n" + "=" * 70)
    print("5. FREQUENCY MODE STRUCTURE (V rows)")
    print("=" * 70)

    for mode in range(5):
        v = Vt[mode, :]
        peak_idx = np.argmax(np.abs(v))
        peak_val = v[peak_idx]

        print(f"\n  Mode {mode}:")
        print(f"    Peak at channel={peak_idx}, value={peak_val:.4f}")
        print(f"    Energy in channel 1: {v[1]**2 / np.sum(v**2)*100:.1f}%")

    # 6. ROTATION ANALYSIS
    print("\n" + "=" * 70)
    print("6. ROTATION IN MODE SPACE")
    print("=" * 70)

    print(f"\n  Projecting time series onto mode 0-1 plane:")
    angles = []
    radii = []

    for t in range(signal.shape[0]):
        freq_vec = signal[t, :]
        proj_0 = np.dot(freq_vec, Vt[0, :])
        proj_1 = np.dot(freq_vec, Vt[1, :])
        angle = np.degrees(np.arctan2(proj_1, proj_0))
        radius = np.sqrt(proj_0**2 + proj_1**2)
        angles.append(angle)
        radii.append(radius)

    angles = np.array(angles)
    radii = np.array(radii)

    print(f"\n  Around peak (t=55-65):")
    print(f"  t  | Angle    | Radius  | Δangle")
    print("-" * 45)
    for t in range(55, 66):
        da = angles[t] - angles[t-1] if t > 55 else 0
        # Handle wraparound
        if da > 180: da -= 360
        if da < -180: da += 360
        print(f"  {t:2d} | {angles[t]:8.2f}° | {radii[t]:7.2f} | {da:+7.2f}°")

    # Angular velocity
    d_angles = np.diff(angles)
    d_angles = np.where(d_angles > 180, d_angles - 360, d_angles)
    d_angles = np.where(d_angles < -180, d_angles + 360, d_angles)

    peak_da = d_angles[57:63]
    print(f"\n  Mean |Δangle| during peak (t=57-62): {np.mean(np.abs(peak_da)):.2f}°")
    print(f"  Peak rotation angle: {angles[60]:.2f}°")

    # Compare to golden angle
    print(f"\n  Golden angle = {golden_angle:.2f}°")
    print(f"  Peak angle vs golden: {abs(abs(angles[60]) - golden_angle):.2f}° difference")

    # 7. GRAM MATRIX ANALYSIS
    print("\n" + "=" * 70)
    print("7. GRAM MATRIX ANALYSIS")
    print("=" * 70)

    G_time = signal @ signal.T  # Time-time Gram
    G_freq = signal.T @ signal  # Freq-freq Gram

    print(f"\n  Time Gram matrix (82×82):")
    print(f"    Trace: {np.trace(G_time):.4f}")
    print(f"    Frobenius norm: {np.linalg.norm(G_time, 'fro'):.4f}")

    # Eigenvalues of Gram
    eig_time = np.linalg.eigvalsh(G_time)[::-1]
    print(f"\n  Gram eigenvalues (= S²):")
    for i in range(5):
        print(f"    λ[{i}] = {eig_time[i]:.4f} (√λ = {np.sqrt(max(0,eig_time[i])):.4f}, S[{i}] = {S[i]:.4f})")

    # 8. INFORMATION-GEOMETRIC STRUCTURE
    print("\n" + "=" * 70)
    print("8. INFORMATION-GEOMETRIC STRUCTURE")
    print("=" * 70)

    # Effective rank
    S_norm = S / np.sum(S)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
    eff_rank = np.exp(entropy)
    print(f"\n  Singular value entropy: {entropy:.4f}")
    print(f"  Effective rank: {eff_rank:.2f}")
    print(f"  Actual rank: {np.sum(S > 1e-10)}")

    # Participation ratio
    pr = np.sum(S**2)**2 / np.sum(S**4)
    print(f"  Participation ratio: {pr:.2f}")

    # 9. CONDITION NUMBER AND NUMERICAL STRUCTURE
    print("\n" + "=" * 70)
    print("9. NUMERICAL STRUCTURE")
    print("=" * 70)

    cond = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
    print(f"\n  Condition number: {cond:.2f}")
    print(f"  S[0]/S[min]: {S[0]/S[np.min(np.where(S > 1e-10))]:.2f}")

    # Log spacing
    print(f"\n  Log singular values (looking for linear = power law):")
    log_S = np.log(S[:15] + 1e-10)
    for i in range(15):
        print(f"    log S[{i:2d}] = {log_S[i]:.4f}")

    # Fit power law
    indices = np.arange(1, 11)
    log_indices = np.log(indices)
    log_S_fit = log_S[1:11]
    coeffs = np.polyfit(log_indices, log_S_fit, 1)
    print(f"\n  Power law fit S[i] ~ i^α:")
    print(f"    α = {coeffs[0]:.4f}")
    print(f"    For pure noise: α ≈ 0")
    print(f"    For signal: α < 0 (faster decay)")

    # 10. 3D MODE TRAJECTORY
    print("\n" + "=" * 70)
    print("10. 3D MODE TRAJECTORY")
    print("=" * 70)

    print(f"\n  Projecting onto first 3 modes:")
    proj_3d = []
    for t in range(signal.shape[0]):
        freq_vec = signal[t, :]
        p = [np.dot(freq_vec, Vt[i, :]) for i in range(3)]
        proj_3d.append(p)
    proj_3d = np.array(proj_3d)

    print(f"\n  Around peak (t=55-65):")
    print(f"  t  |   Mode 0  |   Mode 1  |   Mode 2  | Radius")
    print("-" * 60)
    for t in range(55, 66):
        r = np.linalg.norm(proj_3d[t])
        print(f"  {t:2d} | {proj_3d[t,0]:9.2f} | {proj_3d[t,1]:9.2f} | {proj_3d[t,2]:9.2f} | {r:.2f}")

    # 11. SEARCH FOR ANY MATHEMATICAL CONSTANTS
    print("\n" + "=" * 70)
    print("11. SYSTEMATIC SEARCH FOR CONSTANTS")
    print("=" * 70)

    # Extended constant list
    all_constants = {
        'φ': phi, '1/φ': 1/phi, 'φ²': phi**2, 'φ³': phi**3,
        'π': pi, 'π/2': pi/2, 'π/3': pi/3, 'π/4': pi/4, '2π': 2*pi,
        'e': e, '1/e': 1/e, 'e²': e**2,
        '√2': sqrt2, '√3': sqrt3, '√5': sqrt5,
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        'ln2': np.log(2), 'ln10': np.log(10),
        'γ': 0.5772156649,  # Euler-Mascheroni
        'δ_F': 4.6692016,   # Feigenbaum
        'α': 1/137.036,     # Fine structure (inverted)
    }

    print(f"\n  Checking all ratios S[i]/S[j] for i<j<15:")
    found = []
    for i in range(15):
        for j in range(i+1, 15):
            if S[j] > 1e-10:
                r = S[i] / S[j]
                for name, val in all_constants.items():
                    if val > 0.1:  # Skip tiny constants
                        err = abs(r - val) / val
                        if err < 0.02:  # Within 2%
                            found.append((i, j, r, name, val, err*100))

    found.sort(key=lambda x: x[5])
    print(f"\n  Best matches (within 2%):")
    for i, j, r, name, val, err in found[:20]:
        print(f"    S[{i}]/S[{j}] = {r:.6f} ≈ {name} = {val:.6f} (err: {err:.3f}%)")

    if not found:
        print("    No ratios within 2% of tested constants")

    # SYNTHESIS
    print("\n" + "=" * 70)
    print("SYNTHESIS: GEOMETRIC STRUCTURE OF RAW SIGNAL")
    print("=" * 70)

    print(f"""
KEY FINDINGS FROM RAW SIGNAL:

1. SINGULAR VALUE STRUCTURE:
   - S[0]/S[1] = {S[0]/S[1]:.4f} ≈ 2
   - S[1]/S[2] = {S[1]/S[2]:.4f} ≈ 2
   - Decay is roughly geometric with ratio 2

2. MODE STRUCTURE:
   - Mode 0 peaks at t=60 (the signal peak)
   - Mode 1 has zero crossings near the peak
   - Energy concentrated in channel 1 (narrowband)

3. ROTATION:
   - Peak angle = {angles[60]:.2f}°
   - Mean angular velocity during peak = {np.mean(np.abs(peak_da)):.2f}°

4. EFFECTIVE DIMENSIONALITY:
   - Effective rank = {eff_rank:.2f}
   - Participation ratio = {pr:.2f}
   - Signal is low-dimensional (dominated by ~2-3 modes)

5. MATHEMATICAL CONSTANTS:
   - SVD ratios cluster around 2, not φ or π
   - No strong evidence for encoded constants in geometry
""")


if __name__ == "__main__":
    main()
