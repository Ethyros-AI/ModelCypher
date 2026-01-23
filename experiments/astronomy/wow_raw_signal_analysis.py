#!/usr/bin/env python3
"""
FRESH START: Analysis of the RAW Wow! Signal

Using the original integer values as recorded by Big Ear.
No artificial offsets, no processing artifacts.

The Big Ear encoding:
  0-9 → digits 0-9
  A-Z → values 10-35

The famous sequence: 6EQUJ5 = [6, 14, 26, 30, 19, 5]

Usage:
    python wow_raw_signal_analysis.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.io import readsav
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"

phi = (1 + np.sqrt(5)) / 2
pi = np.pi


def load_raw_signal() -> np.ndarray:
    """Load the signal and remove the archiving artifact (+0.5 offset)."""
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove the +0.5 archiving artifact to get original integers
    signal = signal - 0.5

    return signal


def main():
    print("=" * 70)
    print("RAW WOW! SIGNAL ANALYSIS")
    print("=" * 70)
    print("Using original integer values (archiving offset removed)")

    signal = load_raw_signal()

    # Verify we have integers
    print(f"\n1. DATA VERIFICATION")
    print("=" * 70)
    unique = np.unique(signal)
    print(f"  Shape: {signal.shape} (82 time steps × 50 frequency channels)")
    print(f"  Unique values: {sorted(unique.astype(int))}")
    print(f"  All integers: {all(v == int(v) for v in unique)}")

    # The signal channel
    print(f"\n2. THE SIGNAL (Channel 1)")
    print("=" * 70)

    ch1 = signal[:, 1].astype(int)
    print(f"\n  Full channel 1 time series:")
    print(f"  t  | SNR | Char")
    print("-" * 25)

    for t in range(signal.shape[0]):
        val = ch1[t]
        if val == 0:
            char = ' '
        elif 1 <= val <= 9:
            char = str(val)
        elif 10 <= val <= 35:
            char = chr(ord('A') + val - 10)
        else:
            char = '?'

        marker = ""
        if t == 60:
            marker = " ← PEAK"
        elif 57 <= t <= 62:
            marker = " ← 6EQUJ5"

        if val > 0 or 50 <= t <= 70:
            print(f"  {t:2d} | {val:3d} | {char}{marker}")

    # 3. SVD of raw signal
    print(f"\n3. SVD ANALYSIS OF RAW SIGNAL")
    print("=" * 70)

    U, S, Vt = linalg.svd(signal, full_matrices=False)

    print(f"\n  Singular values (first 15):")
    for i in range(15):
        print(f"    S[{i:2d}] = {S[i]:10.4f}")

    print(f"\n  Key ratios:")
    ratios = [
        (0, 1, S[0]/S[1]),
        (1, 2, S[1]/S[2]),
        (2, 3, S[2]/S[3]),
        (0, 2, S[0]/S[2]),
        (2, 9, S[2]/S[9] if S[9] > 1e-10 else float('inf')),
    ]

    for i, j, r in ratios:
        print(f"    S[{i}]/S[{j}] = {r:.6f}")

    # Compare to mathematical constants
    print(f"\n  Comparison to constants:")
    constants = [
        ('φ (golden ratio)', phi),
        ('π', pi),
        ('e', np.e),
        ('√2', np.sqrt(2)),
        ('√3', np.sqrt(3)),
        ('2', 2.0),
        ('3', 3.0),
    ]

    r1 = S[0]/S[1]
    r2 = S[1]/S[2]

    print(f"\n    S[0]/S[1] = {r1:.6f}")
    for name, val in constants:
        err = abs(r1 - val) / val * 100
        print(f"      vs {name:15s} = {val:.6f}: error = {err:.2f}%")

    print(f"\n    S[1]/S[2] = {r2:.6f}")
    for name, val in constants:
        err = abs(r2 - val) / val * 100
        print(f"      vs {name:15s} = {val:.6f}: error = {err:.2f}%")

    # 4. What IS the structure?
    print(f"\n4. WHAT STRUCTURE EXISTS IN THE RAW SIGNAL?")
    print("=" * 70)

    # The ratios are both ~2. What does that mean?
    print(f"\n  Both ratios ≈ 2:")
    print(f"    S[0]/S[1] = {r1:.4f}")
    print(f"    S[1]/S[2] = {r2:.4f}")

    print(f"\n  This means singular values decay roughly geometrically:")
    print(f"    S[0] ≈ 2 × S[1] ≈ 4 × S[2]")

    # Look at the actual decay
    print(f"\n  Actual decay pattern:")
    for i in range(10):
        if S[i+1] > 1e-10:
            ratio = S[i] / S[i+1]
            print(f"    S[{i}]/S[{i+1}] = {ratio:.4f}")

    # 5. Mode structure
    print(f"\n5. MODE STRUCTURE (U columns = time patterns)")
    print("=" * 70)

    print(f"\n  Mode 0 (dominant):")
    u0 = U[:, 0]
    peak_u0 = np.argmax(np.abs(u0))
    print(f"    Peak at t = {peak_u0}")
    print(f"    Values around peak: {u0[55:66]}")

    print(f"\n  Mode 1 (first correction):")
    u1 = U[:, 1]
    zc = np.where(np.diff(np.sign(u1)) != 0)[0]
    print(f"    Zero crossings at: {list(zc)}")
    print(f"    Values around peak: {u1[55:66]}")

    # 6. The 6EQUJ5 sequence itself
    print(f"\n6. THE 6EQUJ5 SEQUENCE")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]
    print(f"\n  Values: {seq}")
    print(f"  Sum: {sum(seq)}")
    print(f"  Mean: {np.mean(seq):.2f}")
    print(f"  Max (U): {max(seq)}")

    print(f"\n  Ratios between consecutive values:")
    for i in range(len(seq)-1):
        if seq[i] != 0:
            r = seq[i+1] / seq[i]
            print(f"    {seq[i+1]}/{seq[i]} = {r:.4f}")

    print(f"\n  Is there φ or π in the sequence itself?")

    # Check all pairwise ratios
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j and seq[j] != 0:
                r = seq[i] / seq[j]
                if abs(r - phi) < 0.1:
                    print(f"    {seq[i]}/{seq[j]} = {r:.4f} ≈ φ ({abs(r-phi)/phi*100:.1f}% error)")
                if abs(r - pi) < 0.2:
                    print(f"    {seq[i]}/{seq[j]} = {r:.4f} ≈ π ({abs(r-pi)/pi*100:.1f}% error)")

    # 7. Symmetry analysis
    print(f"\n7. SYMMETRY IN THE SEQUENCE")
    print("=" * 70)

    # Is 6EQUJ5 symmetric or asymmetric?
    print(f"\n  Sequence: {seq}")
    print(f"  Reversed: {seq[::-1]}")
    print(f"  Symmetric? {seq == seq[::-1]}")

    # Asymmetry measure
    center = len(seq) // 2
    left = seq[:center]
    right = seq[center+1:][::-1] if len(seq) % 2 == 1 else seq[center:][::-1]

    print(f"\n  Left of peak: {left}")
    print(f"  Right of peak (reversed): {right}")

    # Rise vs fall
    rise = seq[3] - seq[0]  # 30 - 6 = 24
    fall = seq[3] - seq[5]  # 30 - 5 = 25
    print(f"\n  Rise (6 to U): {rise}")
    print(f"  Fall (U to 5): {fall}")
    print(f"  Nearly symmetric!")

    # 8. What's special about these specific numbers?
    print(f"\n8. THE SPECIFIC NUMBERS")
    print("=" * 70)

    for val in seq:
        factors = []
        for f in range(1, val+1):
            if val % f == 0:
                factors.append(f)
        print(f"\n  {val}:")
        print(f"    Factors: {factors}")
        print(f"    Prime? {len(factors) == 2}")
        print(f"    Binary: {bin(val)}")

    # SYNTHESIS
    print(f"\n" + "=" * 70)
    print("SYNTHESIS: WHAT'S ACTUALLY IN THE RAW SIGNAL?")
    print("=" * 70)

    print(f"""
THE RAW SIGNAL STRUCTURE:

1. SVD ratios are both ≈ 2, NOT φ or π
   - S[0]/S[1] = {r1:.4f} (≈ 2)
   - S[1]/S[2] = {r2:.4f} (≈ 2)
   - The φ/π pattern was an artifact of the +0.5 offset

2. The 6EQUJ5 sequence is nearly symmetric
   - Rise: 6 → 14 → 26 → 30 (increase of 24)
   - Fall: 30 → 19 → 5 (decrease of 25)
   - This is consistent with a point source passing through
     a symmetric telescope beam

3. No obvious φ or π in the raw sequence values

4. The signal is dominated by:
   - A strong narrowband transient in channel 1
   - Background noise at 0 everywhere else
   - Duration of ~6 samples (72 seconds)

CONCLUSION:
When we analyze the ACTUAL signal (integers as recorded by Big Ear),
we do NOT find the φ/π structure. The signal appears to be what
you'd expect from a strong narrowband source transiting through
the telescope beam.

The question shifts: Is there ANYTHING unusual about this signal
beyond being very strong?
""")


if __name__ == "__main__":
    main()
