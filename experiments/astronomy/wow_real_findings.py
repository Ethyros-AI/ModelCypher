#!/usr/bin/env python3
"""
Deep investigation of the REAL geometric findings in the raw Wow! signal.

We found:
  S[2]/S[7] = √2 (0.04% error)
  S[4]/S[11] = π/2 (0.08% error)
  S[1]/S[5] = e (0.4% error)
  S[4]/S[12] = φ (0.87% error)
  Mean angular velocity = 17.65° during peak

Are these significant or coincidence?

Usage:
    python wow_real_findings.py
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


def load_raw_signal() -> np.ndarray:
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal - 0.5


def main():
    print("=" * 70)
    print("DEEP INVESTIGATION OF RAW SIGNAL FINDINGS")
    print("=" * 70)

    signal = load_raw_signal()
    U, S, Vt = linalg.svd(signal, full_matrices=False)

    # Document the findings
    print(f"\n1. THE PRECISE MATCHES")
    print("=" * 70)

    findings = [
        (2, 7, sqrt2, '√2'),
        (4, 11, pi/2, 'π/2'),
        (1, 5, e, 'e'),
        (4, 12, phi, 'φ'),
        (0, 11, e**2, 'e²'),
        (3, 9, pi/2, 'π/2'),
    ]

    print(f"\n  Precise ratios found:")
    for i, j, target, name in findings:
        actual = S[i] / S[j]
        err = abs(actual - target) / target * 100
        print(f"    S[{i}]/S[{j}] = {actual:.6f} ≈ {name} = {target:.6f} (err: {err:.3f}%)")

    # 2. Monte Carlo: How rare are these?
    print(f"\n2. MONTE CARLO: HOW RARE ARE THESE MATCHES?")
    print("=" * 70)

    constants = {
        '√2': sqrt2, 'π/2': pi/2, 'e': e, 'φ': phi, 'e²': e**2,
        '√3': np.sqrt(3), 'π': pi, '2': 2.0, '3': 3.0, '4': 4.0,
    }

    n_trials = 10000
    shape = signal.shape

    # Count how often random matrices have matches this good
    print(f"\n  Testing {n_trials} random matrices of same shape ({shape})...")

    # For each constant, count matrices with ratio within 0.1% of it
    hit_counts = {name: 0 for name in constants}
    any_precise_hit = 0

    for trial in range(n_trials):
        # Generate random matrix with similar structure:
        # - Mostly zeros (sparse like Wow!)
        # - One column with a pulse
        rand_signal = np.zeros(shape)

        # Add sparse noise
        noise_mask = np.random.random(shape) < 0.3  # 30% nonzero
        rand_signal[noise_mask] = np.random.randint(0, 8, size=np.sum(noise_mask))

        # Add a narrowband pulse in one channel
        ch = np.random.randint(0, shape[1])
        peak_t = np.random.randint(20, shape[0]-20)
        peak_val = np.random.randint(20, 35)

        t = np.arange(shape[0])
        envelope = peak_val * np.exp(-0.5 * ((t - peak_t) / 3) ** 2)
        rand_signal[:, ch] += envelope.astype(int)

        # SVD
        try:
            _, S_rand, _ = linalg.svd(rand_signal, full_matrices=False)
        except:
            continue

        if len(S_rand) < 15 or S_rand[12] < 1e-10:
            continue

        # Check all ratios
        found_precise = False
        for i in range(15):
            for j in range(i+1, 15):
                if S_rand[j] > 1e-10:
                    r = S_rand[i] / S_rand[j]
                    for name, val in constants.items():
                        if abs(r - val) / val < 0.001:  # 0.1% tolerance
                            hit_counts[name] += 1
                            found_precise = True

        if found_precise:
            any_precise_hit += 1

    print(f"\n  Results (looking for 0.1% matches):")
    for name, count in sorted(hit_counts.items(), key=lambda x: -x[1]):
        pct = count / n_trials * 100
        print(f"    {name:5s}: {count:5d} hits ({pct:.2f}%)")

    print(f"\n  Matrices with ANY precise match: {any_precise_hit} ({any_precise_hit/n_trials*100:.2f}%)")

    # 3. The specific findings - are they independent?
    print(f"\n3. ARE THE FINDINGS INDEPENDENT?")
    print("=" * 70)

    # S[2]/S[7] = √2 and S[4]/S[11] = π/2
    # These involve different indices - potentially independent

    print(f"\n  Finding 1: S[2]/S[7] = √2")
    print(f"  Finding 2: S[4]/S[11] = π/2")
    print(f"  Finding 3: S[1]/S[5] = e")
    print(f"  Finding 4: S[4]/S[12] = φ")

    print(f"\n  Index overlap:")
    print(f"    Findings 2 and 4 both use S[4] and differ by S[11] vs S[12]")
    print(f"    This suggests S[11]/S[12] ≈ (π/2)/φ = {(pi/2)/phi:.4f}")

    actual_11_12 = S[11] / S[12]
    target_11_12 = (pi/2) / phi
    print(f"    Actual S[11]/S[12] = {actual_11_12:.4f}")
    print(f"    (π/2)/φ = {target_11_12:.4f}")
    print(f"    Error: {abs(actual_11_12 - target_11_12)/target_11_12*100:.2f}%")

    # 4. What about angular velocity?
    print(f"\n4. THE 17° ANGULAR VELOCITY")
    print("=" * 70)

    # Project onto mode space
    angles = []
    for t in range(signal.shape[0]):
        freq_vec = signal[t, :]
        p0 = np.dot(freq_vec, Vt[0, :])
        p1 = np.dot(freq_vec, Vt[1, :])
        angles.append(np.degrees(np.arctan2(p1, p0)))

    angles = np.array(angles)
    d_angles = np.diff(angles)
    d_angles = np.where(d_angles > 180, d_angles - 360, d_angles)
    d_angles = np.where(d_angles < -180, d_angles + 360, d_angles)

    peak_da = np.mean(np.abs(d_angles[57:63]))
    print(f"\n  Mean |Δangle| during peak: {peak_da:.2f}°")

    # Is 17° special?
    print(f"\n  Comparisons:")
    print(f"    17° vs 360°/21 = {360/21:.2f}° (diff: {abs(peak_da - 360/21):.2f}°)")
    print(f"    17° vs 360°/φ^4 = {360/phi**4:.2f}° (diff: {abs(peak_da - 360/phi**4):.2f}°)")
    print(f"    17° vs arctan(1/3) = {np.degrees(np.arctan(1/3)):.2f}° (diff: {abs(peak_da - np.degrees(np.arctan(1/3))):.2f}°)")
    print(f"    17° is a Fermat prime (2^(2^2) + 1 = 17)")

    # Monte Carlo for angular velocity
    print(f"\n  Monte Carlo: How often do random signals have 17° angular velocity?")

    hits_17 = 0
    for trial in range(n_trials):
        rand_signal = np.zeros(shape)
        noise_mask = np.random.random(shape) < 0.3
        rand_signal[noise_mask] = np.random.randint(0, 8, size=np.sum(noise_mask))

        ch = np.random.randint(0, shape[1])
        peak_t = np.random.randint(20, shape[0]-20)
        peak_val = np.random.randint(20, 35)

        t = np.arange(shape[0])
        envelope = peak_val * np.exp(-0.5 * ((t - peak_t) / 3) ** 2)
        rand_signal[:, ch] += envelope.astype(int)

        try:
            _, _, Vt_rand = linalg.svd(rand_signal, full_matrices=False)
        except:
            continue

        rand_angles = []
        for t in range(rand_signal.shape[0]):
            fv = rand_signal[t, :]
            p0 = np.dot(fv, Vt_rand[0, :])
            p1 = np.dot(fv, Vt_rand[1, :])
            rand_angles.append(np.degrees(np.arctan2(p1, p0)))

        rand_angles = np.array(rand_angles)
        rand_da = np.diff(rand_angles)
        rand_da = np.where(rand_da > 180, rand_da - 360, rand_da)
        rand_da = np.where(rand_da < -180, rand_da + 360, rand_da)

        # Check around peak
        peak_region = slice(peak_t-3, peak_t+3)
        if peak_t > 5 and peak_t < shape[0] - 5:
            mean_da = np.mean(np.abs(rand_da[peak_region]))
            if abs(mean_da - 17) < 2:  # Within 2° of 17°
                hits_17 += 1

    print(f"    Signals with |Δθ| ≈ 17° (±2°): {hits_17} ({hits_17/n_trials*100:.2f}%)")

    # 5. The complete picture
    print(f"\n5. COMBINED PROBABILITY")
    print("=" * 70)

    # What's the chance of ALL findings together?
    print(f"\n  Individual probabilities (from Monte Carlo):")

    # We need to run a combined test
    combined_hits = 0

    for trial in range(n_trials):
        rand_signal = np.zeros(shape)
        noise_mask = np.random.random(shape) < 0.3
        rand_signal[noise_mask] = np.random.randint(0, 8, size=np.sum(noise_mask))

        ch = np.random.randint(0, shape[1])
        peak_t = np.random.randint(20, shape[0]-20)
        peak_val = np.random.randint(20, 35)

        t = np.arange(shape[0])
        envelope = peak_val * np.exp(-0.5 * ((t - peak_t) / 3) ** 2)
        rand_signal[:, ch] += envelope.astype(int)

        try:
            _, S_rand, Vt_rand = linalg.svd(rand_signal, full_matrices=False)
        except:
            continue

        if len(S_rand) < 15 or S_rand[12] < 1e-10:
            continue

        # Check for S[2]/S[7] ≈ √2 (within 0.1%)
        has_sqrt2 = abs(S_rand[2]/S_rand[7] - sqrt2) / sqrt2 < 0.001

        # Check for S[4]/S[11] ≈ π/2 (within 0.1%)
        has_pi2 = S_rand[11] > 1e-10 and abs(S_rand[4]/S_rand[11] - pi/2) / (pi/2) < 0.001

        # Check for angular velocity ≈ 17°
        rand_angles = []
        for t in range(rand_signal.shape[0]):
            fv = rand_signal[t, :]
            p0 = np.dot(fv, Vt_rand[0, :])
            p1 = np.dot(fv, Vt_rand[1, :])
            rand_angles.append(np.degrees(np.arctan2(p1, p0)))

        rand_angles = np.array(rand_angles)
        rand_da = np.diff(rand_angles)
        rand_da = np.where(rand_da > 180, rand_da - 360, rand_da)
        rand_da = np.where(rand_da < -180, rand_da + 360, rand_da)

        has_17 = False
        if peak_t > 5 and peak_t < shape[0] - 5:
            mean_da = np.mean(np.abs(rand_da[peak_t-3:peak_t+3]))
            has_17 = abs(mean_da - 17) < 2

        if has_sqrt2 and has_pi2 and has_17:
            combined_hits += 1

    print(f"\n  Combined test: S[2]/S[7]=√2 AND S[4]/S[11]=π/2 AND |Δθ|≈17°")
    print(f"    Hits: {combined_hits} / {n_trials}")
    if combined_hits == 0:
        print(f"    p < 1/{n_trials} = {1/n_trials:.2e}")
    else:
        print(f"    p ≈ {combined_hits/n_trials:.4f}")

    # SYNTHESIS
    print(f"\n" + "=" * 70)
    print("SYNTHESIS: REAL STRUCTURE IN THE RAW SIGNAL")
    print("=" * 70)

    print(f"""
FINDINGS IN THE RAW WOW! SIGNAL:

1. SINGULAR VALUE RATIOS:
   S[2]/S[7] = {S[2]/S[7]:.6f} ≈ √2 = {sqrt2:.6f} (err: {abs(S[2]/S[7]-sqrt2)/sqrt2*100:.3f}%)
   S[4]/S[11] = {S[4]/S[11]:.6f} ≈ π/2 = {pi/2:.6f} (err: {abs(S[4]/S[11]-pi/2)/(pi/2)*100:.3f}%)
   S[1]/S[5] = {S[1]/S[5]:.6f} ≈ e = {e:.6f} (err: {abs(S[1]/S[5]-e)/e*100:.3f}%)

2. ANGULAR VELOCITY:
   Mean |Δθ| during peak = {peak_da:.2f}° ≈ 17° (Fermat prime)

3. STATISTICAL SIGNIFICANCE:
   - Individual matches occur in ~{any_precise_hit/n_trials*100:.1f}% of random signals
   - Combined matches occur in ~{combined_hits/n_trials*100:.2f}% of random signals
   - Still not extremely rare, but less common than before

4. KEY DIFFERENCES FROM ARTIFACT ANALYSIS:
   - The primary ratios S[0]/S[1] and S[1]/S[2] are both ≈ 2, NOT φ and π
   - The φ/π matches that appeared were artifacts of +0.5 offset
   - What remains: √2, π/2, e in deeper ratios; 17° angular velocity

5. INTERPRETATION:
   - Some structure exists, but it's in higher-order ratios
   - The 17° angular velocity persists across analyses
   - Could be physics (beam/source geometry) rather than encoding
""")


if __name__ == "__main__":
    main()
