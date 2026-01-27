#!/usr/bin/env python3
"""Find trajectory outliers and analyze their output properties.

The entropy curve shows:
- Distance saturates to π/e ≈ 1.12
- Growth rate ≈ √2

Trajectories that deviate from this curve are GEOMETRICALLY UNUSUAL.
Do they also produce STATISTICALLY UNUSUAL outputs?

If so, we can constrain the search space by filtering on trajectory geometry.
"""

import sys
from pathlib import Path
import struct
import hashlib
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PI_OVER_E = math.pi / math.e
SQRT2 = math.sqrt(2)

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

H_INIT = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def get_trajectory_and_hash(message: bytes, check_rounds: list[int] = [6, 10, 16]):
    """Get state at specific rounds and final hash."""
    msg_len = len(message)
    padded = message + b'\x80'
    padded = padded + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
    padded = padded + struct.pack('>Q', msg_len * 8)
    chunk = padded[:64]

    w = list(struct.unpack('>16I', chunk))
    for i in range(16, 64):
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

    a, b, c, d, e, f, g, h = H_INIT
    states = {}

    for r in range(64):
        if r in check_rounds:
            states[r] = np.array([a, b, c, d, e, f, g, h], dtype=np.float64) / 0xFFFFFFFF

        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        temp1 = (h + S1 + ch + K[r] + w[r]) & 0xFFFFFFFF
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xFFFFFFFF

        h, g, f = g, f, e
        e = (d + temp1) & 0xFFFFFFFF
        d, c, b = c, b, a
        a = (temp1 + temp2) & 0xFFFFFFFF

    # Final hash
    final_hash = hashlib.sha256(message).digest()

    return states, final_hash


def count_leading_zeros(hash_bytes: bytes) -> int:
    """Count leading zero bits in hash."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            # Count leading zeros in this byte
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    break
                count += 1
            break
    return count


def main():
    np.random.seed(42)

    print("Trajectory Outlier Analysis")
    print("=" * 70)
    print("Looking for correlation between trajectory geometry and output properties.")
    print()

    header = b"outlier_test"
    n_samples = 10000
    check_rounds = [6, 10, 16]

    # Compute reference state (from H_INIT, for measuring distance)
    h_init_norm = np.array(H_INIT, dtype=np.float64) / 0xFFFFFFFF

    # Collect trajectories and outputs
    print(f"Analyzing {n_samples} trajectories...")

    all_data = []
    for i in range(n_samples):
        nonce = i.to_bytes(4, 'big') + np.random.bytes(28)
        message = header + nonce

        states, final_hash = get_trajectory_and_hash(message, check_rounds)
        leading_zeros = count_leading_zeros(final_hash)

        # Compute trajectory metrics
        dist_6 = np.linalg.norm(states[6] - h_init_norm) if 6 in states else 0
        dist_10 = np.linalg.norm(states[10] - h_init_norm) if 10 in states else 0
        dist_16 = np.linalg.norm(states[16] - h_init_norm) if 16 in states else 0

        # Deviation from expected saturation (π/e)
        dev_from_pi_e = abs(dist_10 - PI_OVER_E)

        all_data.append({
            'nonce': nonce,
            'leading_zeros': leading_zeros,
            'dist_6': dist_6,
            'dist_10': dist_10,
            'dist_16': dist_16,
            'dev_from_pi_e': dev_from_pi_e,
            'hash': final_hash.hex()[:16],
        })

    # Analyze correlation between trajectory and output
    leading_zeros = np.array([d['leading_zeros'] for d in all_data])
    dist_10 = np.array([d['dist_10'] for d in all_data])
    dev_from_pi_e = np.array([d['dev_from_pi_e'] for d in all_data])

    # Correlation coefficients
    corr_dist_zeros = np.corrcoef(dist_10, leading_zeros)[0, 1]
    corr_dev_zeros = np.corrcoef(dev_from_pi_e, leading_zeros)[0, 1]

    print()
    print("-" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)
    print(f"Correlation(distance_at_round_10, leading_zeros): {corr_dist_zeros:.6f}")
    print(f"Correlation(deviation_from_π/e, leading_zeros):   {corr_dev_zeros:.6f}")

    # Compare statistics for high-zero vs low-zero outputs
    print()
    print("-" * 70)
    print("TRAJECTORY STATISTICS BY OUTPUT QUALITY")
    print("-" * 70)

    # Define "good" outputs as having more leading zeros than median
    median_zeros = np.median(leading_zeros)
    good_mask = leading_zeros > median_zeros
    bad_mask = ~good_mask

    print(f"Median leading zeros: {median_zeros}")
    print(f"Good outputs (>{median_zeros} zeros): {np.sum(good_mask)}")
    print(f"Bad outputs (≤{median_zeros} zeros):  {np.sum(bad_mask)}")
    print()

    print(f"{'Metric':<25} {'Good outputs':<15} {'Bad outputs':<15} {'Difference':<15}")
    print("-" * 70)

    for metric in ['dist_6', 'dist_10', 'dist_16', 'dev_from_pi_e']:
        vals = np.array([d[metric] for d in all_data])
        good_mean = np.mean(vals[good_mask])
        bad_mean = np.mean(vals[bad_mask])
        diff = good_mean - bad_mean
        print(f"{metric:<25} {good_mean:<15.6f} {bad_mean:<15.6f} {diff:<15.6f}")

    # Look at extreme outliers
    print()
    print("-" * 70)
    print("EXTREME TRAJECTORY OUTLIERS")
    print("-" * 70)

    # Sort by deviation from π/e
    sorted_by_dev = sorted(all_data, key=lambda x: x['dev_from_pi_e'], reverse=True)

    print("Top 10 trajectories with HIGHEST deviation from π/e:")
    print(f"{'Rank':<6} {'Dev from π/e':<14} {'Leading zeros':<14} {'Distance@10':<14}")
    for i, d in enumerate(sorted_by_dev[:10]):
        print(f"{i+1:<6} {d['dev_from_pi_e']:<14.6f} {d['leading_zeros']:<14} {d['dist_10']:<14.6f}")

    print()
    print("Top 10 trajectories with LOWEST deviation from π/e:")
    for i, d in enumerate(sorted_by_dev[-10:]):
        print(f"{i+1:<6} {d['dev_from_pi_e']:<14.6f} {d['leading_zeros']:<14} {d['dist_10']:<14.6f}")

    # The key question: do outlier trajectories produce outlier outputs?
    print()
    print("-" * 70)
    print("KEY QUESTION: Do trajectory outliers produce output outliers?")
    print("-" * 70)

    # Top 10% by trajectory deviation
    n_outliers = n_samples // 10
    outlier_trajectories = sorted_by_dev[:n_outliers]
    normal_trajectories = sorted_by_dev[n_outliers:]

    outlier_zeros = np.mean([d['leading_zeros'] for d in outlier_trajectories])
    normal_zeros = np.mean([d['leading_zeros'] for d in normal_trajectories])

    print(f"Mean leading zeros for trajectory outliers (top 10%): {outlier_zeros:.4f}")
    print(f"Mean leading zeros for normal trajectories:          {normal_zeros:.4f}")

    if abs(outlier_zeros - normal_zeros) > 0.1:
        print("\n*** POTENTIAL SIGNAL: Trajectory outliers have different output statistics!")
    else:
        print("\nNo significant difference detected.")


if __name__ == "__main__":
    main()
