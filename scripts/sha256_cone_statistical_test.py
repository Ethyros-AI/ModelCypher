#!/usr/bin/env python3
"""Statistical validation of cone search advantage.

Initial results showed:
- Cone_V1: 39% fewer hashes at 16 zeros
- Cone_V2: 37% fewer hashes at 12 zeros

Is this real or noise? Let's run a proper statistical test.
"""

import hashlib
import struct
import numpy as np
from scipy import stats
import time
from collections import defaultdict

# From previous analysis
NONCE_INFLUENCE = {
    0: 574, 1: 549, 2: 555, 3: 556, 4: 552, 5: 553, 6: 549, 7: 595,
    8: 567, 9: 537, 10: 529, 11: 550, 12: 576, 13: 572, 14: 556, 15: 574,
    16: 545, 17: 574, 18: 577, 19: 550, 20: 531, 21: 526, 22: 571, 23: 600,
    24: 548, 25: 538, 26: 504, 27: 525, 28: 499, 29: 570, 30: 568, 31: 540
}

HIGH_INFLUENCE_BITS = sorted(NONCE_INFLUENCE.keys(), key=lambda b: -NONCE_INFLUENCE[b])
LOW_INFLUENCE_BITS = sorted(NONCE_INFLUENCE.keys(), key=lambda b: NONCE_INFLUENCE[b])


def count_leading_zeros(hash_bytes: bytes) -> int:
    n = 0
    for byte in hash_bytes:
        if byte == 0:
            n += 8
        else:
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    return n
                n += 1
    return n


def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def uniform_search(header: bytes, target_zeros: int, max_hashes: int) -> int:
    """Return number of hashes to find target."""
    for nonce in range(max_hashes):
        h = double_sha256(header + struct.pack('<I', nonce))
        if count_leading_zeros(h) >= target_zeros:
            return nonce + 1
    return max_hashes


def cone_v1_search(header: bytes, target_zeros: int, max_hashes: int) -> int:
    """Cone V1: High-influence bits first."""
    hashes = 0
    top_bits = HIGH_INFLUENCE_BITS[:8]
    low_bits = LOW_INFLUENCE_BITS[:8]

    # Phase 1: High-influence exploration
    for hi_combo in range(256):
        base_nonce = 0
        for i, bit in enumerate(top_bits):
            if hi_combo & (1 << i):
                base_nonce |= (1 << bit)

        for lo_combo in range(16):
            if hashes >= max_hashes:
                return hashes

            nonce = base_nonce
            for i, bit in enumerate(low_bits[:4]):
                if lo_combo & (1 << i):
                    nonce |= (1 << bit)

            h = double_sha256(header + struct.pack('<I', nonce))
            hashes += 1

            if count_leading_zeros(h) >= target_zeros:
                return hashes

    # Phase 2: Sequential from best region
    best_base = 0  # Could track best from phase 1
    for offset in range(max_hashes - hashes):
        if hashes >= max_hashes:
            return hashes

        nonce = (best_base + offset) % (2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        hashes += 1

        if count_leading_zeros(h) >= target_zeros:
            return hashes

    return hashes


def run_statistical_test(trials: int, target_zeros: int, max_hashes: int):
    """Run multiple trials and compute statistics."""

    uniform_results = []
    cone_results = []

    print(f"Running {trials} trials at {target_zeros} zeros, max {max_hashes} hashes...")
    print()

    for trial in range(trials):
        header = b"Statistical test block " + struct.pack('>I', trial)

        # Uniform search
        uniform_hashes = uniform_search(header, target_zeros, max_hashes)
        uniform_results.append(uniform_hashes)

        # Cone search
        cone_hashes = cone_v1_search(header, target_zeros, max_hashes)
        cone_results.append(cone_hashes)

        if trial % 10 == 0:
            print(f"  Trial {trial}: uniform={uniform_hashes}, cone={cone_hashes}")

    uniform_results = np.array(uniform_results)
    cone_results = np.array(cone_results)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print(f"Uniform search:")
    print(f"  Mean: {uniform_results.mean():.1f}")
    print(f"  Std:  {uniform_results.std():.1f}")
    print(f"  Median: {np.median(uniform_results):.1f}")
    print()

    print(f"Cone V1 search:")
    print(f"  Mean: {cone_results.mean():.1f}")
    print(f"  Std:  {cone_results.std():.1f}")
    print(f"  Median: {np.median(cone_results):.1f}")
    print()

    # Statistical comparison
    improvement = (uniform_results.mean() - cone_results.mean()) / uniform_results.mean() * 100

    print(f"Improvement: {improvement:.1f}%")
    print()

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(uniform_results, cone_results)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print()

    if p_value < 0.05:
        if t_stat > 0:
            print("*** SIGNIFICANT: Cone search uses FEWER hashes (p < 0.05) ***")
        else:
            print("*** SIGNIFICANT: Uniform search uses FEWER hashes (p < 0.05) ***")
    else:
        print("Not statistically significant (p >= 0.05)")

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(uniform_results, cone_results, alternative='greater')
    print()
    print(f"Mann-Whitney U test (uniform > cone):")
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value: {p_value_mw:.6f}")

    if p_value_mw < 0.05:
        print("*** SIGNIFICANT: Cone search is better (p < 0.05) ***")

    return uniform_results, cone_results


print("SHA-256 CONE SEARCH STATISTICAL VALIDATION")
print("=" * 70)
print()

# Test at different difficulties
print("=" * 70)
print("TEST 1: 8 leading zeros (easy)")
print("=" * 70)
u1, c1 = run_statistical_test(trials=50, target_zeros=8, max_hashes=10000)

print()
print("=" * 70)
print("TEST 2: 12 leading zeros (medium)")
print("=" * 70)
u2, c2 = run_statistical_test(trials=50, target_zeros=12, max_hashes=100000)

print()
print("=" * 70)
print("TEST 3: 16 leading zeros (hard)")
print("=" * 70)
u3, c3 = run_statistical_test(trials=30, target_zeros=16, max_hashes=500000)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("If cone search consistently shows p < 0.05 improvement,")
print("the manifold structure provides exploitable advantage.")
print()
print("The improvement should scale with difficulty if real.")


if __name__ == "__main__":
    pass
