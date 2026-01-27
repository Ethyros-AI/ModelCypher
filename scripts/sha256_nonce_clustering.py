#!/usr/bin/env python3
"""SHA-256 Nonce Clustering Analysis.

Question: Do valid nonces (those producing low hashes) CLUSTER?

If they do, then once we find one valid nonce, nearby nonces
might also be valid - narrowing the search cone.

If they don't, SHA-256 is truly random and no cone narrowing is possible.
"""

import hashlib
import struct
import numpy as np
from scipy import stats
from collections import defaultdict
import time

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


def find_valid_nonces(header: bytes, target_zeros: int, max_nonces: int, random_sample: bool = True) -> list:
    """Find nonces producing at least target_zeros leading zeros.

    If random_sample=True, sample randomly from full 32-bit space.
    Otherwise, search sequentially from 0.
    """
    valid = []

    if random_sample:
        # Random sample from full 32-bit nonce space
        tested_nonces = set()
        while len(tested_nonces) < max_nonces:
            nonce = np.random.randint(0, 2**32, dtype=np.uint32)
            if nonce in tested_nonces:
                continue
            tested_nonces.add(nonce)

            h = double_sha256(header + struct.pack('<I', int(nonce)))
            zeros = count_leading_zeros(h)
            if zeros >= target_zeros:
                valid.append((int(nonce), zeros))
    else:
        for nonce in range(max_nonces):
            h = double_sha256(header + struct.pack('<I', nonce))
            zeros = count_leading_zeros(h)
            if zeros >= target_zeros:
                valid.append((nonce, zeros))

    return valid


def analyze_clustering(valid_nonces: list, max_search: int):
    """Analyze if valid nonces cluster together."""
    if len(valid_nonces) < 2:
        return None

    nonces = sorted([n for n, z in valid_nonces])

    # Compute gaps between consecutive valid nonces
    gaps = np.diff(nonces)

    # For uniform random distribution, gaps should be exponential
    expected_mean_gap = max_search / len(nonces)

    print(f"Valid nonces found: {len(nonces)}")
    print(f"Search space: {max_search}")
    print(f"Expected mean gap: {expected_mean_gap:.1f}")
    print(f"Actual mean gap: {gaps.mean():.1f}")
    print(f"Gap std: {gaps.std():.1f}")
    print(f"Min gap: {gaps.min()}")
    print(f"Max gap: {gaps.max()}")
    print()

    # Test if gaps follow exponential distribution (random)
    normalized_gaps = gaps / gaps.mean()
    _, p_exponential = stats.kstest(normalized_gaps, 'expon')

    print(f"Kolmogorov-Smirnov test vs exponential:")
    print(f"  p-value: {p_exponential:.6f}")

    if p_exponential < 0.05:
        print("  *** REJECTS exponential (p < 0.05) - gaps are NOT random! ***")
    else:
        print("  Cannot reject exponential - gaps appear random")

    print()

    # Check for clustering (gaps smaller than expected)
    small_gap_threshold = expected_mean_gap / 10
    cluster_count = sum(1 for g in gaps if g < small_gap_threshold)
    expected_clusters = len(gaps) * (1 - np.exp(-0.1))  # For exponential dist

    print(f"Gaps < {small_gap_threshold:.0f} (clustering indicator):")
    print(f"  Found: {cluster_count}")
    print(f"  Expected (random): {expected_clusters:.1f}")

    if cluster_count > expected_clusters * 1.5:
        print("  *** MORE clustering than random! ***")
    elif cluster_count < expected_clusters * 0.5:
        print("  *** LESS clustering than random! ***")
    else:
        print("  Consistent with random")

    return gaps


def analyze_neighborhood(header: bytes, valid_nonces: list, window: int = 1000):
    """
    For each valid nonce, check if neighbors are more likely to be valid.
    """
    print(f"Analyzing {window}-nonce neighborhood of each valid nonce...")
    print()

    total_neighbors_checked = 0
    valid_neighbors = 0

    for nonce, zeros in valid_nonces[:100]:  # Sample first 100
        for offset in range(-window//2, window//2):
            if offset == 0:
                continue
            neighbor = nonce + offset
            if neighbor < 0 or neighbor >= 2**32:
                continue

            h = double_sha256(header + struct.pack('<I', neighbor))
            neighbor_zeros = count_leading_zeros(h)

            total_neighbors_checked += 1
            if neighbor_zeros >= zeros:
                valid_neighbors += 1

    expected_valid = total_neighbors_checked * (1 / 2**zeros)

    print(f"Neighbors checked: {total_neighbors_checked}")
    print(f"Valid neighbors found: {valid_neighbors}")
    print(f"Expected (random): {expected_valid:.2f}")
    print()

    if valid_neighbors > expected_valid * 1.5:
        print("*** NEIGHBORS ARE MORE VALID THAN RANDOM! ***")
        print("This indicates clustering - cone narrowing might work!")
    else:
        print("Neighbors are not more valid than random.")
        print("No exploitable clustering.")

    return valid_neighbors, expected_valid


def analyze_bit_patterns(valid_nonces: list):
    """Check if valid nonces share common bit patterns."""
    if len(valid_nonces) < 10:
        return

    print("Analyzing bit patterns in valid nonces...")
    print()

    nonces = [n for n, z in valid_nonces]

    # For each bit position, count how often it's set in valid nonces
    bit_counts = np.zeros(32)
    for nonce in nonces:
        for bit in range(32):
            if nonce & (1 << bit):
                bit_counts[bit] += 1

    bit_fractions = bit_counts / len(nonces)

    print("Bit frequencies in valid nonces (expected: 0.5 for random):")
    deviations = []
    for bit in range(32):
        deviation = abs(bit_fractions[bit] - 0.5)
        deviations.append((bit, bit_fractions[bit], deviation))

    deviations.sort(key=lambda x: -x[2])

    print("Most deviant bits:")
    for bit, frac, dev in deviations[:10]:
        expected_dev = 1 / (2 * np.sqrt(len(nonces)))  # ~std for binomial
        significance = dev / expected_dev
        print(f"  Bit {bit:2d}: freq={frac:.3f}, deviation={dev:.3f}, σ={significance:.1f}")

    print()

    # Test each bit individually for bias from 0.5
    n = len(nonces)
    biased_bits = 0
    for bit in range(32):
        # Binomial test: is this bit's frequency significantly different from 0.5?
        successes = int(bit_counts[bit])
        p_value = stats.binom_test(successes, n, 0.5, alternative='two-sided') if hasattr(stats, 'binom_test') else 1.0
        if p_value < 0.05:
            biased_bits += 1

    print(f"Bits significantly biased from 0.5 (p < 0.05): {biased_bits} / 32")

    if biased_bits > 2:  # More than ~5% false positive rate
        print("  *** Some bits appear biased! ***")
    else:
        print("  Bits appear uniformly distributed")


print("SHA-256 NONCE CLUSTERING ANALYSIS")
print("=" * 70)
print()
print("Question: Do valid nonces cluster, or are they randomly distributed?")
print()

header = b"Nonce clustering analysis block 2026"

# Test at different difficulties
for target_zeros in [8, 10, 12]:
    print("=" * 70)
    print(f"TARGET: {target_zeros} leading zeros")
    print("=" * 70)
    print()

    max_search = 2**(target_zeros + 6)  # Expect ~64 valid nonces
    print(f"Searching {max_search} nonces...")

    start = time.time()
    valid = find_valid_nonces(header, target_zeros, max_search)
    elapsed = time.time() - start

    print(f"Found {len(valid)} valid nonces in {elapsed:.2f}s")
    print()

    if len(valid) >= 2:
        gaps = analyze_clustering(valid, max_search)
        print()
        analyze_bit_patterns(valid)
        print()

        if len(valid) >= 10:
            analyze_neighborhood(header, valid, window=500)

    print()


print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("If valid nonces cluster (gaps not exponential), we could:")
print("  1. Find one valid nonce")
print("  2. Search nearby for more")
print("  3. This would narrow the cone")
print()
print("If valid nonces are random (gaps exponential):")
print("  No cone narrowing is possible")
print("  SHA-256 is functioning as designed")


if __name__ == "__main__":
    pass
