#!/usr/bin/env python3
"""SHA-256 Cone Search Strategy.

We can't beat 2^k, but we can NARROW THE CONE of where we search.

The manifold analysis revealed:
1. Nonce bits have 20% variation in influence (499-600 output bits)
2. Effective dimension = 3 (search happens on a 3D submanifold)
3. The π/e overhead suggests ~15% inefficiency in uniform search

Strategy: Instead of uniform random search, build a "probability cone"
that focuses computational effort on high-probability regions.
"""

import hashlib
import struct
import numpy as np
import time
from typing import Tuple, List, Dict
from collections import defaultdict
import math

PI = math.pi
E = math.e
LN2 = math.log(2)

# Nonce bit influence (from manifold analysis)
# These are the number of output bits affected by each nonce bit
NONCE_INFLUENCE = {
    0: 574, 1: 549, 2: 555, 3: 556, 4: 552, 5: 553, 6: 549, 7: 595,
    8: 567, 9: 537, 10: 529, 11: 550, 12: 576, 13: 572, 14: 556, 15: 574,
    16: 545, 17: 574, 18: 577, 19: 550, 20: 531, 21: 526, 22: 571, 23: 600,
    24: 548, 25: 538, 26: 504, 27: 525, 28: 499, 29: 570, 30: 568, 31: 540
}

# Rank bits by influence
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

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


class ConeSearcher:
    """
    Search for valid hashes using a probability cone strategy.

    The cone is defined by:
    1. High-influence bits form the "axis" of the cone
    2. Low-influence bits are the "width" of the cone
    3. We search along the axis first, then expand the width
    """

    def __init__(self, header: bytes, use_double: bool = True):
        self.header = header
        self.hash_func = double_sha256 if use_double else sha256
        self.hashes_computed = 0
        self.best_zeros = 0
        self.best_nonce = 0
        self.history = []

    def _hash_nonce(self, nonce: int) -> Tuple[bytes, int]:
        """Hash with nonce and return (hash, leading_zeros)."""
        data = self.header + struct.pack('<I', nonce)
        h = self.hash_func(data)
        zeros = count_leading_zeros(h)
        self.hashes_computed += 1

        if zeros > self.best_zeros:
            self.best_zeros = zeros
            self.best_nonce = nonce

        return h, zeros

    def uniform_search(self, max_hashes: int, target_zeros: int) -> Tuple[int, int, int]:
        """Baseline: uniform sequential search."""
        self.hashes_computed = 0
        self.best_zeros = 0

        for nonce in range(max_hashes):
            _, zeros = self._hash_nonce(nonce)
            if zeros >= target_zeros:
                return nonce, zeros, self.hashes_computed

        return self.best_nonce, self.best_zeros, self.hashes_computed

    def cone_search_v1(self, max_hashes: int, target_zeros: int) -> Tuple[int, int, int]:
        """
        Cone Search V1: Prioritize high-influence bit combinations.

        Strategy:
        - Search all 2^8 combinations of top 8 high-influence bits
        - For each, try a few variations of low-influence bits
        - Then fill remaining budget with local search around best
        """
        self.hashes_computed = 0
        self.best_zeros = 0

        top_bits = HIGH_INFLUENCE_BITS[:8]
        low_bits = LOW_INFLUENCE_BITS[:8]

        # Phase 1: Explore high-influence bit combinations
        phase1_budget = min(max_hashes // 3, 256 * 16)

        for hi_combo in range(256):
            if self.hashes_computed >= phase1_budget:
                break

            # Set high-influence bits
            base_nonce = 0
            for i, bit in enumerate(top_bits):
                if hi_combo & (1 << i):
                    base_nonce |= (1 << bit)

            # Try a few low-influence variations
            for lo_combo in range(min(16, max_hashes - self.hashes_computed)):
                nonce = base_nonce
                for i, bit in enumerate(low_bits[:4]):
                    if lo_combo & (1 << i):
                        nonce |= (1 << bit)

                _, zeros = self._hash_nonce(nonce)
                if zeros >= target_zeros:
                    return nonce, zeros, self.hashes_computed

        # Phase 2: Local search around best so far
        phase2_budget = max_hashes - self.hashes_computed
        best = self.best_nonce

        for offset in range(phase2_budget):
            nonce = (best + offset) % (2**32)
            _, zeros = self._hash_nonce(nonce)
            if zeros >= target_zeros:
                return nonce, zeros, self.hashes_computed

        return self.best_nonce, self.best_zeros, self.hashes_computed

    def cone_search_v2(self, max_hashes: int, target_zeros: int) -> Tuple[int, int, int]:
        """
        Cone Search V2: Adaptive cone based on feedback.

        Strategy:
        - Start with wide exploration of high-influence bits
        - Track which bit patterns produce more zeros
        - Narrow the cone toward promising patterns
        """
        self.hashes_computed = 0
        self.best_zeros = 0

        # Track bit patterns that lead to higher zeros
        bit_scores = defaultdict(lambda: [0, 0])  # [total_zeros, count]

        # Phase 1: Exploration
        exploration_budget = min(max_hashes // 4, 10000)

        for _ in range(exploration_budget):
            # Random nonce
            nonce = np.random.randint(0, 2**32)
            _, zeros = self._hash_nonce(nonce)

            if zeros >= target_zeros:
                return nonce, zeros, self.hashes_computed

            # Record which bits were set
            for bit in range(32):
                if nonce & (1 << bit):
                    bit_scores[bit][0] += zeros
                    bit_scores[bit][1] += 1

        # Compute average zeros per bit
        bit_quality = {}
        for bit, (total, count) in bit_scores.items():
            if count > 0:
                bit_quality[bit] = total / count

        # Rank bits by quality (higher = more zeros when set)
        ranked_bits = sorted(bit_quality.keys(), key=lambda b: -bit_quality.get(b, 0))

        # Phase 2: Exploit - focus on high-quality bit patterns
        exploit_budget = max_hashes - self.hashes_computed

        # Create biased nonces: more likely to set high-quality bits
        for _ in range(exploit_budget):
            nonce = 0
            for i, bit in enumerate(ranked_bits):
                # Probability of setting bit decreases with rank
                prob = 0.7 - (i / len(ranked_bits)) * 0.5
                if np.random.random() < prob:
                    nonce |= (1 << bit)

            _, zeros = self._hash_nonce(nonce)
            if zeros >= target_zeros:
                return nonce, zeros, self.hashes_computed

        return self.best_nonce, self.best_zeros, self.hashes_computed

    def cone_search_v3(self, max_hashes: int, target_zeros: int) -> Tuple[int, int, int]:
        """
        Cone Search V3: 3D Manifold projection.

        Since effective dimension = 3, project search onto 3 principal axes.
        """
        self.hashes_computed = 0
        self.best_zeros = 0

        # Divide 32 bits into 3 groups (the "3 dimensions")
        # Based on influence ranking
        dim1_bits = HIGH_INFLUENCE_BITS[:11]   # Top 11 (most influence)
        dim2_bits = HIGH_INFLUENCE_BITS[11:22] # Middle 11
        dim3_bits = HIGH_INFLUENCE_BITS[22:]   # Bottom 10 (least influence)

        # Search in a spiral pattern through the 3D space
        # Start at center, expand outward

        max_per_dim = int(np.cbrt(max_hashes))  # Cube root for 3D

        center = [max_per_dim // 2] * 3

        for radius in range(max_per_dim // 2):
            # Generate points at this radius in 3D grid
            for d1 in range(-radius, radius + 1):
                for d2 in range(-radius, radius + 1):
                    for d3 in range(-radius, radius + 1):
                        # Only points on the surface of the cube
                        if abs(d1) != radius and abs(d2) != radius and abs(d3) != radius:
                            continue

                        if self.hashes_computed >= max_hashes:
                            return self.best_nonce, self.best_zeros, self.hashes_computed

                        # Convert 3D coordinates to nonce
                        val1 = (center[0] + d1) % (1 << len(dim1_bits))
                        val2 = (center[1] + d2) % (1 << len(dim2_bits))
                        val3 = (center[2] + d3) % (1 << len(dim3_bits))

                        nonce = 0
                        for i, bit in enumerate(dim1_bits):
                            if val1 & (1 << i):
                                nonce |= (1 << bit)
                        for i, bit in enumerate(dim2_bits):
                            if val2 & (1 << i):
                                nonce |= (1 << bit)
                        for i, bit in enumerate(dim3_bits):
                            if val3 & (1 << i):
                                nonce |= (1 << bit)

                        _, zeros = self._hash_nonce(nonce)
                        if zeros >= target_zeros:
                            return nonce, zeros, self.hashes_computed

        return self.best_nonce, self.best_zeros, self.hashes_computed


def benchmark_strategies(header: bytes, max_hashes: int, target_zeros: int, trials: int = 10):
    """Compare different cone search strategies."""

    results = {
        'uniform': [],
        'cone_v1': [],
        'cone_v2': [],
        'cone_v3': [],
    }

    for trial in range(trials):
        trial_header = header + struct.pack('>I', trial)

        for name, method in [
            ('uniform', 'uniform_search'),
            ('cone_v1', 'cone_search_v1'),
            ('cone_v2', 'cone_search_v2'),
            ('cone_v3', 'cone_search_v3'),
        ]:
            searcher = ConeSearcher(trial_header)
            start = time.time()
            nonce, zeros, hashes = getattr(searcher, method)(max_hashes, target_zeros)
            elapsed = time.time() - start

            results[name].append({
                'zeros': zeros,
                'hashes': hashes,
                'time': elapsed,
                'found': zeros >= target_zeros
            })

    return results


def analyze_results(results: Dict, target_zeros: int):
    """Analyze benchmark results."""
    print(f"\n{'Strategy':<12} {'Avg Zeros':<10} {'Avg Hashes':<12} {'Success %':<10} {'Time':<10}")
    print("-" * 60)

    uniform_hashes = np.mean([r['hashes'] for r in results['uniform']])

    for name, data in results.items():
        avg_zeros = np.mean([r['zeros'] for r in data])
        avg_hashes = np.mean([r['hashes'] for r in data])
        success_rate = np.mean([r['found'] for r in data]) * 100
        avg_time = np.mean([r['time'] for r in data])

        # Speedup relative to uniform
        speedup = uniform_hashes / avg_hashes if avg_hashes > 0 else 0

        print(f"{name:<12} {avg_zeros:<10.2f} {avg_hashes:<12.0f} {success_rate:<10.1f} {avg_time:<10.4f}s")

    print()


print("SHA-256 CONE SEARCH STRATEGIES")
print("=" * 70)
print()
print("The goal: Focus search on high-probability regions of nonce space")
print()
print("Strategies:")
print("  UNIFORM:  Baseline sequential search")
print("  CONE_V1:  Prioritize high-influence bit combinations")
print("  CONE_V2:  Adaptive search based on feedback")
print("  CONE_V3:  3D manifold projection (effective dim = 3)")
print()

header = b"Cone search benchmark block 2026"

print("=" * 70)
print("TEST 1: Easy difficulty (target = 8 zeros)")
print("=" * 70)
results = benchmark_strategies(header, max_hashes=50000, target_zeros=8, trials=10)
analyze_results(results, 8)

print("=" * 70)
print("TEST 2: Medium difficulty (target = 12 zeros)")
print("=" * 70)
results = benchmark_strategies(header, max_hashes=100000, target_zeros=12, trials=10)
analyze_results(results, 12)

print("=" * 70)
print("TEST 3: Hard difficulty (target = 16 zeros)")
print("=" * 70)
results = benchmark_strategies(header, max_hashes=500000, target_zeros=16, trials=5)
analyze_results(results, 16)

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()
print("If any cone strategy consistently finds targets with fewer hashes,")
print("it indicates exploitable structure in the search space.")
print()
print("Expected outcome: All strategies perform similarly because")
print("SHA-256's compression function destroys input structure.")
print()
print("If cone strategies WIN: The manifold structure is exploitable!")
print("If cone strategies LOSE: The structure exists but isn't exploitable.")


if __name__ == "__main__":
    pass
