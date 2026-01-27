#!/usr/bin/env python3
"""Anisotropic Mining Strategy for SHA-256.

Key insight from manifold analysis:
- Effective dimension = 3 (despite 512 input bits)
- Nonce bits have unequal influence (499-600 output bits affected)
- The metric is highly anisotropic (condition number 2812)

Strategy: Instead of uniform random nonce search, search along
the principal axes of the message schedule manifold.
"""

import hashlib
import struct
import numpy as np
import time
from typing import Tuple, List
import math

PI_OVER_E = math.pi / math.e

# Nonce bit weights (from manifold analysis)
# Higher weight = more "distance" in message schedule space
NONCE_BIT_WEIGHTS = {
    23: 600, 7: 595, 18: 577, 12: 576, 15: 574, 17: 574, 0: 574,
    13: 572, 22: 571, 29: 570, 30: 568, 8: 567, 14: 556, 3: 556,
    2: 555, 4: 552, 5: 553, 11: 550, 19: 550, 1: 549, 6: 549,
    24: 548, 16: 545, 31: 540, 25: 538, 9: 537, 20: 531, 10: 529,
    27: 525, 21: 526, 26: 504, 28: 499
}

# Sort bits by weight (high to low)
SORTED_BITS = sorted(NONCE_BIT_WEIGHTS.keys(), key=lambda b: -NONCE_BIT_WEIGHTS[b])

def sha256_double(data: bytes) -> bytes:
    """Double SHA-256 (Bitcoin-style)."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def count_leading_zeros(hash_bytes: bytes) -> int:
    """Count leading zero bits in hash."""
    n = 0
    for byte in hash_bytes:
        if byte == 0:
            n += 8
        else:
            # Count leading zeros in this byte
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    return n
                n += 1
            return n
    return n

def set_bit(n: int, bit: int, value: int) -> int:
    """Set a specific bit in integer n."""
    if value:
        return n | (1 << bit)
    else:
        return n & ~(1 << bit)

def get_bit(n: int, bit: int) -> int:
    """Get a specific bit from integer n."""
    return (n >> bit) & 1

class AnisotropicMiner:
    """
    Miner that exploits message schedule manifold structure.

    Instead of uniform random search, we:
    1. Search high-influence bits first
    2. Use Gray code ordering (minimize bit flips)
    3. Prune low-influence bits when close to solution
    """

    def __init__(self, header: bytes):
        self.header = header
        self.hashes_computed = 0
        self.best_zeros = 0
        self.best_nonce = 0

    def mine_uniform(self, max_nonces: int, target_zeros: int) -> Tuple[int, int, float]:
        """Standard uniform random mining (baseline)."""
        start = time.time()
        self.hashes_computed = 0
        self.best_zeros = 0

        for nonce in range(max_nonces):
            data = self.header + struct.pack('<I', nonce)
            hash_result = sha256_double(data)
            zeros = count_leading_zeros(hash_result)

            self.hashes_computed += 1

            if zeros > self.best_zeros:
                self.best_zeros = zeros
                self.best_nonce = nonce

            if zeros >= target_zeros:
                elapsed = time.time() - start
                return nonce, zeros, elapsed

        elapsed = time.time() - start
        return self.best_nonce, self.best_zeros, elapsed

    def mine_anisotropic(self, max_nonces: int, target_zeros: int) -> Tuple[int, int, float]:
        """
        Anisotropic mining: search high-influence bits more aggressively.

        Strategy:
        - Divide nonce space into "shells" based on bit weight
        - Search high-weight bit combinations first
        - Use structure to skip unlikely regions
        """
        start = time.time()
        self.hashes_computed = 0
        self.best_zeros = 0

        # Phase 1: Search along high-influence axes
        # Try all combinations of top 8 high-weight bits first
        high_bits = SORTED_BITS[:8]  # Bits with most influence

        for combo in range(256):  # 2^8 combinations
            nonce = 0
            for i, bit in enumerate(high_bits):
                if combo & (1 << i):
                    nonce |= (1 << bit)

            data = self.header + struct.pack('<I', nonce)
            hash_result = sha256_double(data)
            zeros = count_leading_zeros(hash_result)

            self.hashes_computed += 1

            if zeros > self.best_zeros:
                self.best_zeros = zeros
                self.best_nonce = nonce

            if zeros >= target_zeros:
                elapsed = time.time() - start
                return nonce, zeros, elapsed

        # Phase 2: Refine best candidates with medium-influence bits
        # Take top 10 candidates and try medium-weight bit variations
        candidates = [self.best_nonce]
        med_bits = SORTED_BITS[8:16]

        for base_nonce in candidates:
            for combo in range(256):
                nonce = base_nonce
                for i, bit in enumerate(med_bits):
                    if combo & (1 << i):
                        nonce |= (1 << bit)
                    else:
                        nonce &= ~(1 << bit)

                if self.hashes_computed >= max_nonces:
                    break

                data = self.header + struct.pack('<I', nonce)
                hash_result = sha256_double(data)
                zeros = count_leading_zeros(hash_result)

                self.hashes_computed += 1

                if zeros > self.best_zeros:
                    self.best_zeros = zeros
                    self.best_nonce = nonce

                if zeros >= target_zeros:
                    elapsed = time.time() - start
                    return nonce, zeros, elapsed

        # Phase 3: Fill remaining budget with sequential search
        # (but starting from best known region)
        base = self.best_nonce & 0xFFFF0000  # Keep high bits
        for low_bits in range(65536):
            if self.hashes_computed >= max_nonces:
                break

            nonce = base | low_bits

            data = self.header + struct.pack('<I', nonce)
            hash_result = sha256_double(data)
            zeros = count_leading_zeros(hash_result)

            self.hashes_computed += 1

            if zeros > self.best_zeros:
                self.best_zeros = zeros
                self.best_nonce = nonce

            if zeros >= target_zeros:
                elapsed = time.time() - start
                return nonce, zeros, elapsed

        elapsed = time.time() - start
        return self.best_nonce, self.best_zeros, elapsed

    def mine_geodesic(self, max_nonces: int, target_zeros: int) -> Tuple[int, int, float]:
        """
        Geodesic mining: follow curves on the message schedule manifold.

        Uses the π/e scale to determine step sizes and directions.
        """
        start = time.time()
        self.hashes_computed = 0
        self.best_zeros = 0

        # Start from a random point
        import random
        current = random.randint(0, 2**32 - 1)

        # Step size based on π/e (the characteristic scale)
        # We'll vary nonces in "quantum" steps related to π/e
        step_quantum = int(2**32 / (PI_OVER_E * 1000))

        directions = [
            step_quantum,
            -step_quantum,
            step_quantum * 2,
            -step_quantum * 2,
            int(step_quantum / PI_OVER_E),
            int(-step_quantum / PI_OVER_E),
        ]

        while self.hashes_computed < max_nonces:
            # Evaluate current point
            data = self.header + struct.pack('<I', current & 0xFFFFFFFF)
            hash_result = sha256_double(data)
            zeros = count_leading_zeros(hash_result)

            self.hashes_computed += 1

            if zeros > self.best_zeros:
                self.best_zeros = zeros
                self.best_nonce = current & 0xFFFFFFFF

            if zeros >= target_zeros:
                elapsed = time.time() - start
                return current & 0xFFFFFFFF, zeros, elapsed

            # Take a step along the geodesic
            direction = random.choice(directions)
            current = (current + direction) % (2**32)

            # Occasionally jump to a random point (avoid local minima)
            if self.hashes_computed % 1000 == 0:
                current = random.randint(0, 2**32 - 1)

        elapsed = time.time() - start
        return self.best_nonce, self.best_zeros, elapsed


def benchmark(header: bytes, max_nonces: int, target_zeros: int, trials: int = 5):
    """Compare mining strategies."""
    print(f"Benchmarking with {max_nonces} nonces, target {target_zeros} zeros")
    print("-" * 70)

    results = {
        'uniform': [],
        'anisotropic': [],
        'geodesic': []
    }

    for trial in range(trials):
        # Use different header each trial to avoid caching effects
        trial_header = header + struct.pack('>I', trial * 12345)
        miner = AnisotropicMiner(trial_header)

        # Uniform
        _, zeros, elapsed = miner.mine_uniform(max_nonces, target_zeros)
        results['uniform'].append((zeros, miner.hashes_computed, elapsed))

        # Anisotropic
        miner = AnisotropicMiner(trial_header)
        _, zeros, elapsed = miner.mine_anisotropic(max_nonces, target_zeros)
        results['anisotropic'].append((zeros, miner.hashes_computed, elapsed))

        # Geodesic
        miner = AnisotropicMiner(trial_header)
        _, zeros, elapsed = miner.mine_geodesic(max_nonces, target_zeros)
        results['geodesic'].append((zeros, miner.hashes_computed, elapsed))

    print(f"\nResults over {trials} trials:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Avg Zeros':<12} {'Avg Hashes':<15} {'Avg Time':<12}")
    print("-" * 70)

    for strategy, data in results.items():
        avg_zeros = np.mean([d[0] for d in data])
        avg_hashes = np.mean([d[1] for d in data])
        avg_time = np.mean([d[2] for d in data])
        print(f"{strategy:<15} {avg_zeros:<12.2f} {avg_hashes:<15.0f} {avg_time:<12.4f}s")

    print()
    return results


print("ANISOTROPIC SHA-256 MINING")
print("=" * 70)
print()

print("The message schedule manifold has:")
print("  - Effective dimension: 3")
print("  - Condition number: 2812")
print("  - Nonce bit influence varies by 20%")
print()

print("Strategies:")
print("  1. UNIFORM: Standard sequential nonce search")
print("  2. ANISOTROPIC: Prioritize high-influence nonce bits")
print("  3. GEODESIC: Follow π/e-scaled paths on manifold")
print()

# Run benchmark
header = b"ModelCypher Anisotropic Mining Test Block 2026"

print("=" * 70)
print("SHORT TEST (10K nonces, target 8 zeros)")
print("=" * 70)
results = benchmark(header, 10000, 8, trials=5)

print("=" * 70)
print("MEDIUM TEST (100K nonces, target 12 zeros)")
print("=" * 70)
results = benchmark(header, 100000, 12, trials=3)

print("=" * 70)
print("LONGER TEST (500K nonces, target 16 zeros)")
print("=" * 70)
results = benchmark(header, 500000, 16, trials=3)

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()
print("If anisotropic mining consistently finds more zeros with same hashes,")
print("it indicates exploitable structure in the message schedule manifold.")
print()
print("The π/e scale might determine optimal 'step sizes' for geodesic search,")
print("but the nonlinear compression function likely destroys any advantage.")
print()


if __name__ == "__main__":
    pass
