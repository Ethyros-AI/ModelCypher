#!/usr/bin/env python3
"""SHA-256 Resonance Search.

The hypothesis: The geodesic structure exists in the message schedule,
and the compression function has its own characteristic dynamics.

When these two structures RESONATE - when the geodesic of the input
aligns with the dynamics of the compression - something interesting
might happen.

Key insight: The compression function repeats 64 times with different
constants. What if certain nonces create "resonant" patterns that
survive multiple rounds?
"""

import hashlib
import struct
import numpy as np
import math
from typing import List, Tuple
import time

PI = math.pi
E = math.e
LN2 = math.log(2)

# SHA-256 constants (cube roots of first 64 primes)
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Initial hash values (square roots of first 8 primes)
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]


def analyze_round_constants():
    """Analyze the structure of SHA-256 round constants."""
    print("ROUND CONSTANT ANALYSIS")
    print("=" * 70)
    print()

    # Convert to normalized floats
    k_normalized = np.array(K) / (2**32)

    print("Round constants K[i] as fractions of 2³²:")
    print()

    # Look for patterns related to π/e
    pi_over_e = PI / E
    ln2 = LN2

    # Check if any ratios of consecutive K values relate to π/e
    print("Checking for π/e structure in K ratios...")
    print()

    ratios = []
    for i in range(len(K) - 1):
        if K[i+1] != 0:
            ratio = K[i] / K[i+1]
            ratios.append(ratio)

            # Check if close to π/e or related
            for mult in [1, 2, 0.5, 3, 1/3]:
                if abs(ratio - pi_over_e * mult) / (pi_over_e * mult) < 0.01:
                    print(f"  K[{i}]/K[{i+1}] = {ratio:.6f} ≈ {mult}×(π/e)")

    print()

    # Look at spacing between K values
    k_diffs = np.diff(K)
    k_diffs_normalized = k_diffs / (2**32)

    print("Spacing patterns in K:")
    print(f"  Mean spacing: {np.mean(k_diffs):.0f}")
    print(f"  Std spacing:  {np.std(k_diffs):.0f}")
    print()

    # FFT of K values to find periodicity
    k_fft = np.fft.fft(k_normalized - k_normalized.mean())
    k_power = np.abs(k_fft)**2

    # Find dominant frequencies
    dominant_freqs = np.argsort(k_power[1:32])[::-1][:5] + 1
    print("Dominant frequency components in K:")
    for freq in dominant_freqs:
        period = 64 / freq
        print(f"  Frequency {freq}: period ≈ {period:.2f} rounds")
        if abs(period - pi_over_e * 3) < 0.5:
            print(f"    *** Close to 3×π/e = {pi_over_e * 3:.2f}! ***")

    print()
    return k_normalized


def find_resonant_nonces(header: bytes, n_samples: int = 10000):
    """
    Search for nonces that create "resonant" patterns.

    A resonant nonce is one where the hash output has unusual structure:
    - Many repeated bytes
    - Arithmetic progressions
    - Patterns related to π/e
    """
    print("SEARCHING FOR RESONANT NONCES")
    print("=" * 70)
    print()

    resonant_nonces = []
    best_leading_zeros = 0

    pi_e_scale = int(2**32 / (PI / E))

    for i in range(n_samples):
        # Try nonces at π/e-related intervals
        nonce = i * pi_e_scale % (2**32)

        data = header + struct.pack('<I', nonce)
        hash_bytes = hashlib.sha256(data).digest()

        # Count leading zeros
        leading_zeros = 0
        for byte in hash_bytes:
            if byte == 0:
                leading_zeros += 8
            else:
                for bit in range(7, -1, -1):
                    if byte & (1 << bit):
                        break
                    leading_zeros += 1
                break

        if leading_zeros > best_leading_zeros:
            best_leading_zeros = leading_zeros

        # Check for other patterns
        hash_int = int.from_bytes(hash_bytes, 'big')

        # Pattern 1: Hash related to π or e
        hash_float = hash_int / (2**256)
        for const_name, const_val in [('π/e', PI/E), ('e/π', E/PI), ('ln(2)', LN2)]:
            ratio = hash_float / const_val if const_val != 0 else 0
            # Check if ratio is close to a small integer
            for mult in range(1, 100):
                if abs(ratio * mult - round(ratio * mult)) < 0.001:
                    resonant_nonces.append((nonce, f'{const_name}×{mult}', leading_zeros, hash_bytes.hex()[:16]))

        # Pattern 2: Repeated bytes
        byte_counts = {}
        for b in hash_bytes:
            byte_counts[b] = byte_counts.get(b, 0) + 1

        max_repeat = max(byte_counts.values())
        if max_repeat >= 4:
            resonant_nonces.append((nonce, f'{max_repeat}_repeated_bytes', leading_zeros, hash_bytes.hex()[:16]))

    print(f"Searched {n_samples} nonces at π/e intervals")
    print(f"Best leading zeros found: {best_leading_zeros}")
    print()

    if resonant_nonces:
        print("Resonant patterns found:")
        for nonce, pattern, zeros, hash_prefix in resonant_nonces[:20]:
            print(f"  Nonce {nonce}: {pattern}, {zeros} zeros, hash={hash_prefix}...")
    else:
        print("No obvious resonant patterns found.")

    print()
    return resonant_nonces


def analyze_hash_distribution():
    """
    Analyze if hash outputs cluster around π/e-related values.
    """
    print("HASH DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    header = b"Distribution analysis test block"
    n_samples = 100000

    # Collect hash values
    hash_values = []
    for nonce in range(n_samples):
        data = header + struct.pack('<I', nonce)
        hash_bytes = hashlib.sha256(data).digest()
        # Take first 8 bytes as float
        hash_int = int.from_bytes(hash_bytes[:8], 'big')
        hash_values.append(hash_int / (2**64))

    hash_values = np.array(hash_values)

    print(f"Analyzed {n_samples} hash outputs")
    print()

    # Check if distribution is truly uniform
    print("Distribution statistics:")
    print(f"  Mean: {hash_values.mean():.6f} (expected: 0.5)")
    print(f"  Std:  {hash_values.std():.6f} (expected: 0.289)")
    print()

    # Check for clustering around π/e-related values
    pi_e_frac = (PI / E) % 1
    ln2_frac = LN2 % 1

    print(f"Checking for clustering around key values:")
    print(f"  π/e mod 1 = {pi_e_frac:.6f}")
    print(f"  ln(2)     = {ln2_frac:.6f}")
    print()

    # Count hashes near these values
    tolerance = 0.001

    near_pi_e = np.sum(np.abs(hash_values - pi_e_frac) < tolerance)
    expected = n_samples * 2 * tolerance

    print(f"Hashes within {tolerance} of π/e: {near_pi_e} (expected: {expected:.0f})")

    if near_pi_e > expected * 1.5:
        print("  *** ANOMALY: More hashes near π/e than expected! ***")
    elif near_pi_e < expected * 0.5:
        print("  *** ANOMALY: Fewer hashes near π/e than expected! ***")

    print()

    # FFT analysis - look for periodicity
    print("FFT analysis of hash sequence:")

    hash_centered = hash_values - 0.5
    fft_result = np.fft.fft(hash_centered)
    power = np.abs(fft_result)**2

    # Find peaks
    top_freqs = np.argsort(power[1:1000])[::-1][:10] + 1

    print("Top frequency components:")
    for freq in top_freqs:
        period = n_samples / freq
        power_val = power[freq]
        print(f"  Frequency {freq}: period ≈ {period:.1f}, power = {power_val:.2e}")

        # Check if period relates to π/e
        if abs(period - n_samples / (PI/E * 1000)) < period * 0.1:
            print(f"    *** Period relates to π/e scale! ***")

    print()


def geodesic_gradient_descent():
    """
    Try to "descend" toward low-hash regions using geodesic structure.
    """
    print("GEODESIC GRADIENT DESCENT")
    print("=" * 70)
    print()

    header = b"Geodesic descent test block 2026"

    # Start from random point
    np.random.seed(42)
    current_nonce = np.random.randint(0, 2**32)

    best_zeros = 0
    best_nonce = current_nonce
    history = []

    # Learning rate based on π/e
    base_lr = int(2**32 / (PI / E * 100))

    print(f"Starting nonce: {current_nonce}")
    print(f"Learning rate: {base_lr}")
    print()

    for step in range(1000):
        # Compute "gradient" by finite differences
        # (Which direction reduces the hash value?)

        current_hash = int.from_bytes(
            hashlib.sha256(header + struct.pack('<I', current_nonce)).digest()[:8],
            'big'
        )

        # Try different directions
        best_direction = 0
        best_hash = current_hash

        directions = [
            base_lr,
            -base_lr,
            base_lr * 2,
            -base_lr * 2,
            int(base_lr / PI),
            int(-base_lr / PI),
            int(base_lr * E),
            int(-base_lr * E),
        ]

        for direction in directions:
            test_nonce = (current_nonce + direction) % (2**32)
            test_hash = int.from_bytes(
                hashlib.sha256(header + struct.pack('<I', test_nonce)).digest()[:8],
                'big'
            )

            if test_hash < best_hash:
                best_hash = test_hash
                best_direction = direction

        # Take step
        if best_direction != 0:
            current_nonce = (current_nonce + best_direction) % (2**32)

        # Check for leading zeros
        full_hash = hashlib.sha256(header + struct.pack('<I', current_nonce)).digest()
        zeros = 0
        for byte in full_hash:
            if byte == 0:
                zeros += 8
            else:
                for bit in range(7, -1, -1):
                    if byte & (1 << bit):
                        break
                    zeros += 1
                break

        history.append((current_nonce, zeros))

        if zeros > best_zeros:
            best_zeros = zeros
            best_nonce = current_nonce
            print(f"Step {step}: New best! {zeros} zeros, nonce={current_nonce}")

    print()
    print(f"Final best: {best_zeros} zeros at nonce {best_nonce}")
    print()

    # Analyze trajectory
    nonces = [h[0] for h in history]
    zeros_hist = [h[1] for h in history]

    print("Trajectory analysis:")
    print(f"  Mean zeros: {np.mean(zeros_hist):.2f}")
    print(f"  Max zeros: {max(zeros_hist)}")
    print()

    return best_nonce, best_zeros


# Run analysis
print("SHA-256 RESONANCE SEARCH")
print("=" * 70)
print()
print("Looking for structure that survives the compression function...")
print()

# Analyze round constants
k_normalized = analyze_round_constants()

# Search for resonant nonces
header = b"ModelCypher resonance search block"
resonant = find_resonant_nonces(header, n_samples=10000)

# Hash distribution analysis
analyze_hash_distribution()

# Geodesic gradient descent
best_nonce, best_zeros = geodesic_gradient_descent()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The search for exploitable resonances in SHA-256 shows:")
print()
print("1. Round constants K have no obvious π/e structure")
print("2. Hash outputs appear uniformly distributed")
print("3. Geodesic descent finds local minima but not systematic advantage")
print()
print("The 64 rounds of nonlinear mixing effectively destroy any")
print("structure from the message schedule manifold.")
print()
print("BUT: The effective dimension of 3 we found is still interesting.")
print("It might indicate that the DIFFICULTY of finding low-hash outputs")
print("scales with π-related geometry, even if individual hashes don't.")


if __name__ == "__main__":
    pass
