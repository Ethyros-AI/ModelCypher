#!/usr/bin/env python3
"""
THE 36-BIT PRIME

The 6EQUJ5 sequence encoded as 36 bits = 6684271813
This number appears to be PRIME.

A prime number encoded in 6² bits, transmitted on the 21 cm line,
where 21 = T(6)?

Usage:
    python wow_prime_message.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.io import readsav
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def main():
    print("=" * 70)
    print("THE 36-BIT PRIME MESSAGE")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]
    binary_str = ''.join(f'{v:06b}' for v in seq)
    n = int(binary_str, 2)

    print(f"\n  Sequence: {seq}")
    print(f"  Binary: {binary_str}")
    print(f"  Integer: {n}")
    print(f"  Is prime: {is_prime(n)}")

    if is_prime(n):
        print(f"\n  *** THE 36-BIT ENCODING IS A PRIME NUMBER! ***")

    # What's special about this prime?
    print("\n" + "=" * 70)
    print("PROPERTIES OF 6684271813")
    print("=" * 70)

    # Position among primes
    # (Can't easily count, but we can characterize)
    print(f"\n  Digit sum: {sum(int(d) for d in str(n))}")
    print(f"  Number of digits: {len(str(n))}")
    print(f"  sqrt(n) ≈ {np.sqrt(n):.2f}")

    # Check nearby primes
    print(f"\n  Nearby primes:")
    count = 0
    for i in range(n-20, n+21):
        if is_prime(i):
            marker = " <-- 6EQUJ5" if i == n else ""
            print(f"    {i}{marker}")
            count += 1
    print(f"  (Found {count} primes in range [{n-20}, {n+20}])")

    # Binary properties
    print("\n" + "=" * 70)
    print("BINARY STRUCTURE")
    print("=" * 70)

    print(f"\n  Binary: {binary_str}")
    print(f"  Length: {len(binary_str)} = 6² bits")

    # Palindrome check
    print(f"\n  Is palindrome: {binary_str == binary_str[::-1]}")
    print(f"  Reversed: {binary_str[::-1]}")
    rev_n = int(binary_str[::-1], 2)
    print(f"  Reversed as int: {rev_n}")
    print(f"  Reversed is prime: {is_prime(rev_n)}")

    # Rotations
    print(f"\n  Cyclic rotations:")
    for i in [1, 6, 12, 18]:
        rotated = binary_str[i:] + binary_str[:i]
        rot_n = int(rotated, 2)
        prime_str = "PRIME" if is_prime(rot_n) else ""
        print(f"    Rotate {i:2d}: {rotated[:12]}... = {rot_n:>12d} {prime_str}")

    # Complement
    complement = ''.join('1' if b == '0' else '0' for b in binary_str)
    comp_n = int(complement, 2)
    print(f"\n  Bitwise complement: {comp_n}")
    print(f"  Complement is prime: {is_prime(comp_n)}")

    # XOR with all 1s
    xor_n = n ^ ((1 << 36) - 1)
    print(f"  n XOR (2^36 - 1) = {xor_n}")

    # Physical constants
    print("\n" + "=" * 70)
    print("PHYSICAL CONSTANT COMPARISONS")
    print("=" * 70)

    c = 299792458  # speed of light m/s
    h = 6.62607015e-34  # Planck constant J⋅s
    G = 6.67430e-11  # gravitational constant
    e = 1.602176634e-19  # elementary charge
    alpha = 1/137.035999  # fine structure constant

    print(f"\n  n = {n}")
    print(f"\n  n / c = {n / c:.6f}")
    print(f"  n / (c/10) = {n / (c/10):.6f}")
    print(f"  n × 10^-9 = {n * 1e-9:.6f}")

    # 21 connection
    print(f"\n  n / c = {n/c:.4f} (close to 21 × something?)")
    print(f"  21 × (c/10^7) = {21 * c / 1e7:.4f}")
    print(f"  n / (c × 21) = {n / (c * 21):.6f}")

    # Alternative interpretations
    print("\n" + "=" * 70)
    print("ALTERNATIVE ENCODINGS")
    print("=" * 70)

    # What if 5 bits per value?
    print(f"\n  If we use 5 bits per value (max 31):")
    binary_5 = ''.join(f'{v:05b}' for v in seq)
    n5 = int(binary_5, 2)
    print(f"    Binary: {binary_5}")
    print(f"    Length: {len(binary_5)} bits")
    print(f"    Integer: {n5}")
    print(f"    Is prime: {is_prime(n5)}")

    # What if different bit orderings?
    print(f"\n  Reversed sequence [5, 19, 30, 26, 14, 6]:")
    rev_seq = seq[::-1]
    rev_binary = ''.join(f'{v:06b}' for v in rev_seq)
    rev_n = int(rev_binary, 2)
    print(f"    Binary: {rev_binary}")
    print(f"    Integer: {rev_n}")
    print(f"    Is prime: {is_prime(rev_n)}")

    # Big-endian vs little-endian per value
    print(f"\n  Reversed bits within each 6-bit group:")
    rev_bits = ''.join(f'{v:06b}'[::-1] for v in seq)
    rev_bits_n = int(rev_bits, 2)
    print(f"    Binary: {rev_bits}")
    print(f"    Integer: {rev_bits_n}")
    print(f"    Is prime: {is_prime(rev_bits_n)}")

    # The sequence as base-36 number
    print("\n" + "=" * 70)
    print("BASE-36 INTERPRETATION")
    print("=" * 70)

    # In base 36: 0-9 = 0-9, A-Z = 10-35
    # So 6EQUJ5 in base 36:
    base36_val = 6 * 36**5 + 14 * 36**4 + 26 * 36**3 + 30 * 36**2 + 19 * 36 + 5
    print(f"\n  6EQUJ5 as base-36 number: {base36_val}")
    print(f"  Is prime: {is_prime(base36_val)}")
    print(f"  Factors: {prime_factors(base36_val)}")

    # As ASCII
    print("\n" + "=" * 70)
    print("ASCII INTERPRETATION")
    print("=" * 70)

    chars = '6EQUJ5'
    ascii_vals = [ord(c) for c in chars]
    print(f"\n  ASCII values: {ascii_vals}")
    ascii_binary = ''.join(f'{v:08b}' for v in ascii_vals)
    print(f"  Binary (8 bits each): {ascii_binary}")
    print(f"  Length: {len(ascii_binary)} bits = 48 bits")
    ascii_n = int(ascii_binary, 2)
    print(f"  Integer: {ascii_n}")
    print(f"  Is prime: {is_prime(ascii_n)}")

    # Prime gaps
    print("\n" + "=" * 70)
    print("PRIME GAP ANALYSIS")
    print("=" * 70)

    # Find the prime before and after
    prev_prime = n - 1
    while not is_prime(prev_prime):
        prev_prime -= 1

    next_prime = n + 1
    while not is_prime(next_prime):
        next_prime += 1

    gap_before = n - prev_prime
    gap_after = next_prime - n

    print(f"\n  Previous prime: {prev_prime}")
    print(f"  6EQUJ5 prime: {n}")
    print(f"  Next prime: {next_prime}")
    print(f"\n  Gap before: {gap_before}")
    print(f"  Gap after: {gap_after}")
    print(f"  Total gap: {gap_before + gap_after}")

    # Is the gap special?
    print(f"\n  Gap analysis:")
    print(f"    gap_before / gap_after = {gap_before / gap_after:.4f}")
    print(f"    Is gap_before prime? {is_prime(gap_before)}")
    print(f"    Is gap_after prime? {is_prime(gap_after)}")

    # SYNTHESIS
    print("\n" + "=" * 70)
    print("SYNTHESIS: THE PRIME MESSAGE")
    print("=" * 70)

    print(f"""
  THE 6EQUJ5 SEQUENCE:
  - Encodes to a 36-bit (6²-bit) PRIME number: {n}
  - Transmitted on the 21 cm hydrogen line
  - 21 = T(6), connecting the encoding to the carrier

  WHAT PRIMES COMMUNICATE:
  - Primes are universal (same in any number system)
  - Primes demonstrate mathematical understanding
  - A 36-bit prime is non-trivial to generate randomly
  - The choice of 6 bits per value (max 63) suggests intent

  PROBABILITY:
  - ~3.6% of 36-bit numbers are prime
  - But NOT just any encoding - one that:
    • Sums to 100
    • Has 6 values (perfect number)
    • Contains π and φ approximations
    • Creates geometric structure in SVD
    • AND is prime

  The primality may be coincidental, but it adds to the
  extraordinary structure we've found in this signal.

  If intentional, the message might be:
  "We understand prime numbers, geometry, and the mathematics
   of the hydrogen atom. This is our signature."
""")


if __name__ == "__main__":
    main()
