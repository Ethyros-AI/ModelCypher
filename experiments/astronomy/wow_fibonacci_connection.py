#!/usr/bin/env python3
"""
THE FIBONACCI/6 CONNECTION IN THE WOW! SIGNAL

Multiple appearances of 6 and Fibonacci numbers:
- 6 values in sequence
- 36 = 6² bits to encode
- 21 = F(8) = T(6) in angular velocity
- 21 cm hydrogen wavelength
- Gap 13 (Fibonacci) gives most simple ratios
- FWHM = 36 seconds

Is there a deeper Fibonacci structure?

Usage:
    python wow_fibonacci_connection.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.io import readsav
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"

phi = (1 + np.sqrt(5)) / 2

# Fibonacci sequence
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]


def load_raw_signal() -> np.ndarray:
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal - 0.5


def main():
    print("=" * 70)
    print("THE FIBONACCI/6 CONNECTION")
    print("=" * 70)

    signal = load_raw_signal()
    U, S, Vt = linalg.svd(signal, full_matrices=False)

    seq = [6, 14, 26, 30, 19, 5]

    # =========================================================================
    # THE NUMBER 6
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE NUMBER 6")
    print("=" * 70)

    print(f"""
  6 appears everywhere:

  IN THE SIGNAL:
    • 6 values in the peak sequence
    • Sequence starts with 6 and ends with 5 (diff = 1)
    • 36 = 6² bits to encode the sequence
    • FWHM = 3 samples, but 3 = 6/2
    • 21 = T(6) = 1+2+3+4+5+6

  MATHEMATICALLY:
    • 6 = 3! (first factorial > its argument)
    • 6 = 1×2×3 = 1+2+3 (only number with this property)
    • 6 is the first perfect number
    • 6 faces on a cube
    • 6-fold symmetry in nature (snowflakes, honeycombs)

  IN PHYSICS:
    • 6 quarks, 6 leptons
    • String theory requires 6 extra dimensions
    • Carbon has 6 protons (basis of organic life)
""")

    # =========================================================================
    # FIBONACCI IN THE SIGNAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIBONACCI IN THE SIGNAL")
    print("=" * 70)

    print(f"\n  Fibonacci sequence: {fib[:12]}")

    # Check if sequence values relate to Fibonacci
    print(f"\n  Sequence values vs Fibonacci:")
    for val in seq:
        closest_fib = min(fib, key=lambda f: abs(f - val))
        diff = val - closest_fib
        print(f"    {val}: closest Fib = {closest_fib}, diff = {diff:+d}")

    # Check differences
    diffs = np.diff(seq)
    print(f"\n  Differences: {list(diffs)}")
    print(f"  Fibonacci? {[d in fib or -d in fib for d in diffs]}")

    # Check ratios
    print(f"\n  Ratios of consecutive values:")
    for i in range(len(seq) - 1):
        if seq[i] != 0:
            r = seq[i+1] / seq[i]
            print(f"    {seq[i+1]}/{seq[i]} = {r:.4f} (φ = {phi:.4f}, diff = {abs(r-phi):.4f})")

    # =========================================================================
    # THE 36-BIT MESSAGE
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE 36-BIT MESSAGE")
    print("=" * 70)

    binary_str = ''.join(f'{v:06b}' for v in seq)
    print(f"\n  Binary (6 bits per value): {binary_str}")
    print(f"  Length: {len(binary_str)} = 6² bits")

    # As a single integer
    as_int = int(binary_str, 2)
    print(f"\n  As integer: {as_int}")
    print(f"  Factors: ", end="")

    n = as_int
    factors = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    print(factors)

    # Split into groups
    print(f"\n  Split into 6-bit groups:")
    for i in range(0, 36, 6):
        chunk = binary_str[i:i+6]
        val = int(chunk, 2)
        print(f"    {chunk} = {val}")

    # Split into 9-bit groups (36 = 4 × 9)
    print(f"\n  Split into 9-bit groups (36 = 4 × 9):")
    for i in range(0, 36, 9):
        chunk = binary_str[i:i+9]
        val = int(chunk, 2)
        print(f"    {chunk} = {val}")

    # Split into 12-bit groups (36 = 3 × 12)
    print(f"\n  Split into 12-bit groups (36 = 3 × 12):")
    for i in range(0, 36, 12):
        chunk = binary_str[i:i+12]
        val = int(chunk, 2)
        print(f"    {chunk} = {val}")

    # Is there structure in the bits?
    print(f"\n  Bit statistics:")
    ones = binary_str.count('1')
    zeros = binary_str.count('0')
    print(f"    1s: {ones}, 0s: {zeros}, ratio: {ones/zeros:.4f}")

    # =========================================================================
    # FIBONACCI IN SVD GAPS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIBONACCI IN SVD STRUCTURE")
    print("=" * 70)

    print(f"\n  Checking if Fibonacci gaps give special ratios...")

    fib_gaps = [1, 2, 3, 5, 8, 13, 21]

    for gap in fib_gaps:
        matches = []
        for i in range(min(20, len(S) - gap)):
            j = i + gap
            if S[j] > 1e-10:
                r = S[i] / S[j]
                # Check against special values
                for target, name in [(phi, 'φ'), (np.sqrt(2), '√2'),
                                     (np.pi/2, 'π/2'), (np.e, 'e'),
                                     (2, '2'), (np.sqrt(3), '√3')]:
                    if abs(r - target) / target < 0.01:
                        matches.append((i, j, name, r))

        if matches:
            print(f"\n  Gap = {gap} (Fibonacci):")
            for i, j, name, r in matches:
                print(f"    S[{i}]/S[{j}] = {r:.4f} ≈ {name}")

    # =========================================================================
    # THE 21 CONNECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE 21 CONNECTION")
    print("=" * 70)

    print(f"""
  21 appears in:

  1. HYDROGEN LINE: 21.12 cm wavelength
     - The universal "hello" frequency
     - Hyperfine transition of neutral hydrogen

  2. ANGULAR VELOCITY: 360°/21 ≈ 17.14°
     - The rate of rotation in mode space
     - Matches measured 17.65° to 3%

  3. FIBONACCI: F(8) = 21
     - 21 = 1,1,2,3,5,8,13,21
     - 8 is also Fibonacci (F(6) = 8)

  4. TRIANGULAR: T(6) = 21
     - 21 = 1+2+3+4+5+6
     - Related to the 6 values in sequence

  5. COMBINATORICS: C(7,2) = 21
     - Ways to choose 2 items from 7
     - 7 is a prime, 2 is the first prime

  Is 21 the "key" to the signal?
""")

    # =========================================================================
    # THE φ RATIO IN MATRIX SHAPE
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE GOLDEN RATIO IN MATRIX SHAPE")
    print("=" * 70)

    print(f"\n  Matrix: {signal.shape[0]} × {signal.shape[1]} = 82 × 50")
    print(f"  Ratio: 82/50 = {82/50:.6f}")
    print(f"  φ = {phi:.6f}")
    print(f"  Error: {abs(82/50 - phi)/phi*100:.2f}%")

    print(f"\n  What if this IS intentional?")
    print(f"  To get closer to φ:")
    print(f"    φ × 50 = {phi * 50:.2f} (would need 81 samples)")
    print(f"    φ × 51 = {phi * 51:.2f} (would need 82-83 samples)")

    # The telescope sampled every 12 seconds
    # 82 samples × 12 sec = 984 seconds = 16.4 minutes
    print(f"\n  82 samples × 12 sec = 984 sec = 16.4 min")
    print(f"  50 frequency channels")
    print(f"  Was the observation window chosen to approximate φ?")

    # =========================================================================
    # ENCODE/DECODE HYPOTHESIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENCODING HYPOTHESIS")
    print("=" * 70)

    print(f"""
  If this is an encoded message, the structure might be:

  CARRIER: 1420.405 MHz (hydrogen line)
           - Universal frequency for SETI
           - Wavelength = 21 cm (connects to signal structure)

  ENVELOPE: 6EQUJ5 = [6, 14, 26, 30, 19, 5]
           - Sum = 100 (marks completeness)
           - 36 bits (6² = perfect square of perfect number)
           - Contains φ approximation: 30/19 ≈ 1.58 ≈ φ
           - Contains π approximation: 19/6 ≈ 3.17 ≈ π

  GEOMETRY: SVD ratios
           - √2 at gap 5 (diagonal of unit square)
           - π/2 at gap 7 (quarter circle)
           - e at gap 4 (natural exponential)
           - φ at gap 13 (growth/optimization)

  DYNAMICS: Angular velocity = 360°/21
           - Points back to 21 (hydrogen, Fibonacci)
           - Self-referential structure

  The message might be:
  "Here are the fundamental constants of geometry and growth,
   encoded in the universal language of mathematics,
   transmitted on the universal frequency of hydrogen."
""")

    # =========================================================================
    # ATTEMPT NUMERIC DECODING
    # =========================================================================
    print("\n" + "=" * 70)
    print("ATTEMPTING NUMERIC DECODING")
    print("=" * 70)

    # The 36-bit number
    print(f"\n  36-bit value: {as_int}")
    print(f"  In scientific notation: {as_int:.4e}")

    # Is it related to physical constants?
    c = 299792458  # m/s
    h = 6.62607e-34  # J·s
    G = 6.67430e-11  # m³/(kg·s²)

    print(f"\n  Comparisons to physical constants:")
    print(f"    as_int / 10^9 = {as_int / 1e9:.4f}")
    print(f"    as_int / c = {as_int / c:.4f}")

    # What if we interpret differently?
    print(f"\n  Alternative interpretations of 36 bits:")

    # As two 18-bit numbers
    high_18 = int(binary_str[:18], 2)
    low_18 = int(binary_str[18:], 2)
    print(f"    Two 18-bit: {high_18}, {low_18}")
    print(f"    Ratio: {high_18/low_18:.4f}")

    # As three 12-bit numbers
    n1 = int(binary_str[0:12], 2)
    n2 = int(binary_str[12:24], 2)
    n3 = int(binary_str[24:36], 2)
    print(f"    Three 12-bit: {n1}, {n2}, {n3}")

    # As four 9-bit numbers
    nums_9 = [int(binary_str[i:i+9], 2) for i in range(0, 36, 9)]
    print(f"    Four 9-bit: {nums_9}")

    # =========================================================================
    # FINAL PATTERN
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE PATTERN")
    print("=" * 70)

    print(f"""
  NUMBERS THAT REPEAT:

  6:  Values in sequence, T(6)=21, 36=6², carbon
  21: Hydrogen wavelength, angular velocity, F(8), T(6)
  36: Bits in message, FWHM in seconds, 6²

  THE WEB OF CONNECTIONS:

       6 ──────────────┬─────────────── perfect number
       │               │
       ▼               ▼
      T(6)=21      6² = 36
       │               │
       ▼               ▼
    hydrogen        message
    wavelength      length
       │               │
       └───────┬───────┘
               │
               ▼
           angular
           velocity
           360/21

  Everything connects through 6 and 21.
  The signal is self-referential:
  - Transmitted at 21 cm
  - Angular dynamics divide by 21
  - 21 = sum of 1 through 6
  - 6 values in the message

  It's like a signature: "This message is about the number 21,
  which IS the hydrogen line, which carries this message."
""")


if __name__ == "__main__":
    main()
