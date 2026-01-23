#!/usr/bin/env python3
"""
THE SPEED OF LIGHT CONNECTION

6684271813 / c = 22.296 ≈ 21

Is the signal encoding a relationship to the speed of light?

Usage:
    python wow_speed_of_light.py
"""

from __future__ import annotations

import numpy as np

phi = (1 + np.sqrt(5)) / 2
pi = np.pi
c = 299792458  # m/s
h = 6.62607015e-34  # Planck constant
hbar = h / (2 * np.pi)


def main():
    print("=" * 70)
    print("THE SPEED OF LIGHT CONNECTION")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]
    binary_str = ''.join(f'{v:06b}' for v in seq)
    n = int(binary_str, 2)

    print(f"\n  36-bit encoding: {n}")
    print(f"  Speed of light c = {c} m/s")
    print(f"\n  n / c = {n / c:.6f}")

    # This is remarkably close to 21!
    ratio = n / c
    print(f"\n  Comparison to 21:")
    print(f"    n / c = {ratio:.6f}")
    print(f"    21 = 21.000000")
    print(f"    Difference: {ratio - 21:.6f}")
    print(f"    Relative error: {abs(ratio - 21) / 21 * 100:.2f}%")

    # What if the signal encodes 21 × c?
    print("\n" + "=" * 70)
    print("HYPOTHESIS: n = 21 × c × k for some k")
    print("=" * 70)

    k = n / (21 * c)
    print(f"\n  n / (21 × c) = {k:.6f}")
    print(f"  This k is close to: {round(k, 2)}")

    # Is k related to anything?
    print(f"\n  k comparisons:")
    print(f"    k × 10 = {k * 10:.4f}")
    print(f"    1/k = {1/k:.4f}")
    print(f"    k × φ = {k * phi:.4f}")
    print(f"    k × π = {k * pi:.4f}")

    # What value would give EXACTLY 21?
    print("\n" + "=" * 70)
    print("WHAT WOULD GIVE EXACTLY 21?")
    print("=" * 70)

    target = 21 * c
    print(f"\n  21 × c = {target:.0f}")
    print(f"  Our n = {n}")
    print(f"  Difference = {n - target:.0f}")

    # What sequence would give exactly 21 × c?
    target_binary = bin(int(target))[2:].zfill(36)
    print(f"\n  21 × c in binary (36 bits): {target_binary}")

    # Decode back to sequence
    target_seq = []
    for i in range(0, 36, 6):
        target_seq.append(int(target_binary[i:i+6], 2))
    print(f"  Would require sequence: {target_seq}")

    # Compare
    print(f"\n  Actual sequence:   {seq}")
    print(f"  21×c would need:   {target_seq}")
    print(f"  Differences:       {[a - b for a, b in zip(seq, target_seq)]}")

    # The hydrogen line connection
    print("\n" + "=" * 70)
    print("THE HYDROGEN LINE CONNECTION")
    print("=" * 70)

    freq_H = 1420.405751e6  # Hz (hydrogen hyperfine)
    wavelength_H = c / freq_H  # meters

    print(f"\n  Hydrogen line frequency: {freq_H/1e6:.6f} MHz")
    print(f"  Hydrogen wavelength: {wavelength_H * 100:.2f} cm = 21.1 cm")

    # Is there a relationship?
    print(f"\n  n / freq_H = {n / freq_H:.6f}")
    print(f"  n × wavelength_H = {n * wavelength_H:.6f}")
    print(f"  n / (c / 21) = {n / (c / 21):.6f}")

    # Planck units
    print("\n" + "=" * 70)
    print("PLANCK SCALE CONNECTIONS")
    print("=" * 70)

    print(f"\n  n × h = {n * h:.6e} J·s")
    print(f"  n × hbar = {n * hbar:.6e} J·s")
    print(f"  n / h = {n / h:.6e} Hz")

    # Time interpretation
    print(f"\n  If n represents time in some unit:")
    print(f"    n nanoseconds = {n * 1e-9:.4f} seconds ≈ 6.68 seconds")
    print(f"    n microseconds = {n * 1e-6:.4f} seconds ≈ 6684 seconds ≈ 1.86 hours")

    # Distance interpretation
    print(f"\n  If n represents distance:")
    print(f"    n meters = {n:.0f} m = {n/1000:.0f} km")
    print(f"    n × c = {n * c:.2e} (n light-seconds in meters)")

    # The 22 connection
    print("\n" + "=" * 70)
    print("THE NUMBER 22")
    print("=" * 70)

    print(f"\n  n / c ≈ 22.3")
    print(f"\n  22 is interesting:")
    print(f"    22 = 2 × 11 (semiprime)")
    print(f"    22/7 = {22/7:.6f} ≈ π (classic approximation)")
    print(f"    22 = number of letters in Hebrew alphabet")
    print(f"    22 = atomic number of Titanium")

    print(f"\n  If the message encodes '22':")
    print(f"    22 × 7 = 154 (if pointing to π ≈ 22/7)")
    print(f"    22 - 21 = 1 (off by 1 from hydrogen)")

    # More precise analysis
    print("\n" + "=" * 70)
    print("PRECISE ANALYSIS")
    print("=" * 70)

    # n/c = 22.296331
    # What is 22.296331?

    x = n / c
    print(f"\n  x = n/c = {x:.10f}")

    # Try to express as simple fraction
    from fractions import Fraction
    frac = Fraction(n, c).limit_denominator(1000)
    print(f"  As fraction (limit 1000): {frac} = {float(frac):.6f}")

    frac2 = Fraction(n, c).limit_denominator(100)
    print(f"  As fraction (limit 100): {frac2} = {float(frac2):.6f}")

    # Express in terms of constants
    print(f"\n  x in terms of constants:")
    print(f"    x / φ = {x / phi:.6f}")
    print(f"    x / π = {x / pi:.6f}")
    print(f"    x / e = {x / np.e:.6f}")
    print(f"    x - 21 = {x - 21:.6f}")
    print(f"    (x - 21) × φ = {(x - 21) * phi:.6f}")
    print(f"    (x - 21) × π = {(x - 21) * pi:.6f}")

    # What if x = 21 + φ/something?
    diff = x - 21
    print(f"\n  x - 21 = {diff:.6f}")
    print(f"  φ / (x - 21) = {phi / diff:.6f}")
    print(f"  (x - 21) / φ = {diff / phi:.6f}")
    print(f"  (x - 21) × 5 = {diff * 5:.6f}")

    # SYNTHESIS
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    print(f"""
  THE CONNECTION:

  n = 6684271813 (the 36-bit encoding of 6EQUJ5)
  c = 299792458 (speed of light in m/s)
  n / c = 22.296

  This is approximately 21 + 1.3

  21 appears everywhere in this signal:
  - Hydrogen wavelength: 21 cm
  - Angular velocity: 360°/21
  - Triangular number: T(6) = 21
  - Fibonacci: F(8) = 21

  The deviation from 21:
  - n/c - 21 = 1.296
  - 1.296 ≈ 1.3 ≈ 4/3 ≈ 1/φ² + something

  INTERPRETATION:

  If intentional, this could mean:
  "The encoded value divided by the speed of light
   gives (approximately) the hydrogen wavelength in cm."

  It's like a self-referential signature:
  - Transmitted on 21 cm
  - Encodes a number related to 21 × c
  - Contains 21 in its angular dynamics

  The ~6% discrepancy (22.3 vs 21) might be:
  - Noise/imprecision in the signal
  - An additional encoded value
  - A pointer to something else (22/7 ≈ π?)
  - Or coincidence

  But the near-miss to such a fundamental relationship
  (encoding × wavelength = number of bits × c) is striking.
""")


if __name__ == "__main__":
    main()
