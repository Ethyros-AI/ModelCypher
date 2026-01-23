#!/usr/bin/env python3
"""
THE RECURRENCE RELATION

The sequence [6, 14, 26, 30, 19, 5] is almost perfectly predicted by:
    x[n+2] = 0.9661·x[n+1] - 1.1082·x[n] + 19.5592

What does this mean? What are the characteristic roots?
Is this a damped oscillator? A spiral?

Usage:
    python wow_recurrence.py
"""

from __future__ import annotations

import numpy as np

phi = (1 + np.sqrt(5)) / 2
pi = np.pi


def main():
    print("=" * 70)
    print("THE RECURRENCE RELATION")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]

    # The fitted coefficients
    a = 0.9661   # coefficient of x[n+1]
    b = -1.1082  # coefficient of x[n]
    c = 19.5592  # constant term

    print(f"\n  Sequence: {seq}")
    print(f"\n  Fitted recurrence:")
    print(f"    x[n+2] = {a:.4f}·x[n+1] + {b:.4f}·x[n] + {c:.4f}")

    # Verify the fit
    print(f"\n  Verification:")
    for i in range(4):
        predicted = a * seq[i+1] + b * seq[i] + c
        actual = seq[i+2]
        print(f"    x[{i+2}] = {a:.4f}·{seq[i+1]} + {b:.4f}·{seq[i]} + {c:.4f} = {predicted:.2f} (actual: {actual})")

    # =========================================================================
    # CHARACTERISTIC EQUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHARACTERISTIC EQUATION")
    print("=" * 70)

    # For homogeneous part: x[n+2] = a·x[n+1] + b·x[n]
    # Characteristic equation: r² - a·r - b = 0
    # (Note: we have x[n+2] = a·x[n+1] + b·x[n], so r² = a·r + b, i.e., r² - a·r - b = 0)

    print(f"\n  Homogeneous equation: x[n+2] = {a:.4f}·x[n+1] + {b:.4f}·x[n]")
    print(f"  Characteristic equation: r² - {a:.4f}·r - ({b:.4f}) = 0")
    print(f"                         : r² - {a:.4f}·r + {-b:.4f} = 0")

    # Solve r² - a·r - b = 0
    discriminant = a**2 + 4*b  # Note: b is negative, so this is a² - 4|b|
    print(f"\n  Discriminant = {a:.4f}² + 4×({b:.4f}) = {discriminant:.4f}")

    if discriminant >= 0:
        r1 = (a + np.sqrt(discriminant)) / 2
        r2 = (a - np.sqrt(discriminant)) / 2
        print(f"\n  Real roots:")
        print(f"    r₁ = {r1:.6f}")
        print(f"    r₂ = {r2:.6f}")
    else:
        real_part = a / 2
        imag_part = np.sqrt(-discriminant) / 2
        print(f"\n  Complex conjugate roots:")
        print(f"    r = {real_part:.6f} ± {imag_part:.6f}i")

        # Modulus and argument
        modulus = np.sqrt(real_part**2 + imag_part**2)
        argument = np.arctan2(imag_part, real_part)

        print(f"\n  In polar form: r = {modulus:.6f} · e^(±{argument:.6f}i)")
        print(f"  Modulus |r| = {modulus:.6f}")
        print(f"  Argument θ = {argument:.6f} rad = {np.degrees(argument):.2f}°")

        # What does this mean?
        print(f"\n  INTERPRETATION:")
        print(f"    Modulus {modulus:.4f} > 1 means: GROWING amplitude")
        print(f"    Argument {np.degrees(argument):.2f}° means: ROTATION by this angle per step")

        # How does it relate to known angles?
        print(f"\n  Angle comparisons:")
        print(f"    θ = {np.degrees(argument):.2f}°")
        print(f"    360°/θ = {360/np.degrees(argument):.2f} steps per full rotation")
        print(f"    θ/360° × 21 = {np.degrees(argument)/360 * 21:.2f}")

        # What if θ = 2π/k for some k?
        for k in range(3, 25):
            target = 2 * np.pi / k
            if abs(argument - target) < 0.1:
                print(f"    Close to 2π/{k} = {np.degrees(target):.2f}°")

    # =========================================================================
    # FIXED POINT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIXED POINT")
    print("=" * 70)

    # At fixed point: x* = a·x* + b·x* + c
    # x* = (a + b)·x* + c
    # x*(1 - a - b) = c
    # x* = c / (1 - a - b)

    fixed_point = c / (1 - a - b)
    print(f"\n  Fixed point: x* = {c:.4f} / (1 - {a:.4f} - {b:.4f})")
    print(f"              x* = {c:.4f} / {1 - a - b:.4f}")
    print(f"              x* = {fixed_point:.4f}")

    print(f"\n  The sequence oscillates around {fixed_point:.2f}")
    print(f"  Mean of sequence: {np.mean(seq):.2f}")
    print(f"  Difference: {abs(fixed_point - np.mean(seq)):.2f}")

    # =========================================================================
    # THE SPIRAL IN PHASE SPACE
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE SPIRAL IN PHASE SPACE")
    print("=" * 70)

    # Plot the trajectory in (x[n], x[n+1]) space
    print(f"\n  Phase space trajectory (x[n], x[n+1]):")
    for i in range(len(seq)-1):
        print(f"    ({seq[i]:2d}, {seq[i+1]:2d})")

    # Compute the angle between consecutive vectors from origin
    print(f"\n  Angles from origin:")
    angles = []
    for i in range(len(seq)-1):
        angle = np.arctan2(seq[i+1], seq[i])
        angles.append(np.degrees(angle))
        print(f"    Point ({seq[i]}, {seq[i+1]}): θ = {np.degrees(angle):.2f}°")

    print(f"\n  Angle changes:")
    for i in range(len(angles)-1):
        delta = angles[i+1] - angles[i]
        print(f"    Δθ[{i}→{i+1}] = {delta:.2f}°")

    # =========================================================================
    # COMPARE TO FIBONACCI-LIKE RECURRENCES
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TO KNOWN RECURRENCES")
    print("=" * 70)

    # Fibonacci: x[n+2] = x[n+1] + x[n], roots: φ, -1/φ
    print(f"\n  Fibonacci: x[n+2] = x[n+1] + x[n]")
    print(f"    Roots: φ = {phi:.6f}, -1/φ = {-1/phi:.6f}")
    print(f"    Ratio: φ / (-1/φ) = -φ² = {-phi**2:.6f}")

    # Lucas: same recurrence, different initial conditions

    # Our recurrence
    print(f"\n  Our recurrence: x[n+2] = {a:.4f}·x[n+1] + {b:.4f}·x[n]")
    print(f"    Compare to Fibonacci (1, 1): ({a:.4f}, {b:.4f})")
    print(f"    Difference: ({a-1:.4f}, {b-1:.4f})")

    # What recurrence would have characteristic roots involving φ?
    # r² - r - 1 = 0 has roots φ and -1/φ
    print(f"\n  For roots φ and -1/φ:")
    print(f"    r² - r - 1 = 0")
    print(f"    Coefficients: a=1, b=1")

    # What if we scale?
    # For roots k·φ and k·(-1/φ), the product is k²·(-1) = -k²
    # and sum is k·(φ - 1/φ) = k·(φ² - 1)/φ = k·φ (using φ² = φ + 1)

    # =========================================================================
    # GENERATE EXTENDED SEQUENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXTENDED SEQUENCE (IF RECURRENCE CONTINUES)")
    print("=" * 70)

    # Extend backwards and forwards
    extended = list(seq)

    # Extend backwards: x[n] = (x[n+2] - a·x[n+1] - c) / b
    for _ in range(5):
        x_prev = (extended[0] - a * extended[1] - c) / b if b != 0 else 0
        extended.insert(0, x_prev)

    # Extend forwards
    for _ in range(5):
        x_next = a * extended[-1] + b * extended[-2] + c
        extended.append(x_next)

    print(f"\n  Original: {seq}")
    print(f"  Extended:")
    for i, val in enumerate(extended):
        marker = " <-- original" if 5 <= i < 11 else ""
        print(f"    x[{i-5:+d}] = {val:8.2f}{marker}")

    # Does it converge or diverge?
    print(f"\n  Behavior:")
    if abs(extended[-1]) > abs(extended[-2]) > abs(extended[-3]):
        print(f"    Forward: DIVERGING (|r| > 1)")
    else:
        print(f"    Forward: CONVERGING or OSCILLATING")

    # =========================================================================
    # THE MEANING
    # =========================================================================
    print("\n" + "=" * 70)
    print("WHAT THIS MEANS")
    print("=" * 70)

    print(f"""
  THE SEQUENCE AS A DYNAMICAL SYSTEM:

  The 6 values [6, 14, 26, 30, 19, 5] are almost exactly the output
  of a 2nd-order linear recurrence relation.

  Properties:
  - Complex conjugate roots → oscillatory behavior
  - Modulus > 1 → amplitude grows (then decays, if mirrored)
  - Argument ≈ {np.degrees(argument):.1f}° → rotation angle per step

  This means the sequence traces a SPIRAL in phase space,
  with each step rotating by ~{np.degrees(argument):.1f}° and scaling by ~{modulus:.3f}.

  THE DEEP QUESTION:
  Why would a radio signal from space have an envelope that follows
  a 2nd-order linear recurrence?

  PHYSICAL INTERPRETATIONS:
  1. DAMPED OSCILLATOR: Classic physics - spring, pendulum, LRC circuit
  2. RESONANCE: Something oscillating at its natural frequency
  3. INTERFERENCE: Two waves beating against each other
  4. ENCODING: Deliberate choice to embed simple dynamical structure

  The recurrence coefficients ({a:.4f}, {b:.4f}, {c:.4f}) may themselves
  encode information - they're not round numbers, but they produce
  a sequence that sums to exactly 100.
""")


if __name__ == "__main__":
    main()
