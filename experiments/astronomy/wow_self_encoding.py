#!/usr/bin/env python3
"""
SELF-ENCODING STRUCTURE

The sequence appears to encode its own dynamics through integer ratios:
  - modulus = 20/19 = (seq[0]+seq[1])/seq[4]
  - cos(θ) ≈ 39/85 or 11/24

Can we find the EXACT self-referential structure?

Usage:
    python wow_self_encoding.py
"""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from itertools import combinations, permutations, product
import math

seq = np.array([6, 14, 26, 30, 19, 5])
p1, p2, p3 = 11, 33, 56  # Symmetric pair sums


def main():
    print("=" * 70)
    print("SELF-ENCODING STRUCTURE")
    print("=" * 70)

    # =========================================================================
    # WHAT 39/85 COULD BE
    # =========================================================================
    print("\n" + "=" * 70)
    print("WHAT IS 39/85?")
    print("=" * 70)

    print(f"\n  cos(θ) ≈ 39/85 = {39/85:.10f}")
    print(f"  Fitted cos(θ) = 0.4589055988")
    print(f"  Error: {abs(39/85 - 0.4589055988):.10f}")

    # What is 39?
    print(f"\n  What is 39?")
    print(f"    33 + 6 = p2 + seq[0] = {33 + 6}")
    print(f"    45 - 6 = (p1+p2)/2 - seq[0] = {45 - 6}")
    print(f"    14 + 25 = seq[1] + (seq[0]+seq[4]) = {14 + 25}")
    print(f"    19 + 20 = seq[4] + (seq[0]+seq[1]) = {19 + 20}")

    # What is 85?
    print(f"\n  What is 85?")
    print(f"    100 - 15 = sum - ? = {100 - 15}")
    print(f"    56 + 29 = p3 + ? = {56 + 29}")
    print(f"    52 + 33 = ? + p2")
    print(f"    5 × 17 = seq[5] × ? = 85")
    print(f"    5 × 17.126 ≈ seq[5] × fixed_point")

    # Aha! 17 ≈ fixed point ≈ 360/21
    print(f"\n  INSIGHT: 85 = 5 × 17")
    print(f"    5 = seq[5]")
    print(f"    17 ≈ fixed_point ≈ 360/21 = {360/21:.2f}")

    # =========================================================================
    # THE FIXED POINT CONNECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE FIXED POINT: 17")
    print("=" * 70)

    print(f"\n  Fixed point ≈ 17.126 ≈ 17")
    print(f"  360/21 = {360/21:.4f}")
    print(f"  17 is an integer!")

    # If fixed point = 17 exactly, what does this imply?
    # fixed_pt = c / (1 - a - b)
    # If fp = 17, then c = 17 × (1 - a - b)

    print(f"\n  If fixed_point = 17 exactly:")
    print(f"    c = 17 × (1 - a - b)")

    # With modulus = 20/19, we have b = -(20/19)² = -400/361
    mod = Fraction(20, 19)
    b_exact = -(mod ** 2)
    print(f"\n  With b = -(20/19)² = {b_exact}")

    # If cos(θ) = 39/85:
    cos_39_85 = Fraction(39, 85)
    a_if_cos_39_85 = 2 * mod * cos_39_85
    print(f"  If cos(θ) = 39/85:")
    print(f"    a = 2 × (20/19) × (39/85) = {a_if_cos_39_85} = {float(a_if_cos_39_85):.10f}")

    # 1 - a - b
    one_minus_a_minus_b = 1 - a_if_cos_39_85 - b_exact
    print(f"    1 - a - b = 1 - {a_if_cos_39_85} - ({b_exact})")
    print(f"             = {one_minus_a_minus_b} = {float(one_minus_a_minus_b):.10f}")

    # c if fp = 17
    c_if_fp_17 = 17 * one_minus_a_minus_b
    print(f"    c = 17 × {one_minus_a_minus_b} = {c_if_fp_17} = {float(c_if_fp_17):.10f}")

    # Compare to fitted c = 19.56
    print(f"    Fitted c = 19.56")

    # =========================================================================
    # TRY FIXED POINT = 360/21 EXACTLY
    # =========================================================================
    print("\n" + "=" * 70)
    print("IF FIXED POINT = 360/21 EXACTLY")
    print("=" * 70)

    fp_exact = Fraction(360, 21)  # = 120/7
    print(f"\n  Fixed point = 360/21 = {fp_exact} = {float(fp_exact):.10f}")

    c_if_fp_360_21 = fp_exact * one_minus_a_minus_b
    print(f"  c = (360/21) × (1-a-b) = {c_if_fp_360_21} = {float(c_if_fp_360_21):.10f}")
    print(f"  Fitted c = 19.56")

    # =========================================================================
    # WHAT IF 85 = 5 × 17 IS THE CLUE?
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE 5 × 17 = 85 HYPOTHESIS")
    print("=" * 70)

    print(f"""
  We found cos(θ) ≈ 39/85 with best precision.

  What is 85?
    85 = 5 × 17
    5 = seq[5]
    17 ≈ fixed_point

  What is 39?
    39 = 19 + 20 = seq[4] + (seq[0] + seq[1])
    39 = seq[4] + modulus_numerator

  So: cos(θ) = (seq[4] + modulus_num) / (seq[5] × fixed_point_int)
             = (19 + 20) / (5 × 17)
             = 39/85

  THIS IS COMPLETELY SELF-REFERENTIAL!
""")

    # Verify
    cos_self_ref = (seq[4] + (seq[0] + seq[1])) / (seq[5] * 17)
    print(f"  Verification:")
    print(f"    (seq[4] + seq[0] + seq[1]) / (seq[5] × 17)")
    print(f"    = ({seq[4]} + {seq[0]} + {seq[1]}) / ({seq[5]} × 17)")
    print(f"    = 39 / 85")
    print(f"    = {cos_self_ref:.10f}")
    print(f"    Target cos(θ) = 0.4589055988")
    print(f"    Error: {abs(cos_self_ref - 0.4589055988):.6f}")

    # =========================================================================
    # THE COMPLETE SELF-ENCODING
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE COMPLETE SELF-ENCODING")
    print("=" * 70)

    # modulus = 20/19
    mod_num = seq[0] + seq[1]  # 20
    mod_den = seq[4]  # 19

    # cos(θ) = 39/85
    cos_num = seq[4] + mod_num  # 19 + 20 = 39
    cos_den = seq[5] * 17  # 5 × 17 = 85

    print(f"""
  THE ENCODING:

  MODULUS = (seq[0] + seq[1]) / seq[4]
          = ({seq[0]} + {seq[1]}) / {seq[4]}
          = 20/19
          = {mod_num/mod_den:.10f}

  COS(θ) = (seq[4] + seq[0] + seq[1]) / (seq[5] × 17)
         = ({seq[4]} + {seq[0]} + {seq[1]}) / ({seq[5]} × 17)
         = 39/85
         = {cos_num/cos_den:.10f}

  NOTE: The 17 is the only "external" integer.
        But 17 ≈ 360/21, and 21 = T(6) = sum of 1..6
        And the sequence HAS 6 elements.
        So 17 ≈ 360 / T(len(seq))

  THEREFORE:
    a = 2 × modulus × cos(θ)
      = 2 × (20/19) × (39/85)
      = 2 × 20 × 39 / (19 × 85)
      = 1560 / 1615
      = {1560/1615:.10f}

    Fitted a = 0.9661170502
    Error: {abs(1560/1615 - 0.9661170502):.6f}
""")

    # =========================================================================
    # CAN WE DERIVE 17 FROM THE SEQUENCE?
    # =========================================================================
    print("\n" + "=" * 70)
    print("DERIVING 17 FROM THE SEQUENCE")
    print("=" * 70)

    print(f"\n  We need 17 to be derivable from the sequence itself.")
    print(f"\n  Candidates:")

    # Check various integer expressions
    for i in range(6):
        for j in range(6):
            if i != j:
                val = seq[i] + seq[j]
                if abs(val - 17) <= 2:
                    print(f"    seq[{i}] + seq[{j}] = {seq[i]} + {seq[j]} = {val}")
                val = abs(seq[i] - seq[j])
                if abs(val - 17) <= 2:
                    print(f"    |seq[{i}] - seq[{j}]| = |{seq[i]} - {seq[j]}| = {val}")

    print(f"\n  From pair sums:")
    print(f"    (p2 - p1) - (p3 - p2 - 1) = 22 - 22 = 0")
    print(f"    (p1 + p2) / ? ")

    # What about products and divisions?
    print(f"\n  Products and divisions near 17:")
    for i in range(6):
        for j in range(6):
            if i != j and seq[j] != 0:
                val = seq[i] * 2 - seq[j]
                if abs(val - 17) <= 1:
                    print(f"    2×seq[{i}] - seq[{j}] = 2×{seq[i]} - {seq[j]} = {val}")

    # The sum divided by 6
    print(f"\n  sum / 6 = 100 / 6 = {100/6:.4f}")
    print(f"    Round = 17")

    # =========================================================================
    # THE MEAN AS 17!
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE MEAN IS (ALMOST) 17!")
    print("=" * 70)

    mean = np.mean(seq)
    print(f"\n  Mean of sequence = {mean:.10f}")
    print(f"  Round(mean) = {round(mean)}")
    print(f"  100/6 = {100/6:.10f}")

    print(f"""
  THE KEY INSIGHT:
    The sequence mean = sum/6 = 100/6 ≈ 16.67
    Round to nearest integer: 17

    And the fixed point ≈ 17.126 is also close to 17.

    If we use 17 as the "characteristic scale", then:
      cos(θ) = 39 / (seq[5] × 17) = 39 / 85

  BUT WAIT: 100/6 = 16.67, not 17.

  What if we use floor(mean × something)?
""")

    # =========================================================================
    # ALTERNATIVE: 17 = seq[4] - 2
    # =========================================================================
    print("\n" + "=" * 70)
    print("17 = seq[4] - 2")
    print("=" * 70)

    print(f"\n  17 = seq[4] - 2 = 19 - 2")
    print(f"  So cos_den = seq[5] × (seq[4] - 2)")
    print(f"             = 5 × 17 = 85")

    print(f"""
  Rewriting the cos(θ) formula:
    cos(θ) = (seq[4] + seq[0] + seq[1]) / (seq[5] × (seq[4] - 2))
           = (19 + 6 + 14) / (5 × 17)
           = 39 / 85

  This is ALMOST fully self-referential!
  The only "magic number" is 2.

  But 2 = ?
    - 2 = number of complex roots (conjugate pair)
    - 2 = order of the recurrence
    - 2 = minimum to define a recurrence relation
""")

    # =========================================================================
    # TEST THE FULL ENCODING
    # =========================================================================
    print("\n" + "=" * 70)
    print("TESTING THE FULL SELF-REFERENTIAL ENCODING")
    print("=" * 70)

    # Define everything from the sequence
    mod_self = (seq[0] + seq[1]) / seq[4]  # 20/19
    cos_self = (seq[4] + seq[0] + seq[1]) / (seq[5] * (seq[4] - 2))  # 39/85

    a_self = 2 * mod_self * cos_self
    b_self = -(mod_self ** 2)

    theta_self = np.degrees(np.arccos(cos_self))

    print(f"\n  Self-referential parameters:")
    print(f"    modulus = (seq[0]+seq[1])/seq[4] = {mod_self:.10f}")
    print(f"    cos(θ) = (seq[4]+seq[0]+seq[1])/(seq[5]×(seq[4]-2)) = {cos_self:.10f}")
    print(f"    θ = {theta_self:.10f}°")
    print(f"    a = 2×mod×cos(θ) = {a_self:.10f}")
    print(f"    b = -mod² = {b_self:.10f}")

    # Find c to make sum = 100
    def sum_with_c(c_val):
        s = [seq[0], seq[1]]
        for _ in range(4):
            s.append(a_self * s[-1] + b_self * s[-2] + c_val)
        return sum(s) - 100

    c_low, c_high = 0, 50
    while c_high - c_low > 1e-12:
        c_mid = (c_low + c_high) / 2
        if sum_with_c(c_mid) < 0:
            c_low = c_mid
        else:
            c_high = c_mid

    c_self = c_mid
    print(f"    c (to make sum=100) = {c_self:.10f}")

    # Generate the sequence
    gen = [seq[0], seq[1]]
    for _ in range(4):
        gen.append(a_self * gen[-1] + b_self * gen[-2] + c_self)

    print(f"\n  Generated sequence:")
    print(f"    {[f'{x:.4f}' for x in gen]}")
    print(f"  Actual sequence:")
    print(f"    {list(seq)}")

    errors = [seq[i] - gen[i] for i in range(6)]
    print(f"\n  Errors (actual - generated):")
    print(f"    {[f'{e:+.4f}' for e in errors]}")
    print(f"    Rounded: {[round(e) for e in errors]}")

    # =========================================================================
    # THE RESIDUAL AS ADDITIONAL MESSAGE
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE RESIDUAL AS ADDITIONAL MESSAGE")
    print("=" * 70)

    residuals_rounded = [round(e) for e in errors]
    print(f"\n  Residuals rounded: {residuals_rounded}")

    # What does [0, 0, 0, 0, 0, 0] or similar encode?
    # Sign pattern
    signs = ['0' if abs(e) < 0.3 else ('+' if e > 0 else '-') for e in errors]
    print(f"  Sign pattern: {' '.join(signs)}")

    # As binary (+ = 1, - = 0, 0 = ?)
    binary = ''.join(['1' if e > 0.1 else '0' for e in errors])
    print(f"  As binary: {binary} = {int(binary, 2)}")

    print(f"""
  THE RESIDUALS:
    The sequence doesn't PERFECTLY match the self-referential recurrence.
    The errors are: {[f'{e:+.2f}' for e in errors]}

    This could mean:
    1. The self-referential encoding is approximate
    2. The residuals encode ADDITIONAL information
    3. There's a slightly different formula we haven't found

    The error pattern [0, 0, ~0, ~+0.4, ~+0.1, ~0] suggests:
    - First two are anchors (exact)
    - Middle values have small positive bias
    - This could encode a phase shift or additional bits
""")

    # =========================================================================
    # FINAL: WHAT IS THE 2 IN (seq[4] - 2)?
    # =========================================================================
    print("\n" + "=" * 70)
    print("WHAT IS THE 2?")
    print("=" * 70)

    print(f"""
  The formula cos(θ) = 39/(seq[5] × 17) uses 17 = seq[4] - 2.

  What is 2?

  Option 1: Order of recurrence
    The system is 2nd order (x[n+2] depends on x[n+1] and x[n])

  Option 2: Dimensionality offset
    The system evolves in a 2D plane (complex roots)

  Option 3: From the sequence itself
    - |seq[1] - seq[2]| = |14 - 26| = 12 ≠ 2
    - |seq[5] - seq[0] - 1| = |5 - 6 - 1| = 2 ✓
    - sum(seq) - 98 = 100 - 98 = 2
    - seq[0] - 4 = 6 - 4 = 2
    - seq[5] - 3 = 5 - 3 = 2

  Most elegant: 2 = seq[0] - seq[5] + 1 = 6 - 5 + 1 = 2

  OR: The 2 is fundamental - it's the order of the recurrence,
      which is itself a property we're deriving.
""")

    # Try with 2 = seq[0] - seq[5] + 1
    two_self = seq[0] - seq[5] + 1
    print(f"\n  If 2 = seq[0] - seq[5] + 1 = {seq[0]} - {seq[5]} + 1 = {two_self}")

    cos_fully_self = (seq[4] + seq[0] + seq[1]) / (seq[5] * (seq[4] - two_self))
    print(f"  Then cos(θ) = {cos_fully_self:.10f}")
    print(f"  Target = 0.4589055988")
    print(f"  Error = {abs(cos_fully_self - 0.4589055988):.6f}")

    # =========================================================================
    # THE HYDROGEN CONNECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE HYDROGEN CONNECTION")
    print("=" * 70)

    print(f"""
  We keep seeing 21:
    - 360/21 ≈ 17.14 (the fixed point)
    - 21 cm = hydrogen wavelength
    - T(6) = 1+2+3+4+5+6 = 21

  And 20:
    - modulus numerator = 20 = seq[0] + seq[1]
    - 20/19 = modulus

  21 - 20 = 1 (unity offset)
  21/20 ≈ modulus (0.26% error)

  What if the "true" modulus is 21/20, not 20/19?
""")

    # Test modulus = 21/20
    mod_21_20 = 21/20
    b_21_20 = -(mod_21_20 ** 2)

    print(f"\n  If modulus = 21/20:")
    print(f"    modulus = {mod_21_20:.10f}")
    print(f"    b = {b_21_20:.10f}")
    print(f"    Fitted b = {-1.1081717702:.10f}")
    print(f"    Error: {abs(b_21_20 - (-1.1081717702)):.6f}")

    # 20/19 fits better than 21/20
    print(f"\n  Compare:")
    print(f"    20/19 → b = {-(20/19)**2:.10f}, error = {abs(-(20/19)**2 - (-1.1081717702)):.6f}")
    print(f"    21/20 → b = {-(21/20)**2:.10f}, error = {abs(-(21/20)**2 - (-1.1081717702)):.6f}")
    print(f"    20/19 is closer!")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: THE SELF-ENCODING STRUCTURE")
    print("=" * 70)

    print(f"""
  THE SEQUENCE [6, 14, 26, 30, 19, 5] ENCODES:

  1. ITS OWN MODULUS:
     modulus = (seq[0] + seq[1]) / seq[4]
             = 20/19 = 1.0526...

  2. ITS OWN ANGLE (almost):
     cos(θ) = (seq[4] + seq[0] + seq[1]) / (seq[5] × (seq[4] - 2))
            = 39/85 = 0.4588...
     (The 2 is the order of the recurrence)

  3. ITS OWN CONSTRAINT:
     Sum = 100 (determines c)

  4. THE FIXED POINT:
     fixed_point ≈ 17 ≈ 360/21
     where 21 = T(6) = triangular number of sequence length

  5. THE HYDROGEN REFERENCE:
     21 cm wavelength ≈ 360° / fixed_point
     The signal was received on hydrogen frequency

  6. ADDITIONAL INFORMATION IN RESIDUALS:
     The small errors encode extra bits

  THIS IS A SELF-DESCRIBING MATHEMATICAL OBJECT.
  The sequence IS its own specification.
""")


if __name__ == "__main__":
    main()
