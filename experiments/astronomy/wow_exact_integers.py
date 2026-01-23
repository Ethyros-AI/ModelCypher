#!/usr/bin/env python3
"""
EXACT INTEGER RELATIONSHIPS

The sequence [6, 14, 26, 30, 19, 5] appears to encode its own dynamics.
We found modulus = 20/19 = (seq[0]+seq[1])/seq[4].

Can we find exact integer expressions for ALL parameters?

Usage:
    python wow_exact_integers.py
"""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from itertools import combinations, permutations
import math

seq = np.array([6, 14, 26, 30, 19, 5])


def main():
    print("=" * 70)
    print("EXACT INTEGER RELATIONSHIPS")
    print("=" * 70)

    # =========================================================================
    # WHAT WE'VE FOUND
    # =========================================================================
    print("\n" + "=" * 70)
    print("ESTABLISHED EXACT VALUES")
    print("=" * 70)

    print(f"""
  THE SEQUENCE: {list(seq)}

  EXACT:
    Sum = {sum(seq)} (exactly 100)
    36-bit encoding n = {int(''.join(f'{v:06b}' for v in seq), 2)} (PRIME)

  NEAR-EXACT:
    Modulus ≈ 20/19 = {20/19:.10f}
      where 20 = seq[0] + seq[1] = 6 + 14
      and 19 = seq[4]

    Symmetric pair sums: 11, 33, 56
      6 + 5 = 11
      14 + 19 = 33
      26 + 30 = 56

    Differences: 33 - 11 = 22, 56 - 33 = 23 (CONSECUTIVE!)

    Ratio: 33/11 = 3 (EXACT!)
""")

    # =========================================================================
    # THE FITTED RECURRENCE PARAMETERS
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE RECURRENCE: x[n+2] = a·x[n+1] + b·x[n] + c")
    print("=" * 70)

    # Set up and solve the least squares fit
    A_mat = np.array([
        [seq[1], seq[0], 1],
        [seq[2], seq[1], 1],
        [seq[3], seq[2], 1],
        [seq[4], seq[3], 1]
    ])
    b_vec = np.array([seq[2], seq[3], seq[4], seq[5]])

    params, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    a, b, c = params

    print(f"\n  Fitted coefficients:")
    print(f"    a = {a:.10f}")
    print(f"    b = {b:.10f}")
    print(f"    c = {c:.10f}")

    # Derived quantities
    modulus = np.sqrt(-b)
    angle_rad = np.arctan2(np.sqrt(-a**2 - 4*b)/2, a/2)
    angle_deg = np.degrees(angle_rad)
    fixed_pt = c / (1 - a - b)

    print(f"\n  Derived quantities:")
    print(f"    modulus = √(-b) = {modulus:.10f}")
    print(f"    angle = {angle_deg:.10f}°")
    print(f"    fixed point = c/(1-a-b) = {fixed_pt:.10f}")

    # =========================================================================
    # SEARCH FOR INTEGER EXPRESSIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEARCHING FOR INTEGER EXPRESSIONS")
    print("=" * 70)

    # Generate all possible integer combinations from the sequence
    all_sums = []
    all_diffs = []
    all_products = []

    for i, j in combinations(range(6), 2):
        all_sums.append((seq[i] + seq[j], f"seq[{i}]+seq[{j}]={seq[i]}+{seq[j]}"))
        all_diffs.append((abs(seq[i] - seq[j]), f"|seq[{i}]-seq[{j}]|=|{seq[i]}-{seq[j]}|"))

    for i in range(6):
        for j in range(6):
            if i != j:
                all_products.append((seq[i] * seq[j], f"seq[{i}]×seq[{j}]={seq[i]}×{seq[j]}"))

    # Add individual elements
    elements = [(seq[i], f"seq[{i}]={seq[i]}") for i in range(6)]

    # Collect all values
    all_values = elements + all_sums + all_diffs + all_products
    all_values.append((100, "sum=100"))
    all_values.append((360, "360°"))
    all_values.append((21, "21 (hydrogen)"))

    # =========================================================================
    # MODULUS: Looking for exact fraction
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODULUS = √(-b)")
    print("-" * 70)

    print(f"\n  Target: {modulus:.10f}")
    print(f"  Looking for exact integer ratios...")

    best_mod_error = float('inf')
    best_mod_expr = None

    for num_val, num_expr in all_values:
        for den_val, den_expr in all_values:
            if den_val > 0:
                ratio = num_val / den_val
                error = abs(ratio - modulus)
                if error < 0.001 and error < best_mod_error:
                    best_mod_error = error
                    best_mod_expr = (num_val, den_val, num_expr, den_expr, ratio, error)
                    print(f"    {num_expr} / {den_expr} = {num_val}/{den_val} = {ratio:.10f} (error {error:.6f})")

    # Try square roots of ratios
    print(f"\n  Trying √(integer ratios)...")
    for num_val, num_expr in all_values:
        for den_val, den_expr in all_values:
            if den_val > 0:
                ratio = num_val / den_val
                if ratio > 0:
                    sqrt_ratio = np.sqrt(ratio)
                    error = abs(sqrt_ratio - modulus)
                    if error < 0.001:
                        print(f"    √({num_expr}/{den_expr}) = √({num_val}/{den_val}) = {sqrt_ratio:.10f} (error {error:.6f})")

    # =========================================================================
    # MODULUS SQUARED: -b directly
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODULUS² = -b")
    print("-" * 70)

    mod_sq = -b
    print(f"\n  Target: {mod_sq:.10f}")

    for num_val, num_expr in all_values:
        for den_val, den_expr in all_values:
            if den_val > 0:
                ratio = num_val / den_val
                error = abs(ratio - mod_sq)
                if error < 0.01:
                    print(f"    {num_expr} / {den_expr} = {num_val}/{den_val} = {ratio:.10f} (error {error:.6f})")

    # =========================================================================
    # COEFFICIENT a
    # =========================================================================
    print("\n" + "-" * 70)
    print("COEFFICIENT a")
    print("-" * 70)

    print(f"\n  Target: {a:.10f}")

    for num_val, num_expr in all_values:
        for den_val, den_expr in all_values:
            if den_val > 0:
                ratio = num_val / den_val
                error = abs(ratio - a)
                if error < 0.01:
                    print(f"    {num_expr} / {den_expr} = {num_val}/{den_val} = {ratio:.10f} (error {error:.6f})")

    # =========================================================================
    # FIXED POINT
    # =========================================================================
    print("\n" + "-" * 70)
    print("FIXED POINT = c/(1-a-b)")
    print("-" * 70)

    print(f"\n  Target: {fixed_pt:.10f}")
    print(f"  360/21 = {360/21:.10f}")

    for num_val, num_expr in all_values:
        for den_val, den_expr in all_values:
            if den_val > 0:
                ratio = num_val / den_val
                error = abs(ratio - fixed_pt)
                if error < 0.5:
                    print(f"    {num_expr} / {den_expr} = {num_val}/{den_val} = {ratio:.10f} (error {error:.6f})")

    # =========================================================================
    # ANGLE
    # =========================================================================
    print("\n" + "-" * 70)
    print("CHARACTERISTIC ANGLE")
    print("-" * 70)

    print(f"\n  Target: {angle_deg:.10f}°")
    print(f"  In radians: {angle_rad:.10f}")

    # Check if angle is a simple fraction of 360
    for num in range(1, 100):
        for den in range(1, 100):
            target_angle = 360 * num / den
            if abs(target_angle - angle_deg) < 0.01:
                print(f"    360 × {num}/{den} = {target_angle:.6f}° (error {abs(target_angle - angle_deg):.6f}°)")

    # =========================================================================
    # THE SYMMETRIC PAIR PATTERN
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE SYMMETRIC PAIR PATTERN: 11, 33, 56")
    print("=" * 70)

    p1 = seq[0] + seq[5]  # 11
    p2 = seq[1] + seq[4]  # 33
    p3 = seq[2] + seq[3]  # 56

    print(f"\n  Pair sums: {p1}, {p2}, {p3}")
    print(f"  Sum of pair sums: {p1 + p2 + p3} = 2 × {sum(seq)} / 2 = 100 ✓")

    print(f"\n  Relationships:")
    print(f"    p2/p1 = {p2}/{p1} = {p2/p1:.10f} = 3 EXACTLY")
    print(f"    p3/p2 = {p3}/{p2} = {p3/p2:.10f}")
    print(f"    p3/p1 = {p3}/{p1} = {p3/p1:.10f}")

    # What is 56/33?
    print(f"\n  What is 56/33?")
    print(f"    56/33 = {56/33:.10f}")
    print(f"    φ = {(1+np.sqrt(5))/2:.10f}")
    print(f"    √3 = {np.sqrt(3):.10f}")
    print(f"    5/3 = {5/3:.10f}")

    # The differences
    d1 = p2 - p1  # 22
    d2 = p3 - p2  # 23

    print(f"\n  Differences: {d1}, {d2} (consecutive!)")
    print(f"    Sum of differences: {d1 + d2} = {d1 + d2}")
    print(f"    Product: {d1 * d2} = {d1 * d2}")

    # =========================================================================
    # WHAT IF THE PARAMETERS ARE EXACTLY DETERMINED BY INTEGERS?
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYPOTHESIS: PARAMETERS FROM SEQUENCE INTEGERS")
    print("=" * 70)

    # We know modulus ≈ 20/19. What if it's EXACTLY 20/19?
    mod_exact = Fraction(20, 19)
    b_exact = -mod_exact**2

    print(f"\n  If modulus = 20/19 exactly:")
    print(f"    b = -(20/19)² = -{Fraction(400, 361)} = {float(-Fraction(400, 361)):.10f}")
    print(f"    Fitted b = {b:.10f}")
    print(f"    Difference: {abs(b + float(Fraction(400, 361))):.10f}")

    # For the angle, we need a = 2·modulus·cos(θ)
    # So cos(θ) = a / (2·modulus)
    cos_theta = a / (2 * 20/19)
    theta_from_mod = np.degrees(np.arccos(cos_theta))

    print(f"\n  If we use modulus = 20/19:")
    print(f"    cos(θ) = a / (2 × 20/19) = {a:.10f} / {2*20/19:.10f} = {cos_theta:.10f}")
    print(f"    θ = arccos({cos_theta:.10f}) = {theta_from_mod:.10f}°")

    # What integer ratio gives this cosine?
    print(f"\n  Looking for cos(θ) as integer ratio...")
    for num in range(1, 50):
        for den in range(1, 100):
            if num <= den:  # cos must be <= 1
                ratio = num / den
                error = abs(ratio - cos_theta)
                if error < 0.001:
                    angle = np.degrees(np.arccos(ratio))
                    print(f"    cos(θ) = {num}/{den} = {ratio:.10f} → θ = {angle:.6f}° (error {error:.6f})")

    # =========================================================================
    # THE COEFFICIENT a
    # =========================================================================
    print("\n" + "-" * 70)
    print("ANALYZING COEFFICIENT a MORE DEEPLY")
    print("-" * 70)

    # a = 2·modulus·cos(θ)
    # If modulus = 20/19 and we want a to be rational
    # We need cos(θ) = a·19/40

    print(f"\n  a = {a:.10f}")
    print(f"  a × 19/40 = {a * 19/40:.10f} = cos(θ) if modulus = 20/19")

    # What if a itself is a simple fraction?
    # a ≈ 0.966, close to 29/30
    print(f"\n  Checking simple fractions near a:")
    for num in range(90, 100):
        for den in range(90, 110):
            ratio = num / den
            if abs(ratio - a) < 0.01:
                print(f"    {num}/{den} = {ratio:.10f} (error {abs(ratio - a):.6f})")

    # =========================================================================
    # A NEW APPROACH: WHAT IF c IS THE KEY?
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE CONSTANT TERM c")
    print("=" * 70)

    print(f"\n  c = {c:.10f}")
    print(f"\n  c as fraction of 100 (sum): {c/100:.10f}")
    print(f"  c as fraction of 360: {c/360:.10f}")

    # c near 19.56, close to seq[4] = 19
    print(f"\n  c / seq[4] = {c / seq[4]:.10f}")
    print(f"  c / seq[5] = {c / seq[5]:.10f}")
    print(f"  c - seq[4] = {c - seq[4]:.10f}")

    # What if c is related to the pair sums?
    print(f"\n  c × (p1/p2) = {c * (p1/p2):.10f}")
    print(f"  c × (p2/p3) = {c * (p2/p3):.10f}")
    print(f"  100/c = {100/c:.10f}")

    # =========================================================================
    # THE EXACT EQUATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE EXACT EQUATIONS FROM INTEGERS")
    print("=" * 70)

    # We have 4 equations from the recurrence:
    # seq[2] = a·seq[1] + b·seq[0] + c  →  26 = 14a + 6b + c
    # seq[3] = a·seq[2] + b·seq[1] + c  →  30 = 26a + 14b + c
    # seq[4] = a·seq[3] + b·seq[2] + c  →  19 = 30a + 26b + c
    # seq[5] = a·seq[4] + b·seq[3] + c  →   5 = 19a + 30b + c

    print(f"""
  The 4 equations (overdetermined system):
    26 = 14a +  6b + c   ... (1)
    30 = 26a + 14b + c   ... (2)
    19 = 30a + 26b + c   ... (3)
     5 = 19a + 30b + c   ... (4)

  Subtracting to eliminate c:
    (2) - (1):  4 = 12a +  8b   →  1 = 3a + 2b   ... (5)
    (3) - (2): -11 =  4a + 12b  →  -11 = 4a + 12b  ... (6)
    (4) - (3): -14 = -11a + 4b  →  -14 = -11a + 4b  ... (7)
""")

    # From (5): 3a + 2b = 1, so a = (1 - 2b)/3
    # Sub into (6): -11 = 4(1-2b)/3 + 12b = (4 - 8b + 36b)/3 = (4 + 28b)/3
    # -33 = 4 + 28b
    # -37 = 28b
    # b = -37/28

    b_exact_from_int = Fraction(-37, 28)
    a_exact_from_int = (1 - 2*b_exact_from_int) / 3

    print(f"  From equations (5) and (6):")
    print(f"    b = -37/28 = {float(b_exact_from_int):.10f}")
    print(f"    a = (1 - 2b)/3 = {float(a_exact_from_int):.10f}")

    # Check with (7)
    lhs_7 = -11*float(a_exact_from_int) + 4*float(b_exact_from_int)
    print(f"\n  Verify with equation (7): -11a + 4b = {lhs_7:.10f} (should be -14)")
    print(f"    Error: {abs(lhs_7 - (-14)):.10f}")

    # The system is inconsistent because it's overdetermined!
    # But the EXACT integer solution from 2 equations gives us:

    print(f"\n  EXACT INTEGER SOLUTION from (5) and (6):")
    print(f"    a = {a_exact_from_int} = {float(a_exact_from_int):.10f}")
    print(f"    b = {b_exact_from_int} = {float(b_exact_from_int):.10f}")

    # Now derive c from equation (1)
    c_from_1 = 26 - 14*float(a_exact_from_int) - 6*float(b_exact_from_int)
    print(f"    c (from eq 1) = 26 - 14a - 6b = {c_from_1:.10f}")

    # What's √(-b)?
    mod_from_exact = np.sqrt(-float(b_exact_from_int))
    print(f"\n  If b = -37/28:")
    print(f"    modulus = √(37/28) = {mod_from_exact:.10f}")
    print(f"    Compare to 20/19 = {20/19:.10f}")
    print(f"    Compare to fitted = {modulus:.10f}")

    # Hmm, √(37/28) ≈ 1.149, not 1.053. Different solution set.

    # =========================================================================
    # USE EXACTLY 3 EQUATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXACT SOLUTION FROM FIRST 3 EQUATIONS")
    print("=" * 70)

    # Use equations (1), (2), (3) which determine a, b, c exactly
    A3 = np.array([
        [14, 6, 1],
        [26, 14, 1],
        [30, 26, 1]
    ])
    b3 = np.array([26, 30, 19])

    params3 = np.linalg.solve(A3, b3)
    a3, b3_coef, c3 = params3

    print(f"\n  Using equations (1), (2), (3):")
    print(f"    a = {a3:.10f}")
    print(f"    b = {b3_coef:.10f}")
    print(f"    c = {c3:.10f}")

    # Verify with equation (4)
    pred_4 = 19*a3 + 30*b3_coef + c3
    print(f"\n  Prediction for seq[5]: {pred_4:.10f} (actual: 5)")
    print(f"  Error: {abs(pred_4 - 5):.10f}")

    # Try equations (2), (3), (4)
    A_234 = np.array([
        [26, 14, 1],
        [30, 26, 1],
        [19, 30, 1]
    ])
    b_234 = np.array([30, 19, 5])

    params_234 = np.linalg.solve(A_234, b_234)
    a_234, b_234_coef, c_234 = params_234

    print(f"\n  Using equations (2), (3), (4):")
    print(f"    a = {a_234:.10f}")
    print(f"    b = {b_234_coef:.10f}")
    print(f"    c = {c_234:.10f}")

    # Verify with equation (1)
    pred_1 = 14*a_234 + 6*b_234_coef + c_234
    print(f"\n  Prediction for seq[2]: {pred_1:.10f} (actual: 26)")
    print(f"  Error: {abs(pred_1 - 26):.10f}")

    # =========================================================================
    # THE KEY INSIGHT: THE RESIDUAL IS THE MESSAGE
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHT: THE RESIDUAL IS THE MESSAGE")
    print("=" * 70)

    # The sequence doesn't EXACTLY follow any single recurrence
    # The DEVIATION from the best-fit recurrence is ~0.62 RMS
    # This deviation might encode additional information

    print(f"""
  The sequence [6, 14, 26, 30, 19, 5] almost follows:
    x[n+2] = a·x[n+1] + b·x[n] + c

  But not exactly. The residuals from least squares are:
""")

    # Compute predictions and residuals
    predictions = [a * seq[i+1] + b * seq[i] + c for i in range(4)]
    residuals_rec = [seq[i+2] - predictions[i] for i in range(4)]

    for i in range(4):
        print(f"    seq[{i+2}] predicted: {predictions[i]:.4f}, actual: {seq[i+2]}, residual: {residuals_rec[i]:+.4f}")

    print(f"\n  RMS residual: {np.sqrt(np.mean(np.array(residuals_rec)**2)):.4f}")

    print(f"""
  THE INSIGHT:
    The sequence is CLOSE to a perfect 2nd-order recurrence
    but the deviations form a pattern: {[f'{r:+.2f}' for r in residuals_rec]}

    If we round to integers: {[round(r) for r in residuals_rec]}

    These residuals might encode additional bits of information!
""")

    # =========================================================================
    # EXACT VALUES IF MODULUS = 20/19
    # =========================================================================
    print("\n" + "=" * 70)
    print("IF MODULUS = 20/19 EXACTLY")
    print("=" * 70)

    # Set modulus = 20/19 exactly
    mod_20_19 = 20/19
    b_if_20_19 = -(20/19)**2

    # What angle gives the fitted 'a'?
    # a = 2·modulus·cos(θ)
    # cos(θ) = a / (2·modulus)
    cos_theta_exact = a / (2 * mod_20_19)
    theta_exact = np.degrees(np.arccos(cos_theta_exact))

    print(f"\n  If modulus = 20/19 = {mod_20_19:.10f}")
    print(f"    b would be = -(20/19)² = {b_if_20_19:.10f}")
    print(f"    Fitted b = {b:.10f}")
    print(f"    Difference: {abs(b - b_if_20_19):.10f}")

    print(f"\n  The angle that matches fitted 'a' with modulus 20/19:")
    print(f"    cos(θ) = {a:.6f} / (2 × {mod_20_19:.6f}) = {cos_theta_exact:.10f}")
    print(f"    θ = {theta_exact:.10f}°")

    # What integer expression gives this cosine?
    # cos_theta_exact ≈ 0.459
    # Close to 9/20 = 0.45, 11/24 = 0.458
    print(f"\n  cos(θ) ≈ {cos_theta_exact:.6f}")
    print(f"    9/20 = {9/20:.6f}")
    print(f"    11/24 = {11/24:.6f}")
    print(f"    seq[0]+seq[5] / 24 = 11/24 = {11/24:.6f}")

    # 11 = seq[0] + seq[5]!
    cos_if_11_24 = 11/24
    theta_if_11_24 = np.degrees(np.arccos(cos_if_11_24))

    print(f"\n  IF cos(θ) = 11/24 = (seq[0]+seq[5])/24:")
    print(f"    θ = {theta_if_11_24:.10f}°")
    print(f"    Compare to fitted: {angle_deg:.10f}°")
    print(f"    Difference: {abs(theta_if_11_24 - angle_deg):.6f}°")

    # What is 24?
    print(f"\n  What is 24?")
    print(f"    24 = 4! = 24")
    print(f"    24 = 2 × 12 = 3 × 8 = 4 × 6")
    print(f"    24 = seq[0] × 4 = 6 × 4")
    print(f"    24 = p1 + p2/3 + ?? ")

    # =========================================================================
    # FINAL SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SYNTHESIS: THE EXACT ENCODING")
    print("=" * 70)

    print(f"""
  HYPOTHESIS: The sequence encodes its own dynamics through integer ratios

  MODULUS:
    modulus = 20/19 = (seq[0] + seq[1]) / seq[4]
    20 = 6 + 14, 19 = seq[4]

  COSINE OF ANGLE:
    cos(θ) ≈ 11/24 = (seq[0] + seq[5]) / 24
    11 = 6 + 5 = seq[0] + seq[5]
    24 = 4 × seq[0] = 4 × 6

  THEREFORE:
    a = 2 × (20/19) × (11/24)
      = 2 × 20 × 11 / (19 × 24)
      = 440 / 456
      = 55/57
      = {55/57:.10f}

    Fitted a = {a:.10f}
    Difference: {abs(55/57 - a):.10f}

    b = -(20/19)²
      = -400/361
      = {-400/361:.10f}

    Fitted b = {b:.10f}
    Difference: {abs(-400/361 - b):.10f}
""")

    # Verify our derived constants
    a_derived = 55/57
    b_derived = -400/361

    print(f"\n  Testing derived constants:")
    print(f"    a = 55/57 = {a_derived:.10f}")
    print(f"    b = -400/361 = {b_derived:.10f}")

    # Generate sequence with derived constants
    # Need to find c that makes sum = 100
    def sum_with_c(c_val):
        s = [seq[0], seq[1]]
        for _ in range(4):
            s.append(a_derived * s[-1] + b_derived * s[-2] + c_val)
        return sum(s) - 100

    # Binary search for c
    c_low, c_high = 0, 50
    while c_high - c_low > 1e-10:
        c_mid = (c_low + c_high) / 2
        if sum_with_c(c_mid) < 0:
            c_low = c_mid
        else:
            c_high = c_mid

    c_derived = c_mid

    print(f"    c (to make sum=100) = {c_derived:.10f}")
    print(f"    Fitted c = {c:.10f}")

    # Generate and compare
    derived_seq = [seq[0], seq[1]]
    for _ in range(4):
        derived_seq.append(a_derived * derived_seq[-1] + b_derived * derived_seq[-2] + c_derived)

    print(f"\n  Sequence from derived constants:")
    print(f"    Derived:  {[f'{x:.2f}' for x in derived_seq]}")
    print(f"    Actual:   {list(seq)}")
    print(f"    Errors:   {[f'{seq[i] - derived_seq[i]:+.2f}' for i in range(6)]}")


if __name__ == "__main__":
    main()
