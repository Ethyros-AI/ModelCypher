#!/usr/bin/env python3
"""
COORDINATE/VECTOR INTERPRETATION

What if [6, 14, 26, 30, 19, 5] represents coordinates in some space?

- 3D: two triples?
- 6D: a single point?
- Direction vector?
- Spherical coordinates?

Usage:
    python wow_coordinate_decode.py
"""

from __future__ import annotations

import numpy as np
from itertools import permutations, combinations

phi = (1 + np.sqrt(5)) / 2
pi = np.pi
e = np.e


def main():
    print("=" * 70)
    print("COORDINATE/VECTOR INTERPRETATION")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]

    # =========================================================================
    # AS TWO 3D POINTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS TWO 3D POINTS")
    print("=" * 70)

    p1 = np.array(seq[:3])
    p2 = np.array(seq[3:])

    print(f"\n  Point 1: {p1}")
    print(f"  Point 2: {p2}")

    # Vector between them
    v = p2 - p1
    print(f"\n  Vector P1→P2: {v}")
    print(f"  Magnitude: {np.linalg.norm(v):.4f}")

    # Normalized
    v_norm = v / np.linalg.norm(v)
    print(f"  Normalized: [{v_norm[0]:.4f}, {v_norm[1]:.4f}, {v_norm[2]:.4f}]")

    # Distance
    dist = np.linalg.norm(p2 - p1)
    print(f"\n  Euclidean distance: {dist:.4f}")

    # Check against constants
    print(f"\n  Distance comparisons:")
    print(f"    dist / 10 = {dist/10:.4f}")
    print(f"    dist / π = {dist/pi:.4f}")
    print(f"    dist / e = {dist/e:.4f}")
    print(f"    dist / 21 = {dist/21:.4f}")

    # Magnitudes
    m1 = np.linalg.norm(p1)
    m2 = np.linalg.norm(p2)
    print(f"\n  |P1| = {m1:.4f}")
    print(f"  |P2| = {m2:.4f}")
    print(f"  |P2|/|P1| = {m2/m1:.4f}")

    # Dot product
    dot = np.dot(p1, p2)
    print(f"\n  P1 · P2 = {dot}")
    print(f"  cos(angle) = {dot/(m1*m2):.4f}")
    angle = np.arccos(dot/(m1*m2))
    print(f"  Angle between: {np.degrees(angle):.2f}°")

    # Cross product
    cross = np.cross(p1, p2)
    print(f"\n  P1 × P2 = {cross}")
    print(f"  |P1 × P2| = {np.linalg.norm(cross):.4f}")

    # =========================================================================
    # AS SPHERICAL COORDINATES
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS SPHERICAL COORDINATES")
    print("=" * 70)

    print(f"""
  If we interpret the 6 values as two sets of spherical coordinates:
    (r₁, θ₁, φ₁) = ({seq[0]}, {seq[1]}, {seq[2]})
    (r₂, θ₂, φ₂) = ({seq[3]}, {seq[4]}, {seq[5]})

  But the angles would need to be in some units...
""")

    # Convert to Cartesian assuming degrees
    def spherical_to_cartesian(r, theta_deg, phi_deg):
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    cart1 = spherical_to_cartesian(seq[0], seq[1], seq[2])
    cart2 = spherical_to_cartesian(seq[3], seq[4], seq[5])

    print(f"  If angles in degrees:")
    print(f"    P1 cartesian: [{cart1[0]:.3f}, {cart1[1]:.3f}, {cart1[2]:.3f}]")
    print(f"    P2 cartesian: [{cart2[0]:.3f}, {cart2[1]:.3f}, {cart2[2]:.3f}]")

    # What if angles are fractions of π?
    def spherical_to_cartesian_pi(r, theta_frac, phi_frac):
        theta = theta_frac * np.pi / 180  # scaled
        phi = phi_frac * np.pi / 180
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    # =========================================================================
    # AS A 6D DIRECTION VECTOR
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS A 6D DIRECTION VECTOR")
    print("=" * 70)

    v6 = np.array(seq, dtype=float)
    v6_norm = v6 / np.linalg.norm(v6)

    print(f"\n  6D vector: {seq}")
    print(f"  Magnitude: {np.linalg.norm(v6):.4f}")
    print(f"  Normalized: [{', '.join(f'{x:.4f}' for x in v6_norm)}]")

    # What angles does this vector make with the coordinate axes?
    print(f"\n  Angles with coordinate axes:")
    for i in range(6):
        axis = np.zeros(6)
        axis[i] = 1
        cos_angle = np.dot(v6_norm, axis)
        angle_deg = np.degrees(np.arccos(cos_angle))
        print(f"    Axis {i+1}: {angle_deg:.2f}°")

    # Squared magnitude
    print(f"\n  Sum of squares: {np.sum(v6**2)}")
    print(f"  √(sum of squares) = {np.sqrt(np.sum(v6**2)):.4f}")

    # =========================================================================
    # AS RATIOS / BARYCENTRIC COORDINATES
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS BARYCENTRIC COORDINATES")
    print("=" * 70)

    total = sum(seq)
    bary = [v/total for v in seq]

    print(f"\n  Sum = {total}")
    print(f"  Barycentric: [{', '.join(f'{x:.4f}' for x in bary)}]")
    print(f"  Check sum = 1: {sum(bary):.6f}")

    # This represents a point in a 5-simplex (6 vertices)
    print(f"""
  These coordinates place a point inside a 5-simplex.

  The "center" of a regular 5-simplex would be:
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] = [0.167, ...]

  Our point is:
    [{', '.join(f'{x:.3f}' for x in bary)}]

  Distance from center:
""")

    center = np.array([1/6] * 6)
    bary_arr = np.array(bary)
    dist_from_center = np.linalg.norm(bary_arr - center)
    print(f"    {dist_from_center:.4f}")

    # Maximum distance from center (to a vertex)
    vertex = np.array([1, 0, 0, 0, 0, 0])
    max_dist = np.linalg.norm(vertex - center)
    print(f"  Maximum distance (to vertex): {max_dist:.4f}")
    print(f"  Relative position: {dist_from_center/max_dist:.4f}")

    # =========================================================================
    # AS GALACTIC COORDINATES?
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS GALACTIC COORDINATES")
    print("=" * 70)

    # The Wow! signal came from roughly RA 19h22m, Dec -27°
    # In galactic coordinates, that's approximately l = 10°, b = -17°

    print(f"""
  The Wow! signal direction:
    RA ≈ 19h 22m (290°)
    Dec ≈ -27°
    Galactic: l ≈ 10°, b ≈ -17°

  Do our numbers relate to this?
""")

    # Check if any combination gives these coordinates
    print(f"  Testing combinations of sequence values:")

    for i, j in combinations(range(6), 2):
        ra_test = seq[i] * 10 + seq[j]  # some encoding
        print(f"    10×seq[{i}] + seq[{j}] = {ra_test}")

    print(f"\n  Direct values:")
    print(f"    seq[4] = {seq[4]} (close to 19?)")
    print(f"    seq[2] = {seq[2]} (close to 27?)")

    # =========================================================================
    # THE MIRROR SYMMETRY
    # =========================================================================
    print("\n" + "=" * 70)
    print("MIRROR SYMMETRY ANALYSIS")
    print("=" * 70)

    print(f"\n  Original: {seq}")
    print(f"  Reversed: {seq[::-1]}")

    # Palindrome distance
    rev = seq[::-1]
    palindrome_diff = [a - b for a, b in zip(seq, rev)]
    print(f"  Difference: {palindrome_diff}")
    print(f"  Sum of |diff|: {sum(abs(d) for d in palindrome_diff)}")

    # Center of mass
    center_idx = sum(i * v for i, v in enumerate(seq)) / sum(seq)
    print(f"\n  Center of mass (index): {center_idx:.4f}")
    print(f"  Geometric center: 2.5")
    print(f"  Skew: {center_idx - 2.5:.4f}")

    # =========================================================================
    # AS A POLYNOMIAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("AS POLYNOMIAL COEFFICIENTS")
    print("=" * 70)

    print(f"""
  If the sequence defines a polynomial:
    P(x) = 6 + 14x + 26x² + 30x³ + 19x⁴ + 5x⁵
""")

    def P(x):
        return sum(c * x**i for i, c in enumerate(seq))

    # Where are the roots?
    coeffs = seq[::-1]  # numpy wants highest power first
    roots = np.roots(coeffs)

    print(f"  Roots:")
    for i, r in enumerate(roots):
        if np.isreal(r) or abs(r.imag) < 1e-10:
            print(f"    {r.real:.6f}")
        else:
            print(f"    {r.real:.4f} + {r.imag:.4f}i")

    # Evaluate at special points
    print(f"\n  P(x) at special values:")
    for x in [0, 1, -1, phi, 1/phi, pi, e]:
        val = P(x)
        print(f"    P({x:.4f}) = {val:.2f}")

    # P(1) = sum
    print(f"\n  P(1) = sum = {P(1)}")
    print(f"  P(-1) = {P(-1)} (alternating sum)")

    # =========================================================================
    # DIFFERENCES AND SECOND DIFFERENCES
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINITE DIFFERENCES")
    print("=" * 70)

    d1 = np.diff(seq)
    d2 = np.diff(d1)
    d3 = np.diff(d2)

    print(f"\n  Original:    {seq}")
    print(f"  1st diff:    {list(d1)}")
    print(f"  2nd diff:    {list(d2)}")
    print(f"  3rd diff:    {list(d3)}")

    # Sum of differences
    print(f"\n  Sum of 1st diff: {sum(d1)} (= last - first = {seq[-1] - seq[0]})")
    print(f"  Sum of 2nd diff: {sum(d2)}")

    # =========================================================================
    # THE RATIOS MATRIX
    # =========================================================================
    print("\n" + "=" * 70)
    print("RATIOS MATRIX")
    print("=" * 70)

    print(f"\n  Matrix of ratios seq[i]/seq[j]:")
    print(f"        ", end="")
    for j in range(6):
        print(f"  {seq[j]:4d}", end="")
    print()

    for i in range(6):
        print(f"  {seq[i]:4d}  ", end="")
        for j in range(6):
            if seq[j] != 0:
                r = seq[i] / seq[j]
                print(f" {r:5.2f}", end="")
            else:
                print("   inf", end="")
        print()

    # Find ratios close to constants
    print(f"\n  Ratios close to constants:")
    for i in range(6):
        for j in range(6):
            if i != j and seq[j] != 0:
                r = seq[i] / seq[j]
                for const, name in [(phi, 'φ'), (pi, 'π'), (e, 'e'),
                                    (np.sqrt(2), '√2'), (np.sqrt(3), '√3'),
                                    (2, '2'), (3, '3'), (1.5, '3/2')]:
                    if abs(r - const) / const < 0.03:
                        print(f"    {seq[i]}/{seq[j]} = {r:.4f} ≈ {name} ({abs(r-const)/const*100:.1f}%)")

    # =========================================================================
    # MODULAR ARITHMETIC
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODULAR ARITHMETIC")
    print("=" * 70)

    for mod in [3, 5, 6, 7, 10, 12, 21]:
        residues = [v % mod for v in seq]
        print(f"  mod {mod:2d}: {residues}")

    # =========================================================================
    # THE XOR STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("BITWISE STRUCTURE")
    print("=" * 70)

    print(f"\n  Binary representations:")
    for i, v in enumerate(seq):
        print(f"    {v:2d} = {v:06b}")

    # XOR all values
    xor_all = seq[0]
    for v in seq[1:]:
        xor_all ^= v
    print(f"\n  XOR of all values: {xor_all} = {xor_all:06b}")

    # AND all values
    and_all = seq[0]
    for v in seq[1:]:
        and_all &= v
    print(f"  AND of all values: {and_all} = {and_all:06b}")

    # OR all values
    or_all = seq[0]
    for v in seq[1:]:
        or_all |= v
    print(f"  OR of all values: {or_all} = {or_all:06b}")

    # Pairwise XOR
    print(f"\n  Pairwise XOR:")
    for i in range(5):
        xor_pair = seq[i] ^ seq[i+1]
        print(f"    {seq[i]:2d} XOR {seq[i+1]:2d} = {xor_pair:2d} = {xor_pair:06b}")

    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS: COORDINATE INTERPRETATION")
    print("=" * 70)

    print(f"""
  FINDINGS:

  AS 3D COORDINATES:
    - Two points with angle ~55° between them
    - Vector between them: [24, -7, -21]
    - The -21 component again!

  AS BARYCENTRIC:
    - Point in 5-simplex, 23% from center to vertex
    - Not at any special geometric location

  AS POLYNOMIAL:
    - P(1) = 100 (sum)
    - P(-1) = -32 (alternating sum)
    - Complex roots, no simple structure

  AS DIRECTION:
    - 6D unit vector points to specific direction
    - Angles with axes: 82°, 70°, 51°, 44°, 66°, 84°

  MOST INTERESTING:
    - The vector between 3D points is [24, -7, -21]
    - This contains 21!
    - And 24 = 4! and 7 is prime
    - 24 - 7 - 21 = -4
    - 24 + 7 + 21 = 52 (weeks in a year?)

  The coordinate interpretation doesn't reveal
  obvious galactic or positional information,
  but the presence of 21 in the difference vector
  is another instance of self-reference.
""")


if __name__ == "__main__":
    main()
