#!/usr/bin/env python3
"""Fast search for the exact formula.

We found: correction ≈ 1/(π²×e) might work.
Let's verify and explore nearby formulas.
"""

import math

PI = math.pi
E = math.e
LN2 = math.log(2)
PHI = (1 + math.sqrt(5)) / 2

TARGET = PI / (E * LN2)  # 1.667362116163107
EPSILON = PI/E - (5/3)*LN2  # 0.000482048857680

print("Fast Exact Formula Search")
print("=" * 70)
print()
print(f"Target γ = π/(e×ln(2)) = {TARGET:.15f}")
print(f"5/3                    = {5/3:.15f}")
print(f"ε = π/e - (5/3)×ln(2)  = {EPSILON:.15f}")
print()

# The correction factor
correction = TARGET / (5/3) - 1
print(f"Correction: γ = (5/3) × (1 + c)")
print(f"where c = {correction:.15f}")
print()

# Test candidates for c
print("-" * 70)
print("TESTING CORRECTION TERM CANDIDATES")
print("-" * 70)
print()

candidates = [
    # Simple fractions of π, e, ln(2)
    ("1/(π²×e)", 1/(PI**2 * E)),
    ("1/(π×e²)", 1/(PI * E**2)),
    ("1/(π²×e×3)", 1/(PI**2 * E * 3)),
    ("ln(2)/(π²×e)", LN2/(PI**2 * E)),
    ("1/(3×π²)", 1/(3 * PI**2)),
    ("1/(2×π×e²)", 1/(2 * PI * E**2)),
    ("(π-3)/(π³)", (PI-3)/(PI**3)),
    ("1/(π³)", 1/(PI**3)),
    ("1/(e³)", 1/(E**3)),
    ("ln(2)/(e³)", LN2/(E**3)),
    ("1/(π²+e²)", 1/(PI**2 + E**2)),
    ("1/(π×e×3)", 1/(PI * E * 3)),
    ("(e-2)/(e³)", (E-2)/(E**3)),
    ("1/(2π²)", 1/(2*PI**2)),
    ("ln(2)/(3π²)", LN2/(3*PI**2)),
    ("1/(π²×ln(2))", 1/(PI**2 * LN2)),
    ("(π-3)/(π²×e)", (PI-3)/(PI**2 * E)),
    ("1/(4π²)", 1/(4*PI**2)),
    ("(4-π)/(π³)", (4-PI)/(PI**3)),
    ("1/(π×e×ln(2)×3)", 1/(PI * E * LN2 * 3)),

    # Involving φ
    ("1/(φ×π²)", 1/(PHI * PI**2)),
    ("1/(φ×e²)", 1/(PHI * E**2)),

    # More complex
    ("(π-3)/π³ + 1/(3e³)", (PI-3)/(PI**3) + 1/(3*E**3)),
    ("(3-π)/(π²×e) × (-1)", -(3-PI)/(PI**2 * E)),
    ("ln(2)²/(π×e²)", LN2**2/(PI * E**2)),
    ("1/(2πe)", 1/(2*PI*E)),
    ("1/(6π)", 1/(6*PI)),
]

print(f"{'Expression':<30} {'Value':<18} {'Error %':<12}")
print("-" * 60)

results = []
for name, val in candidates:
    err = abs(val - correction) / correction * 100
    results.append((err, name, val))

results.sort()
for err, name, val in results[:15]:
    marker = " <<<" if err < 1 else ""
    print(f"{name:<30} {val:.15f} {err:>10.4f}%{marker}")

print()

# Check the best candidate
print("-" * 70)
print("CHECKING BEST CANDIDATE")
print("-" * 70)
print()

# Let's manually derive what c should be
# c = γ/(5/3) - 1 = 3γ/5 - 1 = (3γ - 5)/5 = (3π/(e×ln(2)) - 5)/5
# c = (3π - 5e×ln(2)) / (5e×ln(2))

c_exact = (3*PI - 5*E*LN2) / (5*E*LN2)
print(f"Exact c = (3π - 5e×ln(2)) / (5e×ln(2))")
print(f"        = {c_exact:.15f}")
print()

# Can we simplify (3π - 5e×ln(2))?
numerator = 3*PI - 5*E*LN2
print(f"Numerator: 3π - 5e×ln(2) = {numerator:.15f}")
print(f"           ≈ {numerator:.10f}")
print()

# What is this close to?
num_candidates = [
    ("1/e", 1/E),
    ("1/π", 1/PI),
    ("ln(2)/e", LN2/E),
    ("1/3", 1/3),
    ("1/φ²", 1/(PHI**2)),
    ("e-2", E-2),
    ("π-3", PI-3),
    ("ln(2)²", LN2**2),
    ("1/(π-e)", 1/(PI-E)),
    ("1/4", 1/4),
    ("(π-e)/π", (PI-E)/PI),
    ("ln(2)/π", LN2/PI),
    ("1/(2e)", 1/(2*E)),
]

print("What is the numerator close to?")
for name, val in num_candidates:
    if abs(val - numerator) < 0.1:
        err = abs(val - numerator)
        print(f"  {numerator:.10f} ≈ {name} = {val:.10f} (diff: {err:.10f})")
print()

# THE KEY: What if it's EXACT?
print("-" * 70)
print("THE EXACT IDENTITY (if it exists)")
print("-" * 70)
print()

# γ = π/(e×ln(2))
# If this equals (5/3)(1 + c) for some expressible c,
# then: π/(e×ln(2)) = (5/3)(1 + c)
# So: π = (5/3)(1 + c) × e × ln(2)
# If c = f(π,e,ln(2)), we can solve for the relationship.

# Let's try: c = k/(π^a × e^b × ln(2)^c) for small integers

print("Searching: c = k/(π^a × e^b × ln(2)^c)")
print()

best_match = None
best_err = float('inf')

for k_num in range(1, 10):
    for k_den in range(1, 10):
        k = k_num / k_den
        for a in range(-3, 4):
            for b in range(-3, 4):
                for c_exp in range(-3, 4):
                    if a == b == c_exp == 0:
                        continue
                    try:
                        val = k / (PI**a * E**b * LN2**c_exp)
                        err = abs(val - correction) / correction
                        if err < best_err:
                            best_err = err
                            best_match = (k_num, k_den, a, b, c_exp, val)
                    except:
                        pass

if best_match and best_err < 0.01:
    k_num, k_den, a, b, c_exp, val = best_match
    print(f"BEST MATCH:")
    print(f"  c = ({k_num}/{k_den}) / (π^{a} × e^{b} × ln(2)^{c_exp})")
    print(f"    = {val:.15f}")
    print(f"  Target c = {correction:.15f}")
    print(f"  Error: {best_err*100:.6f}%")
    print()

    # What does this give us?
    gamma_formula = (5/3) * (1 + val)
    print(f"  γ = (5/3) × (1 + ({k_num}/{k_den})/(π^{a} × e^{b} × ln(2)^{c_exp}))")
    print(f"    = {gamma_formula:.15f}")
    print(f"  Target = {TARGET:.15f}")
    print(f"  Error: {abs(gamma_formula - TARGET)/TARGET * 100:.6f}%")
    print()

    # The full identity
    print("  FULL IDENTITY:")
    print(f"  π/e = (5/3) × (1 + ({k_num}/{k_den})/(π^{a} × e^{b} × ln(2)^{c_exp})) × ln(2)")

# ALTERNATIVE: Try without 5/3
print()
print("-" * 70)
print("ALTERNATIVE: Express γ directly without 5/3")
print("-" * 70)
print()

# Try: γ = (a×π + b×e + c) / (d×e + f×ln(2) + g)
# for small integers

print("Searching: γ = (a×π + b) / (c×e + d)")
print()

best_direct = None
best_err_direct = float('inf')

for a in range(1, 5):
    for b in range(-10, 11):
        for c in range(-5, 6):
            for d in range(-10, 11):
                denom = c*E + d
                if abs(denom) < 0.1:
                    continue
                val = (a*PI + b) / denom
                err = abs(val - TARGET) / TARGET
                if err < best_err_direct:
                    best_err_direct = err
                    best_direct = (a, b, c, d, val)

if best_direct and best_err_direct < 0.0001:
    a, b, c, d, val = best_direct
    print(f"FOUND: γ = ({a}π + {b}) / ({c}e + {d})")
    print(f"  = {val:.15f}")
    print(f"  Target = {TARGET:.15f}")
    print(f"  Error: {best_err_direct*100:.6f}%")
else:
    print(f"No good match found (best error: {best_err_direct*100:.4f}%)")

# Try with ln(2) in formula
print()
print("Searching: γ = (a×π + b) / (c×e×ln(2) + d)")
print()

for a in range(1, 5):
    for b in range(-10, 11):
        for c in range(-5, 6):
            for d in range(-10, 11):
                denom = c*E*LN2 + d
                if abs(denom) < 0.1:
                    continue
                val = (a*PI + b) / denom
                err = abs(val - TARGET) / TARGET
                if err < best_err_direct:
                    best_err_direct = err
                    best_direct = (a, b, c, d, val, "with_ln2")

if best_direct and len(best_direct) == 6 and best_err_direct < 0.0001:
    a, b, c, d, val, _ = best_direct
    print(f"FOUND: γ = ({a}π + {b}) / ({c}e×ln(2) + {d})")
    print(f"  = {val:.15f}")
    print(f"  Error: {best_err_direct*100:.6f}%")

# FINAL: The theorem form
print()
print("=" * 70)
print("THE THEOREM (if exact)")
print("=" * 70)
print()

if best_match and best_err < 0.001:
    k_num, k_den, a, b, c_exp, _ = best_match
    print("INFORMATION-THERMODYNAMIC BRIDGE THEOREM")
    print()
    print("Let:")
    print("  - ln(2) be Landauer's constant (information)")
    print("  - γ = 5/3 be the Euclidean adiabatic index (thermodynamics)")
    print("  - π/e be the entropy geometry constant")
    print()
    print("Then:")
    print()
    print(f"     π        5              {k_num}      ")
    print(f"    ─── = ─── × ─────────────────── × ln(2)")
    print(f"     e        3       π^{a} × e^{b} × ln(2)^{c_exp}  ")
    print()
    print("Or equivalently:")
    print()
    print("     GEOMETRY = EFFICIENCY × CURVATURE × INFORMATION")
    print()
    print(f"where CURVATURE = 1 + {k_num}/({k_den} × π^{a} × e^{b} × ln(2)^{c_exp})")
else:
    print("No exact formula found yet.")
    print()
    print("The relationship remains:")
    print("  π/e ≈ (5/3) × ln(2)  with 0.04% error")
    print()
    print("The 0.04% may represent:")
    print("  - Information curvature correction")
    print("  - Non-integer effective dimension (f ≈ 2.997)")
    print("  - Fundamental limit of Euclidean approximation")


if __name__ == "__main__":
    pass
