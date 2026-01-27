#!/usr/bin/env python3
"""Find the EXACT formula for π/(e×ln(2)).

Goal: Express γ = π/(e×ln(2)) as a closed-form expression.

If we find it, we have:
  π/e = γ_exact × ln(2)

Which is a theorem connecting geometry, thermodynamics, and information.
"""

import math
import itertools
from fractions import Fraction

# Constants with high precision
PI = math.pi
E = math.e
LN2 = math.log(2)
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
GAMMA_EM = 0.5772156649015329  # Euler-Mascheroni

# Target
TARGET = PI / (E * LN2)

print("Finding the Exact Formula for π/(e×ln(2))")
print("=" * 70)
print()
print(f"Target: γ = π/(e×ln(2)) = {TARGET:.15f}")
print()

# Strategy 1: γ might be expressible WITHOUT π
# If γ = f(e, ln(2)), then π = e × ln(2) × f(e, ln(2))
# This would be extraordinary

print("-" * 70)
print("STRATEGY 1: Can γ be expressed without π?")
print("-" * 70)
print()

# Try: γ = a/b + c×e + d×ln(2) + e×e×ln(2) for simple fractions
best_no_pi = None
best_error_no_pi = float('inf')

for a in range(-10, 11):
    for b in range(1, 11):
        for c_num in range(-5, 6):
            for c_den in range(1, 6):
                for d_num in range(-5, 6):
                    for d_den in range(1, 6):
                        c = c_num / c_den if c_den != 0 else 0
                        d = d_num / d_den if d_den != 0 else 0

                        val = a/b + c*E + d*LN2
                        err = abs(val - TARGET)
                        if err < best_error_no_pi and err < 0.001:
                            best_error_no_pi = err
                            best_no_pi = (a, b, c_num, c_den, d_num, d_den, val)

if best_no_pi:
    a, b, c_num, c_den, d_num, d_den, val = best_no_pi
    print(f"Best (no π): {a}/{b} + ({c_num}/{c_den})×e + ({d_num}/{d_den})×ln(2)")
    print(f"  = {val:.15f}")
    print(f"  Error: {best_error_no_pi:.2e}")
else:
    print("No good match without π found.")
print()

# Strategy 2: γ involves a ratio of sums
print("-" * 70)
print("STRATEGY 2: γ as a ratio")
print("-" * 70)
print()

# Try: γ = (a + b×π) / (c + d×e) for simple integers
best_ratio = None
best_error_ratio = float('inf')

for a in range(-20, 21):
    for b in range(-10, 11):
        for c in range(-20, 21):
            for d in range(-10, 11):
                if c + d*E == 0:
                    continue
                try:
                    val = (a + b*PI) / (c + d*E)
                    err = abs(val - TARGET)
                    if err < best_error_ratio and err < 1e-6:
                        best_error_ratio = err
                        best_ratio = (a, b, c, d, val)
                except:
                    pass

if best_ratio:
    a, b, c, d, val = best_ratio
    print(f"Best ratio: ({a} + {b}×π) / ({c} + {d}×e)")
    print(f"  = {val:.15f}")
    print(f"  Error: {best_error_ratio:.2e}")
else:
    print("No good ratio found with error < 1e-6")

# Try with ln(2) involved
for a in range(-10, 11):
    for b in range(-5, 6):
        for c in range(-10, 11):
            for d in range(-5, 6):
                for e_coef in range(-5, 6):
                    if c + d*E + e_coef*LN2 == 0:
                        continue
                    try:
                        val = (a + b*PI) / (c + d*E + e_coef*LN2)
                        err = abs(val - TARGET)
                        if err < best_error_ratio and err < 1e-8:
                            best_error_ratio = err
                            best_ratio = (a, b, c, d, e_coef, val, "with_ln2")
                    except:
                        pass

if best_ratio and len(best_ratio) == 7:
    a, b, c, d, e_coef, val, _ = best_ratio
    print(f"Better ratio: ({a} + {b}×π) / ({c} + {d}×e + {e_coef}×ln(2))")
    print(f"  = {val:.15f}")
    print(f"  Error: {best_error_ratio:.2e}")
print()

# Strategy 3: γ = 1 + 2/f where f involves π
print("-" * 70)
print("STRATEGY 3: γ = 1 + 2/f where f is exact")
print("-" * 70)
print()

f_exact = 2 / (TARGET - 1)
print(f"Exact f = {f_exact:.15f}")
print()

# What is f? Is it expressible?
print("Testing f against expressions:")

f_candidates = {
    "3": 3,
    "π": PI,
    "e": E,
    "3 - 1/π²": 3 - 1/(PI**2),
    "3 - 1/(π×e)": 3 - 1/(PI*E),
    "3 - ln(2)/(π×e)": 3 - LN2/(PI*E),
    "3 × (1 - 1/(π²×e))": 3 * (1 - 1/(PI**2 * E)),
    "3 × (1 - ln(2)/(π²×e))": 3 * (1 - LN2/(PI**2 * E)),
    "π - 1/7": PI - 1/7,
    "π - 1/e²": PI - 1/(E**2),
    "π - ln(2)/e": PI - LN2/E,
    "e + 1/π": E + 1/PI,
    "3 - 1/(3×π×e)": 3 - 1/(3*PI*E),
    "π × (1 - 1/(π×e×ln(2)))": PI * (1 - 1/(PI*E*LN2)),
    "3 × e / (e + ln(2)/π)": 3 * E / (E + LN2/PI),
    "3 - (π-3)/(π×e)": 3 - (PI-3)/(PI*E),
    "3 × (1 - (π-3)/π³)": 3 * (1 - (PI-3)/(PI**3)),
}

results = []
for name, val in f_candidates.items():
    err = abs(val - f_exact) / f_exact * 100
    if err < 1:
        results.append((err, name, val))

results.sort()
for err, name, val in results[:10]:
    print(f"  f ≈ {name:<35} = {val:.12f} (error: {err:.6f}%)")
print()

# Strategy 4: Direct search for π = e × ln(2) × g(e, ln(2))
print("-" * 70)
print("STRATEGY 4: π = e × ln(2) × g where g has structure")
print("-" * 70)
print()

# We know π = e × ln(2) × γ
# So g = γ = π/(e×ln(2))
#
# Can we express this as g = h(e, ln(2)) where h doesn't use π?
# That would give: π = e × ln(2) × h(e, ln(2))
# A formula for π in terms of e and ln(2)!

print("If π = e × ln(2) × g, and g can be expressed without π,")
print("then we have a formula for π in terms of e and ln(2).")
print()

# Let's try: g = (a + b×e + c×ln(2)) / (d + e×e + f×ln(2))
print("Searching for g = (a + b×e + c×ln(2)) / (d + f×e + h×ln(2))...")

best_g = None
best_error_g = float('inf')

for a in range(-20, 21):
    for b in range(-10, 11):
        for c in range(-10, 11):
            for d in range(-20, 21):
                for f in range(-10, 11):
                    for h in range(-10, 11):
                        denom = d + f*E + h*LN2
                        if abs(denom) < 0.001:
                            continue
                        try:
                            g = (a + b*E + c*LN2) / denom
                            err = abs(g - TARGET)
                            if err < best_error_g:
                                best_error_g = err
                                best_g = (a, b, c, d, f, h, g)
                        except:
                            pass

if best_g and best_error_g < 1e-10:
    a, b, c, d, f, h, g = best_g
    print(f"FOUND: g = ({a} + {b}×e + {c}×ln(2)) / ({d} + {f}×e + {h}×ln(2))")
    print(f"  = {g:.15f}")
    print(f"  Target = {TARGET:.15f}")
    print(f"  Error: {best_error_g:.2e}")
    print()
    print("This would give:")
    print(f"  π = e × ln(2) × [({a} + {b}×e + {c}×ln(2)) / ({d} + {f}×e + {h}×ln(2))]")
else:
    print(f"Best found has error {best_error_g:.2e}")
print()

# Strategy 5: The EXACT identity must involve all three
print("-" * 70)
print("STRATEGY 5: Exact symmetric identity")
print("-" * 70)
print()

# What if: π^a × e^b × ln(2)^c = constant?
# Taking logs: a×ln(π) + b + c×ln(ln(2)) = ln(constant)

# We know: π/e ≈ (5/3) × ln(2)
# So: π ≈ (5/3) × e × ln(2)
# Taking logs: ln(π) ≈ ln(5/3) + 1 + ln(ln(2))

lhs = math.log(PI)
rhs = math.log(5/3) + 1 + math.log(LN2)
print(f"ln(π) = {lhs:.15f}")
print(f"ln(5/3) + 1 + ln(ln(2)) = {rhs:.15f}")
print(f"Difference: {lhs - rhs:.15f}")
print()

# The exact relationship
gamma_exact = TARGET
print(f"For EXACT equality: π = e × ln(2) × γ")
print(f"where γ = {gamma_exact:.15f}")
print()
print(f"Taking logs: ln(π) = 1 + ln(ln(2)) + ln(γ)")
print(f"So: ln(γ) = ln(π) - 1 - ln(ln(2))")
print(f"    ln(γ) = {math.log(gamma_exact):.15f}")
print(f"    ln(π) - 1 - ln(ln(2)) = {math.log(PI) - 1 - math.log(LN2):.15f}")
print()

# Strategy 6: Try expressions with sqrt
print("-" * 70)
print("STRATEGY 6: Involving square roots")
print("-" * 70)
print()

sqrt_candidates = {
    "√(π/e) + 1/2": math.sqrt(PI/E) + 0.5,
    "√(π×e) / e": math.sqrt(PI*E) / E,
    "1 + 2/√(π×e)": 1 + 2/math.sqrt(PI*E),
    "(√π + √e) / √(πe)": (math.sqrt(PI) + math.sqrt(E)) / math.sqrt(PI*E),
    "√(e/π) + 1": math.sqrt(E/PI) + 1,
    "(1 + √(e/π)) × √(e/π)": (1 + math.sqrt(E/PI)) * math.sqrt(E/PI),
    "e / √(π×e - 1)": E / math.sqrt(PI*E - 1),
    "√(π²/e² + 1/3)": math.sqrt((PI/E)**2 + 1/3),
    "(π + e) / (√π × e)": (PI + E) / (math.sqrt(PI) * E),
    "1 + 1/√(πe)": 1 + 1/math.sqrt(PI*E),
    "5/(3×√(1 - ε))": 5/(3*math.sqrt(1 - 0.00125)),  # If ε relates to curvature
    "5/3 × √(1 + 1/400)": 5/3 * math.sqrt(1 + 1/400),
}

print(f"{'Expression':<35} {'Value':<18} {'Error %':<12}")
print("-" * 65)
for name, val in sorted(sqrt_candidates.items(), key=lambda x: abs(x[1] - TARGET)):
    err = abs(val - TARGET) / TARGET * 100
    if err < 1:
        print(f"{name:<35} {val:<18.12f} {err:<12.6f}")
print()

# THE KEY INSIGHT
print("=" * 70)
print("THE KEY INSIGHT")
print("=" * 70)
print()

print("The relationship π/e = γ × ln(2) can be rewritten as:")
print()
print("  π = e × γ × ln(2)")
print()
print("If γ can be expressed exactly WITHOUT using π, then we have")
print("a formula for π in terms of e and ln(2).")
print()
print("We know:")
print(f"  γ = {TARGET:.15f}")
print(f"  5/3 = {5/3:.15f}")
print(f"  Ratio = γ/(5/3) = {TARGET / (5/3):.15f}")
print()

# The ratio is very close to 1
ratio = TARGET / (5/3)
print(f"γ = (5/3) × {ratio:.15f}")
print(f"  = (5/3) × (1 + {ratio - 1:.15f})")
print()

# Is the correction expressible?
correction = ratio - 1
print(f"Correction term: {correction:.15f}")
print()

# Try to express correction
corr_candidates = {
    "1/(4×π²)": 1/(4*PI**2),
    "1/(π²×e)": 1/(PI**2 * E),
    "ln(2)/(π²×e)": LN2/(PI**2 * E),
    "1/(π×e²)": 1/(PI * E**2),
    "(π-3)/(π³)": (PI-3)/(PI**3),
    "1/(2×π×e²)": 1/(2*PI*E**2),
    "1/(3×π²)": 1/(3*PI**2),
    "(e-2)/(π×e²)": (E-2)/(PI*E**2),
    "1/(π²+e²)": 1/(PI**2 + E**2),
    "ln(2)/(3×π²)": LN2/(3*PI**2),
}

print(f"Searching for correction term {correction:.12f}:")
print()
for name, val in sorted(corr_candidates.items(), key=lambda x: abs(x[1] - correction)):
    err = abs(val - correction) / correction * 100
    if err < 20:
        print(f"  {correction:.12f} ≈ {name:<20} = {val:.12f} (error: {err:.4f}%)")

# THE FORMULA
print()
print("=" * 70)
print("CANDIDATE EXACT FORMULA")
print("=" * 70)
print()

# Best candidate: γ = (5/3) × (1 + 1/(π²×e))
gamma_candidate = (5/3) * (1 + 1/(PI**2 * E))
error_candidate = abs(gamma_candidate - TARGET) / TARGET * 100

print("CANDIDATE:")
print()
print("     γ = (5/3) × (1 + 1/(π²×e))")
print()
print(f"     = (5/3) × (1 + 1/{PI**2 * E:.10f})")
print(f"     = (5/3) × {1 + 1/(PI**2 * E):.15f}")
print(f"     = {gamma_candidate:.15f}")
print()
print(f"Target = {TARGET:.15f}")
print(f"Error: {error_candidate:.6f}%")
print()

if error_candidate < 0.01:
    print("*** CLOSE MATCH! ***")
    print()
    print("If this is exact, then:")
    print()
    print("     π/e = (5/3) × (1 + 1/(π²×e)) × ln(2)")
    print()
    print("     π/e = (5/3) × ln(2) + (5/3) × ln(2)/(π²×e)")
    print()
    print("     π/e - (5/3) × ln(2) = (5/3) × ln(2)/(π²×e)")
    print()
    print("     ε = (5/3) × ln(2)/(π²×e)")
    print()
    eps_formula = (5/3) * LN2 / (PI**2 * E)
    eps_actual = PI/E - (5/3)*LN2
    print(f"     Predicted ε = {eps_formula:.15f}")
    print(f"     Actual ε    = {eps_actual:.15f}")
    print(f"     Error: {abs(eps_formula - eps_actual)/eps_actual * 100:.4f}%")


if __name__ == "__main__":
    pass
