#!/usr/bin/env python3
"""Explore the correction term that makes π/e = (5/3) × ln(2) + ε exact.

The 0.04% error is not noise - it's structure. What is ε in terms of
fundamental constants?

If we can express ε cleanly, we have a real identity.
"""

import math
from fractions import Fraction

# Fundamental constants
PI = math.pi
E = math.e
LN2 = math.log(2)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
GAMMA = 5/3  # Adiabatic index

# Derived constants
PI_OVER_E = PI / E
FIVE_THIRDS_LN2 = GAMMA * LN2

# The error term
EPSILON = PI_OVER_E - FIVE_THIRDS_LN2

print("Exploring the Correction Term")
print("=" * 70)
print()
print(f"π/e           = {PI_OVER_E:.15f}")
print(f"(5/3) × ln(2) = {FIVE_THIRDS_LN2:.15f}")
print(f"ε = difference = {EPSILON:.15f}")
print()

# Is ε expressible in terms of fundamental constants?
print("-" * 70)
print("SEARCHING FOR ε IN TERMS OF FUNDAMENTAL CONSTANTS")
print("-" * 70)
print()

# Build a library of candidate expressions
candidates = {
    # Simple constants
    "1": 1,
    "π": PI,
    "e": E,
    "ln(2)": LN2,
    "φ": PHI,
    "√2": SQRT2,
    "√3": SQRT3,

    # Ratios
    "π/e": PI_OVER_E,
    "e/π": E/PI,
    "ln(2)/π": LN2/PI,
    "ln(2)/e": LN2/E,
    "1/φ": 1/PHI,
    "φ-1": PHI - 1,
    "1/e": 1/E,
    "1/π": 1/PI,

    # Powers
    "π²": PI**2,
    "e²": E**2,
    "ln(2)²": LN2**2,
    "√π": math.sqrt(PI),
    "√e": math.sqrt(E),
    "√(π/e)": math.sqrt(PI/E),

    # Products
    "π×ln(2)": PI * LN2,
    "e×ln(2)": E * LN2,
    "π×e": PI * E,

    # Special combinations
    "π-e": PI - E,
    "e-π": E - PI,
    "π/e - 1": PI/E - 1,
    "1 - e/π": 1 - E/PI,
    "ln(π)": math.log(PI),
    "ln(e)": 1,  # =1 by definition
    "ln(φ)": math.log(PHI),
    "ln(2)/ln(3)": LN2 / math.log(3),

    # Euler-Mascheroni
    "γ_EM": 0.5772156649,  # Euler-Mascheroni constant

    # Catalan's constant
    "G": 0.9159655941,  # Catalan's constant

    # Apéry's constant
    "ζ(3)": 1.2020569032,  # Riemann zeta(3)
}

print("Testing: ε = c × (constant) for simple c")
print()

matches = []
for name, value in candidates.items():
    if abs(value) < 1e-15:
        continue

    # What coefficient c makes ε = c × value?
    c = EPSILON / value

    # Check if c is a simple fraction
    for num in range(-20, 21):
        for den in range(1, 21):
            if num == 0:
                continue
            frac = num / den
            if abs(frac - c) < 1e-6:
                # Found a match!
                predicted = frac * value
                rel_error = abs(predicted - EPSILON) / abs(EPSILON) * 100
                if rel_error < 0.1:  # Within 0.1%
                    matches.append((name, num, den, rel_error, c))

matches.sort(key=lambda x: x[3])
print(f"{'Expression':<30} {'Fraction':<10} {'Error %':<12} {'Actual c':<15}")
print("-" * 70)
for name, num, den, err, c in matches[:15]:
    print(f"ε = ({num}/{den}) × {name:<15} {num}/{den:<6} {err:<12.6f} {c:<15.10f}")

# More complex: ε = a×X + b×Y
print()
print("-" * 70)
print("SEARCHING FOR TWO-TERM EXPRESSIONS")
print("-" * 70)
print()

# Try: ε = a×ln(2)² + b×(something)
# Or: ε = (small fraction) × (combination)

# Key insight: maybe ε involves higher-order terms
print("Testing specific hypotheses:")
print()

# Hypothesis 1: ε = ln(2)³ × (simple factor)
ln2_cubed = LN2**3
ratio1 = EPSILON / ln2_cubed
print(f"ε / ln(2)³ = {ratio1:.10f}")
# Check against simple fractions
for num in range(1, 20):
    for den in range(1, 20):
        if abs(num/den - ratio1) < 0.01:
            err = abs(num/den - ratio1) / ratio1 * 100
            print(f"  ≈ {num}/{den} = {num/den:.10f} (error: {err:.4f}%)")

# Hypothesis 2: ε = (π/e)² × (simple factor)
pi_e_squared = PI_OVER_E**2
ratio2 = EPSILON / pi_e_squared
print(f"\nε / (π/e)² = {ratio2:.10f}")
for num in range(1, 20):
    for den in range(1, 20):
        if abs(num/den - ratio2) < 0.01:
            err = abs(num/den - ratio2) / ratio2 * 100
            print(f"  ≈ {num}/{den} = {num/den:.10f} (error: {err:.4f}%)")

# Hypothesis 3: ε involves the difference (5/3 - φ)
diff_gamma_phi = GAMMA - PHI
ratio3 = EPSILON / diff_gamma_phi
print(f"\nε / (5/3 - φ) = {ratio3:.10f}")
print(f"  Note: 5/3 - φ = {diff_gamma_phi:.10f}")

# Hypothesis 4: second-order continued fraction correction
# The CF is [1, 1, 2, 159, ...]
# The 159 suggests the next term
cf_correction = 1 / (159 * 3)  # From CF structure
print(f"\nContinued fraction correction (1/159×3):")
print(f"  1/(159×3) = {cf_correction:.10f}")
print(f"  ε         = {EPSILON:.10f}")
print(f"  Ratio     = {EPSILON / cf_correction:.6f}")

# Hypothesis 5: ε = f(ln(2), π, e) where f is a polynomial
print()
print("-" * 70)
print("POLYNOMIAL SEARCH: ε = Σ c_ijk × ln(2)^i × π^j × e^k")
print("-" * 70)

best_poly = None
best_error = float('inf')

# Try small integer coefficients
for i in range(-3, 4):
    for j in range(-3, 4):
        for k in range(-3, 4):
            if i == j == k == 0:
                continue

            try:
                term = (LN2**i) * (PI**j) * (E**k)
            except:
                continue

            if abs(term) < 1e-10 or abs(term) > 1e10:
                continue

            coeff = EPSILON / term

            # Check if coefficient is a simple fraction
            for num in range(-10, 11):
                for den in range(1, 11):
                    if num == 0:
                        continue
                    frac = num / den
                    predicted = frac * term
                    rel_error = abs(predicted - EPSILON) / abs(EPSILON)

                    if rel_error < best_error and rel_error < 0.001:
                        best_error = rel_error
                        best_poly = (num, den, i, j, k, rel_error * 100)

if best_poly:
    num, den, i, j, k, err = best_poly
    print(f"Best match: ε = ({num}/{den}) × ln(2)^{i} × π^{j} × e^{k}")
    print(f"Error: {err:.6f}%")
    predicted = (num/den) * (LN2**i) * (PI**j) * (E**k)
    print(f"Predicted ε = {predicted:.15f}")
    print(f"Actual ε    = {EPSILON:.15f}")

# The REAL identity
print()
print("=" * 70)
print("CONSTRUCTING THE EXACT IDENTITY")
print("=" * 70)
print()

# If π/e = (5/3)×ln(2) + ε, then we need ε exactly
# Let's see what ε actually IS in terms of the constants

# Key observation: the continued fraction tells us
# π/(e×ln(2)) = 5/3 + 1/(159 + ...)
# So: π/e = (5/3)×ln(2) + ln(2)/(159 + ...)

# The 159 is suspicious. Is it related to constants?
print(f"The continued fraction term 159:")
print(f"  159 = 160 - 1 = 2^5 × 5 - 1")
print(f"  159 ≈ π × 50.6 = {PI * 50.6:.2f}")
print(f"  159 ≈ e × 58.5 = {E * 58.5:.2f}")
print(f"  159 ≈ 100/ln(2) × 1.1 = {100/LN2 * 1.1:.2f}")

# More precise: what is 159 in terms of constants?
for a in range(-5, 6):
    for b in range(-5, 6):
        for c in range(-5, 6):
            if a == b == c == 0:
                continue
            try:
                expr = (PI**a) * (E**b) * (LN2**c)
                if abs(expr - 159) < 1 and abs(expr) > 100:
                    print(f"  159 ≈ π^{a} × e^{b} × ln(2)^{c} = {expr:.4f}")
            except:
                pass

# Alternative: express the FULL identity
print()
print("-" * 70)
print("THE FULLER PICTURE")
print("-" * 70)
print()

# π/e = (5/3)×ln(2) × (1 + δ) where δ is small
delta = PI_OVER_E / FIVE_THIRDS_LN2 - 1
print(f"π/e = (5/3)×ln(2) × (1 + δ)")
print(f"where δ = {delta:.15f}")
print()

# Is δ expressible?
print(f"δ × 1000 = {delta * 1000:.10f}")
print(f"δ × 10000 = {delta * 10000:.10f}")
print()

# Check: δ ≈ 1/2400?
print(f"1/2400 = {1/2400:.15f}")
print(f"1/2393 = {1/2393:.15f}")  # Closer match
print(f"δ      = {delta:.15f}")
print()

# Is 2393 special?
print(f"2393 factorization: {2393} = ?")
# 2393 is prime!
print(f"  2393 is prime")
print(f"  2393 ≈ 3 × 797 + 2 = {3*797+2}")
print(f"  Note: 797/478 is the NEXT convergent after 5/3!")

# Final identity attempt
print()
print("=" * 70)
print("CANDIDATE IDENTITY")
print("=" * 70)
print()

# The pattern: 5/3 and 797/478 are consecutive convergents
# 5/3 = 1.666666...
# 797/478 = 1.667364...
# Actual = 1.667362...

actual_ratio = PI_OVER_E / LN2
print(f"π / (e × ln(2)) = {actual_ratio:.15f}")
print()
print("Continued fraction convergents:")
print(f"  5/3      = {5/3:.15f}  (error: {abs(5/3 - actual_ratio)/actual_ratio*100:.6f}%)")
print(f"  797/478  = {797/478:.15f}  (error: {abs(797/478 - actual_ratio)/actual_ratio*100:.6f}%)")
print()

# The mediant?
mediant_num = 5 + 797
mediant_den = 3 + 478
print(f"Mediant (5+797)/(3+478) = {mediant_num}/{mediant_den} = {mediant_num/mediant_den:.15f}")
print(f"  Error: {abs(mediant_num/mediant_den - actual_ratio)/actual_ratio*100:.6f}%")

# Check: is the actual ratio algebraic?
print()
print("-" * 70)
print("ALGEBRAIC CHECK")
print("-" * 70)
print()
print("If π/(e×ln(2)) were algebraic, we'd have:")
print("  π = α × e × ln(2) for some algebraic α")
print()
print("But π, e, and ln(2) are all transcendental and")
print("algebraically independent (conjectured, not proven).")
print()
print("So the relationship π/e ≈ (5/3)×ln(2) is likely a")
print("'near-miss' - close but not exact.")
print()
print("HOWEVER: The fact that 5/3 = γ (adiabatic index) appears")
print("suggests this near-miss has PHYSICAL meaning, even if")
print("it's not an exact mathematical identity.")


if __name__ == "__main__":
    pass
