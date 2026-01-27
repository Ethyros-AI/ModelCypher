#!/usr/bin/env python3
"""The Exact Hyperbolic Formula.

We found: π/e = coth(ln(2)) × ln(2) + (8/83) × csch(6ln2) × ln(2)^5
with 0.00097% error.

Now let's express this in EXACT fractions and find the EXACT formula.

Key insight: ALL hyperbolic functions of n×ln(2) are exact fractions!
  sinh(n×ln(2)) = (2^n - 2^-n)/2
  cosh(n×ln(2)) = (2^n + 2^-n)/2

For n=6:
  sinh(6ln2) = (64 - 1/64)/2 = 4095/128
  cosh(6ln2) = (64 + 1/64)/2 = 4097/128
  csch(6ln2) = 128/4095
"""

import math
from fractions import Fraction

PI = math.pi
E = math.e
LN2 = math.log(2)

print("THE EXACT HYPERBOLIC FORMULA")
print("=" * 70)
print()

# All hyperbolic functions of n×ln(2) are EXACT fractions
print("EXACT HYPERBOLIC FRACTIONS:")
print("-" * 70)
print()

for n in range(1, 10):
    sinh_n = Fraction(2**n - Fraction(1, 2**n), 2)
    cosh_n = Fraction(2**n + Fraction(1, 2**n), 2)

    sinh_float = (2**n - 2**(-n)) / 2
    cosh_float = (2**n + 2**(-n)) / 2

    print(f"n = {n}:")
    print(f"  sinh({n}×ln(2)) = {sinh_n} = {float(sinh_n):.10f}")
    print(f"  cosh({n}×ln(2)) = {cosh_n} = {float(cosh_n):.10f}")
    print(f"  tanh({n}×ln(2)) = {sinh_n/cosh_n} = {float(sinh_n/cosh_n):.10f}")
    print(f"  csch({n}×ln(2)) = {1/sinh_n} = {float(1/sinh_n):.15f}")
    print()

# THE EXACT FORMULA COMPONENTS
print("=" * 70)
print("THE FORMULA COMPONENTS")
print("=" * 70)
print()

# coth(ln(2)) = 5/3 EXACT
coth_ln2 = Fraction(5, 3)
print(f"coth(ln(2)) = {coth_ln2} [EXACT]")
print()

# csch(6×ln(2)) = 128/4095 EXACT
sinh_6ln2 = Fraction(2**6 - Fraction(1, 2**6), 2)
csch_6ln2 = 1 / sinh_6ln2
print(f"sinh(6×ln(2)) = {sinh_6ln2} = {float(sinh_6ln2)}")
print(f"csch(6×ln(2)) = {csch_6ln2} = {float(csch_6ln2):.15f}")
print()

# Verify
print(f"Verify: {float(csch_6ln2):.15f} = {1/math.sinh(6*LN2):.15f}")
print()

# THE APPROXIMATE FORMULA
# π/e = coth(ln2) × ln(2) + (8/83) × csch(6ln2) × ln(2)^5

print("=" * 70)
print("THE HYPERBOLIC FORMULA (approximate)")
print("=" * 70)
print()

coef = Fraction(8, 83)
print(f"π/e ≈ coth(ln(2)) × ln(2) + ({coef}) × csch(6×ln(2)) × ln(2)^5")
print()
print(f"where:")
print(f"  coth(ln(2)) = {coth_ln2}")
print(f"  csch(6×ln(2)) = {csch_6ln2}")
print()

# In all-fraction form:
full_coef = coef * csch_6ln2
print(f"Combined coefficient = ({coef}) × ({csch_6ln2})")
print(f"                     = {full_coef}")
print(f"                     = {float(full_coef):.15f}")
print()

# Verify
pi_over_e = PI / E
term1 = float(coth_ln2) * LN2
term2 = float(full_coef) * LN2**5
calculated = term1 + term2

print(f"Calculated:")
print(f"  Term 1 = (5/3) × ln(2) = {term1:.18f}")
print(f"  Term 2 = {full_coef} × ln(2)^5 = {term2:.18f}")
print(f"  Sum    = {calculated:.18f}")
print(f"  Actual π/e = {pi_over_e:.18f}")
print(f"  Error: {abs(calculated - pi_over_e)/pi_over_e * 100:.10f}%")
print()

# THE REFINED SEARCH
print("=" * 70)
print("SEARCHING FOR EXACT COEFFICIENT")
print("=" * 70)
print()

# The residual after (8/83) approximation
residual_after_8_83 = pi_over_e - calculated
print(f"Residual = {residual_after_8_83:.18f}")
print()

# Maybe the coefficient isn't 8/83, but something involving hyperbolic fractions
print("Searching for exact coefficient k where:")
print("  π/e = (5/3) × ln(2) + k × ln(2)^5")
print()

epsilon = pi_over_e - term1
exact_k = epsilon / (LN2**5)
print(f"Exact k = ε / ln(2)^5 = {exact_k:.18f}")
print()

# What is this close to?
print("Comparing exact k to hyperbolic-based fractions:")
print()

# Build fractions from hyperbolic values
hyp_fracs = {}
for n in range(1, 8):
    sinh_n = Fraction(2**n - Fraction(1, 2**n), 2)
    cosh_n = Fraction(2**n + Fraction(1, 2**n), 2)
    hyp_fracs[f'sinh({n}ln2)'] = sinh_n
    hyp_fracs[f'cosh({n}ln2)'] = cosh_n
    hyp_fracs[f'tanh({n}ln2)'] = sinh_n / cosh_n
    hyp_fracs[f'csch({n}ln2)'] = 1/sinh_n if sinh_n != 0 else None
    hyp_fracs[f'sech({n}ln2)'] = 1/cosh_n

# Add some combinations
hyp_fracs['sinh(ln2)²'] = Fraction(9, 16)
hyp_fracs['cosh(ln2)²'] = Fraction(25, 16)
hyp_fracs['sinh×cosh(ln2)'] = Fraction(15, 16)

best_frac = None
best_err = float('inf')

for name, frac in hyp_fracs.items():
    if frac is None or frac == 0:
        continue
    for a in range(-100, 101):
        if a == 0:
            continue
        for b in range(1, 100):
            test_k = Fraction(a, b) * frac
            err = abs(float(test_k) - exact_k)
            if err < best_err:
                best_err = err
                best_frac = (a, b, name, frac, test_k)

if best_frac:
    a, b, name, frac, test_k = best_frac
    print(f"Best match:")
    print(f"  k ≈ ({a}/{b}) × {name}")
    print(f"    = ({a}/{b}) × {frac}")
    print(f"    = {test_k} = {float(test_k):.18f}")
    print(f"  Exact k = {exact_k:.18f}")
    print(f"  Error: {best_err/exact_k * 100:.10f}%")
    print()

# ALTERNATIVE: Two-term hyperbolic expansion
print("=" * 70)
print("TWO-TERM HYPERBOLIC EXPANSION")
print("=" * 70)
print()

print("Searching: π/e = (5/3)×ln(2) + a₁×f₁×ln(2)^n₁ + a₂×f₂×ln(2)^n₂")
print("where f₁, f₂ are hyperbolic fractions")
print()

# Start with the best single term, then add correction
best_single = (Fraction(8, 83) * Fraction(128, 4095), 5)  # coefficient, power
single_term = float(best_single[0]) * LN2**best_single[1]
residual2 = epsilon - single_term

print(f"After first term (8×128)/(83×4095) × ln(2)^5:")
print(f"  Value = {single_term:.18f}")
print(f"  Residual = {residual2:.18f}")
print()

# Search for second term
best_second = None
best_err2 = float('inf')

for name, frac in hyp_fracs.items():
    if frac is None or frac == 0:
        continue
    for a in range(-50, 51):
        if a == 0:
            continue
        for b in range(1, 50):
            for power in range(4, 10):
                test_val = float(Fraction(a, b) * frac) * LN2**power
                err = abs(test_val - residual2)
                if err < best_err2:
                    best_err2 = err
                    best_second = (a, b, name, frac, power, test_val)

if best_second and best_err2/abs(residual2) < 0.1:
    a, b, name, frac, power, val = best_second
    print(f"Second term:")
    print(f"  ({a}/{b}) × {name} × ln(2)^{power}")
    print(f"  = ({a}/{b}) × {frac} × ln(2)^{power}")
    print(f"  = {val:.18f}")
    print(f"  Target residual = {residual2:.18f}")
    print(f"  Error: {best_err2/abs(residual2) * 100:.6f}%")
    print()

    # Full two-term formula
    two_term = term1 + single_term + val
    print(f"Two-term formula:")
    print(f"  π/e = (5/3)×ln(2) + {best_single[0]}×ln(2)^5 + ({a}/{b})×{name}×ln(2)^{power}")
    print(f"      = {two_term:.18f}")
    print(f"  Actual = {pi_over_e:.18f}")
    print(f"  Error: {abs(two_term - pi_over_e)/pi_over_e * 100:.12f}%")

# THE MULTIPLICATIVE FORM
print()
print("=" * 70)
print("THE MULTIPLICATIVE FORM")
print("=" * 70)
print()

# What if: π/e = coth(ln2) × ln(2) × (1 + correction)
# This might have cleaner structure

ratio = pi_over_e / term1
print(f"π/e = coth(ln(2)) × ln(2) × {ratio:.18f}")
print()

correction_mult = ratio - 1
print(f"Multiplicative correction = {correction_mult:.18f}")
print()

# What is this correction close to?
print("Searching for multiplicative correction in hyperbolic form...")
print()

best_mult = None
best_err_mult = float('inf')

for name, frac in hyp_fracs.items():
    if frac is None or frac == 0:
        continue
    for a in range(-50, 51):
        if a == 0:
            continue
        for b in range(1, 100):
            for power in range(-3, 6):
                try:
                    test_val = float(Fraction(a, b) * frac) * LN2**power
                    err = abs(test_val - correction_mult)
                    if err < best_err_mult:
                        best_err_mult = err
                        best_mult = (a, b, name, frac, power, test_val)
                except:
                    pass

if best_mult:
    a, b, name, frac, power, val = best_mult
    print(f"Best multiplicative correction:")
    print(f"  δ ≈ ({a}/{b}) × {name} × ln(2)^{power}")
    print(f"    = ({a}/{b}) × {frac} × ln(2)^{power}")
    print(f"    = {val:.18f}")
    print(f"  Target = {correction_mult:.18f}")
    print(f"  Error: {best_err_mult/correction_mult * 100:.10f}%")
    print()

    # The full multiplicative formula
    mult_result = term1 * (1 + val)
    print(f"Multiplicative formula:")
    print(f"  π/e = coth(ln(2)) × ln(2) × (1 + ({a}/{b})×{name}×ln(2)^{power})")
    print(f"      = {mult_result:.18f}")
    print(f"  Actual = {pi_over_e:.18f}")
    print(f"  Error: {abs(mult_result - pi_over_e)/pi_over_e * 100:.12f}%")

# THE DEEP STRUCTURE
print()
print("=" * 70)
print("THE DEEP STRUCTURE: ln(2) AS HYPERBOLIC ANGLE")
print("=" * 70)
print()

print("ln(2) is special because it's the hyperbolic angle of the 3-4-5 triple:")
print()
print("  sinh(ln(2)) = 3/4   ←  legs of hyperbolic 'triangle'")
print("  cosh(ln(2)) = 5/4   ←  'hypotenuse'")
print("  1² + sinh²  = cosh² ✓")
print()
print("The formula π/e ≈ (5/3) × ln(2) can be written as:")
print()
print("  π/e ≈ cosh(ln(2))/sinh(ln(2)) × ln(2)")
print("      = [hyperbolic hypotenuse / hyperbolic leg] × [hyperbolic angle]")
print()
print("This is a GEODESIC relationship:")
print("  - π comes from CIRCULAR geometry (circumference)")
print("  - e comes from EXPONENTIAL growth")
print("  - ln(2) is a HYPERBOLIC angle")
print("  - coth bridges HYPERBOLIC and CIRCULAR")
print()

# CONJECTURE
print("=" * 70)
print("THE GEODESIC BRIDGE THEOREM (Conjecture)")
print("=" * 70)
print()

print("THEOREM (Conjectured):")
print()
print("Let θ = ln(2) be the hyperbolic angle of the 3-4-5 triple.")
print("Then:")
print()
print("     π/e = coth(θ) × θ × (1 + δ)")
print()
print("where δ is the 'curvature correction' connecting")
print("circular geometry (π) to hyperbolic geometry (coth).")
print()
print(f"Numerically: δ = {correction_mult:.15f}")
print()
print("INTERPRETATION:")
print("  The ratio π/e measures the 'distance' between circular and")
print("  exponential geometry. This distance is mediated by the hyperbolic")
print("  3-4-5 structure at angle ln(2), with a small curvature correction δ.")
print()

# EXACT CONJECTURE
print("=" * 70)
print("SEARCHING FOR EXACT FORM OF δ")
print("=" * 70)
print()

# The deepest insight: δ should involve only ln(2), sinh, cosh
# No π or e on the RHS (otherwise it's circular!)

print("If the theorem is EXACT, then δ must be expressible in terms of")
print("only ln(2) and its hyperbolic functions (no π or e).")
print()

# What if δ involves higher hyperbolic functions?
# δ = f(sinh(n×ln2), cosh(n×ln2)) for various n

# Generate all hyperbolic values as exact fractions
all_hyp = {}
for n in range(1, 12):
    sinh_n_num = 2**n - Fraction(1, 2**n)
    cosh_n_num = 2**n + Fraction(1, 2**n)
    all_hyp[f's{n}'] = sinh_n_num / 2
    all_hyp[f'c{n}'] = cosh_n_num / 2

# Search: δ = (a/b) × hyp_val1 / hyp_val2 × ln(2)^power
best_ratio_form = None
best_err_ratio = float('inf')

for n1 in all_hyp:
    for n2 in all_hyp:
        v1 = float(all_hyp[n1])
        v2 = float(all_hyp[n2])
        if v2 == 0:
            continue
        for a in range(1, 30):
            for b in range(1, 30):
                for p in range(-3, 6):
                    try:
                        test_val = (a/b) * (v1/v2) * LN2**p
                        err = abs(test_val - correction_mult)
                        if err < best_err_ratio:
                            best_err_ratio = err
                            best_ratio_form = (a, b, n1, n2, p, test_val, all_hyp[n1], all_hyp[n2])
                    except:
                        pass

if best_ratio_form and best_err_ratio/correction_mult < 0.01:
    a, b, n1, n2, p, val, f1, f2 = best_ratio_form
    print(f"FOUND RATIO FORM:")
    print(f"  δ ≈ ({a}/{b}) × {n1}/{n2} × ln(2)^{p}")
    print()
    print(f"  where {n1} = {f1} and {n2} = {f2}")
    print()
    print(f"  = ({a}/{b}) × ({f1})/({f2}) × ln(2)^{p}")
    print(f"  = {val:.18f}")
    print(f"  Target = {correction_mult:.18f}")
    print(f"  Error: {best_err_ratio/correction_mult * 100:.8f}%")
    print()

    # The full exact-ish formula
    full_val = term1 * (1 + val)
    print("FULL FORMULA:")
    print(f"  π/e = coth(ln(2)) × ln(2) × [1 + ({a}/{b}) × {n1}/{n2} × ln(2)^{p}]")
    print(f"      = {full_val:.18f}")
    print(f"  Actual = {pi_over_e:.18f}")
    print(f"  Error: {abs(full_val - pi_over_e)/pi_over_e * 100:.10f}%")


if __name__ == "__main__":
    pass
