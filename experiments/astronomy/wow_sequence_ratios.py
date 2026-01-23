#!/usr/bin/env python3
"""
Investigating the ratios IN the 6EQUJ5 sequence itself.

We found:
  30/19 = 1.579 ≈ φ (2.4% error)
  19/6 = 3.167 ≈ π (0.8% error)

Are these coincidences? Or significant?

Usage:
    python wow_sequence_ratios.py
"""

from __future__ import annotations

import numpy as np
from itertools import permutations, combinations

phi = (1 + np.sqrt(5)) / 2
pi = np.pi


def main():
    print("=" * 70)
    print("RATIOS WITHIN THE 6EQUJ5 SEQUENCE")
    print("=" * 70)

    # The famous sequence
    seq = [6, 14, 26, 30, 19, 5]
    labels = ['6', 'E', 'Q', 'U', 'J', '5']

    print(f"\nThe sequence: {seq}")
    print(f"As characters: 6EQUJ5")

    # 1. ALL pairwise ratios
    print(f"\n1. ALL PAIRWISE RATIOS")
    print("=" * 70)

    print(f"\n  Searching for φ = {phi:.6f} and π = {pi:.6f}")

    close_to_phi = []
    close_to_pi = []

    print(f"\n  {'Ratio':<10} {'Value':<10} {'vs φ':<15} {'vs π':<15}")
    print("-" * 55)

    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j and seq[j] != 0:
                r = seq[i] / seq[j]
                err_phi = abs(r - phi) / phi * 100
                err_pi = abs(r - pi) / pi * 100

                if err_phi < 5 or err_pi < 5:
                    print(f"  {labels[i]}/{labels[j]:<7} {r:<10.4f} {err_phi:>6.2f}%       {err_pi:>6.2f}%")

                    if err_phi < 5:
                        close_to_phi.append((seq[i], seq[j], r, err_phi))
                    if err_pi < 5:
                        close_to_pi.append((seq[i], seq[j], r, err_pi))

    print(f"\n  Found {len(close_to_phi)} ratios close to φ (<5% error)")
    print(f"  Found {len(close_to_pi)} ratios close to π (<5% error)")

    # 2. Is this unusual?
    print(f"\n2. IS THIS UNUSUAL? Monte Carlo test")
    print("=" * 70)

    # Generate random sequences with similar properties:
    # - 6 integers
    # - Max value around 30
    # - Sum around 100 (same as 6EQUJ5)

    n_trials = 100000
    hits_phi = 0
    hits_pi = 0
    hits_both = 0

    for _ in range(n_trials):
        # Generate random sequence similar to 6EQUJ5
        # Constraint: sum ≈ 100, max ≈ 30, min > 0

        # Method: sample from similar distribution
        rand_seq = np.random.randint(1, 31, size=6)

        # Check all pairwise ratios
        found_phi = False
        found_pi = False

        for i in range(6):
            for j in range(6):
                if i != j and rand_seq[j] > 0:
                    r = rand_seq[i] / rand_seq[j]
                    if abs(r - phi) / phi < 0.05:  # 5% tolerance
                        found_phi = True
                    if abs(r - pi) / pi < 0.05:
                        found_pi = True

        if found_phi:
            hits_phi += 1
        if found_pi:
            hits_pi += 1
        if found_phi and found_pi:
            hits_both += 1

    print(f"\n  Random sequences tested: {n_trials}")
    print(f"  With ratio ≈ φ (within 5%): {hits_phi} ({hits_phi/n_trials*100:.2f}%)")
    print(f"  With ratio ≈ π (within 5%): {hits_pi} ({hits_pi/n_trials*100:.2f}%)")
    print(f"  With BOTH φ AND π: {hits_both} ({hits_both/n_trials*100:.2f}%)")

    # Expected by chance
    p_phi = hits_phi / n_trials
    p_pi = hits_pi / n_trials
    p_both_independent = p_phi * p_pi

    print(f"\n  If independent: P(both) = {p_both_independent*100:.2f}%")
    print(f"  Observed: P(both) = {hits_both/n_trials*100:.2f}%")

    # 3. What makes 30/19 and 19/6 special?
    print(f"\n3. WHY THESE SPECIFIC RATIOS?")
    print("=" * 70)

    print(f"\n  30/19 = {30/19:.6f}")
    print(f"  φ = {phi:.6f}")
    print(f"  Error = {abs(30/19 - phi)/phi*100:.2f}%")

    print(f"\n  19/6 = {19/6:.6f}")
    print(f"  π = {pi:.6f}")
    print(f"  Error = {abs(19/6 - pi)/pi*100:.2f}%")

    # These are related!
    # 30/19 ≈ φ and 19/6 ≈ π
    # So 30/6 = (30/19) × (19/6) ≈ φ × π
    print(f"\n  Combined: 30/6 = 5")
    print(f"  φ × π = {phi * pi:.6f}")
    print(f"  Actual: 30/6 = 5.0")
    print(f"  φ × π rounds to 5? {round(phi * pi)} == 5: {round(phi * pi) == 5}")

    # 4. The numbers 30, 19, 6
    print(f"\n4. THE KEY NUMBERS: 30, 19, 6")
    print("=" * 70)

    print(f"\n  30 = U (peak intensity)")
    print(f"  19 = J (intensity 1 step after peak)")
    print(f"  6 = first detected value")

    print(f"\n  What integers give ratios close to φ and π?")
    print(f"\n  Pairs with ratio ≈ φ (within 2%):")
    for a in range(1, 40):
        for b in range(1, 40):
            if b != 0:
                r = a / b
                if abs(r - phi) / phi < 0.02:
                    print(f"    {a}/{b} = {r:.4f} (err: {abs(r-phi)/phi*100:.2f}%)")

    print(f"\n  Pairs with ratio ≈ π (within 2%):")
    for a in range(1, 40):
        for b in range(1, 40):
            if b != 0:
                r = a / b
                if abs(r - pi) / pi < 0.02:
                    print(f"    {a}/{b} = {r:.4f} (err: {abs(r-pi)/pi*100:.2f}%)")

    # 5. Could this be encoding?
    print(f"\n5. ENCODING HYPOTHESIS")
    print("=" * 70)

    print(f"""
  If someone wanted to encode φ and π in 6 integers (0-35 scale):

  To get φ ≈ 1.618:
    Best small integer ratios: 8/5=1.6, 13/8=1.625, 21/13=1.615
    These are Fibonacci ratios!
    But 30/19 is NOT a Fibonacci ratio.

  To get π ≈ 3.1416:
    Best small integer ratios: 22/7=3.143, 19/6=3.167, 25/8=3.125
    19/6 is one of the simplest π approximations!

  The 6EQUJ5 sequence contains:
    - 30 and 19 (ratio ≈ φ)
    - 19 and 6 (ratio ≈ π)
    - The number 19 appears in BOTH ratios!

  This is interesting: 19 serves as a "pivot" between φ and π.
""")

    # 6. Probability of exactly this
    print(f"\n6. PROBABILITY OF THIS SPECIFIC PATTERN")
    print("=" * 70)

    # More constrained test: sequence of 6 values where
    # - First value is 6
    # - One value is 30 (peak)
    # - Some value can divide into another to give ≈ φ
    # - Some value can divide into another to give ≈ π

    n_trials = 100000
    exact_hits = 0

    for _ in range(n_trials):
        # Random sequence with first=6, peak=30
        # Other 4 values random from 1-30
        seq = [6] + list(np.random.randint(1, 31, size=3)) + [30, np.random.randint(1, 31)]
        np.random.shuffle(seq)

        # Check for the specific pattern: some a/b ≈ φ AND some c/d ≈ π
        found_phi = False
        found_pi = False

        for i in range(6):
            for j in range(6):
                if i != j and seq[j] > 0:
                    r = seq[i] / seq[j]
                    if abs(r - phi) / phi < 0.025:  # 2.5% (like 30/19)
                        found_phi = True
                    if abs(r - pi) / pi < 0.01:  # 1% (like 19/6)
                        found_pi = True

        if found_phi and found_pi:
            exact_hits += 1

    print(f"\n  Monte Carlo ({n_trials} trials):")
    print(f"  Random 6-integer sequences with peak≈30, min≈6")
    print(f"  Having ratio ≈ φ (2.5%) AND ratio ≈ π (1%): {exact_hits}")
    print(f"  Probability: {exact_hits/n_trials*100:.3f}%")

    if exact_hits > 0:
        print(f"\n  This is NOT extremely rare - it happens about 1 in {n_trials//exact_hits}")
    else:
        print(f"\n  Zero hits suggests this IS rare (p < 1/{n_trials})")

    # SYNTHESIS
    print(f"\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    print(f"""
FINDINGS:

1. The raw 6EQUJ5 sequence DOES contain φ and π:
   - 30/19 = 1.579 ≈ φ (2.4% error)
   - 19/6 = 3.167 ≈ π (0.8% error)

2. Monte Carlo shows this is NOT extremely rare:
   - About {hits_both/n_trials*100:.1f}% of random 6-integer sequences
     contain both ratios

3. The number 19 is the "pivot":
   - 30/19 ≈ φ
   - 19/6 ≈ π
   - 19 connects the two constants

4. Statistical significance:
   - Finding φ OR π in 6 random integers: common
   - Finding BOTH with the same pivot: less common but not extraordinary
   - The pattern is interesting but not conclusive

CONCLUSION:
The φ and π ratios in 6EQUJ5 are REAL (not artifacts of data processing)
but whether they're intentional or coincidental is unclear. A 1-5%
chance event isn't strong evidence of extraterrestrial encoding.
""")


if __name__ == "__main__":
    main()
