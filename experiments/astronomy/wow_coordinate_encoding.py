#!/usr/bin/env python3
"""
COORDINATE ENCODING HYPOTHESIS

The sequence [6, 14, 26, 30, 19, 5] might encode the signal's source coordinates!

Evidence:
- 10×seq[2] + seq[3] = 260 + 30 = 290 ≈ RA in degrees (19h22m = 290°)
- seq[4] = 19 = RA hours
- seq[2] = 26 ≈ |Dec| = 27°

Is this just coincidence, or is the signal pointing to itself?

Usage:
    python wow_coordinate_encoding.py
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic

try:
    from astropy.coordinates import Galactic
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False


def main():
    print("=" * 70)
    print("COORDINATE ENCODING HYPOTHESIS")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]

    # =========================================================================
    # THE WOW! SIGNAL POSITION
    # =========================================================================
    print("\n" + "=" * 70)
    print("ACTUAL WOW! SIGNAL POSITION")
    print("=" * 70)

    # The Wow! signal position has uncertainty due to the dual-beam
    # Two possible positions:
    # 1) RA 19h22m24.64s, Dec -27°03'
    # 2) RA 19h25m31.25s, Dec -26°57'

    print(f"""
  The Big Ear had TWO feed horns, so the source could be at:

  Position A (positive horn):
    RA = 19h 22m 24.64s = 19.373° × 15 = 290.60°
    Dec = -27° 03' = -27.05°

  Position B (negative horn):
    RA = 19h 25m 31.25s = 19.425° × 15 = 291.38°
    Dec = -26° 57' = -26.95°

  Center estimate:
    RA ≈ 290.5° - 291.5°
    Dec ≈ -27°
""")

    ra_deg_A = 19 + 22/60 + 24.64/3600
    ra_deg_A_total = ra_deg_A * 15  # Convert to degrees
    dec_deg_A = -(27 + 3/60)

    ra_deg_B = 19 + 25/60 + 31.25/3600
    ra_deg_B_total = ra_deg_B * 15
    dec_deg_B = -(26 + 57/60)

    print(f"  Position A: RA = {ra_deg_A_total:.2f}°, Dec = {dec_deg_A:.2f}°")
    print(f"  Position B: RA = {ra_deg_B_total:.2f}°, Dec = {dec_deg_B:.2f}°")

    # =========================================================================
    # ENCODING TEST 1: Direct sequence values
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENCODING TEST 1: DIRECT VALUES")
    print("=" * 70)

    print(f"\n  Sequence: {seq}")
    print(f"\n  Comparing to actual coordinates:")
    print(f"    seq[4] = {seq[4]} vs RA hours = 19 → EXACT MATCH!")
    print(f"    seq[2] = {seq[2]} vs |Dec| = 27 → ERROR: {abs(seq[2] - 27)}")
    print(f"    seq[5] = {seq[5]} vs Dec minutes/10? = 0.3 → ?")

    # =========================================================================
    # ENCODING TEST 2: Combined values
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENCODING TEST 2: COMBINED VALUES")
    print("=" * 70)

    print(f"\n  Testing combinations:")

    # RA encoding attempts
    print(f"\n  RA encodings:")
    print(f"    10×seq[2] + seq[3] = {10*seq[2] + seq[3]} vs {ra_deg_A_total:.1f}° → ERROR: {abs(10*seq[2] + seq[3] - ra_deg_A_total):.1f}°")
    print(f"    seq[4]×15 + seq[3]/2 = {seq[4]*15 + seq[3]/2:.1f} vs {ra_deg_A_total:.1f}° → ERROR: {abs(seq[4]*15 + seq[3]/2 - ra_deg_A_total):.1f}°")
    print(f"    seq[4]×15 + seq[1] = {seq[4]*15 + seq[1]} vs {ra_deg_A_total:.1f}° → ERROR: {abs(seq[4]*15 + seq[1] - ra_deg_A_total):.1f}°")

    # Dec encoding attempts
    print(f"\n  Dec encodings:")
    print(f"    -seq[2] = {-seq[2]} vs {dec_deg_A:.1f}° → ERROR: {abs(-seq[2] - dec_deg_A):.1f}°")
    print(f"    -(seq[2] + seq[5]/6) = {-(seq[2] + seq[5]/6):.2f} vs {dec_deg_A:.2f}° → ERROR: {abs(-(seq[2] + seq[5]/6) - dec_deg_A):.2f}°")

    # =========================================================================
    # ENCODING TEST 3: The 290 match
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE 290 MATCH")
    print("=" * 70)

    print(f"""
  10 × seq[2] + seq[3] = 10 × 26 + 30 = 290

  Actual RA in degrees:
    Position A: {ra_deg_A_total:.2f}°
    Position B: {ra_deg_B_total:.2f}°

  Difference:
    290 - {ra_deg_A_total:.2f} = {290 - ra_deg_A_total:.2f}°
    290 - {ra_deg_B_total:.2f} = {290 - ra_deg_B_total:.2f}°

  This is within ~1° of the actual position!
  Is this coincidence?
""")

    # =========================================================================
    # PROBABILITY OF COINCIDENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROBABILITY OF COINCIDENCE")
    print("=" * 70)

    print(f"""
  What's the probability that random values would encode the position?

  For RA:
    - Sky covers 0-360° in RA
    - We need to match within ~1°
    - Using 10×A + B for A,B in 0-36 gives 0-396
    - Probability of hitting ±1° of any target ≈ 2/396 ≈ 0.5%

  For Dec:
    - Sky covers -90° to +90°
    - We need to match within ~1°
    - Using a single value 0-36 to encode |Dec| 0-90
    - Probability of exact match: 1/37 ≈ 2.7%

  For RA hours:
    - 24 possible hours
    - We hit 19 exactly
    - Probability: 1/24 ≈ 4%

  Combined probability (if independent):
    - P(all three) ≈ 0.005 × 0.027 × 0.04 ≈ 0.0005%

  BUT: We chose the encoding scheme after seeing the data.
  This is "p-hacking" unless the encoding is somehow natural.
""")

    # =========================================================================
    # TESTING ALL POSSIBLE ENCODINGS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SYSTEMATIC ENCODING SEARCH")
    print("=" * 70)

    print(f"\n  Searching for ANY encoding that gives RA ≈ {ra_deg_A_total:.1f}°...")

    found_ra = []
    for i in range(6):
        for j in range(6):
            for mult in [1, 10, 15, 12, 100]:
                val = mult * seq[i] + seq[j]
                if abs(val - ra_deg_A_total) < 2:
                    found_ra.append((mult, i, j, val))

    if found_ra:
        print(f"  Found {len(found_ra)} encodings within 2°:")
        for mult, i, j, val in found_ra:
            print(f"    {mult}×seq[{i}] + seq[{j}] = {mult}×{seq[i]} + {seq[j]} = {val}")
    else:
        print("  No simple encodings found.")

    print(f"\n  Searching for ANY encoding that gives Dec ≈ {abs(dec_deg_A):.1f}°...")

    found_dec = []
    for i in range(6):
        if abs(seq[i] - abs(dec_deg_A)) < 2:
            found_dec.append((i, seq[i]))
        for j in range(6):
            val = seq[i] + seq[j]
            if abs(val - abs(dec_deg_A)) < 2:
                found_dec.append((f"{i}+{j}", val))

    if found_dec:
        print(f"  Found {len(found_dec)} encodings within 2°:")
        for encoding, val in found_dec[:5]:
            print(f"    seq[{encoding}] = {val}")

    # =========================================================================
    # THE SELF-REFERENTIAL INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("SELF-REFERENTIAL INTERPRETATION")
    print("=" * 70)

    print(f"""
  IF the signal encodes its own position:

  This would be the ultimate self-reference:
    - Transmitted on 21 cm (hydrogen, universal)
    - Angular dynamics divide by 21
    - Encodes its own sky position

  The message would be:
    "Look here (these coordinates) on this frequency (21 cm)
     to find this message (about 21 and geometry)."

  It's like a cosmic "You Are Here" sign.

  But this interpretation has problems:
    1. Why would a natural source encode its position?
    2. We had to search for the encoding scheme
    3. The match is approximate, not exact
""")

    # =========================================================================
    # ALTERNATIVE: RANDOM CHANCE CALCULATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("ALTERNATIVE: RANDOM CHANCE")
    print("=" * 70)

    # How many possible "encodings" did we try?
    # For RA: 6 × 6 × 5 = 180 combinations of (mult, i, j)
    # Plus direct values: 6
    # Total: ~200 encodings

    # For any target in [0, 360], probability of a hit within 2° is about 4/360 = 1.1%
    # With 200 trials, expected hits: 200 × 0.011 = 2.2

    print(f"""
  Number of encoding schemes tried: ~200
  Range of possible values: 0-400+
  Probability of hitting target within 2°: ~1%
  Expected number of "hits" by chance: ~2

  We found ~2-3 hits.

  CONCLUSION: The coordinate encoding could be coincidence.

  The 10×26 + 30 = 290 match is striking, but not statistically
  significant given the number of schemes we tried.
""")

    # =========================================================================
    # WHAT IF WE CHECK OTHER SIGNALS?
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: APPLY TO RANDOM SIGNALS")
    print("=" * 70)

    # Generate random 6-value sequences and check if they "encode" a position
    np.random.seed(42)
    n_trials = 10000
    matches = 0

    for _ in range(n_trials):
        # Random sequence with values 0-36, sum = 100
        while True:
            rand_seq = np.random.randint(0, 37, size=6)
            if rand_seq.sum() == 100:
                break

        # Check if 10×rand[2] + rand[3] is within 2 of 290
        encoded_ra = 10 * rand_seq[2] + rand_seq[3]
        if abs(encoded_ra - 290) < 2:
            matches += 1

    print(f"\n  Checking {n_trials} random sequences (sum=100, values 0-36):")
    print(f"  Sequences where 10×val[2] + val[3] ≈ 290 (±2): {matches}")
    print(f"  Probability: {matches/n_trials*100:.2f}%")

    # What about matching BOTH RA and Dec?
    matches_both = 0
    for _ in range(n_trials):
        while True:
            rand_seq = np.random.randint(0, 37, size=6)
            if rand_seq.sum() == 100:
                break

        encoded_ra = 10 * rand_seq[2] + rand_seq[3]
        encoded_dec = rand_seq[2]

        if abs(encoded_ra - 290) < 2 and abs(encoded_dec - 27) < 2:
            matches_both += 1

    print(f"\n  Matching BOTH RA (290±2) AND Dec (27±2): {matches_both}")
    print(f"  Probability: {matches_both/n_trials*100:.3f}%")

    # =========================================================================
    # FINAL ASSESSMENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    print(f"""
  THE COORDINATE ENCODING:
    10 × 26 + 30 = 290 ≈ RA in degrees ✓
    26 ≈ |Dec| = 27 ✓
    19 = RA hours ✓

  PROBABILITY ASSESSMENT:
    Random sequences matching both criteria: ~{matches_both/n_trials*100:.1f}%

  This is {'unusual' if matches_both/n_trials < 0.01 else 'not particularly rare'}.

  INTERPRETATION:
    The sequence MIGHT encode coordinates, or it might be
    coincidence given the many ways we could interpret 6 numbers.

    If it DOES encode coordinates, the message is:
    "The source of this signal is at these coordinates" -
    a self-locating message that only makes sense for an
    artificial beacon designed to be found.

    If coincidence, it's just another intriguing pattern
    in a signal full of patterns.
""")


if __name__ == "__main__":
    main()
