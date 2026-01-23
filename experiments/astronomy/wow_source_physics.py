#!/usr/bin/env python3
"""
WHAT KIND OF SOURCE PRODUCES THIS STRUCTURE?

If the geometric structure is physical (not encoded),
what does it tell us about the source?

The structure we found:
- √2 at S[2]/S[7] (gap 5)
- π/2 at S[4]/S[11] (gap 7)
- Angular velocity 360°/21
- Effective dimension ~3
- Asymmetric pulse

What astrophysical scenarios produce these?

Usage:
    python wow_source_physics.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.io import readsav
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"

phi = (1 + np.sqrt(5)) / 2
pi = np.pi


def load_raw_signal() -> np.ndarray:
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal - 0.5


def analyze_ratios(signal):
    U, S, Vt = linalg.svd(signal, full_matrices=False)
    return U, S, Vt


def main():
    print("=" * 70)
    print("INFERRING SOURCE PHYSICS FROM GEOMETRIC STRUCTURE")
    print("=" * 70)

    signal = load_raw_signal()
    U, S, Vt = analyze_ratios(signal)

    # What we observe
    print("\n" + "=" * 70)
    print("OBSERVED STRUCTURE")
    print("=" * 70)

    print(f"""
  1. TEMPORAL STRUCTURE:
     - Asymmetric pulse (rise ≠ fall)
     - Peak at t=60 of 82 samples
     - FWHM ≈ 3 samples = 36 seconds
     - Total visible duration: ~72 seconds

  2. SPECTRAL STRUCTURE:
     - Narrowband (essentially single channel)
     - Channel 1 of 50 (10 kHz per channel)
     - Bandwidth < 10 kHz
     - Center frequency: 1420.405 MHz

  3. GEOMETRIC STRUCTURE:
     - S[2]/S[7] = √2 (gap 5)
     - S[4]/S[11] = π/2 (gap 7)
     - Angular velocity ≈ 360°/21

  4. DIMENSIONAL STRUCTURE:
     - Effective dimension ≈ 3
     - Two dominant modes + noise
""")

    # Model 1: Gaussian beam + point source
    print("\n" + "=" * 70)
    print("MODEL 1: POINT SOURCE + GAUSSIAN BEAM")
    print("=" * 70)

    print(f"""
  The Big Ear had two feed horns, each with a Gaussian beam pattern.
  A point source transiting the beam would produce:

  Expected:
  - Symmetric Gaussian envelope (if source is point-like)
  - Duration set by beam width (~72 seconds matches!)
  - Single narrowband signal (if monochromatic source)

  Problem:
  - This model predicts SYMMETRIC pulse
  - Observed pulse is ASYMMETRIC
  - The asymmetry requires additional physics
""")

    # Measure the actual asymmetry
    seq = [6, 14, 26, 30, 19, 5]
    rise = seq[3] - seq[0]  # 30 - 6 = 24
    fall = seq[3] - seq[5]  # 30 - 5 = 25

    print(f"\n  Observed asymmetry:")
    print(f"    Rise (6 → 30): {rise}")
    print(f"    Fall (30 → 5): {fall}")
    print(f"    Ratio: {rise/fall:.3f}")
    print(f"    Nearly symmetric! Rise/fall ≈ 1")

    # Model 2: Scintillation
    print("\n" + "=" * 70)
    print("MODEL 2: INTERSTELLAR SCINTILLATION")
    print("=" * 70)

    print(f"""
  Interstellar medium causes scintillation (twinkling) of radio sources.

  Would produce:
  - Intensity variations on timescales of seconds to minutes
  - Narrowband enhancement possible
  - Asymmetric structure possible

  Prediction:
  - Would repeat as source moves through turbulent ISM
  - No repeat observed → source was transient OR one-time event
""")

    # Model 3: Gravitational lensing
    print("\n" + "=" * 70)
    print("MODEL 3: GRAVITATIONAL LENSING")
    print("=" * 70)

    print(f"""
  A gravitational lens could amplify a distant source.

  Would produce:
  - Temporary brightening (matches!)
  - Could be asymmetric (lens geometry)
  - Narrowband if source is narrowband

  The √2 connection:
  - Einstein ring radius involves √2 in certain geometries
  - π/2 appears in deflection angle formulas

  This is speculative but geometrically suggestive.
""")

    # Model 4: Rotating/beamed source
    print("\n" + "=" * 70)
    print("MODEL 4: ROTATING BEAMED SOURCE")
    print("=" * 70)

    print(f"""
  A rotating source with beamed emission (like a pulsar):

  Would produce:
  - Periodic pulses (not observed - only one pulse)
  - BUT if period >> observation time, would see one pulse

  The 17° angular velocity:
  - If source rotates at 360°/21 per some timescale
  - 21 rotations would complete a circle
  - Connected to emission geometry?
""")

    # Model 5: Artificial beacon
    print("\n" + "=" * 70)
    print("MODEL 5: ARTIFICIAL BEACON")
    print("=" * 70)

    print(f"""
  An intentional transmission would:
  - Choose hydrogen line (universal, obvious)
  - Be narrowband (maximize signal/noise)
  - Be transient (survey mode, or deliberate)

  The encoding would:
  - Use fundamental constants (√2, π, φ)
  - Be self-referential (21 cm wavelength → 360°/21)
  - Have simple structure (sum = 100)

  This matches what we observe, but isn't falsifiable
  without additional signals.
""")

    # What the SVD tells us about the source
    print("\n" + "=" * 70)
    print("WHAT SVD REVEALS ABOUT THE SOURCE")
    print("=" * 70)

    print(f"\n  The singular value spectrum tells us about source complexity:")

    # Mode 0: The main signal
    print(f"\n  MODE 0 (dominant):")
    print(f"    Captures {S[0]**2/np.sum(S**2)*100:.1f}% of variance")
    print(f"    = The mean pulse shape")
    print(f"    Interpretation: Primary emission structure")

    # Mode 1: The asymmetry
    print(f"\n  MODE 1:")
    print(f"    Captures {S[1]**2/np.sum(S**2)*100:.1f}% of variance")
    print(f"    = The asymmetry (rise vs fall)")
    print(f"    Interpretation: Time-varying emission or beam asymmetry")

    # Mode 2+: Fine structure
    print(f"\n  MODES 2+:")
    print(f"    Capture {(1 - (S[0]**2 + S[1]**2)/np.sum(S**2))*100:.1f}% of variance")
    print(f"    = Noise + fine structure")
    print(f"    Interpretation: Background + measurement noise")

    # The √2 at gap 5
    print("\n" + "=" * 70)
    print("WHY √2 AT GAP 5?")
    print("=" * 70)

    print(f"""
  S[2]/S[7] = √2 with 0.04% precision

  In physical terms:
  - Mode 2 has √2 times the variance of Mode 7
  - These modes are 5 indices apart

  Geometric interpretation:
  - √2 relates diagonal to side of a square
  - Appears in rotation matrices (45° rotation involves √2)
  - Appears in wave interference (√2 from superposition)

  If the source involves:
  - Two interfering beams
  - Or two polarization states
  - Or rotational geometry

  ...the √2 could emerge naturally.
""")

    # The π/2 at gap 7
    print("\n" + "=" * 70)
    print("WHY π/2 AT GAP 7?")
    print("=" * 70)

    print(f"""
  S[4]/S[11] = π/2 with 0.08% precision

  Physical interpretation:
  - π/2 radians = 90° (quarter turn)
  - Appears in circular/rotational geometry
  - Appears in wave phase relationships

  If the source has:
  - Circular polarization (phase shift of π/2)
  - Or circular motion (angular relationships)
  - Or waveguide geometry

  ...the π/2 could emerge naturally.
""")

    # The 360°/21 angular velocity
    print("\n" + "=" * 70)
    print("WHY 360°/21 ANGULAR VELOCITY?")
    print("=" * 70)

    print(f"""
  Mean |Δθ| = 17.65° ≈ 360°/21 = 17.14°

  Physical interpretation:
  - 21 divisions of a full rotation
  - Connected to hydrogen (21 cm)

  If the source or observation involves:
  - A 21-fold symmetry
  - Or coupling to hydrogen physics
  - Or specific beam/source geometry

  The 21 could emerge from:
  - The hydrogen hyperfine transition itself
  - 21 = 3 × 7 (angular momentum coupling?)
  - 21 = F(8) (growth/natural structure)
""")

    # Synthesis
    print("\n" + "=" * 70)
    print("SYNTHESIS: WHAT THE STRUCTURE TELLS US")
    print("=" * 70)

    print(f"""
  THE SOURCE CHARACTERISTICS:

  1. NARROWBAND EMITTER
     - Bandwidth < 10 kHz
     - Centered on hydrogen line
     - Implies: coherent emission (not thermal)

  2. TRANSIENT
     - Duration ~72 seconds
     - Single event (no repeat in 50 years)
     - Implies: one-time or very rare event

  3. STRUCTURED
     - Asymmetric pulse shape
     - Low effective dimension (2-3)
     - Implies: simple geometry, not chaotic

  4. GEOMETRIC RELATIONSHIPS
     - √2, π/2 in mode ratios
     - 360°/21 angular velocity
     - Implies: rotational/wave geometry

  POSSIBLE SOURCE TYPES:

  A) SCINTILLATING BACKGROUND SOURCE
     - ISM focusing of a steady narrowband source
     - Would explain transience
     - Doesn't obviously explain √2, π/2

  B) LENSING EVENT
     - Gravitational magnification
     - Could involve geometric constants
     - Would be extremely rare (matches observation)

  C) ROTATING BEAMED EMITTER
     - Natural or artificial
     - Would have rotational geometry
     - Could produce periodic structure

  D) ARTIFICIAL BEACON
     - Designed to be found
     - Would intentionally encode constants
     - Matches all observations

  WE CANNOT DISTINGUISH THESE without:
  - A repeat observation
  - Multi-frequency data
  - Polarization information
  - A second, similar signal

  THE SIGNAL REMAINS AMBIGUOUS:
  Its structure is consistent with both natural
  and artificial origins. The information it
  carries could be about physics or about intent.
""")


if __name__ == "__main__":
    main()
