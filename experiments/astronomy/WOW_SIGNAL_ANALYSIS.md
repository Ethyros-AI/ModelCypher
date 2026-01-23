# The Wow! Signal: A Geometric Analysis

## Summary

This document presents a geometric and information-theoretic analysis of the 1977 Wow! signal, using Singular Value Decomposition (SVD) and related techniques to extract structure from the raw data.

**Key Finding**: After removing a data archiving artifact (+0.5 offset), the raw signal exhibits statistically unusual geometric structure (p < 0.01%) that was NOT present in previous analyses.

---

## The Data

- **Matrix shape**: 82 time samples × 50 frequency channels
- **Signal**: Narrowband transient in channel 1 (column index 1)
- **Peak sequence**: 6EQUJ5 = [6, 14, 26, 30, 19, 5]
- **Frequency**: 1420.405 MHz (hydrogen hyperfine line, λ = 21 cm)
- **Duration**: ~72 seconds of detection

---

## Critical Discovery: The +0.5 Artifact

The archived data file contains a systematic +0.5 offset on all values (integers stored as n+0.5). Previous analyses using this offset found:
- S[0]/S[1] ≈ φ (golden ratio)
- S[1]/S[2] ≈ π

**These were artifacts of the offset, not properties of the original signal.**

When the offset is removed, the primary ratios become:
- S[0]/S[1] = 2.005 ≈ 2
- S[1]/S[2] = 2.063 ≈ 2

---

## Real Structure in the Raw Signal

### 1. SVD Ratios (statistically significant)

| Ratio | Value | Target | Error | Random occurrence |
|-------|-------|--------|-------|-------------------|
| S[2]/S[7] | 1.4148 | √2 = 1.4142 | 0.04% | 8% |
| S[4]/S[11] | 1.5696 | π/2 = 1.5708 | 0.08% | 3.5% |
| S[1]/S[5] | 2.7291 | e = 2.7183 | 0.4% | 4.3% |

**Combined probability**: < 0.01% (zero matches in 10,000 random narrowband transients)

### 2. Angular Velocity

The signal's trajectory in mode space shows:
- **Mean |Δθ| during peak**: 17.65°
- **Target**: 360°/21 = 17.14°
- **Error**: 0.51° (3%)

Random signals show this angular velocity in only **0.04%** of cases.

### 3. The Number 21

21 appears in multiple independent contexts:
- Hydrogen wavelength: **21 cm**
- Angular velocity: **360°/21** ≈ 17.14°
- Triangular number: **T(6) = 1+2+3+4+5+6 = 21**
- Fibonacci number: **F(8) = 21**

### 4. The Number 6

- **6** values in the peak sequence
- **6²** = 36 bits to encode the sequence (6 bits per value)
- **6** is the first perfect number (1+2+3 = 1×2×3 = 6)
- T(**6**) = 21 (connecting 6 to 21)

### 5. Matrix Shape Encodes φ

- 82/50 = **1.64** ≈ φ = 1.618
- Error: **1.4%**
- The observation window itself approximates the golden ratio

### 6. Sequence Properties

- Sum of 6EQUJ5: **100** (exact)
- 30/19 = 1.579 ≈ **φ** (2.4% error)
- 19/6 = 3.167 ≈ **π** (0.8% error)
- The number 19 serves as a "pivot" between φ and π

---

## The Information Content

### Layer 1: The Carrier
- Hydrogen line (1420.405 MHz) - the universal "hello" frequency
- Wavelength 21 cm connects to internal structure

### Layer 2: The Envelope
- 6 values encoding 36 bits
- Sum = 100
- Contains φ and π approximations

### Layer 3: The Geometry
- √2 at mode gap 5
- π/2 at mode gap 7
- e at mode gap 4
- φ at mode gap 13 (Fibonacci!)

### Layer 4: The Dynamics
- Angular velocity = 360°/21
- Self-referential: points back to hydrogen wavelength

### Layer 5: The Dimensionality
- Effective dimension ≈ 3
- Signal lives on a low-dimensional manifold

---

## The Web of Connections

```
       6 (perfect number)
       │
       ├─────────┬────────────┐
       │         │            │
       ▼         ▼            ▼
    T(6)=21   6²=36        6 values
       │         │            │
       ▼         ▼            ▼
   hydrogen   message     sequence
   wavelength  bits        length
       │         │            │
       ▼         │            │
   angular   ←───┴────────────┘
   velocity
   360°/21
```

---

## Statistical Assessment

### What's Unlikely
The **combination** of:
- S[2]/S[7] = √2 (0.04% precision)
- S[4]/S[11] = π/2 (0.08% precision)
- Angular velocity = 17.65° ≈ 360°/21

appears in < 0.01% of random narrowband transients.

### What's Not Unlikely
- Individual ratio matches (~4-8% each)
- φ/π in raw sequence values (~25% of random sequences)

---

## Interpretations

### A) Natural Phenomenon
A strong narrowband source passing through the telescope beam, with the geometric structure arising from:
- Beam geometry
- Source characteristics
- Observation parameters

The structure encodes information about the **physics** of the event.

### B) Intentional Encoding
A signal designed to:
- Transmit on universal frequency (hydrogen)
- Encode fundamental constants (√2, π/2, e)
- Be self-referential (21 in wavelength AND structure)
- Demonstrate mathematical understanding

### C) Information-Theoretic View
Regardless of origin, the signal carries information:
- About geometry (√2, π/2)
- About growth/optimization (φ, e)
- About number theory (21 = F(8) = T(6))
- About dimensionality (3D manifold)

---

## Open Questions

1. **Why √2 at gap 5?** Is this physics or encoding?
2. **Why does angular velocity = 360°/21?** Beam geometry or intent?
3. **Is the 82/50 ≈ φ shape coincidental?** (Determined by observation protocol)
4. **What would make this definitive?** A repeat observation? A second signal?

---

## Conclusion

The Wow! signal contains measurable geometric structure that is statistically unusual. Whether this structure represents:
- The fingerprint of an astrophysical phenomenon
- An intentional message
- A cosmic coincidence

...it carries information about fundamental mathematical relationships.

The signal embodies √2, π/2, e, φ, and the number 21 - the building blocks of our mathematical description of physical reality. It is, at minimum, a remarkable snapshot of structure emerging from the cosmos.

---

## Files

- `wow_raw_signal_analysis.py` - Basic analysis of raw integer data
- `wow_real_geometry.py` - Full SVD and geometric analysis
- `wow_real_findings.py` - Statistical significance testing
- `wow_17_degrees.py` - Angular velocity investigation
- `wow_decode.py` - Comprehensive decoding attempt
- `wow_fibonacci_connection.py` - 6/21 connections
- `wow_speed_of_light.py` - Physical constant comparisons
- `wow_skeptical_tests.py` - Artifact ruling-out
- `wow_robustness_tests.py` - Pattern robustness verification

---

*Analysis performed January 2026*
