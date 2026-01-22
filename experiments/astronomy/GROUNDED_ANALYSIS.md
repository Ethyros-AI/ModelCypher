# Wow! Signal: Grounded Geometric Analysis

**Date**: 2026-01-21
**Methodology**: SVD analysis with proper null hypothesis testing against FRB control group

---

## Data

- **Source**: `wow_signal.sav` (IDL format)
- **Content**: SNR matrix, shape [82 time × 50 frequency]
- **Peak**: 30.5 at time=60, freq=1
- **Background**: median ≈ 0.5

---

## Verified Claims

### 1. SVD Ratios

| Ratio | Value | Claimed Match | Error |
|-------|-------|---------------|-------|
| S[0]/S[1] | 1.5630 | φ (1.618) | 3.4% |
| S[1]/S[2] | 3.2942 | π (3.142) | 4.9% |

### 2. Statistical Significance (vs 45 FRBs)

| Ratio | FRB Mean | FRB Std | Wow! | z-score | Unusual? |
|-------|----------|---------|------|---------|----------|
| S[0]/S[1] | 1.15 | 0.33 | 1.56 | **1.25** | NO |
| S[1]/S[2] | 1.12 | 0.38 | 3.29 | **3.86** | YES |

### 3. Mode Structure

- Mode 0+1 explain **80%** of variance
- Mode 0+1+2 explain **82%** of variance
- Gap between mode 1 and mode 2 is unusually large

---

## Debunked Claims

### ❌ "S[0]/S[1] ≈ φ is special"

**False.** FRBs have similar ratios (z=1.25, well within normal range). The value 1.56 is not unusual for radio signals.

### ❌ "Random matrices don't produce these ratios"

**Misleading.** Random Gaussian matrices are the wrong comparison. When compared to actual radio signals (FRBs), the S[0]/S[1] ratio is normal.

### ❌ "11-dimensional winding"

**Artifact.** Real-valued SVD produces Zak phase quantization. The "11" was counting sign flips, not measuring continuous phase.

---

## Genuine Anomalies

### ✓ S[1]/S[2] = 3.29 is unusual (z=3.86)

Only 1/45 FRBs has S[1]/S[2] > 3.29. This means the Wow! signal has an unusually **clean 2-mode structure**.

**Physical interpretation**: The signal concentrates its energy in exactly two orthogonal patterns with a large gap before the third pattern. Most FRBs have more distributed spectral structure.

### ✓ Is proximity to π meaningful?

**Probably not.** Testing 10,000 random rank-2 dominated matrices, 0% produced S[1]/S[2] within 5% of π. This suggests:
- The value ~3.3 may be constrained by having exactly 2 dominant modes
- Proximity to π (4.9% error) is likely coincidental
- No physical mechanism has been proposed for π appearing here

---

## What the Signal Actually Is

Based on grounded analysis:

1. **A narrowband, 2-mode signal** - Two orthogonal spectral/temporal patterns dominate
2. **Cleaner than typical FRBs** - Unusually large gap between mode 2 and mode 3
3. **Not "encoded" with mathematical constants** - φ proximity is unremarkable; π proximity is coincidental

---

## Open Questions (Answerable)

1. **What physical mechanism produces exactly 2 dominant modes?**
   - Narrowband maser?
   - Coherent emission process?
   - Instrumental artifact?

2. **Does the 2-mode structure appear in other SETI candidates?**
   - Compare to other "interesting" signals in archives

3. **What is FRB20180906A?**
   - This FRB has S[1]/S[2] = 3.56, higher than Wow!
   - Is it also a "2-mode" signal?

---

## Files

**Verified Analysis:**
- `audit_wow_data.py` - Basic data audit
- `audit_wow_structure.py` - Synthetic and FRB comparison
- `audit_wow_r2_anomaly.py` - Detailed S[1]/S[2] analysis

**Supporting Analysis (FRB geometry):**
- `exp16_geodesic_structure.py` - Geodesic structure of FRB feature space
- `exp17_null_space.py` - Null space analysis
- `exp74_neural_spectrum_structure.py` - Applies same methods to neural network activations

**Data:**
- `data/` - Wow! signal and FRB data
- `shared/` - Data loading utilities

---

## Cleanup Note (2026-01-21)

Removed ~70 experiments that made speculative claims without proper statistical backing:
- "Decoding" experiments that assumed intentional encoding
- "Semantic highway" experiments that compared signal to LLM embeddings
- Numerology experiments looking for φ, π, e without FRB controls
- "11-dimensional winding" (debunked as phase quantization artifact)
- Claims of z-scores like -144σ (wrong comparisons to random noise, not FRBs)

What remains is geometry that can be measured, with proper null hypothesis testing.

---

*"The signal is geometrically clean. That's the finding. Everything else is interpretation."*
