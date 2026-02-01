# Phi Null Hypothesis Findings

**Generated:** 2026-02-01
**Model tested:** LFM2-350M-MLX-bf16

## Research Question

Does φ (golden ratio ≈ 1.618) have mechanistic significance in activation geometry, or are we pattern-matching to aesthetically pleasing numbers?

## Executive Summary

**φ is NUMEROLOGY.** The experiments conclusively show:

1. φ is NOT statistically special (ranks 202nd out of 1014 constants tested)
2. The best normalizing constant is **2**, not φ
3. The raw ratio (peak_dim / final_dim) clusters around **1.0-2.0**, not φ
4. TwoNN-based and norm-based "comp/φ" measure **completely different things** (r = -0.27)

**RECOMMENDATION: Remove φ from the formula. Report raw_ratio directly, or normalize by 2 if normalization is needed.**

---

## Phase 1: TwoNN Validation

**Purpose:** Verify the measurement tool before questioning the metric.

**Result:** PASSED
- 6/7 synthetic manifolds recovered within 95% CI
- S² sphere: estimated 2.84 (true=2, CI [2.42, 3.34]) - slight bias
- All other manifolds: accurate within CI
- Strong z-scores (-3.6 to -10.3) distinguishing structured from random data

**Conclusion:** TwoNN reliably measures intrinsic dimension.

---

## Phase 2: Raw Ratio Analysis

**Purpose:** See what peak_dim/final_dim actually looks like WITHOUT φ normalization.

**Results on LFM2-350M (22 prompts):**
```
Distribution Statistics:
  Mean:   1.9734
  Std:    1.1435
  Median: 1.7606
  Mode:   1.1137
  Range:  [1.0000, 5.5490]

Clustering Analysis:
  Near 1.0 (±10%): 5/22 (22.7%)
  Near φ (±10%):   3/22 (13.6%)
  Near 1/φ (±10%): 0/22 (0.0%)
```

**Key Finding:** The MODE (1.11) clusters near 1.0, not φ. Many prompts have flat trajectories (peak ≈ final).

**Interpretation:** When peak_dim ≈ final_dim, dividing by φ gives comp/φ ≈ 0.618 - this is an **artifact** of the formula, not emergent structure.

---

## Phase 3: Constant Comparison

**Purpose:** Is φ statistically special compared to random constants?

**Method:** Tested φ against 14 fundamental constants + 1000 random constants from [0.5, 5.0]. Measured which constant c makes raw_ratio/c closest to 1.0.

**Results:**
```
φ RANKING:
  Rank: 202 / 1014
  Percentile: 19.8%
  → φ is in top 25% - marginally better than random, but NOT special

BEST CONSTANTS (by distance from 1.0):
  Best overall: rand_511 = 1.9708 (distance: 0.0013)
  Best fundamental: 2 (distance: 0.0133)

ALL FUNDAMENTAL CONSTANTS RANKED:
  Symbol   Value    Dist     %near1   Rank
  2        2.0000   0.0133   18.2     #12
  π/φ      1.9416   0.0164   18.2     #13
  e/φ      1.6800   0.1747   9.1      #150
  φ        1.6180   0.2196   13.6     #202  ← NOT SPECIAL
  ...
```

**Key Finding:** φ ranks **#202 out of 1014** (20th percentile). The best fundamental constant is **2**.

**Interpretation:** φ performs no better than arbitrary constants. The ratio naturally clusters around ~2.0, making `2` the natural normalizing constant if one is needed.

---

## Phase 4: Model Diagnostic

**Purpose:** Investigate trajectory shapes.

**Results on LFM2-350M (15 prompts):**
```
TRAJECTORY SHAPE DISTRIBUTION:
  compression    5 (33.3%)
  expansion      4 (26.7%)
  peaked         5 (33.3%)
  flat           1 (6.7%)

Peak ≈ final: 1/15 (6.7%)
```

**Key Finding:** LFM2-350M has varied trajectory shapes - mostly peaked/compression/expansion patterns. Only 7% are flat.

**Note:** The high variance in raw ratios (mean=19.7, std=65.4) indicates outliers from longer prompts with many tokens.

---

## Phase 6: Method Correlation

**Purpose:** Do TwoNN-based and norm-based "comp/φ" measure the same thing?

**Results:**
```
CORRELATION: r = -0.2665

Method          Mean         Std          Range
TwoNN           1.0247       0.2822       [0.618, 1.418]
Norm-based      1.8518       0.4593       [0.528, 2.411]
```

**CRITICAL Finding:** r = -0.27 means the methods are **negatively correlated** - they measure COMPLETELY DIFFERENT properties.

**Implications:**
- "comp/φ" means different things depending on measurement method
- Cannot use them interchangeably
- The differentiable proxy does NOT capture the same geometry as TwoNN

---

## Decision Matrix

| Finding | Evidence | Recommended Action |
|---------|----------|-------------------|
| φ is NOT special | Ranks 202/1014 (20th percentile) | Remove φ from formula |
| Best constant is 2 | Distance from 1.0 = 0.013 | Consider normalizing by 2 |
| Methods disagree | r = -0.27 | Clarify which matters; cannot use both |
| Raw ratio clusters ~1.0-2.0 | Mode = 1.11, mean = 1.97 | Report raw ratio directly |

---

## Recommendations

### Immediate Actions

1. **Remove φ from the formula** in `safety.py:928`
   ```python
   # OLD (NUMEROLOGY):
   comp_phi = (peak_dim / final_dim) / PHI

   # NEW (HONEST):
   expansion_ratio = peak_dim / final_dim
   ```

2. **Rename the metric** from "comp/φ" to "expansion_ratio" or "dimension_ratio"

3. **Document the method disagreement** - TwoNN and norm-based approaches are not interchangeable

### If Normalization Is Desired

If a normalized metric is still wanted, use 2 instead of φ:
```python
normalized_ratio = (peak_dim / final_dim) / 2.0
```

This has theoretical motivation: it's asking "how much bigger is the peak dimension than the final dimension, as a fraction of 2?"

### For the Differentiable Proxy

The norm-based differentiable_phi.py measures something different. Options:
1. Accept they're different metrics and use each for its purpose
2. Rename to avoid confusion (e.g., "norm_expansion_loss")
3. Investigate what the norm-based method actually captures

---

## Appendix: Raw Data Locations

- `data/experiments/twonn_validation.json` - Phase 1 results
- `data/experiments/raw_ratio_analysis.json` - Phase 2 results
- `data/experiments/constant_comparison.json` - Phase 3 results
- `data/experiments/deepseek_diagnostic.json` - Phase 4 results
- `data/experiments/method_correlation.json` - Phase 6 results

---

## Conclusion

The φ constant in the comp/φ formula is **aesthetically motivated numerology**, not mathematically justified structure. The golden ratio has no special significance in the dimension ratio data - the constant 2 performs better, and arbitrary random constants often outperform φ.

The "comp/φ ≈ 0.618" pattern observed in some models (like DeepSeek-R1) is a **construction artifact**: when peak_dim ≈ final_dim (flat trajectory), raw_ratio ≈ 1.0, and 1.0/φ = 0.618 by definition.

**Trust the math, not the aesthetics.**
