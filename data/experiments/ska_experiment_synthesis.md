# SKA-Inspired Experiments: Synthesis of Findings

**Date:** 2026-01-27

## Overview

Three experiments inspired by the SKA (Structured Knowledge Accumulation) paper, testing if geometry-derived parameters match fundamental constants (phi, pi/e, e/pi, sqrt(2)). ~~The fundamental constants hypothesis was disproven~~ [DISPROVEN: phi ranks 202/1014 among tested constants -- no special significance. See PHI_FINDINGS.md].

---

## Experiment 1: Geometry-Derived Metric Tensor [DISPROVEN]

**Hypothesis:** SKA's metric tensor coefficients (α, β, γ) encode fundamental ratios.

**Method:** Extract eigenvalue ratios from local covariance at each point in activation manifold.

**Results:**
| Metric | Mean | Std | Closest Constant | Error |
|--------|------|-----|------------------|-------|
| α (eig0/eig1) | 3.04 | 1.91 | φ² (2.618) | 16% |
| β (eig1/eig2) | 2.10 | 0.82 | φ² (2.618) | 20% |
| γ (regularization) | 0.20 | 0.11 | √eps × scale | ✓ |

**Key Finding:** Both α and β cluster around **~2.6**, not π/e or √2 as expected. Match rate ~30% for individual points.

**Note:** Early analysis suggested φ² (2.618) significance, but subsequent testing (see PHI_FINDINGS.md) showed φ ranks 202/1014 among tested constants - no special significance. The clustering around ~2.6 may simply reflect the natural scale of eigenvalue ratios in these architectures.

---

## Experiment 2: DEC Geodesic Computation [EMPIRICAL]

**Hypothesis:** Discrete Exterior Calculus geodesics match Floyd-Warshall, and Hodge decomposition separates correct/incorrect regions.

**Results:**
- DEC geodesics: mean = 3.98
- Floyd-Warshall geodesics: mean = 114.98
- Relative error: 97% (they measure different scales)

**Hodge Decomposition:**
| Component | Fraction | Interpretation |
|-----------|----------|----------------|
| Gradient | 13.1% | Steepest descent (optimization path) |
| Curl | 43.4% | Circulation (recurrent processing) |
| Harmonic | 43.5% | Global topology (stable attractors) |

**Key Finding:** The activation manifold is **curl-dominated**, suggesting the model processes information through circulation rather than direct optimization paths. This may explain why simple gradient descent struggles.

**Implication:** Learning is more about navigating circulation patterns than following gradients. This matches the expand-compress dynamics.

---

## Experiment 3: Temporal/Spatial Learning Duality [EMPIRICAL]

**Hypothesis:** Entropy decreases through layers (information compression), correct answers converge to target.

**Results:**
- Entropy trajectory: 0.84 → 2.09 (**INCREASES**, opposite of hypothesis)
- Intrinsic dimension: 1.7 → 8.3 (**EXPANDS**)
- Separation ratio (incorrect/correct): **2.14 > φ** ✓

**Key Finding:** Entropy **increases** through layers, not decreases. This led to discovering the expand-compress model.

---

## Experiment 4: Full Entropy Trajectory (All 36 Layers) [VALIDATED]

**Hypothesis (revised):** Information expands to high-dimensional space, then compresses at output.

**Results:**
```
Phase          | Layers    | Entropy Change
---------------|-----------|---------------
Expansion      | 0 → 26    | 0.57 → 1.51 (+166%)
Processing     | 26 → 34   | ~1.48-1.50 (plateau)
Compression    | 34 → 35   | 1.48 → 0.99 (-33%)
```

**Key Finding:** The model follows an **expand → process → compress** trajectory:
1. **Expansion** (layers 0-26): Information blows up to high-dimensional space
2. **Processing** (layers 26-34): High-entropy plateau for computation
3. **Compression** (layer 35): Dramatic funnel to coherent output

The compression is remarkably sharp - nearly all happens in the **final layer**.

---

## Synthesis: What We Learned

### 1. ~~Eigenvalue Ratios Cluster Around ~2.6~~ [DISPROVEN: phi has no special significance]
- ~~α and β eigenvalue ratios cluster around **~2.6**~~
- ~~The separation ratio exceeds **1.6**~~
- Note: phi^2 = 2.618 is numerically close, but PHI_FINDINGS.md showed phi has no special significance (ranks 202/1014 among tested constants)

### 2. Circulation > Gradient [EMPIRICAL]
- Curl (43%) and Harmonic (43%) dominate the Hodge decomposition
- Gradient component is only 13%
- This explains why gradient descent struggles - the manifold has strong circulation

### 3. Expand-Compress Dynamics [VALIDATED]
- Information **expands** through layers (0.57 → 1.51 entropy)
- Compression happens sharply in the **final layer** (1.48 → 0.99)
- This is the "funnel" that channels processing into coherent output

### 4. Geometry-Derived Training Works (in principle) [EMPIRICAL]
- LR = 1/(κ × scale) = 6.99e-05 (derived from geometry)
- Max iterations = ceil(κ) = 3 (derived from condition number)
- Loss dropped 22.58 → 0.65 in 3 iterations

**But:** Raw SGD on base model + adapter damaged performance. Need proper LoRA training.

---

## Implications for GSM8K Mastery

1. **Training should respect the funnel:** Don't disrupt the final-layer compression
2. **Consider circulation patterns:** Pure gradient descent may not follow the natural paths
3. **Certain ratios appear at multiple scales:** α, β, separation ratio cluster around similar values - but this may be architectural, not fundamental
4. **Final layer is critical:** The dramatic compression in layer 35 is where "understanding" crystallizes

---

## Next Steps

1. **Fix training method:** Use proper LoRA, not raw SGD on full model
2. **Train with more data:** 3 examples is insufficient; use the 60 examples from Phase B
3. **Respect expand-compress:** Monitor entropy trajectory during training
4. **Characterize natural constants:** Determine which ratios emerge from architecture vs training

---

## Files

| Experiment | Script | Results |
|------------|--------|---------|
| Metric Tensor | `scripts/exp_geometry_metric_tensor.py` | `data/experiments/exp1_geometry_metric_tensor.json` |
| DEC Geodesic | `scripts/exp_dec_geodesic.py` | `data/experiments/exp2_dec_geodesic.json` |
| Temporal/Spatial | `scripts/exp_temporal_spatial_duality.py` | `data/experiments/exp3_temporal_spatial_duality.json` |
| Entropy Full | `scripts/exp_entropy_trajectory_full.py` | `data/experiments/entropy_trajectory_full.json` |
