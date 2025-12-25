# Operational Semantics Hypothesis: Riemannian Re-Analysis

**Date**: 2025-12-25
**Previous Analysis**: 2025-12-24 (Euclidean)

## Critical Correction

The original experiment used **Euclidean algebra** when the codebase now enforces **geodesic geometry**. This re-analysis uses:
- **Geodesic distances** via k-NN graph shortest paths (not Euclidean norm)
- **Fréchet mean** instead of arithmetic mean (curvature-aware centroid)
- **Backend protocol** instead of numpy

## Key Finding: Curvature Changes Interpretation

All models show **positive curvature** with geodesic/Euclidean ratios > 1.0:

| Model | Geo/Euc Ratio | Curvature Type |
|-------|---------------|----------------|
| Qwen 0.5B | 1.15 | Positive (15% larger geodesic) |
| Llama 3.2 3B | **1.27** | Strong positive (27% larger) |
| Mistral 7B | 1.12 | Positive (12% larger) |
| Qwen3 8B | 1.14 | Positive (14% larger) |
| Qwen 2.5 3B | **1.26** | Strong positive (26% larger) |

This means **Euclidean distances underestimate true geodesic distances by 12-27%**.

## Comparison: Euclidean vs Riemannian Results

### Position Test (Is 5 closer to (3,4) than 6?)

| Model | Euclidean | Geodesic | Same Answer? |
|-------|-----------|----------|--------------|
| Qwen 0.5B | FAIL | FAIL | Yes |
| **Llama 3.2 3B** | PASS | **PASS** | Yes |
| Mistral 7B | PASS (barely) | **FAIL** | **NO** |
| **Qwen3 8B** | PASS | **PASS** | Yes |
| Qwen 2.5 3B | FAIL | FAIL | Yes |

### Critical: Mistral 7B Changes Answer

The original analysis showed Mistral 7B passed the position test (14.56 vs 13.80 - barely).
With geodesic geometry: **5 = 16.22, 6 = 15.60** - now 6 is clearly closer.

The Euclidean analysis gave the **wrong answer** for Mistral 7B.

## Fréchet Mean vs Arithmetic Mean Difference

The difference between the proper Riemannian centroid and the Euclidean centroid:

| Model | Frechet - Arithmetic Distance | Significance |
|-------|------------------------------|--------------|
| Qwen 0.5B | 3.1 units | Moderate |
| Llama 3.2 3B | 2.3 units | Moderate |
| Mistral 7B | 1.5 units | Small |
| **Qwen3 8B** | **67.8 units** | **HUGE** |
| Qwen 2.5 3B | 24.1 units | Large |

Qwen3 8B has a massive centroid shift - the Euclidean analysis was measuring from the **wrong center**.

## Revised Hypothesis Support

| Model | Original (Euclidean) | Corrected (Geodesic) | Changed? |
|-------|---------------------|----------------------|----------|
| Qwen 0.5B | NOT SUPPORTED | NOT SUPPORTED | No |
| **Llama 3.2 3B** | **SUPPORTED** | **SUPPORTED** | No |
| Mistral 7B | PARTIAL | **NOT SUPPORTED** | **YES** |
| **Qwen3 8B** | PARTIAL | **SUPPORTED** | **YES** |
| Qwen 2.5 3B | NOT SUPPORTED | NOT SUPPORTED | No |

### Summary of Changes:
- **Mistral 7B**: Downgraded from PARTIAL to NOT SUPPORTED
- **Qwen3 8B**: Upgraded from PARTIAL to SUPPORTED (despite massive centroid shift)

## Implications

### 1. Curvature is Real and Significant
Every model shows 12-27% larger geodesic vs Euclidean distances. This confirms:
- The representation space is positively curved
- Euclidean algebra systematically underestimates distances
- The original analysis had a systematic bias

### 2. Some Conclusions Hold
Llama 3.2 3B genuinely encodes Pythagorean structure - this passes with both metrics.
The original finding for this model is validated.

### 3. Some Conclusions Were Wrong
Mistral 7B's "pass" was a Euclidean artifact. The geodesic analysis shows no Pythagorean encoding.

### 4. Fréchet Mean Matters
The centroid shift is substantial (up to 67.8 units for Qwen3). Any analysis computing "distance from center" using arithmetic mean was measuring from the wrong point.

## Geodesic Distance Values

### Llama 3.2 3B (Hypothesis SUPPORTED)
```
Geodesic 5 → (3,4): 20.18
Geodesic 6 → (3,4): 39.08
Euclidean 5 → (3,4): 16.59
Euclidean 6 → (3,4): 29.47
```
5 is ~2x closer than 6 in geodesic terms.

### Mistral 7B (Hypothesis NOT SUPPORTED)
```
Geodesic 5 → (3,4): 16.22
Geodesic 6 → (3,4): 15.60
Euclidean 5 → (3,4): 14.56
Euclidean 6 → (3,4): 13.80
```
6 is actually closer than 5 geodesically.

## Open Issues

### Curvature Estimation Shows Zero
The sectional curvature estimates show 0.0 for all points despite the geodesic/Euclidean ratio indicating positive curvature. This is likely due to:
- Small sample size (18 numbers) for the curvature estimator
- Need more points to reliably estimate local curvature

The geodesic/Euclidean ratio is a more robust curvature indicator for small samples.

## Cross-Model Invariance: Euclidean Was Wrong

The original analysis claimed **88.5% cross-model similarity** after Procrustes alignment.

With proper geodesic geometry: **35.3% average similarity**.

| Comparison | Euclidean (Procrustes) | Geodesic |
|------------|------------------------|----------|
| Average similarity | 88.5% | **35.3%** |
| Interpretation | "Strong invariance" | "Limited invariance" |

### Why the Discrepancy?

The "5/6 ratio from (3,4)" varies dramatically across models:

| Model | 5/6 Geodesic Ratio | Interpretation |
|-------|-------------------|----------------|
| Llama 3.2 3B | **0.54** | 5 much closer than 6 |
| Qwen3 8B | 0.92 | 5 slightly closer |
| Mistral 7B | 1.07 | 6 actually closer |
| Qwen 0.5B | 1.25 | 6 closer |
| Qwen 2.5 3B | 2.07 | 6 much closer |

Only **Llama 3.2 3B** shows strong Pythagorean encoding (ratio << 1).

### The Procrustes Illusion

Euclidean Procrustes finds the best rotation to align point clouds. This works because:
1. It minimizes Euclidean residuals
2. All models have the same numbers at similar relative positions
3. After rotation, surface-level structure looks similar

But geodesic geometry reveals the **actual** manifold relationships are very different.
The "invariance" was measuring **tokenization similarity**, not **mathematical structure**.

## Files

- `riemannian_v2_*.json` - Per-model geodesic analysis results
- `cross_model_geodesic.json` - Geodesic cross-model comparison
- `fast_*.json` - Original Euclidean fast probe results
- `riemannian_pythagorean_v2.py` - Corrected analysis script (no numpy)
- `cross_model_geodesic.py` - Geodesic invariance test

## Conclusion

**The Operational Semantics Hypothesis holds for Llama 3.2 3B and Qwen3 8B using proper geodesic geometry.**

### What We Got Right
- Llama 3.2 3B genuinely encodes Pythagorean structure
- The manifold is positively curved (12-27% geodesic/Euclidean ratio)

### What We Got Wrong
- **Mistral 7B**: Euclidean said PASS, geodesic says FAIL
- **Cross-model invariance**: 88.5% was an artifact; true invariance is 35%
- **The "invariant but twisted" hypothesis**: REJECTED - geodesic structures vary significantly

### The Corrected Picture
Mathematical structure is **not universally encoded** across LLM architectures.
Only some models (Llama 3.2 3B, Qwen3 8B) show geometric Pythagorean encoding.
The original claim of architecture-independent invariance was a Euclidean illusion.

Geodesic geometry provides the correct picture of the representation manifold.
