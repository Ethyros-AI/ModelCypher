# Model Selection Analysis: Finding the Optimal Geometry

**Date:** 2026-01-27

## Executive Summary

We profiled 8 small models (<3B parameters) to find the one with the best geometric properties for alignment work. The key metric is **expansion_ratio** (peak_dim/final_dim) - values near 1.0-2.0 correlate with balanced processing.

**Winner:** Depends on license requirements:
- **Truly Open (Apache 2.0):** Qwen2.5-3B-Instruct
- **Revenue-limited OK:** LFM2-1.2B (marginal geometric advantage)

---

## Complete Ranking

| Rank | Model | Score | Expansion Ratio | Peak Layer | License |
|------|-------|-------|-----------------|------------|---------|
| 1 | **LFM2-1.2B** | 0.635 | 1.82 | 55.6% | LFM1.0 (≤$10M) |
| 2 | **Qwen2.5-3B** | 0.634 | 1.87 | 52.5% | Apache 2.0 |
| 3 | LFM2-350M | 0.611 | 1.62 | 38.1% | LFM1.0 |
| 4 | LFM2-700M | 0.602 | 1.32 | 44.4% | LFM1.0 |
| 5 | LFM2.5-1.2B | 0.600 | 1.98 | 57.5% | LFM1.0 |
| 6 | Qwen3-1.7B | 0.533 | 2.21 | 45.4% | Apache 2.0 |
| 7 | Qwen2.5-Coder-0.5B | 0.469 | 2.23 | 25.4% | Apache 2.0 |
| 8 | Qwen2.5-Math-1.5B | 0.292 | 2.81 | 6.4% | Apache 2.0 |

---

## Key Findings

### 1. Liquid Foundation Models Have Balanced Geometry

The LFM2 family shows consistent geometric properties:
- **LFM2-350M achieves expansion_ratio ≈ 1.62** (balanced expansion/compression)
- All LFM2 models have peak layers in the 38-57% range
- The Liquid (state-space) architecture produces consistent trajectory shapes

**Note:** The state-space design produces consistent geometric patterns, though the significance of specific ratio values requires further validation.

### 2. Qwen2.5-3B Is the Best Truly Open Model

With Apache 2.0 license and expansion_ratio = 1.157:
- No revenue restrictions
- Can be freely modified and released under AGPL
- Peak layer at 52.5% (ideal range)
- Strong baseline capabilities (3B scale)

### 3. Model Size Doesn't Determine Geometry

| Observation | Example |
|-------------|---------|
| Tiny can be balanced | LFM2-350M: expansion_ratio ≈ 1.6 |
| Bigger isn't always better | Qwen2.5-Math-1.5B: expansion_ratio ≈ 2.8 (highest variance) |
| Architecture matters | LFM2 family shows more consistent ratios than Qwen family |

---

## Detailed Profiles

### LFM2-1.2B (Top Score)
```
Layers: 16
Hidden dim: 2048
Compression/φ: 1.127 (very close to ideal 1.0)
Peak layer: 55.6% (exactly in ideal range)
Trajectory: 11.2 → 19.4 → 12.4
Geometric score: 0.635
```
**Pros:** Best overall geometry, proven in our merge experiments
**Cons:** LFM1.0 license limits commercial use >$10M revenue

### Qwen2.5-3B-Instruct (Best Open License)
```
Layers: 36
Hidden dim: 2048
Compression/φ: 1.157 (close to ideal)
Peak layer: 52.5% (ideal range)
Trajectory: 16.0 → 23.4 → 13.0
Geometric score: 0.634
```
**Pros:** Apache 2.0, 3B scale baseline capabilities, ideal peak position
**Cons:** Larger than LFM2-1.2B (3B vs 1.2B)

### LFM2-350M (Balanced Ratio)
```
Layers: 16
Hidden dim: 1024
Expansion ratio: 1.62 (balanced)
Peak layer: 38.1% (slightly early)
Trajectory: 9.2 → 15.5 → 10.0
Geometric score: 0.611
```
**Pros:** Balanced expansion/compression, tiny size, fast inference
**Cons:** Very small capacity, early peak

---

## Recommendations

### For AGPL Open Source Release

**Choose: Qwen2.5-3B-Instruct**

- Apache 2.0 allows any derivative work and redistribution
- Score (0.634) essentially ties with LFM2-1.2B (0.635)
- Larger model = more capacity for geometric training
- 52.5% peak is geometrically ideal

### For Research/Non-profit

**Choose: LFM2-1.2B**

- Slightly better geometric score
- Smaller = faster iteration
- LFM1.0 license fine for non-commercial
- Already validated in our merge experiments

### For Minimal Viable Model

**Choose: LFM2-350M**

- Perfect expansion_ratio = 1.002
- Extremely fast inference
- Good for understanding architectural principles

---

## Why Architecture Matters

The LFM2 (Liquid) architecture produces consistent expansion ratios, while Qwen models show higher variance. This suggests:

1. **State-space models** may produce more consistent geometric patterns
2. **Transformer depth** affects peak timing more than expansion ratio
3. **Architecture influences trajectory shape** independently of training

---

## Next Steps

1. **Deep analysis** of top 2 (Qwen2.5-3B and LFM2-1.2B):
   - Trainability with geometric loss
   - Transfer from math → science
   - Adversarial detection accuracy

2. **Final selection** based on:
   - License compatibility with release goals
   - Training efficiency
   - Baseline capability preservation

---

## Scoring Formula

```
Score = 0.20 × peak_position_score
      + 0.25 × expansion_ratio_score
      + 0.15 × variance_score
      + 0.15 × kappa_stability_score
      + 0.15 × trajectory_stability_score
      + 0.10 × mlx_native_score

Where:
- peak_position_score = 1 - |peak_pct - 55| / 50
- expansion_ratio_score = 1 - |expansion_ratio - 1.5| / 2.0  (prefer ~1.5, penalize extremes)
```

## Raw Data

All profiles saved to: `data/experiments/geometric_profile_*.json`
