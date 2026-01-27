# Model Selection Analysis: Finding the Optimal Geometry

**Date:** 2026-01-27

## Executive Summary

We profiled 8 small models (<3B parameters) to find the one with the best geometric properties for alignment work. The key metric is **compression/φ** (ideal = 1.0) which governs correct reasoning.

**Winner:** Depends on license requirements:
- **Truly Open (Apache 2.0):** Qwen2.5-3B-Instruct
- **Revenue-limited OK:** LFM2-1.2B (marginal geometric advantage)

---

## Complete Ranking

| Rank | Model | Score | Comp/φ | Peak Layer | License |
|------|-------|-------|--------|------------|---------|
| 1 | **LFM2-1.2B** | 0.635 | 1.127 | 55.6% | LFM1.0 (≤$10M) |
| 2 | **Qwen2.5-3B** | 0.634 | 1.157 | 52.5% | Apache 2.0 |
| 3 | LFM2-350M | 0.611 | **1.002** | 38.1% | LFM1.0 |
| 4 | LFM2-700M | 0.602 | 0.814 | 44.4% | LFM1.0 |
| 5 | LFM2.5-1.2B | 0.600 | 1.225 | 57.5% | LFM1.0 |
| 6 | Qwen3-1.7B | 0.533 | 1.368 | 45.4% | Apache 2.0 |
| 7 | Qwen2.5-Coder-0.5B | 0.469 | 1.379 | 25.4% | Apache 2.0 |
| 8 | Qwen2.5-Math-1.5B | 0.292 | 1.738 | 6.4% | Apache 2.0 |

---

## Key Findings

### 1. Liquid Foundation Models Have Natural φ Geometry

The LFM2 family shows remarkable geometric properties:
- **LFM2-350M achieves comp/φ = 1.002** (essentially perfect!)
- All LFM2 models have peak layers in the 38-57% range
- The Liquid (state-space) architecture naturally implements φ compression

**Why?** Liquid AI's recurrent/state-space design may inherently create the geometric structure we discovered governs correct reasoning. This is architecturally significant.

### 2. Qwen2.5-3B Is the Best Truly Open Model

With Apache 2.0 license and comp/φ = 1.157:
- No revenue restrictions
- Can be freely modified and released under AGPL
- Peak layer at 52.5% (ideal range)
- Strong baseline capabilities (3B scale)

### 3. Model Size Doesn't Determine Geometry

| Observation | Example |
|-------------|---------|
| Tiny can be perfect | LFM2-350M: comp/φ = 1.002 |
| Bigger isn't better | Qwen2.5-Math-1.5B: comp/φ = 1.738 (worst) |
| Architecture matters more | LFM2 family beats Qwen family despite same sizes |

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

### LFM2-350M (Perfect φ Ratio!)
```
Layers: 16
Hidden dim: 1024
Compression/φ: 1.002 (ESSENTIALLY PERFECT)
Peak layer: 38.1% (slightly early)
Trajectory: 9.2 → 15.5 → 10.0
Geometric score: 0.611
```
**Pros:** Nearly perfect compression/φ, tiny size, fast inference
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

- Perfect comp/φ = 1.002
- Extremely fast inference
- Good for understanding architectural principles

---

## Why Architecture Matters

The LFM2 (Liquid) architecture achieves near-perfect φ compression naturally, while Qwen models require training to approach it. This suggests:

1. **State-space models** may be geometrically superior for reasoning
2. **Transformer depth** affects peak timing more than compression ratio
3. **The φ constant is architecturally achievable** without post-training

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
      + 0.25 × compression_phi_score
      + 0.15 × constant_match_score
      + 0.15 × kappa_stability_score
      + 0.15 × trajectory_stability_score
      + 0.10 × mlx_native_score

Where:
- peak_position_score = 1 - |peak_pct - 55| / 50
- compression_phi_score = 1 - |comp/φ - 1.0|
```

## Raw Data

All profiles saved to: `data/experiments/geometric_profile_*.json`
