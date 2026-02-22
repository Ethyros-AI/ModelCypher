# Trajectory Analysis Synthesis [EMPIRICAL]

## Date: 2026-01-30

## Executive Summary

**Finding: Language models converge toward expansion_ratio ≈ 1.0 (peak ≈ final) as they scale up or receive RL training.**

**Note on metrics:** Earlier analysis divided by φ, yielding "0.618" when peak ≈ final. PHI_FINDINGS.md showed φ has no special significance. The raw expansion_ratio (peak_dim/final_dim) is the meaningful metric.

This isn't noise. It's a fundamental geometric property of how transformer representations evolve through layers.

---

## The Three Models

| Model | Size | Training | Layers |
|-------|------|----------|--------|
| LFM2-350M | 350M | Base | 16 |
| LFM2-1.2B | 1.2B | Base | 16 |
| DeepSeek-R1 | 8B | Base + RL | 36 |

---

## Key Metrics Compared

### expansion_ratio Range by Model (raw peak_dim/final_dim)

| Model | Min | Max | Mean | Std |
|-------|-----|-----|------|-----|
| LFM2-350M | 1.0 | 2.05 | 1.40 | 0.32 |
| LFM2-1.2B | 1.0 | 1.24 | 1.07 | 0.07 |
| DeepSeek-R1 | 1.0 | 1.0 | 1.0 | 0.000 |

**Pattern: Variance decreases with scale. RL models converge to expansion_ratio ≈ 1.0 (flat trajectory).**

### Peak Layer Location

| Model | Retrieval | Reasoning | CoT |
|-------|-----------|-----------|-----|
| LFM2-350M | 14/16 (88%) | 14/16 (88%) | 16/16 (100%) |
| LFM2-1.2B | 15/16 (94%) | 16/16 (100%) | 16/16 (100%) |
| DeepSeek-R1 | 36/36 (100%) | 36/36 (100%) | 36/36 (100%) |

**Pattern: Peaks migrate toward final layer with scale. DeepSeek-R1 always peaks at final layer.**

### Expansion vs Compression Layers

| Model | Task | Expansion | Compression |
|-------|------|-----------|-------------|
| LFM2-350M | Retrieval | 11 | 5 |
| LFM2-350M | CoT | 15 | 1 |
| LFM2-1.2B | Retrieval | 11 | 4 |
| LFM2-1.2B | CoT | 15 | 1 |
| DeepSeek-R1 | Retrieval | 36 | 0 |
| DeepSeek-R1 | CoT | 36 | 0 |

**Pattern: DeepSeek-R1 has ZERO compression layers. It's purely expansive.**

---

## The Geometric Story [EMPIRICAL]

### Small Models (350M): Expand-Then-Compress

```
Embedding → Expansion (layers 1-14) → Peak → Compression (layers 15-16) → Output

Shape: ▁▂▃▄█▇▅ (hump with decay)
```

- Clear task differentiation via expansion_ratio (retrieval ≈ 2.0 vs CoT ≈ 1.0)
- Peak typically at ~88% depth
- Compression happens in final layers
- Task type affects how much compression occurs

### Medium Models (1.2B): Late Peak, Minimal Compression

```
Embedding → Expansion (layers 1-15) → Peak → Minimal Compression → Output

Shape: ▁▂▃▄▅▆█▇ (plateau near end)
```

- Narrower expansion_ratio range (all < 0.8)
- Peak at 94-100% depth
- Compression reduced to 1-2 layers
- Task differentiation weakening

### Large RL Models (DeepSeek-R1): Pure Expansion

```
Embedding → Continuous Expansion (all 36 layers) → Peak = Output

Shape: ▁▂▃▄▅▆▇█ (monotonic increase)
```

- expansion_ratio ≈ 1.0 for ALL tasks (flat trajectory)
- Peak ALWAYS at final layer
- ZERO compression layers
- No task differentiation in geometry

---

## Interpretation: Why 1.0? [PROVEN]

### Mathematical Derivation

When peak = final layer:
- expansion_ratio = peak_dim / final_dim = 1.0 (flat trajectory)

**1.0 is the mathematical floor.** It occurs when there's no expansion/compression differential.

**Note:** Earlier analysis divided by φ (1.618), yielding 0.618 when the raw ratio was 1.0. This φ normalization has no theoretical justification (see PHI_FINDINGS.md).

### What This Means

1. **DeepSeek-R1 never compresses** - it expands representations monotonically
2. **Small models compress in final layers** - they "summarize" the expanded representation
3. **Scale removes the need for compression** - larger models have capacity to maintain expanded representations

### Why RL Training Produces Flat Trajectories [CONJECTURAL]

Hypothesis: RL (RLHF/GRPO) optimizes for coherent extended reasoning. The optimal geometry for this is:
- Expand continuously (build up representation)
- Never compress (preserve all computed information)
- Peak at output (maximum information available for generation)

This creates the constant expansion_ratio ≈ 1.0 (flat trajectory) signature.

---

## Trajectory Shape Invariance [EMPIRICAL]

Despite different expansion_ratio values, trajectory SHAPES are nearly identical:

| Model | Avg Similarity Across Tasks |
|-------|----------------------------|
| LFM2-350M | 0.97+ |
| LFM2-1.2B | 0.97+ |
| DeepSeek-R1 | 0.98+ |

**The curve shape is universal. Only the compression magnitude differs.**

This suggests:
- Trajectory shape is determined by architecture (attention patterns, layer norms)
- Compression magnitude is determined by task type and training
- RL training eliminates compression, locking expansion_ratio at the floor

---

## Early Layer Detection: Can We Predict Task Type?

### Early Slope Analysis (First 25% of Layers)

| Model | Retrieval Slope | CoT Slope |
|-------|----------------|-----------|
| LFM2-350M | -0.05 | -0.09 |
| LFM2-1.2B | +0.41 | +0.35 |
| DeepSeek-R1 | +3.59 | +3.63 |

**Pattern: Early slopes become more positive with scale.**

- Small models: Early contraction (negative slope)
- Larger models: Immediate expansion (positive slope)
- DeepSeek-R1: Aggressive early expansion

### Implication

Early layer behavior might distinguish base models from RL models:
- Negative early slope → base model, will compress later
- Positive early slope → likely RL-tuned, no compression coming

---

## Expansion Ratio: How Much Do Models "Expand"?

| Model | Typical Expansion Ratio |
|-------|------------------------|
| LFM2-350M | 5-12x |
| LFM2-1.2B | 21-29x |
| DeepSeek-R1 | 1000-1500x |

DeepSeek-R1 expands representations **100x more** than small base models.

This aligns with the RL hypothesis:
- More capacity → larger representations
- RL optimizes for information preservation → no compression
- Result: massive expansion ratios

---

## Implications for ModelCypher

### 1. expansion_ratio ≈ 1.0 Is Not A Universal Target

We shouldn't train all models toward flat trajectories. That's the natural state for RL-tuned reasoning models. For base models:
- Retrieval tasks: expansion_ratio > 1.5 may be typical
- Balanced tasks: expansion_ratio ≈ 1.2-1.5
- Reasoning tasks: expansion_ratio ≈ 1.0 (flat)

### 2. Geometric Self-Awareness Should Be Task-Aware

The trajectory shape varies by model type:
- For base models: expansion_ratio varies by task (1.0-2.0+ range)
- For RL/specialist models: expansion_ratio ≈ 1.0 is constant

### 3. Model Fingerprinting

We can distinguish model types by geometric signature:
- Wide expansion_ratio variance → base model, small
- Narrow expansion_ratio variance → base model, large
- Zero expansion_ratio variance (constant ~1.0) → RL-tuned or specialist model

### 4. Null-Space Merging Considerations

When merging:
- Source model's geometric signature will partially transfer
- If merging RL source into base target, watch for expansion_ratio drift toward 1.0
- Preservation of target's task-differentiated expansion_ratio may be a quality metric

---

## Next Research Questions

1. **Other RL models**: Do Claude, GPT-4, Gemini show constant flat trajectories?
2. **Instruct vs Base**: Does instruction tuning move expansion_ratio toward 1.0?
3. **Layer-specific analysis**: Which layers drive the compression phase?
4. **Attention patterns**: Is expansion driven by attention entropy increasing?
5. **Can we induce compression?**: Training objective that forces compression in final layers

---

---

## NEW FINDING: Specialization, Not Scale [EMPIRICAL]

### Qwen2.5-Coder-0.5B-Instruct Results

**Note:** All specialist models show expansion_ratio ≈ 1.0 (flat trajectory - peak at final layer).

| Task | Peak | expansion_ratio | Expansion |
|------|------|-----------------|-----------|
| Retrieval | 24/24 | ~1.0 | 213x |
| Arithmetic | 24/24 | ~1.0 | 195x |
| Reasoning | 24/24 | ~1.0 | 218x |
| Logic | 24/24 | ~1.0 | 243x |
| Creative | 24/24 | ~1.0 | 276x |
| Code | 24/24 | ~1.0 | ~250x |
| CoT | 24/24 | ~1.0 | ~230x |

**This 500M model shows the SAME geometric signature as DeepSeek-R1 (8B)!**

### Revised Hypothesis

The constant flat trajectory (expansion_ratio ≈ 1.0) isn't driven by:
- ❌ Model size (Qwen-Coder is 500M, DeepSeek-R1 is 8B)
- ❌ RL training (Qwen-Coder uses supervised fine-tuning)

It IS driven by:
- ✅ **Specialized task training** (coding-specific, reasoning-specific)
- ✅ Training for consistent output patterns

### Evidence

| Model | Training | expansion_ratio Pattern |
|-------|----------|-------------------------|
| LFM2-350M | Base | Wide variance (1.0-2.05) |
| LFM2-1.2B | Base | Narrow variance (1.0-1.24) |
| LFM2.5-Instruct | General instruct | Moderate variance (1.0-1.28) |
| Qwen-Coder | **Code specialist** | Constant ~1.0 |
| DeepSeek-R1 | **Reasoning specialist** | Constant ~1.0 |

### Interpretation

1. **Base models** maintain task-differentiated geometry because they're trained on diverse objectives
2. **General instruct models** maintain some differentiation
3. **Specialist models** converge to flat trajectories because they optimize for ONE type of coherent output

The expansion_ratio ≈ 1.0 floor represents **maximal coherence** - the model has learned to maintain all information (no compression) for its specialized task.

---

## Raw Data Locations

- LFM2-350M: `data/experiments/trajectory_lfm2_350m.json`
- LFM2-1.2B: `data/experiments/trajectory_lfm2_1p2b.json`
- DeepSeek-R1: `data/experiments/trajectory_deepseek_r1.json`
- Qwen2.5-Coder: (in-memory analysis)
- Phi Distribution: `data/experiments/phi_distribution_*.json`
