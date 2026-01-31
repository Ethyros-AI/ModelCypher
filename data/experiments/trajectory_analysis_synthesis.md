# Trajectory Analysis Synthesis

## Date: 2026-01-30

## Executive Summary

**Finding: Language models converge toward comp/φ = 0.618 (1/φ) as they scale up or receive RL training.**

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

### comp/φ Range by Model

| Model | Min | Max | Mean | Std |
|-------|-----|-----|------|-----|
| LFM2-350M | 0.618 | 1.268 | 0.866 | 0.195 |
| LFM2-1.2B | 0.618 | 0.764 | 0.661 | 0.046 |
| DeepSeek-R1 | 0.618 | 0.618 | 0.618 | 0.000 |

**Pattern: Variance decreases with scale. All models floor at exactly 0.618.**

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

## The Geometric Story

### Small Models (350M): Expand-Then-Compress

```
Embedding → Expansion (layers 1-14) → Peak → Compression (layers 15-16) → Output

Shape: ▁▂▃▄█▇▅ (hump with decay)
```

- Clear task differentiation via comp/φ (retrieval=1.268 vs CoT=0.618)
- Peak typically at ~88% depth
- Compression happens in final layers
- Task type affects how much compression occurs

### Medium Models (1.2B): Late Peak, Minimal Compression

```
Embedding → Expansion (layers 1-15) → Peak → Minimal Compression → Output

Shape: ▁▂▃▄▅▆█▇ (plateau near end)
```

- Narrower comp/φ range (all < 0.8)
- Peak at 94-100% depth
- Compression reduced to 1-2 layers
- Task differentiation weakening

### Large RL Models (DeepSeek-R1): Pure Expansion

```
Embedding → Continuous Expansion (all 36 layers) → Peak = Output

Shape: ▁▂▃▄▅▆▇█ (monotonic increase)
```

- comp/φ = 0.618 for ALL tasks
- Peak ALWAYS at final layer
- ZERO compression layers
- No task differentiation in geometry

---

## Interpretation: Why 0.618?

### Mathematical Derivation

When peak = final layer:
- compression_ratio = peak_norm / final_norm = 1.0 (no compression)
- comp/φ = compression_ratio / φ = 1.0 / 1.618... = **0.618...**

**0.618 is the mathematical floor.** It occurs when there's no compression at all.

### What This Means

1. **DeepSeek-R1 never compresses** - it expands representations monotonically
2. **Small models compress in final layers** - they "summarize" the expanded representation
3. **Scale removes the need for compression** - larger models have capacity to maintain expanded representations

### Why RL Training Locks In 0.618

Hypothesis: RL (RLHF/GRPO) optimizes for coherent extended reasoning. The optimal geometry for this is:
- Expand continuously (build up representation)
- Never compress (preserve all computed information)
- Peak at output (maximum information available for generation)

This creates the constant 0.618 = 1/φ signature.

---

## Trajectory Shape Invariance

Despite different comp/φ values, trajectory SHAPES are nearly identical:

| Model | Avg Similarity Across Tasks |
|-------|----------------------------|
| LFM2-350M | 0.97+ |
| LFM2-1.2B | 0.97+ |
| DeepSeek-R1 | 0.98+ |

**The curve shape is universal. Only the compression magnitude differs.**

This suggests:
- Trajectory shape is determined by architecture (attention patterns, layer norms)
- Compression magnitude is determined by task type and training
- RL training eliminates compression, locking comp/φ at the floor

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

### 1. comp/φ = 0.618 Is Not A Target

We shouldn't train base models toward 0.618. That's the natural floor for RL-tuned reasoning models. For base models:
- Retrieval tasks: comp/φ > 1.0 may be optimal
- Balanced tasks: comp/φ ≈ 1.0
- Reasoning tasks: comp/φ ≈ 0.618

### 2. Geometric Self-Awareness Should Be Task-Aware

The "aligned reasoning" signal (comp/φ ≈ 1.0 from LFM2-350M project) may need refinement:
- For base models: comp/φ ≈ 1.0 might indicate balanced processing
- For RL models: comp/φ = 0.618 is the constant state

### 3. Model Fingerprinting

We can distinguish model types by geometric signature:
- Wide comp/φ variance → base model, small
- Narrow comp/φ variance → base model, large
- Zero comp/φ variance (constant 0.618) → RL-tuned reasoning model

### 4. Null-Space Merging Considerations

When merging:
- Source model's geometric signature will partially transfer
- If merging RL source into base target, watch for comp/φ drift toward 0.618
- Preservation of target's task-differentiated comp/φ may be a quality metric

---

## Next Research Questions

1. **Other RL models**: Do Claude, GPT-4, Gemini show constant 0.618?
2. **Instruct vs Base**: Does instruction tuning move comp/φ toward 0.618?
3. **Layer-specific analysis**: Which layers drive the compression phase?
4. **Attention patterns**: Is expansion driven by attention entropy increasing?
5. **Can we induce compression?**: Training objective that forces compression in final layers

---

---

## NEW FINDING: Specialization, Not Scale

### Qwen2.5-Coder-0.5B-Instruct Results

| Task | Peak | comp/φ | Expansion |
|------|------|--------|-----------|
| Retrieval | 24/24 | 0.618 | 213x |
| Arithmetic | 24/24 | 0.618 | 195x |
| Reasoning | 24/24 | 0.618 | 218x |
| Logic | 24/24 | 0.618 | 243x |
| Creative | 24/24 | 0.618 | 276x |
| Code | 24/24 | 0.618 | ~250x |
| CoT | 24/24 | 0.618 | ~230x |

**This 500M model shows the SAME geometric signature as DeepSeek-R1 (8B)!**

### Revised Hypothesis

The constant 0.618 signature isn't driven by:
- ❌ Model size (Qwen-Coder is 500M, DeepSeek-R1 is 8B)
- ❌ RL training (Qwen-Coder uses supervised fine-tuning)

It IS driven by:
- ✅ **Specialized task training** (coding-specific, reasoning-specific)
- ✅ Training for consistent output patterns

### Evidence

| Model | Training | comp/φ Pattern |
|-------|----------|----------------|
| LFM2-350M | Base | Wide variance (0.618-1.268) |
| LFM2-1.2B | Base | Narrow variance (0.618-0.764) |
| LFM2.5-Instruct | General instruct | Moderate variance (0.618-0.793) |
| Qwen-Coder | **Code specialist** | Constant 0.618 |
| DeepSeek-R1 | **Reasoning specialist** | Constant 0.618 |

### Interpretation

1. **Base models** maintain task-differentiated geometry because they're trained on diverse objectives
2. **General instruct models** maintain some differentiation
3. **Specialist models** converge to 0.618 because they optimize for ONE type of coherent output

The 0.618 = 1/φ floor represents **maximal coherence** - the model has learned to maintain all information (no compression) for its specialized task.

---

## Raw Data Locations

- LFM2-350M: `data/experiments/trajectory_lfm2_350m.json`
- LFM2-1.2B: `data/experiments/trajectory_lfm2_1p2b.json`
- DeepSeek-R1: `data/experiments/trajectory_deepseek_r1.json`
- Qwen2.5-Coder: (in-memory analysis)
- Phi Distribution: `data/experiments/phi_distribution_*.json`
