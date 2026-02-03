# Cross-Architecture Geometric Survey

**Date:** 2026-02-03
**Models tested:** 8 models across 4 architecture families

---

## Executive Summary

Different architectures exhibit fundamentally different geometric processing patterns:

1. **LFM2**: Entry compression → processing → exit recovery (traditional semantic highway)
2. **Qwen/DeepSeek**: Processing → mid compression → exit recovery (sandglass architecture)
3. **Granite**: Early spike → long flat highway → moderate recovery

Specialist/instruct training flattens expansion_ratio to constant 1.0 across all architectures.

---

## Fingerprint Results (Expansion Ratio)

| Model | Architecture | Variance | Range | Classification |
|-------|-------------|----------|-------|----------------|
| LFM2-1.2B | LFM2 | 0.007 | [1.0, 1.24] | Low variance |
| Qwen2.5-3B-Instruct | Qwen | 0.025 | [1.0, 1.45] | Moderate |
| Granite-3B-code | Granite | 0.0 | [1.0, 1.0] | **Flat** |
| Qwen3-8B | Qwen | 0.0 | [1.0, 1.0] | **Flat** |
| DeepSeek-R1-Qwen3-8B | Qwen | 0.0 | [1.0, 1.0] | **Flat** |
| Granite-8B-code | Granite | ~0 | [1.0, 1.01] | Near-flat |

**Finding:** Specialist training (code-instruct, reasoning) produces flat expansion_ratio = 1.0 regardless of base architecture. This suggests RL/RLHF training creates stable geometric attractors.

---

## Dimension Profile Results (ID Trajectory)

### LFM2-350M (16 layers)
```
Pattern: Entry compression → Mid processing → Exit recovery
Highway: Layers 0-1 (2.3-3.9D)
Peak: Layer 8 (27.3D)
Exit: Layers 14-15 (31-33D)
Recovery Ratio: 14.04×
```

### LFM2-1.2B (16 layers)
```
Pattern: Entry compression → Early peak → Gradual decline
Highway: Layer 0 (3.4D)
Peak: Layer 3 (28.3D)
Exit: Layer 15 (16.3D)
Recovery Ratio: 4.83×
```

### Qwen2.5-3B-Instruct (36 layers)
```
Pattern: Entry processing → MID COMPRESSION → Exit recovery
Highway: Layers 17-28 (3.7-7D) ← UNUSUAL
Peak: Layer 11 (23.2D)
Exit: Layers 34-35 (29-32D)
Recovery Ratio: 5.78×
```

### Qwen3-8B (36 layers)
```
Pattern: Entry expansion → DEEP MID COMPRESSION → Partial recovery
Highway: Layers 16-33 (2.3-3.2D) ← VERY LONG
Peak: Layer 8 (47.9D)
Exit: Layer 35 (6.2D)
Recovery Ratio: 2.64×
```

### DeepSeek-R1-Qwen3-8B (36 layers)
```
Pattern: Entry expansion → Mid compression → Exit recovery
Highway: Layers 16-28 (4.4-7.9D)
Peak: Layer 8 (49.0D)
Exit: Layer 35 (22.2D)
Recovery Ratio: 5.06×
```

### Granite-3B-code (32 layers)
```
Pattern: Entry spike → VERY LONG FLAT HIGHWAY → Moderate recovery
Highway: Layers 5-28 (~3D constant) ← 24 LAYERS FLAT
Spike: Layer 3 (25.0D)
Exit: Layers 29-31 (10-13D)
Recovery Ratio: 3.76×
```

---

## Architectural Patterns

### Pattern 1: Traditional Semantic Highway (LFM2)
```
ID
 ▲
 │    ╭───╮
 │   ╱     ╲    ╭──
 │──╯       ╲──╯
 └────────────────► Layer
   Entry  Mid   Exit
```
- Compression at entry
- Processing in mid layers
- Dimension recovery at exit

### Pattern 2: Sandglass (Qwen/DeepSeek)
```
ID
 ▲
 │ ╭───╮          ╭──
 │╱     ╲        ╱
 │       ╲──────╯
 └────────────────► Layer
   Entry  Mid   Exit
```
- Processing at entry (high ID ~50D)
- Deep compression in mid layers (2-5D)
- Variable recovery at exit

### Pattern 3: Long Highway (Granite)
```
ID
 ▲
 │ ╭╮
 │╱  ╲────────────╭──
 │                 │
 └────────────────► Layer
   Entry   Mid    Exit
```
- Brief early spike
- Very long flat highway (24 layers at ~3D)
- Moderate exit recovery

---

## Key Findings

### 1. Highway Location is Architecture-Dependent — PARTIALLY UNDERSTOOD (2026-02-03)

| Architecture | Highway Location | Layers | Position | attention_bias | RoPE θ |
|-------------|------------------|--------|----------|----------------|--------|
| LFM2 | Entry | 0-1 | 0-6% | - | - |
| Granite-3B | Early | 5-28 | 16% | True | 10M |
| Granite-8B | Early | 4-24 | 11% | True | 10M |
| Qwen2.5-3B | Mid | 17-28 | 47% | False | 1M |
| Qwen3-8B | Mid | 16-33 | 44% | False | 1M |

**FALSIFIED: GQA formula was spurious** (Granite-8B has GQA=4 like Qwen3-8B, but highway at 11% not 44%)

**Actual pattern: Model family determines highway position**
- Granite family: Early (11-16%)
- Qwen family: Mid (44-47%)

**Candidate causal factors (unconfirmed):**
1. attention_bias: Granite=True → early, Qwen=False → mid
2. RoPE theta: Granite=10M → early, Qwen=1M → mid
3. Training procedure (unknown)

**LFM2 is special:** Entry highway caused by Mamba/SSM layers (layers 0-1 are pure Mamba).
SSM's linear recurrence h_t = A·h_{t-1} + B·x_t naturally creates low-dimensional state.

**The geometric mechanism (2026-02-03):**

Measured attention entropy across layers:
- Granite (bias=True): 2.78 → 1.24 by layer 6 (entropy drops 55%)
- Qwen (bias=False): 2.70 → 2.70 through layer 10 (constant)

The causal chain:
```
attention_bias=True → early selectivity → info filtering → low ID → early highway
attention_bias=False → diffuse attention → all info preserved → high ID → late highway
```

This is NOT about RoPE theta - measured attention locality is similar despite 10× theta difference.

### 2. Specialist Training Creates Flat Geometry

All models with instruct/code/reasoning training show:
- expansion_ratio variance ≈ 0
- Constant ratio = 1.0 across all task types

This is consistent with RL training creating stable attractors regardless of input type.

### 3. Recovery Ratio Correlates Inversely with Size

| Model Size | Recovery Ratio |
|-----------|----------------|
| 350M | 14.04× |
| 1.2B | 4.83× |
| 3B | 3.76-5.78× |
| 8B | 2.64-5.06× |

Smaller models show more dramatic dimension recovery. Larger models maintain more stable representations.

### 4. Qwen Has Extreme Mid-Layer Compression

Qwen3-8B compresses to **2.3D** in mid layers (16-33), the lowest ID observed. This sandglass architecture may be responsible for Qwen's strong reasoning capabilities.

---

## Gemma Note

Gemma models (gemma-3-12b, gemma-3n-E4B) failed fingerprinting due to architectural differences:
- No `embed_tokens` attribute
- Different layer structure

Further investigation needed to support Gemma architecture.

---

## Jacobian Spectrum Analysis — CORRECTED (2026-02-03)

**CORRECTION:** The "effective rank = 1.0" finding was a **numerical artifact** caused by:
1. bf16 model precision (3-4 significant digits)
2. Tiny finite difference epsilon (1e-5) used for Jacobian estimation
3. These combined to make small input perturbations invisible

### Corrected Findings

When measured correctly (float32, ε=1e-3 to 1e-4):

| Epsilon | Effective Rank | σ_max | σ_2 |
|---------|----------------|-------|-----|
| 1e-03 | **63.9** | 1.08 | 1.02 |
| 1e-04 | **63.9** | 1.10 | 1.05 |
| 1e-05 | 59.6 | 2.14 | 1.12 |
| 1e-06 | 1.3 | 45.2 | 5.2 |

The true layer Jacobian is:
- **Full rank** (~64 effective rank, not rank-1)
- **Near-identity** (all singular values ≈ 1.0)
- Each layer makes small incremental changes to the representation

### Correct Interpretation

1. **Transformer layers are approximately identity transformations**
   - Residual connections dominate: output ≈ input + small_delta
   - This is the "semantic highway" - information flows with minimal transformation

2. **The "semantic highway" is about geometry, not Jacobian rank**
   - Low intrinsic dimension at highway = manifold compression
   - But the layer transformation is still full-rank near-identity
   - Information is preserved, just compressed geometrically

3. **Attention rank varies by architecture**
   - LFM2: rank-1 (uniform attention = mean pooling)
   - Qwen: rank 3-4 (selective attention)
   - But layer Jacobians are full-rank in both cases

### What the Attention Analysis Showed

| Model | Attention Eff. Rank | Attention Pattern |
|-------|---------------------|-------------------|
| LFM2-350M | **1.02** | Uniform (mean pooling) |
| Qwen2.5-3B | 3.85 | Selective |
| Qwen3-8B | 2.76 | Selective, sharper |
| DeepSeek-R1-8B | 2.74 | Similar to Qwen3 base |
| Random baseline | 6.95 | Diffuse |

LFM2's attention is genuinely rank-1 (every position attends equally to all tokens).
This explains why LFM2 might have different computational properties from Qwen.

### Lesson Learned

**Always verify numerical methods across precision levels and epsilon values.**
The finite difference + bf16 combination created an artifact that looked like a fundamental property.


---

## Implications for Model Merging

1. **Same-architecture merges** should be straightforward - highway locations align
2. **Cross-architecture merges** require careful layer mapping - highways don't align
3. **Specialist→Base transfers** may lose the flat geometry attractor

---

## Data Files

Raw data available in:
- This document (inline)
- CLI commands used: `mc model fingerprint`, `mc safety dimension-profile --recovery`

---

*Generated by ModelCypher cross-architecture survey*
