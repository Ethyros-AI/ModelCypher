# Geometric Fingerprint Discovery

## Date: 2026-01-30

## The Discovery

**expansion_ratio variance is a reliable fingerprint for model specialization.**

| Model Type | expansion_ratio Variance | Example |
|------------|--------------------------|---------|
| Base model | High (1.0 - 2.1) | LFM2-350M |
| General instruct | Moderate (1.0 - 1.5) | Qwen2.5-3B-Instruct |
| **Specialist** | **Zero (constant ~1.0)** | Qwen-Coder, DeepSeek-R1, Granite-Code |

---

## Complete Model Survey

| Model | Params | Type | expansion_ratio Min | expansion_ratio Max | Variance |
|-------|--------|------|---------------------|---------------------|----------|
| LFM2-350M | 350M | Base | 1.0 | 2.05 | HIGH |
| LFM2-1.2B | 1.2B | Base | 1.0 | 1.24 | MODERATE |
| LFM2.5-1.2B-Instruct | 1.2B | General | 1.0 | 1.28 | MODERATE |
| Qwen2.5-3B-Instruct | 3B | General | 1.0 | 1.45 | MODERATE |
| **Qwen2.5-Coder-0.5B** | 500M | Code | 1.0 | 1.0 | **ZERO** |
| **DeepSeek-R1-8B** | 8B | Reasoning | 1.0 | 1.0 | **ZERO** |
| **Granite-3B-Code** | 3B | Code | 1.0 | 1.0 | **ZERO** |

---

## Key Findings

### 1. The 1.0 Floor Is Universal

Every model tested has the same minimum expansion_ratio = 1.0.

This occurs when peak_dim = final_dim (flat trajectory).

**expansion_ratio = 1.0 represents maximal information preservation (no compression).**

### 2. Specialization Eliminates Variance

Specialist models (coding, reasoning) show constant expansion_ratio ≈ 1.0 regardless of task type.

They've learned one optimal geometric pattern for their domain.

General models maintain task-differentiated geometry.

### 3. Expansion Ratio Scales with Depth and Training

| Model | Layers | Expansion Ratio |
|-------|--------|-----------------|
| LFM2-350M | 16 | 5-12x |
| LFM2-1.2B | 16 | 21-29x |
| Qwen-Coder | 24 | 195-276x |
| DeepSeek-R1 | 36 | 1000-1500x |
| Granite-Code | 32 | 1500-2300x |

Deeper models expand more. Code-specialist models expand MUCH more.

### 4. Task-Type Patterns (Base Models Only)

For base models with variance:

| Task Type | Typical expansion_ratio | Interpretation |
|-----------|-------------------------|----------------|
| Retrieval | 1.5 - 2.0+ | High expansion (retrieve and elaborate) |
| Arithmetic | 1.3 - 1.6 | Moderate expansion (compute and output) |
| Creative | 1.2 - 1.5 | Moderate expansion (generate novelty) |
| Reasoning | 1.0 - 1.3 | Lower expansion (distill to answer) |
| CoT | ~1.0 | Flat trajectory (pure reasoning) |

**Specialist models lose this differentiation** - they apply one geometry to all tasks.

---

## Implications

### For Model Development

1. **expansion_ratio variance** is a diagnostic for generalization vs specialization
2. Training for a specific domain collapses geometry to flat trajectories (expansion_ratio ≈ 1.0)
3. Maintaining task-differentiated geometry may preserve flexibility

### For Capability Transfer (ModelCypher)

When merging a specialist into a base model:
- Monitor expansion_ratio variance before and after
- Decreased variance → model becoming more specialized
- Preserved variance → generalization maintained

Prediction: Merging Qwen-Coder into LFM2-350M will reduce LFM2's expansion_ratio variance.

### For Model Fingerprinting

You can identify model type from a few prompts:
```
if expansion_ratio_variance < 0.05:
    return "specialist model"
elif expansion_ratio_variance < 0.3:
    return "general instruct"
else:
    return "base model"
```

---

## Mathematical Foundation

### Why 1.0?

expansion_ratio = peak_dim / final_dim

When peak = final layer:
- peak_dim = final_dim
- expansion_ratio = 1.0

**1.0 is the mathematical floor when there's no compression.**

**Note:** Earlier analysis divided by φ (1.618), yielding "0.618" when the raw ratio was 1.0. PHI_FINDINGS.md showed this φ normalization has no theoretical justification.

### Why Specialists Converge to 1.0

Specialist training optimizes for one coherent output type.

The optimal geometry is:
1. Expand continuously (build representation)
2. Never compress (preserve all information)
3. Peak at output (maximum info available)

This is the **greedy information preservation** strategy.

### Why General Models Maintain Variance

General training exposes the model to diverse objectives:
- Some tasks benefit from compression (reasoning → answer)
- Some tasks benefit from expansion (retrieval → elaborate)

The model learns task-appropriate geometry.

---

---

## Capability Transfer Experiment

### Setup

- **Source**: Qwen2.5-Coder-0.5B (specialist, constant ~1.0)
- **Target**: LFM2-350M (base, variance 1.0-2.05)
- **Method**: Null-space merge

### Results

| Model | Code Score | Reasoning Score | expansion_ratio Range |
|-------|------------|-----------------|------------------------|
| Source | 80% | 80% | ~1.0 (constant) |
| Target | 60% | 100% | 1.0 - 2.05 |
| **Merged** | **50%** | **100%** | **1.0 - 1.99** |

### Analysis

**Geometry preserved = capability preserved (but not transferred)**

The merged model:
- Preserved target's reasoning (100% → 100%)
- Preserved target's geometric signature (variance maintained)
- Did NOT gain source's coding capability (80% → 50%)

**Why?** The source's specialized geometry (constant ~1.0) is fundamentally incompatible with the target's task-differentiated geometry. Null-space projection protected the target's geometry, which meant the source's capability pattern couldn't transfer.

### Implication

**Geometric signature IS capability.**

You cannot:
- Transfer specialist capability while preserving generalist geometry
- Transfer generalist flexibility while preserving specialist coherence

Capability transfer requires geometric change. Geometry protection prevents capability transfer.

---

## Future Research

1. **Geometry-aware merging**: Allow partial geometry transfer
2. **Selective dimension transfer**: Transfer coding dimensions only
3. **Does variance predict capability?** Higher variance → better generalization?
4. **Geometry as training signal** - Train toward target expansion_ratio per task type

---

## Data Files

| File | Contents |
|------|----------|
| `trajectory_analysis_synthesis.md` | Detailed per-model analysis |
| `phi_distribution_*.json` | Raw measurement data |
| `trajectory_*.json` | Layer-by-layer trajectories |
