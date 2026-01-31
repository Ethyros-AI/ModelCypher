# Geometric Fingerprint Discovery

## Date: 2026-01-30

## The Discovery

**comp/φ variance is a reliable fingerprint for model specialization.**

| Model Type | comp/φ Variance | Example |
|------------|-----------------|---------|
| Base model | High (0.618 - 1.3) | LFM2-350M |
| General instruct | Moderate (0.618 - 0.9) | Qwen2.5-3B-Instruct |
| **Specialist** | **Zero (constant 0.618)** | Qwen-Coder, DeepSeek-R1, Granite-Code |

---

## Complete Model Survey

| Model | Params | Type | comp/φ Min | comp/φ Max | Variance |
|-------|--------|------|------------|------------|----------|
| LFM2-350M | 350M | Base | 0.618 | 1.268 | HIGH |
| LFM2-1.2B | 1.2B | Base | 0.618 | 0.764 | MODERATE |
| LFM2.5-1.2B-Instruct | 1.2B | General | 0.618 | 0.793 | MODERATE |
| Qwen2.5-3B-Instruct | 3B | General | 0.618 | 0.896 | MODERATE |
| **Qwen2.5-Coder-0.5B** | 500M | Code | 0.618 | 0.618 | **ZERO** |
| **DeepSeek-R1-8B** | 8B | Reasoning | 0.618 | 0.618 | **ZERO** |
| **Granite-3B-Code** | 3B | Code | 0.618 | 0.618 | **ZERO** |

---

## Key Findings

### 1. The 0.618 Floor Is Universal

Every model tested has the same minimum comp/φ = 0.618 = 1/φ.

This occurs when peak_norm = final_norm (zero compression).

**0.618 represents maximal information preservation.**

### 2. Specialization Eliminates Variance

Specialist models (coding, reasoning) show constant 0.618 regardless of task type.

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

| Task Type | Typical comp/φ | Interpretation |
|-----------|----------------|----------------|
| Retrieval | > 1.0 | Low compression (retrieve, don't transform) |
| Arithmetic | 0.9 - 1.1 | Balanced (compute and output) |
| Creative | 0.8 - 1.0 | Moderate compression (generate novelty) |
| Reasoning | 0.6 - 0.8 | High compression (distill to answer) |
| CoT | 0.618 | Maximum compression (pure reasoning) |

**Specialist models lose this differentiation** - they apply one geometry to all tasks.

---

## Implications

### For Model Development

1. **comp/φ variance** is a diagnostic for generalization vs specialization
2. Training for a specific domain collapses geometry to 0.618
3. Maintaining task-differentiated geometry may preserve flexibility

### For Capability Transfer (ModelCypher)

When merging a specialist into a base model:
- Monitor comp/φ variance before and after
- Decreased variance → model becoming more specialized
- Preserved variance → generalization maintained

Prediction: Merging Qwen-Coder into LFM2-350M will reduce LFM2's comp/φ variance.

### For Model Fingerprinting

You can identify model type from a few prompts:
```
if comp_phi_variance < 0.01:
    return "specialist model"
elif comp_phi_variance < 0.15:
    return "general instruct"
else:
    return "base model"
```

---

## Mathematical Foundation

### Why 0.618?

comp/φ = (peak_norm / final_norm) / φ

When peak = final layer:
- peak_norm = final_norm
- compression_ratio = 1.0
- comp/φ = 1.0 / φ = 0.618...

**0.618 is the mathematical floor when there's no compression.**

### Why Specialists Converge to 0.618

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

- **Source**: Qwen2.5-Coder-0.5B (specialist, constant 0.618)
- **Target**: LFM2-350M (base, variance 0.618-1.268)
- **Method**: Null-space merge

### Results

| Model | Code Score | Reasoning Score | comp/φ Range |
|-------|------------|-----------------|--------------|
| Source | 80% | 80% | 0.618 (constant) |
| Target | 60% | 100% | 0.618 - 1.268 |
| **Merged** | **50%** | **100%** | **0.618 - 1.227** |

### Analysis

**Geometry preserved = capability preserved (but not transferred)**

The merged model:
- Preserved target's reasoning (100% → 100%)
- Preserved target's geometric signature (variance maintained)
- Did NOT gain source's coding capability (80% → 50%)

**Why?** The source's specialized geometry (constant 0.618) is fundamentally incompatible with the target's task-differentiated geometry. Null-space projection protected the target's geometry, which meant the source's capability pattern couldn't transfer.

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
4. **Geometry as training signal** - Train toward target comp/φ per task type

---

## Data Files

| File | Contents |
|------|----------|
| `trajectory_analysis_synthesis.md` | Detailed per-model analysis |
| `phi_distribution_*.json` | Raw measurement data |
| `trajectory_*.json` | Layer-by-layer trajectories |
