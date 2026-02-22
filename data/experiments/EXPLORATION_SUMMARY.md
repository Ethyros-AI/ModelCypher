# Exploration Summary: Geometry of Language Models [EMPIRICAL]

## Date: 2026-01-30

## Executive Summary

We discovered a multi-level geometric structure that explains model behavior:

```
Level 1: expansion_ratio variance         → Model fingerprint (specialist vs base)
Level 2: Compression gate        → Layer behavior (compress vs expand)
Level 3: Dimension recovery      → Representation structure (collapse vs recover)
Level 4: Weight structure        → Sparse, low-rank projections in specialists
```

**Key insight: Geometry = Capability.** You cannot transfer specialist capability without changing geometry. Null-space merging preserves geometry, which is why capability didn't transfer.

---

## The Discoveries

### Discovery 1: Geometric Fingerprint [EMPIRICAL]

expansion_ratio variance identifies model specialization:

| Type | expansion_ratio Variance | Example |
|------|--------------------------|---------|
| Specialist | 0 (constant ~1.0) | Qwen-Coder, DeepSeek-R1 |
| General instruct | Moderate | Qwen2.5-3B-Instruct |
| Base | High (1.0-2.1) | LFM2-350M |

The floor (expansion_ratio = 1.0) occurs when peak = final layer (no compression).

### Discovery 2: Compression Gate [EMPIRICAL]

Base models have "compression gate" layers (esp. L15-16) that compress representations:

| Model | Always-Compress Layers | Compression Gate? |
|-------|------------------------|-------------------|
| LFM2-350M | L1, L3, L4, L6, L15 | **YES** (L15 = -2.1 units) |
| LFM2.5-Instruct | L1, L3, L4, L6, L7 | Weak (L15 = mixed) |
| Qwen-Coder | None | **NO** |
| DeepSeek-R1 | None | **NO** |

Specialists have no compression gate → constant geometry.

### Discovery 3: Dimension Recovery [EMPIRICAL]

The compression gate is actually a "dimension recovery" mechanism:

| Model | Mid-Layer EffDim | Final EffDim | Recovery? |
|-------|------------------|--------------|-----------|
| LFM2-350M | 1.0 | 3-8 | **YES** |
| DeepSeek-R1 | 1.5 | 1.2-1.5 | NO |
| Qwen-Coder | 2.5 | 1.5-2.0 | NO |

Base models recover effective dimensionality in final layers.
Different tasks recover different amounts → task differentiation.

### Discovery 4: Weight-Space Signature [EMPIRICAL]

Final layer projections differ in base vs specialist:

| Model | Final o_proj Rank | Sparsity |
|-------|-------------------|----------|
| LFM2-350M | 800.5 | 13% |
| Qwen-Coder | 606.5 | **42%** |

Specialists have sparse, low-rank final projections → can't project to high dimensions.

---

## The Unified Theory [CONJECTURAL]

```
Training Objective
       ↓
Weight Structure (sparse vs dense in final layers)
       ↓
Dimension Recovery (recover vs stay collapsed)
       ↓
Compression Gate (compress vs pure expand)
       ↓
expansion_ratio Signature (variance vs constant ~1.0)
       ↓
Capability Profile (task-flexible vs domain-coherent)
```

**Specialization = Learning to NOT recover dimensions.**

When a model is trained on one domain (coding, reasoning), it learns that dimension recovery is unnecessary. The final layers become sparse and low-rank. This locks in constant expansion_ratio ≈ 1.0 (flat trajectory).

---

## Capability Transfer Implications [EMPIRICAL]

### Why Null-Space Merge Preserved Geometry

The merged model showed:
- Same dimension recovery as target (3-8 dims)
- Same expansion_ratio variance as target (1.0-1.95)
- Same capability as target (code: 50%, reasoning: 100%)

Null-space projection protected the target's dimension recovery pattern.
The source's capability (constant ~1.0, no recovery) couldn't transfer.

### To Transfer Capability, Must Transfer Geometry

Options:
1. **Don't protect geometry** - Allow source geometry to modify target
2. **Selective transfer** - Only transfer mid-layers, not final layers
3. **Hybrid merge** - Interpolate geometry instead of null-space addition
4. **LoRA on final layers** - Train dimension recovery into specialist

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `explore_expansion_trajectories.py` | Layer-by-layer norm tracking |
| `layer_contribution_analysis.py` | Per-layer expansion/compression |
| `hidden_state_analysis.py` | Effective dimension per layer |
| `final_layer_weight_analysis.py` | Weight matrix rank and sparsity |

---

## Data Files

| File | Contents |
|------|----------|
| `geometric_fingerprint_discovery.md` | expansion_ratio variance analysis |
| `layer_roles_comparison.md` | Compression gate hypothesis |
| `dimension_recovery_discovery.md` | EffDim trajectory analysis |
| `trajectory_*.json` | Raw trajectory data per model |

---

## Next Research Directions

1. **Induce recovery**: Can LoRA on L15-16 teach dimension recovery?
2. **Geometry-aware merging**: Interpolate instead of null-space project?
3. **Training dynamics**: When does dimension recovery emerge during pretraining?
4. **Benchmark correlation**: Does higher dimension recovery = better downstream?
5. **Cross-architecture**: Do all transformer variants show dimension collapse + recovery?

---

## Commits Made

1. `feat: Discover geometric fingerprint` - expansion_ratio variance = specialization
2. `feat: Discover compression gate` - Layer-level mechanism
3. `feat: Discover dimension recovery` - Representation-level mechanism
4. `feat: Add weight analysis` - Weight-space signature
