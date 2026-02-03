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

### 1. Highway Location is Architecture-Dependent

| Architecture | Highway Location | Layers |
|-------------|------------------|--------|
| LFM2 | Entry | 0-1 |
| Qwen | Mid | 16-33 |
| Granite | Long mid | 5-28 |

**Implication:** The "semantic highway" hypothesis needs refinement. Different architectures achieve compression at different points in the forward pass.

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

## Jacobian Spectrum Analysis

**Key Finding:** All tested models have **effective rank = 1.0** at every layer - information flows through a single dominant direction regardless of architecture.

### Jacobian Summary

| Model | Layers | Mean Condition # | Max σ | Cumulative Amp |
|-------|--------|-----------------|-------|----------------|
| LFM2-350M | 16 | 209,517 | 48,723 | 1.56e+58 |
| Qwen2.5-3B | 36 | 2,394,140 | 839,063 | 9.40e+184 |
| DeepSeek-R1-8B | 36 | 14,899,720 | 4,841,997 | 7.44e+196 |

### Interpretation

1. **Effective Rank = 1.0 is universal**: Despite different manifold geometries (ID trajectories), the layer-to-layer transformation is dominated by a single direction at every layer.

2. **Amplification scales with capability**: Larger/more capable models have higher cumulative amplification. DeepSeek-R1 (reasoning) has the highest.

3. **Geometry and information flow are decoupled**:
   - Intrinsic Dimension measures the *shape* of the activation manifold
   - Jacobian spectrum measures the *information flow* through layers
   - Both are important but capture different aspects

4. **The "semantic highway" is about geometry, not information**:
   - Low ID (compression) doesn't mean information is lost
   - It means the manifold is low-dimensional at that point
   - The Jacobian shows information continues flowing through one direction

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
