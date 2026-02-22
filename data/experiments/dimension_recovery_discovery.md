# Dimension Recovery Discovery [EMPIRICAL]

## Date: 2026-01-30

## The Finding

**The compression gate is a "dimension recovery" mechanism.**

Base models recover effective dimensionality in their final layers.
Specialist models do not - they keep the representation collapsed.

---

## Evidence [EMPIRICAL]

### LFM2-350M (Base Model with Compression Gate)

```
Effective Dimension by Layer:

Emb:  6-10 dims (multi-dimensional representation)
L1-7: 4-8 dims  (moderate dimensionality)
L8:   1 dim     ← BIG COLLAPSE (the "big bang" layer)
L9-14: 1-1.5 dims (stays collapsed)
L15:  1.6-4.5 dims ← RECOVERY STARTS
L16:  3-8 dims  ← FULL RECOVERY
```

**Final layer recovers to multi-dimensional representation!**

### DeepSeek-R1 (Specialist without Compression Gate)

```
Effective Dimension by Layer:

Emb:  10-12 dims
L1-3: 2-3 dims ← IMMEDIATE COLLAPSE (different from LFM2!)
L4-34: 1.5-2.5 dims (stays low)
L35:  1.6-1.7 dims (no recovery)
L36:  1.2-1.5 dims ← STAYS COLLAPSED
```

**Final layer remains collapsed - no recovery!**

---

## Interpretation

### Why Base Models Have Task Differentiation

The dimension recovery in L15-16 is **task-dependent**:

| Task | Final EffDim (LFM2-350M) |
|------|--------------------------|
| Reasoning | 8.4 |
| Code | 5.3 |
| Creative | 4.9 |
| Retrieval | 3.0 |

Different tasks recover different amounts of dimensionality!

Retrieval stays more collapsed (just retrieve one fact).
Reasoning expands to high dimensionality (need multiple concepts).

**This is the source of expansion_ratio variance.**

### Why Specialist Models Have Constant expansion_ratio ≈ 1.0

DeepSeek-R1 has no dimension recovery - final layer is always ~1.5 dims:

| Task | Final EffDim (DeepSeek-R1) |
|------|----------------------------|
| Reasoning | 1.3 |
| Code | 1.5 |
| Creative | 1.4 |
| Retrieval | 1.2 |

All tasks end with same low dimensionality → same geometry → constant expansion_ratio.

---

## The Geometric Story (Revised)

### Base Models: Expand → Collapse → Recover

```
      Effective Dimension
   10 |▓▓▓▓▓
    8 |▓▓▓▓▓▓
    6 |▓▓▓▓▓▓▓▓                              ▓▓
    4 |▓▓▓▓▓▓▓▓▓                           ▓▓▓▓
    2 |▓▓▓▓▓▓▓▓▓▓                        ▓▓▓▓▓▓
    1 |           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
      +-----|-----|-----|-----|-----|-----|--->
          Emb   L4   L8   L12   L15  L16

      ↑ Early exploration
            ↑ Collapse (focus on core signal)
                              ↑ Recovery (task-appropriate expansion)
```

### Specialist Models: Collapse → Stay Collapsed

```
      Effective Dimension
   10 |▓
    8 |▓▓
    6 |▓▓▓
    4 |
    2 |   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    1 |                                     ▓▓
      +-----|-----|-----|-----|-----|-----|---->
          Emb   L10  L20  L30  L35  L36

      ↑ Immediate collapse
                                        ↑ No recovery
```

---

## Implications

### 1. Dimension Recovery = Generalization [CONJECTURAL]

The ability to recover dimensionality in final layers may be key to generalization.
Base models can adapt their final representation to the task.
Specialist models are locked into one representation style.

### 2. Training for Specialization Removes Recovery [CONJECTURAL]

Specialist training (coding, reasoning) seems to optimize away the recovery mechanism.
This makes the model more coherent for its domain but less flexible.

### 3. Capability Transfer Requires Geometry Transfer [CONJECTURAL]

To transfer capability, we may need to transfer the dimension recovery pattern.
Just transferring weights isn't enough - we need to transfer the geometric behavior.

### 4. Possible Intervention

Could we:
- Add a LoRA to final layers that induces dimension recovery?
- Train a specialist model with explicit dimension recovery loss?
- Merge in just the "recovery" layers from a base model?

---

## Metrics Summary

| Model | Big Bang Layer | Min EffDim | Final EffDim | Recovery? |
|-------|----------------|------------|--------------|-----------|
| LFM2-350M | L8 (10x jump) | 1.0 | 3-8 | **YES** |
| DeepSeek-R1 | L1-3 (gradual) | 1.2 | 1.2-1.5 | **NO** |

---

## Next Steps

1. Test LFM2.5-Instruct - does it have partial recovery?
2. Test Qwen-Coder - does it match DeepSeek-R1 pattern?
3. Can we induce recovery with LoRA on L15-16?
4. Is recovery correlated with downstream benchmark performance?
