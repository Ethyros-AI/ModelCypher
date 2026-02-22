# Layer Roles Comparison [EMPIRICAL]

## Date: 2026-01-30

## Summary

The layer analysis reveals WHY specialist models show constant expansion_ratio ≈ 1.0:

**They have no compression layers.**

---

## LFM2-350M (Base Model)

```
Layer Pattern: Contract → Expand → Contract → Big Expand → Compress

L01 [COMPRESS] -------
L02 [EXPAND  ] +++++++
L03 [COMPRESS] -------
L04 [COMPRESS] -------
L05 [EXPAND  ] +++++++
L06 [COMPRESS] -------
L07 [MIXED   ] ++-----
L08 [EXPAND  ] +++++++ ← BIG BANG (+6.8 units)
L09 [MIXED   ]
L10 [MIXED   ] +++++
L11 [MIXED   ] ++++++
L12 [EXPAND  ] +++++++
L13 [EXPAND  ] +++++++
L14 [EXPAND  ] +++++++
L15 [COMPRESS] ------- ← COMPRESSION GATE (-2.1 units)
L16 [MIXED   ] +++----
```

**Key layers:**
- L08: "Big bang" expansion (+6.8 relative units)
- L15: "Compression gate" (-2.1 units) - **this is the geometry-defining layer**

---

## DeepSeek-R1 (Specialist)

```
Layer Pattern: Pure Expansion (with mid-layer fluctuation)

L01-L06 [EXPAND  ] +++++++ (all expand)
L07     [MIXED   ] ++++++-
L08-L12 [EXPAND  ] +++++++ (all expand)
L13     [MIXED   ] ++++--
L14     [EXPAND  ] +++++++
L15-L19 [MIXED   ] (some fluctuation)
L20-L36 [EXPAND  ] +++++++ ← FINAL 17 LAYERS ALL EXPAND

Final layer contributions:
L31: +71 units
L32: +51 units
L33: +78 units
L34: +94 units
L35: +226 units
L36: +347 units ← 50x more than LFM2-350M's total expansion
```

**Key observation:**
- NO compression gate
- Final 17 layers all expand
- L36 alone contributes more expansion than LFM2-350M's entire trajectory

---

## The Compression Gate Hypothesis [EMPIRICAL]

Base models have a "compression gate" in the final layers:

```
Information Flow:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Early     │ → │   Mid-Layer  │ → │  Final      │
│ Processing  │    │  Expansion   │    │ Compression │
└─────────────┘    └──────────────┘    └─────────────┘
                         ↓
                   Peak norm here

                         ↓ Compression gate
                   Final norm (reduced)
```

Specialist models remove this gate:

```
Information Flow:
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Early     │ → │   Mid-Layer  │ → │   Final      │
│ Processing  │    │  Expansion   │    │   Expansion  │
└─────────────┘    └──────────────┘    └──────────────┘
                                              ↓
                                        Peak = Final
                                              ↓
                                        expansion_ratio ≈ 1.0
```

---

## Why This Matters

### For Understanding

1. The flat trajectory (expansion_ratio ≈ 1.0) isn't about RL or model size - it's about **architecture behavior**
2. Compression gates are learned, not architectural - specialist training removes them
3. Task differentiation comes from variable compression, not variable expansion

### For Training

If we want to maintain task differentiation during fine-tuning:
- Preserve the compression gate behavior in final layers
- Don't optimize purely for output coherence

If we want specialist-like coherence:
- Train to reduce compression gate activation
- Optimize for constant output patterns

### For Merging

The compression gate is the geometry-defining structure. To transfer capabilities:
- Transferring mid-layer representations may work (similar across models)
- Transferring final-layer behavior changes geometry (breaks task differentiation)

---

---

## Additional Model Analysis

### LFM2.5-Instruct (General Instruct)

```
Always expand: L2, L5, L8, L10-14 (8 layers)
Always compress: L1, L3, L4, L6, L7 (5 layers)
Mixed: L9, L15, L16 (3 layers)
```

**Key finding**: L15 is now MIXED instead of always-compress.
The compression gate is weakened but not eliminated.
→ Explains narrower (but non-zero) expansion_ratio variance

### Qwen2.5-Coder (Specialist)

```
Always expand: L1-6, L15-23 (15 layers)
Always compress: ZERO
Mixed: L7-14, L24 (9 layers, expansion-biased)
```

**Key finding**: No compression gate at all.
→ Explains constant expansion_ratio ≈ 1.0

---

## Compression Gate Summary

| Model | Type | Compress Layers | Gate Present | expansion_ratio Variance |
|-------|------|-----------------|--------------|-----------------|
| LFM2-350M | Base | 5 (incl. L15) | Strong | High (0.65) |
| LFM2.5-Instruct | General | 5 (L15 weak) | Weakened | Moderate (0.18) |
| Qwen-Coder | Specialist | 0 | **Absent** | Zero |
| DeepSeek-R1 | Specialist | 0 | **Absent** | Zero |

---

## Next Steps

1. Does fine-tuning progressively weaken the compression gate?
2. Can we train a "hybrid" model with both coherence AND task differentiation?
3. Can we surgically restore the compression gate to a specialist model?
4. What happens if we transfer only the mid-layer representations (not final)?
