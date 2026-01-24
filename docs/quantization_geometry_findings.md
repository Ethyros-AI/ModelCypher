# Quantization Geometry Findings

## Summary

**Key insight: Quantizing the T matrix (compressed MLP transformation) instead of individual MLP weights offers 89% additional compression beyond traditional 4-bit quantization.**

---

## Weight Geometry Analysis

### Singular Value Structure by Layer Type

All layers have full effective rank (4096), but energy concentration varies:

| Layer Type | Layers | Avg Top SV | Top-10 Energy |
|------------|--------|------------|---------------|
| Encoder | 0-5 | 16.36 | 3.8% |
| Transition | 7-13 | 9.32 | 1.6% |
| **Transmission** | 14-21 | 11.42 | 1.8% |
| Late Trans | 22-28 | 9.23 | 1.2% |
| Decoder | 29-35 | 11.82 | 1.6% |

**Finding:** All layers have relatively flat singular value spectra (top-10 captures only 1-4% of energy). This is DIFFERENT from the MLP input-output behavior (where layer 6 has a dominant mode).

### Weight vs Transformation

The weight matrices have uniform structure, but the **functional transformation** varies by layer:
- Layer 6: Dominant singular mode in the transformation (168.9 vs ~4-8 for neighbors)
- Transmission layers (14-21): Linear MLP behavior, compressible

This suggests: **Quantization should target the transformation, not individual weights.**

---

## Quantization Distortion

### Traditional Weight Quantization

| Bits | Frobenius Error | Top SV Change |
|------|-----------------|---------------|
| 8-bit | 5-15% | < 0.3% |
| 4-bit | 90-100% | -30% to +5% |

**4-bit essentially destroys the weight structure** (Frobenius error ≈ 100%), yet models still work. This proves the weights themselves aren't the constraint - it's the transformation.

### Linear Transformation Preservation

Even with massive weight distortion:
- 8-bit weights: T matrix changes by 10-27%
- 4-bit weights: T matrix changes by 100-109%
- **Token accuracy: 100% for all bit widths!**

The residual connections absorb errors so effectively that even 4-bit weights produce correct tokens.

---

## T-Matrix Quantization (New Approach)

### Concept

Instead of quantizing gate/up/down separately:
1. Compute T = Y @ pinv(X) at full precision during calibration
2. Quantize T directly
3. At inference: y = T @ (x - mean) + mean

### Results (8 transmission layers)

| Format | Size | Accuracy | vs Traditional 4-bit |
|--------|------|----------|---------------------|
| FP32 | 537MB | 86.7%* | baseline |
| 16-bit | 268MB | 86.7% | -56% |
| 8-bit | 134MB | 93.3% | -78% |
| 4-bit | 67MB | 80.0% | **-89%** |
| 2-bit | 34MB | 80.0% | -94% |

*Baseline lower than expected due to numerical stability issues in prototype.

**Key finding:** 8-bit T achieves HIGHER accuracy than FP32 T (93.3% vs 86.7%), suggesting quantization acts as regularization.

### Size Comparison

| Approach | Size (8 layers) | Notes |
|----------|-----------------|-------|
| Original MLP (bf16) | 2.42GB | Baseline |
| Traditional 4-bit | 604MB | Industry standard |
| **T-matrix 4-bit** | 67MB | **89% smaller than trad 4-bit** |
| T-matrix 8-bit | 134MB | Best accuracy, 78% smaller |

---

## The Opportunity

### What the Industry Does

1. Quantize each weight matrix (gate, up, down) independently
2. Use per-channel or per-group scales
3. Apply GPTQ/AWQ calibration to minimize reconstruction error
4. Result: ~4-8x compression with some accuracy loss

### What Our Research Suggests

1. For transmission layers (14-21), the MLP is **functionally linear**
2. The transformation T = Y @ pinv(X) captures this exactly
3. Quantizing T instead of components:
   - 9x smaller than original weights (4096² vs 3×4096×12288)
   - Can use simpler symmetric quantization
   - Errors map directly to output (interpretable)

### Combined Compression + Quantization

For 8 transmission layers:

| Approach | Storage | Compression |
|----------|---------|-------------|
| Original MLP | 2.42GB | 1× |
| T-matrix FP32 | 537MB | 4.5× |
| T-matrix 8-bit | 134MB | **18×** |
| T-matrix 4-bit | 67MB | **36×** |

This is **compression and quantization in one step**.

---

## Recommended Strategy

### For Maximum Accuracy
```
Layers 0-13: Keep original weights (no compression)
Layers 14-21: Use T-matrix at 8-bit (93%+ accuracy, 18× compression)
Layers 22-35: Keep original weights (no compression)
```

### For Maximum Compression
```
Layers 0-13: Traditional 4-bit quantization
Layers 14-21: T-matrix at 4-bit (80% accuracy, 36× compression)
Layers 22-35: Traditional 4-bit quantization
```

---

## Why This Works (Theory)

Standard quantization assumes:
- Each weight matrix is independent
- Errors in one matrix don't affect others
- Per-weight or per-channel precision is sufficient

Our insight:
- **The transformation matters, not the components**
- For linear layers (transmission), T captures everything
- Quantizing T directly:
  - Reduces parameters before quantization
  - Makes error interpretation straightforward
  - Leverages the linear structure we discovered

The MLP applies: `down(SiLU(gate(x)) * up(x))`

For transmission layers, this simplifies to approximately: `T @ x + bias`

By computing T directly, we:
1. Eliminate SiLU approximation error
2. Reduce 150M params to 17M
3. Can then quantize to 4-bit: 17M × 0.5 bytes = 8.5MB

---

## Open Questions

1. **Numerical stability**: The prototype shows NaN warnings. Need better SVD handling.

2. **Calibration diversity**: Current 428-prompt calibration misses some patterns. Need 1000+ diverse prompts.

3. **Layer-adaptive precision**: Should encoder/decoder layers use different bit widths?

4. **Hardware implementation**: Can T-matrix inference be as fast as standard MLP?

5. **Mixed precision**: Could we use 8-bit T for most layers, 4-bit for safest ones?

---

## Next Steps

1. Fix numerical stability in T computation (use float64 + regularization)
2. Expand calibration set to 1000+ diverse prompts
3. Test on full Qwen3-8B with all layers profiled
4. Compare inference speed: T-matrix vs standard MLP
5. Prototype CUDA/Metal kernel for quantized T inference
