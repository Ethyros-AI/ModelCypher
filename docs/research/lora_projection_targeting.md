# LoRA Projection Targeting: Spectral Analysis `[EMPIRICAL]`

> **⚠️ SUPERSEDED**: This document's analysis is valid but all LFM2-350M adapters were deleted.
> See [`lora_spectral_scale_bound.md`](./lora_spectral_scale_bound.md) for the discovery that
> invalidated the existing adapters. The spectral analysis here remains useful for understanding
> projection geometry, but future adapters will use `apply_lora_geometric()` for safe scaling.

**Status**: Superseded (adapters deleted)
**Date**: 2026-02-03
**Impact**: Critical - explains why standard q_proj targeting fails

## Summary

Standard LoRA practice targets `q_proj` and `v_proj`. This is half-wrong.

- **v_proj**: Safe target (10× spectral decay)
- **q_proj**: Dangerous target (2,810× spectral decay)

The spectral structure of attention projections varies by 250× between projection types.
Targeting the wrong projections guarantees scale violations.

## Analysis Results (Qwen3-8B)

| Projection | σ_k (mean) | Decay Ratio | Safe Scale | Rating |
|------------|------------|-------------|------------|--------|
| v_proj | 0.462 | 10× | ~0.23 | EXCELLENT |
| k_proj | 0.305 | 42× | ~0.15 | EXCELLENT |
| q_proj | 0.005 | 2,810× | ~0.003 | AVOID |
| o_proj | 0.003 | 2,508× | ~0.002 | AVOID |

**Key metric**: Spectral decay = σ_max / σ_k

Higher decay means steeper singular value drop-off, which means smaller σ_k,
which means tighter geometric scale bound.

## Why This Pattern Exists

### Projection Roles in Attention

1. **v_proj (values)**: Projects token representations to value vectors.
   Content-carrying, relatively robust to perturbation.

2. **k_proj (keys)**: Projects to key space for attention addressing.
   Moderate sensitivity - affects what attends to what.

3. **q_proj (queries)**: Projects to query space for attention computation.
   High precision required - small changes affect all attention patterns.

4. **o_proj (output)**: Combines multi-head outputs back to residual stream.
   Critical bottleneck - funnels all heads through single projection.

### Shape Correlation

- q_proj, o_proj: 4096×4096 (full hidden_dim)
- k_proj, v_proj: 1024×1024 (per-head dimension)

The larger matrices have steeper spectral decay (not lower rank - they're nearly
full rank, but with much larger dynamic range).

## Implications for LoRA Training

### Standard Practice (BROKEN)
```python
target_modules = ["q_proj", "v_proj"]
alpha = 16
rank = 8
scale = 2.0  # alpha / rank
```

With this configuration:
- q_proj: scale is 667× over geometric bound (2.0 / 0.003)
- v_proj: scale is 9× over geometric bound (2.0 / 0.23)

### Recommended Practice
```python
target_modules = ["v_proj", "k_proj"]  # Not q_proj!
alpha = 1
rank = 8
scale = 0.125  # Within geometric bounds for both
```

Or use `apply_lora_geometric()` which derives per-layer scale automatically.

### If You Must Target q_proj/o_proj

Use per-layer scaling:
```python
scales = {
    "q_proj": 0.003,
    "k_proj": 0.15,
    "v_proj": 0.23,
    "o_proj": 0.002,
}
```

## Validation

This analysis was performed on Qwen3-8B. The pattern should hold for other
transformer architectures but the exact values will vary. Always run
`mc adapter analyze` on new model/adapter combinations.

## Code

Analysis script: `scripts/analyze_projection_spectra.py`

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation. arXiv:2106.09685
- See also: `docs/research/lora_spectral_scale_bound.md`
