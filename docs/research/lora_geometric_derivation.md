# Complete Geometric Derivation for LoRA `[EMPIRICAL]`

> **⚠️ SUPERSEDED**: This document predates the discovery of the spectral scale bound.
> See [`lora_spectral_scale_bound.md`](./lora_spectral_scale_bound.md) for the current understanding.
> The scale formula derived here is correct but incomplete - it lacks the spectral norm constraint.

**Status**: Superseded
**Date**: 2026-02-03
**Impact**: Foundational - replaces hyperparameter tuning with geometry

## Summary

LoRA has three parameters that practitioners typically "tune": target modules, rank, and scale (alpha/rank). None of these are hyperparameters. All are derived from the geometry of the base weight matrices.

## The Fundamental Constraint `[PROVEN]`

The LoRA update is: W' = W + scale × (B @ A)

For this to respect the learned structure of W:

```
||scale × B @ A||_spectral ≤ σ_k(W)
```

Where σ_k(W) is the structural boundary singular value of W (Shannon effective-rank anchor;
precision diagnostics use max(m,n) × ε × σ_max).

This ensures the perturbation adds information at the edge of W's effective subspace rather than overwhelming its learned structure.

## Derivation 1: Target Modules `[EMPIRICAL]`

**Source**: Spectral decay analysis

| Projection | σ_k | Decay Ratio | Scale Bound |
|------------|-----|-------------|-------------|
| v_proj | 0.46 | 10× | ~0.5 |
| k_proj | 0.30 | 42× | ~0.3 |
| q_proj | 0.005 | 2,810× | ~0.002 |
| o_proj | 0.003 | 2,508× | ~0.002 |

**Geometric answer**: Target v_proj and k_proj. Their σ_k is 100× larger than q_proj/o_proj, providing 100× more room for perturbation.

The standard practice of targeting q_proj + v_proj is geometrically inconsistent: q_proj has a 0.002 scale bound while v_proj has a 0.5 bound.

## Derivation 2: Scale

**Source**: Scale bound formula

```
scale ≤ σ_k(W) / ||B @ A||_spectral
```

For trained adapters, ||B @ A||_spectral ≈ 0.7-1.9 (3-8× larger than initialization).

| Target | σ_k | ||B@A||_trained | Scale Bound |
|--------|-----|-----------------|-------------|
| v_proj | 0.46 | ~0.88 | 0.52 |
| k_proj | 0.30 | ~1.03 | 0.29 |

**Geometric answer**: Scale ≤ 0.3-0.5 for v_proj/k_proj.

The standard scale of 2.0 is 4-7× over bound even for the "good" targets.

## Derivation 3: Rank

**Source**: Spectral energy distribution

The singular value spectrum of W shows:
- 90% of energy in top ~680 dimensions (dominant subspace)
- 10% of energy in remaining ~300 dimensions (tail subspace)

LoRA should add capacity in the tail where there's room:

```
rank ≤ tail_dimensions = full_rank - rank_90 ≈ 300
```

A practical choice is a fraction of the tail:

```
rank = tail_dimensions / c, where c ≈ 3-10
rank ≈ 32-128
```

**Geometric answer**: Rank 32-128 for v_proj/k_proj (within 300 ceiling).

The standard rank of 8 is under-parameterized by geometry.

## The Missing Constraint

Standard LoRA training has no mechanism to constrain ||B @ A||_spectral. During training, it grows 3-8× from initialization, violating the geometric bound even with correct scale settings.

**Solutions**:

### Option 1: Conservative Scale
Use scale = 0.1-0.2, accepting that it will be somewhat under the bound but safe even after training growth.

### Option 2: Spectral Regularization
Add a loss term:
```python
loss = task_loss + λ * spectral_norm(B @ A)
```

This keeps ||B @ A|| bounded during training.

### Option 3: Spectral Normalization (Recommended)
Normalize the LoRA delta at each step:
```python
delta = (B @ A) / spectral_norm(B @ A)
scaled_delta = delta * σ_k  # Exactly at the bound
```

This guarantees the geometric constraint is satisfied throughout training.

## Complete Geometric Configuration

For Qwen3-8B attention layers:

```python
# Derived from geometry, not tuned
config = {
    "target_modules": ["v_proj", "k_proj"],  # From spectral decay
    "rank": 64,  # From tail subspace (300 ceiling)
    "scale": 0.3,  # From σ_k / ||B@A||_expected
    "spectral_norm": True,  # Guarantee constraint
}
```

## Validation

All 9 previously trained adapters violated these constraints:
- Targeted q_proj (2800× decay) instead of just v/k
- Used scale 2.0 (4-3000× over bounds)
- Used rank 8-16 (under-parameterized)

After applying geometric scaling at inference, the adapters produced coherent output (validated on GSM8K).

## Implementation

- Analysis scripts: `scripts/analyze_projection_spectra.py`, `scripts/analyze_sv_spectrum.py`
- Scale checking: `mc adapter analyze`
- Geometric application: `LoRASafetyService.apply_lora_geometric()`

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation. arXiv:2106.09685
- Miyato, T., et al. (2018). Spectral Normalization for GANs. arXiv:1802.05957
- See also: `lora_spectral_scale_bound.md`, `lora_projection_targeting.md`
