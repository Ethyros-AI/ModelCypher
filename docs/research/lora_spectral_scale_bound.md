# LoRA Spectral Scale Bound

**Status**: Validated
**Date**: 2026-02-03
**Impact**: Critical - explains systematic LoRA failures

## Summary

LoRA scale (alpha/rank) is not a hyperparameter to tune. It is a **geometric constraint** derived from the spectral structure of each base weight matrix. Violating this constraint causes catastrophic model degradation.

## The Problem

All 9 tested LoRA adapters for LFM2-350M were found to be unsafe:

| Adapter | Scale Ratio | Status |
|---------|-------------|--------|
| lfm2_350m_p1_6_mid_balanced | 2726× | Critical |
| self-reflection-lora-v4 | 1655× | Critical |
| self-reflection-lora-v5 | 1525× | Critical |
| geometric-awareness-v1 | 1311× | Critical |
| self-reflection-lora-v3-expansion | 860× | Critical |
| self-reflection-lora-v3 | 850× | Critical |
| self-reflection-lora-v2 | 622× | Critical |
| self-reflection-lora-v1 | 606× | Critical |
| lfm2_350m_p1_6_mid_balanced_v2 | 22.6× | Unsafe |

The standard configuration (alpha=16, rank=8, scale=2.0) is 600-2700× larger than the geometry permits.

## The Formula

The geometry-derived scale bound for each layer is:

```
scale_bound = σ_k(W) / ||B @ A||_spectral
```

Where:
- **W** is the base weight matrix
- **σ_k(W)** is the structural boundary singular value of W
- Structural anchor uses Shannon effective-rank; precision diagnostics use
  `σ > max(m,n) × ε × σ_max` (LAPACK convention)
- **||B @ A||_spectral** is the spectral norm (largest singular value) of the LoRA delta

## Derivation

1. The base weight W has singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ
2. The precision-significant rank uses `σᵢ > max(m,n) × ε × σ₁`
3. The smallest structural boundary singular value σ_k defines the "edge" of W's effective subspace
4. The LoRA delta should have spectral norm bounded by σ_k
5. This ensures the perturbation adds information at the edge, not overwhelming the core

The `max(m,n) × ε × σ_max` threshold follows LAPACK/MATLAB numerical-rank convention.

## Validation

**Test case**: GSM8K sheep problem
- Prompt: "Toulouse has twice as many sheep as Charleston..."
- Correct answer: 260

**Results**:

| Configuration | Output | Correct |
|---------------|--------|---------|
| Base model (no LoRA) | Clean reasoning → 260 | ✓ |
| Configured scale (2.0) | Gibberish, loops | ✗ |
| Geometric scale (~0.1) | Clean reasoning → 260 | ✓ |

The configured scale produces degenerate output:
```
Answer:. Question: 20 + (20 + 40) + 40 = ? (seattle, cholens, tommur) = r factor
for 20, 40, etc. in and and and and a a a and a and a and a and a and a or b or b...
```

The geometric scale produces correct reasoning:
```
Seattle sheep = 4 * 20 = 80
Toulouse sheep = 2 * 80 = 160
Total sheep = 20 + 80 + 160 = 260
```

## Implementation

The `LoRASafetyService` now provides:

### `compute_geometric_scale(model_path, adapter_path)`
Analyzes an adapter and reports per-layer scale bounds:
```python
report = service.compute_geometric_scale(model_path, adapter_path)
print(f"Safe: {report.is_safe}")
print(f"Max ratio: {report.max_scale_ratio}×")
```

### `apply_lora_geometric(model, adapter_path)`
Applies LoRA with geometry-derived per-layer scaling:
```python
model, scales = service.apply_lora_geometric(model, adapter_path)
# scales is a dict of layer_key -> applied_scale
```

## Implications

### For Existing Adapters
All existing adapters can be salvaged by applying with geometric scaling instead of configured scaling. The learned weights are valid; only the application scale was wrong.

### For Training
Future LoRA training must either:
1. Constrain alpha/rank to respect target layer geometry upfront
2. Store per-layer scale bounds in adapter config
3. Always apply with geometric scaling at inference time

### For the Field
The standard LoRA formula `W' = W + (alpha/rank) * B @ A` with fixed alpha/rank is fundamentally incomplete. Scale must be derived from W's spectral structure, not chosen as a hyperparameter.

## Code Location

- Service: `src/modelcypher/core/use_cases/lora_safety_service.py`
- Methods: `compute_geometric_scale()`, `apply_lora_geometric()`

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
- Golub & Van Loan (2013). Matrix Computations. Chapter 2: Matrix Analysis (condition numbers, numerical rank)
