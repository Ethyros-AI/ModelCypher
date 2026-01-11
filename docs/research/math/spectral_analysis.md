# Spectral Analysis of Weight Matrices

> Raw spectral measurements for source/target weight pairs.

---

## Why This Matters for Model Merging

Spectral measurements quantify scale and conditioning without introducing
heuristics. ModelCypher uses them to describe how far source and target
weights are apart before null-space addition.

**In ModelCypher**: Implemented in `spectral_analysis.py` and returns raw
metrics only.

---

## Metrics Computed

For source matrix $W_s$ and target matrix $W_t$:

- **Spectral ratio**: $\sigma_{max}(W_s) / \sigma_{max}(W_t)$
- **Spectral ratio symmetry**: $\min(r, 1/r)$ where $r$ is the spectral ratio
- **Condition number**: $\sigma_{max}(W_t) / \sigma_{min}(W_t)$ (capped by dtype threshold)
- **Source/target spectral norms**: $\sigma_{max}$ per matrix
- **Delta Frobenius**: $\|W_s - W_t\|_F$ using geodesic norms

For 1D vectors (biases, layer norms), ModelCypher uses geodesic norms and sets
condition number to 1.0.

---

## Algorithm (ModelCypher)

```python
def compute_spectral_metrics(source, target):
    eps = division_epsilon(target)
    max_cond = condition_threshold(target)

    if source.ndim == 1:
        source_norm = geodesic_norm(source)
        target_norm = geodesic_norm(target)
        delta_norm = geodesic_norm(source - target)
        ratio = source_norm / max(target_norm, eps)
        symmetry = min(ratio, 1 / max(ratio, eps))
        return metrics(condition_number=1.0, ...)

    source_f32 = astype(source, "float32")
    target_f32 = astype(target, "float32")
    _, s_source, _ = geodesic_svd(source_f32)
    _, s_target, _ = geodesic_svd(target_f32)

    sigma_max_s = s_source[0]
    sigma_max_t = s_target[0]
    sigma_min_t = s_target[-1]
    condition = min(sigma_max_t / max(sigma_min_t, eps), max_cond)

    ratio = sigma_max_s / max(sigma_max_t, eps)
    symmetry = min(ratio, 1 / ratio) if ratio > 0 else 0.0

    delta = geodesic_norm(source - target)
    return metrics(...)
```

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/spectral_analysis.py`](../../../src/modelcypher/core/domain/geometry/spectral_analysis.py)

**Key entry points**:
- `compute_spectral_metrics()` - per-weight spectral metrics
- `spectral_summary()` - aggregate statistics across weights

**Design decisions**:
1. **Data-derived thresholds**: Uses `division_epsilon` and `condition_threshold`.
2. **Backend-only math**: Geodesic SVD and norms, no NumPy.
3. **Raw measurements only**: No thresholds or qualitative labels.

---

## Related Concepts

- [intrinsic_dimension.md](intrinsic_dimension.md) - Related to effective rank
- [fisher_information.md](fisher_information.md) - Fisher eigenspectrum
