# Tangent Space Alignment

> Local tangent-space agreement for geometric comparison.

---

## Why This Matters for Model Merging

Global rotations can hide local mismatches. Tangent space alignment measures
local geometric agreement around shared anchors, providing a scale-free signal
before null-space addition.

**In ModelCypher**: Implemented in `tangent_space_alignment.py` and driven by
geodesic neighbor graphs.

---

## What ModelCypher Computes

For each anchor point shared between source and target representations:

1. **Neighbors**: Build a k-NN graph using geodesic distances.
   - `k = sqrt(n_anchors)` clamped to `[2, n_anchors - 1]`.
2. **Tangent basis**: Compute local covariance from neighbor deltas and extract
   the top eigenvectors (tangent directions).
3. **Principal angles**: Compute principal cosines from `B_s^T B_t` and convert
   to angles.
4. **Aggregate** across anchors into per-layer metrics.

Per-layer outputs include:
- Mean/min/max cosine of principal angles
- Mean/median angle (radians)
- Coverage (anchors with valid bases / total)
- Neighbor count and tangent rank

---

## Algorithm (ModelCypher)

```python
def compute_layer_metrics(source_points, target_points):
    n = min(len(source_points), len(target_points))
    neighbor_count = clamp(sqrt(n), 2, n - 1)
    tangent_rank = clamp(neighbor_count // 2, 1, neighbor_count)
    eps = division_epsilon(source_points)

    source_neighbors = knn_geodesic(source_points, neighbor_count)
    target_neighbors = knn_geodesic(target_points, neighbor_count)

    cosines = []
    angles = []
    for i in range(n):
        basis_s = tangent_basis(source_points, source_neighbors[i], tangent_rank, eps)
        basis_t = tangent_basis(target_points, target_neighbors[i], tangent_rank, eps)
        if basis_s is None or basis_t is None:
            continue
        principal = principal_cosines(basis_s, basis_t, eps)
        cosines.extend(principal)
        angles.extend(arccos(clamp_01(c)) for c in principal)

    return metrics(...)
```

Notes:
- Batch mode uses `eigh` on covariance for speed; fallback uses `geodesic_svd`.
- All parameters are derived from data (no configuration classes).

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/tangent_space_alignment.py`](../../../src/modelcypher/core/domain/geometry/tangent_space_alignment.py)

**Key entry points**:
- `TangentSpaceAlignment.compute_layer_metrics()` - per-layer tangent metrics
- `compute_alignment_for_layers()` - batch report across layer mappings

**Design decisions**:
1. **Geodesic neighbors**: k-NN graph built from geodesic distances.
2. **Data-derived parameters**: neighbor count, rank, and epsilon all derived.
3. **Raw metrics only**: No thresholds or qualitative labels.
4. **Coverage tracking**: Reports anchor coverage to detect sparse validity.

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) - k-NN geodesic graph distances
- [frechet_mean.md](frechet_mean.md) - Geodesic averaging (separate module)
- [procrustes_analysis.md](procrustes_analysis.md) - Global alignment baseline
