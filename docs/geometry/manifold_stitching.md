# Manifold Stitching: Cross-Architecture Model Merging

> **Status**: Prototype
> **Core Module**: `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

## The Problem: The "Bag of Numbers" Fallacy

Traditional model merging (Task Arithmetic, TIES, SLERP) works best for models initialized from the **same seed** and fine-tuned on different tasks. This is because they share a linear mode connectivity: neurons and directions are more likely to correspond.

For **disparate models** (different seeds, architectures, or sizes), this assumption fails.
- **Permutation symmetry**: internal features are only defined up to permutation.
- **Coordinate mismatch**: similar features can live in different bases.

Averaging these weights destroys information.

## The Solution: Geometric Manifold Stitching

**Manifold Stitching** treats models as **high-dimensional representation spaces** and aligns activations under a fixed probe setup.

> **Analogy (intuition)**: “stitching” adds a coordinate transform between two spaces so vectors point in comparable directions.
>
> **Non-claim**: a successful stitch on a probe corpus does not guarantee downstream capability retention; it must be evaluated.

### 1. The Intersection Map (Overlap Map)

We first determine *where* two models overlap. We do not assume full alignment.
- **Source Model**: probed with triangulated atlas probes (`TriangulatedProbeBuilder`).
- **Target Model**: probed with the same probe set.
- **Alignment**: compute the **Intersection Map**, which matches source/target dimensions using similarity on activation fingerprints.

This is a “Venn diagram” *analogy* of overlap under the probe setup (see `intersection_maps.md` for the operational details).

### 2. Procrustes Alignment (Rotation)

For the matching intrinsic dimensions, we solve the **Orthogonal Procrustes Problem**:

```
R* = argmin_R ||A R - B||_F^2   subject to  R^T R = I
```

where A and B are activation matrices of the source and target models on the shared probes. The solution is given by SVD:

```
U, Σ, V^T = SVD(B^T A)
R* = U V^T
```

This yields an **orthogonal transform** that aligns Model A’s activations to Model B’s activations under the probe corpus.

### 3. Applying the Alignment

The stitching process outputs rotation matrices and alignment clusters. These are used to **transform activations** (or weight-derived representations) before downstream merge steps. The alignment reduces representational mismatch; downstream behavior still needs evaluation.

## Implementation Details

The implementation in `manifold_stitcher.py` uses the Backend protocol for linear algebra:

```python
# Procrustes Alignment in ModelCypher
m = backend.matmul(backend.transpose(z_source), z_target)
u, _, vt = geodesic_svd(backend, m)
omega = backend.matmul(u, vt)

# Sign Correction (Ensure rotation, not reflection)
omega = _ensure_proper_rotation(u, vt, omega, backend)
```

### Key Components

1. **`TriangulatedProbeBuilder`**: builds probe sets from the atlas registry for comparable activations.
2. **Semantic prime anchors**: canonical inventory in `src/modelcypher/data/semantic_primes.json`, surfaced via `mc geometry primes`.
3. **`ContinuousFingerprint`**: a stable signature of activation geometry (magnitude + entropy).
4. **`IntersectionMap`**: correspondence between fingerprints (pre-alignment similarity only).

## Verification

We evaluate stitching with **raw representation similarity** (e.g., CKA/cosine on stitched activations) and, when available, downstream task checks.

- Compare against baselines for the same probe corpus.
- Avoid fixed thresholds; interpret relative changes and raw deltas.
