# Manifold Alignment: Intersection Maps & Cross-Model Stitching

> **Status**: Core Theory
> **References**:
> - `src/modelcypher/core/domain/geometry/manifold_stitcher.py`
> - `src/modelcypher/core/domain/geometry/intersection_similarity.py`

---

## Part I: Intersection Maps & Representation Overlap

### The Concept

The **Intersection Map** is a diagnostic of **representation overlap** between two models under a fixed probe setup.

> **Analogy (intuition)**: a "Venn diagram" of overlap.
>
> **Operationalization**: overlap is computed from **activation fingerprints** (semantic prime probes), not from weights or raw logits.

### Working Assumption [CONJECTURAL]

Two models trained on broadly similar data may encode partially similar features, even if they encode them in different coordinates. Conceptually, there may exist an approximately shared subspace:

```
M_A ∩ M_B ≈ S
```

where M_A and M_B are activation manifolds induced by the probe corpus and capture method.

### Computing the Map

We do not compare weights directly. We compare **activation fingerprints** for the same probe set.

1. **Probe**: Run semantic-prime probes on both models.
2. **Fingerprint**: Build `ActivationFingerprint` per probe (activated dimensions per layer).
3. **Match**: For each layer, compare source/target activated dimensions using a similarity mode.

### Similarity Modes

The intersection map uses similarity metrics over sparse activation signatures:
- **JACCARD**: set overlap of activated dimensions.
- **WEIGHTED_JACCARD**: magnitude-weighted overlap.
- **CKA**: cosine² on sparse activation vectors.

Each source dimension is paired with the target dimension that yields the **best similarity** in that layer.

### Output Structure

```python
@dataclass
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    raw_fingerprint_similarity: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]
```

`raw_fingerprint_similarity` is **pre-alignment only**. Use CKA separately for post-alignment quality checks.

### Layer-wise Dynamics [EMPIRICAL]

The Intersection Map evolves across depth because fingerprints are tracked per layer:

1. **Early layers**: often higher overlap (shared token/formatting features).
2. **Middle layers**: divergence is common (architecture and data differences).
3. **Late layers**: convergence may appear due to shared logit objectives, but it is not assured.

### Applications

Understanding the Intersection Map allows us to:
1. **Guide merging**: focus on regions with measured overlap to reduce interference.
2. **Support transfer**: decide when adapter or feature transfer is plausible.
3. **Detect drift**: large drops in overlap can indicate representational drift.

---

## Part II: Manifold Stitching (Cross-Architecture Merging)

### The Problem: The "Bag of Numbers" Fallacy

Traditional model merging (Task Arithmetic, TIES, SLERP) works best for models initialized from the **same seed** and fine-tuned on different tasks. This is because they share linear mode connectivity: neurons and directions are more likely to correspond.

For **disparate models** (different seeds, architectures, or sizes), this assumption fails:
- **Permutation symmetry**: internal features are only defined up to permutation.
- **Coordinate mismatch**: similar features can live in different bases.

Averaging these weights destroys information.

### The Solution: Geometric Manifold Stitching

**Manifold Stitching** treats models as **high-dimensional representation spaces** and aligns activations under a fixed probe setup.

> **Analogy (intuition)**: "stitching" adds a coordinate transform between two spaces so vectors point in comparable directions.
>
> **Non-claim**: a successful stitch on a probe corpus does not ensure downstream capability retention; it must be evaluated.

### Step 1: The Intersection Map (Overlap Map)

We first determine *where* two models overlap. We do not assume full alignment.
- **Source Model**: probed with triangulated atlas probes (`TriangulatedProbeBuilder`).
- **Target Model**: probed with the same probe set.
- **Alignment**: compute the **Intersection Map**, which matches source/target dimensions using similarity on activation fingerprints.

This is a "Venn diagram" *analogy* of overlap under the probe setup.

### Step 2: Procrustes Alignment (Rotation) [PROVEN]

For the matching intrinsic dimensions, we solve the **Orthogonal Procrustes Problem**:

```
R* = argmin_R ||A R - B||_F^2   subject to  R^T R = I
```

where A and B are activation matrices of the source and target models on the shared probes. The solution is given by SVD:

```
U, Σ, V^T = SVD(B^T A)
R* = U V^T
```

This yields an **orthogonal transform** that aligns Model A's activations to Model B's activations under the probe corpus.

### Step 3: Applying the Alignment

The stitching process outputs rotation matrices and alignment clusters. These are used to **transform activations** (or weight-derived representations) before downstream merge steps. The alignment reduces representational mismatch; downstream behavior still needs evaluation.

### Implementation Details

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
2. **Semantic prime anchors**: canonical inventory in `src/modelcypher/data/semantic_primes.json`, surfaced via `mc analyze concept-volume`.
3. **`ContinuousFingerprint`**: a stable signature of activation geometry (magnitude + entropy).
4. **`IntersectionMap`**: correspondence between fingerprints (pre-alignment similarity only).

### Verification

We evaluate stitching with **raw representation similarity** (e.g., CKA/cosine on stitched activations) and, when available, downstream task checks.

- Compare against baselines for the same probe corpus.
- Avoid fixed thresholds; interpret relative changes and raw deltas.
