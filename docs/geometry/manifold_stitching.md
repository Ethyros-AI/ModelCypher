# Manifold Stitching: Cross-Architecture Model Merging

> **Status**: Prototype / In progress (see `../PARITY.md`)
> **Core Module**: `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

## The Problem: The "Bag of Numbers" Fallacy

Traditional model merging (Task Arithmetic, TIES, SLERP) works well for models initialized from the **same seed** and fine-tuned on different tasks. This is because they share a "Linear Mode Connectivity" (LMC)—their loss landscapes are connected by a linear path, and "Neuron 42" in Model A roughly corresponds to "Neuron 42" in Model B.

However, for **disparate models** (different random seeds, different architectures, or even different sizes), this assumption fails.
-   **Permutation Symmetry**: Many internal features are only defined up to permutations of hidden dimensions across equivalent parameterizations.
-   **Coordinate mismatch**: Even when two models encode similar features, they may do so in different bases; similarity metrics are often insensitive to orthogonal transforms.

Trying to average these weights destroys the information.

## The Solution: Geometric Manifold Stitching

**Manifold Stitching** treats models as **high-dimensional representation spaces** and attempts to reduce mismatch by aligning activations under a fixed probe setup.

> **Analogy (intuition)**: “stitching” is like adding a coordinate transform between two spaces so vectors point in more comparable directions.
>
> **Non-claim**: a successful stitch on a probe corpus does not guarantee downstream capability retention; it must be evaluated.

### 1. The Intersection Map ("Venn Diagram")
We first determine *where* two models overlap. We don't assume full alignment.
-   **Source Model**: Probed with a standardized probe corpus (see `src/modelcypher/core/domain/geometry/probe_corpus.py`).
-   **Target Model**: Probed with the same corpus.
-   **Alignment**: We compute the **Intersection Map**, identifying which dimensions in Model A correlate with which in Model B.

This creates a “Venn diagram” *analogy* of overlap under that probe setup (see `intersection_maps.md` for details).

### 2. Procrustes Alignment (Rotation)
For the matching intrinsic dimensions, we solve the **Orthogonal Procrustes Problem**:

$$ R^* = \arg\min_R \|A R - B\|_F^2 \quad \text{s.t.} \quad R^T R = I $$

Where $A$ and $B$ are the activation matrices of the source and target models on the shared probe corpus.
The solution is given by SVD:
$$ U, \Sigma, V^T = \text{SVD}(B^T A) $$
$$ R^* = U V^T $$

This yields an **orthogonal transform** (a best-fit rotation/reflection constraint) that aligns Model A’s activations to Model B’s activations under the probe corpus.

### 3. Stitching Layer
We insert a learnable (or computed) linear layer between the models.
-   **Input**: Model A's hidden states (rotated).
-   **Output**: Model B's expected input states.
-   **Result**: Reduced representational mismatch, measured by similarity metrics; downstream behavior may still change and must be evaluated.

## Implementation Details

The implementation in `manifold_stitcher.py` uses rigorous `MLX` linear algebra:

```python
# Procrustes Alignment in ModelCypher
m = z_source.T @ z_target
u, _, vt = mx.linalg.svd(m)
omega = u @ vt

# Sign Correction (Ensure rotation, not reflection)
if mx.linalg.det(omega) < 0:
    # ... flip sign of last column ...
```

### Key Components

1.  **`ProbeCorpus`**: Standardized prompt corpus used to elicit comparable activations (not the semantic prime inventory itself).
2.  **Semantic prime anchors**: Canonical inventory lives in `src/modelcypher/data/semantic_primes.json` and is surfaced via `mc geometry primes …`.
3.  **`ContinuousFingerprint`**: A stable signature of a model's activation geometry, preserving magnitude and entropy.
4.  **`IntersectionMap`**: The calculated correspondence (Venn diagram) between two fingerprints.

## Verification

We evaluate stitching with **representation similarity** (e.g., CKA/cosine on stitched activations) and, when available, downstream task checks.

- Similarity thresholds are heuristic and depend on architecture, probe corpus, and layer.
- Passing similarity checks is necessary but not sufficient for “safe to merge”.
