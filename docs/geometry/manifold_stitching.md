# Manifold Stitching: Cross-Architecture Model Merging

> **Status**: Implemented & Verified
> **Core Module**: `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

## The Problem: The "Bag of Numbers" Fallacy

Traditional model merging (Task Arithmetic, TIES, SLERP) works well for models initialized from the **same seed** and fine-tuned on different tasks. This is because they share a "Linear Mode Connectivity" (LMC)—their loss landscapes are connected by a linear path, and "Neuron 42" in Model A roughly corresponds to "Neuron 42" in Model B.

However, for **disparate models** (different random seeds, different architectures, or even different sizes), this assumption fails.
-   **Permutation Symmetry**: Networks are invariant to neuron permutations. Model A might have a "cat detector" at index 0, while Model B has it at index 5.
-   **Rotational Misalignment**: Deep networks learn representations that are fundamentally the same up to an orthogonal rotation (Procrustes theorem), but the raw coordinates differ.

Trying to average these weights destroys the information.

## The Solution: Geometric Manifold Stitching

**Manifold Stitching** treats models not as lists of numbers, but as **high-dimensional geometric spaces**. It aligns the "shape" of the knowledge before merging.

### 1. The Intersection Map ("Venn Diagram")
We first determine *where* two models overlap. We don't assume full alignment.
-   **Source Model**: Probed with a standardized probe corpus (see `src/modelcypher/core/domain/geometry/probe_corpus.py`).
-   **Target Model**: Probed with the same corpus.
-   **Alignment**: We compute the **Intersection Map**, identifying which dimensions in Model A correlate with which in Model B.

This creates a "Venn Diagram" of shared knowledge.

### 2. Procrustes Alignment (Rotation)
For the matching intrinsic dimensions, we solve the **Orthogonal Procrustes Problem**:

$$ R^* = \arg\min_R \|A R - B\|_F^2 \quad \text{s.t.} \quad R^T R = I $$

Where $A$ and $B$ are the activation matrices of the source and target models on the shared probe corpus.
The solution is given by SVD:
$$ U, \Sigma, V^T = \text{SVD}(B^T A) $$
$$ R^* = U V^T $$

This yields a **Rotation Matrix** that physically rotates Model A's activation space to align with Model B's.

### 3. Stitching Layer
We insert a learnable (or computed) linear layer between the models.
-   **Input**: Model A's hidden states (rotated).
-   **Output**: Model B's expected input states.
-   **Result**: Knowledge flows seamlessly from the specific representation of Model A into the processing machinery of Model B.

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

We verify stitching using **CKA (Centered Kernel Alignment)** and direct **Cosine Similarity** of the stitched activations.

-   **High CKA (>0.9)**: Successful manifold alignment.
-   **Low CKA**: Attempted stitching of disjoint concepts (no "Venn" overlap).
