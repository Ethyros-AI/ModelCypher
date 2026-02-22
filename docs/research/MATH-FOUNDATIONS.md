# Mathematical Foundations Reference `[PROVEN]`

> Consolidated reference for geometric and high-dimensional concepts in ModelCypher.
> Standard mathematical definitions and theorems with citations.

---

## Overview

ModelCypher is built on a fundamental observation:

> **Neural network representations live on curved manifolds. Euclidean geometry is the approximation; geodesic geometry is the reality.**

This has profound implications:
- **Averaging**: Arithmetic mean is wrong on curved manifolds → use Fréchet mean
- **Distance**: Euclidean distance is wrong in high dimensions → use geodesic via k-NN
- **Similarity**: Pointwise comparison fails across dimensions → use CKA on Gram matrices
- **Transfer**: Linear projection loses geometry → use Gromov-Wasserstein transport

---

## Part 1: Core Geometry

### Geodesic Distance

The true distance on discrete manifolds via k-NN graphs.

**Key insight**: Neural network activations form a discrete manifold. The k-NN graph approximates the continuous manifold, and shortest paths on the graph ARE geodesics on the discrete manifold.

**Definition**: For points X with k-NN graph, geodesic distance is the shortest path:
$$d_{geo}(x_i, x_j) = \min_{\text{paths } p: i \to j} \sum_{(a,b) \in p} w_{ab}$$

**Geodesic defect** (curvature measure):
$$\delta(x, y) = \frac{d_{geo}(x, y)}{d_{euc}(x, y)} - 1$$

**Implementation**: `riemannian_utils.py` - `RiemannianGeometry.geodesic_distances()`

---

### Fréchet Mean (Karcher Mean)

The Riemannian generalization of arithmetic mean for curved spaces.

**Definition** (Fréchet 1948):
$$\bar{x} = \arg\min_{y \in M} \sum_{i=1}^{n} w_i \cdot d(x_i, y)^2$$

**Algorithm**: Gradient descent on manifold with geodesic medoid initialization.

**Log map approximation** on discrete manifolds:
$$\log_\mu(x) = (x - \mu) \cdot \frac{d_{geo}(\mu, x)}{d_{euc}(\mu, x)}$$

**Implementation**: `riemannian_utils.py` - `RiemannianGeometry.frechet_mean()`

---

### Intrinsic Dimension

Measuring the true dimensionality of neural network representations.

**Definition**: The minimum number of coordinates needed to represent data without significant information loss. If data lies on a d-dimensional manifold M ⊂ R^D, then ID(X) = d.

**TwoNN Estimator** (Facco et al. 2017):
$$\hat{d} = \frac{n}{\sum_{i=1}^{n} \log \mu_i}$$
where μᵢ = r₂(xᵢ)/r₁(xᵢ) is the ratio of second to first nearest neighbor distance.

**Empirical findings**: ID increases through early layers, peaks at intermediate layers, decreases toward output.

**Implementation**: `intrinsic_dimension.py` - TwoNN regression with geodesic distances

---

### Manifold Curvature

How neural network representation spaces bend.

**Types**:
- **Sectional curvature**: Curvature of 2D plane in tangent space
- **Ricci curvature**: Average of sectional curvatures
- **Ollivier-Ricci** (discrete): κ(x,y) = 1 - W₁(mₓ, mᵧ)/d(x,y)

**Interpretation**:
- Positive curvature (sphere-like): geodesics converge
- Negative curvature (saddle-like): geodesics diverge
- Flat: Euclidean intuitions hold locally

**Implementation**: `manifold_curvature.py` - `SectionalCurvatureEstimator`, `OllivierRicciCurvature`

---

### Tangent Space

Local linearization for Riemannian operations.

**Use in ModelCypher**: Tangent space alignment measures local geometric agreement around shared anchors, providing scale-free signal before null-space addition.

**Algorithm**:
1. Build k-NN graph using geodesic distances
2. Compute local covariance from neighbor deltas
3. Extract tangent basis (top eigenvectors)
4. Compute principal angles between source/target bases

**Implementation**: `tangent_space_alignment.py`

---

## Part 2: Representation Similarity

### CKA (Centered Kernel Alignment)

Cross-dimensional representation similarity via Gram matrices.

**Definition** (Kornblith et al. 2019):
$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where K = XX^T, L = YY^T are Gram matrices.

**Linear CKA** (simplified):
$$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}$$

**Key insight**: Gram matrices are the same size regardless of feature dimension. They capture pairwise relationships, not individual features.

**Invariances**:
- Orthogonal transformation: CKA(X, Y) = CKA(XQ, Y)
- Isotropic scaling: CKA(X, Y) = CKA(αX, Y)

**Implementation**: `cka.py` - `compute_cka()`, supports bias correction (Murphy 2024, Chun 2025)

---

### HSIC (Hilbert-Schmidt Independence Criterion)

The kernel-based foundation underlying CKA.

**Definition** (Gretton et al. 2005):
$$\text{HSIC}(X, Y) = \|\mathcal{C}_{XY}\|_{HS}^2$$

where C_XY is the cross-covariance operator in RKHS.

**Key property**: HSIC(X, Y) = 0 ⟺ X ⊥ Y (with characteristic kernels)

**Unbiased estimator** (Song et al. 2012) available for bias correction.

**Implementation**: Computed internally in `cka.py`

---

### Procrustes Analysis

Optimal orthogonal alignment of representation spaces.

**Orthogonal Procrustes Problem**: Find Ω* = argmin_{Ω∈O(d)} ||XΩ - Y||_F

**Closed-form solution** via SVD of X^T Y = UΣV^T: Ω* = UV^T

**Procrustes distance**: d_Proc(X, Y) = √(||X||²_F + ||Y||²_F - 2||X^TY||_*)

**Connection to CKA** (Harvey 2024):
$$d_{Proc}^2(X, Y) = \|X\|_F^2 + \|Y\|_F^2 - 2\|X\|_F \|Y\|_F \sqrt{\text{CKA}(X, Y)}$$

**Implementation**: `generalized_procrustes.py`, `backend_matrix_utils.py`

---

### Gromov-Wasserstein Transport

Transport between metric spaces with different dimensions.

**Definition** (Memoli 2011):
$$\text{GW}_p(X, Y) = \left( \inf_{\pi} \int |d_X(x, x') - d_Y(y, y')|^p \, d\pi(x,y) \, d\pi(x',y') \right)^{1/p}$$

**Key insight**: Compares *structure* by matching pairwise distances, not point coordinates.

**ModelCypher use**: Cross-dimensional alignment via Gram matrices with Frank-Wolfe solver.

**Implementation**: `gromov_wasserstein.py`, `cross_dimensional_projection.py`

---

### Relative Representations

Dimension-agnostic transfer via anchor similarities (Moschella et al. 2023).

**Core insight**: Instead of absolute coordinates x ∈ R^d, use relative coordinates:
$$r(x) = [s(x, a_1), s(x, a_2), \ldots, s(x, a_k)]$$

where s is similarity (typically cosine) and {aᵢ} are anchor points.

**Properties**:
- Isometry invariant: φ_A(Qz) = φ_{QA}(Qz)
- Scale invariant: φ_A(αz) = φ_A(z)

**Implementation**: `relative_representation.py` - anchors from atlas probes, Procrustes alignment

---

## Part 3: Weight Analysis

### Spectral Analysis

Raw spectral measurements for source/target weight pairs.

**Metrics computed**:
- Spectral ratio: σ_max(W_s) / σ_max(W_t)
- Condition number: σ_max / σ_min (capped by dtype threshold)
- Delta Frobenius: ||W_s - W_t||_F

**Implementation**: `spectral_analysis.py` - raw measurements only, no thresholds

---

### DoRA Decomposition

Separating magnitude and direction for fine-tuning analysis.

**Decomposition**: W = ||W|| · Ŵ (magnitude × unit direction)

**Metrics**:
- Magnitude ratio: ||W_current|| / ||W_base||
- Direction cosine: geodesic cosine similarity
- Directional drift: 1 - direction_cosine

**Implementation**: `dora_decomposition.py` - diagnostics for adapter geometry

---

## Part 4: Merge Methods (Reference)

### SLERP (Spherical Linear Interpolation)

Geodesic interpolation on hypersphere for **diagnostics only**.

**Formula** (Shoemake 1985):
$$\text{SLERP}(v_0, v_1, t) = \frac{\sin((1-t)\theta)}{\sin\theta} v_0 + \frac{\sin(t\theta)}{\sin\theta} v_1$$

**Note**: ModelCypher uses null-space addition for merging, not interpolation.

---

### TIES-Merging / DARE / Task Singular Vectors

**Status**: Not implemented in ModelCypher. These are parameter-space heuristics from external literature. ModelCypher uses geometric alignment and null-space addition.

**DARE sparsity analysis**: `dare_sparsity.py` analyzes delta sparsity with data-derived thresholds.

---

### Fisher Information

Information geometry for curvature diagnostics.

**Definition**: F_θ = E[∇log p(y|x) ∇log p(y|x)^T]

**Note**: Fisher-weighted averaging is prohibited in ModelCypher. Fisher may be used for diagnostics/constraints only.

---

## Part 5: Topological Features

### Persistent Homology

Multi-scale topological fingerprints from activation point clouds.

**Features captured**:
- H0: connected components
- H1: loops/cycles

**Implementation**: `topological_fingerprint.py` - Vietoris-Rips filtration with geodesic distances

**Comparison metrics**: Bottleneck distance, Wasserstein distance, Betti difference

---

## Part 6: Special Topics

### Riemannian Density

Modeling concept activations as probability mass on curved manifold.

**ConceptVolume includes**: centroid, covariance (curvature-aware), geodesic radius, influence type

**Implementation**: `riemannian_density.py`

---

### Prime Spectral Geometry

Eigenvalue analysis of prime sequences for universal alignment anchors.

**Core idea**: Prime structure provides training-invariant anchors. Time-delay embedding + Gram matrix analysis reveals hidden structure.

**Implementation**: `prime_geometry.py`

---

### Permutation Alignment (Git Re-Basin)

Resolving permutation symmetries before comparing weights.

**Note**: ModelCypher emphasizes Gram/CKA alignment rather than permutation alignment.

---

## Key Relationships

```
                REPRESENTATION SPACE
                (High-dimensional, curved)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Geodesic         Intrinsic        Manifold
   Distance         Dimension        Curvature
        │                │                │
        ▼                ▼                ▼
   Tangent          Persistent       Spectral
   Space            Homology         Analysis
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Fréchet          CKA (HSIC)       Gromov-
   Mean             Similarity       Wasserstein
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
            NULL-SPACE ADDITION (MERGE)
```

---

## Implementation Principles

1. **Backend-only math**: Use Backend protocol, no NumPy in core paths
2. **Data-derived thresholds**: From machine epsilon and dtype, not magic numbers
3. **Geodesic distances**: k-NN graph shortest paths, not Euclidean
4. **Raw measurements**: No qualitative labels, only numeric metrics
5. **Caching**: Gram matrices, SVDs, Fréchet means cached per session

---

## Key Citations

### Foundational
- Fréchet (1948) - Metric space means
- Gretton et al. (2005) - HSIC
- Kornblith et al. (2019) - CKA
- Memoli (2011) - Gromov-Wasserstein
- Schönemann (1966) - Procrustes
- Facco et al. (2017) - TwoNN intrinsic dimension

### Modern (2023-2025)
- Moschella et al. (2023) - Relative representations
- Murphy et al. (2024) - CKA bias correction
- Chun et al. (2025) - Feature-sampling correction

---

## Archived Individual Documents

Original detailed reference documents archived to:
`/Volumes/CodeCypher/archive/modelcypher-legacy/docs/math/`

---

*Last consolidated: 2026-01-29*
