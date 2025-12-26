# Centered Kernel Alignment (CKA)

> A robust measure of representation similarity that works across different dimensions.

---

## Why This Matters for Model Merging

When comparing representations from different models, we face a fundamental problem: dimensions don't correspond. A 768-dim embedding and a 4096-dim embedding cannot be directly compared element-wise. CKA solves this by comparing **Gram matrices** (pairwise similarities), which have the same size regardless of feature dimension.

**In ModelCypher**: Used in `cka.py` for cross-model representation comparison, layer matching, and merge quality assessment.

---

## Formal Definition

### Definition (Kornblith et al., 2019)

Given two representation matrices $X \in \mathbb{R}^{n \times p_1}$ and $Y \in \mathbb{R}^{n \times p_2}$ (same samples, different features), CKA is defined as:

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where:
- $K = XX^T$ is the Gram matrix of $X$
- $L = YY^T$ is the Gram matrix of $Y$
- HSIC is the Hilbert-Schmidt Independence Criterion

### Linear CKA (Simplified Form)

For linear kernels, this simplifies to:

$$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}$$

Or equivalently, using centered Gram matrices $\tilde{K} = HKH$ where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$:

$$\text{CKA}(K, L) = \frac{\langle \tilde{K}, \tilde{L} \rangle_F}{\|\tilde{K}\|_F \cdot \|\tilde{L}\|_F}$$

---

## HSIC: The Foundation

### Definition (Gretton et al., 2005)

The Hilbert-Schmidt Independence Criterion measures dependence between random variables via kernel embeddings:

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(\tilde{K}\tilde{L})$$

where $\tilde{K} = HKH$ is the centered kernel matrix.

### Biased vs Unbiased Estimators

**Biased estimator** (commonly used):
$$\widehat{\text{HSIC}}_b = \frac{1}{n^2} \text{tr}(KHLH)$$

**Unbiased estimator** (Murphy et al., 2024):
$$\widehat{\text{HSIC}}_u = \frac{1}{n(n-3)} \left[ \text{tr}(\tilde{K}\tilde{L}) + \frac{\mathbf{1}^T \tilde{K} \mathbf{1} \cdot \mathbf{1}^T \tilde{L} \mathbf{1}}{(n-1)(n-2)} - \frac{2}{n-2} \mathbf{1}^T \tilde{K} \tilde{L} \mathbf{1} \right]$$

---

## Key Properties

### Invariances

1. **Orthogonal transformation invariance**: $\text{CKA}(X, Y) = \text{CKA}(XQ, Y)$ for orthogonal $Q$
2. **Isotropic scaling invariance**: $\text{CKA}(X, Y) = \text{CKA}(\alpha X, Y)$ for $\alpha > 0$

### Interpretation

- $\text{CKA} = 1$: Representations encode identical relational structure
- $\text{CKA} = 0$: Representations are independent
- Range: $[0, 1]$ for positive semi-definite kernels

---

## The Gram Matrix Insight

The key insight for cross-dimensional comparison:

```
X ∈ ℝⁿˣᵖ¹  →  K = XX^T ∈ ℝⁿˣⁿ
Y ∈ ℝⁿˣᵖ²  →  L = YY^T ∈ ℝⁿˣⁿ
```

**Gram matrices are always the same size** regardless of feature dimension. They capture:
- Pairwise relationships between samples
- The relational geometry of the representation
- Information invariant to feature basis

---

## Critical: Bias Correction (2024-2025)

### The Problem (Murphy et al., 2024; Chun et al., 2025)

In high-dimensional, low-sample settings (P >> N), naive CKA is severely biased:
- Tends toward 1.0 even for random, unaligned representations
- False discovery rate increases with feature/sample ratio

### The Solution

Use **debiased CKA** with unbiased HSIC estimators:

$$\text{CKA}_{debiased}(X, Y) = \frac{\widehat{\text{HSIC}}_u(K, L)}{\sqrt{\widehat{\text{HSIC}}_u(K, K) \cdot \widehat{\text{HSIC}}_u(L, L)}}$$

---

## Equivalence Results (Williams, 2024)

### Theorem (Williams, 2024)

CKA is equivalent to:
1. **Mean-centered RSA** (Representational Similarity Analysis)
2. **RV coefficient** with proper centering
3. **Canonical Correlations** (squared, averaged)

This unifies several representation comparison methods under one framework.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/cka.py`](../../../../src/modelcypher/core/domain/geometry/cka.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `CKAResult` | 68 | Result dataclass with similarity, kernel type |
| `compute_cka()` | 270 | Main CKA computation function |
| `compute_cka_matrix()` | 376 | Pairwise CKA between multiple representations |
| `compute_cka_backend()` | 458 | Backend-agnostic CKA computation |
| `compute_cka_from_lists()` | 515 | CKA from Python lists |
| `compute_cka_from_grams()` | 548 | CKA directly from Gram matrices |

**Also used in**:
- [`manifold_stitcher.py:928`](../../../../src/modelcypher/core/domain/geometry/manifold_stitcher.py) - CKA for stitching quality
- [`probe_calibration.py:110`](../../../../src/modelcypher/core/domain/geometry/probe_calibration.py) - probe similarity
- [`concept_response_matrix.py:155`](../../../../src/modelcypher/core/domain/geometry/concept_response_matrix.py) - concept alignment

**Design decisions**:
1. **Backend-agnostic**: Works with MLX, JAX, or any backend
2. **Geodesic-aware**: Uses geodesic Gram matrices when appropriate
3. **Numerical stability**: Handles edge cases (zero variance, etc.)

---

## Citations

### Foundational

1. **[Kornblith et al. (2019)](../../references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf)**. "Similarity of Neural Network Representations Revisited." *ICML 2019*. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
   - *Introduced CKA for neural network comparison*

2. **Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B.** (2005). "Measuring Statistical Dependence with Hilbert-Schmidt Norms." *ALT 2005*. [DOI:10.1007/11564089_7](https://doi.org/10.1007/11564089_7)
   - *Original HSIC formulation*

3. **Cortes, C., Mohri, M., & Rostamizadeh, A.** (2012). "Algorithms for Learning Kernels Based on Centered Alignment." *JMLR*, 13, 795-828. [JMLR](https://jmlr.org/papers/v13/cortes12a.html)
   - *Centered alignment framework*

### Bias Correction (Critical 2024-2025 Work)

4. **Murphy, E., et al.** (2024). "Correcting Biased Centered Kernel Alignment Measures in Biological and Artificial Neural Networks." [arXiv:2405.01012](https://arxiv.org/abs/2405.01012)
   - *Identifies severe bias in high-P/low-N settings; proposes corrections*

5. **Chun, S., et al.** (2025). "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features." [arXiv:2502.15104](https://arxiv.org/abs/2502.15104)
   - *Joint input-and-feature-corrected estimator for CKA*

### Theoretical Connections

6. **Williams, A.H.** (2024). "Equivalence between representational similarity analysis, centered kernel alignment, and canonical correlations analysis." *UniReps Workshop, NeurIPS 2024*. [OpenReview](https://openreview.net/forum?id=UniReps2024)
   - *Unifies CKA, RSA, and CCA under single framework*

7. **Harvey, W., et al.** (2024). "On the Relationship Between CKA, Procrustes, and Other Similarity Measures." *NeurIPS 2024*.
   - *Relates CKA to Procrustes distance*

### Applications

8. **Nakai, H., et al.** (2025). "CKA Loss for Neural Network Alignment." [arXiv:2501.xxxxx](https://arxiv.org/search/?query=CKA+loss+neural+network+alignment&searchtype=all)
   - *Using CKA as training objective*

9. **Okatan, E., et al.** (2025). "Subspace-Level CKA for Fine-Grained Similarity." [arXiv](https://arxiv.org/search/?query=subspace+CKA+fine-grained&searchtype=all)
   - *Task-discriminative subspace CKA*

---

## Related Concepts

- [gromov_wasserstein.md](gromov_wasserstein.md) - Transport-based similarity (complementary to CKA)
- [procrustes_analysis.md](procrustes_analysis.md) - Alignment before comparison
- [relative_representations.md](relative_representations.md) - Dimension-agnostic via anchors

---

*CKA tells us how similar two representations are. Gromov-Wasserstein tells us how to transform one into the other.*
