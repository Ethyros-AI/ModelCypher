# Procrustes Analysis

> Optimal orthogonal alignment of representation spaces.

---

## Why This Matters for Model Merging

Before comparing or merging representations from different models, we need to align them. Procrustes analysis finds the optimal orthogonal transformation (rotation/reflection) that minimizes the difference between two sets of representations.

**In ModelCypher**: Used in `generalized_procrustes.py` for multi-model alignment and `procrustes_alignment.py` for pairwise alignment.

---

## The Classical Problem

### Orthogonal Procrustes Problem

Given two matrices $X, Y \in \mathbb{R}^{n \times d}$ (same samples, same features), find:

$$\Omega^* = \arg\min_{\Omega \in O(d)} \|X\Omega - Y\|_F$$

where $O(d)$ is the orthogonal group (rotations and reflections).

### Closed-Form Solution

Via SVD of the cross-covariance matrix:

$$X^T Y = U \Sigma V^T$$

The optimal orthogonal transformation is:

$$\Omega^* = UV^T$$

### Procrustes Distance

The minimum achievable distance:

$$d_{Proc}(X, Y) = \|X\Omega^* - Y\|_F = \sqrt{\|X\|_F^2 + \|Y\|_F^2 - 2\|X^TY\|_*}$$

where $\|\cdot\|_*$ is the nuclear norm (sum of singular values).

---

## Generalized Procrustes Analysis (GPA)

### The Problem

Align $M$ matrices $\{X_1, \ldots, X_M\}$ to a common consensus:

$$\min_{\Omega_1, \ldots, \Omega_M, \bar{X}} \sum_{i=1}^{M} \|X_i \Omega_i - \bar{X}\|_F^2$$

### Algorithm

```
1. Initialize consensus X̄ = X₁ (or average)
2. Repeat until convergence:
   a. For each i: compute Ωᵢ by solving Procrustes(Xᵢ, X̄)
   b. Update consensus: X̄ = (1/M) Σᵢ Xᵢ Ωᵢ
3. Return {Ωᵢ} and X̄
```

### Fréchet Mean Extension

For curved manifolds, replace arithmetic mean with Fréchet mean:

$$\bar{X} = \text{FrechetMean}(X_1 \Omega_1, \ldots, X_M \Omega_M)$$

---

## Procrustes vs Other Alignments

| Method | Transformation | Preserves |
|--------|---------------|-----------|
| **Procrustes** | Orthogonal | Distances, angles |
| **Affine** | Linear | Ratios, parallelism |
| **CCA** | Linear projections | Correlations |
| **GW Transport** | Soft correspondence | Relational structure |

### When to Use Procrustes

- Same dimensionality
- Want to preserve geometric structure
- Need interpretable transformation
- Fast, closed-form solution

---

## Partial Procrustes

### Problem

When matrices have different sizes, use partial alignment.

**Same features, different samples**: No alignment needed (Gram matrices handle this)

**Same samples, different features**: Truncate or pad, then align.

### Solution for Different Feature Dimensions

```python
# Align to smaller dimension
d_min = min(d1, d2)
X_trunc = X[:, :d_min]
Y_trunc = Y[:, :d_min]
Omega = procrustes(X_trunc, Y_trunc)
```

---

## Connections to Other Measures

### Theorem (Harvey et al., 2024)

Procrustes distance is related to CKA:

$$d_{Proc}^2(X, Y) = \|X\|_F^2 + \|Y\|_F^2 - 2\|X\|_F \|Y\|_F \sqrt{\text{CKA}(X, Y)}$$

for centered matrices with specific normalization.

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/generalized_procrustes.py`

```python
class GeneralizedProcrustes:
    """Generalized Procrustes Analysis with Fréchet mean support."""

    def align(
        self,
        matrices: list[Array],
        config: GPAConfig,
    ) -> GPAResult:
        """
        Align multiple matrices to consensus.

        Uses Fréchet mean for consensus when config.frechet_mean.enabled=True.
        """
```

**Design decisions**:
1. **Fréchet mean option**: Curvature-aware consensus computation
2. **Dimension handling**: Auto-truncates to shared dimension
3. **Convergence tracking**: Reports alignment error and iterations

---

## Applications in Model Merging

### 1. Layer Alignment

Before merging, align layer representations:
```python
aligned_layers = procrustes_align(source_layers, target_layers)
merged = 0.5 * aligned_layers + 0.5 * target_layers
```

### 2. Cross-Model Comparison

Compare representations after alignment:
```python
aligned_source = source @ omega
similarity = cka(aligned_source, target)
```

### 3. Multi-Model Consensus

Find shared representation space:
```python
consensus, transformations = gpa(model_representations)
```

---

## Citations

### Foundational

1. **Schönemann, P.H.** (1966). "A generalized solution of the orthogonal Procrustes problem." *Psychometrika*, 31(1), 1-10.
   DOI: 10.1007/BF02289451
   - *Original closed-form solution*

2. **Gower, J.C.** (1975). "Generalized Procrustes Analysis." *Psychometrika*, 40(1), 33-51.
   DOI: 10.1007/BF02291478
   - *Multi-matrix extension*

3. **Goodall, C.** (1991). "Procrustes methods in the statistical analysis of shape." *Journal of the Royal Statistical Society B*, 53(2), 285-339.
   - *Comprehensive treatment*

### Neural Network Applications

4. **Hamilton, W.L., Leskovec, J., & Jurafsky, D.** (2016). "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." *ACL 2016*.
   arXiv: 1605.09096
   - *Procrustes for word embedding alignment*

5. **Ding, F., et al.** (2021). "Grounding Representation Similarity with Statistical Testing." *NeurIPS 2021*.
   arXiv: 2108.01661
   - *Statistical framework for Procrustes*

### 2024-2025 Advances

6. **Zielnicki, A., & Hsiao, D.** (2025). "When Embedding Models Meet: Procrustes Bounds and Alignment." *arXiv*.
   arXiv: 2510.13406
   - *Procrustes for embedding model versions*

7. **Harvey, W., et al.** (2024). "Duality of Bures and Shape Distances with Implications for Representation Similarity." *CCN 2024*.
   - *Connects Procrustes to other similarity measures*

8. **Chen, Y., et al.** (2025). "ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Procrustes." *ACL Findings 2025*.
   - *Procrustes for LLM compression*

9. **Similarity Survey** (2023). "Similarity of Neural Networks: A Survey of Functional and Representational Measures." *arXiv*.
   arXiv: 2305.06329
   - *Comprehensive comparison including Procrustes*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA after Procrustes alignment
- [gromov_wasserstein.md](gromov_wasserstein.md) - Alternative for cross-dimensional
- [relative_representations.md](relative_representations.md) - Alignment-free alternative

---

*Procrustes finds the rotation that best aligns two spaces. It's the foundation for representation comparison in same-dimensional settings.*
