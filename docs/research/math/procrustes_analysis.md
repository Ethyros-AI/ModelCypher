# Procrustes Analysis `[PROVEN]`

> Optimal orthogonal alignment of representation spaces.
> *(Schonemann 1966; Gower 1975)*

---

## Why This Matters for Model Merging

Before comparing or merging representations from different models, we need to align them. Procrustes analysis finds the optimal orthogonal transformation (rotation/reflection) that minimizes the difference between two sets of representations.

**In ModelCypher**: `generalized_procrustes.py` handles multi-model alignment. Pairwise alignment uses `backend_matrix_utils.py` (core rotation). Cross-dimensional cases go through `_project_procrustes` in `cross_dimensional_projection.py`, and vocabulary alignment uses `vocabulary/embedding_projector.py`.

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

### Algorithm (ModelCypher)

```
1. Initialize consensus X̄ = X₁
2. Repeat until convergence:
   a. For each i: compute Ωᵢ by solving Procrustes(Xᵢ, X̄) with det(Ωᵢ)=1
   b. Update consensus:
      - M > 2: Fréchet mean of {Xᵢ Ωᵢ}
      - M ≤ 2: arithmetic mean (k-NN graph degenerates)
3. Return {Ωᵢ} and X̄
```

### Consensus on Curved Manifolds

ModelCypher uses the Fréchet mean whenever $M > 2$:

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

## Cross-Dimensional Procrustes (ModelCypher)

When dimensions differ, ModelCypher uses `_project_procrustes` with geodesic SVD
and zero-padding to preserve geometry:

- **Rows match, columns differ**:
  - If $d_s > d_t$: project via geodesic SVD to the top-$k$ subspace, align to target,
    and pad with zeros if rank < $d_t$.
  - If $d_s < d_t$: align shared dimensions and zero-pad expansion.
- **Columns match, rows differ**: transpose, apply the same logic, transpose back.
- **Both rows and columns differ**: fall back to Gram-transport projection.

Reflections are corrected so $\det(\Omega)=1$, and zero padding is used to avoid
introducing spurious correlations.

---

## Connections to Other Measures

### Theorem (Harvey et al., 2024)

Procrustes distance is related to CKA:

$$d_{Proc}^2(X, Y) = \|X\|_F^2 + \|Y\|_F^2 - 2\|X\|_F \|Y\|_F \sqrt{\text{CKA}(X, Y)}$$

for centered matrices with specific normalization.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/generalized_procrustes.py`](../../../src/modelcypher/core/domain/geometry/generalized_procrustes.py)

**Key entry points**:
- `GeneralizedProcrustes.align()` - multi-model alignment with Fréchet consensus
- `BackendMatrixUtils.procrustes_rotation()` / `procrustes_align()` - pairwise alignment
- `cross_dimensional_projection._project_procrustes()` - one-dimension mismatch
- `EmbeddingProjector.project()` - vocabulary alignment (in `vocabulary/`)

**Design decisions**:
1. **Fréchet mean**: Used for $M > 2$; arithmetic mean only when $M \le 2$.
2. **Shape requirements**: GPA requires matching shapes; cross-dimensional projection handles mismatches.
3. **Reflections and scaling**: Disabled; rotations are enforced with $\det(\Omega)=1$.
4. **Convergence**: Threshold from machine epsilon; max iterations from model count.

---

## Applications in Model Merging

### 1. Layer Alignment (Null-Space Addition)

Before merging, align representations and add only the null-space component:
```python
utils = BackendMatrixUtils(backend)
omega = utils.procrustes_rotation(source_layers, target_layers).rotation
aligned_source = backend.matmul(source_layers, omega)
delta = aligned_source - target_layers
filter = GeodesicNullSpaceFilter(backend)
projected = filter.filter_delta(delta, target_activations).filtered_delta
merged = target_layers + projected
```

### 2. Cross-Model Comparison

Compare representations after alignment:
```python
aligned_source = backend.matmul(source, omega)
similarity = cka(aligned_source, target)
```

### 3. Multi-Model Consensus

Find shared representation space:
```python
result = GeneralizedProcrustes().align(model_representations)
consensus = result.consensus
transformations = result.rotations
```

---

## Citations

### Foundational

1. **Schönemann, P.H.** (1966). "A generalized solution of the orthogonal Procrustes problem." *Psychometrika*, 31(1), 1-10. [DOI:10.1007/BF02289451](https://doi.org/10.1007/BF02289451)
   - *Original closed-form solution*

2. **Gower, J.C.** (1975). "Generalized Procrustes Analysis." *Psychometrika*, 40(1), 33-51. [DOI:10.1007/BF02291478](https://doi.org/10.1007/BF02291478)
   - *Multi-matrix extension*

3. **Goodall, C.** (1991). "Procrustes methods in the statistical analysis of shape." *Journal of the Royal Statistical Society B*, 53(2), 285-339. [JSTOR](https://www.jstor.org/stable/2345744)
   - *Comprehensive treatment*

### Neural Network Applications

4. **[Hamilton et al. (2016)](../../references/arxiv/Hamilton_2016_Diachronic_Word_Embeddings_Reveal_Statistical_Laws.pdf)**. "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." *ACL 2016*. [arXiv:1605.09096](https://arxiv.org/abs/1605.09096)
   - *Procrustes for word embedding alignment*

5. **[Ding et al. (2021)](../../references/arxiv/Ding_2021_Grounding_Representation_Similarity_Statistical_Testing.pdf)**. "Grounding Representation Similarity with Statistical Testing." *NeurIPS 2021*. [arXiv:2108.01661](https://arxiv.org/abs/2108.01661)
   - *Statistical framework for Procrustes*

### 2024-2025 Advances

6. **Zielnicki, A., & Hsiao, D.** (2025). "When Embedding Models Meet: Procrustes Bounds and Alignment." [arXiv:2510.13406](https://arxiv.org/abs/2510.13406)
   - *Procrustes for embedding model versions*

7. **Harvey, W., et al.** (2024). "Duality of Bures and Shape Distances with Implications for Representation Similarity." *CCN 2024*. [CCN](https://2024.ccneuro.org/)
   - *Connects Procrustes to other similarity measures*

8. **Chen, Y., et al.** (2025). "ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Procrustes." *ACL Findings 2025*. [ACL Anthology](https://aclanthology.org/)
   - *Procrustes for LLM compression*

9. **[Klabunde et al. (2023)](../../references/arxiv/Klabunde_2023_Similarity_Neural_Network_Models_Survey_Functional.pdf)**. "Similarity of Neural Networks: A Survey of Functional and Representational Measures." [arXiv:2305.06329](https://arxiv.org/abs/2305.06329)
   - *Comprehensive comparison including Procrustes*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA after Procrustes alignment
- [gromov_wasserstein.md](gromov_wasserstein.md) - Alternative for cross-dimensional
- [relative_representations.md](relative_representations.md) - Alignment-free alternative

---

*Procrustes finds the rotation that minimizes Frobenius alignment error between two spaces. It's the foundation for representation comparison in same-dimensional settings.*
