# Gromov-Wasserstein Optimal Transport

> Transport between metric spaces with different dimensions.

---

## Why This Matters for Model Merging

Standard Wasserstein distance requires spaces of the same dimension. When merging models with different hidden dimensions, we need **Gromov-Wasserstein (GW)** distance, which compares the *structure* of metric spaces rather than point positions directly.

**In ModelCypher**: Used in `gromov_wasserstein.py` and `cross_dimensional_projection.py` for projecting weights between models with different vocabulary sizes or hidden dimensions.

---

## Formal Definition

### Definition (Mémoli, 2011)

Let $(X, d_X, \mu)$ and $(Y, d_Y, \nu)$ be two metric measure spaces. The **Gromov-Wasserstein distance** is:

$$\text{GW}_p(X, Y) = \left( \inf_{\pi \in \Pi(\mu, \nu)} \int_{X \times Y} \int_{X \times Y} |d_X(x, x') - d_Y(y, y')|^p \, d\pi(x,y) \, d\pi(x',y') \right)^{1/p}$$

where $\Pi(\mu, \nu)$ is the set of couplings (joint distributions with marginals $\mu$ and $\nu$).

### Discrete Formulation

For finite point sets with distance matrices $C^{(1)} \in \mathbb{R}^{n \times n}$ and $C^{(2)} \in \mathbb{R}^{m \times m}$:

$$\text{GW}(C^{(1)}, C^{(2)}) = \min_{\pi \in \Pi(\mathbf{p}, \mathbf{q})} \sum_{i,j,k,l} L(C^{(1)}_{ij}, C^{(2)}_{kl}) \cdot \pi_{ik} \cdot \pi_{jl}$$

where:
- $L$ is a loss function (typically squared error: $L(a,b) = (a-b)^2$)
- $\mathbf{p}, \mathbf{q}$ are probability vectors on the point sets
- $\pi_{ik}$ is the transport mass from point $i$ to point $k$

---

## The Key Insight: Structure Matching

GW compares **pairwise distances**, not point positions:

```
Space X (n points, dim p₁):     Space Y (m points, dim p₂):
    d_X(x_i, x_j)         ↔         d_Y(y_k, y_l)

The coupling π finds correspondences such that:
    "if x_i maps to y_k and x_j maps to y_l,
     then d_X(x_i, x_j) ≈ d_Y(y_k, y_l)"
```

This enables comparison across:
- Different dimensions (768-dim vs 4096-dim)
- Different sample sizes
- Different metric spaces entirely

---

## Fused Gromov-Wasserstein (FGW)

When we have both structure AND features, use FGW (Vayer et al., 2019):

$$\text{FGW}_\alpha = \min_{\pi} (1-\alpha) \sum_{i,k} \pi_{ik} \cdot d(a_i, b_k)^q + \alpha \sum_{i,j,k,l} |C^{(1)}_{ij} - C^{(2)}_{kl}|^q \cdot \pi_{ik} \cdot \pi_{jl}$$

where:
- $\alpha = 0$: pure Wasserstein (feature-based)
- $\alpha = 1$: pure Gromov-Wasserstein (structure-based)
- $\alpha \in (0,1)$: interpolation

---

## Entropic Regularization

### Sinkhorn-GW Algorithm

The unregularized GW problem is non-convex. Entropic regularization makes it tractable:

$$\text{GW}_\epsilon = \min_{\pi \in \Pi(\mathbf{p}, \mathbf{q})} \sum_{i,j,k,l} L(C^{(1)}_{ij}, C^{(2)}_{kl}) \cdot \pi_{ik} \cdot \pi_{jl} - \epsilon H(\pi)$$

where $H(\pi) = -\sum_{ik} \pi_{ik} \log \pi_{ik}$ is the entropy.

### Algorithm: Sinkhorn-GW

```
Input: C¹, C², p, q, ε, max_iter
Output: Transport plan π

1. Initialize π ← pq^T (outer product)
2. Repeat until convergence:
   a. Compute gradient: G_ij = 2 Σ_{kl} (C¹_ij - C²_kl)² π_kl
   b. K ← exp(-G/ε)
   c. Sinkhorn iterations on K to get π with marginals p, q
3. Return π
```

---

## Theoretical Properties

### Theorem 1: Metric Property (Mémoli, 2011)

$\text{GW}_p$ is a metric on the space of metric measure spaces (up to isometry).

### Theorem 2: Invariance

GW is invariant to:
- Isometries of either space
- Rescaling (when using normalized distance matrices)

### Theorem 3: Computational Complexity

The discrete GW problem is NP-hard in general, but the entropic-regularized version admits polynomial-time algorithms.

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/gromov_wasserstein.py`

```python
def gromov_wasserstein_transport(
    source_gram: Array,
    target_gram: Array,
    config: GWConfig,
    backend: Backend,
) -> GWResult:
    """
    Compute Gromov-Wasserstein transport plan.

    Uses geodesic distance matrices (not Euclidean) for
    accurate manifold-aware transport.
    """
```

**Design decisions**:
1. **Geodesic Gram matrices**: Input distance matrices use geodesic, not Euclidean
2. **Multi-initialization**: Random restarts to escape local minima
3. **Entropic regularization**: Controllable smoothness via ε parameter

---

## Application: Cross-Dimensional Projection

The transport plan $\pi$ gives us a soft correspondence between source and target points. For projecting a weight matrix $W \in \mathbb{R}^{n \times p_1}$ to $\mathbb{R}^{m \times p_2}$:

$$W_{projected} = D_\pi^{-1} \cdot \pi^T \cdot W$$

where $D_\pi = \text{diag}(\pi^T \mathbf{1})$ normalizes the transport.

---

## Citations

### Foundational

1. **Mémoli, F.** (2011). "Gromov-Wasserstein Distances and the Metric Approach to Object Matching." *Foundations of Computational Mathematics*, 11(4), 417-487.
   DOI: 10.1007/s10208-011-9093-5
   - *Original formulation of GW distance*

2. **Peyré, G., Cuturi, M., & Solomon, J.** (2016). "Gromov-Wasserstein Averaging of Kernel and Distance Matrices." *ICML 2016*.
   - *Computational framework for GW*

3. **Vayer, T., Chapel, L., Flamary, R., Tavenard, R., & Courty, N.** (2019). "Fused Gromov-Wasserstein Distance for Structured Objects." *ICML 2019*.
   arXiv: 1811.02834
   - *FGW combining structure and features*

### Word Embeddings and NLP

4. **Alvarez-Melis, D., & Jaakkola, T.S.** (2018). "Gromov-Wasserstein Alignment of Word Embedding Spaces." *EMNLP 2018*.
   arXiv: 1809.00013
   - *GW for cross-lingual embedding alignment*

### Neural Network Applications

5. **Alvarez-Melis, D., & Fusi, N.** (2020). "Geometric Dataset Distances via Optimal Transport." *NeurIPS 2020*.
   arXiv: 2002.02923
   - *GW for comparing datasets*

6. **Sato, R., et al.** (2025). "Unsupervised alignment in neuroscience: Introducing a toolbox for Gromov-Wasserstein optimal transport." *Journal of Neuroscience Methods*, 419, 110443.
   DOI: 10.1016/j.jneumeth.2025.110443
   - *GWOT for neural representation alignment*

### Computational Advances (2024-2025)

7. **Carrasco, X.A., et al.** (2023). "Neural Gromov-Wasserstein Optimal Transport." *arXiv*.
   arXiv: 2303.05978
   - *Scalable neural method for GW*

8. **Gromov-Wasserstein Tutorial** (2024). "Recent Advances in Optimal Transport for Machine Learning." *arXiv*.
   arXiv: 2306.16156
   - *Comprehensive survey including GW*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA for similarity measurement
- [procrustes_analysis.md](procrustes_analysis.md) - Orthogonal alignment (same-dimension)
- [geodesic_distance.md](geodesic_distance.md) - Distance matrices for GW input

---

*CKA measures similarity. GW computes the optimal transport plan. Together they enable cross-dimensional model merging.*
