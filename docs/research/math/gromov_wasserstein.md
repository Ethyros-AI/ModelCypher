# Gromov-Wasserstein Optimal Transport `[PROVEN]`

> Transport between metric spaces with different dimensions.
> *(Memoli 2011; Peyre, Cuturi & Solomon 2016)*

---

## Why This Matters for Model Merging

Standard Wasserstein distance requires spaces of the same dimension. When
merging models with different hidden sizes, **Gromov-Wasserstein (GW)** compares
*structure* by matching pairwise distances instead of point coordinates.

**In ModelCypher**: Used in `gromov_wasserstein.py` and
`cross_dimensional_projection.py` for cross-dimensional alignment and projection.

---

## Formal Definition

### Definition (Memoli, 2011)

Let $(X, d_X, \mu)$ and $(Y, d_Y, \nu)$ be two metric measure spaces. The
**Gromov-Wasserstein distance** is:

$$\text{GW}_p(X, Y) = \left( \inf_{\pi \in \Pi(\mu, \nu)} \int_{X \times Y} \int_{X \times Y} |d_X(x, x') - d_Y(y, y')|^p \, d\pi(x,y) \, d\pi(x',y') \right)^{1/p}$$

### Discrete Formulation

For distance matrices $C^{(1)} \in \mathbb{R}^{n \times n}$ and
$C^{(2)} \in \mathbb{R}^{m \times m}$:

$$\text{GW}(C^{(1)}, C^{(2)}) = \min_{\pi \in \Pi(\mathbf{p}, \mathbf{q})} \sum_{i,j,k,l} L(C^{(1)}_{ij}, C^{(2)}_{kl}) \cdot \pi_{ik} \cdot \pi_{jl}$$

with $L(a,b) = (a-b)^2$ by default.

---

## ModelCypher Implementation

ModelCypher computes GW using a **Frank-Wolfe** (conditional gradient) solver
with multiple restarts. Each inner step solves a linear OT subproblem using
Sinkhorn iterations, with epsilon derived from the cost scale (no user-tuned
regularization).

Key implementation details:
- **Exact permutation search** for small square matrices ($n \le 8$).
- **Uniform marginals** by default.
- **Random restarts** to escape local minima (GW is non-convex).
- **Data-derived epsilon** for Sinkhorn stability.

---

## Application: Cross-Dimensional Projection

`cross_dimensional_projection.project_cross_dimensional` applies GW to Gram
matrices, with a column-first strategy for tractability:

- Column Gram: $G_{col} = W^T W$ (hidden-dimension sized, tractable)
- Column coupling $\pi_{col}$ projects columns: $W' = W \pi_{col}$
- Row coupling $\pi_{row}$ is applied only when row dimensions are tractable:
  $W'' = \pi_{row}^T W$

This preserves relational structure while respecting dimension mismatches.

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/gromov_wasserstein.py`

**Key classes**:
- `GromovWassersteinDistance`
- `Result`

**Related usage**:
- `src/modelcypher/core/domain/geometry/cross_dimensional_projection.py`
- `src/modelcypher/core/use_cases/geometry_metrics_service.py`
- `src/modelcypher/core/domain/geometry/geometry_validation_suite.py`

---

## Citations

1. **Memoli, F.** (2011). "Gromov-Wasserstein Distances and the Metric Approach to Object Matching." *Foundations of Computational Mathematics*, 11(4), 417-487. [DOI:10.1007/s10208-011-9093-5](https://doi.org/10.1007/s10208-011-9093-5)
2. **Peyre, G., Cuturi, M., & Solomon, J.** (2016). "Gromov-Wasserstein Averaging of Kernel and Distance Matrices." *ICML 2016*. [Paper](https://proceedings.mlr.press/v48/peyre16.html)
3. **[Vayer et al. (2019)](../../references/arxiv/Vayer_2018_Fused_GromovWasserstein_distance_structured_objects_theoretical.pdf)**. "Fused Gromov-Wasserstein Distance for Structured Objects." *ICML 2019*. [arXiv:1811.02834](https://arxiv.org/abs/1811.02834)
4. **[Alvarez-Melis & Jaakkola (2018)](../../references/arxiv/AlvarezMelis_2018_GromovWasserstein_Alignment_Word_Embedding_Spaces.pdf)**. "Gromov-Wasserstein Alignment of Word Embedding Spaces." *EMNLP 2018*. [arXiv:1809.00013](https://arxiv.org/abs/1809.00013)
5. **[Peyre & Cuturi (2019)](../../references/arxiv/Peyre_2018_Computational_Optimal_Transport.pdf)**. "Computational Optimal Transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA for similarity measurement
- [procrustes_analysis.md](procrustes_analysis.md) - Orthogonal alignment (same-dimension)
- [geodesic_distance.md](geodesic_distance.md) - Distance matrices for GW input

---

*CKA measures similarity. GW computes a transport plan that aligns structures across dimensions.*
