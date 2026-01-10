# Manifold Curvature Estimation

> Measuring how neural network representation spaces bend.

---

## Why This Matters for Model Merging

Curvature determines whether Euclidean intuitions hold:
- **Flat**: Euclidean operations are locally valid
- **Positive curvature** (sphere-like): geodesics converge, Euclidean underestimates distance
- **Negative curvature** (saddle-like): geodesics diverge, Euclidean overestimates distance

**In ModelCypher**: Curvature is used for diagnostics and curvature-aware
operations (e.g., covariance correction in `riemannian_density.py`).

---

## Types of Curvature (Background)

### Sectional Curvature

For a 2D plane $\sigma$ in the tangent space $T_pM$:

$$K(\sigma) = \frac{R(u, v, v, u)}{g(u, u)g(v, v) - g(u, v)^2}$$

### Ricci Curvature

Average of sectional curvatures through a direction $v$:

$$\text{Ric}(v, v) = \sum_{i=1}^{n-1} K(\sigma_i)$$

### Scalar Curvature

Trace of Ricci curvature:

$$S = \sum_{i,j} g^{ij} \text{Ric}_{ij}$$

---

## Discrete Curvature for k-NN Graphs

### Ollivier-Ricci Curvature

For an edge $(x, y)$ on a graph:

$$\kappa(x, y) = 1 - \frac{W_1(m_x, m_y)}{d(x, y)}$$

where $m_x, m_y$ are neighborhood measures and $W_1$ is Wasserstein-1.

---

## ModelCypher Implementation

### Sectional Curvature (Local)

`SectionalCurvatureEstimator` estimates local curvature using geodesic deviation:
- Samples random orthogonal directions
- Estimates local metric from covariance (or optional metric_fn)
- Computes sectional, scalar, and principal curvatures

### Ollivier-Ricci Curvature (Graph)

`OllivierRicciCurvature` computes edge and node curvatures on the k-NN graph using
optimal transport. All parameters are data-derived (no manual tuning).

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/manifold_curvature.py`

**Key types**:
- `CurvatureSign`
- `LocalCurvature`
- `ManifoldCurvatureProfile`
- `SectionalCurvatureEstimator`
- `OllivierRicciCurvature`

**Related usage**:
- `src/modelcypher/core/domain/geometry/riemannian_density.py`
- `src/modelcypher/core/domain/geometry/riemannian_utils.py`

---

## Citations

1. **do Carmo, M.P.** (1992). *Riemannian Geometry*. Birkhauser. [DOI:10.1007/978-1-4757-2201-7](https://doi.org/10.1007/978-1-4757-2201-7)
2. **Ollivier, Y.** (2009). "Ricci curvature of Markov chains on metric spaces." *JFA*, 256(3), 810-864. [DOI:10.1016/j.jfa.2008.11.001](https://doi.org/10.1016/j.jfa.2008.11.001)
3. **Pennec, X.** (2006). "Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements." *J. Math. Imaging Vis.*, 25, 127-154. [DOI:10.1007/s10851-006-6228-4](https://doi.org/10.1007/s10851-006-6228-4)

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) - Curvature affects geodesic computation
- [frechet_mean.md](frechet_mean.md) - Mean computation depends on curvature
- [intrinsic_dimension.md](intrinsic_dimension.md) - Curvature affects ID estimation

---

*Curvature tells us when Euclidean intuitions fail. Use it as a diagnostic, not a heuristic.*
