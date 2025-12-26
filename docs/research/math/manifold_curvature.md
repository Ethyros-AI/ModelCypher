# Manifold Curvature Estimation

> Measuring how neural network representation spaces bend.

---

## Why This Matters for Model Merging

Curvature determines whether Euclidean intuitions hold:
- **Flat**: Euclidean operations work
- **Positive curvature** (sphere-like): Geodesics converge, Euclidean underestimates distance
- **Negative curvature** (saddle-like): Geodesics diverge, Euclidean overestimates distance

**In ModelCypher**: Used in `manifold_curvature.py` for curvature-aware merge strategies and geodesic computation.

---

## Types of Curvature

### 1. Sectional Curvature

For a 2-dimensional plane $\sigma$ in the tangent space $T_pM$:

$$K(\sigma) = \frac{R(u, v, v, u)}{g(u, u)g(v, v) - g(u, v)^2}$$

where $R$ is the Riemann curvature tensor and $g$ is the metric.

**Interpretation**:
- $K > 0$: Sphere-like (geodesics converge)
- $K = 0$: Flat (Euclidean)
- $K < 0$: Saddle-like (geodesics diverge)

### 2. Ricci Curvature

Average of sectional curvatures through a direction $v$:

$$\text{Ric}(v, v) = \sum_{i=1}^{n-1} K(\sigma_i)$$

where $\{\sigma_i\}$ are planes containing $v$.

### 3. Scalar Curvature

Trace of Ricci curvature (single number at each point):

$$S = \sum_{i,j} g^{ij} \text{Ric}_{ij}$$

---

## Discrete Curvature for k-NN Graphs

### Ollivier-Ricci Curvature

For edges in a graph, Ollivier (2009) defined:

$$\kappa(x, y) = 1 - \frac{W_1(m_x, m_y)}{d(x, y)}$$

where:
- $m_x, m_y$ are probability measures at $x$ and $y$ (typically uniform on neighbors)
- $W_1$ is the Wasserstein-1 distance
- $d(x,y)$ is the edge length

**Interpretation**:
- $\kappa > 0$: Neighbors of $x$ and $y$ are closer than $x$ and $y$ (positive curvature)
- $\kappa < 0$: Neighbors are farther apart (negative curvature)

### Forman-Ricci Curvature

Combinatorial curvature based on edge-triangle relationships:

$$F(e) = w_e \left( \frac{w_{v_1} + w_{v_2}}{w_e} - \sum_{e' \sim e, e' \neq e} \frac{w_e}{\sqrt{w_e w_{e'}}} \right)$$

where $w$ are weights (typically 1 for unweighted graphs).

---

## Geodesic Defect Method

### Definition

The geodesic defect compares geodesic to Euclidean distance:

$$\delta(x, y) = \frac{d_{geo}(x, y) - d_{euc}(x, y)}{d_{euc}(x, y)}$$

**Curvature estimation** (local average):
$$\hat{K}(x) = -\frac{6}{r^2} \cdot \mathbb{E}_{y \in B_r(x)}[\delta(x, y)]$$

For a sphere of radius $R$:
- $\frac{d_{geo}}{d_{euc}} \approx 1 + \frac{K \cdot r^2}{6}$ for small $r$

---

## Ricci Flow Connection

### Ricci Flow Equation

$$\frac{\partial g}{\partial t} = -2 \text{Ric}$$

The metric evolves to smooth curvature:
- Positive curvature regions shrink
- Negative curvature regions expand

### Neural Networks as Ricci Flow (2024)

Recent work shows deep networks approximately implement discrete Ricci flow:
- Each layer transforms the representation geometry
- The transformation tends to smooth curvature
- This explains feature learning dynamics

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/manifold_curvature.py`](../../../../src/modelcypher/core/domain/geometry/manifold_curvature.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `CurvatureSign` | 52 | Enum for positive/negative/mixed curvature |
| `CurvatureConfig` | 62 | Configuration with method, neighbors, thresholds |
| `LocalCurvature` | 79 | Per-point curvature result |
| `ManifoldCurvatureProfile` | 135 | Full manifold curvature profile |
| `SectionalCurvatureEstimator` | 231 | Main estimator class with Ollivier/Forman/defect methods |

**Also in**:
- [`riemannian_utils.py:142`](../../../../src/modelcypher/core/domain/geometry/riemannian_utils.py) - `CurvatureEstimate` dataclass
- [`loss_landscape_mlx.py:75`](../../../../src/modelcypher/core/domain/training/loss_landscape_mlx.py) - `CurvatureMetrics` for training

**Design decisions**:
1. **Multiple methods**: Ollivier, Forman, geodesic defect
2. **Per-point estimates**: Curvature can vary across manifold
3. **Classification**: Determines curvature regime (positive/negative/mixed)

---

## Curvature Regimes in Neural Networks

### Empirical Observations

From recent literature:

1. **Early layers**: Often near-flat (feature extraction)
2. **Middle layers**: Mixed curvature (representation learning)
3. **Late layers**: Tends toward positive curvature (compression)

### Implications for Merging

| Curvature | Merge Strategy |
|-----------|---------------|
| Flat | Standard averaging works |
| Positive | Use Fréchet mean; Euclidean underestimates |
| Negative | Use Fréchet mean; Euclidean overestimates |
| Mixed | Adaptive strategies; per-region handling |

---

## Citations

### Foundational Differential Geometry

1. **do Carmo, M.P.** (1992). *Riemannian Geometry*. Birkhäuser. [DOI:10.1007/978-1-4757-2201-7](https://doi.org/10.1007/978-1-4757-2201-7)
   - *Comprehensive reference on curvature*

2. **Lee, J.M.** (2018). *Introduction to Riemannian Manifolds*. Springer. [DOI:10.1007/978-3-319-91755-9](https://doi.org/10.1007/978-3-319-91755-9)
   - *Modern treatment of Riemannian geometry*

### Discrete Curvature

3. **Ollivier, Y.** (2009). "Ricci curvature of Markov chains on metric spaces." *Journal of Functional Analysis*, 256(3), 810-864. [DOI:10.1016/j.jfa.2008.11.001](https://doi.org/10.1016/j.jfa.2008.11.001)
   - *Ollivier-Ricci curvature definition*

4. **Forman, R.** (2003). "Bochner's method for cell complexes and combinatorial Ricci curvature." *Discrete and Computational Geometry*, 29(3), 323-374. [DOI:10.1007/s00454-002-0743-x](https://doi.org/10.1007/s00454-002-0743-x)
   - *Forman-Ricci curvature*

### Neural Networks and Curvature

5. **Bronstein, M.M., et al.** (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
   - *Comprehensive geometric deep learning survey*

6. **Weber, M., et al.** (2024). "Neural Feature Geometry Evolves as Discrete Ricci Flow." [arXiv:2509.22362](https://arxiv.org/abs/2509.22362)
   - *Deep learning as Ricci flow*

### 2024-2025 Advances

7. **Torbati, N., et al.** (2024). "Exploring Geometric Representational Alignment through Ollivier-Ricci Curvature and Ricci Flow." *NeurIPS 2024 Workshop*. [OpenReview](https://openreview.net/forum?id=neurips2024workshop)
   - *Curvature for representation alignment*

8. **Farzam, A., et al.** (2024). "On the Ricci Curvature of Attention Maps and Transformers Training and Robustness." *NeurIPS 2024*. [OpenReview](https://openreview.net/forum?id=neurips2024)
   - *Curvature analysis of transformers*

9. **Curvature-based Network Analysis** (2025). "A roadmap for curvature-based geometric data analysis." [arXiv:2510.22599](https://arxiv.org/abs/2510.22599)
   - *Comprehensive survey of discrete curvature methods*

10. **Shi, X., et al.** (2024). "Deep learning as Ricci flow." *Scientific Reports*, 14, 74045. [DOI:10.1038/s41598-024-74045-9](https://doi.org/10.1038/s41598-024-74045-9)
    - *Ricci flow interpretation of deep learning*

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) - Curvature affects geodesic computation
- [frechet_mean.md](frechet_mean.md) - Mean computation depends on curvature
- [intrinsic_dimension.md](intrinsic_dimension.md) - Curvature affects ID estimation

---

*Curvature is the fundamental invariant that determines whether Euclidean intuitions hold. In neural networks, they usually don't.*
