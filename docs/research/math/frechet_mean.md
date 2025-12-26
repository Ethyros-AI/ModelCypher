# Fréchet Mean (Karcher Mean)

> The Riemannian generalization of the arithmetic mean for curved spaces.

---

## Why This Matters for Model Merging

When averaging embeddings or representations from neural networks, the arithmetic mean is **geometrically incorrect** on curved manifolds. The Fréchet mean minimizes the sum of squared geodesic distances, giving the true center of mass on the manifold.

**In ModelCypher**: Used in `riemannian_utils.py` for curvature-aware averaging of embeddings, activation patterns, and layer-wise representations during merge operations.

---

## Formal Definition

### Definition (Fréchet, 1948)

Let $(M, d)$ be a metric space and let $\mu$ be a probability measure on $M$. The **Fréchet mean** (or barycenter) is defined as:

$$\bar{x} = \arg\min_{y \in M} \int_M d(x, y)^2 \, d\mu(x)$$

For a finite set of points $\{x_1, \ldots, x_n\}$ with weights $\{w_1, \ldots, w_n\}$:

$$\bar{x} = \arg\min_{y \in M} \sum_{i=1}^{n} w_i \cdot d(x_i, y)^2$$

### Riemannian Formulation (Karcher, 1977)

On a Riemannian manifold $(M, g)$, the Fréchet mean satisfies the **gradient condition**:

$$\sum_{i=1}^{n} w_i \cdot \log_{\bar{x}}(x_i) = 0$$

where $\log_{\bar{x}}$ is the logarithmic map (inverse exponential map) at $\bar{x}$.

---

## The Iterative Algorithm

We implement the gradient descent algorithm on the manifold:

### Algorithm: Fréchet Mean via Gradient Descent

```
Input: Points {x_1, ..., x_n}, weights {w_1, ..., w_n}, tolerance ε
Output: Fréchet mean μ

1. Initialize μ ← weighted Euclidean mean (reasonable starting point)
2. Compute geodesic distances from all points to μ
3. Repeat until convergence:
   a. Compute weighted gradient: g = Σᵢ wᵢ · log_μ(xᵢ)
   b. Update: μ ← exp_μ(η · g)  where η is step size
   c. If ‖g‖ < ε: break
4. Return μ
```

### Log Map Approximation on Discrete Manifolds

For a discrete manifold defined by a k-NN graph:

$$\log_\mu(x) = (x - \mu) \cdot \frac{d_{geo}(\mu, x)}{d_{euc}(\mu, x)}$$

The ratio $\frac{d_{geo}}{d_{euc}}$ is the **curvature correction factor**:
- Ratio > 1: negative curvature (hyperbolic-like)
- Ratio < 1: positive curvature (sphere-like)
- Ratio = 1: flat space

---

## Key Theorems

### Theorem 1: Existence (Sturm, 2003)

On Hadamard spaces (complete CAT(0) spaces), the Fréchet mean exists and is unique for any probability measure with finite second moment.

### Theorem 2: Uniqueness on Riemannian Manifolds (Afsari, 2011)

On a complete Riemannian manifold $M$ with sectional curvature $K$, if all data points lie in a geodesic ball of radius $r < \frac{\pi}{2\sqrt{K_{max}}}$, then the Fréchet mean is unique.

### Theorem 3: Differentiability (Lou et al., 2020)

The Fréchet mean is differentiable with respect to input points, enabling backpropagation through mean computation in neural networks.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/riemannian_utils.py`](../../../../src/modelcypher/core/domain/geometry/riemannian_utils.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `FrechetMeanResult` | 121 | Result dataclass with mean, iterations, convergence |
| `RiemannianGeometry.frechet_mean()` | 164 | Instance method for Fréchet mean computation |
| `frechet_mean()` | 1146 | Standalone function wrapping the class method |

**Also used in**: [`generalized_procrustes.py:32`](../../../../src/modelcypher/core/domain/geometry/generalized_procrustes.py) - `FrechetMeanConfig` for GPA consensus

**Key design decisions**:
1. **No Euclidean fallback** - If geodesic computation fails, we raise an error
2. **No scale clamping** - Extreme curvature is reported, not hidden
3. **Weighted support** - Handles importance-weighted averaging

---

## Citations

### Foundational

1. **Fréchet, M.** (1948). "Les éléments aléatoires de nature quelconque dans un espace distancié." *Annales de l'Institut Henri Poincaré*, 10(4), 215-310.
   - *Original definition of metric space mean*

2. **Karcher, H.** (1977). "Riemannian center of mass and mollifier smoothing." *Communications on Pure and Applied Mathematics*, 30(5), 509-541.
   - *Riemannian formulation and gradient descent algorithm*

3. **Sturm, K.-T.** (2003). "Probability measures on metric spaces of nonpositive curvature." *Contemporary Mathematics*, 338, 357-390.
   DOI: 10.1090/conm/338/06080
   - *Existence and uniqueness in Hadamard spaces*

### Neural Network Applications

4. **Lou, A., Katsman, I., Jiang, Q., Belongie, S., Lim, S.-N., & De Sa, C.** (2020). "Differentiating through the Fréchet Mean." *ICML 2020*.
   arXiv: 2003.00335
   - *Enables backprop through Fréchet mean; hyperbolic neural networks*

5. **Chakraborty, R., & Vemuri, B.C.** (2015). "Recursive Fréchet mean computation on the Grassmannian and its applications to computer vision." *ICCV 2015*.
   DOI: 10.1109/ICCV.2015.483
   - *Efficient computation for subspace manifolds*

6. **Chakraborty, R., Bouza, J., Manton, J.H., & Vemuri, B.C.** (2020). "ManifoldNet: A Deep Neural Network for Manifold-valued Data with Applications." *IEEE TPAMI*, 44(2), 799-810.
   DOI: 10.1109/TPAMI.2020.3003846
   - *General framework for manifold-valued deep learning*

### 2024-2025 Advances

7. **Iao, Y., et al.** (2025). "DFNN: A Deep Fréchet Neural Network Framework." *arXiv*.
   arXiv: 2510.17072
   - *Deep learning for Fréchet regression on metric spaces*

8. **Yang, Z., et al.** (2023). "Poincaré Fréchet mean." *Pattern Recognition*, 136, 109232.
   DOI: 10.1016/j.patcog.2023.109232
   - *Specialized algorithms for hyperbolic space*

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) - How we measure distances on the manifold
- [manifold_curvature.md](manifold_curvature.md) - Curvature affects mean computation
- [riemannian_density.md](../riemannian_density.md) - Density estimation using Fréchet mean

---

*The arithmetic mean is a special case of the Fréchet mean when the manifold is flat (Euclidean). Neural network representations are rarely flat.*
