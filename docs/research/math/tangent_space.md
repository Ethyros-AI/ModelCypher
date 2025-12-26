# Tangent Space and Riemannian Operations

> Local linearization for computation on curved manifolds.

---

## Why This Matters for Model Merging

Curved manifolds don't allow direct arithmetic. The tangent space provides:
1. **Local Euclidean approximation**: Do linear operations in tangent space
2. **Geodesic computation**: Exponential map traces geodesics
3. **Transport between points**: Move vectors along the manifold

**In ModelCypher**: Implemented in `tangent_space_alignment.py` and `riemannian_utils.py`.

---

## The Core Insight

At each point $p$ on a manifold $M$, there's a **tangent space** $T_pM$—the best local linear approximation. We can:
1. **Lift** from manifold to tangent space (logarithm map)
2. **Compute** in the linear tangent space
3. **Project** back to the manifold (exponential map)

---

## Formal Definitions

### Tangent Space

The **tangent space** $T_pM$ at point $p \in M$ is the vector space of all tangent vectors to curves through $p$.

**Intuition**: For a 2D surface in 3D, the tangent space at $p$ is the tangent plane.

### Exponential Map

The **exponential map** $\text{Exp}_p: T_pM \to M$ maps a tangent vector to the manifold:

$$\text{Exp}_p(v) = \gamma(1)$$

where $\gamma$ is the geodesic starting at $p$ with initial velocity $v$.

**Properties**:
- $\text{Exp}_p(0) = p$
- $\text{Exp}_p(tv) = \gamma(t)$ for the geodesic

### Logarithm Map

The **logarithm map** $\text{Log}_p: M \to T_pM$ is the inverse of $\text{Exp}_p$:

$$\text{Log}_p(q) = v \text{ such that } \text{Exp}_p(v) = q$$

**Properties**:
- $\text{Log}_p(p) = 0$
- $\|\text{Log}_p(q)\| = d(p, q)$ (geodesic distance)

### Parallel Transport

**Parallel transport** $\Gamma_{p \to q}: T_pM \to T_qM$ moves vectors along geodesics while preserving:
- Length: $\|v\| = \|\Gamma_{p \to q}(v)\|$
- Angle: Inner products preserved

---

## Riemannian Operations in Practice

### Fréchet Mean via Tangent Space

```python
def frechet_mean_tangent(points: list[Array], base: Array, n_iter: int = 10):
    """
    Compute Fréchet mean using tangent space iterations.

    1. Lift all points to tangent space at current estimate
    2. Compute Euclidean mean in tangent space
    3. Project back to manifold
    4. Repeat until convergence
    """
    mean = base

    for _ in range(n_iter):
        # Lift to tangent space
        tangent_vectors = [log_map(mean, p) for p in points]

        # Euclidean mean in tangent space
        tangent_mean = sum(tangent_vectors) / len(tangent_vectors)

        # Project back
        mean = exp_map(mean, tangent_mean)

    return mean
```

### Geodesic Interpolation

```python
def geodesic_interpolation(p: Array, q: Array, t: float):
    """
    Interpolate along geodesic from p to q.

    Equivalent to SLERP for spherical manifolds.
    """
    # Direction in tangent space
    v = log_map(p, q)

    # Walk fraction t along geodesic
    return exp_map(p, t * v)
```

---

## Specific Manifolds

### Sphere $S^{n-1}$

For unit vectors on the sphere:

**Exponential map**:
$$\text{Exp}_p(v) = \cos(\|v\|) p + \sin(\|v\|) \frac{v}{\|v\|}$$

**Logarithm map**:
$$\text{Log}_p(q) = \frac{\theta}{\sin\theta}(q - \cos\theta \cdot p)$$

where $\theta = \arccos(p \cdot q)$.

### Symmetric Positive Definite (SPD) Matrices

For the SPD manifold (covariance matrices):

**Exponential map** (Affine-Invariant metric):
$$\text{Exp}_P(V) = P^{1/2} \exp(P^{-1/2} V P^{-1/2}) P^{1/2}$$

**Logarithm map**:
$$\text{Log}_P(Q) = P^{1/2} \log(P^{-1/2} Q P^{-1/2}) P^{1/2}$$

### Grassmann Manifold

For subspaces (relevant for neural network layers):

The Grassmannian $\text{Gr}(k, n)$ is the space of $k$-dimensional subspaces of $\mathbb{R}^n$.

---

## Applications in Neural Networks

### Representation Geometry

Neural representations lie on manifolds, not in flat Euclidean space:
- Normalize layers → spherical manifold
- Covariance structures → SPD manifold
- Subspace representations → Grassmannian

### Tangent Classifier (2024-2025)

Recent work on **tangent classifiers** (Chen et al., 2024):
- Map features to tangent space via logarithm
- Apply Euclidean classifier in tangent space
- Respects underlying geometry

### Wrapped Gaussian Processes (2025)

For data on Riemannian manifolds:
- Define GP in tangent space
- "Wrap" onto manifold via exponential map
- Enables Bayesian inference on curved spaces

---

## For Model Merging

### Tangent Space Averaging

Instead of Euclidean averaging:

```python
def tangent_space_merge(weights: list[Array], base: Array):
    """
    Merge weights via tangent space averaging.

    More appropriate than Euclidean when weights lie on manifold.
    """
    # Lift to tangent space at base
    tangent_vectors = [log_map(base, w) for w in weights]

    # Average in tangent space
    mean_tangent = sum(tangent_vectors) / len(tangent_vectors)

    # Project back
    return exp_map(base, mean_tangent)
```

### Connection to SLERP

For spherical manifolds, geodesic interpolation via exp/log maps is equivalent to SLERP. The tangent space framework generalizes this to arbitrary manifolds.

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/tangent_space_alignment.py`

```python
class TangentSpaceOperations:
    """Tangent space operations for Riemannian manifolds."""

    def exp_map(
        self,
        base: Array,
        tangent_vector: Array,
        manifold: str = "sphere",
    ) -> Array:
        """
        Exponential map: tangent space → manifold.
        """

    def log_map(
        self,
        base: Array,
        point: Array,
        manifold: str = "sphere",
    ) -> Array:
        """
        Logarithm map: manifold → tangent space.
        """

    def parallel_transport(
        self,
        vector: Array,
        from_point: Array,
        to_point: Array,
    ) -> Array:
        """
        Parallel transport vector along geodesic.
        """
```

**Also in**: `src/modelcypher/core/domain/geometry/riemannian_utils.py`
- Fréchet mean using tangent space iterations
- Geodesic distances

---

## Citations

### Foundational

1. **Lee, J.M.** (2018). *Introduction to Riemannian Manifolds*. 2nd ed. Springer.
   - *Standard graduate reference*

2. **do Carmo, M.P.** (1992). *Riemannian Geometry*. Birkhäuser.
   - *Classic treatment*

### Computational

3. **Absil, P.-A., Mahony, R., & Sepulchre, R.** (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
   - *Algorithms for Riemannian optimization*

4. **Pennec, X., Fillard, P., & Ayache, N.** (2006). "A Riemannian Framework for Tensor Computing." *IJCV*, 66(1), 41-66.
   - *SPD manifold operations*

### Neural Network Applications

5. **Bronstein, M.M., et al.** (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges."
   arXiv: 2104.13478
   - *Comprehensive geometric DL survey*

6. **NeurIPS 2020**: "Deep Riemannian Manifold Learning."
   - *Learned Riemannian metrics*

### 2024-2025 Advances

7. **Chen, X., et al.** (2024). "Matrix Functions and Tangent Classifiers."
   arXiv: 2407.10484
   - *Tangent classifiers for SPD data*

8. **Rozo, L., et al.** (2025). "Riemann²: Learning Riemannian Submanifolds from Riemannian Data."
   arXiv: 2503.05540
   - *Wrapped GP models*

9. **OpenReview ICLR 2025**: "Riemannian Transformation Layers for Generative Models."
   - *Exp/log maps for generative modeling*

10. **arXiv 2501.12678** (2025). "Manifold learning and optimization using tangent space proxies."
    - *Modern tangent space optimization*

---

## Related Concepts

- [frechet_mean.md](frechet_mean.md) - Computed via tangent space iterations
- [geodesic_distance.md](geodesic_distance.md) - Computed from log map norms
- [slerp.md](slerp.md) - Geodesic interpolation on spheres

---

*The tangent space is the bridge between curved manifolds and linear algebra—enabling Riemannian computation via local Euclidean approximations.*
