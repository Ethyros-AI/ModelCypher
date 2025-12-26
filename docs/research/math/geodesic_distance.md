# Geodesic Distance on k-NN Graphs

> The true distance on discrete manifolds.

---

## Why This Matters for Model Merging

In high-dimensional curved spaces, **Euclidean distance is wrong**:
- Positive curvature: Euclidean underestimates true distance
- Negative curvature: Euclidean overestimates true distance

Geodesic distance via k-NN graphs gives the **exact** distance on the discrete manifold.

**In ModelCypher**: Foundational operation in `riemannian_utils.py`, used by Fréchet mean, CKA, curvature estimation, and all geometric computations.

---

## The Key Insight

Neural network activations form a **discrete manifold**:
- We observe finite samples (activation vectors)
- The k-NN graph approximates the continuous manifold
- Shortest paths on the graph ARE geodesics on the discrete manifold

This is **not** an approximation—it's exact for the discrete structure we actually have.

---

## Formal Definition

### k-NN Graph Construction

Given points $X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d$ and neighborhood size $k$:

1. For each point $x_i$, find $k$ nearest neighbors by Euclidean distance
2. Create edges: $(i, j) \in E$ if $x_j$ is among $k$-nearest neighbors of $x_i$
3. Edge weight: $w_{ij} = \|x_i - x_j\|_2$

### Geodesic Distance

The geodesic distance is the shortest path on the graph:

$$d_{geo}(x_i, x_j) = \min_{\text{paths } p: i \to j} \sum_{(a,b) \in p} w_{ab}$$

Computed via Dijkstra's algorithm or Floyd-Warshall.

---

## Mathematical Foundation

### Continuous Setting

On a Riemannian manifold $(M, g)$, the geodesic distance is:

$$d_g(p, q) = \inf_{\gamma} \int_0^1 \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt$$

where $\gamma$ ranges over smooth paths from $p$ to $q$.

### Discrete Approximation Guarantee

**Theorem (Tenenbaum et al., 2000)**

For $n$ points sampled from a $d$-dimensional manifold $M$ with bounded curvature, the k-NN graph geodesic converges to the true geodesic:

$$|d_{geo}^{graph}(x_i, x_j) - d_g(x_i, x_j)| \to 0 \text{ as } n \to \infty$$

with appropriate $k = O(n^{2/(d+4)})$.

---

## Why Euclidean Fails

### Example: Sphere

For two points on a sphere with angular separation $\theta$:
- **Euclidean**: $d_{euc} = 2R\sin(\theta/2)$ (chord length)
- **Geodesic**: $d_{geo} = R\theta$ (arc length)

For $\theta = \pi/2$: Euclidean underestimates by 11%.

### Neural Network Manifolds

Empirical observations:
- Early layers: Often positive curvature
- Middle layers: Mixed curvature
- Late layers: Complex curvature patterns

Euclidean distance can be systematically wrong by 10-50% in curved regions.

---

## The Geodesic Defect

We use the geodesic defect to measure curvature:

$$\delta(x, y) = \frac{d_{geo}(x, y)}{d_{euc}(x, y)} - 1$$

- $\delta > 0$: Geodesic longer than Euclidean (positive curvature)
- $\delta < 0$: Geodesic shorter than Euclidean (negative curvature)
- $\delta = 0$: Flat

---

## Algorithm: Geodesic Distance Matrix

```python
def geodesic_distances(points, k_neighbors):
    """
    Compute all-pairs geodesic distances via k-NN graph.

    1. Compute Euclidean distance matrix
    2. Build k-NN graph (keep k nearest neighbors per point)
    3. Symmetrize: edge exists if either direction qualifies
    4. Run Floyd-Warshall or Dijkstra for shortest paths
    5. Handle disconnected components (distance = inf)
    """
    # Euclidean distances
    euc_dist = pairwise_euclidean(points)

    # k-NN graph adjacency
    adj = build_knn_graph(euc_dist, k_neighbors)

    # Shortest paths
    geo_dist = floyd_warshall(adj)

    return geo_dist
```

---

## Handling Disconnected Components

If the k-NN graph is disconnected:
- Some pairs have infinite geodesic distance
- This indicates the manifold has multiple components

**ModelCypher approach**: Raise explicit error rather than fall back to Euclidean.

```python
if np.any(np.isinf(geo_dist)):
    raise ValueError(
        "k-NN graph is disconnected. Increase k_neighbors or "
        "check for outliers in the data."
    )
```

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/riemannian_utils.py`](../../../../src/modelcypher/core/domain/geometry/riemannian_utils.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `GeodesicDistanceResult` | 131 | Result dataclass with distances, graph, defect |
| `RiemannianGeometry.geodesic_distances()` | 293 | Instance method for k-NN graph geodesics |
| `geodesic_distance_matrix()` | 1172 | Standalone function for pairwise distances |

**Also in**: [`riemannian_density.py:261`](../../../../src/modelcypher/core/domain/geometry/riemannian_density.py) - point-to-point geodesic distance

**Design decisions**:
1. **No Euclidean fallback**: Disconnection is an error, not a fallback case
2. **Adaptive k**: Default based on sample size
3. **Symmetric graph**: Edge exists if either direction qualifies
4. **Backend-agnostic**: Works with any Backend implementation

---

## Relationship to Other Methods

| Method | Space | Use Case |
|--------|-------|----------|
| **k-NN Geodesic** | Discrete manifold | General neural representations |
| ISOMAP | Embedding + geodesic | Dimensionality reduction |
| Diffusion Distance | Markov chain | Multi-scale structure |
| Earth Mover's Distance | Distributions | Distribution comparison |

---

## Choosing k

### Too Small k
- Graph disconnects
- Poor geodesic approximation
- Sensitive to noise

### Too Large k
- Short-circuits the manifold
- Geodesic → Euclidean
- Loses curvature information

### Rule of Thumb

$$k \approx 2d + 1$$

where $d$ is the intrinsic dimension. For neural networks:
- $k = 10$ is a reasonable default
- Increase if disconnection occurs
- Decrease if geodesic ≈ Euclidean everywhere

---

## Citations

### Foundational

1. **Dijkstra, E.W.** (1959). "A note on two problems in connexion with graphs." *Numerische Mathematik*, 1, 269-271. [DOI:10.1007/BF01386390](https://doi.org/10.1007/BF01386390)
   - *Shortest path algorithm*

2. **Tenenbaum, J.B., de Silva, V., & Langford, J.C.** (2000). "A Global Geometric Framework for Nonlinear Dimensionality Reduction." *Science*, 290(5500), 2319-2323. [DOI:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
   - *ISOMAP and geodesic approximation guarantees*

3. **Bernstein, M., de Silva, V., Langford, J.C., & Tenenbaum, J.B.** (2000). "Graph Approximations to Geodesics on Embedded Manifolds." *Technical Report*. [Stanford](https://web.stanford.edu/class/cs368a/readings/bernstein_graph_approx.pdf)
   - *Convergence analysis*

### Modern Treatment

4. **Peyre, G., & Cuturi, M.** (2019). "Computational Optimal Transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
   - *Chapter on geodesic computation*

5. **Lee, J.A., & Verleysen, M.** (2007). *Nonlinear Dimensionality Reduction*. Springer. [DOI:10.1007/978-0-387-39351-3](https://doi.org/10.1007/978-0-387-39351-3)
   - *Comprehensive treatment of manifold methods*

---

## Related Concepts

- [frechet_mean.md](frechet_mean.md) - Uses geodesic distance for curvature-aware averaging
- [manifold_curvature.md](manifold_curvature.md) - Curvature from geodesic/Euclidean ratios
- [intrinsic_dimension.md](intrinsic_dimension.md) - Geodesic-corrected ID estimation

---

*Euclidean distance is a convenient lie. Geodesic distance on the k-NN graph is the truth our geometry is built on.*
