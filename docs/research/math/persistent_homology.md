# Persistent Homology and Topological Data Analysis

> Extracting multi-scale topological features from neural representations.

---

## Why This Matters for Model Merging

Neural network representations have **shape**—connected components, loops, voids. Persistent homology captures this shape across scales:
1. **Topological fingerprinting**: Unique signature of representation structure
2. **Merge compatibility**: Models with similar topology merge better
3. **Scale-invariant features**: Robust to magnitude differences

**In ModelCypher**: Implemented in `topological_fingerprint.py` for representation topology analysis.

---

## The Core Insight

Instead of asking "how far apart are these points?", TDA asks "what is the shape of this point cloud?" This shape—captured via **homology groups**—describes:
- $H_0$: Connected components
- $H_1$: Loops/cycles
- $H_2$: Voids/cavities
- Higher $H_k$: Higher-dimensional holes

Persistent homology tracks how these features appear and disappear across scales.

---

## Formal Definitions

### Simplicial Complex

A **simplicial complex** $K$ is a collection of simplices (points, edges, triangles, tetrahedra, ...) such that:
1. Every face of a simplex in $K$ is also in $K$
2. The intersection of any two simplices is a face of both

### Filtration

A **filtration** is a nested sequence of simplicial complexes:

$$\emptyset = K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_n = K$$

Built from data via **Vietoris-Rips complex**:
- At scale $\epsilon$, connect points within distance $\epsilon$
- Add higher simplices for complete subgraphs

### Homology Groups

The **$k$-th homology group** $H_k(K)$ captures $k$-dimensional holes:

$$H_k(K) = \ker(\partial_k) / \text{im}(\partial_{k+1})$$

where $\partial_k$ is the boundary operator.

**Betti numbers**: $\beta_k = \text{rank}(H_k)$ = number of $k$-dimensional holes

### Persistence

A feature is **born** at scale $\epsilon_b$ and **dies** at scale $\epsilon_d$. The **persistence** is:

$$\text{persistence} = \epsilon_d - \epsilon_b$$

Long-lived features are topologically significant; short-lived features are noise.

---

## Persistence Diagrams and Barcodes

### Persistence Diagram

A multiset of points $(b_i, d_i)$ in the extended plane, where:
- $b_i$ = birth time of feature $i$
- $d_i$ = death time of feature $i$

Points far from the diagonal $y = x$ are significant (long persistence).

### Barcode

Equivalent representation as a collection of intervals $[b_i, d_i)$.

### Vectorization for ML

To use persistence in machine learning, convert to vectors:

1. **Persistence Images** (Adams et al., 2017): Gaussian-weighted pixelization
2. **Persistence Landscapes** (Bubenik, 2015): Piecewise-linear functions
3. **Betti Curves**: $\beta_k(\epsilon)$ as a function of scale

---

## Algorithm: Computing Persistent Homology

```python
def compute_persistent_homology(points: Array, max_dim: int = 2) -> PersistenceDiagram:
    """
    Compute persistent homology of a point cloud.

    Args:
        points: [n, d] point cloud
        max_dim: Maximum homology dimension to compute

    Returns:
        Persistence diagrams for H_0, H_1, ..., H_{max_dim}
    """
    # 1. Compute pairwise distances
    distances = pairwise_distances(points)

    # 2. Build Vietoris-Rips filtration
    #    (or use alpha complex for efficiency)
    filtration = build_vietoris_rips(distances, max_epsilon)

    # 3. Compute persistent homology via matrix reduction
    #    (standard algorithm or optimized variants)
    diagrams = reduce_boundary_matrix(filtration, max_dim)

    return diagrams
```

**Complexity**: $O(n^3)$ for standard algorithm; faster with optimizations.

---

## Applications to Neural Networks

### Representation Topology

Analyze the topology of:
- **Activation spaces**: What shape do representations form?
- **Layer progression**: How does topology evolve through layers?
- **Training dynamics**: How does topology change during training?

### Topological Loss (Stucki et al., 2023)

Use persistence in training objectives:

$$\mathcal{L}_{topo} = \sum_{(b,d) \in D} w(b,d) \cdot |d - b|$$

Encourages specific topological structure in learned representations.

### Generalization Prediction

Topology correlates with generalization:
- More complex topology (higher Betti numbers) often indicates overfitting
- Persistent features correlate with robust representations

---

## For Model Merging

### Topological Fingerprints

Compare models by their representation topology:

```python
def topological_similarity(model_a, model_b, data):
    """Compare models by their topological fingerprints."""
    # Get activations
    acts_a = model_a.get_activations(data)
    acts_b = model_b.get_activations(data)

    # Compute persistence diagrams
    diagram_a = compute_persistent_homology(acts_a)
    diagram_b = compute_persistent_homology(acts_b)

    # Compare diagrams (Wasserstein or bottleneck distance)
    return persistence_distance(diagram_a, diagram_b)
```

### Merge Compatibility

Models with similar topological structure:
- Share similar representation geometry
- Merge more smoothly
- Preserve topological features after merging

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/topological_fingerprint.py`](../../../../src/modelcypher/core/domain/geometry/topological_fingerprint.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `TopologicalFingerprint` | 153 | Main class with `compute_fingerprint()` and `compare_fingerprints()` |

**Also in**:
- [`geometry_metrics_service.py:78`](../../../../src/modelcypher/core/use_cases/geometry_metrics_service.py) - `TopologicalFingerprintResult`

**Design decisions**:
1. **Multi-scale**: Capture features at all scales
2. **Efficient computation**: Use alpha complexes for high-dimensional data
3. **Vectorized output**: Compatible with downstream ML

---

## Distance Between Persistence Diagrams

### Bottleneck Distance

$$d_\infty(D_1, D_2) = \inf_{\gamma} \sup_{x \in D_1} \|x - \gamma(x)\|_\infty$$

where $\gamma$ ranges over bijections (including matching to diagonal).

### Wasserstein Distance

$$W_p(D_1, D_2) = \left( \inf_{\gamma} \sum_{x \in D_1} \|x - \gamma(x)\|_\infty^p \right)^{1/p}$$

$W_2$ is most common for ML applications.

---

## Citations

### Foundational

1. **Edelsbrunner, H., Letscher, D., & Zomorodian, A.** (2002). "Topological Persistence and Simplification." *Discrete & Computational Geometry*, 28(4), 511-533.
   DOI: 10.1007/s00454-002-2885-2
   - *Original persistent homology algorithm*

2. **Zomorodian, A., & Carlsson, G.** (2005). "Computing Persistent Homology." *Discrete & Computational Geometry*, 33(2), 249-274.
   - *Efficient computation*

3. **Carlsson, G.** (2009). "Topology and Data." *Bulletin of the AMS*, 46(2), 255-308.
   - *Foundational survey*

### Vectorization

4. **Adams, H., et al.** (2017). "Persistence Images: A Stable Vector Representation of Persistent Homology." *JMLR*, 18(8), 1-35.
   - *Persistence images*

5. **Bubenik, P.** (2015). "Statistical Topological Data Analysis using Persistence Landscapes." *JMLR*, 16, 77-102.
   - *Persistence landscapes*

### Neural Network Applications

6. **Hensel, F., et al.** (2021). "A Survey of Topological Machine Learning Methods." *Frontiers in AI*, 4, 681108.
   DOI: 10.3389/frai.2021.681108
   - *Comprehensive TDA+ML survey*

7. **Rieck, B., et al.** (2019). "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology." *ICLR 2019*.
   - *Topology for analyzing neural networks*

### 2024-2025 Advances

8. **NeurIPS 2024 Poster**: "Boosting Graph Pooling with Persistent Homology."
   - *TDA for graph neural networks*

9. **Nature Scientific Reports** (2025). "Machine learning of time series data using persistent homology."
   DOI: 10.1038/s41598-025-06551-3
   - *Recent applications*

10. **Neurocomputing** (2024). "Predicting the generalization gap in neural networks using topological data analysis."
    - *Topology predicts generalization*

11. **Neurocomputing** (2025). "A comprehensive review of deep neural network interpretation using topological data analysis."
    DOI: 10.1016/j.neucom.2024.128840
    - *Comprehensive 2024-2025 review*

---

## Related Concepts

- [intrinsic_dimension.md](intrinsic_dimension.md) - Complementary geometric measure
- [manifold_curvature.md](manifold_curvature.md) - Local vs global (topological) geometry
- [geodesic_distance.md](geodesic_distance.md) - Distance underlying Vietoris-Rips

---

*Persistent homology captures the shape of data—connected components, loops, voids—providing a topological fingerprint for neural representations.*
