# Intrinsic Dimension Estimation

> Measuring the true dimensionality of neural network representations.

---

## Why This Matters for Model Merging

Neural networks operate in high-dimensional spaces (4096 dimensions for LLMs), but representations often lie on low-dimensional manifolds. Understanding intrinsic dimension helps us:
1. **Choose compression rank**: TSV and LoRA rank selection
2. **Detect overfitting**: High ID may indicate memorization
3. **Assess merge compatibility**: Similar ID suggests similar geometry

**In ModelCypher**: Used in `intrinsic_dimension.py` for manifold complexity analysis and merge quality prediction.

---

## Formal Definition

### Definition

The **intrinsic dimension** (ID) of a dataset $X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^D$ is the minimum number of coordinates needed to represent $X$ without significant information loss.

Formally, if $X$ lies on a $d$-dimensional manifold $\mathcal{M} \subset \mathbb{R}^D$, then $\text{ID}(X) = d$.

### Extrinsic vs Intrinsic

- **Extrinsic dimension**: $D$ (ambient space dimension, e.g., 4096)
- **Intrinsic dimension**: $d$ (manifold dimension, often $d \ll D$)

---

## Estimation Methods

### 1. Maximum Likelihood Estimator (MLE)

**Levina & Bickel (2004)**

Treats k-nearest neighbor distances as a Poisson process:

$$\hat{d}_{MLE}(x_i) = \left( \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{r_k(x_i)}{r_j(x_i)} \right)^{-1}$$

where $r_j(x_i)$ is the distance to the $j$-th nearest neighbor of $x_i$.

**Global estimate**:
$$\hat{d} = \frac{1}{n} \sum_{i=1}^{n} \hat{d}_{MLE}(x_i)$$

**Robust version** (averaged over $k$ values):
$$\hat{d}_{robust} = \frac{1}{k_2 - k_1} \sum_{k=k_1}^{k_2} \hat{d}_{MLE}^{(k)}$$

### 2. TwoNN Estimator

**Facco et al. (2017)**

Uses only the ratio of first and second nearest neighbor distances:

$$\mu_i = \frac{r_2(x_i)}{r_1(x_i)}$$

Under the assumption of uniform density on a $d$-dimensional manifold:
$$P(\mu \leq \mu_0) = 1 - \mu_0^{-d}$$

**Estimator**:
$$\hat{d} = \frac{n}{\sum_{i=1}^{n} \log \mu_i}$$

**Advantages**:
- Minimal neighborhood dependency (only 2 neighbors)
- More robust to curvature
- Less sensitive to density variations

---

## Geodesic-Corrected ID

### The Problem with Euclidean ID

Standard ID estimators use Euclidean distances, which are incorrect on curved manifolds:
- **Positive curvature**: Euclidean underestimates true distance → ID overestimated
- **Negative curvature**: Euclidean overestimates true distance → ID underestimated

### ModelCypher Solution

Use geodesic distances (via k-NN graph) for ID estimation:

$$\hat{d}_{geo}(x_i) = \left( \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{d_{geo}(x_i, x_{(k)})}{d_{geo}(x_i, x_{(j)})} \right)^{-1}$$

where $d_{geo}$ is the shortest path distance on the k-NN graph.

---

## Key Theorems

### Theorem 1: Consistency (Levina & Bickel, 2004)

The MLE estimator is consistent: $\hat{d} \xrightarrow{p} d$ as $n \to \infty$ for manifolds with bounded curvature.

### Theorem 2: Bias (Facco et al., 2017)

The TwoNN estimator has bias $O(1/d)$ for finite samples, which decreases as intrinsic dimension increases.

### Theorem 3: Scale Dependence (Noia et al., 2024)

ID estimation is inherently scale-dependent. Different scales may reveal different intrinsic dimensions (multiscale structure).

---

## ID in Neural Networks

### Empirical Findings

From Ansuini et al. (2019) and subsequent work:

1. **ID increases through layers** (for classification)
2. **ID peaks at intermediate layers** (for generative models)
3. **Overparameterized networks have lower ID**
4. **ID correlates with generalization**

### Layer-wise ID Profile

```
Input Layer:   ID ≈ data dimension
Early Layers:  ID increases (feature extraction)
Middle Layers: ID peaks (representation learning)
Late Layers:   ID decreases (compression to output)
Output Layer:  ID ≈ number of classes
```

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/intrinsic_dimension.py`

```python
def estimate_intrinsic_dimension(
    points: Array,
    method: str = "mle",  # "mle" or "twonn"
    k_neighbors: int = 10,
    use_geodesic: bool = True,
    backend: Backend | None = None,
) -> IntrinsicDimensionResult:
    """
    Estimate intrinsic dimension of a point cloud.

    Uses geodesic distances by default for curvature-aware
    estimation on neural network manifolds.
    """
```

**Design decisions**:
1. **Geodesic by default**: Uses k-NN graph distances
2. **Robust averaging**: Averages over multiple k values
3. **Local ID support**: Can compute per-point ID for heterogeneous manifolds

---

## Citations

### Foundational

1. **Levina, E., & Bickel, P.J.** (2004). "Maximum Likelihood Estimation of Intrinsic Dimension." *NeurIPS 2004*.
   - *The MLE estimator*

2. **Facco, E., d'Errico, M., Rodriguez, A., & Laio, A.** (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information." *Scientific Reports*, 7, 12140.
   DOI: 10.1038/s41598-017-11873-y
   - *The TwoNN estimator*

### Neural Network Applications

3. **Ansuini, A., Laio, A., Macke, J.H., & Zoccolan, D.** (2019). "Intrinsic dimension of data representations in deep neural networks." *NeurIPS 2019*.
   arXiv: 1905.12784
   - *ID analysis of deep networks*

4. **Pope, P., et al.** (2021). "The Intrinsic Dimension of Images and Its Impact on Learning." *ICLR 2021*.
   arXiv: 2104.08894
   - *ID of image datasets*

### 2024-2025 Advances

5. **Konz, N., et al.** (2024). "Unraveling Learning Differences via Intrinsic Dimension." *ICLR 2024*.
   - *ID for understanding learning dynamics*

6. **Noia, A., et al.** (2024). "Scale-dependent intrinsic dimension estimation." *arXiv*.
   - *Multiscale ID analysis*

7. **Kataiwa, K., et al.** (2025). "Robust estimation of the intrinsic dimension of data sets." *Scientific Reports*, 15, 91676.
   DOI: 10.1038/s41598-025-91676-8
   - *Noise-robust ID estimation*

8. **Valeriani, L., et al.** (2024). "Intrinsic dimension correlation in neural networks." *arXiv*.
   arXiv: 2406.15812
   - *ID as a tool for network analysis*

### LLM-Specific

9. **Cheng, X., et al.** (2023). "Intrinsic Dimension of LLM Representations." *arXiv*.
   - *ID in large language models*

10. **Lee, S., et al.** (2024). "A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension." *RepL4NLP 2025*.
    - *ID across learning paradigms*

---

## Related Concepts

- [manifold_curvature.md](manifold_curvature.md) - Curvature affects ID estimation
- [task_singular_vectors.md](task_singular_vectors.md) - Low-rank structure relates to ID
- [geodesic_distance.md](geodesic_distance.md) - Distance computation for geodesic ID

---

*Intrinsic dimension tells us the true complexity of a representation space. A 4096-dimensional embedding might live on a 50-dimensional manifold.*
