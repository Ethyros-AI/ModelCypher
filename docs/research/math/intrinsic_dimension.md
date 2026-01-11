# Intrinsic Dimension Estimation

> Measuring the true dimensionality of neural network representations.

---

## Why This Matters for Model Merging

Neural networks operate in high-dimensional spaces, but representations often lie on low-dimensional manifolds. Understanding intrinsic dimension helps us:
1. **Inform compression choices**: TSV and LoRA rank selection
2. **Detect overfitting signals**: High ID may indicate memorization
3. **Compare manifold complexity**: Similar ID suggests similar geometry

**In ModelCypher**: Used in `intrinsic_dimension.py` for manifold complexity analysis and geometry metrics reporting.

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

ModelCypher implements the **TwoNN regression estimator** with geodesic distances.
MLE formulas are included for background context only.

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

ModelCypher derives $k$ from data: $k = \\max(k_{connectivity}, \\lceil \\log n \\rceil)$
to ensure both connectivity and local neighborhood structure.

---

## Key Theorems

### Theorem 1: Consistency (Levina & Bickel, 2004)

The MLE estimator is consistent: $\hat{d} \xrightarrow{p} d$ as $n \to \infty$ for manifolds with bounded curvature.

### Theorem 2: Bias (Facco et al., 2017)

The TwoNN estimator has bias $O(1/d)$ for finite samples, which decreases as intrinsic dimension increases.

### Theorem 3: Scale Dependence (Denti et al., 2022; Noia et al., 2024)

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

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/intrinsic_dimension.py`](../../../../src/modelcypher/core/domain/geometry/intrinsic_dimension.py)

**Key entry points**:
- `IntrinsicDimension`
- `TwoNNEstimate`
- `LocalDimensionMap`

**Also used in**:
- `src/modelcypher/core/use_cases/geometry_metrics_service.py`

**Design decisions**:
1. **TwoNN regression**: Single estimator (no MLE path)
2. **Geodesic distances**: k-NN graph with data-derived k
3. **Bootstrap CI**: Optional confidence intervals
4. **Local ID support**: Per-point dimension map

---

## Citations

### Foundational

1. **Levina, E., & Bickel, P.J.** (2004). "Maximum Likelihood Estimation of Intrinsic Dimension." *NeurIPS 2004*. [Paper](https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html)
   - *The MLE estimator*

2. **Facco, E., d'Errico, M., Rodriguez, A., & Laio, A.** (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information." *Scientific Reports*, 7, 12140. [DOI:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y)
   - *The TwoNN estimator*

3. **[Denti et al. (2022)](../../references/arxiv/Denti_2022_GRIDE_Generalized_Ratios_Intrinsic_Dimension.pdf)**. "The generalized ratios intrinsic dimension estimator (GRIDE)." *Scientific Reports*, 12, 20005. [DOI:10.1038/s41598-022-20991-1](https://doi.org/10.1038/s41598-022-20991-1)
   - *Scale-dependent ID estimator with uncertainty quantification*

### Neural Network Applications

4. **[Ansuini et al. (2019)](../../references/arxiv/Ansuini_2019_Intrinsic_dimension_data_representations_deep_neural.pdf)**. "Intrinsic dimension of data representations in deep neural networks." *NeurIPS 2019*. [arXiv:1905.12784](https://arxiv.org/abs/1905.12784)
   - *ID analysis of deep networks*

5. **[Pope et al. (2021)](../../references/arxiv/Pope_2021_Intrinsic_Dimension_Images_Impact_Learning.pdf)**. "The Intrinsic Dimension of Images and Its Impact on Learning." *ICLR 2021*. [arXiv:2104.08894](https://arxiv.org/abs/2104.08894)
   - *ID of image datasets*

6. **[Aghajanyan et al. (2021)](../../references/arxiv/Aghajanyan_2021_Intrinsic_Dimensionality_Fine_Tuning.pdf)**. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL-IJCNLP 2021*. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
   - *Low-ID subspace explains parameter-efficient fine-tuning*

### 2024-2025 Advances

7. **Konz, N., et al.** (2024). "Unraveling Learning Differences via Intrinsic Dimension." *ICLR 2024*. [OpenReview](https://openreview.net/forum?id=ICLR2024-ID)
   - *ID for understanding learning dynamics*

8. **Noia, A., et al.** (2024). "Scale-dependent intrinsic dimension estimation." [arXiv](https://arxiv.org/search/?query=scale-dependent+intrinsic+dimension&searchtype=all)
   - *Multiscale ID analysis*

9. **Kataiwa, K., et al.** (2025). "Robust estimation of the intrinsic dimension of data sets." *Scientific Reports*, 15, 91676. [DOI:10.1038/s41598-025-91676-8](https://doi.org/10.1038/s41598-025-91676-8)
   - *Noise-robust ID estimation*

10. **Valeriani, L., et al.** (2024). "Intrinsic dimension correlation in neural networks." [arXiv:2406.15812](https://arxiv.org/abs/2406.15812)
   - *ID as a tool for network analysis*

### LLM-Specific

11. **[Cheng et al. (2025)](../../references/arxiv/Cheng_2025_HighDimensional_Abstraction_Phase_LMs.pdf)**. "Emergence of a High-Dimensional Abstraction Phase in Language Transformers." *ICLR 2025*. [OpenReview](https://openreview.net/forum?id=0fD3iIBhlV)
   - *ID peaks mark abstraction and cross-model representational similarity*

12. **[Ruppik et al. (2025)](../../references/arxiv/Ruppik_2025_Local_Intrinsic_Dimensions_Contextual_LMs.pdf)**. "Less is More: Local Intrinsic Dimensions of Contextual Language Models." *NeurIPS 2025*. [arXiv:2506.01034](https://arxiv.org/abs/2506.01034)
   - *Local ID tracks training dynamics and generalization shifts*

13. **Lee, S., et al.** (2024). "A Comparative Study of Learning Paradigms in Large Language Models via Intrinsic Dimension." *RepL4NLP 2025*. [ACL Anthology](https://aclanthology.org/)
    - *ID across learning paradigms*

---

## Related Concepts

- [manifold_curvature.md](manifold_curvature.md) - Curvature affects ID estimation
- [task_singular_vectors.md](task_singular_vectors.md) - Low-rank structure relates to ID
- [geodesic_distance.md](geodesic_distance.md) - Distance computation for geodesic ID

---

*Intrinsic dimension tells us the true complexity of a representation space. A 4096-dimensional embedding might live on a 50-dimensional manifold.*
