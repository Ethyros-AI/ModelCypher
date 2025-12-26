# Fisher Information Matrix

> Importance-weighted parameter merging via information geometry.

---

## Why This Matters for Model Merging

Not all parameters are equally important. The Fisher Information Matrix quantifies **how much each parameter contributes to the model's predictions**. In merging, we use Fisher information to:
1. **Weight parameters by importance** during averaging
2. **Preserve critical parameters** from each model
3. **Avoid catastrophic interference** between tasks

**In ModelCypher**: Implemented in `fisher_blending.py` for importance-weighted model merging.

---

## Formal Definition

### Definition

For a model $p_\theta(y|x)$ parameterized by $\theta$, the **Fisher Information Matrix** is:

$$F_\theta = \mathbb{E}_{x,y} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^T \right]$$

**Alternative form** (via Hessian):
$$F_\theta = -\mathbb{E}_{x,y} \left[ \nabla_\theta^2 \log p_\theta(y|x) \right]$$

### Interpretation

- **Diagonal elements**: Sensitivity of loss to each parameter
- **Off-diagonal elements**: Parameter interactions
- **High Fisher value**: Parameter is important; small changes have large effects
- **Low Fisher value**: Parameter can vary without affecting predictions

---

## Fisher Merging

### The Algorithm (Matena & Raffel, 2022)

Given models $\theta_1, \ldots, \theta_T$ with Fisher matrices $F_1, \ldots, F_T$:

$$\theta_{merged} = \left( \sum_{i=1}^{T} \lambda_i F_i \right)^{-1} \left( \sum_{i=1}^{T} \lambda_i F_i \theta_i \right)$$

where $\lambda_i$ are model-level weights.

### Diagonal Approximation

Full Fisher is intractable ($|\theta|^2$ elements). Use diagonal:

$$\theta_{merged,j} = \frac{\sum_{i=1}^{T} \lambda_i F_{i,jj} \theta_{i,j}}{\sum_{i=1}^{T} \lambda_i F_{i,jj}}$$

**Interpretation**: Each parameter is averaged, weighted by its importance in each model.

---

## Geometric Interpretation

### Information Geometry (Amari, 2016)

The Fisher matrix defines a **Riemannian metric** on parameter space:

$$ds^2 = d\theta^T F_\theta d\theta$$

This metric makes the parameter space a **statistical manifold** where:
- Distance reflects distinguishability of distributions
- Geodesics are natural paths between models

### Connection to Natural Gradient

The natural gradient uses Fisher to normalize gradients:

$$\tilde{\nabla}_\theta = F_\theta^{-1} \nabla_\theta$$

Fisher merging is the natural generalization of averaging in this geometry.

---

## Eigendecomposition View (Tam et al., 2024)

### Theorem

Fisher merging can be written as:

$$\theta^* = \left( \sum_{i=1}^{M} Q_i \Lambda_i Q_i^T \right)^{-1} \left( \sum_{i=1}^{M} Q_i \Lambda_i Q_i^T \theta_i \right)$$

where $Q_i \Lambda_i Q_i^T$ is the eigendecomposition of $F_i$.

**Interpretation**: Parameters are upweighted along "important" eigenvector directions, preserving useful components during merging.

---

## Efficient Computation

### The Problem

Full Fisher requires:
1. Computing gradients for many examples
2. Outer products of gradient vectors
3. Storage of $|\theta|^2$ elements

### Solutions

1. **Diagonal approximation**: Only store diagonal (most common)
2. **K-FAC** (Kronecker-Factored Approximate Curvature): Block-diagonal with Kronecker structure
3. **Empirical Fisher**: Use training gradients directly
4. **Recycled gradients** (Li et al., 2025): Reuse optimizer accumulators

### From Optimizer State (Li et al., 2025)

Adam's squared gradient accumulator approximates diagonal Fisher:

$$v_t \approx \mathbb{E}[g_t^2] \approx \text{diag}(F)$$

This gives "Fishers for Free" without extra computation.

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/fisher_blending.py`

```python
def fisher_weighted_merge(
    models: list[dict[str, Array]],
    fisher_matrices: list[dict[str, Array]],
    model_weights: list[float] | None = None,
    config: FisherConfig | None = None,
    backend: Backend | None = None,
) -> dict[str, Array]:
    """
    Merge models using Fisher-weighted averaging.

    Each parameter is weighted by its importance (Fisher value)
    in each source model.
    """
```

**Design decisions**:
1. **Diagonal by default**: Full Fisher is impractical for LLMs
2. **Per-layer Fisher**: Compute separately for each layer
3. **Fallback weighting**: If no Fisher, use equal weights

---

## Relationship to Other Methods

| Method | Weighting Scheme |
|--------|-----------------|
| Simple Average | Uniform |
| **Fisher Merge** | Importance-weighted |
| TIES-Merge | Sign-based with trimming |
| DARE | Random with rescaling |
| Task Arithmetic | Task vector addition |

Fisher merging is complementary to these; can combine approaches.

---

## Continual Learning Connection

Fisher information is central to **Elastic Weight Consolidation (EWC)**:

$$L_{EWC} = L_{task} + \lambda \sum_j F_j (\theta_j - \theta_j^*)^2$$

This penalizes changes to important parameters, preventing catastrophic forgetting.

**Merging interpretation**: Fisher merging is EWC applied to combining multiple models rather than sequential learning.

---

## Citations

### Foundational

1. **Fisher, R.A.** (1925). "Theory of Statistical Estimation." *Proceedings of the Cambridge Philosophical Society*, 22, 700-725.
   - *Original Fisher information*

2. **Amari, S.** (2016). *Information Geometry and Its Applications*. Springer.
   - *Comprehensive treatment of information geometry*

### Neural Network Merging

3. **Matena, M.S., & Raffel, C.** (2022). "Merging Models with Fisher-Weighted Averaging." *NeurIPS 2022*.
   arXiv: 2111.09832
   - *Fisher merging for neural networks*

4. **Tam, D., et al.** (2024). "Dynamic Fisher-weighted Model Merging via Bayesian Optimization." *NAACL 2025*.
   - *Geometric analysis of Fisher merging*

5. **Li, Y.X., Dangel, F., Tam, D., & Raffel, C.** (2025). "Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator." *ICML 2025 Spotlight*.
   OpenReview: https://openreview.net/forum?id=m3zrHhiCCj
   - *Efficient Fisher from optimizer state*

### Federated Learning

6. **Jhunjhunwala, D., et al.** (2024). "FedFisher: Leveraging Fisher Information for One-Shot Federated Learning." *AISTATS 2024*.
   - *Fisher for federated merging*

### Continual Learning

7. **Kirkpatrick, J., et al.** (2017). "Overcoming catastrophic forgetting in neural networks." *PNAS*, 114(13), 3521-3526.
   DOI: 10.1073/pnas.1611835114
   - *Elastic Weight Consolidation (EWC)*

### Approximations

8. **Martens, J., & Grosse, R.** (2015). "Optimizing Neural Networks with Kronecker-Factored Approximate Curvature." *ICML 2015*.
   arXiv: 1503.05671
   - *K-FAC approximation*

---

## Related Concepts

- [task_singular_vectors.md](task_singular_vectors.md) - Complementary importance measure
- [procrustes_analysis.md](procrustes_analysis.md) - Alignment before Fisher merging
- [frechet_mean.md](frechet_mean.md) - Fisher metric defines Riemannian geometry

---

*Fisher information tells us which parameters matter. This knowledge is essential for intelligent model merging.*
