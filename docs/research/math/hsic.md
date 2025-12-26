# HSIC: Hilbert-Schmidt Independence Criterion

> The kernel-based foundation underlying CKA and representation similarity.

---

## Why This Matters for Model Merging

HSIC is the **mathematical core** of CKA and other representation similarity measures. Understanding HSIC reveals:
1. **What CKA actually measures**: Statistical dependence in kernel space
2. **Why it's dimension-agnostic**: Operates on Gram matrices
3. **Kernel choice effects**: How different kernels capture different structure

**In ModelCypher**: HSIC underlies our CKA implementation in `cka.py`.

---

## The Core Insight

HSIC measures statistical dependence between two random variables by:
1. Mapping each to a **reproducing kernel Hilbert space** (RKHS)
2. Computing the **cross-covariance operator**
3. Measuring its **Hilbert-Schmidt norm**

This captures **all orders of correlation**, not just linear.

---

## Formal Definition

### Definition (Gretton et al., 2005)

Given random variables $X$ and $Y$ with kernels $k$ and $l$, the **Hilbert-Schmidt Independence Criterion** is:

$$\text{HSIC}(X, Y) = \mathbb{E}_{xx'yy'}[k(x,x')l(y,y')] - 2\mathbb{E}_{xy}[\mathbb{E}_{x'}[k(x,x')]\mathbb{E}_{y'}[l(y,y')]] + \mathbb{E}_{xx'}[k(x,x')]\mathbb{E}_{yy'}[l(y,y')]$$

where $(x,y)$ and $(x',y')$ are independent copies from the joint distribution.

### Compact Form

$$\text{HSIC}(X, Y) = \|\mathcal{C}_{XY}\|_{HS}^2$$

where $\mathcal{C}_{XY}$ is the cross-covariance operator between the RKHS embeddings.

### Key Property

$$\text{HSIC}(X, Y) = 0 \iff X \perp Y$$

(when using characteristic kernels like RBF)

---

## Empirical Estimator

### Biased Estimator

Given samples $\{(x_i, y_i)\}_{i=1}^n$, kernel matrices $K_{ij} = k(x_i, x_j)$ and $L_{ij} = l(y_i, y_j)$:

$$\widehat{\text{HSIC}} = \frac{1}{(n-1)^2} \text{tr}(KHLH)$$

where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

### Unbiased Estimator

$$\widehat{\text{HSIC}}_u = \frac{1}{n(n-3)}\left[\text{tr}(\tilde{K}\tilde{L}) + \frac{\mathbf{1}^T\tilde{K}\mathbf{1} \cdot \mathbf{1}^T\tilde{L}\mathbf{1}}{(n-1)(n-2)} - \frac{2}{n-2}\mathbf{1}^T\tilde{K}\tilde{L}\mathbf{1}\right]$$

where $\tilde{K}, \tilde{L}$ have zeroed diagonals.

---

## Connection to CKA

### CKA Definition

**Centered Kernel Alignment** is normalized HSIC:

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}$$

### Interpretation

- HSIC: Unnormalized dependence measure
- CKA: Normalized to [0, 1] (or [-1, 1] for some kernels)

### Linear Kernel Special Case

With linear kernel $k(x, y) = x^T y$:

$$\text{CKA}_{linear}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \cdot \|Y^T Y\|_F}$$

This equals the RV coefficient from multivariate statistics.

---

## Algorithm

```python
def hsic(K: Array, L: Array, unbiased: bool = True) -> float:
    """
    Compute HSIC between two kernel matrices.

    Args:
        K: Kernel matrix for X [n, n]
        L: Kernel matrix for Y [n, n]
        unbiased: Use unbiased estimator

    Returns:
        HSIC value
    """
    n = K.shape[0]

    if unbiased:
        # Zero diagonals
        K_tilde = K - diag(diag(K))
        L_tilde = L - diag(diag(L))

        # Unbiased estimator
        term1 = trace(K_tilde @ L_tilde)
        term2 = (sum(K_tilde) * sum(L_tilde)) / ((n-1) * (n-2))
        term3 = (2 / (n-2)) * sum(K_tilde @ L_tilde)

        return (term1 + term2 - term3) / (n * (n-3))
    else:
        # Biased estimator with centering
        H = eye(n) - ones((n, n)) / n
        return trace(K @ H @ L @ H) / (n-1)**2


def cka(X: Array, Y: Array, kernel: str = "linear") -> float:
    """
    Compute CKA similarity.
    """
    # Compute kernel matrices
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)

    # Normalize HSIC
    return hsic(K, L) / sqrt(hsic(K, K) * hsic(L, L))
```

---

## Kernel Choices

### Linear Kernel

$$k(x, y) = x^T y$$

- Captures linear relationships
- Computationally efficient
- Most common for CKA

### RBF (Gaussian) Kernel

$$k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

- Captures nonlinear relationships
- Characteristic kernel (HSIC=0 ⟺ independence)
- Sensitive to bandwidth $\sigma$

### Polynomial Kernel

$$k(x, y) = (x^T y + c)^d$$

- Captures interactions up to degree $d$
- $d=1, c=0$: linear kernel

---

## Statistical Properties

### Independence Testing

HSIC can test the null hypothesis $H_0: X \perp Y$:

1. Compute $\widehat{\text{HSIC}}$
2. Compare to null distribution (permutation or asymptotic)
3. Reject if $\widehat{\text{HSIC}} > \text{threshold}$

### Asymptotic Distribution

Under $H_0: X \perp Y$:
$$n \cdot \widehat{\text{HSIC}} \xrightarrow{d} \sum_{i} \lambda_i z_i^2$$

where $z_i \sim N(0,1)$ and $\lambda_i$ are eigenvalues of the centered kernel.

---

## Applications Beyond CKA

### Feature Selection

HSIC measures dependence between features and labels:
$$\text{score}(f) = \text{HSIC}(f(X), Y)$$

Select features maximizing HSIC with target.

### Self-Supervised Learning

Barlow Twins and related methods minimize redundancy via HSIC-like objectives:
- Maximize HSIC between augmented views
- Minimize HSIC between embedding dimensions

### Domain Adaptation

HSIC measures domain shift:
$$\text{shift} = \text{HSIC}(\text{source features}, \text{domain indicator})$$

---

## Code Implementation

HSIC is implemented as part of the CKA module (CKA = normalized HSIC).

**Primary Location**: [`src/modelcypher/core/domain/geometry/cka.py`](../../../../src/modelcypher/core/domain/geometry/cka.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `CKAResult` | 68 | Result dataclass (includes HSIC internally) |
| `compute_cka()` | 270 | CKA via normalized HSIC |
| `compute_cka_from_grams()` | 548 | Direct Gram matrix version |

The HSIC computation is embedded within the CKA functions (centering, trace product, normalization).

---

## Bias Correction (2024-2025)

### The Problem (Murphy et al., 2024)

Standard HSIC/CKA estimators have bias that depends on:
- Sample size
- Feature dimensionality
- Kernel choice

This can give misleading similarity scores.

### Solutions

1. **Unbiased HSIC estimator** (Song et al., 2012)
2. **Debiased CKA** (Murphy et al., 2024)
3. **Feature correction** (Chun et al., 2025)

See [centered_kernel_alignment.md](centered_kernel_alignment.md) for details.

---

## Citations

### Foundational

1. **Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B.** (2005). "Measuring Statistical Dependence with Hilbert-Schmidt Norms." *ALT 2005*.
   - *Original HSIC definition*

2. **Gretton, A., Fukumizu, K., Teo, C.H., Song, L., Schölkopf, B., & Smola, A.** (2008). "A Kernel Statistical Test of Independence." *NeurIPS 2008*.
   - *HSIC for independence testing*

3. **Song, L., Smola, A., Gretton, A., Bedo, J., & Borgwardt, K.** (2012). "Feature Selection via Dependence Maximization." *JMLR*, 13, 1393-1434.
   - *HSIC for feature selection*

### Neural Network Applications

4. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.** (2019). "Similarity of Neural Network Representations Revisited." *ICML 2019*.
   arXiv: 1905.00414
   - *CKA (normalized HSIC) for neural networks*

5. **Nguyen, T., Raghu, M., & Kornblith, S.** (2021). "Do Wide and Deep Networks Learn the Same Things?" *ICLR 2021*.
   - *CKA applications*

### 2024-2025 Advances

6. **Murphy, D., et al.** (2024). "Debiased Similarity Measures for Neural Network Representations."
   - *Bias correction for HSIC/CKA*

7. **Podsiadly, M., et al.** (2025). "HSIC-based objectives for self-supervised learning."
   - *Modern SSL applications*

8. **Nature Scientific Reports** (2022). "A fast kernel independence test for cluster-correlated data."
   DOI: 10.1038/s41598-022-26278-9
   - *HSIC for correlated data*

9. **arXiv 2508.21815** (2025). "Achieving Hilbert-Schmidt Independence Under Rényi Differential Privacy."
   - *Privacy-preserving HSIC*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA = normalized HSIC
- [gromov_wasserstein.md](gromov_wasserstein.md) - Alternative cross-dimensional metric
- [relative_representations.md](relative_representations.md) - Another invariant similarity

---

*HSIC is the mathematical foundation of representation similarity: it measures all-order statistical dependence in kernel space, enabling dimension-agnostic comparison of neural network representations.*
