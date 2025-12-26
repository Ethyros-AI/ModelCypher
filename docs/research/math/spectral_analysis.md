# Spectral Analysis of Neural Network Weights

> Eigenvalue distributions reveal model structure and training dynamics.

---

## Why This Matters for Model Merging

The eigenvalue spectrum of weight matrices encodes:
1. **Effective rank**: How many dimensions are actually used
2. **Training stage**: Spectra evolve characteristically during training
3. **Merge compatibility**: Similar spectra suggest compatible representations

**In ModelCypher**: Implemented in `spectral_analysis.py` for weight matrix analysis.

---

## The Core Insight

Random matrices have well-characterized spectra (Marchenko-Pastur law). Training **systematically departs** from this:
- Develops "bulk + tail" structure
- Tail eigenvalues correlate with learned features
- Spectral properties predict generalization

---

## Random Matrix Theory Background

### Marchenko-Pastur Law

For a random matrix $X \in \mathbb{R}^{n \times p}$ with i.i.d. entries, the eigenvalue distribution of $\frac{1}{n}X^TX$ converges to:

$$\rho_{MP}(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi \gamma \lambda}$$

for $\lambda \in [\lambda_-, \lambda_+]$, where:
- $\gamma = p/n$ (aspect ratio)
- $\lambda_\pm = (1 \pm \sqrt{\gamma})^2$

### At Initialization

Neural network weights at initialization follow MP closely:
- Bulk of eigenvalues within MP bounds
- Edge statistics follow Tracy-Widom distribution

### After Training

Training induces characteristic deviations:
- **Bulk**: Still approximately MP
- **Tail**: Heavy-tailed distribution of outlier eigenvalues
- Tail eigenvalues encode task-relevant information

---

## Spectral Signatures of Training

### Bulk + Tail Structure (Martin & Mahoney, 2021)

$$\rho(\lambda) = (1-\alpha) \rho_{MP}(\lambda) + \alpha \rho_{tail}(\lambda)$$

where $\rho_{tail}$ is typically power-law:

$$\rho_{tail}(\lambda) \propto \lambda^{-\mu}$$

The exponent $\mu$ correlates with:
- Smaller $\mu$: More heavy-tailed → better generalization
- Larger $\mu$: Lighter tail → possible overfitting

### Effective Rank

The **effective rank** (Roy & Vetterli, 2007):

$$\text{erank}(W) = \exp\left(-\sum_i \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$

where $\tilde{\sigma}_i = \sigma_i / \sum_j \sigma_j$ are normalized singular values.

---

## Spectral Analysis Algorithm

```python
def spectral_analysis(W: Array) -> SpectralSignature:
    """
    Analyze eigenvalue distribution of weight matrix.

    Args:
        W: Weight matrix [d_out, d_in]

    Returns:
        SpectralSignature with eigenvalues, effective rank,
        bulk/tail decomposition, and power-law exponent
    """
    # Singular value decomposition
    U, S, Vh = svd(W)

    # Eigenvalues of W^T W
    eigenvalues = S ** 2

    # Effective rank
    S_norm = S / sum(S)
    effective_rank = exp(-sum(S_norm * log(S_norm + eps)))

    # Fit bulk + tail model
    bulk_edge = estimate_mp_edge(W.shape)
    tail_eigenvalues = eigenvalues[eigenvalues > bulk_edge]
    bulk_eigenvalues = eigenvalues[eigenvalues <= bulk_edge]

    # Power-law exponent for tail
    alpha = fit_power_law(tail_eigenvalues)

    return SpectralSignature(
        eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        bulk_edge=bulk_edge,
        tail_fraction=len(tail_eigenvalues) / len(eigenvalues),
        power_law_exponent=alpha,
    )
```

---

## WeightWatcher Methodology

Martin & Mahoney's **WeightWatcher** analyzes pre-trained models:

### Key Metrics

1. **Alpha ($\alpha$)**: Power-law exponent of eigenvalue tail
   - $\alpha < 2$: Heavy-tailed (good)
   - $\alpha > 4$: Thin-tailed (concerning)

2. **Lambda ($\lambda$)**: Spectral norm / maximum eigenvalue

3. **Log Spectral Norm**: $\log(\lambda_{max})$

### Quality Prediction

Without any training data:
$$\text{Test Accuracy} \approx f(\bar{\alpha}, \bar{\lambda})$$

where $\bar{\alpha}$ is the average power-law exponent across layers.

---

## Implications for Model Merging

### Spectral Compatibility

Models with similar spectral signatures:
- Have similar effective dimensionality
- Use weight space similarly
- Merge with less interference

### Spectral-Aware Merging

```python
def spectral_aware_merge(models: list, weights: list = None):
    """
    Merge models with spectral analysis guidance.
    """
    for layer_name in layer_names:
        # Analyze each model's spectrum
        spectra = [spectral_analysis(m[layer_name]) for m in models]

        # Check compatibility
        rank_variance = var([s.effective_rank for s in spectra])
        if rank_variance > threshold:
            # Use projection to shared subspace
            merged = project_to_shared_subspace(...)
        else:
            # Standard merge is safe
            merged = weighted_average(...)
```

### Truncation Strategies (STAR, 2025)

**STAR: Spectral Truncation and Rescale**:
1. Compute SVD of task vectors
2. Truncate to top-$k$ singular values
3. Rescale remaining components
4. Merge in reduced space

---

## Layer-wise Spectral Patterns

### Empirical Observations

| Layer Position | Typical $\alpha$ | Effective Rank | Interpretation |
|---------------|-----------------|----------------|----------------|
| Early | 2.5-3.5 | Higher | Feature extraction |
| Middle | 2.0-3.0 | Variable | Representation learning |
| Late | 1.5-2.5 | Lower | Task-specific |

### Training Dynamics

Spectra evolve during training:
1. **Early**: Near MP (random initialization)
2. **Middle**: Tail develops, bulk shrinks
3. **Late**: Stable bulk + heavy tail

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/spectral_analysis.py`](../../../../src/modelcypher/core/domain/geometry/spectral_analysis.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `SpectralMetrics` | 62 | Core metrics dataclass (eigenvalues, effective rank, etc.) |
| `SpectralConfig` | 95 | Configuration for spectral analysis |
| `spectral_summary()` | 363 | Aggregate statistics across layers |

---

## Citations

### Random Matrix Theory Foundations

1. **Marčenko, V.A., & Pastur, L.A.** (1967). "Distribution of eigenvalues for some sets of random matrices." *Mathematics of the USSR-Sbornik*, 1(4), 457-483.
   - *Marchenko-Pastur law*

2. **Tracy, C.A., & Widom, H.** (1996). "On orthogonal and symplectic matrix ensembles." *Communications in Mathematical Physics*, 177(3), 727-754.
   - *Edge statistics*

3. **Wigner, E.P.** (1955). "Characteristic vectors of bordered matrices with infinite dimensions." *Annals of Mathematics*, 62, 548-564.
   - *Semicircle law*

### Neural Network Applications

4. **Martin, C.H., & Mahoney, M.W.** (2021). "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning." *JMLR*, 22(165), 1-73.
   - *WeightWatcher methodology*

5. **Saxe, A.M., McClelland, J.L., & Ganguli, S.** (2014). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." *ICLR 2014*.
   - *Spectral dynamics during training*

6. **Papyan, V.** (2020). "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra." *JMLR*, 21(252), 1-64.
   - *Class structure in spectra*

### 2024-2025 Advances

7. **STAR** (2025). "Spectral Truncation and Rescale for Model Merging." *NAACL 2025*.
   - *Spectral methods for merging*

8. **From SGD to Spectra** (2025). "A Theory of Neural Network Weight Dynamics." *arXiv:2507.12709*.
   - *Spectral SDE framework*

9. **Mahoney, M.W.** (2025). "Random Matrix Theory and Modern Machine Learning." *UC Berkeley Lecture Notes*.
   - *Recent comprehensive treatment*

10. **Nature Communications** (2025). "SPectral ARchiteCture Search for neural network models."
    DOI: 10.1038/s44387-025-00039-1
    - *Spectral methods for architecture search*

---

## Related Concepts

- [task_singular_vectors.md](task_singular_vectors.md) - SVD for task-specific components
- [intrinsic_dimension.md](intrinsic_dimension.md) - Related to effective rank
- [fisher_information.md](fisher_information.md) - Fisher eigenspectrum

---

*The eigenvalue spectrum of weight matrices reveals the learned structure: random at initialization, structured after training, with heavy tails encoding task-relevant features.*
