# DARE: Drop And REscale

> Sparse delta parameters for efficient model merging.

---

## Why This Matters for Model Merging

Fine-tuned models have **highly redundant** delta parameters. DARE exploits this:
1. **90% of deltas can be dropped** without significant performance loss
2. **Reduces interference** between merged models
3. **Enables scaling** to many models simultaneously

**In ModelCypher**: Implemented in `dare_sparsity.py` for sparsification before merging.

---

## The Core Insight

When fine-tuning from a pre-trained model $\theta_{pre}$, the resulting model $\theta_{ft}$ differs by:

$$\Delta\theta = \theta_{ft} - \theta_{pre}$$

Most of these delta parameters are **redundant**—removing them barely affects performance. DARE randomly drops deltas and rescales the rest.

---

## Formal Definition

### DARE Algorithm (Yu et al., 2024)

Given:
- Pre-trained weights: $\theta_{pre}$
- Fine-tuned weights: $\theta_{ft}$
- Drop rate: $p \in [0, 1]$

**Step 1: Compute delta parameters**
$$\Delta\theta = \theta_{ft} - \theta_{pre}$$

**Step 2: Random dropping with mask $M$**
$$M_i \sim \text{Bernoulli}(1 - p)$$
$$\tilde{\Delta\theta} = M \odot \Delta\theta$$

**Step 3: Rescale**
$$\hat{\Delta\theta} = \frac{\tilde{\Delta\theta}}{1 - p}$$

**Step 4: Reconstruct**
$$\theta_{DARE} = \theta_{pre} + \hat{\Delta\theta}$$

### Why Rescaling?

Without rescaling, the expected value of $\tilde{\Delta\theta}$ would be $(1-p) \cdot \Delta\theta$. Rescaling by $\frac{1}{1-p}$ ensures:

$$\mathbb{E}[\hat{\Delta\theta}] = \Delta\theta$$

This preserves the expected effect of the fine-tuning.

---

## Mathematical Justification

### Lottery Ticket Hypothesis Connection

The effectiveness of DARE supports the view that:
- Fine-tuning creates **sparse, important** updates
- Most parameter changes are noise or redundancy
- The "winning tickets" are a small subset of parameters

### Dropout Analogy

DARE is related to dropout:
- Dropout: Random masking during training
- DARE: Random masking of parameter updates post-training

Both exploit network redundancy, but at different stages.

---

## Algorithm

```python
def dare_sparsify(
    pretrained: dict[str, Array],
    finetuned: dict[str, Array],
    drop_rate: float = 0.9,
    seed: int = 42,
) -> dict[str, Array]:
    """
    Apply DARE sparsification to fine-tuned model.

    Args:
        pretrained: Pre-trained model weights
        finetuned: Fine-tuned model weights
        drop_rate: Fraction of deltas to drop (0.9 = drop 90%)
        seed: Random seed for reproducibility

    Returns:
        DARE-sparsified model weights
    """
    random.seed(seed)
    result = {}

    for name in pretrained.keys():
        # Compute delta
        delta = finetuned[name] - pretrained[name]

        # Create random mask
        mask = random.bernoulli(1 - drop_rate, shape=delta.shape)

        # Apply mask and rescale
        sparse_delta = mask * delta / (1 - drop_rate)

        # Reconstruct
        result[name] = pretrained[name] + sparse_delta

    return result
```

---

## Experimental Results

### Drop Rate Tolerance (Yu et al., 2024)

| Drop Rate | MMLU Accuracy | Relative to Full |
|-----------|--------------|------------------|
| 0% (full) | 74.5% | 100% |
| 50% | 74.2% | 99.6% |
| 70% | 73.8% | 99.1% |
| **90%** | **73.1%** | **98.1%** |
| 95% | 70.4% | 94.5% |
| 99% | 58.2% | 78.1% |

**Key finding**: 90% of parameters can be dropped with only ~2% performance loss.

### Merging Improvement

| Merge Method | Without DARE | With DARE |
|--------------|-------------|-----------|
| Task Arithmetic | 71.4% | 73.8% |
| TIES-Merge | 79.6% | 81.2% |

DARE reduces interference, improving merge quality.

---

## Combining DARE with Other Methods

### DARE-TIES

Combine DARE sparsification with TIES sign election:

```python
def dare_ties_merge(models, pretrained, drop_rate=0.9, density=0.2):
    # 1. Apply DARE to each model
    dare_models = [dare_sparsify(pretrained, m, drop_rate) for m in models]

    # 2. Compute task vectors
    task_vectors = [m - pretrained for m in dare_models]

    # 3. Apply TIES (trim, elect, merge)
    merged_tv = ties_merge(task_vectors, density=density)

    # 4. Add back to pretrained
    return pretrained + merged_tv
```

### DARE-SLERP

Apply DARE before spherical interpolation:

```python
def dare_slerp_merge(model_a, model_b, pretrained, drop_rate=0.9, t=0.5):
    # Sparsify both models
    dare_a = dare_sparsify(pretrained, model_a, drop_rate)
    dare_b = dare_sparsify(pretrained, model_b, drop_rate)

    # SLERP interpolation
    return slerp_merge(dare_a, dare_b, t)
```

---

## DAREx: Enhanced DARE (2024)

### Problem with Extreme Drop Rates

At very high drop rates (>95%), DARE degrades significantly due to:
- Variance explosion
- Mean shift

### DAREx Solution

1. **Adjusted rescaling factor**:
   $$\hat{\Delta\theta} = \frac{\tilde{\Delta\theta}}{1 - p + \epsilon}$$

2. **Sparsity constraints**: Encourage structured sparsity

3. **Variance balancing**: Control variance of rescaled deltas

---

## Theoretical Analysis

### Expected Value Preservation

$$\mathbb{E}[\theta_{DARE}] = \theta_{pre} + \mathbb{E}[\hat{\Delta\theta}] = \theta_{pre} + \Delta\theta = \theta_{ft}$$

DARE preserves the expected fine-tuned weights.

### Variance Analysis

$$\text{Var}[\theta_{DARE}] = \frac{p}{1-p} \cdot \text{diag}(\Delta\theta \odot \Delta\theta)$$

Variance increases with drop rate—hence the limit on practical $p$.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/dare_sparsity.py`](../../../../src/modelcypher/core/domain/geometry/dare_sparsity.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `Configuration` | 31 | Config with sparsity threshold, droppable percentile |
| `LayerSparsityMetrics` | 47 | Per-layer sparsity result |
| `SparsityAnalysis` | 71 | Full DARE analysis result |
| `DARESparsityAnalyzer` | 87 | Main analyzer class with `analyze()` method |

**Also in**:
- [`geometry_adapter_service.py:618`](../../../../src/modelcypher/core/use_cases/geometry_adapter_service.py) - `dare_merge_readiness()` helper

---

## Citations

### Primary Reference

1. **Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y.** (2024). "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." *ICML 2024*. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)
   - *Original DARE paper*

### Extensions

2. **Deng, Z., et al.** (2024). "DAREx: Drop And REscale with Extreme Sparsity." [arXiv](https://arxiv.org/search/?query=DAREx+drop+rescale+extreme&searchtype=all)
   - *Improved DARE for extreme drop rates*

3. **Davari, M.-J., & Belilovsky, E.** (2024). "Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks." *ECCV 2024*. [arXiv:2312.06795](https://arxiv.org/abs/2312.06795)
   - *Magnitude-based alternative to random dropping*

### Theoretical Analysis

4. **Wang, X., et al.** (2024). "FREE-Merging: Fourier-based Redundancy Elimination for Efficient Model Merging." [arXiv](https://arxiv.org/search/?query=FREE-Merging+Fourier+redundancy&searchtype=all)
   - *Frequency-domain analysis of redundancy*

### 2025 Applications

5. **NAACL 2025**: "STAR: Spectral Truncation and Rescale for Model Merging." [ACL Anthology](https://aclanthology.org/)
   - *Combines spectral methods with DARE concepts*

6. **Engineering Applications of AI** (2025). "Research on enhancing model performance by merging using DARE." [Elsevier](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
   - *Production applications of DARE*

---

## Related Concepts

- [ties_merge.md](ties_merge.md) - Often combined with DARE
- [task_singular_vectors.md](task_singular_vectors.md) - Alternative sparsification via SVD
- [spectral_analysis.md](spectral_analysis.md) - Understanding why DARE works

---

*DARE reveals that 90% of fine-tuning parameters are redundant—dropping them reduces interference and improves model merging.*
