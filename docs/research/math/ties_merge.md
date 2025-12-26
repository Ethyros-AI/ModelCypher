# TIES-Merging: Trim, Elect Sign, and Merge

> Resolving parameter interference through sign consensus.

---

## Why This Matters for Model Merging

When merging multiple fine-tuned models, parameters often **conflict**:
- Same parameter, opposite signs → cancellation
- Redundant parameters → noise amplification

TIES-Merging addresses both through systematic conflict resolution.

**In ModelCypher**: Conceptually informs our merge strategies; combined with geometric methods.

---

## The Core Insight

Task vectors (fine-tuning deltas) have two types of interference:
1. **Redundant parameters**: Small-magnitude changes that add noise
2. **Sign conflicts**: Parameters that moved in opposite directions

TIES resolves both: trim the redundant, elect the dominant sign.

---

## Formal Definition

### Task Vectors

Given pre-trained model $\theta_{pre}$ and fine-tuned model $\theta_t$:

$$\tau_t = \theta_t - \theta_{pre}$$

This is the **task vector**—the direction of fine-tuning.

### TIES-Merging Algorithm (Yadav et al., 2023)

**Input**: Task vectors $\{\tau_1, \ldots, \tau_T\}$, density $k\%$

**Step 1: TRIM** - Remove low-magnitude parameters

For each task vector $\tau_t$:
$$\hat{\tau}_t = \text{TopK}(\tau_t, k\%)$$

Keep only the top $k\%$ by magnitude; set rest to zero.

**Step 2: ELECT** - Resolve sign conflicts

For each parameter position $j$:
$$\gamma_j = \text{sign}\left(\sum_t \hat{\tau}_{t,j}\right)$$

The elected sign is determined by the **aggregate direction**.

**Step 3: MERGE** - Disjoint mean with sign agreement

For each parameter $j$:
$$\theta^*_j = \theta_{pre,j} + \gamma_j \cdot \frac{\sum_t |\hat{\tau}_{t,j}| \cdot \mathbb{1}[\text{sign}(\hat{\tau}_{t,j}) = \gamma_j]}{|\{t : \text{sign}(\hat{\tau}_{t,j}) = \gamma_j\}|}$$

Only average parameters that agree with the elected sign.

---

## Algorithm Implementation

```python
def ties_merge(
    pretrained: dict[str, Array],
    finetuned_models: list[dict[str, Array]],
    density: float = 0.2,  # Keep top 20%
    lambda_weight: float = 1.0,
) -> dict[str, Array]:
    """
    TIES-Merging: Trim, Elect Sign, and Merge.

    Args:
        pretrained: Pre-trained model weights
        finetuned_models: List of fine-tuned model weights
        density: Fraction of parameters to keep (0.2 = top 20%)
        lambda_weight: Scaling factor for merged task vector

    Returns:
        Merged model weights
    """
    result = {}

    for name in pretrained.keys():
        # Compute task vectors
        task_vectors = [m[name] - pretrained[name] for m in finetuned_models]

        # TRIM: Keep top-k% by magnitude
        trimmed = []
        for tv in task_vectors:
            threshold = quantile(abs(tv), 1 - density)
            mask = abs(tv) >= threshold
            trimmed.append(tv * mask)

        # Stack for vectorized operations
        stacked = stack(trimmed)  # [T, *shape]

        # ELECT: Determine sign by sum of trimmed vectors
        sign_sum = sum(stacked, axis=0)
        elected_sign = sign(sign_sum)

        # MERGE: Average only sign-agreeing parameters
        # Mask for sign agreement
        sign_agree = (sign(stacked) == elected_sign) | (stacked == 0)

        # Disjoint mean: average over agreeing, count agreeing
        agreeing_values = where(sign_agree, abs(stacked), 0)
        agreeing_count = sum(sign_agree & (stacked != 0), axis=0)
        agreeing_count = maximum(agreeing_count, 1)  # Avoid division by zero

        merged_magnitude = sum(agreeing_values, axis=0) / agreeing_count
        merged_tv = elected_sign * merged_magnitude

        # Reconstruct
        result[name] = pretrained[name] + lambda_weight * merged_tv

    return result
```

---

## Experimental Results

### NeurIPS 2023 Benchmarks

| Method | T5-Base Avg | T5-Large Avg | ViT-B/32 Avg |
|--------|-------------|--------------|--------------|
| Task Arithmetic | 65.4% | 70.2% | 72.8% |
| Fisher Merging | 67.1% | 71.8% | 74.3% |
| **TIES-Merging** | **69.2%** | **74.5%** | **76.1%** |

TIES consistently outperforms prior methods across modalities.

### Ablation: Component Importance

| Configuration | Performance |
|---------------|-------------|
| Full TIES | 74.5% |
| − Trim | 72.1% |
| − Elect | 71.8% |
| − Disjoint Mean | 73.2% |

All three components contribute; Elect and Trim are most critical.

---

## The Sign Conflict Problem

### Why Signs Matter

Consider parameter $j$ with:
- Model A: $\tau_{A,j} = +0.5$
- Model B: $\tau_{B,j} = -0.5$

**Simple average**: $\frac{+0.5 + (-0.5)}{2} = 0$ → Information destroyed!

**TIES**: Elects dominant sign, preserves useful information.

### Geometric Interpretation

Sign conflicts indicate:
- Different models learned **opposite strategies**
- Simply averaging creates **null effect**
- Need to choose which strategy to preserve

---

## Combining TIES with Other Methods

### TIES + DARE

```python
def dare_ties_merge(models, pretrained, dare_rate=0.9, ties_density=0.2):
    # First: DARE sparsification
    dare_models = [dare_sparsify(pretrained, m, dare_rate) for m in models]

    # Then: TIES merging
    return ties_merge(pretrained, dare_models, density=ties_density)
```

### TIES + Fisher Weighting

```python
def fisher_ties_merge(models, pretrained, fisher_matrices, density=0.2):
    # Use Fisher to weight the sign election
    weighted_task_vectors = [
        tv * fisher for tv, fisher in zip(task_vectors, fisher_matrices)
    ]
    # Apply TIES with weighted vectors
    ...
```

---

## Theoretical Justification

### Interference Reduction

Let $\tau^{(i)}_j$ be parameter $j$ for task $i$. Interference occurs when:

$$\sum_i \tau^{(i)}_j \approx 0 \text{ but } \sum_i |\tau^{(i)}_j| \gg 0$$

TIES reduces interference by:
1. **Trimming**: Removes parameters contributing noise
2. **Electing**: Resolves conflicting directions
3. **Disjoint Mean**: Only combines agreeing updates

### Connection to Voting

Elect Sign is a **majority vote** on the direction of change:
- Each model "votes" for positive or negative
- Majority wins
- Minority discarded (not averaged in)

---

## ModelCypher Integration

While TIES is a parameter-space method, it connects to our geometric framework:

### Geometric View of Task Vectors

Task vectors live in parameter space, but:
- Trimming = projection to sparse subspace
- Sign election = angular clustering
- Disjoint mean = magnitude-weighted averaging

### Combining with Geometry

```python
def geometric_ties_merge(models, pretrained, activations, density=0.2):
    # 1. TIES for parameter-space merging
    ties_merged = ties_merge(pretrained, models, density)

    # 2. Verify with geometric metrics
    cka = compute_cka(
        get_activations(ties_merged, activations),
        [get_activations(m, activations) for m in models]
    )

    # 3. Adjust if geometric similarity is low
    if mean(cka) < threshold:
        # Fall back to geometry-guided merging
        ...
```

---

## Citations

### Primary Reference

1. **Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M.** (2023). "TIES-Merging: Resolving Interference When Merging Models." *NeurIPS 2023*.
   arXiv: 2306.01708
   OpenReview: https://openreview.net/forum?id=xtaX3WyCj1
   - *The foundational TIES paper*

### Task Arithmetic Foundation

2. **Ilharco, G., Ribeiro, M.T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A.** (2023). "Editing Models with Task Arithmetic." *ICLR 2023*.
   arXiv: 2212.04089
   - *Task vector foundation*

### Extensions and Variants

3. **TIES-SVD** (Stoica et al., 2024). "Integrating SVD for refined fusion."
   - *SVD extension of TIES*

4. **CAT Merging** (2025). "Conflict-Aware Task Merging."
   - *Improved conflict detection*

5. **Wang, X., et al.** (2024). "TALL Masks / Consensus: Deactivating irrelevant parameters through binary masking."
   - *Binary masking alternative*

### Applications

6. **SemEval 2025**: "Unlearning via Model Merging using TIES."
   - *TIES for model unlearning*

7. **NVIDIA Developer Blog** (2024). "An Introduction to Model Merging for LLMs."
   - *Practical TIES guide*

---

## Related Concepts

- [dare_sparsity.md](dare_sparsity.md) - Often combined with TIES
- [task_singular_vectors.md](task_singular_vectors.md) - SVD-based alternative
- [fisher_information.md](fisher_information.md) - Importance weighting for TIES

---

*TIES-Merging resolves the fundamental conflict problem: when models disagree on direction, elect the consensus and only merge the agreeing parameters.*
