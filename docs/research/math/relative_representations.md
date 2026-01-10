# Relative Representations

> Dimension-agnostic transfer via anchor similarities (ICLR 2023).

---

## Why This Matters for Model Merging

Neural networks trained with different seeds, architectures, or data produce **incoherent latent spaces** that cannot be directly compared. Relative representations solve this by:
1. Defining representations relative to **anchor points**
2. Enabling **zero-shot stitching** between models
3. Being **invariant to isometries** (rotations, reflections)

**In ModelCypher**: `relative_representation.py` computes anchor embeddings from atlas probes, builds geodesic-cosine relative representations, aligns them with Procrustes, and transfers back to target space via cached pseudo-inverse of anchor similarities.

---

## The Core Insight

Instead of using absolute coordinates:
$$x \in \mathbb{R}^d$$

Use relative coordinates (similarities to anchors):
$$r(x) = [s(x, a_1), s(x, a_2), \ldots, s(x, a_k)] \in \mathbb{R}^k$$

where $s$ is a similarity function (typically cosine) and $\{a_i\}$ are anchor points.

---

## Formal Definition

### Definition (Moschella et al., 2023)

Given:
- A latent space $\mathcal{Z} \subseteq \mathbb{R}^d$
- Anchor set $A = \{a_1, \ldots, a_k\} \subset \mathcal{Z}$
- Similarity function $s: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$

The **relative representation** used in ModelCypher is:

$$\phi_A(z) = \left( s(z, a_1), \ldots, s(z, a_k) \right)$$

Normalized variants appear in the literature, but ModelCypher uses the raw
similarity vector and handles scale during alignment.

### Similarity Function (ModelCypher)

**Geodesic cosine similarity**:
$$s_{cos}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}$$

---

## Invariance Properties

### Theorem 1: Isometry Invariance

For any orthogonal transformation $Q \in O(d)$:
$$\phi_A(Qz) = \phi_{QA}(Qz)$$

If anchors transform with the space, relative representations are preserved.

### Theorem 2: Scale Invariance

For any scalar $\alpha > 0$:
$$\phi_A(\alpha z) = \phi_A(z)$$

(when using cosine similarity)

### Implication

Relative representations are invariant to the nuisance transformations that differ across training runs:
- Random initialization leads to rotated spaces
- Different architectures have different scales
- But relative structure is preserved

---

## Zero-Shot Model Stitching

### The Stitching Problem

Given:
- Encoder $E_1$ trained on task 1
- Decoder $D_2$ trained on task 2

Can we compose $D_2 \circ E_1$ without retraining?

### Relative Representation Solution

```
1. Choose shared anchors A (semantic concepts both models know)
2. For input x:
   a. Encode: z = E₁(x)
   b. Compute relative repr: r = φ_A(z)
   c. (Optional) Align anchor space with paired samples
   d. Project back to target space with a pseudo-inverse of anchor similarities
   e. Decode: y = D₂(projected)
```

**Key insight**: If both models learned similar relative structure (which they do for semantically similar tasks), stitching works.

---

## Anchor Selection

### Requirements

1. **Semantic coverage**: Anchors should span the semantic space
2. **Parallel**: Same semantic meaning across models
3. **Diverse**: Not redundant

### Strategies

1. **Class prototypes**: Use mean embeddings per class
2. **Random sampling**: Works surprisingly well
3. **Bootstrapping**: Learn anchors from unlabeled data (Cannistraci et al., 2023)

**ModelCypher default**: Anchors come from the atlas probe registry via
`compute_anchor_embeddings()`, which averages token embeddings for each probe's
support texts. Custom probe sets can be injected when needed.

---

## Connecting to ModelCypher's Thesis

Relative representations align with our geometric framework:

1. **Gram matrices**: Relative representations are essentially Gram matrices with a subset of points (anchors)
2. **CKA connection**: $\text{CKA}(X, Y) \approx \text{correlation}(\phi_A(X), \phi_A(Y))$
3. **GW transport**: Relative representations can be inputs to GW

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/relative_representation.py`](../../../../src/modelcypher/core/domain/geometry/relative_representation.py)

**Key entry points**:
- `compute_anchor_embeddings()` - build anchor embeddings from atlas probes
- `compute_relative_representation()` - cosine similarities to anchors
- `align_relative_representations()` - Procrustes in anchor space
- `transfer_via_relative_space()` - pseudo-inverse projection to target space
- `cross_dimension_transfer()` - full transfer pipeline

**Design decisions**:
1. **Geodesic cosine only**: No RBF path in core implementation.
2. **No per-sample normalization**: Raw similarity vectors are aligned downstream.
3. **Caching**: Gram and SVD results are cached for repeated projections.
4. **Proper rotations**: Alignment enforces $\det(R)=1$ and reports normalized geodesic error.

---

## Applications

### 1. Cross-Architecture Comparison

Compare CNN and Transformer representations:
```python
anchors, _ = compute_anchor_embeddings(embedding_matrix, tokenizer)
rel_cnn = compute_relative_representation(cnn_features, anchors)
rel_transformer = compute_relative_representation(transformer_features, anchors)
R, error = align_relative_representations(rel_cnn, rel_transformer)
```

### 2. Model Stitching

Compose components from different training runs:
```python
source_anchors, _ = compute_anchor_embeddings(source_embedding, source_tokenizer)
target_anchors, _ = compute_anchor_embeddings(target_embedding, target_tokenizer)
transferred = transfer_via_relative_space(source_hidden, source_anchors, target_anchors)
```

### 3. Cross-Lingual Transfer

Transfer between languages via semantic anchors:
```python
result = cross_dimension_transfer(
    source_hidden,
    source_embedding,
    target_embedding,
    source_tokenizer,
    target_tokenizer,
)
aligned = result.relative_representation
```

---

## Citations

### Primary Reference

1. **[Moschella et al. (2023)](../../references/arxiv/Moschella_2022_Relative_representations_enable_zeroshot_latent_space.pdf)**. "Relative representations enable zero-shot latent space communication." *ICLR 2023* (Notable Top 5%). [arXiv:2209.15430](https://arxiv.org/abs/2209.15430) · [OpenReview](https://openreview.net/forum?id=SrC-nwieGJ)
   - *The foundational paper*

### Extensions and Applications

2. **Cannistraci, I., et al.** (2023). "Bootstrapping Parallel Anchors for Relative Representations." *ICLR 2023 Tiny Papers*. [OpenReview](https://openreview.net/forum?id=xKWWepBdMZ)
   - *Learning anchors without labels*

3. **Jian, Z., et al.** (2023). "Policy Stitching: Learning Transferable Robot Policies." *CoRL 2023*. [OpenReview](https://openreview.net/forum?id=fOqaLJNORCv)
   - *Relative representations for robotics*

4. **[Ricciardi et al. (2024)](../../references/arxiv/Ricciardi_2024_R3L_Relative_Representations_Reinforcement_Learning.pdf)**. "R3L: Relative Representations for Reinforcement Learning." [arXiv:2404.12917](https://arxiv.org/abs/2404.12917)
   - *RL with relative representations*

### Cross-Lingual

5. **Norelli, A., et al.** (2024). "Model Stitching with Static Word Embeddings for Crosslingual Zero-Shot Transfer." *Insights Workshop 2024*. [ACL Anthology](https://aclanthology.org/events/insights-2024/)
   - *Cross-lingual stitching*

### Theoretical Connections

6. **Yu, H., et al.** (2025). "Connecting Neural Models Latent Geometries with Relative Geodesic Representations." [arXiv:2506.01599](https://arxiv.org/abs/2506.01599)
   - *Geodesic extension of relative representations*

7. **Moschella, L.** (2023). *Latent Communication in Artificial Neural Networks*. PhD Thesis, Sapienza University of Rome.
   - *Comprehensive theoretical treatment*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA relates to relative repr via Gram matrices
- [procrustes_analysis.md](procrustes_analysis.md) - Alternative alignment approach
- [`src/modelcypher/core/domain/geometry/anchor_invariance_analyzer.py`](../../../../src/modelcypher/core/domain/geometry/anchor_invariance_analyzer.py) - Finding stable anchors

---

*Relative representations solve the problem of misaligned latent spaces by focusing on relationships rather than absolute positions.*
