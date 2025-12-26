# Relative Representations

> Dimension-agnostic transfer via anchor similarities (ICLR 2023).

---

## Why This Matters for Model Merging

Neural networks trained with different seeds, architectures, or data produce **incoherent latent spaces** that cannot be directly compared. Relative representations solve this by:
1. Defining representations relative to **anchor points**
2. Enabling **zero-shot stitching** between models
3. Being **invariant to isometries** (rotations, reflections)

**In ModelCypher**: Implemented in `relative_representation.py` for cross-architecture model comparison and transfer.

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

The **relative representation** of $z \in \mathcal{Z}$ is:

$$\phi_A(z) = \left( \frac{s(z, a_1)}{\|s(z, A)\|}, \ldots, \frac{s(z, a_k)}{\|s(z, A)\|} \right)$$

where $\|s(z, A)\| = \sqrt{\sum_i s(z, a_i)^2}$ normalizes the representation.

### Similarity Functions

**Cosine similarity** (most common):
$$s_{cos}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}$$

**RBF kernel**:
$$s_{rbf}(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

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
   c. Decode: y = D₂(r)
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

---

## Connecting to ModelCypher's Thesis

Relative representations align with our geometric framework:

1. **Gram matrices**: Relative representations are essentially Gram matrices with a subset of points (anchors)
2. **CKA connection**: $\text{CKA}(X, Y) \approx \text{correlation}(\phi_A(X), \phi_A(Y))$
3. **GW transport**: Relative representations can be inputs to GW

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/relative_representation.py`](../../../../src/modelcypher/core/domain/geometry/relative_representation.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `RelativeRepresentation` | 54 | Class encapsulating relative representation computation |
| `compute_relative_representation()` | 134 | Standalone function for computing relative representations |
| `align_relative_representations()` | 170 | Align representations across models |

**Design decisions**:
1. **Multiple similarity functions**: Cosine (default) and RBF
2. **Normalization option**: For downstream compatibility
3. **Geodesic-aware**: Can use geodesic distances for anchor similarities

---

## Applications

### 1. Cross-Architecture Comparison

Compare CNN and Transformer representations:
```python
anchors = get_shared_anchors()
rel_cnn = relative_repr(cnn_features, anchors)
rel_transformer = relative_repr(transformer_features, anchors)
similarity = cosine_sim(rel_cnn, rel_transformer)
```

### 2. Model Stitching

Compose components from different training runs:
```python
encoder = load_encoder(run_1)
decoder = load_decoder(run_2)
anchors = load_shared_anchors()

def stitched_model(x):
    z = encoder(x)
    r = relative_repr(z, anchors)
    return decoder(r)
```

### 3. Cross-Lingual Transfer

Transfer between languages via semantic anchors:
```python
en_anchors = get_english_prototypes()
de_anchors = get_german_prototypes()
aligned = relative_repr(en_embeddings, en_anchors)
```

---

## Citations

### Primary Reference

1. **Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., & Rodolà, E.** (2023). "Relative representations enable zero-shot latent space communication." *ICLR 2023* (Notable Top 5%).
   arXiv: 2209.15430
   OpenReview: https://openreview.net/forum?id=SrC-nwieGJ
   - *The foundational paper*

### Extensions and Applications

2. **Cannistraci, I., et al.** (2023). "Bootstrapping Parallel Anchors for Relative Representations." *ICLR 2023 Tiny Papers*.
   - *Learning anchors without labels*

3. **Jian, Z., et al.** (2023). "Policy Stitching: Learning Transferable Robot Policies." *CoRL 2023*.
   - *Relative representations for robotics*

4. **Ricciardi, A.P., et al.** (2025). "Zero-Shot Stitching in Reinforcement Learning using Relative Representations." *EWRL 2023*.
   arXiv: 2404.12917
   - *RL with relative representations*

5. **Model Stitching Survey** (2025). "Model Stitching in Neural Networks." *Emergent Mind Topic*.
   - *Overview of stitching methods*

### Cross-Lingual

6. **Norelli, A., et al.** (2024). "Model Stitching with Static Word Embeddings for Crosslingual Zero-Shot Transfer." *Insights Workshop 2024*.
   - *Cross-lingual stitching*

### Theoretical Connections

7. **Fumero, M., et al.** (2024). "Connecting Neural Models Latent Geometries with Relative Geodesic Representations." *arXiv*.
   - *Geodesic extension of relative representations*

8. **Moschella, L.** (2023). *Latent Communication in Artificial Neural Networks*. PhD Thesis, Sapienza University of Rome.
   - *Comprehensive theoretical treatment*

---

## Related Concepts

- [centered_kernel_alignment.md](centered_kernel_alignment.md) - CKA relates to relative repr via Gram matrices
- [procrustes_analysis.md](procrustes_analysis.md) - Alternative alignment approach
- [anchor_invariance_analyzer.py](../anchor_invariance.md) - Finding stable anchors

---

*Relative representations solve the problem of incompatible latent spaces by focusing on relationships rather than absolute positions.*
