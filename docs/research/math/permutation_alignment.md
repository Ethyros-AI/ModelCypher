# Permutation Alignment (Git Re-Basin)

> Aligning neural networks modulo permutation symmetries.

---

## Why This Matters for Model Merging

Neural networks have **permutation symmetries**: reordering hidden units (with
compensating weight changes) preserves the function. Without alignment:
- Unaligned merges destroy learned features
- Loss barriers appear even between equivalent solutions

**In ModelCypher**: Permutation alignment is implemented in
`src/modelcypher/core/domain/geometry/permutation_aligner.py` and is applied
in the merge pipeline before transplant.

---

## Permutation Symmetries

For consecutive layers with weights $W_1, W_2$:

$$W_1' = P W_1, \quad W_2' = W_2 P^T$$

This preserves the overall function when $P$ permutes hidden units.

---

## ModelCypher Alignment Strategy

### Anchor-Guided Matching

`PermutationAligner` builds **signatures** by projecting weights through
anchor embeddings (or per-layer anchor activations), then computes geodesic
cosine similarity between signatures. The optimal assignment is solved via
the Hungarian algorithm.

### Sign Correction

Signed similarities determine per-neuron sign flips. A diagonal sign matrix
is applied consistently across the MLP triplet.

### MLP-Only Re-Basin

Permutation alignment is applied to MLP blocks only:

$$W_{up}' = S P W_{up}$$
$$W_{gate}' = S P W_{gate}$$
$$W_{down}' = W_{down} P^T S$$

Attention weights are not permuted by the generic aligner.

---

## Algorithm Sketch (ModelCypher)

1. Build anchor signatures for source and target.
2. Compute geodesic cosine similarity matrix.
3. Convert similarity to cost and solve Hungarian assignment.
4. Build permutation + sign matrices (dense or sparse).
5. Apply to MLP up/gate rows and down columns.

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/permutation_aligner.py`

**Key entry points**:
- `PermutationAligner.align(...)`
- `PermutationAligner.align_via_anchor_activations(...)`
- `PermutationAligner.rebasin_mlp_with_activations(...)`

**Pipeline usage**:
- `src/modelcypher/core/use_cases/merge/stages/permute.py`

---

## Citations

1. **Ainsworth, S.K., Hayase, J., & Srinivasa, S.S.** (2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries." *ICLR 2023*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)
2. **Entezari, R., et al.** (2022). "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks." *ICLR 2022*. [OpenReview](https://openreview.net/forum?id=dNigytemkL)

---

*Permutation alignment is a prerequisite for geometric merging, not a merge method itself.*
