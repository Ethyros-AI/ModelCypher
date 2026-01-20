# Geometric Zipper: MLP Block Consistency in Model Merging

> **Status**: ARCHIVED - Historical Reference
> The permute stage and permutation_aligner.py have been removed from the codebase
> as of 2026-01. This document is retained for research context only.
> Current merge uses GRAM_TRANSPORT projection (see `merge/stages/transplant.py`).

## Overview

The "geometric zipper" was a rule that each MLP block applies a consistent permutation across its up/gate/down projections, preserving functional equivalence while re-basing neuron orderings.

This document described a former implementation based on the Git Re-Basin algorithm. The permutation alignment stage was removed in favor of GRAM_TRANSPORT projection in the transplant stage.

---

## Theoretical Foundation

### Git Re-Basin (Ainsworth et al., 2022)

**Paper**: [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)

**Core Thesis**: Neural network loss landscapes contain nearly a single basin after accounting for permutation symmetries. Different trained models can be aligned by finding the right permutation of neurons.

**Key Insight**: There are more prediction-preserving permutations than atoms in the observable universe. Finding the right permutation enables zero-barrier linear interpolation between models.

### Permutation Symmetry

For a weight matrix W with shape [out_dim, in_dim]:
- **Output permutation**: P @ W (P is [out_dim, out_dim])
- **Input permutation**: W @ P^T (P is [in_dim, in_dim])

The zipper constraint ensures consistency within the MLP triplet:
```
W_up' = S @ P @ W_up
W_gate' = S @ P @ W_gate
W_down' = W_down @ P^T @ S
```

This maintains functional equivalence: f(x; Θ) = f(x; Θ') for all inputs.

---

## Weight Matching Algorithm

### Anchor-Guided Signatures

Each model provides its own anchor embeddings (from `embed_tokens` / `wte`).
For each MLP block, signatures are computed by projecting weights through the anchors
(or through per-layer anchor activations when available).

### Similarity Matrix and Assignment

Similarity is computed as geodesic cosine between signature sets. The assignment is
solved with the internal Hungarian implementation on the Backend (no NumPy/SciPy).

```python
similarity = geodesic_cosine_between_sets(source_signatures, target_signatures, backend)
cost = backend.max(backend.abs(similarity)) - backend.abs(similarity)
assignment = hungarian_assignment(cost, backend)
```

### Permutation + Sign Construction

The assignment is materialized as either a dense permutation matrix or a sparse
index list (for large intermediate dimensions). Signed similarities determine
per-neuron sign flips, which are applied consistently across the MLP triplet.

---

## Implementation in ModelCypher

### Location

`src/modelcypher/core/use_cases/merge/stages/permute.py`
`src/modelcypher/core/domain/geometry/permutation_aligner.py`

### Key Methods

1. **`PermutationAligner.rebasin_mlp_with_activations(...)`**
   - Aligns each MLP block (up/gate/down) using anchor-guided signatures
   - Applies permutation + sign correction, returns mean match quality

2. **`PermutationAligner.align_via_anchor_activations(...)`**
   - Computes assignment from anchor-projected signatures using Hungarian matching

3. **`stage_permute(...)`**
   - Selects embedding anchors, enforces exact kernel alignment (CKA=1), then runs MLP re-basin

### Zipper Flow in Merge Loop

```
1. Extract source/target embedding anchors (embed_tokens / wte).
2. If anchor CKA < 1 - eps, compute a polar-decomposition rotation and apply it
   to all source weights operating on the hidden dimension.
3. For each MLP block (up/gate/down):
   - Build signatures via anchors (or per-layer anchor activations).
   - Compute assignment + sign flips via Hungarian matching.
   - Apply P and signs to up/gate rows; apply P^T and signs to down columns.
4. Pass aligned weights forward to the transplant stage (no blending here).
```

### Configuration

Permutation alignment runs by default; there is no configuration toggle.

---

## Mathematical Properties

### Permutation Matrix Properties

- **Orthogonal**: P @ P^T = I
- **Inverse equals transpose**: P^{-1} = P^T
- **Determinant**: det(P) = ±1
- **Composition**: P1 @ P2 is also a permutation

### Why Permutations Over Rotations

1. **Exact**: No numerical error accumulates
2. **Discrete**: Maps neurons 1:1 (interpretable)
3. **Composable**: Chain permutations cleanly
4. **Efficient**: Hungarian is O(n³) on intermediate dims; full rotations require dense SVDs

### High-Dimensional Geometry Considerations

In high-dimensional spaces (hidden_dim ~ 4096):
- Random vectors are nearly orthogonal
- Small misalignments scramble semantic content
- Low-rank approximations miss most of the "volume"

The permutation-based zipper uses the **full intermediate dimension** without low-rank shortcuts.
Rotations are only used to bring embedding anchors to exact kernel alignment (CKA=1).

---

## Test Coverage

See:

- `tests/test_stage_permute.py::test_stage_permute_aligns_mlp_blocks`
- `tests/test_permutation_aligner.py` (Hungarian assignment + permutation application)
- `tests/test_permutation_aligner_properties.py` (permutation invariants)
- `tests/test_permutation_aligner_advanced.py` (anchor activations + sparse permutations)
- `tests/test_permutation_aligner_mlx.py` (backend-specific validation)

---

## References

1. **[Ainsworth, S.K., Hayase, J., & Srinivasa, S.S. (2023)](../references/arxiv/Ainsworth_2023_Git_ReBasin.pdf)**. Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)

2. **[Singh, S.P., & Jaggi, M. (2020)](../references/arxiv/Singh_2019_Model_Fusion_Optimal_Transport.pdf)**. Model Fusion via Optimal Transport. *NeurIPS 2020*. (Related: soft neuron alignment via OT)

3. **[Ilharco, G., et al. (2023)](../references/arxiv/Ilharco_2023_Task_Arithmetic.pdf)**. Editing Models with Task Arithmetic. *ICLR 2023*. (Task vectors as separable geometric structures)

4. **[Yadav, P., et al. (2023)](../references/arxiv/Yadav_2023_TIES_Merging.pdf)**. TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. (Sign consensus for merge stability)

---

## Future Work

1. **Activation Matching**: Use probe activations instead of weights for matching (data-dependent)
2. **Iterative Refinement**: Coordinate descent over layers (STE matching from Git Re-Basin)
3. **Soft Permutations**: Doubly-stochastic matrices for gradient-based optimization
4. **Cross-Architecture**: Extend to models with different widths via OT-based padding
