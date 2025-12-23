# Geometric Zipper: Layer-to-Layer Propagation in Model Merging

## Overview

The "geometric zipper" is a layer-propagation mechanism ensuring that transformations applied at layer N are properly compensated at layer N+1, maintaining functional equivalence during model merging.

This document describes the implementation in `unified_geometric_merge.py` based on the Git Re-Basin algorithm.

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

The zipper constraint ensures consistency:
```
W_ℓ' = P @ W_ℓ           (permute output neurons at layer ℓ)
W_{ℓ+1}' = W_{ℓ+1} @ P^T  (compensate at layer ℓ+1 inputs)
```

This maintains functional equivalence: f(x; Θ) = f(x; Θ') for all inputs.

---

## Weight Matching Algorithm

### The Linear Assignment Problem

Given source weights S and target weights T, find permutation π that maximizes:

```
max_π Σ_i ⟨S[i,:], T[π(i),:]⟩
```

This is solved via the **Hungarian algorithm** in O(n³) time.

### Similarity Matrix

```python
S = source_w @ target_w.T  # [n, n]
# S[i,j] measures similarity between source neuron i and target neuron j
```

### Permutation Matrix Construction

```python
from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(-S)  # Minimize negative = maximize

P = np.zeros((n, n))
P[col_ind, row_ind] = 1.0  # P @ source aligns neurons to target
```

---

## Implementation in ModelCypher

### Location

`src/modelcypher/core/use_cases/unified_geometric_merge.py`

### Key Methods

1. **`_compute_weight_matching_permutation(source_w, target_w)`**
   - Computes optimal permutation matrix using LAP
   - Falls back to greedy matching when scipy unavailable
   - Returns P ∈ ℝ^{n×n} permutation matrix

2. **`_compute_full_rank_rotation(source_w, target_w)`**
   - Continuous relaxation via Orthogonal Procrustes
   - R = UV^T where M = target @ source^T = UΣV^T
   - Returns R ∈ O(n) orthogonal matrix

### Zipper Flow in Merge Loop

```
For each weight in model:
  1. Identify layer index and weight type

  2. If RESIDUAL OUTPUT (o_proj, down_proj):
     - Compute P = weight_matching(source, target)
     - Apply: source' = P @ source
     - Store P for layer propagation

  3. If INPUT PROJECTION (q/k/v_proj, gate/up_proj):
     - Retrieve P from previous layer
     - Apply: source' = source @ P^T

  4. Blend transformed source with target
```

### Configuration

```python
@dataclass
class UnifiedMergeConfig:
    enable_zipper: bool = True
    zipper_use_weight_matching: bool = True  # True=permutation, False=rotation
```

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
4. **Efficient**: Hungarian is O(n³), not O(n^3) like full SVD

### High-Dimensional Geometry Considerations

In high-dimensional spaces (hidden_dim ~ 4096):
- Random vectors are nearly orthogonal
- Small misalignments scramble semantic content
- Low-rank approximations miss most of the "volume"

The permutation-based zipper uses the **full hidden dimension**, avoiding the rank mismatch that plagued the earlier spectral approach.

---

## Test Coverage

See `tests/test_unified_geometric_merge.py::TestZipperPropagation`:

- `test_weight_matching_permutation_identity`: Identical matrices → identity P
- `test_weight_matching_permutation_shuffled`: Recovers shuffled neurons
- `test_permutation_is_orthogonal`: Verifies P @ P^T = I
- `test_full_rank_rotation_*`: Tests the continuous relaxation

---

## References

1. **Ainsworth, S.K., Hayase, J., & Srinivasa, S.S. (2023)**. Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)

2. **Singh, S.P., & Jaggi, M. (2020)**. Model Fusion via Optimal Transport. *NeurIPS 2020*. (Related: soft neuron alignment via OT)

3. **Ilharco, G., et al. (2023)**. Editing Models with Task Arithmetic. *ICLR 2023*. (Task vectors as separable geometric structures)

4. **Yadav, P., et al. (2023)**. TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. (Sign consensus for merge stability)

---

## Future Work

1. **Activation Matching**: Use probe activations instead of weights for matching (data-dependent)
2. **Iterative Refinement**: Coordinate descent over layers (STE matching from Git Re-Basin)
3. **Soft Permutations**: Doubly-stochastic matrices for gradient-based optimization
4. **Cross-Architecture**: Extend to models with different widths via OT-based padding
