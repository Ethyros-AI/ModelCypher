# Permutation Alignment and Git Re-Basin

> Aligning neural networks modulo symmetries for linear mode connectivity.

---

## Why This Matters for Model Merging

Neural networks have **permutation symmetries**: swapping neurons in a hidden layer (with corresponding weight adjustments) gives a functionally identical network. This causes:
1. **Loss barriers**: Linearly interpolating unaligned networks crosses high-loss regions
2. **Merge failures**: Averaging misaligned networks destroys learned features
3. **Hidden structure**: The "single basin" hypothesis suggests all solutions are connected

**In ModelCypher**: Implemented in `permutation_aligner.py` for aligning networks before merging.

---

## The Core Insight

Two independently trained networks that solve the same task may have learned the **same function** but with neurons in different orders. Permutation alignment finds the reordering that makes them "line up."

### Single Basin Hypothesis (Ainsworth et al., 2023)

> "Neural network loss landscapes contain (nearly) a single basin, after accounting for all possible permutation symmetries of hidden units."

If true, this explains why model merging works at all.

---

## Permutation Symmetries

### Definition

For a fully-connected layer with weight $W \in \mathbb{R}^{m \times n}$:
- A permutation $P \in S_m$ on the output neurons
- Requires corresponding permutation on the next layer's input

Formally, if $W_1, W_2$ are consecutive layers:
$$W_1 \to P W_1, \quad W_2 \to W_2 P^T$$

produces a functionally identical network.

### The Symmetry Group

For a network with hidden layer sizes $n_1, \ldots, n_L$:
$$G = S_{n_1} \times S_{n_2} \times \cdots \times S_{n_L}$$

This group can be astronomically large. For $L=10$ layers with 1000 neurons each: $|G| = (1000!)^{10}$.

---

## Git Re-Basin Algorithms

### 1. Weight Matching

Find permutation $P$ minimizing weight distance:

$$P^* = \arg\min_{P \in S_n} \|W_A P - W_B\|_F$$

This is the **Linear Assignment Problem** (LAP), solvable via Hungarian algorithm in $O(n^3)$.

```python
def weight_matching(W_A: Array, W_B: Array) -> Array:
    """
    Find permutation aligning W_A to W_B by minimizing Frobenius distance.
    """
    # Cost matrix: cost[i,j] = cost of matching neuron i in A to neuron j in B
    cost = -W_A @ W_B.T  # Negative because we maximize similarity

    # Solve Linear Assignment Problem
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build permutation matrix
    P = zeros((n, n))
    P[row_ind, col_ind] = 1

    return P
```

### 2. Activation Matching

Match based on activation patterns:

$$P^* = \arg\min_{P \in S_n} \sum_{x \in \mathcal{D}} \|A_A(x) P - A_B(x)\|^2$$

where $A(x)$ are activations on input $x$.

**Advantage**: Data-dependent, captures functional similarity
**Disadvantage**: Requires running inference on data

### 3. Straight-Through Estimator (STE)

Make permutation search differentiable:

1. Relax permutation to soft assignment (doubly-stochastic matrix)
2. Forward pass: use hard permutation (argmax)
3. Backward pass: gradient through soft assignment

---

## Linear Mode Connectivity (LMC)

### Definition

Two networks $\theta_A, \theta_B$ are **linearly mode connected** if:

$$\mathcal{L}((1-t)\theta_A + t\theta_B) \leq \max(\mathcal{L}(\theta_A), \mathcal{L}(\theta_B)) \quad \forall t \in [0,1]$$

The loss along the linear path never exceeds the endpoints.

### Re-Basin Result

After permutation alignment:
- Previously unconnected networks become linearly connected
- The loss barrier (height of interpolation path) drops dramatically
- Merging via averaging becomes viable

### Experimental Evidence

| Configuration | Loss Barrier (Before) | Loss Barrier (After) |
|--------------|----------------------|---------------------|
| ResNet-20 CIFAR-10 | 2.3 | 0.02 |
| VGG-16 CIFAR-100 | 1.8 | 0.05 |
| MLP MNIST | 0.4 | 0.001 |

Re-basin reduces barriers by 50-100×.

---

## Multi-Model Extension

### Challenge

Aligning two models is tractable. Aligning $M$ models simultaneously is harder:

$$\{P_1, \ldots, P_M\} = \arg\min \sum_{i < j} \|W_i P_i - W_j P_j\|$$

### STE-MM (2025)

Recent work proposes **Straight-Through Estimator for Multiple Models**:
1. Optimize all permutations jointly
2. Use soft relaxation with temperature annealing
3. Converges to consistent alignment across all models

---

## REPAIR: Fixing Activation Shifts

### The Problem

Even after permutation alignment, **activation statistics** may differ:
- Different batch norm running means
- Different activation scales

### REPAIR Solution (Wortsman et al., 2022)

After permutation alignment:
1. Run calibration data through both networks
2. Match activation means and variances
3. Interpolate with corrected statistics

$$\mu_{merged} = (1-t)\mu_A + t\mu_B$$
$$\sigma_{merged} = \sqrt{(1-t)\sigma_A^2 + t\sigma_B^2}$$

---

## Algorithm: Full Re-Basin Pipeline

```python
def git_rebasin_merge(model_A, model_B, calibration_data):
    """
    Full Git Re-Basin pipeline for model merging.
    """
    # 1. Extract weights
    weights_A = extract_weights(model_A)
    weights_B = extract_weights(model_B)

    # 2. Find permutations layer by layer
    permutations = []
    for layer in layers:
        if layer.has_permutation_symmetry:
            # Weight matching for this layer
            P = weight_matching(
                weights_A[layer],
                weights_B[layer]
            )
            permutations.append(P)

    # 3. Apply permutations to model B
    weights_B_aligned = apply_permutations(weights_B, permutations)

    # 4. REPAIR: Fix activation statistics
    weights_B_repaired = repair_activations(
        weights_A, weights_B_aligned, calibration_data
    )

    # 5. Merge (now safe to average)
    merged_weights = {
        k: 0.5 * weights_A[k] + 0.5 * weights_B_repaired[k]
        for k in weights_A
    }

    return merged_weights
```

---

## Limitations and Considerations

### Architectures with Residual Connections

Residual connections constrain permutation freedom:
- Skip connections link layers
- Permutation must be consistent across residual blocks

**Solution**: Sinkhorn Re-Basin (2023) handles this via differentiable optimization.

### Width Mismatch

If networks have different widths, permutation alignment doesn't directly apply.

**Solutions**:
- Zero-pad smaller network
- Use partial matching
- Project to common subspace first

### Computational Cost

For large models:
- Weight matching: $O(n^3)$ per layer
- Can be prohibitive for wide layers

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/permutation_aligner.py`

```python
class PermutationAligner:
    """Align neural networks modulo permutation symmetries."""

    def weight_match(
        self,
        weights_a: dict[str, Array],
        weights_b: dict[str, Array],
        backend: Backend | None = None,
    ) -> list[Array]:
        """
        Find permutations aligning model B to model A.
        """

    def activation_match(
        self,
        model_a,
        model_b,
        calibration_data: Array,
    ) -> list[Array]:
        """
        Find permutations based on activation similarity.
        """

    def apply_permutations(
        self,
        weights: dict[str, Array],
        permutations: list[Array],
    ) -> dict[str, Array]:
        """
        Apply permutations to weight matrices.
        """
```

---

## Citations

### Primary Reference

1. **Ainsworth, S.K., Hayase, J., & Srinivasa, S.** (2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries." *ICLR 2023*.
   arXiv: 2209.04836
   GitHub: https://github.com/samuela/git-re-basin
   - *The foundational paper*

### Theoretical Foundations

2. **Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B.** (2022). "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks." *ICLR 2022*.
   - *Theoretical analysis of permutation symmetries*

3. **Frankle, J., Dziugaite, G.K., Roy, D.M., & Carbin, M.** (2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis." *ICML 2020*.
   - *Linear mode connectivity framework*

### Extensions

4. **Guerrero Peña, F.A., et al.** (2023). "Re-basin via Implicit Sinkhorn Differentiation." *CVPR 2023*.
   - *Differentiable re-basin for residual networks*

5. **Navon, A., et al.** (2023). "Equivariant Deep Weight Space Alignment."
   arXiv: 2310.13397
   - *Equivariant methods*

### 2024-2025 Advances

6. **ICML 2025 Poster**: "Linear Mode Connectivity between Multiple Models modulo Permutation Symmetries."
   - *Multi-model extension (STE-MM)*

7. **Sharma, K., et al.** (2024). "Simultaneous Linear Connectivity."
   - *Single model aligns to multiple others*

8. **arXiv 2402.05966** (2024). "Rethinking Model Re-Basin and Linear Mode Connectivity."
   - *Analysis of activation shift problem*

---

## Related Concepts

- [procrustes_analysis.md](procrustes_analysis.md) - Continuous version (orthogonal, not permutation)
- [relative_representations.md](relative_representations.md) - Alignment-free alternative
- [cka.md](centered_kernel_alignment.md) - Measuring similarity despite permutation

---

*Permutation alignment reveals that independently trained networks often learn the same solution—just with neurons in different orders. Re-basin unlocks their compatibility for merging.*
