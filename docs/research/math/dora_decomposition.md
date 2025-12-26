# DoRA: Weight-Decomposed Low-Rank Adaptation

> Separating magnitude and direction for better fine-tuning geometry.

---

## Why This Matters for Model Merging

DoRA reveals that fine-tuning affects **magnitude** and **direction** differently. This decomposition:
1. **Explains LoRA limitations**: LoRA couples magnitude and direction updates
2. **Enables geometric analysis**: Magnitude and direction have different merge behaviors
3. **Improves merge quality**: Separate handling prevents interference

**In ModelCypher**: Implemented in `dora_decomposition.py` for geometric analysis of adapter differences.

---

## The Core Insight

Fine-tuning changes weights in two distinct ways:
- **Magnitude**: How "strong" the transformation is
- **Direction**: What the transformation does

LoRA conflates these. DoRA separates them, matching full fine-tuning's learning pattern.

---

## Formal Definition

### Weight Decomposition (Liu et al., 2024)

Any weight matrix $W$ can be decomposed as:

$$W = m \cdot \frac{V}{\|V\|_c} = m \cdot \hat{V}$$

where:
- $m \in \mathbb{R}^{d_{out}}$ is the **magnitude vector** (per-output-channel norms)
- $V \in \mathbb{R}^{d_{out} \times d_{in}}$ is the **directional matrix**
- $\hat{V} = V / \|V\|_c$ is the **unit-normalized direction** (column-wise)
- $\|V\|_c$ denotes column-wise norms

### DoRA Update Rule

Given pre-trained weight $W_0 = m_0 \cdot \hat{V}_0$:

$$W' = (m_0 + \Delta m) \cdot \frac{V_0 + \Delta V}{\|V_0 + \Delta V\|_c}$$

where:
- $\Delta m$ is learned magnitude adjustment
- $\Delta V = BA$ is LoRA-style low-rank directional update

### Comparison with Standard LoRA

**LoRA**: $W' = W_0 + BA$
- Magnitude and direction change are coupled
- Same update affects both components

**DoRA**: $W' = (m_0 + \Delta m) \cdot \widehat{V_0 + BA}$
- Magnitude trained separately ($\Delta m$)
- Direction updated via low-rank ($BA$)
- Matches full fine-tuning patterns

---

## Geometric Analysis

### Theorem: Negative Correlation in Full Fine-Tuning

Liu et al. (2024) observed that full fine-tuning exhibits:

$$\text{Corr}(\Delta\|W\|, \Delta\theta_W) < 0$$

where $\Delta\theta_W$ is the angular change in weight direction.

**Interpretation**: When direction changes significantly, magnitude tends to decrease (and vice versa). This is a regularization effect.

### LoRA's Limitation

Standard LoRA shows:

$$\text{Corr}(\Delta\|W\|, \Delta\theta_W) \approx 0 \text{ or } > 0$$

LoRA cannot reproduce the negative correlation, limiting its expressiveness.

### DoRA's Solution

By separating magnitude ($m$) from direction ($V$), DoRA can learn the appropriate correlation for each task.

---

## Algorithm

```python
def dora_decompose(W: Array) -> tuple[Array, Array]:
    """
    Decompose weight matrix into magnitude and direction.

    Args:
        W: Weight matrix [d_out, d_in]

    Returns:
        m: Magnitude vector [d_out]
        V_hat: Unit-normalized direction [d_out, d_in]
    """
    # Column-wise norms (for each output channel)
    m = norm(W, axis=1, keepdims=True)

    # Normalize to unit direction
    V_hat = W / (m + epsilon)

    return m.squeeze(), V_hat


def dora_merge(W0: Array, delta_m: Array, delta_V: Array) -> Array:
    """
    Apply DoRA update to weight matrix.

    Args:
        W0: Pre-trained weight [d_out, d_in]
        delta_m: Magnitude update [d_out]
        delta_V: Direction update (low-rank) [d_out, d_in]

    Returns:
        Updated weight matrix
    """
    m0, V0 = dora_decompose(W0)

    # New magnitude
    m_new = m0 + delta_m

    # New direction (re-normalize after adding low-rank)
    V_new = V0 * m0.reshape(-1, 1) + delta_V
    V_hat_new = V_new / norm(V_new, axis=1, keepdims=True)

    # Combine
    return m_new.reshape(-1, 1) * V_hat_new
```

---

## Experimental Results

### Performance Comparison (Liu et al., 2024)

| Method | Commonsense Reasoning | Visual Instruction | Parameters |
|--------|----------------------|-------------------|------------|
| Full FT | 83.4% | 82.3% | 100% |
| LoRA | 80.4% | 79.1% | 0.3% |
| **DoRA** | **82.8%** | **81.7%** | **0.3%** |

DoRA achieves near-full-fine-tuning performance with LoRA's parameter efficiency.

### Why It Works

1. **Magnitude learning is cheap**: Only $d_{out}$ parameters
2. **Direction learning via LoRA**: Low-rank is sufficient for directional changes
3. **Decoupled optimization**: Each component optimized appropriately

---

## Implications for Model Merging

### Magnitude-Direction Analysis

For merged models, we can analyze:
- **Magnitude consistency**: Do merged models agree on magnitude?
- **Directional alignment**: Are directions compatible?
- **Interference patterns**: Where do conflicts occur?

### Merge Strategies

1. **Magnitude averaging + directional SLERP**:
   - Average magnitudes: $m_{merged} = \frac{1}{N}\sum_i m_i$
   - SLERP directions: $\hat{V}_{merged} = \text{SLERP}(\hat{V}_1, \hat{V}_2, t)$

2. **Weighted by magnitude change**:
   - Larger magnitude changes → more task-specific
   - Preserve task-specific magnitudes during merge

3. **Sign-aware direction merging** (combine with TIES):
   - Apply TIES to directional components
   - Separately handle magnitude conflicts

---

## ModelCypher Implementation

**Location**: `src/modelcypher/core/domain/geometry/dora_decomposition.py`

```python
class DoRADecomposition:
    """Weight decomposition into magnitude and direction components."""

    def decompose(
        self,
        weights: dict[str, Array],
        backend: Backend | None = None,
    ) -> DoRAComponents:
        """
        Decompose model weights into magnitude and direction.

        Returns per-layer magnitude vectors and directional matrices.
        """

    def analyze_adaptation(
        self,
        base_weights: dict[str, Array],
        adapted_weights: dict[str, Array],
    ) -> AdaptationAnalysis:
        """
        Analyze how adaptation changed magnitude vs direction.

        Returns correlation and per-layer statistics.
        """
```

**Design decisions**:
1. **Per-layer decomposition**: Each layer analyzed independently
2. **Correlation tracking**: Monitor magnitude-direction correlation
3. **Merge guidance**: Provide recommendations based on decomposition

---

## Extensions and Variants

### DoRAN (2025)

Stabilizes DoRA with noise injection and auxiliary networks:
- Addresses training instability in some configurations
- Improves convergence for larger models

### MAP: Revisiting Weight Decomposition (2025)

Further analysis of weight decomposition patterns:
- Confirms magnitude-direction separation benefits
- Proposes refined decomposition strategies

### La-LoRA (2025)

Layer-wise adaptive low-rank adaptation:
- Different ranks per layer based on decomposition analysis
- Guided by magnitude-direction patterns

---

## Citations

### Primary Reference

1. **Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C.F., Cheng, K.-T., & Chen, M.-H.** (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." *ICML 2024* (Oral, 1.5% acceptance rate).
   arXiv: 2402.09353
   - *The foundational paper*

### Theoretical Foundation

2. **Salimans, T., & Kingma, D.P.** (2016). "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks." *NeurIPS 2016*.
   - *Weight normalization that inspired DoRA's decomposition*

### Extensions

3. **DoRAN** (2025). "Stabilizing Weight-Decomposed Low-Rank Adaptation via Noise Injection and Auxiliary Networks."
   - *Training stability improvements*

4. **MAP** (2025). "Revisiting Weight Decomposition for Low-Rank Adaptation."
   - *Refined decomposition analysis*

5. **La-LoRA** (2025). "Parameter-efficient fine-tuning with layer-wise adaptive low-rank adaptation."
   - *Layer-adaptive extension*

### Implementation

6. **NVlabs/DoRA** (2024). Official PyTorch implementation.
   GitHub: https://github.com/NVlabs/DoRA
   - *Reference implementation*

---

## Related Concepts

- [fisher_information.md](fisher_information.md) - Importance weighting (orthogonal to DoRA)
- [task_singular_vectors.md](task_singular_vectors.md) - SVD-based decomposition (complementary)
- [slerp.md](slerp.md) - Directional interpolation after decomposition

---

*DoRA's insight—that magnitude and direction should be learned separately—reveals the geometric structure of fine-tuning and enables better model merging.*
