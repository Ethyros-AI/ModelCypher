# mHC and Null-Space Projection: A Unified Theory `[CONJECTURAL]`

**Author**: ModelCypher Research
**Date**: January 2026
**Status**: Theoretical Analysis (Draft)

## Executive Summary

DeepSeek's Manifold-constrained Hyper-Connectivity (mHC) and ModelCypher's geodesic null-space projection are **mathematically related** through the concept of **invariant-preserving projections onto constrained manifolds**. This document sketches the connection and proposes a unified multi-channel architecture for world model compression.

---

## 1. Two Systems, One Principle

### 1.1 ModelCypher Null-Space Projection

From [geodesic_null_space.py](../../src/modelcypher/core/domain/geometry/geodesic_null_space.py):

```
Goal: Add knowledge without interference

W' = W_target + P_null(A) @ δW

Where:
- P_null = I - Q @ Q^T  (projection onto null space of tangent basis Q)
- A = target activations defining the manifold structure
- δW = source - target weight difference

Invariant preserved: A @ W' = A @ W_target (boundary behavior unchanged)
```

### 1.2 DeepSeek mHC

From the mHC paper (arXiv:2512.24880):

```
Goal: Multi-channel information flow without signal explosion

x_{l+1} = H^res × x_l + H^post × F(H^pre × x_l)

Where:
- H^res is constrained to be doubly stochastic (Birkhoff polytope)
- Doubly stochastic: all rows sum to 1, all columns sum to 1, all entries ≥ 0
- Constraint enforced via Sinkhorn-Knopp projection

Invariant preserved: Spectral norm ≤ 1.0 (no signal amplification)
```

---

## 2. The Mathematical Connection

### 2.1 Both Are Projections Onto Constrained Manifolds

| Property | Null-Space Projection | Birkhoff Projection (mHC) |
|----------|----------------------|---------------------------|
| **Manifold** | Orthogonal complement of tangent space | Set of doubly stochastic matrices |
| **Projection** | P_null = I - Q @ Q^T | Sinkhorn-Knopp iteration |
| **Invariant** | Boundary behavior | Total information flow |
| **Constraint Type** | Orthogonality (L2) | Row/column sums (L1) |
| **Idempotent** | Yes: P_null² = P_null | Converges to fixed point |

### 2.2 Invariant Preservation Propositions

**Proposition 1 (Null-Space Invariance)**:
For any weight delta δW and prior activations A defining tangent basis Q:
```
A @ (W + P_null @ δW) = A @ W
```
*Sketch*: P_null @ v is orthogonal to column(Q) = tangent space. If A defines Q, then A @ P_null @ δW = 0.

**Proposition 2 (Birkhoff Invariance)**:
For any doubly stochastic matrix H and input x:
```
||H @ x||_1 ≤ ||x||_1  (with equality when H is permutation)
Σ_i (H @ x)_i = Σ_j x_j  (total mass preserved)
```
*Sketch*: Rows sum to 1, so (H @ x)_i = Σ_j H_ij x_j. Sum over i: Σ_i (H @ x)_i = Σ_i Σ_j H_ij x_j = Σ_j x_j (column sums = 1).

### 2.3 The Key Insight: Orthogonality in Different Spaces

Null-space projection enforces **L2 orthogonality** in feature space:
```
<P_null @ δW, Q @ v> = 0  for all v
```

Birkhoff projection enforces **L1 conservation** in channel space:
```
Σ_i H_ij = 1, Σ_j H_ij = 1  (soft permutation)
```

**Both prevent interference** by constraining how information flows:
- Null-space: New information cannot project onto existing directions
- Birkhoff: Channels cannot amplify or dominate total signal

---

## 3. Unified Multi-Channel Architecture

### 3.1 The Synthesis

Combining both constraints enables **multi-modal knowledge addition** with:
1. Per-channel null-space projection (prevents intra-channel interference)
2. Cross-channel Birkhoff routing (prevents inter-channel interference)

```python
# Unified Multi-Channel Merge Formula
def multi_channel_merge(
    target_weights: Array,
    channel_deltas: list[Array],      # [δW_visual, δW_temporal, δW_text, ...]
    channel_activations: list[Array], # Prior activations per channel
    H_routing: Array,                 # Doubly stochastic [n_channels, n_channels]
):
    """
    Merge multiple knowledge channels into target.

    Each channel contributes via null-space projection.
    Channels are combined via doubly stochastic routing.
    """
    n_channels = len(channel_deltas)

    # Step 1: Per-channel null-space projection
    projected_deltas = []
    for i in range(n_channels):
        Q_i = compute_tangent_basis(channel_activations[i])
        P_null_i = I - Q_i @ Q_i.T
        δW_safe_i = P_null_i @ channel_deltas[i]
        projected_deltas.append(δW_safe_i)

    # Step 2: Doubly stochastic channel mixing
    # H_routing[i, j] = contribution of channel j to output i
    # Since H is doubly stochastic: no channel dominates, no output is dominated
    merged_delta = sum(
        H_routing[i, j] * projected_deltas[j]
        for i in range(n_channels)
        for j in range(n_channels)
    )

    # Step 3: Add to target (geometric addition, not blending)
    return target_weights + merged_delta
```

### 3.2 Properties of the Unified Approach

**Property 1: CKA = 1.0 per channel**
Each channel's null-space projection preserves the invariant shape:
```
A_i @ (W + δW_safe_i) = A_i @ W  (boundary preserved)
```

**Property 2: Stable combination**
Doubly stochastic routing prevents signal explosion:
```
||merged_delta||_2 ≤ max_i ||δW_safe_i||_2  (bounded growth)
```

**Property 3: Multi-modal coverage**
Different channels encode different aspects of the invariant geometry:
- Visual channel: spatial relationships
- Temporal channel: causal/sequential relationships
- Text channel: linguistic relationships
All project onto the same 4D+ manifold, just with different entry ramps.

---

## 4. Connection to World Models

### 4.1 World Model Channels

A world model encodes three primary channels:

| Channel | Input Type | Encoding | Entry Ramp |
|---------|------------|----------|------------|
| Spatial | Image patches | Visual topology | Vision encoder |
| Temporal | Frame sequences | Causal dynamics | Temporal encoder |
| Symbolic | Text tokens | Linguistic patterns | Tokenizer |

### 4.2 The Compression Path

To compress world model capabilities into a 350M target:

1. **Extract channel-specific activations** from source world model
2. **Compute per-channel null-space projections** onto target's manifold
3. **Learn doubly stochastic routing** that optimizes CKA across all channels
4. **Apply unified merge** to target weights

### 4.3 Why This Works

The Dimensional Compression Statement (Paper 0) states, on aligned probes:
- Models encode a shared 4D+ invariant geometry on the measured probe set
- CKA = 1.0 is achievable regardless of source/target dimensions on probes

The multi-channel extension adds:
- Different modalities = different projections of the same geometry
- Birkhoff routing = stable combination of multiple projections
- Combined effect: world model capabilities in text-only architecture

---

## 5. Derivation Sketches

### 5.1 Proposition: Null-Space Projection Preserves Probe CKA (Linearized)

**Statement**: If alignment achieves CKA(source @ F, target) = 1.0, then:
```
CKA(target + P_null @ δW, target) = 1.0
```

**Sketch (probe-space linearization)**:
1. P_null projects onto directions orthogonal to target's tangent space (at probe activations)
2. CKA measures sample-space relationships (Gram matrix) on those probes
3. Adding orthogonal components leaves probe Gram inner products unchanged
4. Therefore probe CKA is unchanged for those samples

### 5.2 Lemma: Birkhoff Routing Preserves Bounded Norm

**Statement**: For doubly stochastic H and vectors {v_i}:
```
||Σ_j H_ij v_j||_2 ≤ max_j ||v_j||_2
```

**Sketch**:
1. H_ij ≥ 0 and Σ_j H_ij = 1 (convex combination)
2. ||Σ_j H_ij v_j||_2 ≤ Σ_j H_ij ||v_j||_2 (triangle inequality)
3. Σ_j H_ij ||v_j||_2 ≤ max_j ||v_j||_2 × Σ_j H_ij = max_j ||v_j||_2 ∎

### 5.3 Corollary: Conditional Architecture Stability

**Statement**: Under the probe-space assumptions above, multi-channel merge with null-space + Birkhoff is bounded in norm and preserves probe CKA.

**Sketch**: Combine Proposition 5.1 and Lemma 5.2:
- Each channel preserves probe CKA (null-space, on probes)
- Combination is bounded (Birkhoff)
- Therefore: multi-channel merge preserves probe CKA and bounded growth under these assumptions

---

## 6. Implementation Considerations

### 6.1 Changes to ModelCypher Pipeline

To support multi-channel merging:

1. **Extend `GeodesicNullSpaceFilter`**:
   - Accept list of prior_activations (one per channel)
   - Compute channel-specific Q bases
   - Return list of projected deltas

2. **Add `BirkhoffRouter`**:
   - Initialize H as uniform (1/n)
   - Optimize H via Sinkhorn-Knopp projection
   - Learn routing that maximizes CKA across channels

3. **Modify `filter_merge_delta_geodesic`**:
   - Accept channel_deltas and H_routing
   - Apply unified formula

### 6.2 Computational Cost

| Operation | Single-Channel | Multi-Channel (n) |
|-----------|----------------|-------------------|
| Tangent basis (QR) | O(n_samples × d²) | n × O(n_samples × d²) |
| Null-space projection | O(d²) | n × O(d²) |
| Sinkhorn iteration | N/A | O(n² × iterations) |
| Total | O(n_samples × d²) | O(n × n_samples × d² + n² × iter) |

For typical values (n_samples=1000, d=4096, n_channels=3, iterations=20):
- Single-channel: ~16B ops
- Multi-channel: ~50B ops
- Overhead: ~3x (acceptable for multi-modal capability)

---

## 7. Experimental Validation Plan

### 7.1 Hypothesis to Test

1. **CKA preservation**: Multi-channel merge maintains probe CKA (target 1.0) per channel
2. **Capability transfer**: Spatial/temporal capabilities transfer to text-only model
3. **Stability**: No signal explosion across merge iterations

### 7.2 Protocol

1. Select source models:
   - InternVL2 (vision-language)
   - V-JEPA (world model)
   - Qwen2.5 (text-only)

2. Select target: LFM2-350M

3. Extract channel activations:
   - Spatial probes → InternVL2 vision encoder
   - Temporal probes → V-JEPA dynamics encoder
   - Semantic probes → all models' language layers

4. Apply multi-channel merge

5. Measure:
   - CKA per channel (expect: 1.0)
   - World Model Score (expect: increase)
   - Task performance (spatial reasoning, temporal reasoning)

---

## 8. Conclusion

The mathematical connection between mHC and null-space projection reveals a unified principle:

> **Invariant-preserving projections onto constrained manifolds enable stable knowledge addition.**

For null-space: the manifold is the orthogonal complement of tangent space.
For Birkhoff: the manifold is the set of doubly stochastic matrices.

Combining both enables multi-modal knowledge compression while maintaining:
- CKA = 1.0 (geometric invariance)
- Bounded signal growth (numerical stability)
- Multi-channel coverage (world model capabilities)

The path to compressing world model capabilities into a 350M text model is now mathematically well-defined.

---

## References

- DeepSeek-AI. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. arXiv:1905.00414.
- ModelCypher Paper 0: The Shape of Knowledge (January 2026).
- Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices.
