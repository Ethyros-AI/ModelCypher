# Future Directions: Cross-Architecture Transfer Research

> **Status**: Research roadmap consolidating speculative work on geometric knowledge transfer.
>
> This document synthesizes three related research proposals into a unified roadmap
> for cross-architecture capability transfer using geometric principles.

---

## Executive Summary

Three research threads converge on one goal: **transfer knowledge between models with different architectures using geometric invariants**.

| Thread | Core Idea | Key Insight |
|--------|-----------|-------------|
| Anchor-Relative Grafting | Align in anchor space, decode in target space | No foreign coordinates touch target weights |
| Cross-LoRA Transfer | Project adapters via Procrustes rotation | "Write once, run anywhere" adapters |
| Multi-Channel Architecture | Combine null-space + Birkhoff routing | Multi-modal transfer with stable combination |

---

## Thread 1: Anchor-Relative Concept Grafting

### The Problem

Activation-space transforms (F = pinv(X_s) @ X_t) achieve CKA = 1.0 on probes by construction, but applying F directly to weights breaks the target model. F is an activation-space map, not a weight-space map.

### The Solution

Anchor-relative coordinates (Moschella et al., 2023) provide a shared semantic address space invariant to latent isometries and rescalings. Alignment happens in anchor space, not feature space.

```
S_s = cos(A_s, C_s)       # Source anchor-relative representation
S_t = cos(A_t, C_t)       # Target anchor-relative representation

# Align in anchor space
R = Procrustes(S_s, S_t)
Delta_S = S_s @ R - S_t

# Decode into target activation space (target-only)
B = pinv(S_t) @ A_t
Delta_A = (density_weight * Delta_S) @ B

# Constrained weight update (null-space projection)
W_merged = W_target + P_null @ Delta_W
```

### Key Invariants

1. CKA = 1.0 in anchor space on the probe manifold
2. Target boundary behavior preserved by null-space projection
3. No blending—additions are sparse, target-native updates

### Implementation Touchpoints

- Anchor-relative space: `relative_representation.py`
- Procrustes alignment: `gram_aligner.py`
- Density ratios: `knowledge_density.py`
- Null-space projection: `geodesic_null_space.py`

---

## Thread 2: Cross-LoRA Transfer

### The Dream

Train a "coding adapter" for Llama-3 and reuse it on Qwen-2.5 without retraining.

### The Hypothesis

Some fine-tuned behaviors correspond to approximately transferable low-rank structure detectable via anchor-induced relational geometry:

```
ΔW_target ≈ P^T · ΔW_source · P
```

Where P is the orthogonal Procrustes rotation derived from semantic primes.

### The Algorithm

1. **Extract Anchors**: Compute semantic prime activations for source and target
2. **Align**: Find rotation R mapping source → target
3. **Project**: Apply R to LoRA matrices A and B
4. **Smooth**: Fine-tune on small calibration set (orders of magnitude cheaper)

### Rotation Field Roughness

Roughness measures how non-uniformly rotation varies across layers:
- **Low roughness**: Consistent transformation (good for transfer)
- **High roughness**: Non-uniform mapping (may need layer-specific handling)

---

## Thread 3: Multi-Channel Architecture (mHC Connection)

### The Mathematical Connection

ModelCypher's null-space projection and DeepSeek's Manifold-constrained Hyper-Connectivity (mHC) are mathematically related through invariant-preserving projections onto constrained manifolds.

| Property | Null-Space Projection | Birkhoff Projection (mHC) |
|----------|----------------------|---------------------------|
| Manifold | Orthogonal complement | Doubly stochastic matrices |
| Invariant | Boundary behavior | Total information flow |
| Constraint | Orthogonality (L2) | Row/column sums (L1) |

### Unified Multi-Channel Merge

```python
def multi_channel_merge(target_weights, channel_deltas, channel_activations, H_routing):
    # Step 1: Per-channel null-space projection
    projected_deltas = []
    for i, (delta, activations) in enumerate(zip(channel_deltas, channel_activations)):
        Q_i = compute_tangent_basis(activations)
        P_null_i = I - Q_i @ Q_i.T
        projected_deltas.append(P_null_i @ delta)

    # Step 2: Doubly stochastic channel mixing
    merged_delta = sum(H_routing[i,j] * projected_deltas[j] for i,j in indices)

    # Step 3: Geometric addition (not blending)
    return target_weights + merged_delta
```

### Properties

1. **CKA = 1.0 per channel**: Each null-space projection preserves probe geometry
2. **Stable combination**: Birkhoff routing prevents signal explosion
3. **Multi-modal coverage**: Different channels encode different aspects of geometry

---

## Unified Research Roadmap

### Phase A: Validate Anchor-Relative Transfer

1. Implement full anchor-relative pipeline
2. Test on same-architecture pairs (known to work)
3. Test on cross-architecture pairs (LFM2-700M → LFM2-350M)
4. Measure: CKA preservation, capability transfer, boundary stability

### Phase B: Cross-LoRA Experiments

1. Train coding adapter on Llama-3
2. Project to Qwen-2.5 using Procrustes
3. Measure rotation field roughness
4. Calibrate and evaluate

### Phase C: Multi-Channel World Model Compression

1. Extract channel-specific activations from world models (InternVL2, V-JEPA)
2. Compute per-channel null-space projections onto LFM2-350M
3. Learn doubly stochastic routing
4. Evaluate spatial/temporal reasoning transfer

---

## Core Principle

> **Invariant-preserving projections onto constrained manifolds enable stable knowledge addition.**

For null-space: the manifold is the orthogonal complement of tangent space.
For Birkhoff: the manifold is the set of doubly stochastic matrices.
Combined: multi-modal knowledge compression while maintaining CKA = 1.0.

---

## References

- Moschella et al. (2023). Relative representations enable zero-shot latent space communication.
- DeepSeek-AI. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.
- ModelCypher Paper 0: The Shape of Knowledge (January 2026).

---

## Archived Source Documents

The following documents were consolidated into this roadmap and archived:
- `anchor_relative_concept_grafting.md` → Thread 1
- `cross_lora_transfer.md` → Thread 2
- `mhc_null_space_connection.md` → Thread 3

Archived to: `/Volumes/CodeCypher/archive/modelcypher-legacy/docs/research/`
