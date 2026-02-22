# Anchor-Relative Concept Grafting `[CONJECTURAL]`

> **Status**: Design note / research proposal
> **Goal**: Transfer concepts by aligning in anchor-relative space and projecting back into
> target coordinates before null-space addition.

## Problem

Activation-space transforms (F = pinv(X_s) @ X_t) achieve CKA = 1.0 on probes by construction, but applying
F directly to weights breaks the target model. F is an activation-space map, not a weight-space map.

We need a pipeline that:
1) finds what to transfer in a shared, dimension-agnostic space, and
2) applies the update in the target's native coordinates only.

## Core Insight

Anchor-relative coordinates (Moschella et al., 2023) provide a shared semantic address space that
is invariant to latent isometries and rescalings. Alignment happens in anchor space, not feature space.
We then decode back into target activations using the target's own pseudo-inverse.

No foreign coordinates ever touch target weights.

## Canonical Pipeline (No Modes)

Let:
- A_s, A_t: source/target activations at a layer (n x d_s, n x d_t)
- C_s, C_t: anchor embeddings for that layer (m x d_s, m x d_t)
- S_s, S_t: anchor-relative representations (n x m)

```
S_s = cos(A_s, C_s)
S_t = cos(A_t, C_t)

# Align in anchor space
R = Procrustes(S_s, S_t)
Delta_S = S_s @ R - S_t

# Density-derived selection (no thresholds)
rho_s = local_density(S_s)
rho_t = local_density(S_t)
w = rho_s / (rho_s + rho_t)

# Decode into target activation space (target-only)
B = pinv(S_t) @ A_t
Delta_A = (w * Delta_S) @ B

# Constrained weight update (target-only)
Find Delta_W s.t.:
  A_core @ Delta_W ~= Delta_A_core
  A_boundary @ Delta_W = 0

W_merged = W_target + Delta_W
```

Notes:
- `local_density` can be computed from local intrinsic dimension or k-NN density on S_s/S_t.
- The only pseudo-inverse used is from target data (pinv(S_t)), so decoding stays in target coordinates.
- The boundary constraint is enforced via geodesic null-space projection.

## Anchor Selection (Data-Derived)

Anchor choice must be deterministic and geometry-derived. No fixed k, no random sampling,
no hand-tuned thresholds.

### Anchor Sources

1. **Atlas probes (default)**: Use the unified atlas probe registry as shared anchor IDs.
2. **Layer-native anchors**: Prefer anchors computed from per-layer activations (not just token embeddings).
   For each anchor ID, compute the mean activation across its support texts at that layer.

### Anchor Subset Selection

Given a candidate anchor matrix C (m x d) and relative representations S:

1. **Determine target anchor count (m\*)** via spectral gap detection on the anchor Gram spectrum.
   This is the same data-derived rule used in shared-subspace alignment (no manual cutoffs).
2. **Select anchors deterministically** with pivoted QR (max-volume subset). This maximizes span
   without randomness and yields an ordered list of anchors by geometric contribution.
3. **Verify coverage** by CKA invariance in anchor space:
   - Compute CKA(S_full, S_subset).
   - Accept when deviation <= sqrt(machine_epsilon(dtype)).
   - If not, expand the subset by the next QR pivots.

This yields the smallest anchor set that preserves relational structure at numerical precision.

### Cross-Model Consistency

Anchors must be shared across source and target:
- Drop any anchor IDs missing from either model.
- Preserve ordering by anchor ID to keep S_s and S_t aligned.

## Validation

The anchor-relative grafting step is valid only if:
1. CKA(S_s @ R, S_t) == 1.0 within precision
2. Boundary invariance holds (A_boundary @ Delta_W == 0)

Both checks are data-derived and threshold-free (only machine epsilon).

## Layer-Specific Strategy (Data-Derived)

Use measured intrinsic dimension profiles to weight where grafting is strongest.
The "highway core" (low ID in mid-layers) is the primary transfer region.

Example policy:
- Early layers: conservative (high ID, modality-specific)
- Highway core: aggressive (low ID, semantic core)
- Late layers: conservative (task heads)

No hard thresholds: weights come from ID/density measurements only.

## Invariants

1. CKA = 1.0 in anchor space on the probe manifold.
2. Target boundary behavior is preserved by null-space projection.
3. No blending or interpolation; additions are sparse, target-native updates.
4. All weights derived from measured geometry (no heuristic modes).

## Implementation Touchpoints

Existing modules already cover the key operations:
- Anchor-relative space: `src/modelcypher/core/domain/geometry/relative_representation.py`
- Procrustes alignment: `src/modelcypher/core/domain/geometry/gram_aligner.py`
- Intrinsic dimension / density: `src/modelcypher/core/domain/geometry/intrinsic_dimension.py`
- Density ratios: `src/modelcypher/core/domain/geometry/knowledge_density.py`
- Null-space projection: `src/modelcypher/core/domain/geometry/geodesic_null_space.py`
- Transplant constraints: `src/modelcypher/core/domain/geometry/transplant.py`

## References

- Moschella et al. (2023). Relative representations enable zero-shot latent space communication.
  `docs/references/arxiv/Moschella_2022_Relative_representations_enable_zeroshot_latent_space.pdf`
- Aghajanyan et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.
  `docs/references/arxiv/Aghajanyan_2021_Intrinsic_Dimensionality_Fine_Tuning.pdf`
- Denti et al. (2022). GRIDE intrinsic dimension estimator.
  `docs/references/arxiv/Denti_2022_GRIDE_Generalized_Ratios_Intrinsic_Dimension.pdf`
- Cheng et al. (2025). High-dimensional abstraction phase in LLMs.
  `docs/references/arxiv/Cheng_2025_HighDimensional_Abstraction_Phase_LMs.pdf`
- Ruppik et al. (2025). Local intrinsic dimensions of contextual LMs.
  `docs/references/arxiv/Ruppik_2025_Local_Intrinsic_Dimensions_Contextual_LMs.pdf`
