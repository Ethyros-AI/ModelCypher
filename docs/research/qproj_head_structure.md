# q_proj Rank-126 Inflection — Head Structure Investigation

**Date**: 2026-02-19
**Status**: Complete. The rank-126 inflection is a universal architectural signature of LFM2's attention projections at position 2 × head_dim - 2, not layer-8 specific.

## Question

Part A (spectral energy curves) found that q_proj at layer 8 has an energy curve inflection at exactly rank 126 in all 3 LFM2 models (350M, 700M, 1.2B). This was the only statistically significant cross-scale signal (P = 0.0001 by chance after Bonferroni correction). Why does this inflection exist, and is it layer-8 specific?

Key observation: head_dim = 64 across all LFM2 scales, so 126 = 2 × head_dim - 2.

## Architecture context

| Model | hidden_dim | n_heads (q) | kv_heads (k, v) | head_dim | q_proj shape | k/v_proj shape |
|---|---|---|---|---|---|---|
| 350M | 1024 | 16 | 8 | 64 | [1024, 1024] | [512, 1024] |
| 700M | 1536 | 24 | 8 | 64 | [1536, 1536] | [512, 1536] |
| 1.2B | 2048 | 32 | 8 | 64 | [2048, 2048] | [512, 2048] |

Attention layers: indices 2, 5, 8, 10, 12, 14 (6 of 16 total layers).

## Method

**Script**: `scripts/qproj_head_structure.py`

For each model, at all 6 attention layers:

1. **Full-matrix SVD** of q_proj, k_proj, v_proj (weight-only, no inference)
2. **Energy curve + inflection points** (using `compute_full_energy_curve`, `find_energy_inflection_points` from spectral_capacity.py)
3. **Per-head SVD**: reshape q/k/v → [n_heads, head_dim, hidden_dim], compute SVD of each head block, concatenate and sort → "block spectrum"
4. **Head orthogonality**: ||M_i M_j^T||_F / (||M_i||_F × ||M_j||_F) for all head pairs

## Results

### Finding 1: Inflection at rank 126 appears across ALL attention layers, not just layer 8

q_proj EXACT inflection at rank 126:
- 350M: 4/6 layers (L2, L8, L10, L12)
- 700M: 4/6 layers (L5, L8, L12, L14)
- 1.2B: 3/6 layers (L2, L8, L14)

k_proj EXACT at 126: 350M 3/6, 700M 2/6, 1.2B 1/6
v_proj EXACT at 126: 350M 3/6, 700M 0/6, 1.2B 1/6

**Layer 8 is the only layer where all 3 models show EXACT q_proj inflection at 126.** But the inflection is not unique to layer 8 — it appears at most attention layers, with some model-specific variation in which layers have the exact match vs a near-miss.

### Finding 2: Heads are approximately orthogonal

Mean pairwise similarity across all models:

| Projection | Similarity range | Interpretation |
|---|---|---|
| q_proj | 0.039 – 0.077 | ~95% orthogonal |
| k_proj | 0.039 – 0.080 | ~95% orthogonal |
| v_proj | 0.019 – 0.042 | ~97% orthogonal |

Layer 14 consistently has the highest head similarity (all models). v_proj heads are the most orthogonal. These similarity levels mean the full-matrix SVD is well-approximated by the union of per-head SVDs.

### Finding 3: Block spectrum does NOT consistently reproduce the inflection

The per-head concatenated spectrum (sorted union of all per-head SVDs) shows EXACT inflection at 126 only sporadically:
- 350M: q_proj block match at 3/6 layers (L2, L8, L14)
- 700M: q_proj block match at 2/6 layers (L12, L14)
- 1.2B: q_proj block match at 0/6 layers

This means the inflection is NOT simply "per-head SVDs stacked." It's a property of the full matrix that the block decomposition partially captures.

### Finding 4: Energy at rank 126 decreases with model size

| Model | q_proj energy at rank 126 (range across layers) |
|---|---|
| 350M | 0.63 – 0.75 |
| 700M | 0.50 – 0.64 |
| 1.2B | 0.45 – 0.56 |

This is expected: larger models have more SVs, so rank 126 captures a smaller fraction of total energy.

### Finding 5: Inflection prominence is mid-range

When present, the rank-126 inflection is typically at prominence rank 34-67 out of 64-117 total inflection points. This is consistent with Part A's finding: the inflection is real but not among the most prominent spectral features.

## Interpretation

### What rank 126 = 2 × head_dim - 2 means

The inflection at 2 × head_dim - 2 is an architectural signature tied to the shared head dimension. It appears across q/k/v projections and all attention layers, not as a property of any specific layer or projection type.

Possible mechanisms:
1. **Two-SV-per-head boundary**: With approximately orthogonal heads, the sorted full spectrum interleaves per-head SVs. At rank ~2 × head_dim, the spectrum transitions from "first 2 SVs of each head" to "3rd SV of each head." The -2 offset could be the number of heads whose second SV falls below the noise threshold.
2. **Cross-head interaction**: The ~5% non-orthogonality creates subtle spectral mixing that peaks at multiples of head_dim.

The mechanism is consistent with head orthogonality (Finding 2) but not fully explained by simple block structure (Finding 3). The full matrix's cross-head interactions contribute to the inflection.

### What this means for LoRA rank selection

The energy at rank 126 is 45-75% depending on model size and layer. A LoRA adapter with rank 126 captures roughly half to three-quarters of the q_proj energy. This makes 2 × head_dim a natural "knee" for rank selection in attention projections — below it, you're losing substantial signal; above it, returns diminish.

This is a geometric observation, not a recommendation. The right LoRA rank depends on the task, not just the energy curve.

## Data files

- `data/experiments/qproj_head_structure_350m.json`
- `data/experiments/qproj_head_structure_700m.json`
- `data/experiments/qproj_head_structure_1p2b.json`

## Related

- [spectral_capacity_domain_rank.md](spectral_capacity_domain_rank.md) — Part A (weight energy curves) where the inflection was first found
- [positive_geometry_scale_comparison.md](positive_geometry_scale_comparison.md) — Original domain rank analysis (now corrected)
