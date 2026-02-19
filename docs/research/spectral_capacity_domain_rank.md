# Spectral Capacity vs Domain Rank Signatures

**Date**: 2026-02-18
**Status**: Complete (negative result for weight-gap hypothesis; positive finding on attention energy stability)

## Question

Positive geometry analysis found domain-conditioned rank signatures at layers 7,8 that are invariant across LFM2-350M/700M/1.2B:
- **linguistic/mental**: rank 126
- **computational/structural**: rank 211
- **factual**: rank 255

Does the weight matrix singular value spectrum at layers 7,8 show structure at these specific rank values? If yes, the domain signatures are rooted in weight geometry. If no, they emerge from activation dynamics.

## Method

For each of the 3 LFM2 models, ran `mc model capacity` on all 2D weight matrices at layers 7 and 8, extracting:
1. **Spectral gap ratio** at each domain rank position: `sigma[k-1] / sigma[k]` at k = 126, 211, 255
2. **Energy fraction** captured by top k singular values at each position
3. **Effective rank**, **recommended rank** (largest spectral gap), **decay type**

Script: `scripts/spectral_capacity_domain_rank_investigation.py`
Data: `data/experiments/spectral_capacity_domain_rank_*.json`

### Architecture context

- Layer 7 = ShortConv (no attention weights)
- Layer 8 = Attention (q_proj, k_proj, v_proj, out_proj + FFN)
- Hidden sizes: 350M=1024, 700M=1536, 1.2B=2048
- k_proj and v_proj are fixed at 512 rows across all scales (head dimension doesn't scale)

## Results

### Finding 1: No spectral gaps at domain rank positions

Every gap ratio at positions 126, 211, and 255 is between 1.000 and 1.007. There is no meaningful spectral gap at any domain rank boundary in any weight matrix at any scale.

| Weight type | Gap at 126 (range) | Gap at 211 (range) | Gap at 255 (range) |
|---|---|---|---|
| FFN (w1, w2, w3) | 1.000-1.003 | 1.000-1.002 | 1.000-1.002 |
| conv.in/out_proj | 1.000-1.004 | 1.001-1.001 | 1.001-1.002 |
| self_attn.k_proj | 1.002-1.005 | 1.002-1.004 | 1.001-1.003 |
| self_attn.q_proj | 1.006-1.007 | 1.002-1.004 | 1.001-1.003 |
| self_attn.v_proj | 1.002-1.002 | 1.001-1.002 | 1.002-1.002 |

The largest observed "gap" is 1.007 at q_proj rank 126. For comparison, actual spectral gaps in these matrices (at the recommended rank) are 1.2-18x or higher.

**Conclusion: Domain rank signatures are NOT imprinted in the weight spectrum.** The singular values decay smoothly through positions 126, 211, and 255 with no structural break.

### Finding 2: Attention projection energy fractions are scale-stable

The energy fraction at domain rank positions in attention projections (k_proj, v_proj) is remarkably stable across model sizes, while FFN energy fractions decrease with scale:

**k_proj at layer 8** (fixed 512 rows across all scales):

| Domain rank | 350M | 700M | 1.2B | Variation |
|---|---|---|---|---|
| 126 (linguistic) | 0.733 | 0.687 | 0.692 | ~0.05 |
| 211 (computational) | 0.877 | 0.846 | 0.847 | ~0.03 |
| 255 (factual) | 0.921 | 0.897 | 0.896 | ~0.03 |

**v_proj at layer 8** (fixed 512 rows):

| Domain rank | 350M | 700M | 1.2B | Variation |
|---|---|---|---|---|
| 126 (linguistic) | 0.592 | 0.527 | 0.500 | ~0.09 |
| 211 (computational) | 0.779 | 0.721 | 0.696 | ~0.08 |
| 255 (factual) | 0.847 | 0.796 | 0.776 | ~0.07 |

**FFN w1 at layer 8** (scales with hidden_dim):

| Domain rank | 350M | 700M | 1.2B | Variation |
|---|---|---|---|---|
| 126 (linguistic) | 0.337 | 0.261 | 0.222 | ~0.12 |
| 211 (computational) | 0.467 | 0.363 | 0.310 | ~0.16 |
| 255 (factual) | 0.525 | 0.409 | 0.350 | ~0.18 |

k_proj's energy stability makes geometric sense: its row dimension (512) is fixed across scales, so the same rank captures roughly the same fraction. FFN layers grow with hidden_dim, so a fixed rank captures less of a larger space.

### Finding 3: Attention vs FFN structural differences

| Property | Attention (layer 8) | FFN (layers 7,8) | Conv (layer 7) |
|---|---|---|---|
| Decay type | sharp_cliff (mostly) | gradual_slope (mostly) | gradual_slope |
| Effective rank / min_dim | 0.65-0.88 | 0.85-0.93 | 0.63-0.92 |
| Recommended rank | 1 or max-1 | 1 or max-1 | 1 or max-1 |
| Null space fraction | 0.0-0.4% | 0.0% | 0.0-0.2% |

Attention projections have sharper spectral cliffs and lower effective rank relative to their dimensions. They're more "structured" — using a smaller fraction of their available capacity — compared to FFN layers.

### Finding 4: Effective rank scales with hidden dimension, not with domain ranks

| Model | Hidden dim | Mean eff. rank (L7+L8) | Eff. rank / hidden |
|---|---|---|---|
| 350M | 1024 | 735.6 | 0.72 |
| 700M | 1536 | 1078.2 | 0.70 |
| 1.2B | 2048 | 1407.9 | 0.69 |

Effective rank grows proportionally with model size (~70% of hidden dim), while domain ranks (126, 211, 255) remain fixed. The domain ranks occupy an increasingly small fraction of the available spectral capacity as models scale.

## Interpretation

**Domain rank signatures emerge from activation dynamics, not weight structure.**

The weight matrices provide a smooth, gap-free spectral landscape. The domain-specific rank structure crystallizes from the nonlinear interaction between weights and activations during the forward pass. The weights set the capacity ceiling; the activation dynamics determine how that capacity is partitioned across domains.

This is consistent with a picture where:
1. Weights define a high-dimensional geometric landscape (smooth spectrum, no domain-specific breaks)
2. During a forward pass, input activations interact with this landscape
3. Domain-specific concepts activate subspaces of different dimensionality (rank 126 for linguistic, 211 for computational, etc.)
4. These subspace dimensions are determined by the computational structure of the domain, not by pre-existing weight structure

The attention projection energy stability (Finding 2) adds a nuance: the fixed head dimension (512) creates a natural constraint. k_proj consistently concentrates ~69% of its energy in the first 126 dimensions across all scales. This isn't a spectral gap — it's a smooth energy concentration — but it means the linguistic domain's rank=126 captures a consistent fraction of the attention key space regardless of model size. Whether this is coincidence or reflects a deeper geometric relationship between attention head structure and domain encoding remains an open question.

## Data files

- `data/experiments/spectral_capacity_domain_rank_350m.json`
- `data/experiments/spectral_capacity_domain_rank_700m.json`
- `data/experiments/spectral_capacity_domain_rank_1p2b.json`
- `data/experiments/spectral_capacity_domain_rank_crossscale.json`

## Related

- [positive_geometry_scale_comparison.md](positive_geometry_scale_comparison.md) — Source of domain rank signatures
- [POSITIVE-GEOMETRY-ANALYSIS.md](POSITIVE-GEOMETRY-ANALYSIS.md) — Positive geometry methodology
