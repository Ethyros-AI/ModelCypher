# Spectral Capacity vs Domain Rank Signatures

**Date**: 2026-02-18 (initial), 2026-02-19 (updated with Parts A+B)
**Status**: Complete. Domain rank signatures (126, 211, 255) are probe-count artifacts, not geometric invariants. One genuine weight-space signal: q_proj inflection at rank 126.

## Question

Positive geometry analysis found domain-conditioned rank signatures at layers 7,8 that appeared invariant across LFM2-350M/700M/1.2B:
- **linguistic/mental**: rank 126
- **computational/structural**: rank 211
- **factual**: rank 255

Are these genuine geometric properties of the model? Where do they come from?

## Investigation Phases

### Phase 0: Spectral gap ratios (2026-02-18)

**Method**: Checked weight matrix singular value gap ratios at positions 126, 211, 255.

**Result**: All gap ratios 1.000-1.007 — no weight-space spectral gaps at domain rank positions. See Finding 1 below.

### Part A: Weight energy curves + inflection points (2026-02-19)

**Method**: Computed full cumulative energy curves E(k) = sum(σ_i² for i=0..k) / total, found inflection points via second-derivative peaks, overlaid domain rank positions.

**Script**: `scripts/spectral_energy_curves.py`
**Data**: `data/experiments/spectral_energy_curves_*.json`

### Part B: Activation spectral analysis (2026-02-19)

**Method**: Collected mean-pooled hidden states per domain group at layers 7,8. Computed SVD via Gram eigendecomposition. Analyzed energy curves, gap ratios, and inflection points in activation space.

**Script**: `scripts/activation_spectral_analysis.py`
**Data**: `data/experiments/activation_spectral_*.json`

### Architecture context

- Layer 7 = ShortConv (no attention weights)
- Layer 8 = Attention (q_proj, k_proj, v_proj, out_proj + FFN)
- Hidden sizes: 350M=1024, 700M=1536, 1.2B=2048
- k_proj and v_proj are fixed at 512 rows across all scales

---

## Critical Discovery: Domain Ranks Are Probe-Count Artifacts

**The "invariant domain rank signatures" (126, 211, 255) are NOT geometric properties of the model. They are determined by the number of probes used per domain group.**

| Domain group | Probe count | Reported rank | SVD spectral-gap rank (measured) |
|---|---|---|---|
| linguistic + mental | 131 | 126 | **127** |
| computational + structural | 213 | 211 | **211** |
| factual | 256 (capped by --probe-count) | 255 | **255** |

The `spectral-gap` rank method (used by `mc analyze concept-volume --rank-source spectral-gap`) finds the largest relative drop in the singular value sequence. For an activation matrix of [n_probes, hidden_dim], SVD produces n_probes singular values. The last few SVs drop to numerical noise, creating the "gap" at position n_probes - c (where c = 2-5 is the number of noise dimensions).

These ranks appear "invariant across scales" because the same probes are used for all models. The invariance is trivial: same probe set → same matrix size → same noise floor → same spectral-gap position.

**Verification**: Recomputed spectral-gap rank on correctly-collected activations (with LFM2-compatible mask routing) for two of the three groups:
- linguistic_mental (131 probes): SVD spectral_gap = 127, Gram spectral_gap = 1
- computational_structural (213 probes): SVD spectral_gap = 211, Gram spectral_gap = 1
- factual (256 probes): **Not directly verified in Part B** (no factual group in activation script). The prediction (255 = 256 - 1) follows trivially from the same n_probes - c mechanism.

The Gram approach gives rank=1 because the DC component dominates — the largest relative drop is between eigenvalue 1 (DC) and eigenvalue 2.

---

## Results

### Finding 1: No weight-space spectral gaps at domain rank positions

Every gap ratio at positions 126, 211, and 255 is between 1.000 and 1.007. There is no meaningful spectral gap at any domain rank boundary in any weight matrix at any scale.

| Weight type | Gap at 126 (range) | Gap at 211 (range) | Gap at 255 (range) |
|---|---|---|---|
| FFN (w1, w2, w3) | 1.000-1.003 | 1.000-1.002 | 1.000-1.002 |
| conv.in/out_proj | 1.000-1.004 | 1.001-1.001 | 1.001-1.002 |
| self_attn.k_proj | 1.002-1.005 | 1.002-1.004 | 1.001-1.003 |
| self_attn.q_proj | 1.006-1.007 | 1.002-1.004 | 1.001-1.003 |
| self_attn.v_proj | 1.002-1.002 | 1.001-1.002 | 1.002-1.002 |

### Finding 2: q_proj has a genuine inflection at rank 126 (Part A)

**q_proj has a weight-space energy curve inflection at exactly rank 126 in all 3 models** (350M, 700M, 1.2B). This is statistically significant:

- Inflection density: 3.1-7.6% of ranks are inflection points in q_proj
- P(all 3 models hit exact rank by chance) = 0.0001 (passes Bonferroni correction for 12 tests)
- However, this inflection is **low prominence** (rank ~35 of ~70 inflection points by |d²E|)
- The inflection magnitude (|d²E| ≈ 2.5e-5) is ~50× weaker than the top inflections at ranks 3-5

Other attention projections:
- **k_proj/v_proj**: Distance ≤3 from domain ranks, but inflection density is 17-21% — proximity is expected by chance
- **out_proj**: Alignment degrades with model scale (1.2B shows poor alignment)
- **FFN layers**: No alignment at any domain rank (distances 15-200)

### Finding 3: Activation space is dominated by DC component (Part B)

Mean-pooled activation spectra are extremely low-rank:

| Metric | 350M | 700M | 1.2B |
|---|---|---|---|
| Shannon effective rank | 1.1-1.2 | 1.2-1.7 | 1.2-1.4 |
| SV1/SV2 ratio | 33-40× | 25-35× | 30-45× |
| SV1 energy fraction | 98.7-99.3% | 97.8-99.0% | 98.5-99.1% |

After centering (removing mean direction), effective rank improves only slightly to 1.1-1.7. The activation subspace at layers 7-8 is essentially rank-1 with a long tail of low-amplitude variation.

No activation spectral gaps at domain rank positions in any tested group: all gap ratios 1.00-1.08 across 350M/700M/1.2B. Part B tested linguistic_mental (131 probes, can test rank 126), computational_structural (213 probes, can test 126/211), mathematical_logical (208 probes, can test 126), and all_nonfactual (963 probes, can test all three). Rank 255 was only testable via the all_nonfactual group.

### Finding 4: Attention energy fractions are scale-stable (Phase 0)

k_proj at layer 8 (fixed 512 rows across all scales):

| Domain rank | 350M | 700M | 1.2B | Variation |
|---|---|---|---|---|
| 126 | 0.733 | 0.687 | 0.692 | ~0.05 |
| 211 | 0.877 | 0.846 | 0.847 | ~0.03 |
| 255 | 0.921 | 0.897 | 0.896 | ~0.03 |

This stability is explained by k_proj's fixed row dimension (512), not by domain structure.

### Finding 5: `forward_through_backbone` had LFM2 mask bug (FIXED)

The existing `forward_through_backbone()` in `model_backbone.py` created a single numeric causal mask via `backend.create_causal_mask(seq_len, dtype)` and applied it to every layer. LFM2 layers expect string `"causal"` for attention layers and `None` for conv layers. The numeric mask broadcast the batch dimension: `[1, seq_len, hidden_dim]` → `[seq_len, seq_len, hidden_dim]`.

This bug affected all positive geometry analyses run through the CLI on LFM2 models. The activation vectors computed with the bug were deterministic but geometrically incorrect.

**Fix applied**: `_resolve_layer_mask()` in `model_backbone.py` checks `layer.is_attention_layer` for LFM2-style hybrid models and routes to string `"causal"` or `None` accordingly. Standard transformers (which lack this attribute) fall through to the numeric mask. Same fix applied to `collect_trajectory_batch()` in `activation_provider.py`.

**Revalidation completed** (2026-02-19): All positive geometry signatures re-collected with corrected masks (`scripts/revalidate_positive_geometry.py`). Key impact: factual domain at 350M collapsed from rank 255 to rank 1 — the "factual high-rank at all scales" finding was partially a mask artifact. Linguistic rank shifted 126→127 at all scales. See [positive_geometry_scale_comparison.md](positive_geometry_scale_comparison.md) "REVALIDATION" section for full corrected table.

---

## Interpretation

### What the domain ranks actually are

The numbers 126, 211, 255 are a property of the probe set, not the model:
- 126 ≈ 131 (linguistic+mental probes) - 5 noise directions
- 211 ≈ 213 (computational+structural probes) - 2 noise directions
- 255 ≈ 256 (factual probes, capped at --probe-count) - 1 noise direction

The spectral-gap method reports "the last numerically significant dimension before noise." For n probes in a ~1024-dim space, this is approximately n - c.

### What IS geometrically real

1. **Weight-space q_proj inflection at rank 126** — the only statistically significant cross-scale signal in weight spectra. Low prominence but exact position match across 3 models. The coincidence with the linguistic_mental probe count (131) is accidental — follow-up investigation (`qproj_head_structure.md`) confirmed rank 126 = 2 × head_dim - 2 is a universal architectural signature of LFM2 attention layers (appears at most attention layers across q/k/v projections, not just q_proj at layer 8).

2. **Attention vs FFN structural difference** — attention projections have sharper spectral cliffs and lower effective rank relative to their dimensions. They use a smaller fraction of available capacity.

3. **k_proj energy stability across scales** — the fixed head dimension (512) creates a natural constraint where the same rank captures similar energy fractions regardless of model size.

4. **Activation-space DC dominance** — at layers 7-8, mean-pooled activations are essentially rank-1. Domain-specific variation lives in the 0.7-2% tail beyond the first singular value.

### What is NOT geometrically real

1. **"Domain rank signatures invariant across scales"** — these are probe-count artifacts
2. **"Domain ranks emerge from activation dynamics"** — the ranks are determined by matrix dimensions, not by dynamics
3. **Activation spectral structure at positions 126/211/255** — no gaps, no inflections, no structure

---

## Data files

Phase 0:
- `data/experiments/spectral_capacity_domain_rank_350m.json`
- `data/experiments/spectral_capacity_domain_rank_700m.json`
- `data/experiments/spectral_capacity_domain_rank_1p2b.json`
- `data/experiments/spectral_capacity_domain_rank_crossscale.json`

Part A:
- `data/experiments/spectral_energy_curves_350m.json`
- `data/experiments/spectral_energy_curves_700m.json`
- `data/experiments/spectral_energy_curves_1p2b.json`

Part B:
- `data/experiments/activation_spectral_350m.json`
- `data/experiments/activation_spectral_700m.json`
- `data/experiments/activation_spectral_1p2b.json`

## Related

- [positive_geometry_scale_comparison.md](positive_geometry_scale_comparison.md) — Source of the (debunked) domain rank signatures; includes revalidated table with corrected masks
- [qproj_head_structure.md](qproj_head_structure.md) — Follow-up investigation of the q_proj rank-126 inflection (Finding: 2 × head_dim - 2 architectural signature)
- [POSITIVE-GEOMETRY-ANALYSIS.md](POSITIVE-GEOMETRY-ANALYSIS.md) — Positive geometry methodology
