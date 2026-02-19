# Positive Geometry Scale Comparison (LFM2 350M/700M/1.2B)

This note records a scale comparison of positive-geometry signatures across three LFM2 model sizes (350M, 700M, 1.2B). The goal is to isolate domain-specific signatures that are invariant vs scale-dependent.

## Method

Commands (per model, per domain):

```
poetry run mc analyze concept-volume <MODEL_PATH> \
  --layers 7,8 --domains <DOMAIN_SET> --probe-count 256 \
  --rank-source spectral-gap --max-minors 256
```

Settings:
- Rank source: `spectral-gap`
- Minor selection: `lexicographic`
- Max minors: `256`
- Layers: `7,8`
- Domains: `mathematical,logical`, `linguistic,mental`, `computational,structural`,
  `affective,relational`, `temporal,spatial`, `moral`, `safety`, `philosophical`,
  `physical`, `factual`

### Source files

350M:
- `data/experiments/positive_geometry_lfm2_350m_domains_math.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_linguistic.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_computational.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_affective.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_temporal_spatial.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_moral.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_safety.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_philosophical.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_physical.json`
- `data/experiments/positive_geometry_lfm2_350m_domains_factual.json`

700M:
- `data/experiments/positive_geometry_lfm2_700m_domains_math.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_linguistic.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_computational.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_affective.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_temporal_spatial.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_moral.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_safety.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_philosophical.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_physical.json`
- `data/experiments/positive_geometry_lfm2_700m_domains_factual.json`

1.2B:
- `data/experiments/positive_geometry_lfm2_1p2b_domains_math.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_linguistic.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_computational.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_affective.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_temporal_spatial.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_moral.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_safety.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_philosophical.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_physical.json`
- `data/experiments/positive_geometry_lfm2_1p2b_domains_factual.json`

## Compact comparison table

Format per cell: `rank,posFraction,signEntropy,zeroFraction`.

```
domain  layer  350M(rank,pos,ent,zero)         700M(rank,pos,ent,zero)         1.2B(rank,pos,ent,zero)
math    7      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,0.0481,0.1928,0.0000
math    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
ling    7      126,0.4688,0.6912,0.0000       126,0.5000,0.6931,0.0000       126,0.5000,0.7160,0.0039
ling    8      126,0.4297,0.7066,0.0039       126,0.4688,0.6912,0.0000       126,0.4844,0.6927,0.0000
comp    7      211,0.4922,0.6930,0.0000       211,0.5352,0.6907,0.0000       211,0.5156,0.7327,0.0078
comp    8      211,0.4961,0.7334,0.0078       211,0.4688,0.6912,0.0000       211,0.5273,0.6917,0.0000
aff     7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
aff     8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
temp    7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
temp    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
moral   7      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
moral   8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
safety  7      1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
safety  8      1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
phil    7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
phil    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
phys    7      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
phys    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
factual 7      1,1.0000,0.0000,0.0000         255,0.0039,0.0511,0.9922       255,0.0039,0.0511,0.9922
factual 8      255,0.0039,0.0511,0.9922       255,0.0039,0.0511,0.9922       255,0.0039,0.0511,0.9922
```

## Ordering sensitivity (all‑probe, not domain‑sliced)

These summarize 32 shuffle sweeps per layer (seeds 0–3, shuffle_count=8 each) for the full atlas (no domain filter). Values are the mean and variance over shuffles.

Layer 7:
- posFraction mean=0.6212158203125 var=0.07018326222896576
- signEntropy mean=0.5402201859933484 var=0.030728278355419473
- pluckerNorm mean=0.08339609170798212 var=0.00175949450644381

Layer 8:
- posFraction mean=0.4560546875 var=0.08335304260253906
- signEntropy mean=0.5465087199750087 var=0.031979385325156276
- pluckerNorm mean=0.08431766764260828 var=0.0018093362178688138

Shuffle files:
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer7.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer7_seed1.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer7_seed2.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer7_seed3.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer8.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer8_seed1.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer8_seed2.json`
- `data/experiments/positive_geometry_lfm2_350m_shuffle_layer8_seed3.json`

## Logical conclusions from the data

These conclusions are restricted to what the measurements support.

1) **Domain‑specific rank fingerprints recur across scales.**
   - Linguistic/mental rank is 126 at layers 7 and 8 for all three scales.
   - Computational/structural rank is 211 at layers 7 and 8 for all three scales.
   - Factual rank is 255 at layer 8 for all three scales; for layer 7 it is 255 at 700M and 1.2B, while 350M is rank 1 (the only exception in this table).

2) **Sign entropy for high‑rank domains is near‑consistent across scales.**
   - Linguistic/mental signEntropy is ~0.69–0.72 across scales at both layers.
   - Computational/structural signEntropy is ~0.69–0.73 across scales at both layers.

3) **Many domains are rank‑1 across scales, but sign orientation is not invariant.**
   - Affective/relational, temporal/spatial, moral, safety, philosophical, physical, and math are rank‑1 across all scales in this table.
   - For these rank‑1 domains, the sign orientation (pos vs neg) flips across scales and layers (e.g., math, moral, safety, physical, philosophical at layer 8). The measurements show sign consistency is not preserved across scale for these domains.

4) **Factual domain shows a repeated high‑rank + high‑zero pattern at layer 8.**
   - All three scales show rank=255 and zeroFraction=0.9922 at layer 8 with posFraction=0.0039 and signEntropy=0.0511.
   - This same pattern appears at layer 7 for 700M and 1.2B, but not for 350M (rank=1 at 350M layer 7).

5) **Probe ordering measurably affects positive‑geometry statistics.**
   - Across shuffles (full‑atlas, layers 7 and 8), posFraction and signEntropy vary with non‑zero variance.
   - This implies single‑ordering measurements are not sufficient to characterize sign statistics; a distribution over shuffles is the more stable signature.

## What this implies for invariance vs scale effects

Based only on the data above:
- **Scale‑invariant signals (observed across 350M → 1.2B):**
  - Domain‑specific rank signatures for linguistic/mental (126), computational/structural (211), and factual at layer 8 (255 with high zeroFraction).
  - SignEntropy ranges for linguistic/mental and computational/structural remain close across scales.
- **Scale‑variable signals:**
  - Sign orientation in rank‑1 domains (posFraction vs negFraction) changes across scales and layers.
  - Factual domain rank at layer 7 differs between 350M and larger scales.
  - Minor zeroFraction in linguistic/computational domains appears at 1.2B in some layers, but not consistently across all scales.

These points are consistent with the idea that **domain‑conditioned rank patterns are stable across scale**, while **sign orientation is not**. The invariance claim is supported by repeated ranks in the high‑rank domains and the repeated factual pattern at layer 8. The variability claim is supported by sign flips in rank‑1 domains and by shuffle‑driven variance.

## CORRECTION (2026-02-19): Domain ranks are probe-count artifacts

Follow-up investigation (`spectral_capacity_domain_rank.md`) confirmed that the "invariant domain rank signatures" are determined by the number of probes per domain group, not by model geometry:

| Domain group | Probe count | Reported rank | Relationship |
|---|---|---|---|
| linguistic + mental | 131 | 126 | 131 - 5 (noise) |
| computational + structural | 213 | 211 | 213 - 2 (noise) |
| factual | 256 (capped at --probe-count) | 255 | 256 - 1 (noise) |

The `spectral-gap` rank method finds the drop-off at the tail of the SVD spectrum, which occurs at approximately n_probes - c. The "scale invariance" is trivial: same probes → same matrix size → same tail position.

This does NOT invalidate the positive geometry signatures (sign entropy, positive fraction, Plucker norm). Only the rank numbers are artifacts.

## REVALIDATION (2026-02-19): Corrected LFM2 mask routing

The original data above was collected with a bug in `forward_through_backbone()`: it applied a numeric causal mask to ALL layers. LFM2 hybrid models need string `"causal"` for attention layers and `None` for conv layers. The bug was fixed in `_resolve_layer_mask()` (see `spectral_capacity_domain_rank.md` Finding 5).

**Script**: `scripts/revalidate_positive_geometry.py`
**Data**: `data/experiments/revalidated_positive_geometry_*_domains_*.json`
**Original (buggy) data**: `data/experiments/original_positive_geometry/` (recovered from git `4a27a695^`)

### Corrected comparison table

Format per cell: `rank,posFraction,signEntropy,zeroFraction`.

```
domain  layer  350M(rank,pos,ent,zero)         700M(rank,pos,ent,zero)         1.2B(rank,pos,ent,zero)
math    7      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
math    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
ling    7      127,0.5195,0.7323,0.0078       127,0.4805,0.7154,0.0039       127,0.5195,0.6924,0.0000
ling    8      127,0.4414,0.6863,0.0000       127,0.5117,0.6929,0.0000       127,0.4258,0.7055,0.0039
comp    7      211,0.5234,0.7147,0.0039       211,0.4922,0.7159,0.0039       211,0.5078,0.6930,0.0000
comp    8      211,0.5078,0.6930,0.0000       211,0.4609,0.6901,0.0000       211,0.5352,0.6907,0.0000
aff     7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
aff     8      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
temp    7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
temp    8      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
moral   7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
moral   8      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
safety  7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
safety  8      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
phil    7      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
phil    8      1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
phys    7      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000
phys    8      1,1.0000,0.0000,0.0000         1,0.0000,0.0000,0.0000         1,1.0000,0.0000,0.0000
factual 7      1,1.0000,0.0000,0.0000         254,0.0469,0.3875,0.9023       254,0.0469,0.3875,0.9023
factual 8      1,1.0000,0.0000,0.0000         254,0.0352,0.3837,0.9023       254,0.0430,0.3869,0.9023
```

### What changed with corrected masks

**Major changes:**

1. **Linguistic rank: 126 → 127.** All 3 models, both layers. The spectral-gap boundary shifted by 1.

2. **Factual @ 350M: collapsed from rank 255 → rank 1.** The "factual shows high-rank + high-zero across all scales at layer 8" was WRONG for 350M. With correct mask routing, 350M factual is rank 1 at both layers — the high-rank pattern was entirely a mask bug artifact at this scale.

3. **Factual @ 700M/1.2B: rank 255 → 254.** Still high-rank but shifted by 1. Sign statistics changed substantially: signEntropy 0.051 → 0.387, zeroFraction 0.992 → 0.902.

4. **Rank-1 domain sign flips stabilized.** With corrected masks, 350M and 700M rank-1 domains mostly show posFrac=1.0 (positive orientation). The original data had many posFrac=0.0 entries in 700M that now read 1.0 (affective L8, temporal L8, moral, safety, philosophical L8).

**Unchanged:**

- Computational rank stays 211 at all scales.
- Sign entropy for linguistic (~0.69-0.73) and computational (~0.69-0.72) remains consistent across scales.
- Math, affective, temporal, moral, safety, philosophical, physical all remain rank 1.

### Revised conclusions

1. **Domain-specific rank fingerprints are still probe-count artifacts** — ranks shifted by exactly 1 (126→127, 255→254), confirming they track n_probes - c.

2. **The "factual high-rank pattern at all scales" is partially a mask artifact.** It persists at 700M and 1.2B (rank 254) but NOT at 350M (rank 1). The 350M factual collapse suggests the mask bug was creating artificial spectral spread in the activations.

3. **Sign orientation for rank-1 domains is MORE consistent with corrected masks.** Most rank-1 domains at 350M and 700M now show the same sign (posFrac=1.0). The original sign flips were partly caused by buggy activations.

4. **Sign entropy for high-rank domains (linguistic, computational) is robust to the mask fix.** The ~0.69 signEntropy is a genuine signal, not a mask artifact.

## Notes and constraints

- These signatures depend on probe ordering (lexicographic selection over the atlas order). Ordering sensitivity is documented above.
- All values are raw measurements; no thresholds or qualitative labels are applied. The corrected table is the source of truth.
- **Domain rank values are probe-count artifacts.** See `spectral_capacity_domain_rank.md` for full analysis.
- **Original (buggy) data** was collected with numeric causal mask for all layers. See `spectral_capacity_domain_rank.md` Finding 5 for the bug description.
