# Positive Geometry Scale Comparison (LFM2 350M/700M/1.2B)

This note records a scale comparison of positive-geometry signatures across three LFM2 model sizes (350M, 700M, 1.2B). The goal is to isolate domain-specific signatures that are invariant vs scale-dependent.

## Method

Commands (per model, per domain):

```
poetry run mc geometry research positive-geometry <MODEL_PATH> \
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

## Notes and constraints

- These signatures depend on probe ordering (lexicographic selection over the atlas order). Ordering sensitivity is documented above.
- All values are raw measurements; no thresholds or qualitative labels are applied. The table is the source of truth.  
