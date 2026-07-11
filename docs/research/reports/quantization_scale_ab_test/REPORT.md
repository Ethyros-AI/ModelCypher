# Quantization Scale A/B Test

Retained family status: `canonical`

## What This Bundle Keeps

- Canonical paired run summary:
  `results/quantization_scale_ab_test/20260226T193750Z/scale_ab_test.json`

This family now keeps the measured paired comparison and deletes the raw
standard-scale and geometric-scale adapter dumps.

## Key Measurements

- FP baseline perplexity: `10.679181098937988`
- Quantized baseline perplexity: `18.51024054001631`
- Standard-scale post perplexity: `3.4722976696325394`
- Geometric-scale post perplexity: `4.007266763845798`
- Standard-scale mean CKA: `0.9735476020592705`
- Geometric-scale mean CKA: `0.9926576833291525`
- Standard-scale min CKA: `0.6519503539321494`
- Geometric-scale min CKA: `0.8838577248580692`
- Spectral bounds: `standard=OK`, `geometric=OK`

Observed verdict from the retained run:

`ppl(geometric)=4.0073`, `ppl(standard)=3.4723`, `delta=+0.5350`
(`+15.41%`). The geometric arm preserved representation similarity better, but
the standard-scale arm achieved lower post-training perplexity in this run.

## Deleted Raw Artifacts

- `results/quantization_scale_ab_test/20260226T193750Z/adapter_standard_scale`
- `results/quantization_scale_ab_test/20260226T193750Z/adapter_geometric_scale`

Retained adapter fingerprints:

- standard-scale adapter SHA256:
  `f4eb3e844b5a7c8d27eb42a7fdfbc09566b94b399ed75ef76923b60d88222983`
- geometric-scale adapter SHA256:
  `89e7e18f8390efe0197d00aa075fccf67c3f15c90ad58f2da5293c793aefd492`

The deleted adapter payloads accounted for about `598.58 MB` of `.safetensors`
files and are now represented only through the retained comparison JSON.
