# Quantization Frontier

Retained family status: `canonical`

## What This Bundle Keeps

- Owner-local source artifact:
  `results/quantization_frontier/20260227T235714Z/quantization_frontier.json`
- Source SHA256:
  `4bfbea59445a25549b16348cd3e589ab7758b5f7cecdefab7eff32adfdb99595`

The raw result remains owner-local under the repository artifact policy. This
tracked report preserves the measurements used by the public README; it does
not upgrade the open frontier law to a validated claim.

## Key Measurements

Aggregate summary:

- retained models: `3`
- architectures: `Llama`, `Qwen`
- mean recovery ratio: `0.1415`
- `all_ppl_improved = true`
- `all_degen_improved = true`

Per-model rows:

| Model | Correction bits | Baseline CKA | Post CKA | Recovery ratio | PPL delta | Degeneration delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 4 | 0.843599 | 0.876985 | 0.2135 | -0.0409 | -0.0158 |
| Qwen3-1.7B | 4 | 0.894737 | 0.908735 | 0.1330 | -0.0633 | -0.0472 |
| Llama-3.2-3B | 4 | 0.991706 | 0.992353 | 0.0780 | -0.0804 | -0.0555 |

The negative PPL and degeneration deltas are improvements under the retained
artifact's sign convention. These three rows are evidence that the correction
surface was promising on this run family, not a cross-architecture frontier law.

## Open Falsifier

`R4` remains open until one architecture-conditioned statistic orders achieved
CKA floor, fixed-basis feature survival, and degeneration across bit-depth
sweeps and survives a held-out family. The owner-run fixed-basis packet is in
`docs/research/replication/ws4_2/fixed_basis_feature_survival.manifest.json`.
