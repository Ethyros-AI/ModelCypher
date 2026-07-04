# Measurement Atlas Family Report

This directory retains the two April 2, 2026 atlas bundles that close the
live/replay alignment bug on the shipped 350M study pack.

## Retained Bundles

- Pre-fix bundle:
  [`20260402T145540Z-measurement-atlas`](./20260402T145540Z-measurement-atlas/REPORT.md)
- Fixed bundle:
  [`20260402T150954Z-measurement-atlas`](./20260402T150954Z-measurement-atlas/REPORT.md)

Both runs kept the same study pack, model family, and artifact contract:

- `studyCount = 3`
- `variantCount = 16`
- `comparisonCount = 10`
- `onsetEventCount = 22`
- `errorCount = 0`

## Mechanism

The pre-fix bundle was invalid for replay-alignment conclusions because replay
did not preserve the realized continuation token path:

- `full` replay text was built from raw string concatenation, so prompt-final
  and response-initial tokens could collapse into a different tokenization
- standalone replay of the response could reintroduce BOS behavior or newline
  retokenization that was not part of the live decode path

The fixed bundle closed that bug by replaying the exact continuation token ids
and deriving the `full` replay region from the exact concatenated token path
instead of re-encoding `prompt + generated_text`.

## Agreement Delta

| Study | Pre-fix | Fixed |
| --- | --- | --- |
| `measurement_atlas_casing` | `0/4` | `4/4` |
| `measurement_atlas_grounded_hallucination` | `0/2` | `2/2` |
| `measurement_atlas_profanity_tone` | `1/4` | `4/4` |

Earliest divergence steps and grounded-onset counts remained stable where they
should have:

- casing: divergence `1`, grounded onsets `0`
- grounded hallucination: divergence `27`, grounded onsets `2`
- profanity tone: divergence `0`, grounded onsets `0`

## Current Read

Use `20260402T150954Z-measurement-atlas` as the retained proof that replay
alignment is closed on the shipped 350M atlas pack.

Use `20260402T145540Z-measurement-atlas` as the retained pre-fix counterexample
showing why raw text concatenation and BOS/newline retokenization were not
safe for replay-boundary claims.

## Contract-Polish Confirmation

The contract-polish rerun
[`20260402T155048Z-measurement-atlas`](./20260402T155048Z-measurement-atlas/REPORT.md)
kept the fixed agreement counts and `errorCount = 0` while updating the bundle
contract to:

- variant `decode.replaySpaces = ["hidden", "embedding"]`
- variant `decode.liveSpaces = ["hidden"]`
- `run_manifest.frozenSurfaces` records requested vs observed spaces separately
  under schema `mc.measurement_atlas.run_manifest.v2`

## Raw-Locus Cleanup Confirmation

The raw-locus cleanup rerun
[`20260402T160859Z-measurement-atlas`](./20260402T160859Z-measurement-atlas/REPORT.md)
kept the same agreement counts and `errorCount = 0` while making the machine-
readable atlas rows sentinel-free:

- `sequence_metrics.jsonl` now emits `peakLayer = null` /
  `firstBendLayer = null` for embedding rows
- embedding rows now carry explicit `peakLocus = "embedding"` and
  `firstBendLocus = "embedding"`
- `comparisons.jsonl` no longer emits numeric embedding layer deltas and now
  records non-numeric locus checks under `locusComparisons`
- the retained rerun contains no raw `-1` or `"None"` strings in
  `sequence_metrics.jsonl` or `comparisons.jsonl`

## CLI Read-Side

`mc analyze report --bundle ...` now reads both retained atlas manifest
generations:

- `mc.measurement_atlas.run_manifest.v1`
- `mc.measurement_atlas.run_manifest.v2`

Use that shared CLI read-side for quick scans instead of opening the JSONL
files directly when the question is “what moved first, where, and how cleanly
did replay agree with live?”

Use `20260402T160859Z-measurement-atlas` as the sentinel-free read-side
reference bundle. It preserves the fixed agreement counts while also carrying
explicit locus fields and requested-vs-observed replay/live surface reporting.
