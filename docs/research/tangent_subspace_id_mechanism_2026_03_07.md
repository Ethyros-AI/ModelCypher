# Tangent Subspace ID Mechanism Historical Note (2026-03-07)

Status: Historical pre-protocol run, not promotable

Scope:
- 3 models: `LFM2-350M`, `Qwen3.5-0.8B`, `Llama-3.2-3B`
- 60 hand-written legacy prompts
- 3 measurement channels

## What This Artifact Still Supports

The historical run is retained because it provides the baseline evidence that
triggered the recovery work:

- `P1` shared-rotation signal was mixed across models and stage-0-sensitive.
- `P4` local tangent misalignment looked promising on LFM2 and Qwen, but failed
  on Llama.
- Llama's non-stage-0 TwoNN peak was approximately `8`, which now drives the
  repaired rerun probe budget of `256` under the current local-tangent operator.

Historical stage-0-excluded raw correlations from the preserved artifact:

| Observable | LFM2-350M | Qwen3.5-0.8B | Llama-3.2-3B |
|-----------|-----------|--------------|--------------|
| shared rotation vs `|ΔID|` | `r=+0.03`, `p=0.91` | `r=+0.59`, `p=0.003` | `r=+0.17`, `p=0.41` |
| local angle vs `|ΔID|` | `r=+0.54`, `p=0.037` | `r=+0.69`, `p=0.0003` | `r=+0.18`, `p=0.38` |

## What This Artifact Does Not Support

These conclusions are no longer promotable from the 2026-03-07 run:

- `P2 / novel direction count` as a clean elimination or confirmation
- `P5 / local rank change` as a clean elimination
- any causal explanation for the Llama null based on `~4 neighbors` or `5-9D`
  tangent spaces
- any cross-model promotion using literal `0.3` thresholds or median-split gates

Current doctrine:

- `P2` is raw observation only on the historical artifact
- `P5` is `[MEASUREMENT_INVALID]` for TwoNN causal adjudication
- `P4` remains `[EXPLORATORY]`
- overall status remains `[MECHANISM_UNKNOWN]`

## Current Path Forward

Use the repaired rerun path instead:

- Protocol: `docs/research/TANGENT-SUBSPACE-ID-FALSIFIER-PROTOCOL.md`
- Historical baseline artifact: `results/tangent_subspace_id_mechanism/results.json`
- Repaired reruns: `results/tangent_subspace_id_mechanism/<run_id>/`
