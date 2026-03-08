# Tangent Subspace ID Mechanism Repaired Rerun Checkpoint (2026-03-08)

Status: Repaired rerun checkpointed, not yet closed

## Scope

- Protocol: `docs/research/TANGENT-SUBSPACE-ID-FALSIFIER-PROTOCOL.md`
- Frozen manifest: `results/tangent_subspace_id_mechanism/20260308_142536/probe_manifest.json`
- Checkpointed rerun artifact:
  `results/tangent_subspace_id_mechanism/20260308_095800_checkpointed/`

This rerun uses the repaired measurement harness:

- matched-rank `shared_rotation`
- asymmetric `added_direction_signal`
- persisted Measurement B telemetry (`anchor_count`, `neighbor_count`,
  `tangent_rank`, `coverage`)
- Measurement C explicitly marked `[MEASUREMENT_INVALID]` for TwoNN causal
  adjudication
- no in-script pass/fail gates

The frozen manifest contains `324` atlas-backed probes. That count is derived
from the historical Llama baseline:

```text
max non-stage-0 TwoNN ID = 8.992...
ceil(max ID) = 9
neighbor_count = floor(sqrt(N))
tangent_rank = floor(neighbor_count / 2)
first acceptable N = (2 * 9)^2 = 324
```

## Checkpointed Results

The checkpointed artifact currently contains completed rerun results for:

- `LFM2-350M`
- `Qwen3.5-0.8B`

Artifact state:

- `results.json`: present
- `falsifier_outcome.json`: present
- `probe_manifest.json`: present
- `metadata.run_complete = false`

Stage-0-excluded correlation summaries from the checkpointed artifact:

| Model | shared rotation vs `|ΔID|` | added-direction energy vs positive `ΔID` | local angle vs `|ΔID|` |
|------|-----------------------------|-------------------------------------------|------------------------|
| `LFM2-350M` | `r=-0.22`, `p=0.43`, `n=15` | `r=+0.10`, `p=0.77`, `n=11` | `r=+0.51`, `p=0.052`, `n=15` |
| `Qwen3.5-0.8B` | `r=+0.32`, `p=0.14`, `n=23` | `r=+0.70`, `p=0.011`, `n=12` | `r=+0.57`, `p=0.0044`, `n=23` |

Current checkpoint interpretation:

- `shared_rotation`: still `[EXPLORATORY]`
- `added_direction_signal`: now directly measured; Qwen shows a positive signal,
  but promotion remains blocked
- `local_tangent_misalignment`: still `[EXPLORATORY]`
- `local_rank_change`: `[MEASUREMENT_INVALID]`
- overall: `[MECHANISM_UNKNOWN]`

## Current Blocker

`Llama-3.2-3B` still stalls in the local-tangent operator at the derived
`324`-probe budget. The repaired harness now supports:

- partial checkpointing after each completed model
- `--resume` from an existing partial `results.json`
- backend cache clearing between Measurement B layer pairs

Those changes were sufficient to preserve LFM2 and Qwen artifacts, but they did
not yet close the Llama leg.

## Next Falsifier

The next repair target is the operator path inside
`src/modelcypher/core/domain/geometry/tangent_space_alignment.py`:

1. instrument per-layer-pair progress and memory
2. release backend intermediates inside the tangent-alignment core, not only in
   the outer harness
3. resume the checkpointed rerun until `metadata.run_complete = true`

Until that is done, the repaired rerun should be treated as checkpointed but
incomplete.
