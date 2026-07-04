# G5 8B Validation Memory Test Report

Updated: 2026-03-08

## Question

What memory footprint and capacity-profile summary were observed for the
Qwen3-8B bf16 validation path before the full G5 run proceeds?

## Retained Evidence

- Family summary JSON: `results/g5_8b_validation_memtest/summary.json`

## Capacity Summary

- `totalParameters`: `8190735360`
- `analyzedParameters`: `8190427136`
- `analyzedLayers`: `254`
- `meanEffectiveRank`: `2464.9345855118513`
- `meanCapacityUtilization`: `0.7803141310980471`
- `medianCapacityUtilization`: `0.8412397464122264`
- `referenceRankDimension`: `4096`
- decay types: `215` `gradual_slope`, `39` `sharp_cliff`
- lowest-capacity layer:
  `model.layers.15.self_attn.q_proj.weight` with utilization
  `0.5204164386047629`
- highest-capacity layer:
  `model.layers.22.self_attn.v_proj.weight` with utilization
  `0.9263838337790584`
- positive F32 null-space fraction appears in `72 / 254` layers; maximum
  `0.012939453125` at `model.layers.0.self_attn.q_proj.weight`
- module mean capacity utilization:
  - `q_proj`: `0.5800842300696467`
  - `o_proj`: `0.6317618157947125`
  - `k_proj`: `0.7924555689569664`
  - `v_proj`: `0.8869873586354484`
  - `mlp.gate_proj`: `0.8298019733459924`
  - `mlp.down_proj`: `0.8678517374731078`
  - `mlp.up_proj`: `0.8679151478503326`
  - `lm_head`: `0.9242858508788737`

## Memory Trace

- start: RSS `0.0461` GB, backend peak `0.0000` GB, system available
  `111.1524` GB
- after_capacity: RSS `6.3902` GB, backend peak `4.9455` GB, system available
  `104.6764` GB

## Verdict

Keep this family as `summary_only`. It records an 8B preflight capacity and
memory checkpoint, not a completed validation result. The large `capacity_report`
JSON was useful for extracting the aggregate structure above, but it does not
need to stay in the worktree once that structure is summarized.

## Cleanup Performed

- Extracted the retained quantities above into `summary.json`.
- Deleted `seed41/capacity_report.json` after preserving the aggregate and
  layer-extrema statistics in the family summary.
- Deleted `seed41/memory_trace.json`; both recorded snapshots are summarized
  above.
- Deleted `seed41/run.log`; it only captured setup plus the two retained memory
  checkpoints.
- Total deleted raw payload: about `65.56 MB`.
