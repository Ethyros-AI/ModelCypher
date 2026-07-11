# G5 8B Non-Ceiling Capacity Checkpoint Report

Updated: 2026-03-08

## Question

What did the retained non-ceiling Qwen3-8B checkpoint actually measure, and is it
enough to count as a promotable 8B validation result?

## Retained Evidence

- Family summary JSON: `results/g5_8b_validation_non_ceiling/summary.json`

## Observed Values

- Model path: `/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16`
- Analyzed parameters: `4,774,690,816 / 4,774,872,320`
- Layer reports written: `150`
- Failed layers: `0`
- Decay types: `127` `gradual_slope`, `23` `sharp_cliff`
- Mean capacity utilization: `0.7684901475908192`
- Median capacity utilization: `0.8260556509894921`
- Lowest-capacity layer:
  `model.layers.15.self_attn.q_proj.weight` with utilization
  `0.5204164386047629` and recommended rank `4095`
- Highest-capacity layer:
  `model.layers.10.self_attn.v_proj.weight` with utilization
  `0.9156246666177944` and recommended rank `1022`
- Positive F32 null-space fraction appears in `42 / 150` layers; maximum
  `0.012939453125` at `model.layers.0.self_attn.q_proj.weight`
- Module mean capacity utilization:
  - `q_proj`: `0.5690041291179133`
  - `o_proj`: `0.6172924328932269`
  - `k_proj`: `0.7903399745049369`
  - `v_proj`: `0.880277857690664`
  - `mlp.gate_proj`: `0.8023892654372644`
  - `mlp.down_proj`: `0.8572378005607079`
  - `mlp.up_proj`: `0.8541858531830672`
- Logged start-memory snapshot only:
  `rss=0.10943603515625 GB`, `system_used=40.10612487792969 GB`,
  `system_available=86.693359375 GB`

## Verdict

Keep this family as `summary_only`. The retained checkpoint is useful as a
layerwise spectral capacity snapshot, but it did not write a final gate verdict,
behavioral outcome, or completion trace. It is not promotable 8B closure
evidence.

## Cleanup Performed

- Extracted the retained quantities above into `summary.json`.
- Deleted `seed41/capacity_checkpoint.json` after preserving the aggregate and
  layer-extrema statistics in the summary bundle.
- Deleted `seed41/memory_trace.json`; it only recorded the start snapshot.
- Deleted `seed41/run.log`; it ended after the initial probe load and did not
  contain a terminal verdict.
- Total deleted raw payload: about `12.97 MB`.

## Next Falsifier

Rerun the 8B non-ceiling path only if it also writes the completion gate and
behavioral outputs needed to test whether this capacity structure predicts
preservation or failure.
