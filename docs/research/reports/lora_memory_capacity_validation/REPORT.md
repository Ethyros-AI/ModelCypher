# LoRA Memory Capacity Validation Report

Updated: 2026-03-08

## Question

For Qwen3.5-0.8B-bf16 under the `B0` knowledge-memory sweep, how do LoRA rank
cap and training-token budget change exact-match recovery, and which artifacts
need to stay in the worktree after the sweep is summarized?

## Retained Summary Artifacts

- `results/lora_memory_capacity_validation/Qwen3.5-0.8B-bf16/geometry_table.json`
- `results/lora_memory_capacity_validation/Qwen3.5-0.8B-bf16/B0/sweep_summary.json`
- representative retained raw run:
  `results/lora_memory_capacity_validation/Qwen3.5-0.8B-bf16/B0/B0_r4_4000tok`

The retained representative run now keeps only the final adapter bundle plus the
small config and evaluation files. Intermediate checkpoint snapshots are
deleted.

## Sweep Summary

| Run ID | r_cap | tokens | exact_match_rate | train_time_s | eval_time_s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B0_r4_1000tok` | 4 | 1000 | `0.6316` | `717.77` | `16.61` |
| `B0_r4_4000tok` | 4 | 4000 | `0.8889` | `712.79` | `65.41` |
| `B0_r4_8000tok` | 4 | 8000 | `0.8860` | `711.10` | `131.29` |
| `B0_r4_16000tok` | 4 | 16000 | `0.0650` | `721.24` | `291.69` |
| `B0_r16_1000tok` | 16 | 1000 | `0.6316` | `764.40` | `17.01` |
| `B0_r16_4000tok` | 16 | 4000 | `0.0261` | `803.34` | `70.60` |
| `B0_r16_8000tok` | 16 | 8000 | `0.0000` | `937.09` | `129.86` |

## Observed Result

- Best measured arm: `B0_r4_4000tok` with exact-match rate `0.8889`
- Near-tie arm: `B0_r4_8000tok` with exact-match rate `0.8860`
- Rank-4 frontier is non-monotonic: gains rise through `4000`-`8000` tokens and
  then collapse at `16000` tokens.
- Rank-16 is not a safe default in this sweep: performance is already poor at
  `4000` tokens and fully collapsed at `8000` tokens.

## Cleanup Performed

- Retained `B0_r4_4000tok` as the representative raw artifact because it is the
  best-performing measured arm.
- Deleted superseded raw run directories:
  - `B0_r4_1000tok`
  - `B0_r4_8000tok`
  - `B0_r4_16000tok`
  - `B0_r16_1000tok`
  - `B0_r16_4000tok`
  - `B0_r16_8000tok`
  - empty `B0_r16_16000tok`
- Deleted retained-run checkpoint snapshots:
  - `0000500_adapters.safetensors`
  - `0001000_adapters.safetensors`
  - `0001500_adapters.safetensors`

`0001500_adapters.safetensors` had the same SHA256 as the retained final
`adapters.safetensors`, so the retained run now keeps only one final adapter
payload instead of four checkpoint copies.

The retained worktree now preserves the geometry table, the sweep summary, and
one inspectable final adapter artifact instead of the full checkpoint fan-out.
