# Continual Learning

Retained family status: `canonical`

## What This Bundle Keeps

- Cross-architecture aggregate:
  `results/continual_learning/cross_architecture_comparison.json`
- Cross-scale check:
  `results/continual_learning/exp4/exp4_cross_scale_results.json`
- Sequential-forgetting summaries:
  - `results/continual_learning/exp1/seed42/exp1_results.json`
  - `results/continual_learning/exp1/R1/seed42/exp1_results.json`
  - `results/continual_learning/exp1/R_L1/seed42/exp1_results.json`
  - `results/continual_learning/exp1/R_Q1/seed42/exp1_results.json`
  - `results/continual_learning/exp1/R_Q2/seed42/exp1_results.json`
  - `results/continual_learning/exp1/R_Q3/seed42/exp1_results.json`
- Retained gradient-probe summary:
  `results/continual_learning/exp2/R_L1_grad/seed42/exp2_results.json`

The worktree now keeps the small measured summaries and deletes the raw adapter
dumps plus oversized per-layer capacity payloads that were already subsumed by
the retained aggregates.

## Key Measurements

Sequential forgetting on the base 350M run (`exp1/seed42`):

- task count: `2`
- per-task mean CKA: `0.990871`, `0.993187`
- per-task min CKA: `0.958851`, `0.947802`
- Weyl accumulation: `3.685967`
- incremental new dims: `24`, `0`
- depletion rate: `0.0`

Sequential shard variants on `exp1`:

- `R1`: mean/min CKA `0.984744` / `0.963119`, Weyl accumulation `1.193076`,
  incremental new dims `47`
- `R_L1`: mean/min CKA `0.998537` / `0.995661`, Weyl accumulation `0.143859`,
  incremental new dims `49`
- `R_Q1`: mean/min CKA `0.995382` / `0.981390`, Weyl accumulation `3.305492`,
  incremental new dims `273`
- `R_Q2`: mean/min CKA `0.997147` / `0.982820`, Weyl accumulation `1.647738`,
  incremental new dims `254`
- `R_Q3`: mean/min CKA `0.995983` / `0.981525`, Weyl accumulation `2.164314`,
  incremental new dims `273`

Cross-architecture null-space summary from the retained aggregate:

- `base`: params `354418688`, layers `93`, mean null rank `0.559140`
- `1.2B`: params `1170210816`, layers `93`, mean null rank `1.129032`
- `llama32_3b`: params `3212574720`, layers `197`, mean null rank `5.477157`
- `mistral_7b`: params `1132462080`, layers `678`, mean null rank `0.153392`
- `qwen25_3b`: params `3085697024`, layers `253`, mean null rank `1.296443`
- `qwen3_8b`: params `8190427136`, layers `254`, mean null rank `3.421260`

Retained gradient probe summary (`exp2/R_L1_grad`):

- total tail dims: `110`
- cumulative union rank: `49`
- max nights lower bound: `2`
- grad-rank fraction: `0.445455`

Cross-scale check (`exp4`):

- invariance factor: `1.0`
- `h4_passed = true`

## Deleted Raw Artifacts

- all `exp1/*/seed42/adapter_task_*` directories
- all `exp1/*/seed42/capacity_checkpoint.json` files
- raw-only `exp1/R3/seed42` after its unsummarized adapter and checkpoint were
  removed
- `exp2/seed42`
- `exp2_1.2B/seed42`
- `exp2_llama32_3b/seed42`
- `exp2_mistral_7b/seed42`
- `exp2_qwen25_3b/seed42`
- `exp2_qwen3_8b/seed42`
- `exp2/R_Q1_grad/seed42`
- `exp2/R_L1_grad/seed42/capacity_checkpoint.json`

These deletions remove raw adapters and per-layer capacity payloads while
keeping the retained measurement summaries needed to avoid rerunning the same
exploratory work.
