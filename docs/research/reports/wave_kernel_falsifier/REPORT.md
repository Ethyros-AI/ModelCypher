# Wave Kernel Falsifier

Retained family status: `summary_only`

## What This Bundle Keeps

- Promotable falsifier run summary:
  `results/wave_kernel_falsifier/full_promotable_run_20260306/`
- Distance-bias enriched falsifier run summary:
  `results/wave_kernel_falsifier/full_promotable_run_20260307_distance_bias/`

Each retained run keeps:

- `run_manifest.json`
- `model_family_summary.json`
- `falsifier_outcome.json`
- `artifact_validation.json`

This family keeps the retained falsifier summaries, not the large per-head
tables or the superseded smoke run.

## Retained Outcome

Both retained promotable runs reach the same top-line result:

- `overall = falsified_by_decay`
- `promotion_blocked = true`
- reason:
  `All families favored M1 on non-boundary holdout heads.`

Family-level retained outcomes:

- `full_promotable_run_20260306`
  - probes: `24`
  - `LFM2`: `decay_favored`, non-boundary heads `123`
  - `Llama`: `decay_favored`, non-boundary heads `664`
  - `Qwen`: `decay_favored`, non-boundary heads `7`
- `full_promotable_run_20260307_distance_bias`
  - probes: `24`
  - same falsifier verdict and family directions as `20260306`
  - adds retained distance-bias diagnostics such as:
    `mean_content_residual_variance_fraction`,
    `mean_calibration_holdout_weighted_correlation`, and
    `mean_calibration_positive_slope_mass`

## Deleted Raw Or Superseded Artifacts

- `results/wave_kernel_falsifier/manual_smoke_20260306`
- `results/wave_kernel_falsifier/full_promotable_run_20260306/per_head_fit_table.jsonl`
- `results/wave_kernel_falsifier/full_promotable_run_20260307_distance_bias/per_head_fit_table.jsonl`

The deleted payload is about `40.22 MB`:

- superseded smoke run total: `10.01 MB`
- removed per-head table from `full_promotable_run_20260306`: `14.93 MB`
- removed per-head table from `full_promotable_run_20260307_distance_bias`:
  `15.28 MB`

Those deletions keep the falsifier conclusion and family-level measurements
while removing the bulky repeated per-head tables and the earlier smoke pass.
