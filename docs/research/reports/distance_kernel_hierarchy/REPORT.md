# Distance Kernel Hierarchy

Retained family status: `summary_only`

## What This Bundle Keeps

- Promotable falsifier run summary:
  `results/distance_kernel_hierarchy/full_promotable_run_20260307/`

The retained run keeps:

- `run_manifest.json`
- `model_family_summary.json`
- `falsifier_outcome.json`
- `artifact_validation.json`

This family keeps the retained falsifier summary, not the bulky per-head
classification tables or the superseded smoke run.

## Retained Outcome

The retained promotable run reports:

- probes: `24`
- models evaluated: `3`
- `overall = partial_falsification`
- `promotion_blocked = true`
- reason: `Failed predictions: P-DKH-5`

## Deleted Raw Or Superseded Artifacts

- `results/distance_kernel_hierarchy/20260307_153241`
- `results/distance_kernel_hierarchy/full_promotable_run_20260307/per_head_classification.jsonl`

The deleted payload is about `24.29 MB`:

- superseded smoke run total: `9.68 MB`
- removed per-head classification table from `full_promotable_run_20260307`:
  `14.61 MB`

Those deletions keep the retained falsifier verdict while removing the earlier
smoke pass and the bulky per-head classification table.
