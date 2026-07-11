# Null-Space Investigation

Retained family status: `summary_only`

## What This Bundle Keeps

- Investigation output:
  `results/null_space_investigation/investigation_results.json`

This family keeps the retained analysis summary only. It does not keep the
original adapted trial directory that was analyzed.

## Historical Provenance

The retained `investigation_results.json` records the original adapter path used
when the analysis was run:

- `results/pipeline_validation/350M/phase5_artifacts/trial_000_seed_4231027559`

That path is now deleted under the repo retention rule. It remains in the JSON
only as historical provenance for the original run.

## Rerun Rule

`scripts/null_space_investigation.py` no longer hardcodes that adapter path.
Reruns must now pass an explicit retained trial directory:

`poetry run python scripts/null_space_investigation.py --adapter-path /path/to/trial_dir`

The supplied trial directory must contain both:

- `adapters.safetensors`
- `geometry_manifest.json`
