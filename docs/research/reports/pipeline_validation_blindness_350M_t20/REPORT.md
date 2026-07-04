# Pipeline Validation Blindness 350M T20

Retained family status: `canonical`

## What This Bundle Keeps

- Aggregate verdict:
  `results/pipeline_validation_blindness_350M_t20/verdict.json`
- Per-scale trial summary:
  `results/pipeline_validation_blindness_350M_t20/350M/result.json`

This family now keeps the aggregate trial measurements and deletes the raw
per-trial adapter artifacts plus the runner log.

## Key Measurements

Aggregate verdict (`verdict.json`):

- timestamp: `2026-02-25T03:18:28.751624+00:00`
- trials per model: `20`
- scales: `350M`
- `all_pass = true`
- `all_structural_pass = true`
- `all_inference_pass = true`

Per-scale summary (`350M/result.json`):

- pass count: `20 / 20`
- structural pass count: `20 / 20`
- inference pass count: `20 / 20`
- phase-5 probe count: `10`
- phase-5 probe seed: `3475334679`
- mean loss delta: `0.7889143830883715`
- min loss delta: `0.5232124283767878`
- mean perplexity delta: `10.887178869354896`
- min perplexity delta: `8.30431945054324`

Worst-case trial diagnostics among the retained `trial_results`:

- lowest min CKA:
  `0.9315346048482969` at layer `15` on trial `2` / seed `4231027561`
- max blindness ratio:
  `55.03113202131845` on trial `18` / seed `4231027577`
- minimum behavioral-preserved null-access fraction:
  `0.004797473250492749` at layer `8`
- best max 4-gram repeat delta remained non-degrading:
  `-0.08935629587803506` on trial `17` / seed `4231027576`

## Deleted Raw Artifacts

- `results/pipeline_validation_blindness_350M_t20/350M/phase5_artifacts`
- `results/pipeline_validation_blindness_350M_t20/run.log`

The deleted `phase5_artifacts` tree contained `20` raw adapter runs totaling
about `771.18 MB` of `.safetensors` files plus matching config and geometry
manifests. Those runs are now represented only through the retained
`trial_results` payload in `350M/result.json`.
