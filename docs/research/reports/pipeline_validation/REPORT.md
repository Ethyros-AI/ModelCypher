# Pipeline Validation

Retained family status: `canonical`

## What This Bundle Keeps

- Aggregate verdict:
  `results/pipeline_validation/verdict.json`
- Per-scale trial summary:
  `results/pipeline_validation/350M/result.json`

This family now keeps the aggregate verdict and retained `trial_results`
measurements. The raw phase-5 adapter payloads and runner logs are deleted.

## Key Measurements

Aggregate verdict (`verdict.json`):

- timestamp: `2026-02-25T01:45:32.374221+00:00`
- git hash: `3808b5a5`
- trials per model: `5`
- `all_pass = false`
- `all_structural_pass = true`
- `all_inference_pass = false`

Per-scale summary (`350M/result.json`):

- pass count: `3 / 5`
- structural pass count: `5 / 5`
- inference pass count: `3 / 5`
- phase-5 probe count: `10`
- phase-5 probe seed: `3475334679`
- mean loss delta: `0.9840814639514399`
- min loss delta: `0.5715736496235642`
- mean perplexity delta: `12.567903220520218`
- min perplexity delta: `8.874624862846403`

Worst-case trial diagnostics among the retained `trial_results`:

- lowest min CKA:
  `0.9323521445735921` at layer `15` on trial `0` / seed `4231027559`
- max blindness ratio:
  `17.356273235045485` on trial `1` / seed `4231027560`
- minimum behavioral-preserved null-access fraction:
  `0.005462362115171986` at layer `8` on trial `1` / seed `4231027560`
- largest loss delta:
  `1.1657124188826196` on trial `0` / seed `4231027559`
- largest perplexity delta:
  `14.030564000954609` on trial `0` / seed `4231027559`

## Failure Cases Retained In Summary

- trial `0` / seed `4231027559`
  - failure modes: `online_eval_degraded`, `fourgram_degenerated`
  - stop reason:
    `certificate (‖g‖=8.15e-01, Δmax=0.00e+00<CI=3.12e-01, epoch=10)`
  - cooccurrence class: `cka_shift_and_inference_degraded`
  - `loss_delta = 1.1657124188826196`
  - `perplexity_delta = 14.030564000954609`
  - `min_cka = 0.9323521445735921` at layer `15`
  - `online_eval_delta_correct = -1`
  - `max_4gram_repeat_delta = 0.11228338863836185`
  - `null_access_min_behavioral_preserved_fraction = 0.006602507612182352`
    at layer `8`
  - `cka_blindness_ratio = 14.165508425614297` at layer `7`
  - `margin_mean_delta = 0.6687500000000001`
- trial `1` / seed `4231027560`
  - failure modes: `online_eval_degraded`
  - stop reason:
    `online_eval_degraded (stage=pre_outcome, 1/2 correct, epoch=1)`
  - cooccurrence class: `cka_shift_and_inference_degraded`
  - `loss_delta = 0.5715736496235642`
  - `perplexity_delta = 8.874624862846403`
  - `min_cka = 0.9443118206570901` at layer `15`
  - `online_eval_delta_correct = -1`
  - `max_4gram_repeat_delta = -0.17152412804586725`
  - `null_access_min_behavioral_preserved_fraction = 0.005462362115171986`
    at layer `8`
  - `cka_blindness_ratio = 17.356273235045485` at layer `7`
  - `margin_mean_delta = -0.10625000000000007`

## Deleted Raw Artifacts

- `results/pipeline_validation/350M/phase5_artifacts`
- `results/pipeline_validation/350M/run_restart.log`
- `results/pipeline_validation/run.log`

The deleted `phase5_artifacts` tree contained `5` raw adapter runs totaling
about `192.79 MB` of `.safetensors` files plus matching config and geometry
manifests. Those runs are now represented through the retained `trial_results`
payload in `350M/result.json`.

## CLI Read-Side

Use the shared report reader for quick R2 scans instead of opening the retained
JSON files directly:

```bash
poetry run mc --output text analyze report --bundle results/pipeline_validation
poetry run mc --output text analyze report --bundle results/pipeline_validation_blindness_350M_t20
```

That shared read-side now covers:

- failing retained pipeline-validation families such as `results/pipeline_validation`
- all-pass blindness families such as `results/pipeline_validation_blindness_350M_t20`
- the same `workflow`, `summary`, `sections`, and `markdown` envelope already
  used for observation bundles and retained measurement-atlas artifacts
