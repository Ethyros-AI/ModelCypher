# Pipeline Validation Cert 350M 5T

Retained family status: `canonical`

## What This Bundle Keeps

- Aggregate verdict:
  `results/pipeline_validation_cert_350m_5t/verdict.json`
- Per-scale trial summary:
  `results/pipeline_validation_cert_350m_5t/350M/result.json`

This family now keeps the aggregate verdict and retained `trial_results`
measurements. The raw phase-5 adapter payloads and runner log are deleted.

## Key Measurements

Aggregate verdict (`verdict.json`):

- timestamp: `2026-02-25T20:45:51.324965+00:00`
- git hash: `b65c10c6`
- trials per model: `5`
- `all_pass = false`
- `all_structural_pass = true`
- `all_inference_pass = false`

Per-scale summary (`350M/result.json`):

- pass count: `4 / 5`
- structural pass count: `5 / 5`
- inference pass count: `4 / 5`
- phase-5 probe count: `10`
- phase-5 probe seed: `3475334679`
- mean loss delta: `0.8378183838497835`
- min loss delta: `0.5389641912752872`
- mean perplexity delta: `11.330377782379799`
- min perplexity delta: `8.520545504278582`

Worst-case trial diagnostics among the retained `trial_results`:

- lowest min CKA:
  `0.9507843733329425` at layer `15` on trial `3` / seed `4231027562`
- max blindness ratio:
  `34.75214130859887` on trial `2` / seed `4231027561`
- minimum behavioral-preserved null-access fraction:
  `0.005457635997862622` at layer `8` on trial `2` / seed `4231027561`
- largest loss delta:
  `1.1141399492938195` on trial `0` / seed `4231027559`
- largest perplexity delta:
  `13.738520053129076` on trial `0` / seed `4231027559`

## Failure Case Retained In Summary

- trial `2` / seed `4231027561`
  - failure modes: `argmax_not_certified`
  - stop reason:
    `online_eval_degraded (stage=pre_outcome, 4/10 correct, epoch=1)`
  - cooccurrence class: `cka_shift_and_inference_degraded`
  - `loss_delta = 0.5389641912752872`
  - `perplexity_delta = 8.520545504278582`
  - `min_cka = 0.9586101219510835` at layer `15`
  - `online_eval_delta_correct = 1`
  - `max_4gram_repeat_delta = -0.35624523990860635`
  - `null_access_min_behavioral_preserved_fraction = 0.005457635997862622`
    at layer `8`
  - `cka_blindness_ratio = 34.75214130859887` at layer `10`
  - `margin_mean_delta = -0.1875`

## Deleted Raw Artifacts

- `results/pipeline_validation_cert_350m_5t/350M/phase5_artifacts`
- `results/pipeline_validation_cert_350m_5t/run.log`

The deleted `phase5_artifacts` tree contained `5` raw adapter runs totaling
about `192.79 MB` of `.safetensors` files plus matching config and geometry
manifests. Those runs are now represented through the retained `trial_results`
payload in `350M/result.json`.
