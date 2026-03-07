# Wave-Kernel Falsifier Protocol

**Status:** Registered, exploratory only  
**Protocol ID:** `F-WAVE-01`  
**Runner:** `scripts/wave_field_analysis.py`  
**Validator:** `scripts/validate_wave_kernel_falsifier_artifacts.py`

## Claim Form

All claims in this protocol use the repository contract:

`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`

For this protocol:

- `geometry_state`: per-head distance-conditioned attention profile
- `architecture_state`: model family and attention implementation
- `scale_state`: parameter scale and layer/head index
- `precision_state`: checkpoint precision from model path metadata
- `measurement_operator`: holdout profile error after calibration-only fitting

## Narrow Claim

The narrow claim under test is:

> A damped oscillation kernel explains attention distance profiles better than monotone exponential decay once boundary-equivalent M2 fits are excluded and evaluation is done on holdout prompts.

This protocol does **not** test:

- that attention is "literally a wave equation"
- that wave kernels should replace attention
- that merge logic should move into wave space

## Observables

Per model, layer, head, and prompt:

- nonparametric distance-explained variance
- distance-conditioned mean profile
- per-distance sample counts

Per model, layer, and head after split aggregation:

- M0 fit: constant baseline
- M1 fit: monotone exponential decay
- M2 fit: damped oscillation
- calibration SSE and RMSE
- holdout SSE and RMSE using calibration-fitted parameters only
- AICc and BIC on calibration support
- `boundary_equivalent` for M2

Primary observable:

- `holdout_rmse_delta_m2_minus_m1 = RMSE_holdout(M2) - RMSE_holdout(M1)`

Interpretation rule:

- negative values favor M2
- positive values favor M1

## Split Rule

Probe source: `docs/research/wave_kernel_probe_manifest.json`

- Probes are grouped by `family`.
- Within each family, probes are sorted by a stable SHA-256-derived hash of `family:id:text`.
- Even indices go to calibration.
- Odd indices go to holdout.
- A single-probe family remains calibration-only and is non-adjudicating.

The existing 16 prompt set is retained as `smoke_only=true`. It is for harness verification only and is not promotable evidence.

## Boundary-Equivalent Rule

M2 counts as genuinely oscillatory only if its predicted curve over the observed distance support has:

- an interior extremum, or
- a zero-crossing

If neither occurs, the fit is marked `boundary_equivalent=true`.

Boundary-equivalent M2 fits do **not** count as wave support even if they achieve lower error, because they are operationally acting as monotone or flat boundary cases rather than as oscillations.

## Falsifiers

F1. If all families with non-boundary holdout data favor M1, the wave claim is falsified by decay.

F2. If families disagree in direction, the claim is architecture-conditioned and promotion is blocked.

F3. If M2 wins only through boundary-equivalent fits, the apparent support is rejected as an identifiability artifact.

F4. If promotable probes are unavailable for holdout within a family, that family is non-adjudicating.

## Artifact Schema

Each run writes:

- `results/wave_kernel_falsifier/<run_id>/run_manifest.json`
- `results/wave_kernel_falsifier/<run_id>/per_head_fit_table.jsonl`
- `results/wave_kernel_falsifier/<run_id>/model_family_summary.json`
- `results/wave_kernel_falsifier/<run_id>/falsifier_outcome.json`
- `results/wave_kernel_falsifier/<run_id>/artifact_validation.json`

`artifact_validation.json` is produced from the validator and is itself required for a complete run directory.

## Promotion Rule

No global doctrine or roadmap promotion is allowed unless:

- every adjudicating family agrees in direction, and
- that direction favors M2 on non-boundary holdout heads, and
- the artifact validator passes on the final run directory

If families disagree in direction, the outcome remains exploratory and architecture-conditioned.
