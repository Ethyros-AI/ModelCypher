# Wave-Kernel Closeout (2026-03-06)

## Scope

This note captures the first full execution of the registered
wave-kernel falsifier protocol:

- protocol: `docs/research/WAVE-KERNEL-FALSIFIER-PROTOCOL.md`
- runner: `scripts/wave_field_analysis.py`
- validator: `scripts/validate_wave_kernel_falsifier_artifacts.py`
- full promotable run:
  `results/wave_kernel_falsifier/full_promotable_run_20260306/`

The goal was narrow:

> test whether a damped oscillation kernel explains attention distance profiles
> better than monotone decay once boundary-equivalent M2 fits are excluded and
> evaluation is done on held-out probes

This was not an architecture-design pass.
No merge logic or doctrine promotion was changed.

## What was run

Default 3-family matrix:

- `Qwen3.5-0.8B-bf16`
- `Llama-3.2-3B-Instruct-bf16`
- `LFM2.5-1.2B-Base-bf16`

Committed promotable probe set:

- `docs/research/wave_kernel_probe_manifest.json`
- 24 promotable probes across 5 families

Command:

```bash
poetry run python scripts/wave_field_analysis.py \
  --output results/wave_kernel_falsifier/full_promotable_run_20260306
```

Validation:

```bash
poetry run python scripts/validate_wave_kernel_falsifier_artifacts.py \
  --run-dir results/wave_kernel_falsifier/full_promotable_run_20260306
```

Validator result:

- `PASS`

## What the measurement says

Protocol verdict from
`results/wave_kernel_falsifier/full_promotable_run_20260306/falsifier_outcome.json`:

- `overall = falsified_by_decay`
- `promotion_blocked = true`

Family-level result from
`results/wave_kernel_falsifier/full_promotable_run_20260306/model_family_summary.json`:

- `LFM2`: `direction = decay_favored`
  - `nonboundary_head_count = 123`
  - `mean_holdout_delta_m2_minus_m1 = +0.0606`
- `Llama`: `direction = decay_favored`
  - `nonboundary_head_count = 664`
  - `mean_holdout_delta_m2_minus_m1 = +0.0599`
- `Qwen`: `direction = decay_favored`
  - `nonboundary_head_count = 7`
  - `mean_holdout_delta_m2_minus_m1 = +0.0557`

Model-level details:

- `Qwen3.5-0.8B-bf16`
  - highest positional component of the three
  - `mean_prompt_distance_r2 = 0.3887`
  - but `boundary_equivalent_head_fraction = 0.8542`
  - and all non-boundary heads still favored M1 over M2
- `Llama-3.2-3B-Instruct-bf16`
  - low positional explainability overall
  - `mean_prompt_distance_r2 = 0.1028`
  - `holdout_best_model_counts = {m0: 362, m1: 303, m2: 7}`
- `LFM2.5-1.2B-Base-bf16`
  - similarly low positional explainability overall
  - `mean_prompt_distance_r2 = 0.1035`
  - `holdout_best_model_counts = {m0: 84, m1: 62, m2: 46}`
  - only `2` non-boundary heads qualified as wave-supporting

## Decision

The broad wave-kernel direction is **not** a good primary research path for
this repository.

Reason:

1. The full promotable protocol did not merely fail to support the claim.
   It returned a coherent cross-family negative result:
   all adjudicating families favored monotone decay over damped oscillation.
2. The strongest apparent positive signal in earlier exploratory runs came from
   boundary-equivalent M2 fits.
   The protocol removed that artifact, and the support largely vanished.
3. The dominant axis in these measurements is not oscillation.
   It is whether distance explains much of the head at all.

## Next direction

The useful continuation is **not** "wave attention."

The useful continuation is an architecture-conditioned
**distance-kernel hierarchy**:

1. Measure how much of each head is explained by distance at all.
2. For the distance-explained component, test the simplest sufficient model:
   - `M0`: constant / near-content-only
   - `M1`: monotone locality / decay
3. Treat the unexplained remainder as the content-dependent residual, not as an
   invitation to add oscillation by default.
4. Keep this thread in the analysis / measurement lane first.
   Do not move it into merge design or architecture design until a new
   architecture-conditioned claim survives falsification.

## What not to do next

Do not:

- build a wave-parameterized merge path
- build a wave-attention replacement for ModelCypher experiments
- promote "attention is waves" into roadmap or doctrine

Those moves would outrun the measurement.

## Recommended follow-up

Open the next narrow question:

> Is attention better described by a kernel hierarchy
> (`constant`, `monotone decay`, `content residual`)
> than by a wave hypothesis?

That next pass should remain evidence-bearing only and should reuse the current
artifact discipline rather than creating a new architecture branch.
