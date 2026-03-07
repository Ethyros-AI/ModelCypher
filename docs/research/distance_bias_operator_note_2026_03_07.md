# Distance-Bias Operator Note (2026-03-07)

## Scope

This note explains **why** the wave-kernel falsifier failed and what survives as
useful signal for the analysis pipeline.

Artifact set:

- protocol runner:
  `scripts/wave_field_analysis.py`
- validated run:
  `results/wave_kernel_falsifier/full_promotable_run_20260307_distance_bias/`

This is still an analysis pass only.
No merge logic or architecture logic changed.

## The operator that actually matters

Let:

- `A(i, j)` be one causal attention entry for a fixed head
- `D = i - j` be causal distance

The protocol's nonparametric profile predictor is the conditional mean:

`g(d) = E[A | D = d]`

The recorded `distance_r2` is not a heuristic.
It is the variance decomposition induced by the projector
`A -> E[A | D]`.

For the group-mean predictor:

`distance_r2 = Var(E[A | D]) / Var(A)`

and therefore:

`content_residual_fraction = 1 - distance_r2 = E[Var(A | D)] / Var(A)`

This gives the exact split:

1. `distance_explained_variance_fraction`
2. `content_residual_variance_fraction`

So the first question is not "is the head oscillatory?"
It is:

> how much of the head is explained by distance at all?

## What the validated run says

Overall outcome remained:

- `overall = falsified_by_decay`

from:

- `results/wave_kernel_falsifier/full_promotable_run_20260307_distance_bias/falsifier_outcome.json`

The useful surviving signal is the decomposition above.

Model means from
`results/wave_kernel_falsifier/full_promotable_run_20260307_distance_bias/model_family_summary.json`:

- `Qwen3.5-0.8B-bf16`
  - `mean_prompt_distance_r2 = 0.3887`
  - `mean_content_residual_variance_fraction = 0.6113`
  - `mean_calibration_holdout_weighted_correlation = 0.9816`
  - `mean_calibration_positive_slope_mass = 0.00170`
- `Llama-3.2-3B-Instruct-bf16`
  - `mean_prompt_distance_r2 = 0.1028`
  - `mean_content_residual_variance_fraction = 0.8972`
  - `mean_calibration_holdout_weighted_correlation = 0.8963`
  - `mean_calibration_positive_slope_mass = 0.00756`
- `LFM2.5-1.2B-Base-bf16`
  - `mean_prompt_distance_r2 = 0.1035`
  - `mean_content_residual_variance_fraction = 0.8965`
  - `mean_calibration_holdout_weighted_correlation = 0.8840`
  - `mean_calibration_positive_slope_mass = 0.00675`

Global head-level means from the same run:

- `mean_content_residual_variance_fraction = 0.8820`
- `median_content_residual_variance_fraction = 0.8985`
- `wave_support_count = 5 / 912`

## Why M2 lost

The failure decomposes into three facts.

### 1. Distance is usually a minority component

For Llama and LFM2, about `89.7%` of head variance is residual after conditioning
on distance.

That means a distance-only kernel is trying to model a small bias term, not the
dominant operator.

The wave question was asked at the wrong level of the decomposition.

### 2. The distance bias is stable across prompt split

Calibration vs holdout profile correlation is high:

- `Qwen`: `0.9816`
- `Llama`: `0.8963`
- `LFM2`: `0.8840`

So this is not just noise.
There is a repeatable distance-conditioned component.

What is repeatable is the **bias profile**, not the full head matrix.

### 3. The stable bias is mostly monotone

The new `calibration_positive_slope_mass` measures the count-weighted average
magnitude of upward steps in the calibration distance profile.

Family means are small:

- `Qwen`: `0.00170`
- `Llama`: `0.00756`
- `LFM2`: `0.00675`

The only surviving wave-support heads (`5` total) had much larger upward-step
mass:

- `mean_calibration_positive_slope_mass = 0.02927`
- `mean_calibration_holdout_weighted_correlation = 0.9966`

So the rare heads that favor M2 are the ones with both:

1. stable profiles
2. materially non-monotone interior structure

Most heads fail criterion 2.
Their stable distance component is closer to a monotone decay than to a damped
oscillation.

## What improves the pipeline

Three measurements are worth keeping.

1. `distance_explained_variance_fraction`
   This is the exact variance share captured by the distance projector.

2. `content_residual_variance_fraction`
   This tells us when kernel fitting is secondary because most of the operator is
   content-conditioned residual.

3. `calibration_holdout_weighted_correlation`
   This separates a stable bias from a prompt-unstable bias.

4. `calibration_positive_slope_mass`
   This is raw evidence for non-monotone profile structure before fitting M2.

These belong in the pipeline because they expose the operator split directly.

## What this implies for next measurement

The current protocol fits **post-softmax** attention weights:

`A(i, j) = softmax_j(S(i, j))`

If the score operator decomposes as:

`S(i, j) = b(i - j) + r(i, j)`

then row-wise softmax mixes the distance bias `b` with the content term `r`.
That can preserve a stable bias while shrinking its variance share in `A`.

So the next narrow question is:

> is the distance component cleaner in the pre-softmax score matrix than in the
> normalized attention weights?

That suggests one concrete pipeline addition:

- add backend-level collection of **pre-softmax attention score matrices**
  alongside post-softmax weights

If the score-level decomposition is cleaner, that is the right measurement
operator for future kernel work.

## Direction

The wave hypothesis did useful work as a falsifier.
What it left behind is better than a narrative:

- distance bias vs residual decomposition
- stability of the bias across prompt split
- direct non-monotone mass measurement

That is a good direction for ModelCypher because it remains:

- first-principles
- architecture-conditioned
- measurement-first
- free of merge or doctrine promotion beyond what the artifact supports
