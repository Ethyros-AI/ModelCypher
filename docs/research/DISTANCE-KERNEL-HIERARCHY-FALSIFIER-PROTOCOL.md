# Distance-Kernel Hierarchy Falsifier Protocol

**Status:** Registered, exploratory only
**Protocol ID:** `F-DKH-01`
**Runner:** `scripts/distance_kernel_hierarchy_analysis.py`
**Validator:** `scripts/validate_distance_kernel_hierarchy_artifacts.py`

## Claim Form

All claims in this protocol use the repository contract:

`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`

For this protocol:

- `geometry_state`: per-head distance-conditioned attention profile
- `architecture_state`: model family and attention implementation
- `scale_state`: parameter scale and layer/head index
- `precision_state`: checkpoint precision from model path metadata
- `measurement_operator`: AICc model selection on calibration data, holdout RMSE validation

## Narrow Claim

The narrow claim under test is:

> For attention heads where distance explains substantial variance in the attention profile (as determined by AICc model selection on calibration data), monotone exponential decay (M1) is the sufficient kernel, and the simpler constant model (M0) is inadequate. This holds cross-family.

This protocol does **not** test:

- that distance is the primary driver of attention (content residual may dominate)
- that M1 should replace attention
- that merge logic should move into distance-kernel space

## Background

The wave-kernel falsifier (F-WAVE-01) conclusively showed that damped oscillation (M2) does not beat monotone decay (M1) on attention distance profiles -- falsified across all 3 families (Qwen, Llama, LFM2). The closeout document (`docs/research/wave_kernel_closeout_2026_03_06.md`) identifies the dominant axis as **whether distance explains much of the head at all**, not oscillation.

M2 is dropped entirely (falsified). The study reduces to **M0 vs M1 classification** with holdout validation.

## Observables

Per model, layer, head, and prompt:

- nonparametric distance-explained variance (distance_r2)
- distance-conditioned mean profile
- per-distance sample counts

Per model, layer, and head after split aggregation:

- M0 fit: constant baseline (k=1)
- M1 fit: monotone exponential decay (k=2)
- calibration SSE, RMSE, AICc
- holdout SSE and RMSE using calibration-fitted parameters only
- `delta_aicc = AICc(M0) - AICc(M1)` on calibration support
- `head_classification`: `"m1_class"` if delta_aicc > 0, `"m0_class"` if delta_aicc <= 0
- `classification_clear`: `|delta_aicc| > delta_penalty(n)`
- `holdout_agrees`: holdout best-model matches AICc classification

Primary observable:

- Per-head AICc classification (M0 vs M1), validated against holdout RMSE

## AICc Model Selection

Model selection follows Burnham & Anderson (2002), "Model Selection and Multimodel Inference."

For model with k parameters fitted on n calibration points:

```
AIC  = n * log(MSE) + 2k
AICc = AIC + 2k(k+1) / (n - k - 1)
```

The AICc penalty difference between M1 (k=2) and M0 (k=1) at n calibration points is:

```
delta_penalty(n) = [2*2 + 2*2*3/(n-3)] - [2*1 + 2*1*2/(n-2)]
                 = 2 + 12/(n-3) - 4/(n-2)
```

At typical n=28-30 profile points, delta_penalty is approximately 2.3. This is the minimum improvement M1 must achieve in log-likelihood terms over M0 to justify its extra parameter.

No fixed R2 cutoff. No Hartigan's dip test. AICc is the correct tool because it directly answers "does the extra parameter pay for itself?"

## Split Rule

Probe source: `docs/research/wave_kernel_probe_manifest.json`

- Probes are grouped by `family`.
- Within each family, probes are sorted by a stable SHA-256-derived hash of `family:id:text`.
- Even indices go to calibration.
- Odd indices go to holdout.
- A single-probe family remains calibration-only and is non-adjudicating.

Reuses the same probe set and split rule as F-WAVE-01.

## Pre-Registered Predictions

### P-DKH-1: Hierarchy existence

Within each family, the fraction of M1-class heads (by AICc) is strictly between 0 and 1.

**Falsifier:** Any family has M1 fraction = 0 or M1 fraction = 1 -- hierarchy does not exist for that family.

### P-DKH-2: M1 holdout superiority

For M1-classified heads, M1 achieves lower holdout RMSE than M0 in >50% of cases.

**Falsifier:** M0 holdout RMSE <= M1 holdout RMSE in >= 50% of M1-classified heads in any family.

### P-DKH-3: Cross-family consistency

M1 fraction is architecture-conditioned (within-family variance < between-family variance).

**Falsifier:** Within-family variance >= between-family variance.

**Note:** NON-ADJUDICATING with single-model families. Requires adding second model per family.

### P-DKH-4: Content residual dominance

Mean distance_r2 < 0.5 across all heads in every family (content explains more than distance).

**Falsifier:** mean(distance_r2) > 0.5 for any family.

### P-DKH-5: AICc-holdout concordance

For heads with clear AICc preference (|delta_aicc| > analytic penalty), AICc classification agrees with holdout best-model in >90% of cases.

**Falsifier:** Concordance < 90% for any family.

## Artifact Schema

Each run writes:

- `results/distance_kernel_hierarchy/<run_id>/run_manifest.json`
- `results/distance_kernel_hierarchy/<run_id>/per_head_classification.jsonl`
- `results/distance_kernel_hierarchy/<run_id>/model_family_summary.json`
- `results/distance_kernel_hierarchy/<run_id>/falsifier_outcome.json`
- `results/distance_kernel_hierarchy/<run_id>/artifact_validation.json`

`artifact_validation.json` is produced from the validator and is itself required for a complete run directory.

## Promotion Rule

No global doctrine or roadmap promotion is allowed unless:

- every adjudicating family agrees that a hierarchy exists (P-DKH-1 PASS), and
- M1 holdout superiority holds cross-family (P-DKH-2 PASS), and
- content residual dominance is confirmed (P-DKH-4 PASS), and
- the artifact validator passes on the final run directory

If families disagree, the outcome remains exploratory and architecture-conditioned.
