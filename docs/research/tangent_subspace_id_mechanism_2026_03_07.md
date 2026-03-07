# Tangent Subspace ID Mechanism Experiment (2026-03-07)

Scope:
- `scripts/tangent_subspace_id_mechanism.py`
- 3 models (LFM2-350M, Qwen3.5-0.8B, Llama-3.2-3B), 60 probes, 3 measurement channels

Method:
- Full experiment run on real models (not simulation)
- Pre-registered predictions (P1-P5) and falsification criteria (F1-F3)

## Context

All prior ID mechanism hypotheses were refuted:
- M1 global (covariance rank k_eff): r~0 on real models
- M1 local (kNN-patch covariance): r~-0.2, operator mismatch
- M2 (curvature bias on TwoNN): Spearman=0.16, p=0.56
- M3 (scale change): killed (TwoNN scale-invariant by construction)
- Cumulative curvature r=0.821 is spurious (= layer_index -> ID)

## Hypothesis H_T: Tangent Subspace Rotation

TwoNN ID changes at layer l when the Jacobian J_l = I + dF/dh rotates the data
manifold's tangent subspace. Specifically: local tangent misalignment between
consecutive layers predicts the magnitude of ID change.

Three measurement channels:
- **A (global)**: PCA tangent basis -> Grassmann distance + novel direction count
- **B (local)**: Geodesic k-NN + local PCA -> mean principal angle between layers
- **C (tracked neighbors)**: Same k-NN across layers -> participation ratio change

## Results

### Per-Model Predictions

| Prediction | LFM2-350M | Qwen3.5-0.8B | Llama-3.2-3B |
|-----------|-----------|--------------|--------------|
| P1: d_G -> \|dID\| | r=+0.20, p=0.45, FAIL | r=+0.64, p=0.0008, PASS | r=+0.25, p=0.20, FAIL |
| P2: novel -> dID+ | nan (0 novels) | nan (0 novels) | nan (0 novels) |
| P3: highway stability | PASS | FAIL | FAIL |
| P4: local angle -> \|dID\| | r=+0.62, p=0.011, PASS | r=+0.72, p=0.0001, PASS | r=+0.26, p=0.18, FAIL |
| P5: delta_rank -> dID | r=+0.04, p=0.88, FAIL | r=-0.19, p=0.38, FAIL | r=+0.17, p=0.37, FAIL |

### Excluding Stage 0->1 Embedding Transition

The stage 0 (embedding output) has ID=62-84, creating a massive outlier delta.
Excluding this pair tests whether the mechanism holds for processing layers only.

| Prediction | LFM2-350M | Qwen3.5-0.8B | Llama-3.2-3B |
|-----------|-----------|--------------|--------------|
| P1 (excl stg 0) | r=+0.03, p=0.91 | r=+0.59, p=0.003 | r=+0.17, p=0.41 |
| P4 (excl stg 0) | r=+0.54, p=0.037 | r=+0.69, p=0.0003 | r=+0.18, p=0.38 |

P1 on LFM2 is entirely an artifact of the stage 0 outlier. P4 on LFM2 and Qwen
is genuine (holds after removing the outlier). All Llama results are null.

### Cross-Model Falsification

- **F1 (all P1 > 0.3)**: FALSIFIED. Only Qwen passes.
- **F2 (sign match)**: PASSES. All models show positive P1 correlation.
- **F3 (any P5 > 0.3)**: FALSIFIED. All three fail (0/3).

Overall verdict: `F1_FALSIFIED`

## Key Findings

### 1. Novel direction count = 0 everywhere

Consecutive layers NEVER introduce truly orthogonal directions. All principal
angles have cos(theta) > sqrt(eps). The residual connection h_{l+1} = h_l + F(h_l)
preserves the tangent subspace. Whatever changes TwoNN ID, it is NOT the appearance
of novel subspace directions.

### 2. Rank change does not predict ID change (P5 universal failure)

The effective rank (participation ratio) of tracked-neighbor difference matrices
does NOT change in a way that predicts TwoNN ID. This cleanly eliminates the
"dimension gain/loss" version of the hypothesis. TwoNN ID changes are NOT caused
by the local neighborhood gaining or losing effective dimensions.

### 3. Local tangent misalignment (P4) is a candidate mechanism on 2/3 models

P4 passes on LFM2 (r=+0.54, p=0.037 excl. stage 0) and Qwen (r=+0.69, p=0.0003
excl. stage 0). These are different architecture families (hybrid attention-conv
vs hybrid full_attention + linear_attention).

P4 fails on Llama-3.2-3B (r=+0.18, p=0.38). Two possible explanations:
- **Measurement limitation**: N=60 in d=3072 gives underdetermined local tangent
  bases (~4 neighbors per point, trying to estimate 5-9 dimensional tangent spaces)
- **Mechanism underspecification**: Standard GQA transformer may have different
  per-layer geometry than hybrid architectures

Cannot distinguish these without higher-N runs on Llama.

### 4. Global PCA rotation (P1) is NOT the mechanism

P1 works only on Qwen (1/3). On LFM2, the correlation is entirely an artifact of
the stage 0->1 embedding transition (r drops from +0.20 to +0.03). Global PCA
subspace rotation does not predict |dID| for processing layers.

## Status Assessment

Per CLAUDE.md: "Confirmed on one model, refuted on another" is NOT a conclusion.
It means mechanism underspecification or measurement invalidity.

**H_T is NOT confirmed.** The strongest sub-signal (P4, local tangent misalignment)
passes on 2/3 models but the third failure cannot be attributed to measurement
limitations with certainty.

**Status remains: `[MECHANISM_UNKNOWN]`**

P4 is a candidate mechanism pending:
1. Higher-N validation (200+ probes) on d=3072 models to rule out measurement limitation
2. A second standard-transformer model to test whether Llama failure is architecture-specific
3. Formal derivation connecting local tangent rotation to TwoNN mu-ratio distribution

## What Was Eliminated

These are clean negative results (not measurement failures):
- Novel direction count = 0 everywhere -> subspace novelty is NOT the mechanism
- P5 universal failure -> local rank change is NOT the mechanism
- P1 (excl stage 0) on LFM2 -> global PCA rotation is NOT the mechanism on
  hybrid attention-conv architectures

## Artifacts

- Script: `scripts/tangent_subspace_id_mechanism.py`
- Results: `results/tangent_subspace_id_mechanism/results.json`
- Plan: `/Users/jasonkempf/.claude/plans/functional-prancing-piglet.md`
