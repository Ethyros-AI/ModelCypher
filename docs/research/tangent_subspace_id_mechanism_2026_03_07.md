# Tangent Subspace ID Mechanism Experiment (2026-03-07)

Scope:
- `scripts/tangent_subspace_id_mechanism.py`
- 3 models (LFM2-350M, Qwen3.5-0.8B, Llama-3.2-3B), 60 probes, 3 measurement channels

Method:
- Full experiment run on real models (not simulation)
- Pre-registered predictions (P1-P5) and falsification criteria (F1-F3)
- Note: P1-P5 thresholds (0.3 Spearman, median split) are exploratory cutoffs,
  not derived from geometry or machine precision

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
| P2: novel -> dID+ | r=+0.51, p=0.20, PASS | r=+0.48, p=0.08, PASS | r=+0.52, p=0.039, PASS |
| P3: highway stability | PASS | FAIL | FAIL |
| P4: local angle -> \|dID\| | r=+0.62, p=0.011, PASS | r=+0.72, p=0.0001, PASS | r=+0.26, p=0.18, FAIL |
| P5: delta_rank -> dID | r=+0.04, p=0.88, FAIL | r=-0.19, p=0.38, FAIL | r=+0.17, p=0.37, FAIL |

Note: P2 was nan in the initial run due to a bug that truncated PCA bases to k_min
before measuring novel directions. The corrected operator counts implicit novel
directions (k_l1 - k_l when k_l1 > k_l). P2 now passes on ALL 3 models.

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

### 1. Novel direction count predicts ID increase (P2: 3/3 — post-bugfix)

The initial run reported novel_count = 0 everywhere due to an operator bug: PCA
bases were truncated to k_min before computing principal angles, hiding implicit
novel directions. When k_l1 > k_l, there are k_l1 - k_l directions in V_l1 that
are in the null space of the projection onto V_l — these are automatically novel.

After fixing the operator, P2 passes on ALL THREE models:
- LFM2: r=+0.51 (p=0.20, n=8 increasing pairs)
- Qwen: r=+0.48 (p=0.08, n=14)
- Llama: r=+0.52 (p=0.039, n=16)

This is the only prediction that passes cross-family. The p-values on LFM2/Qwen
are marginal because of small n (P2 filters for increasing-ID pairs only), but
the effect size is consistent across architectures.

**CIRCULAR — NOT A REAL FINDING.** Verified: novel_from_angles = 0 for ALL pairs
across all models. The entire novel_count signal comes from novel_implicit =
max(0, round(ID_{l+1}) - round(ID_l)), which is ≈ round(delta_ID). P2 therefore
measures Spearman(round(delta_ID), delta_ID) — trivially positive by construction.
No actual PCA principal angle is near pi/2 between consecutive layers.

**Substantive conclusion:** Consecutive layers never introduce directions that are
orthogonal to the previous layer's PCA tangent subspace (all cos(theta) > sqrt(eps)).
The residual connection h_{l+1} = h_l + F(h_l) preserves the tangent subspace
orientation. This IS a real geometric finding, just not testable via P2.

### 2. Rank change uncorrelated with ID (P5: 0/3 — NOT a clean elimination)

The effective rank (participation ratio) of tracked-neighbor difference matrices
shows no correlation with TwoNN ID changes across all three models.

**Operator limitation:** Measurement C uses Euclidean KDTree neighborhoods, while
TwoNN uses geodesic distances via k-NN Floyd-Warshall graph. This is the same
commensurability issue flagged in the Phase 2 review
(`covariance_rank_id_phase2_review_2026_03_05.md`). A null P5 means Euclidean-
neighborhood rank change does not track geodesic-neighborhood ID. It does NOT
cleanly eliminate local rank change as a mechanism.

### 3. Local tangent misalignment (P4) is a candidate mechanism on 2/3 models

P4 passes on LFM2 (r=+0.54, p=0.037 excl. stage 0) and Qwen (r=+0.69, p=0.0003
excl. stage 0). These are different architecture families (hybrid attention-conv
vs hybrid full_attention + linear_attention).

P4 fails on Llama-3.2-3B (r=+0.18, p=0.38). The Measurement B operator at N=60
uses neighbor_count=7 (floor(sqrt(60))) and tangent_rank=3 (neighbor_count // 2).
Tangent rank 3 may be insufficient to represent the 5-9 dimensional manifold
geometry that TwoNN reports for Llama's layers. However, this is a hypothesis —
the saved artifact does not support a definitive diagnosis since the same operator
parameters (N=60, neighbor_count=7, tangent_rank=3) also apply to LFM2 and Qwen
where P4 passes.

Cannot distinguish measurement limitation from genuine mechanism underspecification
without higher-N runs or a second standard-transformer model.

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
1. Rerun with patched novel_count operator (full unmatched bases)
2. Higher-N validation on d=3072 models to test tangent_rank resolution
3. A second standard-transformer model to test whether Llama failure is architecture-specific
4. Formal derivation connecting local tangent rotation to TwoNN mu-ratio distribution

## What Was Eliminated

These results are supported by the measurement:
- P1 (excl stage 0) on LFM2 -> global PCA rotation is NOT the mechanism on
  hybrid attention-conv architectures

These results have operator limitations (not promotable as clean eliminations):
- P2 (novel_count): operator bug — bases truncated before measurement
- P5 (local rank change): Euclidean KDTree not commensurable with geodesic TwoNN

## Post-Review Patches (same day)

1. Measurement A: now computes `novel_count_full` on unmatched bases plus
   `novel_count_matched` on truncated bases, with `extra_dims` field
2. Measurement B: logs and saves `neighbor_count` and `tangent_rank` in artifact
3. Measurement C / P5: docstring and prediction output annotated with operator caveat

These patches require a rerun to produce corrected results.

## Artifacts

- Script: `scripts/tangent_subspace_id_mechanism.py`
- Results (pre-patch): `results/tangent_subspace_id_mechanism/results.json`
- Plan: `/Users/jasonkempf/.claude/plans/functional-prancing-piglet.md`
