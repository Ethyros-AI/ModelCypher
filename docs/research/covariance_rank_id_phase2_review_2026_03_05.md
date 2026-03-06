# Covariance Rank Phase 2 Review (2026-03-05)

Scope:
- `scripts/covariance_rank_id_analysis.py`
- Phase 2 additions: E2, E5, E6, E7, and cross-model summary changes

Method:
- Code review only
- No new model run was executed in this review

## Summary

The Phase 2 extension is not ready for promotion or scientific closure in its
current form. Three P1 issues invalidate the main new decision points:

1. `E5` local `k_eff` is rank-capped by neighborhood size.
2. `E2` uses the wrong curvature operator for spheres.
3. The cross-model summary can return a false positive verdict when zero models
   are measured.

## Findings

### 1. Local `k_eff` is not commensurable with the TwoNN observable

`compute_local_keff()` builds a covariance patch from only `k` Euclidean
neighbors and then computes effective rank on that patch. With centered data,
the patch rank is bounded by `k - 1`. In the current script, `k` is derived as
`ceil(log2(N))`, which means:

- smoke mode (`N=12`) caps local centered rank at `3`
- full prompt set (`N=60`) caps local centered rank at `5`

That cap is lower than many observed TwoNN layer IDs, so `E5` cannot test
whether local covariance rank explains the ID trajectory. The current operator
measures a small-patch sample ceiling, not the local tangent-space geometry that
TwoNN is built from.

There is also an operator mismatch:

- `compute_local_keff()` uses Euclidean `KDTree` neighborhoods
- `IntrinsicDimension.compute_two_nn()` uses geodesic distances with
  `max(k_connectivity, ceil(log(n)))`

Until the neighborhood operator is made commensurable, the `M1_local` verdict is
not interpretable.

### 2. `E2` mislabels sphere curvature

The script defines sphere curvature as `kappa = 1 / R`. For a sphere `S^d` of
radius `R`, intrinsic sectional curvature is `1 / R^2` and scalar curvature is
`d(d-1) / R^2`.

This matters because the experiment bins and adjudicates `M2` using the reported
`kappa` values. The current `E2` result therefore does not test the quantity
described in the comments or output.

### 3. Empty measurement sets can falsely confirm `M1`

If every requested model path is missing, `run_experiment()` skips all models and
passes an empty list into `compute_cross_model_summary([])`. The summary then
uses `all(...)` over an empty iterator, which returns `True`, yielding:

- `M1_global_all_pass = True`
- `M1_local_all_pass = True`
- `M1_verdict = "M1_LOCAL_confirmed: local k_eff tracks TwoNN ID"`

That is a false scientific conclusion from zero measurements.

## Required Fixes Before Re-Running

1. Redefine the `E5` operator so the local rank observable is commensurable with
   the TwoNN neighborhood geometry and not hard-capped by an undersized patch.
2. Correct `E2` to use the intended curvature quantity and re-bin any
   curvature-conditioned verdicts from that corrected operator.
3. Make zero-model and zero-layer cases return an explicit insufficiency verdict,
   never a confirmation verdict.

## Fixes Applied (2026-03-05)

All three P1 issues were fixed in the same session:

1. **Local k_eff rank cap**: Changed k derivation from `ceil(log2(N))` to
   `max(ceil(ln(N)), N//2)`, matching TwoNN's Berry & Sauer 2016 rule with a
   floor at N//2 so rank cap (k-1) exceeds expected TwoNN IDs. `compute_local_keff`
   now returns `(mean_keff, rank_cap)` tuple for transparency.

2. **E2 curvature formula**: Corrected from `kappa = 1/R` to `kappa = 1/R²`
   (sectional curvature of S^d(R)). Log labels updated to `κ=1/R²=...`.

3. **Empty summary verdict**: Added `n_models > 0` guard. Empty results now
   return `M1_no_data: no models were evaluated` with `False` for both passes.

**Remaining operator mismatch**: E5 uses Euclidean KDTree neighborhoods while
TwoNN uses geodesic distances. This is a known limitation — local covariance
rank is a different operator than TwoNN's local scaling estimator.

## Smoke Test Results (post-fix)

Ran on LFM2-350M + Qwen3.5-0.8B (12 probes):
- **E2**: M2 insufficient — bias uncorrelated with κ (Spearman=0.16, p=0.56)
- **E5**: Local k_eff also fails to track TwoNN ID (r≈-0.2 both models)
- **E3**: M3 killed (scale invariant) — unchanged
- **D2**: TwoNN tracks synthetic k_eff (r=0.976) — unchanged

## Current Status

The three P1 bugs are fixed. Results in
`results/covariance_rank_id/covariance_rank_id_phase2_results.json` reflect
smoke-mode post-fix numbers. Full-registry run (11 models, 60 probes, with E7)
not yet executed. The script is ready for a full run.
