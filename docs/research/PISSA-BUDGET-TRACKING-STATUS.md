# PiSSA Spectral Budget Tracking — Status & Next Steps

**Updated:** 2026-03-12
**Roadmap link:** R1 (baseline suite against standard practice)
**Status:** Code complete, tested. First A/B comparison shows default > MASS. Root cause identified. Amortization fix implemented, not yet validated on training run.

---

## Problem Statement

R1 training with PiSSA-initialized format-aligned data fails: online eval collapses at epoch 1. The MASS controller had no stateful budget signal for PiSSA — it relied only on instantaneous per-step bounds (`eta_sps`, `eta_weyl`). Without cumulative tracking, accumulated displacement (15%+ of spectral scale) triggered no deceleration.

## What Was Built (This Session)

### 1. PiSSA Displacement-from-Initialization Tracking

**Files:**
- `src/modelcypher/core/domain/training/spectral_budget.py` — `_pissa_delta_spectral_norm_power_iter()` + `compute_pissa_budget_ratios()`
- `src/modelcypher/backends/_mlx_training_adapter_core_mixin.py` — stores frozen init factors + per-layer sigma_k at PiSSA injection
- `tests/domain/training/test_spectral_budget.py` — `TestComputePissaBudgetRatios` (5 tests)

**Mechanism:** Power iteration on the implicit displacement operator `D = a_curr @ b_curr - a_init @ b_init` without forming the full `[in, out]` matrix. 4 matmuls per direction per iteration through rank-r intermediates. Returns `||displacement||_spectral / sigma_k` per layer.

### 2. Sub-Epoch Re-Anchor (replaces per-step Frobenius decrement)

**File:** `src/modelcypher/backends/_mlx_training_adapter_train_mixin.py`

**Problem solved:** Per-step Frobenius decrement (`_advance_remaining_budget`) overestimates cumulative spectral displacement by 3-4x (triangle inequality). V6 showed: one step consumed 100% budget, but true spectral measurement showed 29.3%.

**Fix:** `_reanchor_pissa_budget()` runs exact power iteration every 10 steps. Between re-anchors, `remaining_budget` holds the last measured value. Cost: < 1ms per re-anchor (rank-r intermediates).

### 3. Sqrt-Amortized Conformal Margin

**Files:**
- `src/modelcypher/core/domain/training/mass_step_size.py` — `compute_conformal_margin_rate(remaining, d_norm, amortization_steps)`
- `tests/domain/training/test_mass_step_size.py` — `TestAmortizedConformalMargin` (4 tests)

**Problem solved:** The original formula `eta_margin = remaining_budget / d_norm` equals `eta_weyl = sigma_k / d_norm` when the budget is full. Since Weyl's per-step bound allows `sigma_k` of spectral displacement per step, and the PiSSA budget IS `sigma_k`, one MASS step consumed the entire budget by construction.

**Fix:** `eta_margin = remaining_budget / (d_norm * sqrt(steps_remaining))`. The sqrt model matches empirical random-walk growth of cumulative spectral displacement (confirmed by V6 data: 964 steps consumed only 29.3%).

**Math:** For N steps at rate `eta_margin`, cumulative displacement under random walk ~ `sqrt(N) * eta_margin * d_norm`. Setting this equal to `remaining_budget` gives `eta_margin = remaining_budget / (d_norm * sqrt(N))`.

At step 0 with 964 remaining: `eta_margin = 0.40 / (11.45 * 31.05) = 0.0011`. Per-step displacement = 0.013. Cumulative over 964 steps: `sqrt(964) * 0.013 = 0.40 = sigma_k`. Budget consumed over the full epoch instead of one step.

### 4. Identity Fix

The matched-trace MASS branch was labeling itself `adamw_cosine` in runtime identity. Fixed in `identity.py`, train mixin, and `dataset_training_service.py`. Regression tests added.

---

## R1 Comparison Results (2026-03-12)

**Model:** LFM2-350M-MLX-bf16
**Dataset:** `data/training/r1_quick_aligned_train.jsonl` (gsm8k, arc_easy, boolq)
**Baseline:** gsm8k=50%, arc_easy=90%, boolq=70%

| Run | Optimizer | Post-training | Gate failure | Online eval |
|-----|-----------|--------------|-------------|-------------|
| A (default) | adamw_geometric / cosine | gsm8k=0%, arc=80%, boolq=90% (56.7%) | adapter_saturation_exceeded (1.30) | 11/20 |
| B (MASS) | adamw_matched_trace / MASS | gsm8k=0%, arc=50%, boolq=50% (33.3%) | adapter_saturation_exhausted (7.67) | 2/20 |

**Conclusion:** Default controller is materially safer than MASS matched-trace on this substrate. MASS is not promotable for `mc train run`.

### Diagnostic (96-step measurement pass)

With 10-step re-anchor interval: budget hit 0 at step 10 (first re-anchor). Root cause confirmed: `eta_weyl = sigma_k / ||D_eff||` allows full budget consumption in one step.

Reports: `/tmp/r1_default.json`, `/tmp/r1_mass.json`, `/tmp/r1_mass_diagnostic_96steps.json`

---

## Next Steps

**Canonical next action lives in `results/nblora_vs_standard/REPORT.md`
§ Exact Next Falsifier.** This file documents the PiSSA budget tracking
implementation; the spend decision and run commands are centralized in
REPORT.md to avoid competing pointers.

Summary of what's ready but unvalidated on a training run:

- sqrt-amortized conformal margin (`eta_margin = remaining / (d_norm * sqrt(steps_remaining))`)
- exact spectral re-anchor every 10 steps (replaces Frobenius decrement)
- frozen init factors stored at PiSSA injection for implicit power iteration

Success criterion when run: budget does NOT hit 0 before epoch boundary,
online eval does not collapse, post-benchmark comparable to or better than
the default path.

Diagnostic: log which of `eta_sps`, `eta_weyl`, `eta_margin`, `eta_ceiling` is binding at each step.

---

## Test Suite Status

- 7611 passed, 7 pre-existing failures (unrelated to budget tracking)
- Budget-specific tests: 170/170 passed (spectral_budget + mass_step_size + adapter_strict)
- No new failures introduced

---

## Files Changed (Cumulative)

| File | Change |
|------|--------|
| `src/modelcypher/core/domain/training/spectral_budget.py` | `_pissa_delta_spectral_norm_power_iter()` + `compute_pissa_budget_ratios()` |
| `src/modelcypher/core/domain/training/mass_step_size.py` | `amortization_steps` parameter on `compute_conformal_margin_rate` + `compute_per_step_rates` |
| `src/modelcypher/backends/_mlx_training_adapter_core_mixin.py` | Store init factors + per-layer sigma_k at PiSSA injection |
| `src/modelcypher/backends/_mlx_training_adapter_train_mixin.py` | `_reanchor_pissa_budget()`, 10-step re-anchor, sqrt amortization, identity fix |
| `src/modelcypher/core/domain/training/identity.py` | MASS identity reporting fix |
| `src/modelcypher/core/use_cases/dataset_training_service.py` | Identity propagation fix |
| `tests/domain/training/test_spectral_budget.py` | `TestComputePissaBudgetRatios` (5 tests) |
| `tests/domain/training/test_mass_step_size.py` | `TestAmortizedConformalMargin` (4 tests) |
| `tests/test_mlx_training_adapter_strict.py` | `_reanchor_pissa_budget` test |
| `tests/domain/training/test_identity.py` | Identity regression tests |
| `tests/test_dataset_training_service_strict.py` | Identity regression tests |
