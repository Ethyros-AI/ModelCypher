# Geometry-Only Hard Mode Replacement Matrix

**Date:** 2026-02-23  
**Status:** Pre-registered for module-by-module execution  
**Baseline:** targeted suite already green (`558` passing tests)

This document defines the exact, mechanical experiment plan to replace remaining
magic constants with measured geometric events.

Locked contract:
- Use activations/manifold geometry, not logits/probability heuristics.
- If a value cannot be derived from measured geometry, mark `INCONCLUSIVE` and abort that replacement.
- No fallback constants.

## 1) Global Contract (Applies To Every Replacement)

### 1.1 Measurement stack
For each run, emit raw artifacts:
- `config.json`
- `raw_metrics.jsonl`
- `bootstrap_metrics.json`
- `cross_basis_metrics.json`
- `cross_precision_metrics.json`
- `decision.json`

Store at:
- `results/geometry_only/<experiment_id>/<model_pair>/<domain>/<timestamp>/`

### 1.2 Registered axes (falsification, not confirmation)
Each experiment is run across:
- Multiple domains from the activation atlas (same domains used for alignment holdout checks)
- Multiple model pairs/families used in current merge validation
- Constant sensitivity sweeps (legacy constant varied; geometric replacement fixed)
- Cross-basis checks (orthogonal rotations + alignment)
- Cross-precision checks (`fp32`, `bf16` where supported; sanity only)

### 1.3 Promotion rule (global)
A replacement is promotable only if it beats legacy on raw metrics, with no guardrail regression:
- `holdout_cka` (non-degrading)
- `coherence_preservation` (improves or ties)
- `transfer_strength` (improves or ties)
- `false_alarm` and `miss_rate` for safety (both improve or tie, at least one strict)
- confidence interval width (`ci_width`) narrows or ties

If geometric derivation is unstable out-of-sample, mark unresolved and do not promote.

## 2) Atlas + Alignment Gate (Prerequisite)

Before any constant replacement is judged, run local chart alignment and holdout stability.

Files:
- `src/modelcypher/core/domain/geometry/alignment_validation.py`
- `src/modelcypher/core/domain/geometry/shared_manifold.py`
- `src/modelcypher/experimental/merge/stages/probe_alignment.py`

Per-domain resolved criterion:
- `alignment_gain = holdout_cka - raw_holdout_cka`
- `alignment_gain_ci_low > 0`
- numerical stability: `gram_condition_number * machine_epsilon < 1`

If criterion fails, domain/layer is `UNRESOLVED` and excluded from replacement promotion.

## 3) Per-File Experiment Matrix

| ID | File(s) | Legacy constant / heuristic | Geometric replacement event | Required raw metrics | Pass gate | Inconclusive / abort gate | Status |
|---|---|---|---|---|---|---|---|
| G1 | `src/modelcypher/core/domain/geometry/variance_concentration.py`, `src/modelcypher/experimental/merge/models.py` | Legacy bottleneck rule represented as fixed variance cutoffs (`var_top1 > constant`, historically `0.70`) | Spectrum changepoint event from per-layer singular spectrum (piecewise spectral slope break + bootstrap separation) | `var_top1`, full singular spectrum, changepoint index, changepoint strength CI, bottleneck selection stability, downstream `transfer_strength`, `coherence_preservation` | Top bottleneck changepoint is out-of-sample stable and improves/ties transfer+coherence vs legacy constant sweeps | No unique/stable changepoint across bootstrap or holdout collapse after replacement | **HOLD — code audit clean** |
| G2 | `src/modelcypher/core/domain/geometry/manifold_boundary.py` | `coherence_drop_fraction=0.5` | Boundary radius from coherence-vs-radius knee (`r*` from curvature extremum / derivative sign structure on measured curve) | `coherence(r)`, first/second derivative traces, `r_knee` CI, `boundary_max_relative_diff`, `preserved_fraction`, `transfer_strength` | Knee-derived radius improves/ties coherence preservation and transfer strength versus fixed-fraction baseline | No identifiable knee, multimodal unstable knees, or holdout instability | **HOLD — code audit clean** |
| G3 | `src/modelcypher/core/domain/training/loop_preservation.py` | `n_layers // 6` early/late skip and `n_layers // 3` highway/sample heuristics | Highway from stable ID minima under bootstrap; sampling layers from measured trajectory events (ID minimum, entropy re-expansion inflection, exit) | Per-layer ID distributions, highway index distribution, entropy trajectory derivatives, loop loss, downstream coherence | Stable highway event with narrower CI and non-degrading loop/coherence metrics vs legacy layer heuristics | No stable minimum/inflection (overlapping bootstrap order statistics) | **HOLD — code audit clean** |
| G4 | `src/modelcypher/core/use_cases/geometry_safety_service.py` | Percentile defaults (`0.25/0.50/0.75`, `0.95/0.90/0.05`) | Thresholds from calibrated manifold-distance separations between safe and attack distributions (decision boundary from measured density crossing) | Safe/attack distance distributions, boundary location CI, `false_alarm`, `miss_rate`, AUROC/AUPRC, calibration drift traces | Pareto-improves or ties safety error metrics with tighter/equal CI width across held-out prompts/domains | Safe/attack geometry not separable (no stable crossing boundary) | **HOLD — code audit clean** |
| G5 | `src/modelcypher/core/domain/safety/sidecar/sidecar_safety_policy.py`, `src/modelcypher/core/domain/safety/sidecar/sidecar_safety_session.py` | `hard_percentile=1.0`, `soft_percentile=5.0`, `consent_soft_multiplier=0.5` | Hard/soft boundaries from KL manifold-separation calibration; consent relaxation removed from geometric core and treated as explicit external policy input | KL safe/attack traces, intervention timing, `false_alarm`, `miss_rate`, intervention precision/recall, stability by scenario | Geometric thresholds dominate legacy percentile policy on held-out streams; no hidden multiplier defaults remain in geometry path | If consent behavior requires non-geometric tradeoff and no explicit policy is supplied, return `INCONCLUSIVE`/abort | **HOLD — policy, not geometry** |
| G6 | `src/modelcypher/experimental/thermo/measured_thermodynamics.py` | `attempted_percentile=50.0` fallback and fixed percentile windows | Outcome boundaries from measured entropy manifold separatrices (data-driven class boundary events, no fixed fallback percentile) | Entropy distributions by outcome, boundary CI, confusion matrix, calibration drift across domains/models | Separatrix-based thresholds improve/tie outcome classification metrics and CI width vs percentile fallback | No stable separatrix boundary or class overlap too high for geometric discrimination | **HOLD — code audit clean** |

## 4) Mechanical Runbook By Module

### 4.1 G1 Bottleneck changepoint
1. Export per-layer singular spectra from activation atlas.
2. Fit changepoint events per layer (with bootstrap resampling).
3. Compare against legacy constant sweeps.
4. Evaluate merge outputs on holdout domains.
5. Record promotion decision from registered metrics only.

### 4.2 G2 Boundary knee
1. Measure dense `coherence(r)` trajectories per layer/direction.
2. Compute knee event candidates and bootstrap confidence.
3. Replace fixed fraction path with knee path in shadow mode.
4. Compare merge coherence and transfer metrics.
5. Promote only if holdout improvements persist.

### 4.3 G3 Highway from stable ID minima
1. Build per-layer ID distributions via bootstrap.
2. Detect stable minima and entropy inflection events.
3. Replace `n//6`, `n//3` selectors with event-driven selectors.
4. Re-run loop-preservation metrics and downstream merge checks.
5. Promote only if selection remains stable cross-domain/model.

### 4.4 G4/G5 Safety thresholds
1. Build calibration sets with safe and attacked prompts.
2. Compute manifold-distance / KL separation distributions.
3. Derive decision boundaries from measured distribution crossings.
4. Evaluate on held-out prompts for false alarm/miss.
5. Remove geometric defaults that are policy choices; require explicit policy when needed.

### 4.5 G6 Thermo thresholds
1. Fit class-conditional entropy manifolds.
2. Solve for separatrix boundaries.
3. Compare against percentile fallback behavior.
4. Promote only on held-out classification and CI improvements.

## 5) Tests and Commands (Execution Surface)

Core targeted tests:
- `poetry run pytest tests/domain/geometry/test_alignment_validation.py`
- `poetry run pytest tests/test_geometry_evidence_suite.py`
- `poetry run pytest tests/domain/geometry/test_variance_concentration.py`
- `poetry run pytest tests/domain/geometry/test_manifold_boundary.py`
- `poetry run pytest tests/domain/training/test_loop_preservation.py`
- `poetry run pytest tests/domain/training/test_geometric_context_and_loop_preservation.py`
- `poetry run pytest tests/domain/safety/test_sidecar_safety_policy.py`
- `poetry run pytest tests/domain/safety/test_sidecar_safety_session.py`
- `poetry run pytest tests/cli/commands/test_safety_commands.py`

Safety calibration/eval surface:
- `poetry run mc analyze calibrate-safety --model <model> --prompts <safe_prompts> --output-file <calibration.json>`
- `poetry run mc analyze jailbreak-test --model <model> --prompts <attack_prompts> --calibration <calibration.json>`

## 6) Decision Schema (`decision.json`)

Required fields:
- `experiment_id`
- `module`
- `status` in `{PROMOTE, HOLD, INCONCLUSIVE, DISPROVEN}`
- `resolved_domains`
- `unresolved_domains`
- `metric_deltas`
- `ci_width_comparison`
- `failure_mode` (if not `PROMOTE`)
- `notes`

`PROMOTE` is allowed only when all registered gates pass in resolved domains and no guardrail metric regresses.

## 7) Promotion Log

### 2026-03-01: G-Series Code Audit (Measurement Pending)

**Audit method:** Code-level constant enumeration + citation tracing for every numeric literal in each module.

**Test results:** 180/180 tests pass across 11 registered test files.

**Status:** Code audit confirmed zero magic numbers across all modules. Full measurement pipeline comparison (§1.3 promotion rule) has not been run. All modules at HOLD pending formal measurement.

| ID | Status | Magic Numbers | Derivation Sources |
|---|---|---|---|
| G1 | **HOLD** (code audit clean) | 0 | Bai & Perron (1998) changepoint, Schwarz (1978) BIC |
| G2 | **HOLD** (code audit clean) | 0 | Discrete curvature extremum, IEEE 754 eps, bootstrap stability |
| G3 | **HOLD** (code audit clean) | 0 | Bootstrap-stable ID minimum, Demmel & Kahan (1990) SVD floor, Higham (2002) roundoff energy, IEEE 754 |
| G4 | **HOLD** (code audit clean) | 0 | Neyman-Pearson (1933) distribution crossing, bootstrap CI |
| G5 | **HOLD** (policy, not geometry) | N/A | Percentiles are deployment policy, not geometric constants. Module correctly requires explicit external policy input. |
| G6 | **HOLD** (code audit clean) | 0 | Boltzmann energy E=-T·log(p), CLT confidence 1-1/√n, Neyman-Pearson distribution crossing |

**Decision artifacts:** `results/geometry_only/G{1..6}_*/decision.json`

**Legacy constants replaced in code (pending measurement validation):**
- G1: `var_top1 > 0.70` → spectral changepoint
- G2: `coherence_drop_fraction=0.5` → knee detection via discrete curvature
- G3: `n_layers // 6`, `n_layers // 3` → bootstrap-stable trajectory events
- G4: percentiles `0.25/0.50/0.75`, `0.95/0.90/0.05` → distribution crossing
- G5: `hard_percentile=1.0`, `soft_percentile=5.0` → correctly classified as policy, not geometry
- G6: `attempted_percentile=50.0` → measured outcome thresholds from class-boundary crossing
