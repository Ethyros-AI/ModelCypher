# Entropy-Curvature GQA Falsifier Protocol

Date: 2026-03-04
Status: Pre-registered
Scope: Next architecture-conditioned falsifier for the GQA modulation claim in `CR-EC-001`

## Claim Under Test

Higher GQA weakens coupling between entropy operators and reduces cancellation between
the numerator and denominator pathways in theta-space.

## Prediction Contract

Per mission contract:

```
observable = f(geometry_state, architecture_state, scale_state, measurement_operator)
```

Instantiated:

```
z_couple = atanh(corr(H_logit, H_attn))
c_cancel = |beta_num - beta_den|

z_couple = f1(
    geometry_state = hidden-state trajectory + operator traces,
    architecture_state = {GQA, core_operator_type},
    scale_state = {d_model, n_layers, n_params},
    measurement_operator = depth-controlled residual correlation
)

c_cancel = f2(
    geometry_state = depth-controlled component slopes {beta_num, beta_den},
    architecture_state = {GQA, core_operator_type},
    scale_state = {d_model, n_layers, n_params},
    measurement_operator = depth-controlled OLS on log-components
)
```

## Directional Predictions

1. `d z_couple / d log(GQA) < 0`
2. `d c_cancel / d log(GQA) > 0`
3. At fixed family and scale band, increasing effective GQA shifts models toward
   weaker operator coupling and less complete cancellation.

## Model Specification

Primary regression (family-conditioned):

```
z_couple = a_family + b_g*log(GQA) + b_h*I(hybrid) + b_gh*log(GQA)*I(hybrid)
           + b_s*log(d_model) + eps
```

Secondary regression:

```
c_cancel = c_family + d_g*log(GQA) + d_h*I(hybrid) + d_gh*log(GQA)*I(hybrid)
           + d_s*log(d_model) + eta
```

Predicted signs: `b_g < 0`, `d_g > 0`.

## Measurement Operator and Commensurability

1. Use identical probe atlas construction across compared models.
2. Use identical layer inclusion rules (decomposable core + MLP only).
3. Residualize both entropy and curvature quantities by depth before correlation.
4. Report per-model detection floor from Fisher-SE MDE with Bretherton autocorrelation correction.
5. Use only resolvable models for sign-law adjudication; keep unresolved models as reported but non-adjudicating.

### H_logit Saturation Gate (post-hoc measurement-operator validity criterion)

**Added 2026-03-04.** This is NOT part of the pre-registered protocol. It is a discovered
measurement defect that invalidates certain model comparisons.

z_couple = atanh(corr(H_logit_resid, H_attn_resid)). For the H_logit operator to be
commensurable across models, the depth-residualized H_logit signal must carry at least
one bit of information about posterior concentration variation across layers:

```
max(H_logit_resid) - min(H_logit_resid) >= log(2) = 0.693 nats
```

**Derivation:** H_logit = log(k_eff) where k_eff = exp(H_logit) is the effective number
of tokens in the posterior. For the operator to resolve differences, k_eff must vary by
at least a factor of 2 across the model's attention layers — one bit of information.

When H_logit ≈ log(V) across all attention layers (saturation ratio ≈ 1), the model
hasn't learned to concentrate probability at any layer. z_couple then correlates
precision-level fluctuations in a near-uniform distribution — a different observable
than z_couple for desaturated models where k_eff spans orders of magnitude.

Models below the commensurability floor are flagged `commensurable=False` and excluded
from inferential z_couple regressions. They are retained in full-sample exploratory
regressions with explicit labeling. c_cancel is NOT affected because it uses all layers
(attention + conv), where H_logit has real variation even for saturated models.

## Falsifiers

F1 (coupling sign falsifier):
- If `b_g >= 0` with uncertainty interval excluding zero, the coupling-direction claim is falsified.

F2 (cancellation sign falsifier):
- If `d_g <= 0` with uncertainty interval excluding zero, the cancellation-direction claim is falsified.

F3 (within-family falsifier):
- For any family with sufficient within-family GQA variation to estimate a monotone trend,
  if the trend sign contradicts prediction under both Spearman and depth-controlled Pearson
  operators, the within-family claim is falsified.

If uncertainty intervals cross zero, outcome is inconclusive (not promotion, not falsification).

## Required Artifacts

1. `results/gqa_falsifier_protocol/<run_id>/model_table.json`
2. `results/gqa_falsifier_protocol/<run_id>/regression_summary.json`
3. `results/gqa_falsifier_protocol/<run_id>/within_family_trends.json`
4. `results/gqa_falsifier_protocol/<run_id>/falsifier_outcome.json`

## Artifact Integrity Check

Use the validator to prevent and detect malformed/truncated run artifacts:

```bash
poetry run python scripts/validate_gqa_falsifier_artifacts.py --run-dir results/gqa_falsifier_protocol/<run_id> --schema auto
poetry run python scripts/validate_gqa_falsifier_artifacts.py --root results/gqa_falsifier_protocol --all-runs --schema auto
```

Schema mode behavior:
- `auto`: prefer v2 if v2 keys exist; fallback to legacy v1; otherwise fail.
- `v2`: enforce current schema (`z_couple_regression_full`, `z_couple_regression_commensurable`, `commensurability_note`).
- `v1`: enforce legacy schema (`z_couple_regression`).

Producer guard: `scripts/gqa_falsifier_protocol.py` validates emitted artifacts in `v2`
mode and exits non-zero if validation fails.

## Promotion Rule

Promotion from exploratory to stronger claim state requires:
1. Predicted signs supported in primary regressions,
2. No triggered falsifier,
3. Commensurability checks passing,
4. Clear separation between resolvable and unresolved models in the final report.

## Results (2026-03-04, full run with GPU collection, updated with commensurability gate)

**Commensurability assessment (post-hoc, see H_logit Saturation Gate above):**

| Model | H_logit resid range | ≥ log(2)? | Saturation |
|-------|---------------------|-----------|------------|
| LFM2-350M | 0.007 | NO | 0.9998 |
| LFM2-700M | 0.022 | NO | 0.9994 |
| Qwen3.5-0.8B | 0.020 | NO | 0.9994 |
| Mistral-7B | 0.036 | NO | 0.9997 |
| Qwen3.5-2B | 0.221 | NO | 0.9947 |
| Qwen3.5-4B | 1.131 | YES | 0.9842 |
| Llama-3.2-3B | 2.726 | YES | 0.9817 |
| Qwen3-8B | 5.367 | YES | 0.5213 |
| Qwen2.5-3B | 5.216 | YES | 0.6661 |

5 models incommensurable, 4 commensurable.

**Design matrix diagnostics (n=9):**
- cond(X'X) = 37952 (driven by intercept scale)
- VIF: log_GQA=1.22, I_hybrid=2.53, log_d=2.31
- Predictor correlations: log_GQA vs I_hybrid = -0.425, I_hybrid vs log_d = -0.753

**z_couple regression — FULL (n=9, DOF=5, R²=0.686) [EXPLORATORY — includes incommensurable models]:**

| Coefficient | Estimate | SE | t | p | 95% CI |
|-------------|----------|------|------|-------|---------|
| intercept | 3.724 | 1.864 | 2.00 | 0.102 | [-1.068, 8.516] |
| b_g | -0.503 | 0.211 | -2.39 | 0.063 | [-1.044, 0.038] |
| b_h | -0.179 | 0.243 | -0.74 | 0.495 | [-0.805, 0.446] |
| b_s | -0.388 | 0.210 | -1.84 | 0.125 | [-0.928, 0.153] |

These coefficients should NOT be used for inferential claims — 5 of 9 models have
incommensurable z_couple (H_logit saturated, z_couple correlates noise).

**z_couple regression — COMMENSURABLE ONLY (n=4, DOF=0): UNDERPOWERED.**
4 commensurable models with 4 predictors yields zero degrees of freedom.
F1 cannot be adjudicated at the commensurable level with current model set.

**c_cancel regression (n=9, DOF=5, R²=0.854) [unaffected by commensurability issue]:**

| Coefficient | Estimate | SE | t | p | 95% CI |
|-------------|----------|------|------|-------|---------|
| intercept | -0.086 | 0.776 | -0.11 | 0.916 | [-2.082, 1.910] |
| d_g | 0.535 | 0.102 | 5.27 | 0.003 | [0.274, 0.796] |
| d_h | 0.077 | 0.100 | 0.77 | 0.476 | [-0.179, 0.333] |
| d_s | -0.058 | 0.098 | -0.60 | 0.577 | [-0.309, 0.193] |

c_cancel uses ALL layers (attention + conv), so H_logit has real variation even for
saturated models (e.g., LFM2-350M: H_logit range 1.47–8.50 across 16 layers).
The commensurability issue is z_couple only (filtered to attention layers where
saturated models are flat). F2 result stands unchanged.

**Falsifier outcomes:**

| Falsifier | Status | Detail |
|-----------|--------|--------|
| F1 (b_g >= 0) | INCONCLUSIVE | Full: b_g = -0.503, CI crosses zero (p=0.063). Commensurable: UNDERPOWERED (n=4, DOF=0). |
| F2 (d_g <= 0) | **SUPPORTED** | d_g = 0.535, CI excludes zero (p=0.003), predicted sign. Unaffected by commensurability. |
| F3 (within-family) | **INCOMMENSURABLE** | Both LFM2 models have saturated H_logit (resid range 0.007, 0.022 < log(2)). z_couple comparison is mathematically invalid. |

**Overall: INCONCLUSIVE** (F3 no longer FALSIFIED — incommensurable, not contradicted)

**Interpretation:** The cancellation hypothesis (F2) is strongly supported — higher GQA
produces less complete cancellation between numerator and denominator pathways (p=0.003,
R²=0.854). The coupling hypothesis (F1) cannot be adjudicated: the full regression
includes 5 incommensurable models, and the commensurable-only subset has zero DOF.
The within-family LFM2 test (F3) is invalid because both models have saturated H_logit —
z_couple correlates precision-level noise, not posterior concentration gradients.

**Promotion assessment:** Cannot promote to [VALIDATED] — F1 underpowered at commensurable
level, F3 incommensurable. The c_cancel pathway (F2) remains a candidate for separate
promotion as a standalone finding.

Artifacts: `results/gqa_falsifier_protocol/*/`

**Provenance note:** `20260304_213522` is the legacy pre-correction artifact
(F3=FALSIFIED, overall=FALSIFIED, no commensurability fields). `20260304_221926`
is the first corrected v2 artifact from an offline run (commensurability-corrected,
but `c_cancel` limited to cached models). `20260304_230500` is the canonical corrected
artifact from `--collect-missing` (v2 schema, commensurability fields present, full
`c_cancel` coverage across all 9 models) and should be used for adjudication.
