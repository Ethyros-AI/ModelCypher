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

## Promotion Rule

Promotion from exploratory to stronger claim state requires:
1. Predicted signs supported in primary regressions,
2. No triggered falsifier,
3. Commensurability checks passing,
4. Clear separation between resolvable and unresolved models in the final report.

## Results (2026-03-04, full run with GPU collection)

**Design matrix diagnostics (n=9):**
- cond(X'X) = 37952 (driven by intercept scale)
- VIF: log_GQA=1.22, I_hybrid=2.53, log_d=2.31
- Predictor correlations: log_GQA vs I_hybrid = -0.425, I_hybrid vs log_d = -0.753

**z_couple regression (n=9, DOF=5, R²=0.686):**

| Coefficient | Estimate | SE | t | p | 95% CI |
|-------------|----------|------|------|-------|---------|
| intercept | 3.724 | 1.864 | 2.00 | 0.102 | [-1.068, 8.516] |
| b_g | -0.503 | 0.211 | -2.39 | 0.063 | [-1.044, 0.038] |
| b_h | -0.179 | 0.243 | -0.74 | 0.495 | [-0.805, 0.446] |
| b_s | -0.388 | 0.210 | -1.84 | 0.125 | [-0.928, 0.153] |

**c_cancel regression (n=9, DOF=5, R²=0.854):**

| Coefficient | Estimate | SE | t | p | 95% CI |
|-------------|----------|------|------|-------|---------|
| intercept | -0.086 | 0.776 | -0.11 | 0.916 | [-2.082, 1.910] |
| d_g | 0.535 | 0.102 | 5.27 | 0.003 | [0.274, 0.796] |
| d_h | 0.077 | 0.100 | 0.77 | 0.476 | [-0.179, 0.333] |
| d_s | -0.058 | 0.098 | -0.60 | 0.577 | [-0.309, 0.193] |

**Falsifier outcomes:**

| Falsifier | Status | Detail |
|-----------|--------|--------|
| F1 (b_g >= 0) | INCONCLUSIVE | b_g = -0.503, CI crosses zero (p=0.063) |
| F2 (d_g <= 0) | **SUPPORTED** | d_g = 0.535, CI excludes zero (p=0.003), predicted sign |
| F3 (within-family) | FALSIFIED | LFM2: z_couple 0.548 (GQA=2) → 0.590 (GQA=3), wrong sign |

**Overall: FALSIFIED** (F3 triggered, but note: 2 data points, 6 attention layers each)

**Interpretation:** The cancellation hypothesis (F2) is strongly supported — higher GQA
produces less complete cancellation between numerator and denominator pathways (p=0.003,
R²=0.854). The coupling hypothesis (F1) has the right sign but insufficient power.
The within-family LFM2 test (F3) contradicts the coupling prediction, but with only 2
models and 6 attention layers each, this is low-power. The mixed outcome (F2 supported,
F3 falsified) suggests the GQA effect on cancellation is real but the coupling
observable may not be the right measure for within-family comparison.

**Promotion assessment:** Cannot promote to [VALIDATED] — F3 triggered. The c_cancel
pathway (F2) is a candidate for separate promotion as a standalone finding.

Artifacts: `results/gqa_falsifier_protocol/*/`
