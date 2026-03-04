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
