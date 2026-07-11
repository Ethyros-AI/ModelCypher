# Pipeline Validation Report

**Timestamp:** 2026-02-25T18:56:06.273402+00:00
**Git hash:** 833083b5
**Trials per model:** 1

## Verdict: FAIL

| Model | Structural | Inference | Composite | Structural pass/fail | Inference pass/fail | Composite pass/fail | online_eval_delta_correct (mean) | max_4gram_repeat_delta (max) | CKA worst layer | Null-access min preserved | cka_blindness_ratio (max) | margin_mean_delta (mean) |
|-------|------------|-----------|-----------|----------------------|---------------------|---------------------|----------------------------------|----------------------------|----------------|----------------------------|--------------------------|--------------------------|
| 350M | PASS | FAIL | FAIL | 1/0 | 0/1 | 0/1 | +2.000 | -0.161491 | 0.956090 @ layer 15 | 0.006126 @ layer 8 | 9.6783 | +0.3375 |

## Counterexample Detail

### 350M

- **Seed 4231027559**: argmax_not_certified
  - loss_delta=1.1777, ppl_delta=14.1060
  - stop_reason=certificate (‖g‖=4.39e+00, Δmax=0.00e+00<CI=2.97e-01, epoch=9)
  - cooccurrence_class=cka_shift_and_inference_degraded
  - min_cka=0.956090 (layer=15)
  - adapter_saturation_median_ratio=0.7286
  - online_eval_delta_correct=+2
  - max_4gram_repeat_delta=-0.161491
  - null_access_min_behavioral_preserved_fraction=0.006126 (layer=8)
  - null_observability_max_condition_number=3470338351.328949 (layer=14)
  - inference_min_cka=0.000000 (layer=4)
  - cka_blindness_ratio=9.6783 (worst_layer=10)
  - cka_delta_gap=+0.956090
  - margin_mean_baseline=0.6500
  - margin_mean_adapted=0.9875
  - margin_mean_delta=+0.3375
  - margin_n_near_zero_baseline=2
  - margin_n_near_zero_adapted=0
  - margin_n_flipped_sign=2

