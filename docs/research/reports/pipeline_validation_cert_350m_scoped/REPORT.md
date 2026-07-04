# Pipeline Validation Report

**Timestamp:** 2026-02-25T19:22:45.916413+00:00
**Git hash:** 833083b5
**Trials per model:** 1

## Verdict: FAIL

| Model | Structural | Inference | Composite | Structural pass/fail | Inference pass/fail | Composite pass/fail | online_eval_delta_correct (mean) | max_4gram_repeat_delta (max) | CKA worst layer | Null-access min preserved | cka_blindness_ratio (max) | margin_mean_delta (mean) |
|-------|------------|-----------|-----------|----------------------|---------------------|---------------------|----------------------------------|----------------------------|----------------|----------------------------|--------------------------|--------------------------|
| 350M | PASS | FAIL | FAIL | 1/0 | 0/1 | 0/1 | +3.000 | -0.161491 | 0.956409 @ layer 15 | 0.006080 @ layer 12 | 8.7746 | +0.0875 |

## Counterexample Detail

### 350M

- **Seed 4231027559**: argmax_not_certified
  - loss_delta=1.0510, ppl_delta=13.2582
  - stop_reason=certificate (‖g‖=2.18e+00, Δmax=2.63e-04<CI=2.68e-01, epoch=5)
  - cooccurrence_class=cka_shift_and_inference_degraded
  - min_cka=0.956409 (layer=15)
  - adapter_saturation_median_ratio=0.5892
  - online_eval_delta_correct=+3
  - max_4gram_repeat_delta=-0.161491
  - null_access_min_behavioral_preserved_fraction=0.006080 (layer=12)
  - null_observability_max_condition_number=3470338351.328949 (layer=14)
  - inference_min_cka=0.217291 (layer=7)
  - cka_blindness_ratio=8.7746 (worst_layer=2)
  - cka_delta_gap=+0.739118
  - margin_mean_baseline=0.6500
  - margin_mean_adapted=0.7375
  - margin_mean_delta=+0.0875
  - margin_n_near_zero_baseline=2
  - margin_n_near_zero_adapted=0
  - margin_n_flipped_sign=2

