# Geometry Validation Experiment Results

**Date:** 2026-02-05
**Models:** LFM2-350M-MLX-bf16, LFM2.5-1.2B-Instruct-bf16

## Summary

| Experiment | Result | Status |
|------------|--------|--------|
| V1 (Aggregate Metrics) | NULL | Aggregate layer-averaged metrics show no signal |
| V2 (Token-Level) | **ARTIFACTUAL** | Bug in generation code invalidated results |
| V3 (Rigorous) | **WEAK SIGNAL** | Direction change shows moderate effect, velocity not significant |

## Critical Finding: V2 Was an Artifact

The V2 experiment reported d=1.55 effect size for velocity at answer token. **This was caused by a bug.**

The bug: calling `base_model(current_ids)` instead of `model(current_ids)`:
- `base_model` returns hidden states (dim=2048)
- `model` returns logits (dim=65536)
- Argmax over hidden states produces garbage tokens, not actual model outputs

The "14 correct / 86 incorrect" split in V2 was not model reasoning - it was random noise from broken sampling. The reported d=1.55 effect size was comparing random garbage outputs.

## V3 Results (Rigorous Methodology)

V3 fixes: single forward pass, correct model call, strict numeric parsing, bootstrap CIs.

### GSM8K on LFM2.5-1.2B-Instruct (n=100)

59 correct, 41 incorrect (greedy decoding)

| Metric | Correct | Incorrect | Effect Size d | 95% CI |
|--------|---------|-----------|---------------|--------|
| Velocity at answer | 1.10 | 1.08 | 0.17 | [-0.24, 0.59] |
| Direction change | 0.49 | 0.48 | 0.11 | [-0.30, 0.56] |

**Conclusion: No significant signal on GSM8K reasoning.**

Interesting per-layer pattern:
- Early layers (0-6): Incorrect has HIGHER velocity (d=-0.5 to -0.8)
- Late layers (13-15): Correct has HIGHER velocity (d=+0.2 to +0.5)

This reversal suggests early vs late processing differs, but the aggregate signal washes out.

### Arithmetic on LFM2-350M (n=200, temp=0.5)

189 correct, 11 incorrect (temperature sampling needed to get errors)

| Metric | Correct | Incorrect | Effect Size d | 95% CI |
|--------|---------|-----------|---------------|--------|
| Velocity at answer | 1.27 | 1.16 | 0.87 | [-0.11, 1.86] |
| Direction change | 0.53 | 0.44 | **1.23** | [0.33, 2.19] |

**Direction change is statistically significant** (CI doesn't include zero).

Per-layer pattern (all positive, correct > incorrect):
- Layer 9: d = 1.21
- Layer 13: d = 1.36

Interpretation: On simple arithmetic with temperature-induced errors, correct answers show larger direction changes (the model "turns more sharply" in hidden state space).

Caveat: Only 11 incorrect samples. Wide confidence intervals.

## What the Data Actually Shows

1. **V2's d=1.55 was fake.** The "strong geometric mechanism" claim was based on buggy code.

2. **GSM8K shows no signal.** On real reasoning tasks with an instruct model, velocity/direction at answer token doesn't predict correctness.

3. **Simple arithmetic shows moderate signal in direction change.** On arithmetic with temperature-induced errors, direction change d=1.23 is significant. But this is a limited finding:
   - Small sample of incorrect (n=11)
   - Only on simple arithmetic, not reasoning
   - Requires temperature to induce errors

4. **Layer-specific patterns exist.** Early vs late layers show opposite patterns on GSM8K, suggesting different roles in processing.

## Lessons Learned

1. **Test the generation loop.** The V2 bug produced grammatically coherent garbage that looked plausible. Always verify the model outputs are sensible.

2. **Greedy decoding on capable models gives 100% correct.** Need temperature or harder tasks to get incorrect samples.

3. **Small effect sizes need large samples.** With only 11 incorrect samples, even d=1.23 has wide CIs.

4. **Aggregate metrics hide layer-specific patterns.** The early vs late layer reversal on GSM8K is interesting but invisible in the mean.

## Files

- V3 Experiment: `scripts/geometry_validation_v3.py`
- GSM8K Results: `/tmp/geom_v3_gsm8k/results.jsonl`
- Arithmetic Results: `/tmp/geom_v3_350m_temp/results.jsonl`

## Next Steps

1. Run arithmetic with higher temperature to get more incorrect samples
2. Investigate the early/late layer reversal pattern on GSM8K
3. Try other metrics at the answer token (logit entropy, attention patterns)
4. Test whether direction change signal generalizes beyond simple arithmetic
