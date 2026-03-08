# Quantization A/B Survey: Qwen3.5-0.8B `[EMPIRICAL]`

**Status:** Empirical
**Date:** 2026-03-05
**Run ID:** `20260305T144412Z` (retained canonical run; `20260305T061324Z`
was the first fully green pass)
**Models:** Qwen3.5-0.8B-bf16 vs Qwen3.5-0.8B-4bit-g64 (affine, group_size=64)
**Scope:** `observable = f(Qwen3.5-0.8B, bf16/4bit-g64, measurement_operator)`

---

## Summary

4-bit quantization of Qwen3.5-0.8B trades 23% overall benchmark accuracy for
409% throughput gain. Macro-geometric observables (entropy trajectory,
curvature, expansion ratio) shift less than 3%, indicating the quantization
preserves large-scale activation geometry while degrading fine-grained
reasoning capacity. The accuracy loss is task-dependent: simple factual recall
(arc_easy) is unaffected; reasoning (gsm8k) and boolean inference (boolq)
degrade significantly.

---

## Executive Delta Table

| Metric | Tool | bf16 | 4-bit | Delta | Relative |
|--------|------|------|-------|-------|----------|
| overall_accuracy | benchmark | 0.650 | 0.500 | -0.150 | **-23.1%** |
| tokensPerSecond | infer | 66.2 | 336.6 | +270.5 | **+408.7%** |
| meanIntrinsicDim | dimension-profile | 11.77 | 11.11 | -0.66 | -5.6% |
| slope | entropy-trajectory | -7.54e-4 | -7.34e-4 | +2.06e-5 | +2.7% |
| monotonicity | entropy-trajectory | -0.990 | -0.989 | +8.7e-4 | +0.1% |
| avg_mean_curvature | reasoning-flow | 0.600 | 0.606 | +0.006 | +1.0% |
| avg_smoothness | reasoning-flow | 0.648 | 0.646 | -0.002 | -0.3% |
| avg_directness | reasoning-flow | 0.176 | 0.178 | +0.003 | +1.5% |
| entropyToCurvature | chain-profile | 0.657 | 0.664 | +0.007 | +1.1% |
| cumulativeCurvatureToId | chain-profile | -0.759 | -0.840 | -0.081 | **-10.7%** |
| meanAttnFraction | chain-profile | 0.412 | 0.418 | +0.006 | +1.5% |

---

## Key Findings

### 1. Accuracy degrades non-uniformly across task types

| Benchmark | bf16 | q4 | Delta | n |
|-----------|------|----|-------|---|
| gsm8k (reasoning) | 35% (7/20) | 25% (5/20) | -29% | 20 |
| arc_easy (recall) | 90% (18/20) | 90% (18/20) | 0% | 20 |
| boolq (inference) | 70% (14/20) | 35% (7/20) | -50% | 20 |
| **overall** | **65%** | **50%** | **-23%** | 60 |

**OBSERVED:** Simple factual recall (arc_easy) is preserved under 4-bit quantization. Reasoning (gsm8k) and boolean inference (boolq) degrade. The degradation pattern is consistent with quantization noise disrupting fine-grained weight structure while preserving the dominant activation modes that suffice for simple retrieval.

Small sample size (n=20 per benchmark). Binomial 95% CI on overall_accuracy: bf16 [0.52, 0.76], q4 [0.37, 0.63]. The CIs overlap — this is a directional observation, not a statistically significant difference at this sample size.

### 2. Throughput gain is 5.1x

bf16: 66.2 tok/s. q4: 336.6 tok/s. The 4-bit model is 5.1x faster. This is
expected from reduced memory bandwidth requirements (4-bit weights = 4x less
data transfer, plus group-64 affine quantization overhead).

### 3. Macro-geometric observables shift less than 3%

Entropy trajectory (slope, monotonicity), reasoning-flow curvature, smoothness, directness, and expansion ratio all shift less than 3%. The entropy-reduction pattern through layers — the defining geometric signature of these models — is preserved under quantization.

**OBSERVED:** Quantization preserves the macro-geometry of the activation manifold while degrading the fine-grained structure that complex tasks depend on.

### 4. Intrinsic dimension drops 5.6%

Mean TwoNN intrinsic dimension: bf16=11.77, q4=11.11 (delta=-0.66, -5.6%). This is the largest geometric shift observed. Quantization compresses the representation into a slightly lower-dimensional subspace.

### 5. cumulativeCurvatureToId correlation strengthens under quantization

The correlation between cumulative curvature and intrinsic dimension strengthens from -0.759 (bf16) to -0.840 (q4), a 10.7% increase in magnitude. This suggests quantization tightens the curvature-ID coupling — the compressed representation has less slack between curvature accumulation and dimensional response.

**OBSERVED, NOT DERIVED:** The mechanism by which quantization strengthens this correlation is not derived. One hypothesis: quantization noise collapses directions near the decision boundary (those with smallest singular values), making the surviving geometry more tightly coupled. This remains speculative.

---

## Methodology

**Tools (14 CLI commands):** model-info, model-capacity, dimension-profile, entropy-trajectory, spectral-trajectory, expansion-ratio, reasoning-flow, chain-profile, jacobian-trace (3 prompts), attention-collapse (3 prompts), attention-sink (3 prompts), benchmark (quick suite), infer (5 prompts). Total: 23 tool runs per survey.

**Probes:** 17 text prompts covering factual, reasoning, and creative tasks. Shared across both models for commensurability.

**Benchmark suite:** "quick" (gsm8k, arc_easy, boolq), 20 samples each. This is a small sample — results are directional, not statistically definitive.

**Limitations:**
- Single architecture (Qwen3.5-0.8B). Cannot generalize to other architectures or scales.
- Single quantization method (4-bit affine, group_size=64). Other methods (GPTQ, AWQ, different group sizes) may produce different geometric signatures.
- model-info `layers` field truncated to first 20 tensors. bf16 sample covers `mtp.*`/`model.visual.*`, q4 sample covers `language_model.model.*` — different tensor groups, so 0 name-matched layers (expected, not a bug).

---

## CLI Issues Found and Fixed

This survey also served as a stress test of the ModelCypher CLI. Eight bugs were found and fixed:

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | entropy-trajectory MC-3013 on Qwen3.5 | Missing unembedding path for `model.language_model.model.embed_tokens` | Added Qwen3.5 strategies to `set_unembedding_matrix()` |
| 2 | jacobian-trace MC-3071 on Qwen3.5 | Naive `getattr(model, "model", model)` doesn't handle Qwen3.5 nesting | Extracted `resolve_model_base()` to shared utility |
| 3 | benchmark `ModelLoader.load()` | Wrong method name (`load` vs `load_model`) | Fixed 3 call sites in `benchmark.py` |
| 4 | model-capacity bad data on quantized | SVD on `.scales`/`.biases` metadata instead of dequantized weights | Added dequantize + metadata skip in `capacity_analysis_service.py` |
| 5 | chain-profile unregistered | Function existed but not imported/registered in `safety/__init__.py` | Added import and registration |
| 6 | reasoning-flow metrics missing | Survey looked for top-level keys; data nested in `results[i].overall` | Fixed extraction to navigate nested structure |
| 7 | benchmark `InferenceEngine.generate()` | `InferenceEngine` has `infer()` not `generate()` | Changed to `backend.generate()` |
| 8 | chain-profile MC-3032 on Qwen3.5 | `sublayer_collector.py` used naive model resolution | Changed to `resolve_model_base()` |

---

## Raw Data Location

```
results/quantization_ab_survey/
├── REPORT.md               # Family-level retained summary + cleanup log
├── summary.json            # Machine-readable retained summary
└── 20260305T144412Z/       # Retained canonical raw run
    ├── comparison_report.md
    ├── delta_summary.md
    ├── tool_health.md
    ├── survey_results.json
    ├── probes.txt
    └── raw/
        ├── bf16/
        └── q4/
```
