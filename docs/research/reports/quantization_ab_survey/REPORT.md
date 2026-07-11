# Quantization A/B Survey Report

Updated: 2026-03-08

## Question

What changes when Qwen3.5-0.8B moves from bf16 to 4-bit group-64 affine
quantization across the current ModelCypher CLI measurement surface?

## Retained Canonical Run

Retained run: `20260305T144412Z`

Why this run was retained:

- It has the same fully green tool-health profile as `20260305T061324Z`, the
  first fully green 13-tool / 23-run survey.
- It is the later rerun on the same tool surface.
- It includes `delta_summary.md`, which makes the family-level summary easier to
  retain than the earlier complete run.

Retained artifacts:

- `results/quantization_ab_survey/20260305T144412Z/comparison_report.md`
- `results/quantization_ab_survey/20260305T144412Z/delta_summary.md`
- `results/quantization_ab_survey/20260305T144412Z/survey_results.json`
- `results/quantization_ab_survey/20260305T144412Z/tool_health.md`
- `results/quantization_ab_survey/20260305T144412Z/probes.txt`

## Run Progression

| Run ID | Tool Surface | Status | Why it was superseded |
| --- | --- | --- | --- |
| `20260305T044814Z` | 12 tools / 22 runs | superseded | `entropy-trajectory`, `jacobian-trace`, and `benchmark` crashed; `chain-profile` was still unregistered |
| `20260305T051805Z` | 12 tools / 22 runs | superseded | `q4` `entropy-trajectory` still crashed; `chain-profile` still unregistered |
| `20260305T052732Z` | 12 tools / 22 runs | superseded | First clean 12-tool surface, but `chain-profile` still missing from the survey |
| `20260305T055219Z` | 13 tools / 23 runs | superseded | `chain-profile` was added but still crashed on both models |
| `20260305T061324Z` | 13 tools / 23 runs | superseded | First fully green survey; replaced by later rerun with the same tool-health profile plus `delta_summary.md` |
| `20260305T144412Z` | 13 tools / 23 runs | retained | Latest fully green rerun with the best retained summary artifacts |

## Retained Measurements

- `overall_accuracy`: bf16 `0.6500`, q4 `0.5000`, delta `-0.1500` (`-23.1%`)
- `tokensPerSecond`: bf16 `66.1775`, q4 `336.6494`, delta `+270.4718`
  (`+408.7%`)
- `meanIntrinsicDim`: bf16 `11.7679`, q4 `11.1107`, delta `-0.6572`
  (`-5.6%`)
- `slope`: bf16 `-7.5425e-04`, q4 `-7.3363e-04`, delta `+2.0624e-05`
- `avg_mean_curvature`: bf16 `0.6000`, q4 `0.6060`, delta `+0.0061`
- `cumulativeCurvatureToId`: bf16 `-0.7591`, q4 `-0.8400`, delta `-0.0809`

## Cleanup Performed

- Retained only `20260305T144412Z` as the canonical raw run.
- Deleted superseded runs `20260305T044814Z`, `20260305T051805Z`,
  `20260305T052732Z`, `20260305T055219Z`, and `20260305T061324Z`.
- Preserved the run-progression summary above so the tool-fix sequence is not
  lost when the raw superseded runs leave the worktree.
