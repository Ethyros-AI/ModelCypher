# Merge Architecture

This document describes the consolidated merge subsystem and how the pipeline is wired.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options must come before the command path (example: `mc --output text merge run ...`).

## Entry Points

- CLI: `mc merge` -> `MergePipelineService` (`src/modelcypher/core/use_cases/merge/service.py`)
- API: `UnifiedGeometricMerger.merge()` -> `run_merge()` (`src/modelcypher/core/use_cases/merge/merger.py`)
- Full pipeline (pre-merge + merge + post-merge): `MergePipelineService.run()` (`src/modelcypher/core/use_cases/merge/service.py`)

## Pipeline Stages

Pipeline order (null-space transplant path):

1. Probe (CKA + activations): `src/modelcypher/core/use_cases/merge/stages/probe.py`
2. Density (graft mask): `src/modelcypher/core/use_cases/merge/stages/density.py`
3. Transplant (null-space constrained): `src/modelcypher/core/use_cases/merge/stages/transplant.py`
4. Validate (post-merge checks): `src/modelcypher/core/use_cases/merge/stages/validate.py`

Pre-merge analysis and post-merge validation are orchestrated in
`MergePipelineService` (not part of `run_merge()`).

Permutation alignment note:
- The older permutation stage (Git Re-Basin) is intentionally skipped in the current pipeline; alignment is handled by the probe stage’s Gram/CKA-derived transforms. The merge result still records `permute_metrics` for compatibility.

## Data Models and Metrics

- Merge config/results: `src/modelcypher/core/use_cases/merge/models.py`
- Geometric metrics: `src/modelcypher/core/use_cases/merge/metrics.py`
- Validation service: `src/modelcypher/core/use_cases/merge/validation.py`

## Directory Layout

```
src/modelcypher/core/use_cases/merge/
├── __init__.py
├── merger.py              # UnifiedGeometricMerger + run_merge entry
├── pipeline.py            # run_merge implementation
├── service.py             # MergePipelineService (CLI orchestration)
├── models.py              # UnifiedMergeConfig, UnifiedMergeResult, geometry models
├── metrics.py             # geometric metric aggregation
├── validation.py          # MergeValidationService
├── helpers.py             # loading/utilities
├── infrastructure.py      # adapter wiring helpers
├── stages/
│   ├── probe.py
│   ├── density.py
│   ├── transplant.py
│   ├── validate.py
│   ├── manifest.py
│   └── __init__.py
```

## References

- Null-space transplant: *AlphaEdit* ([PDF](references/arxiv/Fang_2025_AlphaEdit.pdf), [arXiv:2410.02355](https://arxiv.org/abs/2410.02355))
- Permutation alignment (historical context): *Git Re-Basin* ([PDF](references/arxiv/Ainsworth_2023_Git_ReBasin.pdf), [arXiv:2209.04836](https://arxiv.org/abs/2209.04836))
