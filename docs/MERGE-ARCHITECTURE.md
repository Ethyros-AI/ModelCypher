# Merge Architecture

This document describes the consolidated merge subsystem and how the pipeline is wired.

## Entry Points

- CLI: `mc merge` -> `MergePipelineService` (`src/modelcypher/core/use_cases/merge/service.py`)
- API: `UnifiedGeometricMerger.merge()` -> `run_merge()` (`src/modelcypher/core/use_cases/merge/merger.py`)
- Full pipeline (pre-merge + merge + post-merge): `MergePipelineService.run()` (`src/modelcypher/core/use_cases/merge/service.py`)

## Pipeline Stages

Pipeline order (null-space transplant path):

1. Probe (CKA + activations): `src/modelcypher/core/use_cases/merge/stages/probe.py`
2. Density (graft mask): `src/modelcypher/core/use_cases/merge/stages/density.py`
3. Permute (Git Re-Basin): `src/modelcypher/core/use_cases/merge/stages/permute.py`
4. Transplant (null-space constrained): `src/modelcypher/core/use_cases/merge/stages/transplant.py`

Pre-merge analysis and post-merge validation are orchestrated in
`MergePipelineService` (not part of `run_merge()`).

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
│   ├── permute.py
│   ├── transplant.py
│   ├── validate.py
│   ├── manifest.py
│   └── __init__.py
```
