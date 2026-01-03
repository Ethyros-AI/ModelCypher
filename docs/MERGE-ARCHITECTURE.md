# Merge Architecture

This document describes the consolidated merge subsystem and how the pipeline is wired.

## Entry Points

- CLI: `mc merge` -> `MergePipelineService` (`src/modelcypher/core/use_cases/merge/service.py`)
- API: `UnifiedGeometricMerger.merge()` -> `run_merge()` (`src/modelcypher/core/use_cases/merge/merger.py`)
- Full geometry analysis: `run_full_geometry_merge()` -> `GeometricMergeOrchestrator` (`src/modelcypher/core/use_cases/merge/orchestrator.py`)

## Pipeline Stages

Pipeline order (null-space transplant path):

1. Vocabulary alignment: `src/modelcypher/core/use_cases/merge/stages/vocabulary.py`
2. Probe (CKA + activations): `src/modelcypher/core/use_cases/merge/stages/probe.py`
3. Density (graft mask): `src/modelcypher/core/use_cases/merge/stages/density.py`
4. Permute (Git Re-Basin): `src/modelcypher/core/use_cases/merge/stages/permute.py`
5. Transplant (null-space constrained): `src/modelcypher/core/use_cases/merge/stages/transplant.py`
6. Validate: `src/modelcypher/core/use_cases/merge/stages/validate.py`

Full-geometry analysis delegates to the orchestrator for layer-by-layer analysis
and then executes null-space transplant (no blending).

## Data Models and Metrics

- Merge config/results: `src/modelcypher/core/use_cases/merge/models.py`
- Confidence metrics: `src/modelcypher/core/use_cases/merge/confidence.py`
- Validation service: `src/modelcypher/core/use_cases/merge/validation.py`

## Directory Layout

```
src/modelcypher/core/use_cases/merge/
├── __init__.py
├── merger.py              # UnifiedGeometricMerger + run_merge entry
├── pipeline.py            # run_merge implementation
├── service.py             # MergePipelineService (CLI orchestration)
├── orchestrator.py        # GeometricMergeOrchestrator (analysis)
├── models.py              # UnifiedMergeConfig, UnifiedMergeResult, geometry models
├── confidence.py          # geometric_confidence helpers
├── validation.py          # MergeValidationService
├── stages/
│   ├── vocabulary.py
│   ├── probe.py
│   ├── density.py
│   ├── permute.py
│   ├── transplant.py
│   ├── validate.py
│   └── vocab/              # vocabulary alignment submodules
└── analysis/               # orchestrator sub-stages
```
