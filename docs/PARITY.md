# TrainingCypher Parity Tracker

This document tracks CLI and MCP parity between TrainingCypher (Swift) and ModelCypher (Python).
Status values: DONE, PARTIAL, PLANNED.

## CLI Command Groups

- inventory: PARTIAL (outputs differ; missing system/version alignment details)
- explain: PARTIAL (stub output only)
- train: PARTIAL (start/preflight/status/pause/resume/cancel/logs/export implemented; real MLX training not yet)
- job: PARTIAL (list/show/attach/delete implemented; missing filters and loss stats parity)
- checkpoint: PARTIAL (list/delete/export implemented; export formats mostly stubbed)
- model: PARTIAL (list/register/delete/fetch/merge implemented; probe/validate-merge/analyze-alignment/search missing)
- dataset: PARTIAL (validate/preprocess/list/delete/pack-asif implemented; preview/get-row/update-row/add-row/delete-row/convert missing)
- doc: PARTIAL (convert/validate implemented; preflight missing)
- system: PARTIAL (status/probe implemented; readiness details need parity)
- eval: PARTIAL (list/show implemented; run/results missing)
- compare: PARTIAL (list/show implemented; run/checkpoints/baseline/score missing)
- validate: PARTIAL (train/dataset implemented)
- estimate: PARTIAL (train implemented)
- infer: PARTIAL (infer implemented; infer run/suite missing)
- geometry: PARTIAL (semantic prime/gate inventories + atlases ported; CLI subcommands missing)
- permutation aligner core: PARTIAL (anchor-projected align/apply + MLP-safe rebasin + fusion implemented; GPU batched argmax parity pending)
- semantic primes & gates core: PARTIAL (inventories + atlases + drift detector ported; probe tooling missing)
- concept response matrix: PARTIAL (CRM + CKA + comparison ported; HiddenStateExtractor integration missing)
- cross-architecture layer matcher: PARTIAL (DP alignment + H2 validation ported; sparse fingerprint integration pending)
- gromov-wasserstein distance: PARTIAL (solver + pairwise distances ported; validation suite integration pending)
- dare sparsity analysis: PARTIAL (analysis + metrics ported; integration pending)
- affine stitching layer: PARTIAL (training + apply/inverse ported; integration pending)
- generalized procrustes analysis: PARTIAL (GPA + CRM alignment ported; integration pending)
- shared subspace projector: PARTIAL (CCA/shared-SVD/procrustes ported; integration pending)
- intrinsic dimension estimator: PARTIAL (TwoNN + bootstrap ported; integration pending)
- topological fingerprint: PARTIAL (persistence summary + comparison ported; integration pending)
- compositional probes: PARTIAL (probe analysis + consistency ported; integration pending)
- entropy delta + model state: PARTIAL (sample/session metrics + anomaly scoring ported; integration pending)
- conflict analysis: PARTIAL (conflict score aggregation ported; integration pending)
- manifold dimensionality: PARTIAL (entropy features + prior tension + ID estimate ported; integration pending)
- manifold profile: PARTIAL (profile/points/regions + stats ported; clustering/service integration pending)
- manifold clusterer: PARTIAL (DBSCAN + merge logic ported; profile service integration pending)
- intersection map analysis: PARTIAL (analysis + markdown report ported; integration pending)
- thermo path integration: PARTIAL (analysis + measurement assembly ported; inference/gate detection integration pending)
- adapter: PLANNED (project/wrap-mlx/smooth/inspect missing)
- calibration: PLANNED
- thermo: PLANNED
- rag: PLANNED
- stability: PLANNED
- agent-eval: PLANNED
- storage: PLANNED
- dashboard: PLANNED
- research: PLANNED
- help/ask/completions/schema: PLANNED
- afm, ensemble, model-acceptance, sparse-region: PLANNED

## MCP Tools

- tc_inventory: PARTIAL (missing fields and parity with CLI inventory)
- tc_train_start: PARTIAL (no full training engine, limited params)
- tc_job_status: PARTIAL (fields populated from stub job)
- tc_job_list: PARTIAL (filters partial)
- tc_job_cancel: PARTIAL
- tc_job_pause: PARTIAL
- tc_job_resume: PARTIAL
- tc_model_list: PARTIAL
- tc_infer: PARTIAL
- tc_system_status: PARTIAL (readiness details need parity)
- tc_validate_train: PARTIAL
- tc_estimate_train: PARTIAL
- tc_dataset_validate: PARTIAL
- tc_model_fetch: PARTIAL
- tc_checkpoint_export: PARTIAL

## MCP Resources

- tc://models: PARTIAL
- tc://jobs: PARTIAL
- tc://checkpoints: PARTIAL
- tc://datasets: PARTIAL
- tc://system: PARTIAL
