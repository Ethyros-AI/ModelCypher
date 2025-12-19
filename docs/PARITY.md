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
- geometry: PLANNED (all subcommands missing)
- permutation aligner core: PARTIAL (anchor-projected align/apply + MLP-safe rebasin + fusion implemented; GPU batched argmax parity pending)
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
