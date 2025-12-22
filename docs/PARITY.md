# ModelCypher Parity Tracker

This document tracks CLI and MCP parity between the internal research reference implementation and ModelCypher (Python).
Status values: DONE, PARTIAL, PLANNED.

Focus is high-dimensional geometry, core math, and CLI/MCP parity; RAG tooling is optional and de-prioritized.

## CLI Command Groups

- inventory: PARTIAL (outputs differ; missing system/version alignment details)
- explain: PARTIAL (stub output only)
- train: PARTIAL (start/preflight/status/pause/resume/cancel/logs/export implemented; real MLX training not yet)
- job: PARTIAL (list/show/attach/delete implemented; missing filters and loss stats parity)
- checkpoint: PARTIAL (list/delete/export implemented; export formats mostly stubbed)
- model: PARTIAL (list/register/delete/fetch/search/probe/validate-merge/analyze-alignment implemented; merge pipeline incomplete)
- dataset: PARTIAL (validate/preprocess/list/delete/pack-asif/quality/auto-fix implemented; preview/get-row/update-row/add-row/delete-row/convert missing)
- doc: PARTIAL (convert/validate implemented; preflight missing)
- system: PARTIAL (status/probe implemented; readiness details need parity)
- eval: PARTIAL (list/show implemented; run/results missing)
- compare: PARTIAL (list/show implemented; run/checkpoints/baseline/score missing)
- validate: PARTIAL (train/dataset implemented)
- estimate: PARTIAL (train implemented)
- infer: PARTIAL (mlx-lm inference + run/suite/batch wired; security scan still heuristic)
- geometry: PARTIAL (path detect/compare + validate + training/safety/adapter/primes/stitch/crm CLI wired; remaining probes pending)
- permutation aligner core: PARTIAL (anchor-projected align/apply + MLP-safe rebasin + fusion implemented; GPU batched argmax parity pending)
- anchor invariance analyzer: DONE (analyzer + stability scoring + layer alignment ported from TC)
- semantic primes & gates core: PARTIAL (inventories + atlases + probe tooling wired; hidden-state probing pending)
- concept response matrix: DONE (CRM build/compare + HiddenStateExtractor integration + CLI/MCP wiring)
- cross-architecture layer matcher: PARTIAL (DP alignment + H2 validation ported; sparse fingerprint integration pending)
- gromov-wasserstein distance: DONE (solver + pairwise distances + CLI/MCP tools wired)
- dare sparsity analysis: PARTIAL (analysis + metrics ported; CLI/MCP integration wired)
- affine stitching layer: PARTIAL (training + apply/inverse ported; integration pending)
- generalized procrustes analysis: PARTIAL (GPA + CRM alignment ported; integration pending)
- shared subspace projector: PARTIAL (CCA/shared-SVD/procrustes ported; integration pending)
- intrinsic dimension estimator: DONE (TwoNN + bootstrap + CLI/MCP tools wired)
- topological fingerprint: DONE (persistence summary + comparison + CLI/MCP tools wired)
- compositional probes: PARTIAL (probe analysis + consistency ported; integration pending)
- entropy delta + model state: PARTIAL (sample/session metrics + anomaly scoring ported; integration pending)
- conflict analysis: PARTIAL (conflict score aggregation ported; integration pending)
- manifold dimensionality: DONE (entropy features + prior tension + ID estimate + CLI/MCP wired)
- manifold profile: DONE (profile/points/regions + stats + local store + service + CLI/MCP wired)
- manifold clusterer: DONE (DBSCAN + merge logic + service + CLI/MCP wired)
- intersection map analysis: PARTIAL (analysis + markdown report ported; integration pending)
- thermo path integration: PARTIAL (analysis + measurement assembly ported; inference/gate detection integration pending)
- refusal direction: DONE (detector + cache + CLI/MCP wired)
- gate detector: PARTIAL (path detect/compare CLI + MCP wired; integration pending)
- geometry validation suite: PARTIAL (CLI + MCP validate wired; fixtures parity pending)
- transport-guided merger: DONE (OT synthesis + CLI/MCP wired)
- model fingerprints projection: PARTIAL (projection utilities ported; integration pending)
- persona vector monitor: DONE (vector extraction/monitoring + drift + CLI/MCP wired)
- adapter: DONE (blend/ensemble create/list/apply + CLI/MCP wired)
- calibration: PLANNED
- thermo: DONE (measure/ridge-detect/phase/sweep + CLI/MCP wired)
- rag: OPTIONAL (de-prioritized)
- stability: PLANNED
- agent-eval: PLANNED
- storage: PARTIAL (filesystem store + manifold profile store ported; remaining ports pending)
- dashboard: PLANNED
- research: DONE (taxonomy run/cluster/report + CLI wired)
- help/ask/completions/schema: PLANNED
- sparse-region: DONE (domains/locator/prober/validator + CLI/MCP wired)
- afm, ensemble, model-acceptance: PLANNED

## MCP Tools

- mc_inventory: PARTIAL (missing fields and parity with CLI inventory)
- mc_settings_snapshot: DONE (MCP tool wired)
- mc_train_start: PARTIAL (no full training engine, limited params)
- mc_job_status: PARTIAL (fields populated from stub job)
- mc_job_list: PARTIAL (filters partial)
- mc_job_cancel: PARTIAL
- mc_job_pause: PARTIAL
- mc_job_resume: PARTIAL
- mc_model_list: PARTIAL
- mc_model_probe: DONE
- mc_model_validate_merge: DONE
- mc_model_analyze_alignment: DONE
- mc_model_merge: PARTIAL (unified mode not implemented)
- mc_infer: PARTIAL (mlx-lm inference wired; security scan heuristic)
- mc_infer_run: PARTIAL (mlx-lm inference wired; security scan heuristic)
- mc_infer_batch: PARTIAL (mlx-lm inference wired; security scan heuristic)
- mc_infer_suite: PARTIAL (mlx-lm inference wired; security scan heuristic)
- mc_system_status: PARTIAL (readiness details need parity)
- mc_validate_train: PARTIAL
- mc_estimate_train: PARTIAL
- mc_dataset_validate: PARTIAL
- mc_doc_convert: DONE (MCP tool wired)
- mc_model_fetch: PARTIAL
- mc_checkpoint_export: PARTIAL
- mc_geometry_validate: DONE (MCP tool wired)
- mc_geometry_training_status: DONE (MCP tool wired)
- mc_geometry_training_history: DONE (MCP tool wired)
- mc_geometry_path_detect: DONE (MCP tool wired)
- mc_geometry_path_compare: DONE (MCP tool wired)
- mc_geometry_primes_list: DONE (MCP tool wired)
- mc_geometry_primes_probe: DONE (MCP tool wired)
- mc_geometry_primes_compare: DONE (MCP tool wired)
- mc_geometry_crm_build: DONE (MCP tool wired)
- mc_geometry_crm_compare: DONE (MCP tool wired)
- mc_geometry_stitch_analyze: PARTIAL (heuristic stitching)
- mc_geometry_stitch_apply: PARTIAL (heuristic stitching)
- mc_safety_circuit_breaker: DONE (MCP tool wired)
- mc_safety_persona_drift: DONE (MCP tool wired)
- mc_geometry_dare_sparsity: DONE (MCP tool wired)
- mc_geometry_dora_decomposition: DONE (MCP tool wired)
- mc_geometry_gromov_wasserstein: DONE (MCP tool wired)
- mc_geometry_intrinsic_dimension: DONE (MCP tool wired)
- mc_geometry_topological_fingerprint: DONE (MCP tool wired)
- mc_geometry_sparse_domains: DONE (MCP tool wired)
- mc_geometry_sparse_locate: DONE (MCP tool wired)
- mc_geometry_refusal_pairs: DONE (MCP tool wired)
- mc_geometry_refusal_detect: DONE (MCP tool wired)
- mc_geometry_persona_traits: DONE (MCP tool wired)
- mc_geometry_persona_extract: DONE (MCP tool wired)
- mc_geometry_persona_drift: DONE (MCP tool wired)
- mc_geometry_manifold_cluster: DONE (MCP tool wired)
- mc_geometry_manifold_dimension: DONE (MCP tool wired)
- mc_geometry_manifold_query: DONE (MCP tool wired)
- mc_geometry_transport_merge: DONE (MCP tool wired)
- mc_geometry_transport_synthesize: DONE (MCP tool wired)
- mc_rag_build/query/list/delete: PARTIAL (in-memory index, no persistent store yet)
- mc_storage_usage: PARTIAL (field naming parity pending)
- mc_storage_cleanup: DONE (MCP tool wired)
- mc_thermo_analyze/path/entropy/measure/detect/detect_batch: DONE (linguistic thermodynamics + ridge cross + phase transition)
- mc_adapter_inspect: DONE (MCP tool wired)
- mc_ensemble_list/delete: DONE (MCP tool wired)
- mc_adapter_blend: DONE (MCP tool wired)
- mc_ensemble_create/apply: DONE (MCP tool wired)

## MCP Resources

- mc://models: PARTIAL
- mc://jobs: PARTIAL
- mc://checkpoints: PARTIAL
- mc://datasets: PARTIAL
- mc://system: PARTIAL
