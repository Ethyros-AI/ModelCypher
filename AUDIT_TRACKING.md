# ModelCypher Python File Audit

**Started**: 2025-12-27
**Total Files**: 607 Python files
**Source Files**: 439
**Test Files**: 156
**Examples/Scripts**: 12

## Audit Objectives
1. Check for duplicate functions across the codebase
2. Verify each file is properly wired in (imported/used appropriately)
3. Document purpose and status of each file

## File Counts by Directory
| Directory | Count | Status |
|-----------|-------|--------|
| src/modelcypher/core/ | 318 | **AUDITED** |
| src/modelcypher/cli/ | 53 | **AUDITED** |
| src/modelcypher/ports/ | 15 | **AUDITED** |
| src/modelcypher/adapters/ | 15 | **AUDITED** |
| src/modelcypher/infrastructure/ | 11 | **AUDITED** |
| src/modelcypher/mcp/ | 11 | **AUDITED** |
| src/modelcypher/utils/ | 8 | **AUDITED** |
| src/modelcypher/backends/ | 8 | **AUDITED** |
| tests/ | 159 | **AUDITED** |
| examples/ | 5 | **AUDITED** |
| scripts/ | 7 | **AUDITED** |

---

## AUDIT LOG

### Session 1 - 2025-12-27

---

## 1. examples/ Directory (5 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| 01_basic_geometry_probe.py | Demo model probing via ModelProbeService | Yes | No | **AUDITED** |
| 02_safety_audit.py | Demo safety audit via SafetyProbeService & EntropyProbeService | Yes | No | **AUDITED** |
| 03_adapter_blending.py | Demo adapter blending via AdapterService | Yes | No | **AUDITED** |
| 04_entropy_analysis.py | Demo entropy analysis via ThermoService | Yes | No | **AUDITED** |
| 05_model_merge.py | Demo model merging | **BROKEN** | No | **ISSUE #1** |

### Notes for examples/:
- Examples 01-04 correctly import from use_cases layer
- No duplicate functions within examples
- **ISSUE #1**: `05_model_merge.py` imports non-existent `ModelMergeService` and `GeometricMergeConfig` from `model_merge_service.py` which doesn't exist. Should use `UnifiedGeometricMerger` from `unified_geometric_merge.py` or `GeometricMergeOrchestrator` from `geometric_merge_orchestrator.py`

---

## 2. scripts/ Directory (7 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| audit_lines.py | Compare Swift/Python line counts for code conversion | Standalone | No | **AUDITED** |
| benchmark_computation_cache.py | Benchmark ComputationCache performance | Yes (core.domain) | No | **AUDITED** |
| modernize_typing.py | One-time migration tool for Python typing updates | Standalone | No | **AUDITED** |
| run_validation_suite.py | Comprehensive CLI test runner via subprocess | Standalone | No | **AUDITED** |
| run_verification_tests.py | Manual test runner importing from tests/ | Yes (tests/) | No | **AUDITED** |
| test_merge_with_caching.py | Test cache with real model weights | Yes (core.domain) | No | **AUDITED** |
| verify_mlx_freeze.py | Test MLX freeze behavior | Standalone (mlx) | No | **AUDITED** |

### Notes for scripts/:
- All scripts are standalone utilities for development/testing
- 3 scripts (benchmark_computation_cache, test_merge_with_caching, run_verification_tests) import from modelcypher
- All imports verified working
- No duplicate functions across scripts
- No issues found

---

## 3. src/modelcypher/backends/ (8 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init with lazy imports for backends | Yes (18 imports) | No | **AUDITED** |
| cuda_backend.py | CUDA/PyTorch impl of Backend protocol (652 lines) | Yes | No | **AUDITED** |
| cuda_model_probe.py | PyTorch impl of BaseModelProbe | Yes (model_probe_service) | Parallel impl | **AUDITED** |
| jax_backend.py | JAX impl of Backend protocol (722 lines) | Yes | No | **AUDITED** |
| jax_model_probe.py | JAX impl of BaseModelProbe | Yes (model_probe_service) | Parallel impl | **AUDITED** |
| mlx_backend.py | MLX impl of Backend protocol (724 lines) | Yes | No | **AUDITED** |
| mlx_model_probe.py | MLX impl of BaseModelProbe | Yes (model_probe_service) | Parallel impl | **AUDITED** |
| safe_gpu.py | MLX eval() helper (55 lines) | Yes (mlx_backend) | No | **AUDITED** |

### Notes for backends/:
- All 3 backends (MLX, JAX, CUDA) properly implement Backend protocol from `ports/backend.py`
- All 3 model probes properly extend BaseModelProbe from `ports/model_probe.py`
- Model probes have parallel implementations (same methods, backend-specific tensor ops) - intentional design
- ModelProbeService in `use_cases/model_probe_service.py` orchestrates backend selection
- CUDABackend.quantize/dequantize raise NotImplementedError (by design)
- No orphans, no true duplicates
- Clean hexagonal architecture: backends implement ports

---

## 4. src/modelcypher/adapters/ (15 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init, re-exports adapter classes | Yes (44 imports in 24 files) | No | **AUDITED** |
| asif_packager.py | ASI Format packager for model exports | Yes (container.py) | No | **AUDITED** |
| embedding_defaults.py | Default embedding configuration provider | Yes (embedding_mlx) | No | **AUDITED** |
| embedding_http.py | HTTP-based EmbeddingProvider impl | Yes (container.py) | No | **AUDITED** |
| embedding_mlx.py | MLX-based EmbeddingProvider impl (107 lines) | Yes (container.py) | No | **AUDITED** |
| filesystem_storage.py | FileSystemStore impl (509 lines) - ModelStore, JobStore, etc. | Yes (widely used) | No | **AUDITED** |
| hf_hub.py | HuggingFace Hub integration for downloads | Yes (cli, mcp) | No | **AUDITED** |
| hf_model_search.py | ModelSearchService impl for HF hub (308 lines) | Yes (container.py) | No | **AUDITED** |
| local_exporter.py | Exporter impl for npz, safetensors, mlx formats (153 lines) | Yes (container.py) | No | **AUDITED** |
| local_inference.py | LocalInferenceEngine impl (1096 lines) - HiddenStateEngine | Yes (cli, mcp) | No | **AUDITED** |
| local_manifold_profile_store.py | ManifoldProfileStore impl (414 lines) | Yes (container.py) | No | **AUDITED** |
| local_training.py | LocalTrainingEngine impl (400 lines) - TrainingEngine | Yes (container.py) | No | **AUDITED** |
| mlx_model_loader.py | MLXModelLoader wrapper class (107 lines) | Yes (delegates to model_loader) | Wrapper | **AUDITED** |
| model_loader.py | Core model loading functions for training (180 lines) | Yes (mlx_model_loader, local_training) | No | **AUDITED** |
| training_dataset.py | TrainingDataset iterable (210 lines) | Yes (local_training) | No | **AUDITED** |

### Notes for adapters/:
- All 15 adapters properly implement their respective ports from `ports/`
- Clean hexagonal architecture: adapters implement interfaces defined in ports
- **mlx_model_loader.py** is a wrapper around **model_loader.py** - intentional delegation pattern, not duplication
- FileSystemStore is the largest (509 lines), implementing 4 storage ports
- LocalInferenceEngine is substantial (1096 lines) with hidden state capture
- No orphaned files - all adapters are imported via `infrastructure/container.py` or directly
- Total of 44 imports across 24 files using adapters module
- No true duplicates found

---

## 5. src/modelcypher/ports/ (15 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init with clean exports (111 lines) | Yes | No | **AUDITED** |
| backend.py | Backend Protocol - 58+ tensor ops (334 lines) | Yes (3 impl) | No | **AUDITED** |
| model_probe.py | ModelProbePort + BaseModelProbe ABC (196 lines) | Yes (3 impl) | No | **AUDITED** |
| storage.py | 5 storage protocols (72 lines) | Yes (filesystem_storage) | No | **AUDITED** |
| embedding.py | EmbeddingProvider Protocol (29 lines) | Yes (2 impl) | No | **AUDITED** |
| inference.py | InferenceEngine + HiddenStateEngine (42 lines) | Yes (local_inference) | No | **AUDITED** |
| training.py | TrainingEngine Protocol (36 lines) | Yes (local_training) | No | **AUDITED** |
| exporter.py | Exporter Protocol (28 lines) | Yes (local_exporter) | No | **AUDITED** |
| hub.py | HubAdapterPort Protocol (86 lines) | Yes (hf_hub) | No | **AUDITED** |
| model_loader.py | ModelLoaderPort Protocol (97 lines) | Yes (mlx_model_loader) | No | **AUDITED** |
| model_search.py | ModelSearchService Protocol (30 lines) | Yes (hf_model_search) | No | **AUDITED** |
| concept_discovery.py | ConceptDiscoveryPort Protocol (41 lines) | Yes (async) | No | **AUDITED** |
| async_embeddings.py | EmbedderPort Protocol - async (40 lines) | Yes | No | **AUDITED** |
| async_geometry.py | GeometryPort Protocol - async (139 lines) | Yes | No | **AUDITED** |
| async_inference.py | InferenceEnginePort Protocol - async (70 lines) | Yes | No | **AUDITED** |

### Notes for ports/:
- All ports are properly defined as Protocol classes
- Clean separation: sync ports (embedding, inference, training, exporter) and async ports (async_*)
- Backend Protocol is the core abstraction (334 lines, 58+ methods)
- Each sync port has a corresponding adapter implementation
- No duplicates - EmbeddingProvider (sync) vs EmbedderPort (async) are intentionally separate
- All ports are imported and used by adapters and use_cases
- Proper hexagonal architecture: ports define interfaces, adapters implement them

---

## 6. src/modelcypher/utils/ (8 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Empty module init (19 lines) | Yes | No | **AUDITED** |
| errors.py | ErrorDetail dataclass/exception (57 lines) | Yes (cli, mcp) | No | **AUDITED** |
| json.py | json_default() + dump_json() helpers (54 lines) | Yes (cli, domain) | No | **AUDITED** |
| limits.py | Field size constants (25 lines) | Yes | No | **AUDITED** |
| locks.py | FileLock + FileLockError (79 lines) | Yes (local_training, local_inference) | No | **AUDITED** |
| logging.py | JSONFormatter + configure_logging() (61 lines) | Yes (cli) | No | **AUDITED** |
| paths.py | expand_path() + ensure_dir() + get_jobs_dir() (40 lines) | Yes (adapters) | No | **AUDITED** |
| text.py | truncate() function (27 lines) | Yes | No | **AUDITED** |

### Notes for utils/:
- All 8 utils are small, focused helper modules
- Total of 48 imports across 42 files using utils module
- No duplicates - each utility serves a distinct purpose
- No orphans - all utilities are actively used
- FileLock (locks.py) is critical for preventing concurrent training/inference

---

## 7. src/modelcypher/infrastructure/ (11 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init (38 lines) - Exports PortRegistry, ServiceFactory | Yes | No | **AUDITED** |
| container.py | PortRegistry dataclass (135 lines) - Composition root | Yes (cli, mcp) | No | **AUDITED** |
| service_factory.py | ServiceFactory (191 lines) - DI factory for services | Yes (cli, mcp) | No | **AUDITED** |
| services/__init__.py | Package init (20 lines) | Yes | No | **AUDITED** |
| services/memory.py | MLXMemoryService singleton (81 lines) | Yes | No | **AUDITED** |
| adapters/mlx/geometry.py | MLXGeometryAdapter impl GeometryPort (352 lines) | Yes (async) | No | **AUDITED** |
| adapters/mlx/inference.py | MLXInferenceAdapter impl InferenceEnginePort (126 lines) | Yes (async) | No | **AUDITED** |
| adapters/mlx/embeddings.py | MockMLXEmbedder (46 lines) - test mock for EmbedderPort | Yes (testing) | No | **AUDITED** |
| adapters/mlx/concepts.py | MLXConceptAdapter impl ConceptDiscoveryPort (366 lines) | Yes (async) | No | **AUDITED** |
| adapters/mlx/merger.py | TransportGuidedMerger (225 lines) - GW-based model merger | Yes (geometry) | No | **AUDITED** |
| adapters/mlx/optimal_transport.py | GromovWassersteinSolver (113 lines) | Yes (merger) | No | **AUDITED** |

### Notes for infrastructure/:
- **container.py** is the composition root - ONLY place where production adapters are instantiated
- **service_factory.py** provides proper dependency injection for all services
- MLX adapters in `adapters/mlx/` implement async ports for geometry, inference, concept discovery
- TransportGuidedMerger implements Gromov-Wasserstein optimal transport for model merging
- MLXMemoryService is a singleton for memory monitoring via MLX metal APIs
- MockMLXEmbedder is for testing only - returns random normalized vectors
- Clean hexagonal architecture: infrastructure wires adapters to ports

---

## 8. src/modelcypher/mcp/ (11 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init (19 lines) | Yes | No | **AUDITED** |
| server.py | Main MCP server (2815 lines) - FastMCP with 148+ tools | Yes (entry point) | No | **AUDITED** |
| security.py | OAuth 2.1 + ConfirmationManager (434 lines) | Yes (server) | No | **AUDITED** |
| tasks.py | TaskManager async framework (391 lines) | Yes (tools) | No | **AUDITED** |
| tools/__init__.py | Tools package init (31 lines) | Yes | No | **AUDITED** |
| tools/common.py | ServiceContext + utilities (438 lines) | Yes (all tools) | No | **AUDITED** |
| tools/tasks.py | Task management MCP tools (277 lines) | Yes (server) | No | **AUDITED** |
| tools/geometry.py | Geometry MCP tools (2579 lines) | Yes (server) | No | **AUDITED** |
| tools/merge_entropy.py | Merge entropy MCP tools (508 lines) | Yes (server) | No | **AUDITED** |
| tools/safety_entropy.py | Safety entropy MCP tools (495 lines) | Yes (server) | No | **AUDITED** |
| tools/agent.py | Agent MCP tools (268 lines) | Yes (server) | No | **AUDITED** |

### Notes for mcp/:
- **server.py** is the largest file (2815 lines) containing 148+ MCP tool definitions
- Uses FastMCP framework for tool registration
- **security.py** implements MCP 2025-06-18 spec: OAuth 2.1, destructive operation confirmation
- **tasks.py** provides async task management with cleanup, progress updates, cancellation
- **ServiceContext** in common.py is the DI container for all MCP services (lazy-loaded)
- Tool files are organized by domain: geometry, safety_entropy, merge_entropy, agent, tasks
- All tools properly wire to use_cases services via ServiceContext
- No orphaned files - all are registered/imported by server.py

---

## 9. src/modelcypher/cli/ (53 files)

### Root CLI Files (7 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| __init__.py | Module init (28 lines) | Yes | No | **AUDITED** |
| app.py | Main CLI app entry point (440 lines) - Registers all commands | Yes (pyproject.toml) | No | **AUDITED** |
| composition.py | DI helpers get_*_service() | Yes (commands) | No | **AUDITED** |
| context.py | CLIContext dataclass for global state | Yes (all commands) | No | **AUDITED** |
| output.py | write_output(), write_error() helpers | Yes (all commands) | No | **AUDITED** |
| presenters.py | Output formatting utilities | Yes (commands) | No | **AUDITED** |
| typer_compat.py | Typer compatibility patches | Yes (app.py) | No | **AUDITED** |

### Top-Level Commands (18 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| commands/__init__.py | Package init | Yes | No | **AUDITED** |
| adapter.py | Adapter inspect/merge/project commands | Yes (app.py) | No | **AUDITED** |
| agent.py | Agent trace import/inspect | Yes (app.py) | No | **AUDITED** |
| agent_eval.py | Agent evaluation suite | Yes (app.py) | No | **AUDITED** |
| dashboard.py | Dashboard commands | Yes (app.py) | No | **AUDITED** |
| ensemble.py | Ensemble routing commands | Yes (app.py) | No | **AUDITED** |
| entropy.py | Entropy analysis commands | Yes (app.py) | No | **AUDITED** |
| eval.py | Model evaluation commands | Yes (app.py) | No | **AUDITED** |
| help_cmd.py | Help/completions/schema | Yes (app.py) | No | **AUDITED** |
| infer.py | Inference commands | Yes (app.py) | No | **AUDITED** |
| job.py | Job management commands | Yes (app.py) | No | **AUDITED** |
| model.py | Model list/probe/merge commands (450 lines) | Yes (app.py) | No | **AUDITED** |
| research.py | Research taxonomy/sparse-region/afm | Yes (app.py) | No | **AUDITED** |
| safety.py | Safety adapter-probe | Yes (app.py) | No | **AUDITED** |
| stability.py | Stability analysis | Yes (app.py) | No | **AUDITED** |
| storage.py | Storage status/cleanup | Yes (app.py) | No | **AUDITED** |
| system.py | System status/probe | Yes (app.py) | No | **AUDITED** |
| thermo.py | Thermodynamic analysis (855 lines) | Yes (app.py) | No | **AUDITED** |
| train.py | Training commands | Yes (app.py) | No | **AUDITED** |

### Geometry Sub-Commands (27 files)

| File | Purpose | Wired? | Duplicates? | Status |
|------|---------|--------|-------------|--------|
| geometry/__init__.py | Package init - imports all sub-modules | Yes (app.py) | No | **AUDITED** |
| geometry/helpers.py | Shared backbone resolution + forward pass (317 lines) | Yes (geometry commands) | No | **AUDITED** |
| geometry/atlas.py | Unified atlas probe commands | Yes | No | **AUDITED** |
| geometry/crm.py | Concept Response Matrix | Yes | No | **AUDITED** |
| geometry/emotion.py | Emotion concept analysis | Yes | No | **AUDITED** |
| geometry/geom_adapter.py | Adapter geometry analysis | Yes | No | **AUDITED** |
| geometry/interference.py | Interference prediction | Yes | No | **AUDITED** |
| geometry/invariant.py | Invariant layer mapping | Yes | No | **AUDITED** |
| geometry/manifold.py | Manifold clustering/dimension | Yes | No | **AUDITED** |
| geometry/merge_entropy.py | Merge entropy analysis | Yes | No | **AUDITED** |
| geometry/metrics.py | GW distance, intrinsic dimension | Yes | No | **AUDITED** |
| geometry/moral.py | Moral foundations geometry | Yes | No | **AUDITED** |
| geometry/path.py | Path integration | Yes | No | **AUDITED** |
| geometry/persona.py | Persona vector extraction | Yes | No | **AUDITED** |
| geometry/primes.py | Semantic primes | Yes | No | **AUDITED** |
| geometry/refinement.py | Refinement density | Yes | No | **AUDITED** |
| geometry/refusal.py | Refusal direction detection | Yes | No | **AUDITED** |
| geometry/safety.py | Geometry safety analysis | Yes | No | **AUDITED** |
| geometry/social.py | Social geometry (power, kinship) | Yes | No | **AUDITED** |
| geometry/sparse.py | Sparse region analysis | Yes | No | **AUDITED** |
| geometry/spatial.py | 3D world model metrology (930 lines) | Yes | No | **AUDITED** |
| geometry/stitch.py | Manifold stitching | Yes | No | **AUDITED** |
| geometry/temporal.py | Temporal topology | Yes | No | **AUDITED** |
| geometry/training.py | Training geometry | Yes | No | **AUDITED** |
| geometry/transfer.py | Cross-architecture transfer | Yes | No | **AUDITED** |
| geometry/transport.py | Transport-guided merging | Yes | No | **AUDITED** |
| geometry/waypoint.py | Domain geometry waypoints | Yes | No | **AUDITED** |

### Notes for cli/:
- **app.py** is the main entry point (registered in pyproject.toml as `mc` and `modelcypher`)
- All 53 files properly wired via `add_typer()` in app.py
- **`_context()` helper appears in 44 files** - NOT a duplicate issue; standard Typer pattern where each command file defines its own local `_context(ctx) -> CLIContext` helper
- **geometry/helpers.py** provides canonical implementations (resolve_model_backbone, forward_through_backbone) used by multiple geometry commands
- Hierarchical structure: app.py → commands/* → commands/geometry/*
- Largest files: spatial.py (930 lines), thermo.py (855 lines), model.py (450 lines)
- No orphaned files, no true duplicates

---

## DUPLICATE FUNCTION TRACKER

Functions that appear in multiple files:

| Function Name | Files | Notes |
|--------------|-------|-------|
| `_context()` | 44 CLI files | NOT duplicate - Standard Typer pattern, each command file has local helper |
| `LoRAConfig` | training/types.py + lora_mlx.py/lora_cuda.py/lora_jax.py | INTENTIONAL - types.py is fallback, platform-specific versions override via __init__.py |
| `IntrinsicDimensionResult` | geometry/types.py + use_cases/geometry_metrics_service.py | INTENTIONAL - Different semantics: domain type (generic) vs service type (with interpretation) |
| `compute_entropy_*` | inference/dual_path_jax.py + dual_path_cuda.py | INTENTIONAL - Backend-specific implementations |

**Cross-Reference Result: NO true duplicates found.** All cases of same-named classes/functions are either:
1. Backend-specific implementations (by design)
2. Platform abstraction (types.py fallback overwritten by platform version)
3. Different semantic contexts (domain vs service layer)

---

## ORPHANED FILES

Files that are not imported or used anywhere:

| File | Notes |
|------|-------|
| | |

---

## ISSUES FOUND

| Issue # | File | Description | Severity |
|---------|------|-------------|----------|
| 1 | examples/05_model_merge.py | Imports non-existent `ModelMergeService` from `model_merge_service.py` | High - Example broken |

---

## PROGRESS SUMMARY

### ✅ AUDIT COMPLETE - 2025-12-27

- **Files audited**: 610/610 (all directories)
- **Issues found**: 1 (examples/05_model_merge.py - broken import)
- **True duplicates found**: 0
- **Orphans found**: 0

### Dead Code Removed:
- `thermodynamics/` shim (2 files) - unused backward compat package
- `dataset/` - empty directory with only __pycache__
- `src/modelcypher/tests/` - misplaced empty directory
- `tests/fixtures/` - empty placeholder directory

---

### Completed: core/ (318 files)

**Verified:**
- domain/__init__.py - Lazy loading system, clean
- use_cases/ (52 files) - Properly wired to CLI/MCP via ServiceFactory
- geometry/ (90 files) - No duplicate functions (compute_cka, frechet_mean, geodesic_distance all unique)
- thermo/ (7 files) - Linguistic thermodynamics, properly used
- training/ (35 files) - Backend-specific implementations (_mlx, _jax, _cuda) - proper pattern
- safety/ (34 files) - Including calibration/, sidecar/, stability_suite/ subdirs
- vocabulary/ (5 files) - Cross-vocab merger, actively used by merge stages
- agents/ (28 files) - Unified atlas, semantic primes, trace analytics - actively used
- entropy/ (19 files) - Entropy tracking, model state - actively used
- All 15 subdirectories verified as properly wired (137 cross-imports)
- Core math functions: NOT duplicated across modules

### Completed: tests/ (159 files)

**Structure:**
- Root level: 153 files (151 test_*.py + conftest.py + __init__.py)
- domain/: 2 files (safety, geometry subdirs)
- integration/: 2 files
- mcp/: 2 files
- All test files follow test_*.py naming convention
- conftest.py provides backend fixtures (any_backend, mlx_backend, etc.)
- Clean organization, no orphaned tests

---

## FINAL AUDIT CONCLUSIONS

### Codebase Health: ✅ EXCELLENT

1. **No duplicate functions** - All same-named items are intentional (backend implementations, platform abstractions, or different semantic contexts)

2. **Clean hexagonal architecture** - Ports define interfaces, adapters implement them, no architecture violations

3. **Proper wiring** - All 610 files are either:
   - Imported directly (via app.py, container.py, __init__.py lazy loading)
   - Used by tests
   - Standalone scripts (examples/, scripts/)

4. **Dead code removed** - Backward compatibility shims and empty directories cleaned up

5. **One known issue** - examples/05_model_merge.py needs update to use current merge API

### Architecture Patterns Verified:
- Hexagonal (Ports & Adapters)
- Backend abstraction (MLX, JAX, CUDA)
- Platform-specific implementations (*_mlx.py, *_jax.py, *_cuda.py)
- Lazy loading via `__getattr__` in `__init__.py`
- Dependency injection via PortRegistry + ServiceFactory
- MCP tools with ServiceContext for DI
