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
| src/modelcypher/core/ | 316 | Pending |
| src/modelcypher/cli/ | 53 | Pending |
| src/modelcypher/ports/ | 15 | Pending |
| src/modelcypher/adapters/ | 15 | Pending |
| src/modelcypher/infrastructure/ | 11 | Pending |
| src/modelcypher/mcp/ | 11 | Pending |
| src/modelcypher/utils/ | 8 | Pending |
| src/modelcypher/backends/ | 8 | Pending |
| tests/ | 156 | Pending |
| examples/ | 5 | Pending |
| scripts/ | 7 | Pending |

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
| __init__.py | | | | Pending |
| asif_packager.py | | | | Pending |
| embedding_defaults.py | | | | Pending |
| embedding_http.py | | | | Pending |
| embedding_mlx.py | | | | Pending |
| filesystem_storage.py | | | | Pending |
| hf_hub.py | | | | Pending |
| hf_model_search.py | | | | Pending |
| local_exporter.py | | | | Pending |
| local_inference.py | | | | Pending |
| local_manifold_profile_store.py | | | | Pending |
| local_training.py | | | | Pending |
| mlx_model_loader.py | | | | Pending |
| model_loader.py | | | | Pending |
| training_dataset.py | | | | Pending |

---

## DUPLICATE FUNCTION TRACKER

Functions that appear in multiple files:

| Function Name | Files | Notes |
|--------------|-------|-------|
| | | |

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

- Files audited: 20/607 (examples/, scripts/, backends/)
- Issues found: 1
- Duplicates found: 0
- Orphans found: 0
