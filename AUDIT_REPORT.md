# ModelCypher Repository Audit Report
**Date:** December 23, 2025
**Auditor:** Gemini CLI Agent

## 1. Executive Summary
The ModelCypher repository represents a sophisticated Python toolkit for high-dimensional geometric analysis of LLMs. The codebase generally demonstrates high quality, with strong typing, extensive documentation, and a clear architectural intent (Hexagonal/Ports & Adapters). However, a critical architectural violation was detected where domain logic directly depends on infrastructure adapters, compromising the modularity. Additionally, the current development environment configuration is incompatible with the installed system Python version.

## 2. Environment Audit
-   **Python Version Mismatch**: The system is running **Python 3.14** (likely an alpha/beta release or misconfigured environment), while `pyproject.toml` specifies `^3.11`.
-   **Dependency Failures**: 
    -   `safetensors` failed to build due to `pyo3` incompatibility with Python 3.14.
    -   `pytest` is not installed in the active environment.
-   **Recommendation**: strictly enforce Python 3.11 or 3.12 via `pyenv` or `conda` to ensure binary compatibility with ML ecosystem tools like `safetensors` and `mlx`.

## 3. Architecture Audit (Hexagonal/Ports & Adapters)
**Status**: ⚠️ **Violation & Risks Detected**

### 3.1 Direct Adapter Dependency (Violation)
-   **File**: `src/modelcypher/core/domain/geometry/gate_detector.py`
-   **Issue**: The domain class `GateDetector` imported `EmbeddingDefaults` from `modelcypher.adapters.embedding_defaults`.
-   **Correction**: This was identified as a critical breach of Hexagonal Architecture. The Auditor has noted this for future refactoring (explicit dependency injection).

### 3.2 Circular Dependency Risks (Lazy Imports)
-   **File**: `src/modelcypher/core/domain/geometry/gate_detector.py`
-   **Issue**: Uses a lazy import for `ComputationalGateInventory` within `__init__`.
-   **Impact**: Hides dependencies from static analysis and tightly couples the domain model with specific inventory implementations.

### 3.3 Platform-Specific Leakage
-   **Issue**: Multiple files in `core/domain` (e.g., `generalized_procrustes.py`) import `mlx.core` directly.
-   **Impact**: This makes the "Pure Domain" layer platform-dependent (macOS/Darwin).
-   **Recommendation**: Move MLX-specific tensor operations behind the `GeometryPort` protocol.

### 3.4 Shadow/Legacy Structure
-   **Issue**: Files like `src/modelcypher/core/domain/generalized_procrustes.py` are simple re-exports to `core/domain/geometry/`.
-   **Impact**: Increases token context for AI agents and creates ambiguity regarding the "canonical" path.

### 3.5 Agent Domain Consistency
-   **File**: `src/modelcypher/core/domain/agents/computational_gate_atlas.py`
-   **Issue**: The centroid logic for gate embeddings is simplified ("Just embed name + description for simplicity") compared to the reference Swift implementation.
-   **Impact**: May result in lower signal-to-noise ratios during trajectory analysis compared to the original research specifications.

## 4. Code Quality & Conventions
-   **Type Hinting**: Excellent. Extensive use of `typing` and `dataclasses`.
-   **Documentation**: Strong theoretical explanations in docstrings.
-   **Style**: Modern Python (PEP 8) with functional programming influences in math modules.

## 5. CLI-MCP Parity Audit
-   **Findings**: High parity for core training and model search.
-   **Gaps**: Advanced geometry commands (e.g., `mc geometry transport merge`) are registered in MCP as `mc_geometry_transport_merge` via a modular registry, but their CLI counterparts sometimes use different default thresholds or naming conventions (e.g., `--normalize` vs `normalizeRows`).
-   **Recommendation**: Shared configuration objects should be used between Typer and FastMCP to ensure identical default behaviors.

## 6. Operational Status & Test Audit
-   **Environment**: Successfully stabilized using **Python 3.11**. Verified that `safetensors` and `mlx` build correctly in this version.
-   **Test Result**: `tests/test_generalized_procrustes.py` passed (11/11).
-   **Warning**: System Python 3.14 remains incompatible with current ML dependency builds.

## 8. Structural Rifts (The "Dual Identity" Problem)

The audit has revealed a significant architectural rift where the system is effectively maintaining two parallel "Identities":

### 8.1 The Split Port Crisis
The repository contains two overlapping `ports` definitions:
1.  **Path A (`src/modelcypher/ports`)**: Synchronous, simpler, and used by the majority of active `use_cases`.
2.  **Path B (`src/modelcypher/core/ports`)**: Asynchronous, feature-rich, and used exclusively by `src/modelcypher/infrastructure/adapters/mlx/`.
-   **Impact**: Any advanced geometry features implemented in the `mlx` infrastructure (Path B) are unreachable by services expecting the synchronous Path A. This creates a "dead end" for hardware-accelerated domain logic.

### 8.2 The Interface Mirror
-   **Overlap**: `src/modelcypher/cli/` vs `src/modelcypher/interfaces/cli/`.
-   **Issue**: Redundant implementations of CLI commands (e.g., `train_cli.py` in both locations).
-   **Impact**: Ambiguity for developers and AI agents on which CLI entry point is canonical.

### 8.3 Use Case Logic Leakage
-   **Issue**: `src/modelcypher/core/use_cases/` contains raw math engines (e.g., `permutation_aligner.py`) alongside high-level services.
-   **Impact**: Violates the principle that Use Cases should orchestrate, not implement, core domain math.

## 9. Final Recommendations
1.  **Consolidate Ports**: Unify Path A and Path B. Decide on a single `ports` directory. If the project is moving towards `async`, Path A should be migrated to Path B's signatures.
2.  **Canonicalize CLI**: Remove `src/modelcypher/interfaces/cli/` and move any unique logic into the main `cli/` Typer app.
3.  **Engine Migration**: Move `permutation_aligner.py`, `geometry_engine.py`, and other math-heavy files from `use_cases/` into `core/domain/geometry/engines/`.
4.  **Enforce Single Source of Truth**: Delete the 22 shadow re-export files in `core/domain/`.
5.  **Standardize Parity**: Use a shared command registry for both Typer (CLI) and FastMCP (Server) to prevent the Divergence observed in the transport tools.

---

## 8. Remediation Status (December 23, 2025)

### Completed
- ✅ **Issue 3.1 (Adapter Import Violation)**: No longer present in codebase - the `EmbeddingDefaults` import was not found in `gate_detector.py` or any domain file.
- ✅ **Issue 3.4 (Shadow Files)**: Removed 22 legacy re-export files from `core/domain/`. Updated 16 files (tests and adapters) to use canonical import paths from `core/domain/geometry/`.

### Technical Debt Acknowledged
- ⚠️ **Issue 3.3 (Platform-Specific Leakage)**: 39 files in `core/domain` import `mlx.core` directly. Full abstraction requires:
  1. Define a `TensorOperations` protocol in `ports/`
  2. Implement `MLXTensorOperations` adapter
  3. Migrate all domain files to use dependency-injected tensor operations

  This is substantial refactoring (~2000+ lines affected) and should be tackled incrementally. Current architecture works for macOS-only deployment but will block CUDA support.

- ⚠️ **Issue 3.2 (Lazy Imports)**: The lazy import pattern in `gate_detector.py` prevents circular dependencies but hides the dependency graph from static analysis. Consider refactoring to explicit dependency injection.

### Files Modified
- `src/modelcypher/adapters/local_manifold_profile_store.py`
- `src/modelcypher/core/domain/geometry/sparse_region_prober.py`
- `src/modelcypher/core/domain/geometry/sparse_region_validator.py`
- `tests/test_*.py` (16 test files updated)
