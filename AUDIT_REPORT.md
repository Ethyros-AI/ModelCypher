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

- ✅ **Issue 3.2 (Lazy Imports)**: RESOLVED - The redundant lazy import in `gate_detector.py` line 122 was removed. The top-level import on line 44 already imports `ComputationalGateInventory`, and there is no circular dependency (the atlas module does not import from gate_detector).

- ⚠️ **Issue 3.5 (Agent Domain Consistency)**: PARTIALLY ADDRESSED - The simplified centroid logic in `computational_gate_atlas.py` now has an alternative `signature_volume_aware()` method (CABE-4 implementation) that uses triangulated embeddings (name, description, examples, polyglot) to estimate concept volumes with Riemannian density. The default `signature()` method retains simplified centroid for backwards compatibility. `emotion_concept_atlas.py` still uses simplified centroid logic (technical debt for future CABE-4 extension).

### Files Modified
- `src/modelcypher/adapters/local_manifold_profile_store.py`
- `src/modelcypher/core/domain/geometry/sparse_region_prober.py`
- `src/modelcypher/core/domain/geometry/sparse_region_validator.py`
- `tests/test_*.py` (16 test files updated)

---

## 10. Remediation Status Update (December 23, 2025 - Continued)

### Structural Rifts Addressed

#### Issue 8.2 (Interface Mirror) - ✅ RESOLVED
- **Action Taken**: Removed dead code from `src/modelcypher/interfaces/` directory and `src/modelcypher/main.py`
- **Files Deleted**:
  - `src/modelcypher/main.py` (legacy argparse CLI entry point)
  - `src/modelcypher/interfaces/cli/train_cli.py`
  - `src/modelcypher/interfaces/cli/inspect_cli.py`
  - `src/modelcypher/interfaces/cli/dynamics_cli.py`
  - `src/modelcypher/interfaces/cli/eval_cli.py`
  - `src/modelcypher/interfaces/__init__.py`
  - `src/modelcypher/interfaces/cli/__init__.py`
- **Verification**: The actual CLI entry points are correctly defined in `pyproject.toml`:
  - `mc` → `modelcypher.cli.app:app` (Typer)
  - `modelcypher` → `modelcypher.cli.app:app` (Typer)
  - `modelcypher-mcp` → `modelcypher.mcp.server:main`

#### Issue 8.1 (Split Port Crisis) - ⚠️ CLARIFIED (Not a Crisis)
- **Investigation Findings**:
  - `src/modelcypher/ports/` (9 files) - Synchronous ports used by 30 files across the codebase
  - `src/modelcypher/core/ports/` (4 files) - Async ports used exclusively by `infrastructure/adapters/mlx/geometry.py`
- **Assessment**: These are intentionally separate interfaces for different use cases:
  - Sync ports: Used by services, tests, and adapters for blocking operations
  - Async ports: Used by MLX adapter for GPU-accelerated async operations
- **No Action Required**: The dual port structure is intentional and serves different runtime requirements

#### Issue 8.3 (Use Case Logic Leakage) - ⚠️ CLARIFIED (Architecture Sound)
- **Investigation Findings**:
  - `permutation_aligner.py` in `use_cases/` is a **wrapper** around domain logic, not raw math
  - It properly delegates to `DomainPermutationAligner` from `core/domain/geometry/`
  - `geometry_engine.py` uses the `Backend` abstraction for computations (proper dependency injection)
- **Assessment**: The use_cases files are doing orchestration, not implementing raw math
- **No Migration Required**: Current architecture follows hexagonal principles correctly

### New Features Added (CABE Implementation)

#### Cross-Manifold Transfer (formerly "Ghost Anchor Synthesis")
- **Renamed for scientific accuracy** per user request to avoid "science fiction" terminology
- **New Files Created**:
  - `src/modelcypher/core/domain/geometry/manifold_transfer.py` - Landmark MDS-based concept transfer
  - `src/modelcypher/core/domain/geometry/geometric_lora.py` - LoRA generation from geometric specifications
  - `src/modelcypher/cli/commands/geometry/transfer.py` - CLI commands
  - `tests/test_manifold_transfer.py` - 15 tests
  - `tests/test_geometric_lora.py` - 22 tests
- **Academic Citations Added**:
  - de Silva & Tenenbaum (2004) - Sparse MDS using landmark points
  - Cox & Cox (2000) - Multidimensional Scaling
  - Hu et al. (2021) - LoRA: Low-Rank Adaptation
  - Eckart-Young theorem for rank truncation
- **Old Files Deleted**:
  - `src/modelcypher/core/domain/geometry/ghost_anchor_synthesis.py`
  - `src/modelcypher/core/domain/geometry/synthetic_lora.py`
  - `src/modelcypher/cli/commands/geometry/ghost.py`

### Test Results
- All 37 new tests for manifold transfer and geometric LoRA passing
- CLI `mc geometry transfer` commands registered and working

---

## 11. CLI-MCP Parity Investigation (December 23, 2025)

### Investigation Summary
The audit identified potential CLI-MCP parity gaps with "different default thresholds or naming conventions."

### Findings

#### Defaults Match ✅
Verified that CLI and MCP use identical defaults for the `transport merge` command:
| Parameter | CLI Default | MCP Default | Service Default |
|-----------|-------------|-------------|-----------------|
| coupling_threshold | 0.001 | 0.001 | 0.001 |
| blend_alpha | 0.5 | 0.5 | 0.5 |
| normalize_rows | True | True | True |

#### Naming Conventions Are Intentional ✅
The naming convention differences follow industry standards:
- **CLI**: kebab-case flags (`--normalize`) - POSIX/Unix convention
- **MCP**: camelCase parameters (`normalizeRows`) - JSON/JavaScript convention
- **Service/Domain**: snake_case (`normalize_rows`) - PEP8 Python convention

Both interfaces call the same service methods with equivalent parameters. No semantic differences exist.

#### Shared Configuration via Service Layer ✅
The recommendation for "shared configuration objects" is effectively achieved:
- Both CLI (`transport.py`) and MCP (`geometry.py`) call `GeometryTransportService`
- Default values are defined in service dataclasses (`MergeConfig`)
- CLI and MCP extract from service, not hardcoded

### Status: CLARIFIED (No Action Required)
The parity concern was based on naming convention differences, not actual behavioral differences. The architecture correctly separates interface conventions from shared service logic.

---

## 12. Issue 3.3 Resolution Progress (December 23, 2025)

### Problem Statement
35 files in `src/modelcypher/core/domain/` imported `mlx.core` directly, violating hexagonal architecture's dependency rule. This couples the domain layer to macOS/Apple Silicon and blocks future CUDA support.

### Solution Implemented
Extended the existing `Backend` protocol with 25 new methods and created infrastructure for domain file migration:

#### Phase 15A: Backend Protocol Extension ✅ COMPLETE
- **Extended Protocol**: `src/modelcypher/ports/backend.py` now has 45+ methods covering:
  - Array Creation: `eye`, `arange`, `diag`, `full`, `ones_like`, `zeros_like`, `linspace`
  - Shape Manipulation: `stack`, `concatenate`, `broadcast_to`
  - Reductions: `mean`, `min`, `argmax`, `argmin`, `var`, `std`
  - Element-wise: `sign`, `clip`, `where`, `softmax`, `cumsum`
  - Linear Algebra: `dot`, `norm`, `det`, `eigh`, `solve`, `qr`
  - Sorting: `sort`, `argsort`
  - Random: `random_normal`, `random_uniform`, `random_randint`, `random_seed`

- **MLXBackend Updated**: `src/modelcypher/backends/mlx_backend.py` implements all 45+ methods
- **CUDABackend Updated**: `src/modelcypher/backends/cuda_backend.py` implements all methods
- **NumpyBackend Updated**: `tests/conftest.py` NumpyBackend for testing

#### Phase 15B: Default Backend Manager ✅ COMPLETE
- **Created**: `src/modelcypher/core/domain/_backend.py`
- Provides `get_default_backend()` for lazy MLX initialization
- Provides `set_default_backend()` for testing with NumpyBackend

#### Phase 15C-D: Domain File Migration 🔄 IN PROGRESS (6/35 files)
**Files Migrated:**
| File | Status |
|------|--------|
| `semantics/vector_space.py` | ✅ Migrated |
| `geometry/types.py` | ✅ Migrated (unused import removed) |
| `geometry/intrinsic_dimension.py` | ✅ Migrated |
| `geometry/fingerprints.py` | ✅ Migrated |
| `entropy/logit_entropy_calculator.py` | ✅ Migrated |
| `geometry/generalized_procrustes.py` | ✅ Migrated |

**Files Remaining (29):**
- geometry: 15 files
- entropy: 5 files
- merging: 4 files
- training: 5 files
- inference: 2 files
- dynamics: 1 file
- thermo: 1 file

#### Phase 15E: Guard Tests ✅ COMPLETE
- **Created**: `tests/test_no_mlx_in_domain.py`
- Tracks migration progress
- Fails if migrated files regress to MLX imports
- Fails if new MLX imports are added without tracking

### Migration Pattern
All migrated files follow this pattern:
```python
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array, Backend

class SomeAnalyzer:
    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(self, data: Array) -> Array:
        return self._backend.sum(data)
```

### Status: 🔄 IN PROGRESS
- Infrastructure complete (protocol, backends, manager, tests)
- 6/35 files migrated (17%)
- Remaining files should follow the established pattern
- Guard test prevents regression
