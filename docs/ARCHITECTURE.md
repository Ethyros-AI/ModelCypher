# Architectural Overview

ModelCypher follows a strict **Hexagonal Architecture** (also known as Ports and Adapters). This ensures that the core mathematical domain remains pure, testable, and independent of external frameworks (like CLI tools or HTTP APIs).

## Visual Overview

```mermaid
flowchart TB
    subgraph EXTERNAL["External Drivers"]
        CLI["CLI<br/>(mc / modelcypher)"]
        MCP["MCP Server<br/>(150+ tools)"]
    end

    subgraph ADAPTERS["Adapters Layer"]
        HF["hf_hub.py"]
        FS["filesystem_storage.py"]
        LT["local_training.py"]
        LI["local_inference.py"]
    end

    subgraph PORTS["Ports Layer"]
        BE["Backend Protocol<br/>(58 methods)"]
        TR["Training Port"]
        ST["Storage Port"]
        INF["Inference Port"]
    end

    subgraph DOMAIN["Core Domain"]
        GEO["geometry/"]
        SAFE["safety/"]
        TRAIN["training/"]
        ENT["entropy/"]
        THERM["thermo/"]
        MERGE["merging/"]
        AGT["agents/"]
    end

    subgraph BACKENDS["Backend Implementations"]
        MLX["MLXBackend<br/>(macOS)"]
        JAX["JAXBackend<br/>(TPU/GPU)"]
        CUDA["CUDABackend<br/>(NVIDIA)"]
    end

    CLI --> DOMAIN
    MCP --> DOMAIN
    ADAPTERS --> PORTS
    PORTS --> DOMAIN
    BACKENDS --> BE
```

## Layers

### 1. The Core Domain (`src/modelcypher/core/domain/`)
This is the heart of the application. It contains the "business logic" and mathematical models.
-   **No adapter imports**: domain code should not import `modelcypher.adapters` directly.
-   **Deterministic, testable logic**: algorithms + dataclasses + small numeric helpers.
-   **Examples**: `ManifoldStitcher`, `CircuitBreakerIntegration`, `IntersectionMap`.

### 2. Ports (`src/modelcypher/ports/`)
These define the *interfaces* (Abstract Base Classes) that the Domain needs to interact with the outside world.
-   **Interfaces only**.
-   **Examples**: `training`, `storage`, `inference`, `geometry` ports.

### 3. Adapters (`src/modelcypher/adapters/`)
Concrete implementations of the Ports. This is where we talk to the filesystem, Hugging Face Hub, or hardware.
-   **Examples**: `hf_hub.py`, `filesystem_storage.py`, `local_training.py`, `local_inference.py`.

### 4. Interfaces / Infrastructure (`src/modelcypher/cli/`, `src/modelcypher/mcp/`, `src/modelcypher/infrastructure/`)
The entry points that drive the application.
-   **CLI**: `src/modelcypher/cli/app.py` (invoked via `mc` / `modelcypher`).
-   **MCP**: `src/modelcypher/mcp/` (Model Context Protocol server).

## Dependency Rule
**Dependencies point INWARD.**
-   The **CLI** depends on the **Domain**.
-   The **Adapters** depend on the **Ports**.
-   The **Domain** depends on **NOTHING** (except shared types).

## Key Components

### Manifold Stitcher (`domain/geometry/manifold_stitcher.py`)
Responsible for aligning two disparate model manifolds using Procrustes analysis.

### Probe Corpus (`domain/geometry/probe_corpus.py`)
A standardized set of prompts used to elicit comparable activations from different models.

Semantic primes are a separate anchor inventory (see `research/semantic_primes.md` and `src/modelcypher/data/semantic_primes.json`).

### Circuit Breaker (`domain/safety/circuit_breaker_integration.py`)
Monitors the "Regime State" of a model and interrupts generation if it enters a "Refusal Basin" or "Unstable Trajectory".

### MCP Server (`mcp/server.py`, `mcp/tools/`)
The MCP server exposes domain functionality via the Model Context Protocol. Tools are organized into modular registration functions:

```
src/modelcypher/mcp/
├── server.py              # Core server, tool profiles, base tools
├── security.py            # Security config and confirmation manager
└── tools/
    ├── common.py          # ServiceContext (lazy-loaded services), helpers
    ├── geometry.py        # Geometry analysis tools (path, CRM, stitch, etc.)
    ├── safety_entropy.py  # Safety probes, entropy tracking
    ├── agent.py           # Agent trace import/analysis
    └── dataset.py         # Dataset format detection, chunking, templating
```

The `ServiceContext` class provides lazy-loaded access to all domain services, avoiding circular imports and reducing startup time.

## Domain Modules

The core domain is organized by concern:

| Domain | Description |
|--------|-------------|
| `geometry/` | Path detection, manifold analysis, CRM, topological fingerprints |
| `entropy/` | Entropy tracking, divergence calculation, model state classification |
| `safety/` | Adapter safety, dataset scanning, circuit breaker, capability guard |
| `agents/` | Trace analytics, action validation, LoRA expert routing |
| `training/` | Checkpoint management, preflight checks, resource guards |
| `validation/` | Dataset format detection, identity linting, file enumeration |
| `dataset/` | Chat templates, document chunking, streaming shuffle |
| `thermo/` | Linguistic thermodynamics, ridge detection, phase transitions |
| `adapters/` | LoRA merging, adapter blending, ensemble orchestration |
| `inference/` | Dual-path generation, entropy dynamics |

## Backend Protocol

The Backend protocol (58 methods) enables platform-agnostic geometry code. All tensor operations go through this abstraction, allowing the same algorithms to run on MLX, JAX, CUDA, or NumPy.

```mermaid
flowchart LR
    subgraph GEOMETRY["Geometry Domain Code"]
        GW["gromov_wasserstein.py"]
        CKA["cka.py"]
        PROC["generalized_procrustes.py"]
        MANI["manifold_stitcher.py"]
    end

    subgraph PROTOCOL["Backend Protocol"]
        ARRAY["Array Creation<br/>array, zeros, ones, eye"]
        SHAPE["Shape Ops<br/>reshape, transpose, stack"]
        LINALG["Linear Algebra<br/>matmul, svd, eigh, solve"]
        REDUCE["Reductions<br/>sum, mean, max, norm"]
    end

    subgraph IMPLS["Implementations"]
        MLX["MLXBackend"]
        JAX["JAXBackend"]
        CUDA["CUDABackend"]
    end

    GEOMETRY --> PROTOCOL
    MLX --> PROTOCOL
    JAX --> PROTOCOL
    CUDA --> PROTOCOL
```

See [BACKEND-COMPARISON.md](BACKEND-COMPARISON.md) for platform selection guidance.

### MLX Infrastructure Exceptions

The Backend protocol abstracts mathematical operations, but certain files require direct MLX access for infrastructure that cannot be abstracted. These are intentional exceptions tracked in `tests/test_no_mlx_in_domain.py`:

| File | Reason |
|------|--------|
| `training/lora.py` | `mlx.nn.Module` for neural network layers |
| `training/checkpoints.py` | `mx.save_safetensors`, `mx.load` for I/O |
| `training/engine.py` | Training loop orchestration |
| `inference/dual_path.py` | `mlx_lm` for model loading |
| `merging/lora_adapter_merger.py` | SafeTensors file I/O |

These represent infrastructure boundaries, not architecture violations. Run `pytest tests/test_no_mlx_in_domain.py -v` to verify current migration status.

## Data Flow: Model Probing

The `mc model probe` command follows this data flow:

```mermaid
sequenceDiagram
    participant CLI as mc model probe
    participant SVC as ModelProbeService
    participant PROBE as Backend Probe
    participant ATLAS as UnifiedAtlas
    participant GEOM as Geometry Modules

    CLI->>SVC: probe(model_path)
    SVC->>PROBE: load_model(path)
    PROBE-->>SVC: weights, tokenizer

    SVC->>ATLAS: all_probes()
    ATLAS-->>SVC: 343 AtlasProbe objects

    loop For each probe batch
        SVC->>PROBE: get_activations(texts)
        PROBE-->>SVC: layer_activations
    end

    SVC->>GEOM: compute_fingerprint()
    GEOM-->>SVC: GeometryFingerprint

    SVC->>GEOM: compute_intrinsic_dimension()
    GEOM-->>SVC: dimension_estimate

    SVC-->>CLI: ProbeResult
```

## Data Flow: Adapter Merge Pipeline

The geometric merge pipeline aligns adapter weights through multiple stages:

```mermaid
flowchart LR
    subgraph INPUT["Inputs"]
        BASE["Base Model"]
        ADP1["Adapter A"]
        ADP2["Adapter B"]
    end

    subgraph STAGE1["Stage 1: Probe"]
        FP["Fingerprint<br/>via 343 probes"]
        IM["Intersection<br/>Map"]
    end

    subgraph STAGE2["Stage 2: Permute"]
        PERM["Weight<br/>Permutation"]
    end

    subgraph STAGE3["Stage 3: Align"]
        ROT["Procrustes<br/>Rotation"]
        BLEND["Weighted<br/>Blend"]
    end

    subgraph STAGE4["Stage 4: Validate"]
        PPL["Perplexity<br/>Check"]
        DIAG["Geometric<br/>Diagnosis"]
    end

    subgraph OUTPUT["Output"]
        MERGED["Merged<br/>Adapter"]
    end

    BASE --> FP
    ADP1 --> FP
    ADP2 --> FP

    FP --> IM
    IM --> PERM
    PERM --> ROT
    ROT --> BLEND
    BLEND --> PPL
    PPL --> DIAG
    DIAG --> MERGED
```
