# Architectural Overview

ModelCypher follows a strict **Hexagonal Architecture** (also known as Ports and Adapters). This ensures that the core mathematical domain remains pure, testable, and independent of external frameworks (like CLI tools or HTTP APIs).

## Layers

### 1. The Core Domain (`src/modelcypher/core/domain/`)
This is the heart of the application. It contains the "business logic" and mathematical models.
-   **No external dependencies** (except `mlx` and standard lib).
-   **Pure Python**.
-   **Examples**: `ManifoldStitcher`, `CircuitBreaker`, `IntersectionMap`.

### 2. Ports (`src/modelcypher/ports/`)
These define the *interfaces* (Abstract Base Classes) that the Domain needs to interact with the outside world.
-   **Interfaces only**.
-   **Examples**: `ModelLoaderPort`, `DatasetRepositoryPort`.

### 3. Adapters (`src/modelcypher/adapters/`)
Concrete implementations of the Ports. This is where we talk to the filesystem, Hugging Face Hub, or hardware.
-   **Examples**: `HFHubAdapter` (implements `ModelLoaderPort`), `LocalFileSystemAdapter`.

### 4. Interfaces / Infrastructure (`src/modelcypher/interfaces/`, `src/modelcypher/infrastructure/`)
The entry points that drive the application.
-   **CLI**: `src/modelcypher/interfaces/cli/` (e.g., `mc-train`).
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
A collection of "Semantic Primes" used to elicit comparable activations from different models.

### Circuit Breaker (`domain/safety/circuit_breaker_integration.py`)
Monitors the "Regime State" of a model and interrupts generation if it enters a "Refusal Basin" or "Unstable Trajectory".
