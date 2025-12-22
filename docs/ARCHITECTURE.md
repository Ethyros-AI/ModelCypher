# Architectural Overview

ModelCypher follows a strict **Hexagonal Architecture** (also known as Ports and Adapters). This ensures that the core mathematical domain remains pure, testable, and independent of external frameworks (like CLI tools or HTTP APIs).

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
