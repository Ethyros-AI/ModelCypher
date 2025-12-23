# Orientation: Navigating ModelCypher

This repository contains **ModelCypher**, a comprehensive toolkit for geometric analysis of large language models. It serves two functions:
1.  **Software Library**: A `pip`-installable Python package for measuring representation geometry.
2.  **Research Archive**: Published manuscripts, experimental protocols, and reproducible data.

## Primary Workflows

### 1. Research & Methodology
If you are evaluating the theoretical basis or experimental results:
-   **[Papers](papers/README.md)**: Full manuscripts (Papers 1–4) with formal methodology.
-   **[Bibliography](docs/references/BIBLIOGRAPHY.md)**: Index of 37 foundational papers.
-   **[Falsification](docs/research/falsification_experiments.md)**: Explicit criteria for rejecting our hypotheses.

### 2. Engineering & Integration
If you are building tools or integrating geometric signals:
-   **[Getting Started](getting_started.md)**: Installation and basic usage.
-   **[CLI Reference](CLI-REFERENCE.md)**: Command-line interface documentation.
-   **[MCP Server](MCP.md)**: Integration with agentic IDEs.

### 3. Model Analysis
If you are inspecting a specific model:
-   **[Geometry Guide](GEOMETRY-GUIDE.md)**: Interpreting outputs (CKA, entropy, fingerprints).
-   **[Mental Models](geometry/mental_model.md)**: Visualizing high-dimensional operations.

## Documentation Index

### The "Handshake" (Core Vocabulary)
-   [**GLOSSARY.md**](GLOSSARY.md) - **READ THIS FIRST**. Defines "Manifold", "Procrustes", "Refusal Vector".

### Theory (The "Why")
-   [**Linguistic Thermodynamics**](research/linguistic_thermodynamics.md) - Thermodynamic analogy for training and inference stability.
-   [**Semantic Primes**](research/semantic_primes.md) - Candidate anchor inventory (proposed universal in NSM; invariance in LLMs is tested).
-   [**Entropy Safety**](research/entropy_differential_safety.md) - The "Sidecar" architecture and $\Delta H$.
-   [**Mental Models**](geometry/mental_model.md) - Visual diagrams of the geometry.

### Future Frontiers (Experimental)
-   [**Cross-LoRA Transfer**](research/cross_lora_transfer.md) - Using geometry to port skills between models.
-   [**Manifold Swapping**](research/manifold_swapping.md) - Composing models via stitching (highly experimental).

### Practice (The "How")
-   [**Getting Started**](getting_started.md)
-   [**CLI Reference**](CLI-REFERENCE.md)
-   [**AI Assistant Guide**](AI-ASSISTANT-GUIDE.md) - How to prompt *with* ModelCypher.

### Architecture (The "What")
-   [**Architecture Overview**](ARCHITECTURE.md) - Hexagonal architecture.
-   [**Security Policy**](security.md)

## Repository Structure

```
ModelCypher/
├── src/modelcypher/          # The Source Code
│   ├── core/domain/          # Core logic (domain + math)
│   ├── adapters/             # Hardware/OS integrations
│   ├── cli/                  # CLI entrypoint + commands
│   ├── mcp/                  # MCP server (agent integrations)
├── docs/                     # The Library (You are here)
└── tests/                    # The Verification Protocols
```

## Methodological Stance

1.  **Geometric Realism**: we treat representation space as an object of study, applying standard topological (Betti numbers) and metric (CKA) tools to characterize it.
2.  **Operational Definitions**: concepts like "Agency" or "Safety" are defined by measurable properties of trajectories, not anthropomorphic metaphors.
3.  **Falsifiability**: we prioritize hypotheses that can be empirically rejected (see `docs/research/falsification_experiments.md`).

This repository does not claim to solve alignment or explain consciousness; it provides engineering tools for measuring specific geometric properties of language models.

## A Note on Rigor

ModelCypher explores a **knowledge-as-geometry** *hypothesis* using explicit, computable metrics (CKA, Jaccard overlaps on probe sets, entropy and divergence signals).
If you find a claim that survives scrutiny, try to break it using [`research/falsification_experiments.md`](research/falsification_experiments.md).
