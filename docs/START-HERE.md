# Orientation: Navigating ModelCypher

This repository contains **ModelCypher**, a comprehensive toolkit for geometric analysis of large language models. It serves two functions:
1.  **Software Library**: A `pip`-installable Python package for measuring representation geometry.
2.  **Research Archive**: Published manuscripts, experimental protocols, and reproducible data.

## Primary Pathways

Select the path that matches your current objective:

### 🛠️ Path 1: The Casual ML Tinkerer
**Goal**: Combine models safely and efficiently without retraining.
-   [**Getting Started**](getting_started.md) - Quick install and your first merge.
-   [**Model Merge CLI**](CLI-REFERENCE.md#mc-model-merge) - Use `mc model merge` with geometric alignment.
-   [**Verification**](VERIFICATION.md) - See why geometric merging beats naive averaging.

### 🔬 Path 2: The ML Researcher
**Goal**: Test the Manifold Hypothesis and synthesize new knowledge.
-   [**Geometry Guide**](GEOMETRY-GUIDE.md) - How to measure curvature and GW distance.
-   [**Manifold Transfer**](research/manifold_swapping.md) - The math behind Zero-Shot synthesis.
-   [**Research Papers**](../papers/README.md) - Deep dive into the physics of the latent space.

### 🛡️ Path 3: The Safety Auditor
**Goal**: Detect adversarial drift and enforce behavioral boundaries.
-   [**Entropy Differential**](research/entropy_differential_safety.md) - Using $\Delta H$ to catch jailbreaks.
-   [**Safety Polytope**](GLOSSARY.md#safety-polytope) - Grounding safety in geometric constraints.
-   [**AI Assistant Guide**](AI-ASSISTANT-GUIDE.md) - How to explain safety signals to stakeholders.

---

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
