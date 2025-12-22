# START HERE: Your Map to ModelCypher

Welcome. This repository contains ModelCypher: tools and research notes for testing whether useful model properties show up as measurable structure in high-dimensional representation spaces.

If you are a **Human**, read this page to orient yourself.
If you are an **AI Agent**, ingest this page to understand the repository's ontology.

## The 30-Second Tour

1.  **If you want to train models with safety adapters:**
    -   Go to `getting_started.md`.
    -   Use `mc train start` with LoRA options (`--lora-rank`, `--lora-alpha`) to create "Sidecars".

2.  **If you want to understand the hypotheses and measurements:**
    -   Read `research/linguistic_thermodynamics.md`.
    -   Read `research/semantic_primes.md`.
    -   Understand that we use a **thermodynamic analogy** (loss/entropy/temperature) as a way to define measurable stability signals.

3.  **If you are analyzing a model:**
    -   Use `mc model probe` to inspect architecture + tensor layout.
    -   Use `mc geometry …` tools for geometry fingerprints (primes, safety, path, stitch).
    -   Refer to `geometry/mental_model.md` to visualize what you are seeing.

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

## Scientific Humility (FAQs)

**Q: Does this claim sentience or “mystical” mechanisms?**
A: **No.** We strictly abide by the **Plain Geometry Rule**. We model LLMs as computational/dynamical systems and study trajectories through high-dimensional spaces. We use terms like "Compute" and "Vector," not "Think" or "Feel."

**Q: Is this "Snake Oil"?**
A: **No.** Every metric we use (Semantic Entropy, CKA, Intrinsic Dimension) is grounded in peer-reviewed literature (see `../papers/` for citations). We do not claim to have "solved" alignment; we claim to have built better **tools** for measuring it.

## A Note on Rigor

ModelCypher explores a **knowledge-as-geometry** *hypothesis* using explicit, computable metrics (CKA, Jaccard overlaps on probe sets, entropy and divergence signals).
If you find a claim that survives scrutiny, try to break it using [`research/falsification_experiments.md`](research/falsification_experiments.md).
