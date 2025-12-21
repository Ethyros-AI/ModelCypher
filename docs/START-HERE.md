# START HERE: Your Map to ModelCypher

Welcome. This repository is not just a collection of scripts; it is a framework for **Geometric AI**.

If you are a **Human**, read this page to orient yourself.
If you are an **AI Agent**, ingest this page to understand the repository's ontology.

## 🧭 The 30-Second Tour

1.  **If you want to train models safely:**
    -   Go to `docs/getting_started.md`.
    -   Use `mc-train` to create "Sidecars" (adapters that steer without lobbying).

2.  **If you want to understand *why* this works:**
    -   Read `docs/research/linguistic_thermodynamics.md`.
    -   Read `docs/research/semantic_primes.md`.
    -   Understand that we treat models as **Physical Systems** with Energy (Loss) and Entropy.

3.  **If you are analyzing a model:**
    -   Use `mc-inspect` to scan its "Geometric Fingerprint".
    -   Refer to `docs/geometry/mental_model.md` to visualize what you are seeing.

## 📚 Documentation Index

### The "Handshake" (Core Vocabulary)
-   [**GLOSSARY.md**](GLOSSARY.md) - **READ THIS FIRST**. Defines "Manifold", "Procrustes", "Refusal Vector".

### Theory (The "Why")
-   [**Linguistic Thermodynamics**](research/linguistic_thermodynamics.md) - The physics of training.
-   [**Semantic Primes**](research/semantic_primes.md) - The universal anchors of meaning.
-   [**Entropy Safety**](research/entropy_differential_safety.md) - The "Sidecar" architecture and $\Delta H$.
-   [**Mental Models**](geometry/mental_model.md) - Visual diagrams of the geometry.

### Future Frontiers (Experimental)
-   [**Cross-LoRA Transfer**](research/cross_lora_transfer.md) - Using geometry to port skills between models.
-   [**Manifold Swapping**](research/manifold_swapping.md) - Building "Frankenstein" models.

### Practice (The "How")
-   [**Getting Started**](getting_started.md)
-   [**CLI Reference**](CLI-REFERENCE.md)
-   [**AI Assistant Guide**](AI-ASSISTANT-GUIDE.md) - How to prompt *with* ModelCypher.

### Architecture (The "What")
-   [**Architecture Overview**](architecture.md) - Hexagonal architecture.
-   [**Security Policy**](security.md)

## 🏗 Repository Structure

```
ModelCypher/
├── src/modelcypher/          # The Source Code
│   ├── core/domain/          # Pure Math & Business Logic (The Brain)
│   ├── adapters/             # Hardware/OS Integrations (The Body)
│   ├── interfaces/           # CLI & Servers (The Voice)
├── docs/                     # The Library (You are here)
└── tests/                    # The Verification Protocols
```

## 🔬 Scientific Humility (FAQs)

**Q: Is this "Consciousness" or "Magic"?**
A: **No.** We strictly abide by the **Plain Geometry Rule**. We model LLMs as physical systems processing trajectories through a high-dimensional manifold. We use terms like "Compute" and "Vector," not "Think" or "Feel."

**Q: Is this "Snake Oil"?**
A: **No.** Every metric we use (Semantic Entropy, CKA, Intrinsic Dimension) is grounded in peer-reviewed literature (see `papers/` for citations). We do not claim to have "solved" alignment; we claim to have built better **tools** for measuring it.

## ⚠️ A Note on Rigor

This project assumes **Knowledge is Geometry**.
We do not use "vibes". We use rigorous mathematical metrics (CKA, Jaccard, Entropy).
If you find a claim that holds up to scrutiny, check `docs/research/falsification_experiments.md`.
