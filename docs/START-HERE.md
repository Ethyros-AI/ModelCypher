# Start Here: ModelCypher in 5 Minutes

## Quick Install

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install
```

---

## Your First Measurement (60 seconds)

Download a small model and probe its geometry:

```bash
# Fetch a small model (or use any local model path you already have)
poetry run mc model fetch mlx-community/Qwen2.5-0.5B-Instruct-bf16

# Probe it (use the `localPath` from the previous command)
poetry run mc geometry spatial probe-model /path/from/localPath
```

**Output:**
```
============================================================
3D WORLD MODEL ANALYSIS: Qwen2.5-0.5B-Instruct-bf16
============================================================

Anchors Probed: 23/23
Layer Analyzed: 23

World Model Score: 0.40

----------------------------------------
Key Metrics:
  Gravity Correlation: 0.61
  Inverse-Square Compliance: 0.72
  Axis Orthogonality (mean): 94.58%
```

**What you just measured:**
- This output reports spatial geometry measurements for the probed model.
- This probe tests whether internal representations encode consistent 3D spatial relations from text-only prompts.
- Axis Orthogonality (mean) shows how close the axes are to orthogonal; higher is more orthogonal.

If you got different numbers, that's real data about your model. If the command failed, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues).

---

## What Is ModelCypher?

A toolkit for measuring the geometric structure of LLM representations.

| Without ModelCypher | With ModelCypher |
| :--- | :--- |
| "The merge feels off" | "Curvature deltas show where geometry shifted" |
| "It refuses too much" | "Refusal boundary movement quantified" |
| "The models are similar-ish" | "Structural alignment measured via Procrustes/CKA" |
| "Training seems stable" | "Entropy signals tracked per step (raw values)" |

**The idea:** ModelCypher treats internal activations and weights as representation spaces and measures their geometry (distances, curvature, alignment). The outputs are raw measurements you can compare across models or track over time.

---

## Three Pathways

### Path 1: ML Engineer
**Goal**: Merge models without breaking them.

```bash
# Predict interference before merging
poetry run mc geometry interference predict ./source-model ./target-model

# Merge with null-space knowledge addition
poetry run mc merge run -s ./source-model -t ./target-model -o ./merged
```

→ [CLI Reference](CLI-REFERENCE.md) · [Why Geometry Matters](WHY-GEOMETRY-MATTERS.md) · [Verification](VERIFICATION.md)

### Path 2: Researcher
**Goal**: Test hypotheses about representation geometry.

→ [Geometry Guide](GEOMETRY-GUIDE.md) · [Research Papers](../papers/README.md) · [Glossary](GLOSSARY.md)

### Path 3: Safety Auditor
**Goal**: Detect drift and enforce boundaries.

→ [Entropy Safety](research/entropy_differential_safety.md) · [AI Assistant Guide](AI-ASSISTANT-GUIDE.md)

---

## Documentation Index

### Core Vocabulary
- [**GLOSSARY.md**](GLOSSARY.md) — Defines "Manifold", "Procrustes", "Refusal Vector", etc.

### Theory
- [**Geometry Guide**](GEOMETRY-GUIDE.md) — How to interpret metrics
- [**ELIF: Conceptual Map**](ELIF.md) — One narrative, analogy-driven overview (technically precise)
- [**Mental Models**](geometry/mental_model.md) — Visual diagrams
- [**Linguistic Thermodynamics**](research/linguistic_thermodynamics.md) — Entropy and stability

### Evidence
- [**Verification**](VERIFICATION.md) — Empirical results (geometry vs naive merging)
- [**Why Geometry Matters**](WHY-GEOMETRY-MATTERS.md) — Before/after comparisons
- [**Spatial Grounding**](research/spatial_grounding.md) — 3D world models in text-only LLMs
- [**Moral Geometry**](research/moral_geometry.md) — Ethical reasoning structure
- [**Bibliography**](references/BIBLIOGRAPHY.md) — Research citations + local PDFs

### Practice
- [**CLI Reference**](CLI-REFERENCE.md) — All commands
- [**MCP Server**](MCP.md) — AI agent integration
- [**FAQ**](FAQ.md) — Common questions and skepticism

### For AI Assistants
- [**AI Assistant Guide**](AI-ASSISTANT-GUIDE.md) — How to use ModelCypher tools
- [**Skeptic's Guide**](SKEPTICS-GUIDE.md) — Why the math claims are true (code references)

---

## Documentation Map

```
START-HERE.md (you are here)
    │
    ├── For Intuition ────────────────────┐
    │   ├── GEOMETRY-GUIDE.md             │
    │   ├── WHY-GEOMETRY-MATTERS.md       │
    │   └── geometry/mental_model.md      │
    │                                     │
    ├── For Precision ───────────────────>│── GLOSSARY.md (reference)
    │   └── geometry/*.md (6 deep dives)  │
    │                                     │
    └── For Research ─────────────────────┤
        ├── papers/paper-0-the-shape-of-knowledge.md    │  ← Start here for theory
        ├── papers/paper-1-invariant-semantic-structure.md
        ├── papers/paper-2-entropy-safety-signal.md
        ├── papers/paper-3-cross-architecture-transfer.md
        ├── papers/paper-4-modelcypher-toolkit.md
        └── papers/paper-5-semantic-highway.md
                                          │
    All paths converge at:────────────────┘
        └── CLI-REFERENCE.md (how to measure)
```

### Reading Order

**For the Big Picture** (30 min):
1. [Paper 0: The Shape of Knowledge](../papers/paper-0-the-shape-of-knowledge.md) — Framework
2. [Paper 5: The Semantic Highway](../papers/paper-5-semantic-highway.md) — Key observation

**For Implementation** (1 hour):
3. [Paper 1: Invariant Semantic Structure](../papers/paper-1-invariant-semantic-structure.md) — CKA methodology
4. [Paper 3: Cross-Architecture Transfer](../papers/paper-3-cross-architecture-transfer.md) — Merge technique
5. [Paper 4: ModelCypher Toolkit](../papers/paper-4-modelcypher-toolkit.md) — CLI usage

**For Safety** (30 min):
6. [Paper 2: Entropy Safety Signal](../papers/paper-2-entropy-safety-signal.md) — ΔH monitoring

---

## Repository Structure

```
ModelCypher/
├── src/modelcypher/          # Source code
│   ├── core/domain/          # Pure math + business logic
│   ├── adapters/             # Concrete integrations (hf_hub, filesystem)
│   ├── cli/                  # CLI commands
│   └── mcp/                  # MCP server (tool registry)
├── docs/                     # Documentation (you are here)
│   ├── geometry/             # Deep-dive geometry docs
│   ├── research/             # Research methodology
│   └── references/arxiv/     # Reference PDFs
├── papers/                   # Research manuscripts (0-5)
└── tests/                    # Test suite
```

---

## Troubleshooting

**"Model not found"** → Use absolute path; check for `config.json` in model dir

**"Backend not available"** → Linux: `poetry install -E jax` · macOS: MLX auto-detected

**"Out of memory"** → Use quantized model (4-bit/8-bit)

---

## Methodological Stance

1. **Geometric Realism**: Representation space is an object of study with measurable properties
2. **Operational Definitions**: "Safety" and "Agency" are defined by trajectory properties, not metaphors
3. **Falsifiability**: Hypotheses can be empirically rejected (see [falsification experiments](research/falsification_experiments.md))

This toolkit provides engineering tools for measuring geometric properties. It does not claim to solve alignment or explain consciousness.
