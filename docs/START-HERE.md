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
# Download (or use any local model you have)
huggingface-cli download mlx-community/Qwen2.5-0.5B-Instruct-bf16 --local-dir ./models/qwen-0.5b

# Probe it
mc geometry spatial probe-model ./models/qwen-0.5b --output text
```

**Output:**
```
============================================================
3D WORLD MODEL ANALYSIS: Qwen2.5-0.5B-Instruct-bf16
============================================================

Anchors Probed: 23/23
Layer Analyzed: last

Has 3D World Model: NO
World Model Score: 0.40
Physics Engine: DETECTED

----------------------------------------
Key Metrics:
  Euclidean Consistency: 0.47
  Gravity Correlation: 0.61
  Axis Orthogonality: 94.58%

============================================================
ALTERNATIVE GROUNDING - Physics encoded geometrically along non-visual axes.
============================================================
```

**What you just measured:**
- This 0.5B model knows physics (gravity, distance) but encodes it along *linguistic* axes, not visual ones
- We call this the "Blind Physicist" — it knows the math of 3D space without seeing it
- Axis Orthogonality of 94.58% means concepts are cleanly separated (good)

If you got different numbers, that's real data about your model. If the command failed, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues).

---

## What Is ModelCypher?

A toolkit for measuring the geometric structure of LLM representations.

| Without ModelCypher | With ModelCypher |
| :--- | :--- |
| "The merge feels off" | "Layer 12 has 3x higher curvature than baseline" |
| "It refuses too much" | "Refusal boundary expanded 40% after fine-tuning" |
| "The models are similar-ish" | "94.2% structural alignment via Procrustes analysis" |
| "Training seems stable" | "Entropy gradient: -0.003/step (healthy convergence)" |

**The insight:** Inside every language model is a high-dimensional space where concepts live as points. That space has *shape*—curves, boundaries, distances. That shape *is* the model's knowledge. ModelCypher gives you a ruler and a map.

---

## Three Pathways

### Path 1: ML Engineer
**Goal**: Merge models without breaking them.

```bash
# Predict interference before merging
mc geometry interference predict --source ./model-A --target ./model-B

# Merge with geometric alignment
mc model merge --source ./model-A --target ./model-B --method procrustes
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
- [**Mental Models**](geometry/mental_model.md) — Visual diagrams
- [**Linguistic Thermodynamics**](research/linguistic_thermodynamics.md) — Entropy and stability

### Evidence
- [**Verification**](VERIFICATION.md) — Empirical results (geometry vs naive merging)
- [**Why Geometry Matters**](WHY-GEOMETRY-MATTERS.md) — Before/after comparisons
- [**Spatial Grounding**](research/spatial_grounding.md) — 3D world models in text-only LLMs
- [**Moral Geometry**](research/moral_geometry.md) — Ethical reasoning structure

### Practice
- [**CLI Reference**](CLI-REFERENCE.md) — All commands
- [**MCP Server**](MCP.md) — AI agent integration
- [**FAQ**](FAQ.md) — Common questions and skepticism

---

## Repository Structure

```
ModelCypher/
├── src/modelcypher/          # Source code
│   ├── core/domain/          # Pure math + business logic
│   ├── adapters/             # Hardware integrations
│   ├── cli/                  # CLI commands
│   └── mcp/                  # MCP server (148 tools)
├── docs/                     # Documentation (you are here)
├── papers/                   # Research manuscripts
└── tests/                    # 2400+ tests
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
