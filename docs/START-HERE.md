# Start Here: ModelCypher in 5 Minutes

## Reality Check: Measurement-First, Known Algorithms

The terminology can sound speculative because high-dimensional geometry is not intuitive. The implementation is not speculative: ModelCypher uses standard geometry and linear algebra and returns raw measurements, not narrative claims.

What this repo uses (examples, not promises):
- Procrustes alignment + CKA on probe sets to compare representations.
- k-NN graph geodesics, curvature summaries, and density estimates in activation space.
- SVD/eigendecomposition, rank/condition checks, and null-space projection for merge safety.
- Entropy and differential signals computed directly from logits.

If you want receipts, start with [Geometry Guide](GEOMETRY-GUIDE.md) and [Verification](VERIFICATION.md).

## Current Evidence State (2026-03-11)

- `mc train run` is the canonical shipped training surface.
- The repo has not yet closed the promotable claim that geometric training is
  better than standard practice.
- Retained 350M pipeline validation still shows structural pass without full
  inference closure: `5/5` structural, `3/5` inference.
- Merge, continual learning, and stacking remain experimental or partial; the
  active closure order is in [RESEARCH-ROADMAP.md](RESEARCH-ROADMAP.md).

## Quick Install

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install
```

### Prerequisites

| Platform | Requirements |
|----------|--------------|
| macOS (Apple Silicon) | Apple Silicon (M1/M2/M3/M4), 16GB+ RAM, macOS 14.0+, Python 3.11+ |
| Linux (NVIDIA GPU) | NVIDIA GPU, Python 3.11+ |
| Linux/Cloud (TPU/GPU) | TPU or GPU, Python 3.11+ |

### Backend Selection

| Platform | Default Backend | Install Command |
|----------|-----------------|-----------------|
| macOS Apple Silicon | macOS backend | `poetry install` |
| Linux + NVIDIA GPU | NVIDIA backend | `poetry install` |
| Linux + TPU | TPU backend | `poetry install` |

For accelerator backends, enable the optional extras listed in `pyproject.toml`.

Set explicitly: `MC_BACKEND=<backend-key> poetry run mc ...` (see `mc system probe backends`).

---

## Your First Measurement (60 seconds)

Download a small model and probe its geometry:

```bash
# Add a small model (or use any local model path you already have)
poetry run mc model add <org>/<model-id>

# Inspect it (use the `localPath` from the previous command)
poetry run mc model info /path/from/localPath
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

Current repo accounting is intentionally narrow:

- `mc train run` is the canonical shipped surface
- merge, continual learning, and stacking remain experimental or partial
- the closure order lives in [RESEARCH-ROADMAP.md](RESEARCH-ROADMAP.md)

---

## Three Pathways

### Path 1: Train a Model
**Goal**: Run the canonical geometry-derived training path.

This is the shipped training surface. It is not yet a closed head-to-head claim
against standard LoRA recipes; that benchmark remains an active roadmap item.

```bash
# Train — all 15 hyperparameters derived from the weight matrices
poetry run mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter

# Validate with repeated trials
poetry run mc train validate-derived --model /path/to/model --data /path/to/data.jsonl --trials 5
```

→ [Training Guide](TRAINING-GUIDE.md) · [CLI Reference](CLI-REFERENCE.md) · [Mission](MISSION.md)

### Path 2: Analyze a Model
**Goal**: Measure representation geometry and test hypotheses.

```bash
# Intrinsic dimension profile
poetry run mc analyze dimension-profile --model /path/to/model --samples 50

# LoRA adapter spectral analysis
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model
```

→ [Geometry Guide](GEOMETRY-GUIDE.md) · [Research Papers](../papers/README.md) · [Glossary](GLOSSARY.md)

### Path 3: Merge Models (Experimental)
**Goal**: Explore cross-model transfer via the experimental null-space merge stack.

```bash
poetry run mc merge run -s ./source-model -t ./target-model -o ./merged
```

This workflow is useful, but it is not yet counted as canonical mission
closure. The active closure order is in [RESEARCH-ROADMAP.md](RESEARCH-ROADMAP.md).

→ [CLI Reference](CLI-REFERENCE.md) · [Geometry Guide](GEOMETRY-GUIDE.md) · [Verification](VERIFICATION.md)

---

## Documentation Index

### Core Vocabulary
- [**GLOSSARY.md**](GLOSSARY.md) — Defines "Manifold", "Procrustes", "Refusal Vector", etc.

### Theory
- [**Geometry Guide**](GEOMETRY-GUIDE.md) — How to interpret metrics
- [**Mental Models**](research/mental_model.md) — Visual diagrams
- [**Linguistic Thermodynamics**](research/linguistic_thermodynamics.md) — Entropy and stability

### Evidence
- [**Verification**](VERIFICATION.md) — Empirical results (geometry vs naive merging)
- [**Geometry Guide**](GEOMETRY-GUIDE.md) — Why geometry matters + before/after comparisons
- [**Atlas-Based Geometry**](research/ATLAS-BASED-GEOMETRY.md) — Domain probes (spatial, moral, social, temporal, semantic primes)
- [**Bibliography**](references/BIBLIOGRAPHY.md) — Research citations + local PDFs

### Practice
- [**CLI Reference**](CLI-REFERENCE.md) — All commands

### For AI Assistants
- [**AGENTS.md**](../AGENTS.md) — AI coding guide and project philosophy

---

## Documentation Map

```
START-HERE.md (you are here)
    │
    ├── For Intuition ────────────────────┐
    │   ├── GEOMETRY-GUIDE.md             │
    │   └── research/mental_model.md      │
    │                                     │
    ├── For Precision ───────────────────>│── GLOSSARY.md (reference)
    │   └── research/*.md (deep dives)    │
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
│   ├── backends/             # ML framework implementations
│   ├── cli/                  # CLI commands
│   └── experimental/         # Research surfaces not yet canonical
├── docs/                     # Documentation (you are here)
│   ├── research/             # Research methodology
│   └── references/arxiv/     # Reference PDFs
├── papers/                   # Research manuscripts (0-5)
└── tests/                    # Test suite
```

---

## Troubleshooting

**"Model not found"** → Use absolute path; check for `config.json` in model dir

**"Backend not available"** → Install the platform-appropriate backend extra from `pyproject.toml`, then re-run `mc system probe backends`.

**"Out of memory"** → Use quantized model (4-bit/8-bit)

---

## Methodological Stance [CONJECTURAL]

1. **Geometric Realism**: Representation space is an object of study with measurable properties
2. **Operational Definitions**: "Safety" and "Agency" are defined by trajectory properties, not metaphors
3. **Falsifiability**: Hypotheses can be empirically rejected

This toolkit provides engineering tools for measuring geometric properties. It does not claim to solve alignment or explain consciousness.
