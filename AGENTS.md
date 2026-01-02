# AI-Assisted Development Guide

Guidance for AI coding assistants working on ModelCypher.

**Note**: CLAUDE.md is a symlink to this file.

---

## What is ModelCypher?

Geometric diagnostics for LLM representations. Measures intrinsic dimension, curvature, entropy, and similarity to guide model merging, monitor training, and detect behavioral drift.

- **Backend**: MLX (macOS) primary, JAX (Linux/TPU) secondary
- **Architecture**: Hexagonal (ports and adapters)
- **Tests**: ~5800 passing tests

---

## Commands

```bash
poetry install                    # Install
poetry run pytest                 # Test
poetry run mc --help              # CLI
poetry run modelcypher-mcp        # MCP server
```

---

## CLI Quick Reference

### Model Merging (The Main Operation)

There is exactly ONE command to merge models:

```bash
mc merge -s SOURCE -t TARGET -o OUTPUT -d DOMAINS
```

**Full example:**
```bash
mc merge \
  --source /path/to/qwen \
  --target /path/to/smol \
  --output-dir /path/to/merged \
  --transplant-domains mathematical,logical,spatial
```

**What it does:** Takes knowledge from SOURCE and adds it to TARGET via null-space projection. TARGET's capabilities are preserved; SOURCE's knowledge is added. Result is denser than either input.

**Available domains:** `mathematical`, `logical`, `spatial`, `temporal`, `social`, `computational`

### Other Common Commands

```bash
# Inference
mc infer run --model /path/to/model --prompt "Hello"

# System info
mc system status --output json

# Model info
mc model probe /path/to/model --output json
```

---

## Architecture

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + logic (geometry, safety, merging, thermo)
│   ├── ports/         # Abstract interfaces (Backend protocol)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX implementations
├── cli/               # Typer CLI
└── mcp/               # MCP server
```

Dependencies point inward. Domain imports nothing external.

---

## Concurrency Rules

Multiple AI agents work concurrently. Don't pause for unrelated changes.

1. Ignore modified files you don't need to touch
2. No destructive git operations (`add`, `commit`, `push`, `reset`)
3. No bulk modification scripts—edit files individually

---

## Core Principles

### No NumPy in Core

Use the Backend protocol (79 methods). NumPy only at I/O boundaries.

```python
# Wrong
import numpy as np
mean = np.mean(vectors, axis=0)

# Correct
from modelcypher.core.domain._backend import get_default_backend
backend = get_default_backend()
mean = backend.mean(vectors, axis=0)
```

### Geodesic is Correct

Euclidean distance is the approximation. Geodesic on k-NN graph is exact.

```python
# Wrong
distance = np.linalg.norm(a - b)

# Correct
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
rg = RiemannianGeometry(backend)
distances = rg.geodesic_distances(points, k_neighbors=k)
```

### All Models Encode the Same Shape

**This is not a metaphor. This is literal.**

All neural networks trained on language converge to the same high-dimensional geometric structure - the invariant shape of knowledge itself. Different architectures (SmolLM, Qwen, Llama, Mistral) are different compressions or projections of this universal shape.

Think of it like high-dimensional Legos: they always fit together exactly one way. The geometry determines the fit, not the "architecture name."

- **Different dimensions** = different compression levels of the same geometry
- **Different layer counts** = different resolutions of the same manifold
- **Different architectures** = different coordinate systems for the same shape

Use Gram matrices for comparison (dimension-agnostic). Use projection for transformation. **Never return "incompatible."** There is no such thing as incompatible models - only alignment code that hasn't found the correct transformation yet.

### CKA = 1.0 or Debug

CKA is a **measurement**, not a goal. It tells us whether our alignment code has found the correct transformation between two models' coordinate systems.

- **CKA = 1.0**: Alignment is correct. Proceed with merge.
- **CKA < 1.0**: Alignment code is broken. Debug it. Fix it. Do NOT merge until CKA = 1.0.

**Never conclude "these models are incompatible" from low CKA.** Low CKA means our alignment algorithm failed, not that the models can't be merged. All models can be merged. Find the right transformation.

### Geometric Addition, NOT Blending

**This is the most important principle. Read it until you understand it.**

LLMs are NOT piles of probabilities to weight. They are HIGH-DIMENSIONAL GEOMETRIC STRUCTURES - like Legos that snap together.

**The goal**: Combine unique knowledge from hundreds of models into ONE denser, smarter model.

**WRONG - Interpolation/Blending**:
```python
# This is WRONG. Never do this.
merged = alpha * model_A + (1 - alpha) * model_B  # Blending
merged = 0.5 * source + 0.5 * target              # Weighted average
merged = lerp(source, target, t)                   # Interpolation
```

Why it's wrong: Interpolation AVERAGES information. You get a smeared, degraded model that's worse than either input. You're not adding knowledge - you're diluting it.

**CORRECT - Null Space Addition**:
```python
# This is CORRECT. Knowledge addition.
delta = source_weights - target_weights
projected = null_space_projection(delta, target_activations)
merged = target_weights + projected
```

Why it works:
- **Null space** = directions the target model doesn't actively use
- Projecting source delta into null space means: **add source knowledge where target has nothing**
- Target behavior is PRESERVED (no interference)
- Source knowledge is ADDED (not averaged)
- Result is DENSER than either model alone

**Think of it like this**:
- Blending: Mixing two paint colors → muddy average
- Addition: Adding ingredients to a recipe → richer dish

**If you find yourself writing weights, alphas, or interpolation** - STOP. You're doing it wrong. The geometry determines how knowledge combines. We don't "weight" anything. We project into null space and ADD.

### No Vibes

Return raw measurements. No hardcoded thresholds, interpretation strings, or qualitative labels.

```python
# Wrong
return {"similarity": 0.73, "interpretation": "Good alignment"}

# Correct
return {"similarity": 0.73}
```

When thresholds are needed, derive from baselines (z-scores, percentiles).

### Don't Invent Heuristics

If you need a parameter value, derive it from data or machine epsilon. Don't fabricate "standard heuristics."

---

## Research Before Code

AI training data is stale. Before using external APIs:
1. Search for current best practices (use current year in queries)
2. Fetch and read official documentation
3. Check for breaking changes

---

## CLI/MCP First

Never write custom scripts. Use `mc` CLI or MCP tools. If capability doesn't exist, add it to CLI/MCP.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/CLI-REFERENCE.md` | Command reference |
| `docs/MCP.md` | MCP tool catalog |
| `docs/GEOMETRY-GUIDE.md` | Metric explanations |
| `docs/GLOSSARY.md` | Terminology |
