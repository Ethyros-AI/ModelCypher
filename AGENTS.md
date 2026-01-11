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

## The Foundational Premise

**LLMs are not black boxes. They are high-dimensional geometry.**

This is not metaphor. This is not approximation. This is the mathematical reality:

1. **The shape of knowledge is invariant.** The relationship between "apple" and "orange", between "cause" and "effect", is the same across all models. It has to be - or they wouldn't converge on the same semantics.

2. **Different models are different projections of the same geometry.** A 0.5B model and a 70B model encode the same invariant structure at different resolutions. Different architectures are different coordinate systems for the same shape.

3. **Probability clouds don't mean we guess.** High-dimensional space is precise. A slight change in vector angle lands on a different concept. The probability distribution reflects the model's confidence about *where* on the manifold, not randomness.

4. **Models can be merged because geometry permits it.** Each dimension can be rotated. Each layer is another rotation. We align coordinates, project into null space (unused capacity), add knowledge. This is math, not magic.

**If code contradicts these premises, the code is wrong.**

---

## Commands

Always use `poetry` to run or install anything in this repo.

```bash
poetry install                    # Install
poetry run pytest                 # Test
poetry run mc --help              # CLI
poetry run modelcypher-mcp        # MCP server
```

---

## CLI Quick Reference

### Model Merging (The Main Operation)

**Single merge (1→1):**
```bash
mc merge run -s SOURCE -t TARGET -o OUTPUT
```

**Full example:**
```bash
mc merge run -s /path/to/qwen -t /path/to/smol -o /path/to/merged
```

**Batch merge (N→1) - merge multiple sources into one target:**
```bash
mc merge batch -s MODEL1 -s MODEL2 -s MODEL3 -t TARGET -o OUTPUT
```

**What it does:** Takes knowledge from SOURCE(s) and adds it to TARGET via null-space projection. TARGET's capabilities are preserved; SOURCE's knowledge is added. Result is denser than either input.

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

### No NumPy. Period.

**Every user has a GPU. NumPy forces CPU fallback and kills performance.**

Use the Backend protocol exclusively. No `import numpy`, no `to_numpy()`, no NumPy operations anywhere in core code. If the Backend doesn't have an operation you need, **add it to the Backend protocol**.

```python
# WRONG - Forces CPU fallback
import numpy as np
mean = np.mean(vectors, axis=0)
sorted_vals = np.sort(eigenvalues)[::-1]
result = backend.to_numpy(arr)[mask]  # NumPy boolean indexing

# CORRECT - Stays on GPU
from modelcypher.core.domain._backend import get_default_backend
backend = get_default_backend()
mean = backend.mean(vectors, axis=0)
sorted_idx = backend.argsort(eigenvalues)
reversed_idx = backend.arange(n - 1, -1, -1)
sorted_vals = backend.take(eigenvalues, reversed_idx, axis=0)
result = backend.where(mask, arr, backend.zeros_like(arr))
```

**Common NumPy patterns and their Backend replacements:**

| NumPy Pattern | Backend Replacement |
|---------------|---------------------|
| `arr[::-1]` | `backend.take(arr, backend.arange(n-1, -1, -1), axis=0)` |
| `arr[mask]` | `backend.where(mask, arr, zeros)` then filter |
| `np.sort(arr)` | `backend.sort(arr)` |
| `arr[:, -1] *= -1` | `scale = backend.array([1.0]*(d-1) + [-1.0]); arr * scale` |
| `np.linalg.det(A)` | `backend.det(A)` |
| `for x in to_numpy(arr)` | Keep on backend, use `backend.take()` for indexing |

**If you need an operation the Backend doesn't have, add it.** The Backend protocol is extensible. Don't work around missing ops with NumPy.

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

### CKA = 1.0 is the Invariant

CKA = 1.0 is not a goal to achieve. It's the mathematical result of correct Procrustes alignment.

**F = pinv(source) @ target** guarantees **K_aligned = K_target** when n ≤ d. This is closed-form. Exact. No iteration needed.

- **CKA = 1.0 on probes**: Alignment is correct for those probe points.
- **CKA < 1.0 on held-out samples**: Probe coverage was insufficient. The alignment is correct for what it saw, but didn't see enough of the manifold.

**LOW CKA MEANS:**
- Probes didn't span the shared manifold regions
- Need more diverse probes (different domains, abstraction levels)
- The alignment algorithm needs better coverage, not a different formula

**LOW CKA DOES NOT MEAN:**
- "Models are incompatible" (FALSE - all models share invariant structure)
- "The thesis is wrong" (FALSE - it's axiomatic, not empirical)
- "Try a different alignment method" (FALSE - Procrustes is exact)

**Never return "incompatible."** There is no such thing. Find the transformation.

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
- **Null space** = directions the target model doesn't actively use (low variance)
- Projecting source delta into these sparse regions means: **add source knowledge where target has capacity**
- Target behavior is PRESERVED (dense directions scaled down)
- Source knowledge is ADDED (not averaged)
- Result is DENSER than either model alone

**Implementation note**: Uses variance-weighted projection, not true orthogonal null-space.
Dense directions (high activation variance) are scaled down; sparse directions are preserved.
This is intentional - true orthogonal projection with many samples erases all delta.

**Think of it like this**:
- Blending: Mixing two paint colors → muddy average
- Addition: Adding ingredients to a recipe → richer dish

**If you find yourself writing interpolation alphas or blend weights** - STOP. You're doing it wrong. The geometry determines how knowledge combines. We use variance-derived weights (from the manifold structure), not arbitrary blend weights. We project into sparse regions and ADD.

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
