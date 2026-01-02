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

### Models Are Always Compatible

Different dimensions = different compression levels of the same geometry. Use Gram matrices for comparison, projection for transformation. Never return "incompatible."

### CKA = 1.0 or Wrong

If CKA < 1.0, debug the alignment code. Don't claim models are incompatible.

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
