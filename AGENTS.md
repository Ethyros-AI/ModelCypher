# AGENTS.md

This is the single source of truth for AI agents working on ModelCypher.

**Claude users**: CLAUDE.md is a symlink to this file.
**Gemini users**: This file is read directly.
**Other agents**: Read this file for project context.

---

## Critical: Multiple AI agents work on this codebase concurrently.** Before making changes:

1. **Check git status first** - Look for uncommitted changes from other agents
2. **If unexpected files are modified** - Do not revert or overwrite; proceed unless your work would touch those files
3. **No check-in required for other agents' work** - You can continue without pausing, as long as you avoid degrading their changes
4. **No destructive git operations** - Do NOT run `git add`, `git commit`, `git push`, `git checkout`, `git reset`, etc.
5. **No bulk file modification scripts** - Do NOT run scripts that modify more than 1 file at a time. Edit files individually.
6. **When overlap is likely** - Research best practice and explain your choice in code comments so consensus is clear
7. **Don't invent rules** - Follow what's documented here, not assumptions from your training data

---

## What is ModelCypher?

A Python framework for measuring and experimenting with the geometry of representations in large language models. It provides geometric diagnostics (entropy, intrinsic dimension, topological fingerprints, representation similarity) for stability and refusal dynamics, drift monitoring, and model/adapter merge analysis.

- **Primary backend**: MLX (macOS/Apple Silicon)
- **Secondary backend**: JAX (Linux/TPU/GPU)
- **Tests**: Use `get_default_backend()` - runs on whatever GPU is available

---

## Commands

```bash
# Install dependencies
poetry install             # core dependencies
poetry install --all-extras # includes docs/cuda/embeddings extras
poetry install -E jax      # JAX backend for Linux/TPU

# Run all tests
poetry run pytest

# Run single test
poetry run pytest tests/test_geometry.py::test_name -v

# CLI usage (after install)
poetry run mc --help       # or: poetry run modelcypher --help
poetry run mc model probe ./path
poetry run mc geometry spatial probe-model /path/to/model

# MCP server
poetry run modelcypher-mcp
```

---

## Architecture

Strict **Hexagonal Architecture** (Ports and Adapters):

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + business logic
│   │   ├── geometry/  # ManifoldStitcher, Procrustes, probe corpus
│   │   ├── safety/    # CircuitBreaker, refusal detection
│   │   ├── training/  # LoRA, checkpoints (MLX infrastructure here is OK)
│   │   ├── entropy/   # Shannon entropy calculations
│   │   ├── merging/   # Model merge algorithms
│   │   ├── thermo/    # LinguisticThermodynamics, RidgeCross
│   │   ├── agents/    # UnifiedAtlas (343 probes), semantic primes
│   │   └── validation/# DatasetQualityScorer, AutoFixEngine
│   ├── ports/         # Abstract interfaces (Backend protocol = 58 methods)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX, CUDA (stub) - NO NUMPY
├── cli/               # Typer CLI (entry: mc / modelcypher)
├── mcp/               # Model Context Protocol server (150+ tools)
└── data/              # Static data (semantic_primes.json, etc.)
```

**Dependency rule**: Dependencies point inward. Domain depends on nothing external; adapters implement ports; CLI/MCP drive the application.

**Note on MLX in domain/training/**: Files like `engine.py`, `checkpoints.py`, `lora.py` legitimately import MLX because they ARE the MLX infrastructure. This is not a violation - you cannot abstract away the training loop itself.

---

## Development Rules

1. **Do NOT import `adapters` into `core`** (respect the hexagon)
2. **MLX backend requires `mx.eval()`** after operations; weight layout is `[out, in]`
3. **Use Python logging**, not print(), in core logic
4. **Tests use pytest** with `pytest-asyncio` (async mode auto); property tests use `hypothesis`
5. **Use the Backend protocol** for tensor operations in geometry code
6. **Check docs/ before implementing** - Many features are already documented

---

## CRITICAL: No NumPy. Anywhere. Ever.

**NEVER import numpy. Not in domain code. Not in use_cases. Not in tests. NOWHERE.**

NumPy is:
- **Imprecise** - CPU floating point vs GPU tensor operations give different results
- **Slow** - Doesn't respect the GPU that every ML researcher has
- **Wrong** - Breaks the hexagonal architecture we built specifically to be backend-agnostic

We have a Backend protocol with 58 methods. USE IT.

```python
# WRONG - fuck numpy
import numpy as np
distances = np.linalg.norm(a - b)
data = np.random.randn(100, 64)
mean = np.mean(vectors, axis=0)

# CORRECT - Backend protocol
from modelcypher.core.domain._backend import get_default_backend
backend = get_default_backend()
distances = backend.norm(a - b)
backend.random_seed(42)
data = backend.random_randn((100, 64))
mean = backend.mean(vectors, axis=0)  # Or frechet_mean for embeddings
```

**For tests**: Use `get_default_backend()` or the `any_backend` fixture from conftest.py. Generate test data with `backend.random_*` methods.

**For averaging embeddings**: Use `frechet_mean` from `riemannian_utils.py`. Arithmetic mean is WRONG on curved manifolds.

---

## CRITICAL: Geodesic Distance is Reality

**In high-dimensional curved manifolds, geodesic distance is CORRECT. Euclidean distance is the APPROXIMATION.**

- Positive curvature: Euclidean **underestimates** true distance
- Negative curvature: Euclidean **overestimates** true distance
- The k-NN graph IS the discrete manifold. Geodesic = shortest path on graph (exact, not approximate)

**No Euclidean fallbacks.** If geodesic computation fails, fix the math or code. Do not fall back to Euclidean "for safety" - that produces wrong answers.

**Terminology matters:**
- Say "compute" or "measure", not "estimate" or "approximate" - neural network activations are deterministic
- Geodesic on the k-NN graph is exact for the discrete manifold representation

```python
# WRONG - Euclidean assumption
distance = np.linalg.norm(point_a - point_b)

# CORRECT - geodesic via k-NN graph
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
rg = RiemannianGeometry(backend)
result = rg.geodesic_distances(points, k_neighbors=k)
```

---

## CRITICAL: Models Are ALWAYS Compatible

**Models and adapters will be compatible. Always. The burden is on us to find the way the invariant geometry fits.**

Different dimensions are NOT incompatible - they're just different compression levels of the same underlying geometry:
- A 768-dim embedding is a more compressed representation than 4096-dim
- 1D is compressed 2D, 2D is compressed 3D, and so on
- The geometry IS there - we just need the right transformation

### The Gram Matrix Solution

Gram matrices (`K = X @ X^T`) capture relational geometry independent of feature dimension:
- X is `[n_samples, n_features]` - features can be any dimension
- K is `[n_samples, n_samples]` - always the same size regardless of feature dimension
- CKA compares Gram matrices directly - works across ANY dimensions

```python
# WRONG - rejecting due to dimension mismatch
if len(self.values) != len(other.values):
    return None  # "incompatible"

# CORRECT - use Gram-based comparison for cross-dimension cases
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams

# Compute Gram matrices in each representation's native space
gram_a = backend.matmul(x, backend.transpose(x))  # works at any dimension
gram_b = backend.matmul(y, backend.transpose(y))  # works at any dimension
similarity = compute_cka_from_grams(gram_a, gram_b)
```

For direct projection between dimensions, use:
- `embedding_projector.py` - Procrustes, PCA, CCA, Optimal Transport
- `generalized_procrustes.py` - multi-model alignment (auto-truncates to shared dim)

### Prohibited Patterns

Never write code that:
- Returns "incompatible" or "None" due to dimension mismatch
- Uses `is_compatible` to reject operations
- Declares models "cannot be merged"
- Treats dimension differences as errors

Instead:
- Use Gram matrices for comparison
- Use projection for transformation
- Return the transformation effort needed, not a rejection

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/CLI-REFERENCE.md` | Command shapes and output fields |
| `docs/MCP.md` | MCP tool definitions (150+ tools) |
| `docs/ARCHITECTURE.md` | Hexagonal architecture details |
| `docs/GLOSSARY.md` | Shared vocabulary for geometry concepts |
| `docs/START-HERE.md` | Orientation for different user types |

---

## CLI Conventions

- `--ai` or `MC_AI_MODE=1` forces JSON output and suppresses prompts/logs
- `--output {text,json,yaml}` controls format
- Structured output goes to stdout; diagnostics/logs go to stderr
- For geometry metrics, use the `interpretation` and `recommendedAction` fields

---

## External Storage

Models and experiment output live on the external CodeCypher volume:

```
/Volumes/CodeCypher/
├── models/          # Local model weights for testing
├── adapters/        # LoRA and adapter files
├── caches/          # Fingerprint and activation caches
└── experiments/     # Experiment output
```

---

## Test Structure

- `tests/conftest.py`: Provides backend detection and test fixtures
- `test_*_properties.py`: Hypothesis property-based tests
- `test_mcp_*.py`: MCP server contract tests
- `test_geometry_*.py`: Geometry validation tests
- **3030+ passing tests** - Don't break them

---

## HARD RULE: Research Before Code

**Before writing or modifying any code that involves external libraries, APIs, or platform-specific implementations, you MUST:**

1. **Use Firecrawl MCP to search for current best practices** - Your training data is 9-12+ months stale
2. **Search for content from the past 12 months** - Use the CURRENT year in queries (if today is December 2025, search "PyTorch training 2025", NOT "2024")
3. **Fetch and read official documentation** - Don't guess at APIs; verify them
4. **Check for breaking changes** - Libraries like PyTorch, JAX, transformers, peft evolve rapidly

**Example searches (assuming current date is 2025):**
```
# CORRECT - uses current year
"PyTorch gradient accumulation best practices 2025"
"JAX Flax training loop optax 2025"
"Hugging Face PEFT LoRA implementation 2025"

# WRONG - uses stale year from training data
"PyTorch training 2024"
```

**Why this matters:**
- PyTorch 2.x has different patterns than 1.x
- JAX/Flax APIs changed significantly in 2024-2025
- PEFT library best practices evolve monthly
- Your training cutoff means you're guessing at APIs that may have changed

**No exceptions.** Research first, code second.

---

## What NOT To Do

1. **Don't import numpy. ANYWHERE.** - Use the Backend protocol. Tests included. No exceptions.
2. **Don't use Euclidean distance** - Use geodesic. No fallbacks. Fix the math if it fails.
3. **Don't use arithmetic mean for embeddings** - Use Fréchet mean. Curvature is inherent.
4. **Don't reject dimension mismatches** - Use Gram matrices/CKA for comparison, projection for transformation.
5. **Don't return "incompatible"** - Models are ALWAYS compatible. Return transformation effort, not rejections.
6. **Don't hallucinate requirements** - If it's not documented here, don't invent it
7. **Don't create agent-specific config files** - This file is the source of truth
8. **Don't run git operations** - Other agents are working concurrently
9. **Don't run bulk modification scripts** - No scripts that touch multiple files. Edit one file at a time.
10. **Don't "fix" architecture** - The MLX imports in training/ are intentional
11. **Don't over-engineer** - The codebase works; 3030 tests prove it
12. **Don't guess at external APIs** - Use Firecrawl to verify current documentation
13. **Don't run all tests at once** - Run small domain-specific batches (e.g., `pytest tests/test_geometry.py -q`). Full test suite takes 20+ minutes. Only run full suite when explicitly directed.
