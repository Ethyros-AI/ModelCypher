# AI-Assisted Development Guide

This document provides guidance for AI coding assistants (Claude, Gemini, Copilot, etc.) and human contributors working on ModelCypher.

**Note**: CLAUDE.md is a symlink to this file.

---

## Concurrency Rules

Multiple AI agents work on this codebase concurrently. Before making changes:
If other files are modified and you do not need to touch them, continue without pausing; only stop when overlap is required.

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
│   │   ├── agents/    # UnifiedAtlas (439 probes), semantic primes
│   │   └── validation/# DatasetQualityScorer, AutoFixEngine
│   ├── ports/         # Abstract interfaces (Backend protocol = 58 methods)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX, CUDA (stub) - no numpy in core math
├── cli/               # Typer CLI (entry: mc / modelcypher)
├── mcp/               # Model Context Protocol server (148 tools)
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

## CRITICAL: No NumPy in core math

**Do NOT import numpy in core/domain geometry or use_cases.** NumPy is permitted only at:
- I/O boundaries (e.g., `npz` export)
- Backend interop (dtype mapping, `to_numpy`)
- Tests that explicitly require it

NumPy is:
- **Imprecise** - CPU floating point vs GPU tensor operations give different results
- **Slow** - Doesn't respect the GPU that every ML researcher has
- **Wrong** - Breaks the hexagonal architecture we built specifically to be backend-agnostic

We have a Backend protocol with 58 methods. USE IT.

```python
# WRONG - don't use numpy
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
| `docs/MCP.md` | MCP tool definitions (148 tools) |
| `docs/ARCHITECTURE.md` | Hexagonal architecture details |
| `docs/GLOSSARY.md` | Shared vocabulary for geometry concepts |
| `docs/START-HERE.md` | Orientation for different user types |

---

## CLI Conventions

- `--ai` or `MC_AI_MODE=1` forces JSON output and suppresses prompts/logs
- `--output {text,json,yaml}` controls format
- Structured output goes to stdout; diagnostics/logs go to stderr
- Return raw measurements; avoid interpretation strings (see "No Vibes" section below)

---

## Test Structure

- `tests/conftest.py`: Provides backend detection and test fixtures
- `test_*_properties.py`: Hypothesis property-based tests
- `test_mcp_*.py`: MCP server contract tests
- `test_geometry_*.py`: Geometry validation tests
- **3060 passing tests** - Don't break them

---

## HARD RULE: Research Before Code

**Before writing or modifying any code that involves external libraries, APIs, or platform-specific implementations, you MUST:**

1. **Search for current best practices** - AI training data may be 9-12+ months stale
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

## CRITICAL: CLI/MCP-First Principle

**All operations MUST use the CLI (`mc`) or MCP tools. Never write custom Python scripts for one-off tasks.**

We are simultaneously building the tools we use AND making them available to everyone else. If we write custom scripts to accomplish something, so will every other user of this repository. That's unacceptable duplication of effort.

### The Rule

1. **Use existing tools first** - Check `mc --help` and `docs/MCP.md` before writing any code
2. **If a capability doesn't exist, build it** - Add a new CLI command or MCP tool, not a script
3. **No throwaway scripts** - Every capability should be reusable via CLI/MCP

### Examples

```bash
# WRONG - custom script for baseline extraction
python scripts/extract_baselines.py /path/to/model --domain spatial

# CORRECT - CLI command that anyone can use
poetry run mc geometry baseline extract /path/to/model --domain spatial

# WRONG - custom script for model validation
python validate_geometry.py /path/to/model

# CORRECT - CLI command with structured output
poetry run mc geometry validate /path/to/model --output json
```

### Why This Matters

- **Reproducibility**: CLI commands are documented and versioned
- **Discoverability**: Users find capabilities via `--help`, not by reading scripts
- **Testing**: CLI/MCP tools have tests; scripts often don't
- **Consistency**: Same interface for humans and AI agents (MCP)

### When Adding New Capabilities

1. Check if the capability exists in CLI (`mc --help`) or MCP (`docs/MCP.md`)
2. If not, implement it as:
   - CLI command in `src/modelcypher/cli/commands/`
   - MCP tool in `src/modelcypher/mcp/tools/`
3. Add documentation to `docs/CLI-REFERENCE.md` or `docs/MCP.md`
4. Add tests

**Never write a script that you wouldn't want to maintain forever.**

---

## CRITICAL: No Vibes - Let Geometry Speak

**Return raw measurements. Never insert human judgment for the judgment of the geometry itself.**

ModelCypher measures geometric properties of neural network representations. The geometry IS the truth - our job is to measure it accurately, not to interpret what it "means" or whether it's "good."

### Prohibited Patterns

```python
# WRONG - hardcoded thresholds with no geometric basis
if similarity > 0.8:
    return "excellent"
elif similarity > 0.5:
    return "good"
else:
    return "poor"

# WRONG - interpretation strings in output
return {
    "value": 0.73,
    "interpretation": "Good alignment detected",
    "recommendation": "Consider proceeding with merge"
}

# WRONG - qualitative labels
status = "healthy" if entropy < 2.0 else "concerning"
```

### Correct Patterns

```python
# CORRECT - raw measurement only
return {"similarity": 0.73}

# CORRECT - baseline-relative comparison (if threshold needed)
baseline = load_baseline(model_family)
z_score = (measured - baseline.mean) / baseline.std
return {"similarity": 0.73, "z_score": z_score, "baseline_mean": baseline.mean}

# CORRECT - percentile within distribution
percentile = compute_percentile(measured, reference_distribution)
return {"entropy": 1.8, "percentile": percentile}
```

### Why This Matters

1. **Thresholds are model-specific** - What's "good" for GPT-2 is different from LLaMA
2. **Baselines change** - A 2024 model has different geometry than a 2025 model
3. **Context matters** - 0.8 CKA might be great for cross-architecture, poor for same-architecture
4. **We're not the user** - Researchers know their domain; we provide measurements, they decide meaning

### When Thresholds Are Unavoidable

Some operations (alerts, circuit breakers) genuinely need thresholds. When they do:

1. **Derive from baselines** - Use measured distributions, not magic numbers
2. **Make configurable** - Let users override defaults
3. **Document the source** - Explain where the threshold came from
4. **Express in relative terms** - "3σ from baseline" not "entropy > 2.0"

```python
# ACCEPTABLE - threshold derived from baseline with clear provenance
class SafetyMonitor:
    def __init__(self, baseline: BaselineProfile, sigma_threshold: float = 3.0):
        self.alert_threshold = baseline.mean + sigma_threshold * baseline.std

    def check(self, value: float) -> dict:
        sigma_distance = (value - self.baseline.mean) / self.baseline.std
        return {
            "value": value,
            "sigma_from_baseline": sigma_distance,
            "exceeds_threshold": sigma_distance > self.sigma_threshold
        }
```

### The Baseline Philosophy

Every measurement should be interpretable relative to:
- **Same model's baseline** - How does this compare to typical behavior?
- **Model family baseline** - How does this compare to other LLaMA models?
- **Cross-family reference** - How does this compare to all models?

This is how scientists work. We don't say "2.3 is good" - we say "2.3 is 1.5σ above the mean for this architecture."

---

## What NOT To Do

1. **Don't import numpy. ANYWHERE.** - Use the Backend protocol. Tests included. No exceptions.
2. **Don't use Euclidean distance** - Use geodesic. No fallbacks. Fix the math if it fails.
3. **Don't use arithmetic mean for embeddings** - Use Fréchet mean. Curvature is inherent.
4. **Don't reject dimension mismatches** - Use Gram matrices/CKA for comparison, projection for transformation.
5. **Don't return "incompatible"** - Models are ALWAYS compatible. Return transformation effort, not rejections.
6. **Don't hallucinate requirements** - If it's not documented here, don't invent it
7. **Don't "fix" architecture** - The MLX imports in training/ are intentional
8. **Don't over-engineer** - The codebase works; 3060 tests prove it
9. **Don't guess at external APIs** - Research current documentation before implementing
10. **Don't run full test suite casually** - Run domain-specific batches (e.g., `pytest tests/test_geometry.py -q`). Full suite takes 20+ minutes.
11. **Don't write custom scripts** - Use CLI (`mc`) or MCP tools. If a capability doesn't exist, build it into CLI/MCP.
12. **Don't add vibes** - No hardcoded thresholds, interpretation strings, or qualitative labels. Return raw measurements only.
