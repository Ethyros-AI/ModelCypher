# AGENTS.md

This is the single source of truth for AI agents working on ModelCypher.

**Claude users**: CLAUDE.md is a symlink to this file.
**Gemini users**: This file is read directly.
**Other agents**: Read this file for project context.

---

## Critical: Multiple AI agents work on this codebase concurrently.** Before making changes:

1. **Check git status first** - Look for uncommitted changes from other agents
2. **If unexpected files are modified** - STOP and ask the user before proceeding
3. **No destructive git operations** - Do NOT run `git add`, `git commit`, `git push`, `git checkout`, `git reset`, etc.
4. **When overlap is likely** - Research best practice and explain your choice in code comments so consensus is clear
5. **Don't invent rules** - Follow what's documented here, not assumptions from your training data

---

## What is ModelCypher?

A Python framework for measuring and experimenting with the geometry of representations in large language models. It provides geometric diagnostics (entropy, intrinsic dimension, topological fingerprints, representation similarity) for stability and refusal dynamics, drift monitoring, and model/adapter merge analysis.

- **Primary backend**: MLX (macOS/Apple Silicon)
- **Secondary backend**: JAX (Linux/TPU/GPU)
- **Test backend**: NumPy (platform-agnostic)

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
│   ├── ports/         # Abstract interfaces (Backend protocol = 118 methods)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX, CUDA (stub), NumPy (test)
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
└── TrainingCypher/  # Training experiment output
```

---

## Test Structure

- `tests/conftest.py`: Provides `NumpyBackend` fixture for deterministic testing
- `test_*_properties.py`: Hypothesis property-based tests
- `test_mcp_*.py`: MCP server contract tests
- `test_geometry_*.py`: Geometry validation tests
- **2671+ passing tests** - Don't break them

---

## What NOT To Do

1. **Don't hallucinate requirements** - If it's not documented here, don't invent it
2. **Don't create agent-specific config files** - This file is the source of truth
3. **Don't run git operations** - Other agents are working concurrently
4. **Don't "fix" architecture** - The MLX imports in training/ are intentional
5. **Don't over-engineer** - The codebase works; 2671 tests prove it
