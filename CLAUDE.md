# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is ModelCypher?

A Python framework for measuring and experimenting with the geometry of representations in large language models. It provides geometric diagnostics (entropy, intrinsic dimension, topological fingerprints, representation similarity) for stability and refusal dynamics, drift monitoring, and model/adapter merge analysis.

Runs on macOS (MLX) for local research; CUDA backend stub exists for future Linux support.

## Commands

```bash
# Install dependencies
uv sync                    # recommended
uv sync --all-extras       # includes docs/cuda/embeddings extras

# Run all tests
uv run pytest

# Run single test
uv run pytest tests/test_geometry.py::test_name -v

# CLI usage (after install)
mc --help                  # or: modelcypher --help
mc model probe ./path      # probe a local model
mc geometry training status --job <id>
mc geometry safety circuit-breaker --model <path>

# MCP server
uv run modelcypher-mcp
```

## Architecture

Strict **Hexagonal Architecture** (Ports and Adapters):

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + business logic (NO adapter imports)
│   │   ├── geometry/  # ManifoldStitcher, Procrustes, probe corpus
│   │   ├── safety/    # CircuitBreaker, refusal detection
│   │   ├── training/  # LoRA configs, geometric training metrics
│   │   ├── entropy/   # Shannon entropy calculations
│   │   └── merging/   # Model merge algorithms
│   ├── ports/         # Abstract interfaces (ABCs)
│   └── use_cases/     # Service orchestration (geometry_service, training_service, etc.)
├── adapters/          # Concrete implementations (hf_hub, filesystem, local_training)
├── backends/          # MLX (macOS) and CUDA (stub) compute backends
├── cli/               # Typer CLI (entry: mc / modelcypher)
├── mcp/               # Model Context Protocol server
└── data/              # Static data (semantic_primes.json, etc.)
```

**Dependency rule**: Dependencies point inward. Domain depends on nothing; adapters implement ports; CLI/MCP drive the application.

## Key Concepts

- **Manifold Stitcher**: Aligns disparate model manifolds using Procrustes analysis
- **Probe Corpus**: Standardized prompts to elicit comparable activations across models
- **Semantic Primes**: Anchor inventory in `src/modelcypher/data/semantic_primes.json`
- **Circuit Breaker**: Monitors regime state, interrupts generation on refusal basin or unstable trajectory
- **DARE/DoRA**: Adapter sparsity and directional decomposition for merge analysis

## Development Rules

- Do NOT import `adapters` into `core` (respect the hexagon)
- MLX backend requires `mx.eval()` after operations; weight layout is `[out, in]`
- Use Python logging, not print, in core logic
- Tests use `pytest` with `pytest-asyncio` (async mode auto); property tests use `hypothesis`
- If overlap with another agent's work is likely, research best practice and explain choice in code comments

## CLI Conventions

- `--ai` or `MC_AI_MODE=1` forces JSON output and suppresses prompts/logs
- `--output {text,json,yaml}` controls format
- Structured output goes to stdout; diagnostics/logs go to stderr
- For geometry metrics, use the `interpretation` and `recommendedAction` fields when summarizing for humans

## Test Structure

Tests mirror the domain structure in `tests/`. Key patterns:
- `test_*_properties.py`: Hypothesis property-based tests
- `test_mcp_*.py`: MCP server contract tests
- `test_geometry_*.py`: Geometry validation tests
- `conftest.py`: Provides NumpyBackend fixture for deterministic testing
