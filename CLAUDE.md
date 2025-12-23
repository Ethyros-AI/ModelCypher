# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is ModelCypher?

A Python framework for measuring and experimenting with the geometry of representations in large language models. It provides geometric diagnostics (entropy, intrinsic dimension, topological fingerprints, representation similarity) for stability and refusal dynamics, drift monitoring, and model/adapter merge analysis.

Runs on macOS (MLX) for local research; CUDA backend stub exists for future Linux support.

## Virtual Environment

**IMPORTANT**: This project uses a local `.venv` virtual environment. Always use it directly:

```bash
# Activate the venv (preferred)
source .venv/bin/activate

# Or run commands directly with the venv python
.venv/bin/python -m pytest tests/
.venv/bin/python -c "import modelcypher"
```

Do NOT use `poetry run` or `uv run` - they may use cached/stale packages.

**Known Issue**: There's a dataclass field ordering bug in `domain_signal_profile.py` that may cause import errors. If you see "non-default argument follows default argument" errors, check and fix the DomainSignalDecision dataclass field ordering.

## Commands

```bash
# Install dependencies (if needed)
source .venv/bin/activate && pip install -e .

# Run all tests
.venv/bin/python -m pytest

# Run single test
.venv/bin/python -m pytest tests/test_geometry.py::test_name -v

# CLI usage (after install)
mc --help                  # or: modelcypher --help
mc model probe ./path      # probe a local model
mc geometry training status --job <id>
mc geometry safety circuit-breaker --model <path>

# Thermodynamics
mc thermo measure --model <path> --prompt "text"
mc thermo ridge-detect --trajectory <file>
mc thermo phase --model <path>
mc thermo sweep --model <path> --temps 0.1,0.5,1.0

# Adapter management
mc adapter blend --adapters a.safetensors,b.safetensors --weights 0.6,0.4 --output blended.safetensors
mc adapter ensemble create --adapters a.safetensors,b.safetensors --strategy weighted
mc adapter ensemble list
mc adapter ensemble apply --ensemble <id> --model <path>

# Research taxonomy
mc research taxonomy run ./signatures.json --model llama3 --k 5
mc research taxonomy cluster ./signatures.json --k 5
mc research taxonomy report ./signatures.json --model llama3 -o report.md

# Dataset quality/validation
mc dataset quality ./data.jsonl
mc dataset auto-fix ./data.jsonl --output ./data-fixed.jsonl

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
│   │   ├── merging/   # Model merge algorithms
│   │   ├── thermo/    # LinguisticThermodynamics, RidgeCross, PhaseTransition
│   │   ├── adapters/  # AdapterBlender, EnsembleOrchestrator
│   │   ├── research/  # JailbreakEntropyTaxonomy
│   │   └── validation/# DatasetQualityScorer, AutoFixEngine
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
- **Git operations**: Only non-destructive git operations allowed (e.g., `git status`, `git diff`, `git log`). Do NOT run `git add`, `git commit`, `git push`, `git checkout`, `git reset`, etc. Other agents work concurrently on this codebase.

## CLI Conventions

- `--ai` or `MC_AI_MODE=1` forces JSON output and suppresses prompts/logs
- `--output {text,json,yaml}` controls format
- Structured output goes to stdout; diagnostics/logs go to stderr
- For geometry metrics, use the `interpretation` and `recommendedAction` fields when summarizing for humans

## External Storage

Models and experiment output live on the external CodeCypher volume:

```
/Volumes/CodeCypher/
├── models/          # Local model weights for testing
├── adapters/        # LoRA and adapter files
├── caches/          # Fingerprint and activation caches
└── TrainingCypher/  # Training experiment output
```

When running experiments, use paths on this volume (e.g., `/Volumes/CodeCypher/models/qwen2.5-7b`).

## Test Structure

Tests mirror the domain structure in `tests/`. Key patterns:
- `test_*_properties.py`: Hypothesis property-based tests
- `test_mcp_*.py`: MCP server contract tests
- `test_geometry_*.py`: Geometry validation tests
- `conftest.py`: Provides NumpyBackend fixture for deterministic testing
