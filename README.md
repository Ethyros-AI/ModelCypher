# ModelCypher

Geometric diagnostics for LLM representations. Measures intrinsic dimension, curvature, entropy, and representational similarity to guide model merging, monitor training stability, and detect behavioral drift.

## Install

```bash
poetry install
```

Requires Python 3.11+. Primary backend is MLX (macOS/Apple Silicon). JAX backend available for Linux/TPU.

## Quick Start

```bash
# Probe a model's architecture and geometry
mc model probe /path/to/model

# Merge two models with geometric alignment
mc merge --source /path/to/source --target /path/to/target --output-dir /path/to/output

# Analyze spatial geometry encoding
mc geometry spatial probe-model /path/to/model

# Measure entropy dynamics
mc thermo measure "Your prompt here" /path/to/model
```

## Core Capabilities

| Command Group | Purpose |
|--------------|---------|
| `mc model` | Probe, fetch, register, validate models |
| `mc merge` | Cross-architecture model merging pipeline |
| `mc geometry` | Representational geometry analysis (30+ subcommands) |
| `mc thermo` | Linguistic thermodynamics and entropy measurement |
| `mc safety` | Behavioral drift and refusal pattern detection |
| `mc train` | Training with geometry monitoring |
| `mc infer` | Entropy-aware inference with security monitoring |

Run `mc --help` for full command list. Run `mc <command> --help` for subcommands.

## MCP Server

```bash
poetry run modelcypher-mcp
```

155+ tools available. See [docs/MCP.md](docs/MCP.md) for the full catalog.

## Documentation

| Doc | Purpose |
|-----|---------|
| [CLAUDE.md](CLAUDE.md) | AI assistant guidance and architecture |
| [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) | Command reference |
| [docs/GEOMETRY-GUIDE.md](docs/GEOMETRY-GUIDE.md) | Geometry metrics explained |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terminology |

## License

AGPL-3.0. See [LICENSE](LICENSE).
