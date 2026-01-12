# ModelCypher

Geometric diagnostics for LLM representations: intrinsic dimension, curvature, entropy, and representational similarity. Use it to guide model merging, monitor training stability, and detect behavioral drift.

Primary backend is MLX (macOS/Apple Silicon). Optional CUDA and JAX backends are available for Linux.

## Install (Repo)

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install
```

Requires Python 3.11+.

## Quick Start

```bash
# CLI help (JSON to stdout by default)
poetry run mc --help

# Probe a model's architecture and geometry
poetry run mc model probe /path/to/model

# Merge two models with null-space knowledge addition
poetry run mc merge run -s /path/to/source -t /path/to/target -o /path/to/output_dir

# Analyze spatial geometry encoding
poetry run mc geometry spatial probe-model /path/to/model

# Measure entropy dynamics
poetry run mc thermo measure --model /path/to/model "Your prompt here"
```

## Evidence & Reproducibility

ModelCypher supports the claim that LLM representations behave like shared, curved geometry by providing reproducible measurements (not a proof). Key checks:

- Alignment invariance on probes: raw CKA increases to aligned CKA ~1.0 after Procrustes (see `experiments/results/geometry_validation.json`).
- Layer-wise intrinsic dimension compression and domain-specific manifold structure (same report).
- Property-based invariants: extensive Hypothesis tests for null-space projection, CKA invariants, and numerical stability.

Reproduce:

```bash
# Geometry experiments (writes experiments/results/geometry_validation.json)
poetry run python experiments/geometry_validation.py

# Property-based tests (full)
HYPOTHESIS_PROFILE=full poetry run pytest
```

## Limitations

- Alignment CKA = 1.0 is by construction on the probe set; generalization depends on probe diversity and coverage.
- Geodesic distances and curvature are approximations based on k-NN graphs and sampling; they are not global proofs of manifold structure.
- Results are model- and data-dependent; different architectures or domains require fresh measurements.
- Geometry here is descriptive and structural, not a complete causal explanation of behavior.

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

In this repo, run `mc` via `poetry run mc …`. Run `poetry run mc help` for contextual help and schemas.

## MCP Server

```bash
poetry run modelcypher-mcp
```

Tools available. See [docs/MCP.md](docs/MCP.md) for the full catalog.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/START-HERE.md](docs/START-HERE.md) | Main guide and reading paths |
| [docs/getting_started.md](docs/getting_started.md) | Installation + first commands |
| [CLAUDE.md](CLAUDE.md) | AI assistant guidance and architecture |
| [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) | Command reference |
| [docs/GEOMETRY-GUIDE.md](docs/GEOMETRY-GUIDE.md) | Geometry metrics explained |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terminology |
| [docs/references/BIBLIOGRAPHY.md](docs/references/BIBLIOGRAPHY.md) | Local PDFs and research references |

## License

AGPL-3.0. See [LICENSE](LICENSE).
