# ModelCypher

Geometric diagnostics for LLM representations: intrinsic dimension, curvature, entropy, and representational similarity. Use it to guide model merging, monitor training stability, and detect behavioral drift.

Primary backend targets macOS/Apple Silicon. Optional GPU and TPU backends are available for Linux.

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

# Inspect a model's architecture
poetry run mc model info /path/to/model

# Merge two models with null-space knowledge addition
poetry run mc merge run -s /path/to/source -t /path/to/target -o /path/to/output_dir

# Compute layer-wise entropy trajectory
poetry run mc analyze entropy-trajectory --model /path/to/model --prompt "Your prompt here"

# Measure per-layer spectral entropy
poetry run mc analyze spectral-trajectory --model /path/to/model --prompt "Your prompt here"
```

## Evidence & Reproducibility

ModelCypher supports the claim that LLM representations behave like shared, curved geometry by providing reproducible measurements (not a proof). Key checks:

- Cross-model reasoning-geometry validation: per-layer probe AUROC, cognitive pivot effect sizes, and beta-1 topology deltas. Run the validation command below to generate results locally.
- Property-based invariants: extensive Hypothesis tests for null-space projection, CKA invariants, and numerical stability.

Reproduce:

```bash
# Reasoning geometry validation (writes report + per-model JSON)
poetry run mc analyze reasoning-geometry-validation \
  --model LFM2-350M \
  --benchmark arithmetic \
  --samples 20 \
  --output results/reasoning_geometry_validation/smoke

# Property-based tests (full)
HYPOTHESIS_PROFILE=full poetry run pytest
```

## Evidence Suite

Run the evidence suite to quantify generalization, approximation error, cross-model/domain variation, and causal intervention effects.

```bash
# Cross-model reasoning geometry validation
poetry run mc analyze reasoning-geometry-validation \
  --model LFM2-350M \
  --benchmark arithmetic \
  --samples 20

# Per-layer geodesic deviation profile
poetry run mc analyze geodesic-profile --model /path/to/model --prompt "Your prompt here"

# Compare geodesic trajectories across prompt categories
poetry run mc analyze geodesic-compare --model /path/to/model --suite /path/to/prompts.jsonl
```

Evidence outputs (raw measurements):
- Alignment generalization: train/holdout CKA + probe coverage ratio.
- Geodesic + curvature convergence on analytic manifolds (circle/sphere) with error ratios.
- Causal intervention: boundary preservation diffs + core shift residuals.

## Core Capabilities

| Command Group | Purpose |
|--------------|---------|
| `mc train` | Train LoRA adapters with geometry-derived hyperparameters |
| `mc merge` | Geometric model merging via null-space projection |
| `mc infer` | Inference with optional adapter loading and security scanning |
| `mc analyze` | Geometry, safety, and entropy analysis (30+ subcommands) |
| `mc model` | Model registry: inspect, search, quantize |
| `mc system` | System status, probes, and benchmarks |
| `mc adapter` | LoRA adapter analysis and baseline calibration |

In this repo, run `mc` via `poetry run mc …`. Run `poetry run mc --help` for the full command list.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/START-HERE.md](docs/START-HERE.md) | Main guide, installation, and reading paths |
| [AGENTS.md](AGENTS.md) | AI assistant guidance and architecture |
| [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) | Command reference |
| [docs/GEOMETRY-GUIDE.md](docs/GEOMETRY-GUIDE.md) | Geometry metrics explained |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terminology |
| [docs/references/BIBLIOGRAPHY.md](docs/references/BIBLIOGRAPHY.md) | Local PDFs and research references |

## License

AGPL-3.0. See [LICENSE](LICENSE).
