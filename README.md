# ModelCypher

**Geometry-first LoRA training. Every hyperparameter derived from the weight matrix, not tuned.**

## The Thesis

A forward pass is a deterministic geometric map. The industry treats 15 training hyperparameters as knobs to tune — learning rate, rank, scale, warmup, clipping, schedule, decay, dropout, batch size, early stopping, target modules, weight init, epsilon, momentum, residual scaling. Every one of these has a closed-form geometric replacement derived from SVD, IEEE 754 machine precision, or a cited theorem. ModelCypher replaces all 15. See [AGENTS.md](AGENTS.md) for the full derivation philosophy.

## One Command, Zero Configuration

```bash
poetry run mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
```

No learning rate. No rank selection. No warmup schedule. No gradient clipping. The optimizer (Cayley-Stiefel retraction on the Stiefel manifold) and step size (MASS: `eta = min(eta_ceiling, eta_sps, eta_weyl)`) are derived from the weight matrices at initialization.

Need explicit control for research instrumentation? Use `mc train run-research`.

**Validated result** (LFM2-350M): val_loss 1.27 (Cayley-Stiefel) vs 1.38 (plain SGD), with geometric stopping certificate.

## What Gets Derived

| # | What Industry Tunes | What ModelCypher Derives | Source |
|---|---|---|---|
| 1 | Learning rate (`1e-4`) | MASS spectral ceiling | Weyl 1912, Loizou 2020 |
| 2 | Adam epsilon (`1e-8`) | Spectral noise floor | IEEE 754 + SVD |
| 3 | Momentum (`0.9/0.999`) | Cayley-Stiefel retraction | Wen & Yin 2013, Wang 2025 |
| 4 | Weight decay (`0.01`) | Condition ratio `sigma_k / sigma_max` | SVD |
| 5 | Gradient clipping (`1.0`) | **Removed** — MASS bounds by construction | Weyl 1912 |
| 6 | Warmup (5-10% steps) | **Removed** — geometric LR stable from step 0 | Ma & Yarats 2021 |
| 7 | LR schedule (cosine) | **Removed** — MASS is per-step, no schedule needed | Defazio 2024 |
| 8 | Batch size | Gradient noise scale `B_crit` | McCandlish 2018 |
| 9 | Early stopping (patience) | 4 geometric criteria | SVD + IEEE 754 |
| 10 | LoRA scale (`alpha/rank`) | Spectral bound `sigma_k(W) / \|\|BA\|\|` | Weyl perturbation theory |
| 11 | LoRA rank (`8`) | Null-space capacity `tail_dims` | Shannon effective rank |
| 12 | Target modules (`q+v`) | Spectral decay analysis | SVD per-layer |
| 13 | Dropout (`0.1`) | Product of two spectral ratios | Roy & Vetterli 2007 |
| 14 | Weight init (random A, zero B) | Spectral normalized to `sigma_k` | SVD |
| 15 | Residual scaling (`1`) | Per-layer `sigma_max(x) / sigma_max(f(x))` | Power iteration |

Full derivations with formulas: [Geometric Hyperparameter Rosetta Stone](docs/research/geometric_hyperparameter_rosetta_stone.md)

## Quick Start

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install          # Python 3.11+
poetry run mc --help    # Verify CLI install
```

```bash
# Train a LoRA adapter — all hyperparameters derived from geometry
poetry run mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter

# Validate derived training across repeated trials (counterexample search)
poetry run mc train validate-derived --model /path/to/model --data /path/to/data.jsonl --trials 5

# Inspect a model's per-layer geometry
poetry run mc model info /path/to/model

# Layer-wise intrinsic dimension profile
poetry run mc analyze dimension-profile --model /path/to/model --samples 50

# LoRA adapter spectral analysis
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model
```

## Results

### What Worked

| Model | Method | Result | Tag |
|-------|--------|--------|-----|
| LFM2-350M | Cayley-Stiefel + CE | val_loss 1.27 vs 1.38 (plain SGD) | [VALIDATED] |
| LFM2-1.2B | Answer-mask + retention replay | 36/46 (78%), 0 degenerate outputs | [VALIDATED] |
| Cross-family | Weight geometry falsification (LFM2 + Qwen2.5) | Weight space Euclidean, activation space curved | [PROVEN] |
| CKA alignment | Procrustes on training probes | CKA = 1.0 (by construction: `F = pinv(source) @ target`) | [PROVEN] |

### What Failed

| Hypothesis | Result | Tag |
|------------|--------|-----|
| REINFORCE on 350M | Gradient orthogonal to CE; degradation monotonic with steps | [DISPROVEN] |
| SFT on reasoning traces | Format memorization: PPL drops, inference degrades | [DISPROVEN] |
| Pullback metric P = MM^T | P ≈ I throughout training (median deviation 0.001) | [DISPROVEN] |
| Stable rank predicts adapter rank | Pearson r = -0.51 vs tail_dims; measures different property | [DISPROVEN] |
| Constrained training (paired) | Constraints monotonically hurt | [DISPROVEN] |

We publish failures because intellectual honesty is not optional. Full details: [CURRENT-STATE.md](docs/CURRENT-STATE.md)

## Measurement Toolkit

28 analysis subcommands under `mc analyze` across 5 categories:

| Category | What It Measures |
|----------|-----------------|
| Geometric | Intrinsic dimension, geodesic curvature, expansion ratio, spectral entropy, Jacobian spectrum |
| Behavioral | Adapter probes, behavioral signatures, cognitive reflection |
| Safety | Jailbreak entropy, refusal boundaries, red-team probes, circuit breakers |
| Benchmark | LoRA SVD, knowledge typing, curriculum profiling, sparse regions |
| Monitoring | Persona drift, uncertainty modes, entropy baselines |

53 total commands across 7 groups (`train`, `merge`, `infer`, `analyze`, `model`, `system`, `adapter`). Full reference: [CLI-REFERENCE.md](docs/CLI-REFERENCE.md)

## Architecture

Hexagonal (ports-and-adapters) with strict domain boundaries:

- **Core domain** (`core/domain/`) — pure geometry and math, zero framework imports
- **Use cases** (`core/use_cases/`) — orchestration, cannot import from adapters
- **Adapters** (`adapters/`) — HuggingFace Hub, filesystem, model loading
- **Backends** — MLX (primary, Apple Silicon), CUDA, JAX behind a protocol interface

All geometric computations are framework-agnostic. Backend selection is automatic.

## Documentation

| Document | What It Covers |
|----------|---------------|
| [Start Here](docs/START-HERE.md) | Installation, first measurement, reading paths for engineers/researchers/auditors |
| [Geometry Guide](docs/GEOMETRY-GUIDE.md) | Interpreting CKA, intrinsic dimension, curvature, and entropy measurements |
| [Training Guide](docs/TRAINING-GUIDE.md) | LoRA training workflows and dataset preparation |
| [CLI Reference](docs/CLI-REFERENCE.md) | All 53 commands with examples |
| [Mission](docs/MISSION.md) | The 15 hyperparameter replacements and why they work |
| [Glossary](docs/GLOSSARY.md) | 60+ term definitions |
| [Architecture](docs/ARCHITECTURE.md) | Hexagonal architecture and domain boundaries |
| [Bibliography](docs/references/BIBLIOGRAPHY.md) | All cited papers with local reference PDFs |

## Research Papers

| Paper | Status | Thesis |
|-------|--------|--------|
| [The Shape of Knowledge](papers/paper-0-the-shape-of-knowledge.md) | [EMPIRICAL] | Knowledge has measurable geometric structure; inference is trajectory |
| [Invariant Semantic Structure](papers/paper-1-invariant-semantic-structure.md) | [PROVEN] intra-model; [CONJECTURAL] cross-model | CKA alignment invariance across layers (by construction on training probes) |
| [Entropy Safety Signal](papers/paper-2-entropy-safety-signal.md) | [CONJECTURAL] | Behavioral drift detection via entropy differentials |
| [Cross-Architecture Transfer](papers/paper-3-cross-architecture-transfer.md) | [CONJECTURAL] | Knowledge transfer between model families via Procrustes alignment |
| [ModelCypher Toolkit](papers/paper-4-modelcypher-toolkit.md) | [EMPIRICAL] | Implementation methodology and CLI design |
| [The Semantic Highway](papers/paper-5-semantic-highway.md) | [EMPIRICAL] | Layer-wise intrinsic dimension compression (15.8 → 1.8 → 9.6) |

## Test Suite

6,294 tests across 401 test files. Includes Hypothesis property-based tests for numerical invariants (CKA symmetry, spectral bounds, null-space orthogonality).

```bash
poetry run pytest                              # Standard run
HYPOTHESIS_PROFILE=full poetry run pytest       # Full property-based testing
```

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS Apple Silicon (M1-M4) | MLX | Primary (optimized) |
| Linux + NVIDIA GPU | CUDA (PyTorch) | Supported |
| Linux + TPU | JAX | Supported |

## Citation

```bibtex
@software{kempf2026modelcypher,
  author = {Kempf, Jason},
  title = {ModelCypher: Geometry-First LoRA Training for LLMs},
  year = {2026},
  url = {https://github.com/Ethyros-AI/ModelCypher},
  license = {AGPL-3.0}
}
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
