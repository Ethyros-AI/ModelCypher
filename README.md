# ModelCypher

**Geometry-first diagnostic toolkit for Large Language Model internals.** ModelCypher measures the geometric structure of LLM representations — intrinsic dimension, Riemannian curvature, spectral entropy, and representational similarity (CKA) — to guide LoRA adapter training, monitor training stability, and detect behavioral drift. Built on MLX for Apple Silicon, with optional CUDA and JAX backends.

ModelCypher treats neural network activations as points on a high-dimensional manifold and applies differential geometry, spectral analysis, and Procrustes alignment to produce reproducible, quantitative diagnostics. Every threshold in the codebase is derived from SVD, IEEE 754 precision bounds, or peer-reviewed theorems — not heuristics.

## What Does ModelCypher Measure?

ModelCypher provides 32 analysis commands across these measurement domains:

| Domain | What It Measures | Key Methods |
|--------|-----------------|-------------|
| **Representational similarity** | How similar are two models' internal representations? | Procrustes alignment, CKA (Centered Kernel Alignment), probe coverage |
| **Intrinsic dimension** | How many effective dimensions does each layer use? | TwoNN estimator, per-layer dimension profiles |
| **Curvature and geodesics** | How curved is the representation manifold? | Geodesic deviation, sectional curvature, Jacobian spectrum |
| **Spectral structure** | What is the singular value distribution of weight matrices? | SVD decomposition, spectral entropy, condition number analysis |
| **Entropy trajectories** | How does uncertainty evolve across layers? | Layer-wise logit entropy, entropy differentials, semantic certainty |
| **LoRA adapter geometry** | How does a LoRA adapter perturb the base model's geometry? | Spectral scale bounds, Weyl perturbation budget, null-space overlap |
| **Safety and behavioral drift** | Has the model's decision geometry shifted? | Refusal boundary movement, jailbreak entropy analysis, persona drift |

## How Does LoRA Training Differ in ModelCypher? [VALIDATED]

ModelCypher trains LoRA adapters using Cayley-parameterized updates on the Stiefel manifold with a Riemannian natural gradient (Amari 1998, Lezcano-Casado 2019). The key differences from standard LoRA:

- **Norm-bounded by construction** [PROVEN] — adapter factors stay on the Stiefel manifold via Cayley retraction, so spectral norms cannot explode
- **Geometry-derived step sizes** [PROVEN] — learning rate is bounded by `2 / (L * lambda_max(P))` where L is the Lipschitz constant and P is the natural gradient preconditioner
- **Weyl perturbation budget** [PROVEN] — monitors `||BA||_2 / sigma_k(W)` to ensure the adapter stays within the base model's spectral capacity
- **No magic numbers** — every hyperparameter traces back to SVD, IEEE 754 epsilon, or a cited theorem

Validated result (LFM2-350M) [VALIDATED]: validation loss 1.27 (Cayley-Riemannian) vs 1.38 (plain SGD), with geometric stopping certificate.

## Quick Start

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install          # Requires Python 3.11+
```

```bash
# Inspect a model's layer-wise geometry
poetry run mc model info /path/to/model

# Per-layer intrinsic dimension profile
poetry run mc analyze dimension-profile --model /path/to/model --prompt "Your prompt here"

# Layer-wise entropy trajectory
poetry run mc analyze entropy-trajectory --model /path/to/model --prompt "Your prompt here"

# Spectral entropy across layers
poetry run mc analyze spectral-trajectory --model /path/to/model --prompt "Your prompt here"

# LoRA adapter SVD decomposition
poetry run mc analyze lora-svd --adapter /path/to/adapter

# Train a LoRA adapter with geometric constraints
poetry run mc train run --model /path/to/model --data /path/to/data.jsonl --rank 8 --epochs 3
```

## Evidence and Reproducibility [VALIDATED]

ModelCypher validates that LLM representations exhibit shared, curved geometric structure through reproducible measurements. The test suite includes 6,000+ tests across 388 test files, including Hypothesis property-based tests for numerical invariants.

Reproduce key results locally:

```bash
# Cross-model reasoning-geometry validation (writes report + JSON)
poetry run mc analyze reasoning-geometry-validation \
  --model LFM2-350M \
  --benchmark arithmetic \
  --samples 20 \
  --output results/reasoning_geometry_validation/

# Property-based tests for null-space projection, CKA, and numerical stability
HYPOTHESIS_PROFILE=full poetry run pytest

# Geodesic deviation profile (curvature measurement)
poetry run mc analyze geodesic-profile --model /path/to/model --prompt "Your prompt here"
```

Key measurements from published validation:
- **CKA = 1.0** on training probes after Procrustes alignment (by construction) [PROVEN]
- **Intrinsic dimension compression**: 15.8 (early layers) to 1.8 (bottleneck) to 9.6 (output layers), consistent across architectures [EMPIRICAL]
- **Geometric stopping certificate**: 4 conditions (stationarity, improvement bound, worst-group, mechanism drift) for training termination [VALIDATED]
- **Cayley-Riemannian vs plain SGD**: validation loss 1.27 vs 1.38 on LFM2-350M [VALIDATED]

## CLI Command Reference

| Command Group | Purpose |
|--------------|---------|
| `mc train` | Train LoRA adapters with Cayley-parameterized Stiefel manifold updates |
| `mc infer` | Run inference with adapter loading and entropy-based safety scanning |
| `mc merge` | Experimental model merging via null-space projection |
| `mc analyze` | 32 geometry, safety, and entropy analysis subcommands |
| `mc model` | Model registry: inspect architecture, download, search, quantize |
| `mc system` | System status, backend probes, and hardware benchmarks |
| `mc adapter` | LoRA adapter spectral analysis and baseline calibration |

All commands output structured JSON by default. Run `poetry run mc --help` for the full command list.

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS Apple Silicon (M1/M2/M3/M4) | MLX | Primary (optimized) |
| Linux + NVIDIA GPU | CUDA (via PyTorch) | Supported |
| Linux + TPU | JAX | Supported |

## Documentation

| Document | Description |
|----------|-------------|
| [Start Here](docs/START-HERE.md) | Installation, first measurement, reading paths for engineers/researchers/auditors |
| [Geometry Guide](docs/GEOMETRY-GUIDE.md) | How to interpret CKA, intrinsic dimension, curvature, and entropy measurements |
| [CLI Reference](docs/CLI-REFERENCE.md) | Complete command reference with examples for all 7 command groups |
| [Training Guide](docs/TRAINING-GUIDE.md) | LoRA training workflows, geometric hyperparameters, dataset preparation |
| [Glossary](docs/GLOSSARY.md) | Definitions for Procrustes alignment, CKA, intrinsic dimension, Weyl bounds, and 60+ terms |
| [Research Papers](papers/README.md) | 6 research papers on geometric structure, entropy safety, and cross-architecture transfer |
| [Architecture](docs/ARCHITECTURE.md) | Hexagonal architecture, domain boundaries, backend abstraction |
| [Bibliography](docs/references/BIBLIOGRAPHY.md) | Cited papers and local reference PDFs |

## Research Papers

ModelCypher includes 6 research papers documenting the geometric framework:

1. **The Shape of Knowledge** — geometric knowledge thesis (supported by measurements)
2. **Invariant Semantic Structure** — CKA alignment invariance across architectures (supported)
3. **Entropy Safety Signal** — behavioral drift detection via entropy differentials
4. **Cross-Architecture Transfer** — knowledge transfer between model families
5. **ModelCypher Toolkit** — implementation methodology and CLI design
6. **The Semantic Highway** — layer-wise intrinsic dimension compression (supported: 15.8 to 1.8 to 9.6)

## Architecture

ModelCypher uses hexagonal (ports-and-adapters) architecture with strict domain boundaries:

- **Core domain** (`core/domain/`) — pure math and geometry, no framework imports
- **Use cases** (`core/use_cases/`) — orchestration layer, cannot import from adapters
- **Adapters** (`adapters/`) — concrete integrations (HuggingFace Hub, filesystem, MLX)
- **Backend abstraction** — framework-specific code (MLX, JAX, PyTorch) isolated behind a protocol interface

All geometric computations are framework-agnostic. Backend selection is automatic based on platform.

## Citation

If you use ModelCypher in your research, please cite:

```bibtex
@software{kempf2025modelcypher,
  author = {Kempf, Jason},
  title = {ModelCypher: Geometric Diagnostics for LLM Representations},
  year = {2025},
  url = {https://github.com/Ethyros-AI/ModelCypher},
  license = {AGPL-3.0}
}
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
