# ModelCypher

**See what a model is doing below token level.**

ModelCypher is a measurement and observability workbench for open-source model
builders. It gives humans and frontier AI a clear way to inspect geometry,
entropy, curvature, chain structure, and adapter-induced changes through
workflow-first CLI surfaces instead of ad hoc activation scripts.

Current evidence state (2026-04-02): `mc analyze` is the clearest public
entrypoint for prompt capture, prompt-family studies, and checkpoint or
adapter comparison. `mc train run` remains shipped and geometry-derived, but
the repo has not yet closed the promotable same-model same-data same-eval
benchmark needed to claim "better than standard practice." See
[RESEARCH-ROADMAP.md](docs/RESEARCH-ROADMAP.md).

## The Measurement Thesis

A forward pass is a deterministic geometric map, and `mc analyze` is the
shipped workbench for measuring that map below token level. The downstream
training program derives all 15 traditional controls from model geometry,
precision, or measured data where the current derivation is wired; the runtime
status table below names the places where the shipped default still uses a
calibrated AdamW value or an unwired formula. See
[AGENTS.md](AGENTS.md) for the full derivation philosophy.

## Start By Measuring

```bash
poetry run mc analyze capture --model /path/to/model --prompt "Explain geodesics."
poetry run mc analyze family --model /path/to/model --manifest data/probes/prompt_family_minimal_pairs.json
poetry run mc analyze compare --left-model /path/to/base --right-model /path/to/base --right-adapter /path/to/adapter --manifest data/probes/prompt_family_minimal_pairs.json
poetry run mc analyze report --bundle /path/to/bundle
poetry run mc analyze report --bundle results/measurement_atlas
poetry run mc analyze report --bundle results/measurement_atlas/<run_id>
poetry run mc analyze report --bundle results/pipeline_validation
poetry run python scripts/run_measurement_atlas.py --model /path/to/model --manifest data/probes/measurement_atlas_casing.json --manifest data/probes/measurement_atlas_profanity_tone.json --manifest data/probes/measurement_atlas_grounded_hallucination.json --output-root results/measurement_atlas
```

These commands emit an observation bundle under
`results/analysis/<timestamp-slug>/` by default:

- `manifest.json`
- `summary.json`
- `REPORT.md`
- `variants.jsonl`
- `layer_metrics.jsonl`
- `comparisons.jsonl`

The prompt-family interface is explicit in phase 1. Each row includes:
`case_id`, `variant_id`, `text`, optional `tags`, and optional
`comparison_to`.

For research-only generation tracing, the measurement atlas runner writes a
family artifact under `results/measurement_atlas/<run_id>/` with:

- `run_manifest.json`
- `summary.json`
- `REPORT.md`
- `ledger.tsv`
- `variants.jsonl`
- `sequence_metrics.jsonl`
- `step_metrics.jsonl`
- `space_step_metrics.jsonl`
- `comparisons.jsonl`
- `onset_events.jsonl`

The retained replay-alignment closure for the shipped 350M atlas pack has a
tracked report copy at
[`docs/research/reports/measurement_atlas/REPORT.md`](docs/research/reports/measurement_atlas/REPORT.md).
Current observed atlas surfaces are `replay={hidden, embedding}` and
`live={hidden}`; `run_manifest.json` now records requested vs observed
surfaces separately so the bundle does not overclaim unsupported replay space
coverage. `mc analyze report --bundle ...` can now read both the standard
`mc analyze` bundles, the retained atlas family root, individual atlas artifact
directories, and retained pipeline-validation families. Atlas generation itself
remains research-only in `scripts/run_measurement_atlas.py`.

## Train When You Want To Act On The Measurements

```bash
poetry run mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
```

The shipped path derives the adapter surface, rank, batch sizing, scale budget,
and stopping certificate from measured geometry. Its default optimizer is still
the calibrated AdamW/cosine path called out in the status table; MASS remains
available on research modes until the closure benchmark earns promotion.

Need extra instrumentation? Use flags on the same command path, such as
`--benchmark`, `--topo-monitor`, `--dim-monitor`, or `--entropy-reg`.

## 15-Control Runtime Status

<!-- BEGIN GENERATED KNOB MATRIX -->

| # | Training control | Runtime status | Current code truth | Code source |
|---|---|---|---|---|
| 1 | Learning rate | derived+research-mode-only | default: calibrated AdamW 2e-4 cosine; MASS on research modes | `_mlx_training_adapter_train_mixin.py` + `mass_step_size.py` |
| 2 | Adam epsilon | formula-exists-unwired | formula exists in `compute_geometric_epsilon`; shipped AdamW path does not consume it | `geometric_optimizer.py` |
| 3 | Momentum | derived+research-mode-only | default: AdamW betas 0.9/0.999; Fisher/MASS moments only in research modes | `_mlx_training_adapter_train_mixin.py` + `diagonal_fisher_preconditioner.py` |
| 4 | Weight decay | formula-exists-unwired | condition-ratio formula exists; shipped `mc train run` default passes `weight_decay=0.0` | `geometric_optimizer.py`, `dataset_training_service.py` |
| 5 | Gradient clipping | derived+research-mode-only | MASS bounds updates in research modes; canonical AdamW path has no geometric clipper | `mass_step_size.py` |
| 6 | Warmup | derived+research-mode-only | canonical path uses calibrated cosine from step 0; research modes use MASS ceilings | `dataset_training_service.py`, `mass_step_size.py` |
| 7 | LR schedule | derived+research-mode-only | default: cosine over 6 data-epochs; no-schedule MASS only in research modes | `_mlx_training_adapter_train_mixin.py`, `mass_step_size.py` |
| 8 | Batch size | derived+shipped-default | derived from gradient-noise scale, then reduced only for memory-safe micro-batching | `DatasetTrainingService.train_from_dataset` |
| 9 | Early stopping | derived+shipped-default | geometric certificate and measured validation-loss windows are wired into training | `geometric_early_stopping.py`, `_mlx_training_adapter_train_mixin.py` |
| 10 | LoRA scale | derived+shipped-default | adapter scale budget and saturation telemetry are enforced during training | `geometric_lora.py`, `_mlx_training_adapter_train_mixin.py` |
| 11 | LoRA rank | derived+shipped-default | per-module ranks derive from tail dimensions and rank-capacity samples | `geometric_lora.py`, `DatasetTrainingService.build_training_plan` |
| 12 | Target modules | derived+shipped-default | target surface derives from layer spectral geometry | `select_target_modules`, `DatasetTrainingService.build_training_plan` |
| 13 | Dropout | removed | derived dropout formula was deleted because no shipped training adapter consumed it | `compute_geometric_dropout` runtime references=0 |
| 14 | Weight init | removed | default init is PiSSA; the unshipped spectral-normalized helper was deleted | `spectral_normalized_lora_init` runtime references=0 |
| 15 | Residual scaling | removed | standalone residual-scaling formula was deleted because no shipped path consumed it | `residual_scaling.py`; runtime references=0 |

<!-- END GENERATED KNOB MATRIX -->

Research program and formulas:
[15-Hyperparameter Research Program](docs/research/15-HYPERPARAMETER-RESEARCH-PROGRAM.md)
and [Geometric Hyperparameter Rosetta Stone](docs/research/geometric_hyperparameter_rosetta_stone.md).

## Quick Start

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install          # Python 3.11+
poetry run mc --help    # Verify CLI install
```

```bash
# Inspect a model's per-layer geometry
poetry run mc model info /path/to/model

# Build an observation bundle from one prompt
poetry run mc analyze capture --model /path/to/model --prompt "Explain geodesics."

# Build an observation bundle from a prompt family
poetry run mc analyze family --model /path/to/model --manifest data/probes/prompt_family_minimal_pairs.json

# Re-read an existing bundle and print the shared report view
poetry run mc analyze report --bundle /path/to/bundle

# Re-read the retained measurement-atlas family root or one atlas run
poetry run mc analyze report --bundle results/measurement_atlas
poetry run mc analyze report --bundle results/measurement_atlas/<run_id>

# Re-read the retained pipeline-validation family
poetry run mc analyze report --bundle results/pipeline_validation

# Layer-wise intrinsic dimension profile
poetry run mc analyze dimension-profile --model /path/to/model --samples 50

# LoRA adapter spectral analysis
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model

# Train a LoRA adapter after inspecting the model
poetry run mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
```

## Evidence Snapshot

| Question | What retained artifacts show | Tag |
|---|---|---|
| Does the measurement layer exist as a real workflow? | Yes. `mc analyze capture`, `mc analyze family`, and `mc analyze compare` now emit observation bundles with machine-readable artifacts plus a short report | [EMPIRICAL] |
| Canonical training path exists | `mc train run` is the shipped geometry-derived runtime path guarded by `pipeline_gate_v1` | [EMPIRICAL] |
| Does the retained 350M validation bundle close preservation? | No. `results/pipeline_validation/verdict.json` reports structural pass `5/5`, inference pass `3/5`, `all_pass = false` | [EMPIRICAL] |
| Does the retained evidence close "better than standard practice"? | No. `results/nblora_vs_standard/` is retained as `summary_only`, and the retained single-seed LFM2-350M summary does not support superiority of `nb_lora` over the kept baselines | [EMPIRICAL] |
| Does the 8B bundle close efficacy? | No. `results/g5_8b_validation_multiseed/multiseed_gates.json` still fails `cka_ok` and `degenerate_ok` | [EMPIRICAL] |
| Is quantization promising? | Yes as a measurement surface: `results/quantization_frontier/20260227T235714Z/quantization_frontier.json` shows PPL and degeneration improvement on all 3 retained models, but the frontier law is still open | [EMPIRICAL] |

### Falsified Training Claims

| Hypothesis | Result | Tag |
|------------|--------|-----|
| REINFORCE on 350M | Gradient orthogonal to CE; degradation monotonic with steps | [DISPROVEN] |
| SFT on reasoning traces | Format memorization: PPL drops, inference degrades | [DISPROVEN] |
| Pullback metric P = MM^T | P ≈ I throughout training (median deviation 0.001) | [DISPROVEN] |
| Stable rank predicts adapter rank | Pearson r = -0.51 vs tail_dims; measures different property | [DISPROVEN] |
| Constrained training (paired) | Constraints monotonically hurt | [DISPROVEN] |

We publish failures because intellectual honesty is not optional. Current
training blockers and exit criteria live in [MISSION.md](docs/MISSION.md) and
[RESEARCH-ROADMAP.md](docs/RESEARCH-ROADMAP.md).

## Measurement Toolkit

`mc analyze` is organized around five canonical workflows:

- `capture`: measure one prompt or prompt file
- `family`: run explicit minimal-pair or perturbation studies
- `compare`: compare two targets on the same prompt family
- `report`: read an existing bundle and render the shared high-signal view
- `probe`: targeted probe and red-team workflows

Expert instruments remain directly callable when you want the underlying
measurements without the bundle wrapper:

- geometry and trajectory: `reasoning-flow`, `geodesic-profile`,
  `entropy-trajectory`, `chain-profile`, `dimension-profile`, `jacobian-trace`
- probe and monitoring: `calibrate-safety`, `jailbreak-test`,
  `probe-redteam`, `probe-behavioral`, `circuit-breaker`, `entropy-pattern`
- diagnostics and adapter analysis: `lora-svd`, `benchmark`,
  `knowledge-type`, `curriculum-profile`, `crm-build`, `crm-compare`

Full reference: [CLI-REFERENCE.md](docs/CLI-REFERENCE.md)

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
| [Start Here](docs/START-HERE.md) | Installation, first observation bundle, and downstream training |
| [Observation Bundles](docs/OBSERVATION-BUNDLES.md) | `PromptFamilyManifest`, `ObservationBundle`, and ready-to-run perturbation manifests |
| [Geometry Guide](docs/GEOMETRY-GUIDE.md) | Interpreting CKA, intrinsic dimension, curvature, and entropy measurements |
| [Training Guide](docs/TRAINING-GUIDE.md) | Downstream adapter workflows and dataset preparation |
| [CLI Reference](docs/CLI-REFERENCE.md) | Workflow-first `mc analyze` plus expert command examples |
| [Mission](docs/MISSION.md) | Measurement-first mission and derived training standards |
| [15-Hyperparameter Research Program](docs/research/15-HYPERPARAMETER-RESEARCH-PROGRAM.md) | Per-control runtime and evidence status for the derivation program |
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

<!-- TEST-COUNT:START -->
7,735 collected tests. This count is generated from `pytest --collect-only`; refresh it with `poetry run python scripts/update_test_count.py --write`.
<!-- TEST-COUNT:END -->

Includes Hypothesis property-based tests for numerical invariants (CKA symmetry, spectral bounds, null-space orthogonality).

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
  title = {ModelCypher: MLX-Native Measurement Workbench for LLMs},
  year = {2026},
  url = {https://github.com/Ethyros-AI/ModelCypher},
  license = {AGPL-3.0}
}
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
