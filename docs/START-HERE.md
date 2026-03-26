# Start Here: See What A Model Is Doing

ModelCypher is a measurement and observability workbench for open-source model
builders. You bring a model and a prompt family. ModelCypher gives you bundleable
measurements about geometry, entropy, curvature, trajectory shape, and how those
signals move when you change prompt style, formatting, casing, or adapters.

You should not need to memorize MLX internals or hand-roll activation
collection just to ask questions like:

- does ALL CAPS bend the trajectory differently?
- does profanity move entropy or geodesic deviation?
- where does an adapter change the chain profile?
- what changed below token level when behavior changed above it?

## Current Reality (2026-03-26)

- `mc analyze` is now the clearest entrypoint for workflow-first model observation.
- `mc analyze capture`, `mc analyze family`, and `mc analyze compare` create
  observation bundles you can inspect or hand to another agent.
- `mc analyze report --bundle ...` is the read-side companion for existing bundles.
- Training remains shipped and useful, but it is a downstream workflow rather
  than the headline.
- The repo still does **not** claim benchmark superiority for its training path.
- Merge, continual learning, and stacking remain experimental or partial.

That means the product story is honest:

- the workbench exists,
- the derived planning is real,
- benchmark superiority is still being measured rather than assumed.

## Install

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install
```

### Prerequisites

| Platform | Requirements |
| --- | --- |
| macOS (Apple Silicon) | Apple Silicon, 16GB+ RAM, macOS 14+, Python 3.11+ |
| Linux (NVIDIA GPU) | NVIDIA GPU, Python 3.11+ |
| Linux/Cloud (TPU/GPU) | TPU or GPU, Python 3.11+ |

### Check The System

```bash
poetry run mc system status
poetry run mc system probe backends
```

If the backend you expect is not available, install the matching extras from
`pyproject.toml` and re-run the probe.

## The Main Workflow

This is the shortest honest path from "I have a model" to "I can see what
changed below token level."

### 1. Inspect The Model

```bash
poetry run mc model info /path/to/model
poetry run mc model capacity /path/to/model --sort-by recommended-rank
```

Use `mc model info` to verify the model loads and `mc model capacity` to inspect
the spectral structure ModelCypher will use when it derives target modules and
ranks.

### 2. Capture A Prompt

```bash
poetry run mc analyze capture \
  --model /path/to/model \
  --prompt "Explain geodesics."
```

This writes an observation bundle with machine-readable outputs and a short
report. Use this when you want a quick single-prompt view of the model’s
internal measurements.

### 3. Run A Prompt Family

```bash
poetry run mc analyze family \
  --model /path/to/model \
  --manifest data/probes/prompt_family_minimal_pairs.json
```

This is the canonical workflow for controlled perturbation studies such as
control vs ALL CAPS vs profanity vs formatting.

### 4. Compare Two Targets

```bash
poetry run mc analyze compare \
  --left-model /path/to/base \
  --right-model /path/to/base \
  --right-adapter /path/to/adapter \
  --manifest data/probes/prompt_family_minimal_pairs.json
```

Use this when you want to see what an adapter or checkpoint changed on the same
prompt family.

### 5. Read An Existing Bundle

```bash
poetry run mc analyze report --bundle /path/to/bundle
```

Use this when you already have a bundle directory and want the shared report
view without opening files manually.

### 6. Drop Into Expert Metrics

```bash
poetry run mc analyze reasoning-flow --model /path/to/model --prompt "Prove that sqrt(2) is irrational."
poetry run mc analyze chain-profile --model /path/to/model
poetry run mc analyze geodesic-profile --model /path/to/model --prompt "Explain geodesics."
```

These remain available when you want direct access to the underlying geometry
tools instead of the bundle-oriented workflows.

## What ModelCypher Gives You

| Without ModelCypher | With ModelCypher |
| --- | --- |
| Guess a LoRA rank | Derive ranks from spectral structure |
| Pick a learning rate recipe | Let the controller resolve derived step sizes |
| Tune stopping by feel | Stop on measured convergence and certificates |
| Hope the adapter helped | Evaluate and compare with built-in commands |
| Memorize backend quirks | Use one CLI across the supported backends |

Geometry matters here because it makes the tool more useful. The point is not
to make users read theory first. The point is to reduce tuning guesswork and
make training results easier to trust.

## Downstream Training

Training remains in the product when you want to turn these measurements into a
derived adapter workflow:

```bash
poetry run mc train run --model /path/to/model --data /path/to/train.jsonl --plan-only
poetry run mc train run --model /path/to/model --data /path/to/train.jsonl --output /path/to/adapter
poetry run mc train evaluate --model /path/to/model --adapter /path/to/adapter --data /path/to/validation.jsonl
```

The difference is emphasis: training is still shipped, but it now sits
downstream of the measurement layer instead of defining the repo’s headline.

### Merge Models (Experimental)

```bash
poetry run mc merge run \
  -s /path/to/source_model \
  -t /path/to/target_model \
  -o /path/to/output
```

This path is useful, but it is still experimental. Do not treat it as the
repository's main shipped promise today.

## Documentation Map

- [OBSERVATION-BUNDLES.md](OBSERVATION-BUNDLES.md): manifest schema, bundle files, and starter perturbation manifests
- [TRAINING-GUIDE.md](TRAINING-GUIDE.md): the downstream adapter workflow
- [CLI-REFERENCE.md](CLI-REFERENCE.md): live command reference
- [MISSION.md](MISSION.md): product mission and implementation standards
- [VISION.md](VISION.md): where the workbench is headed
- [GEOMETRY-GUIDE.md](GEOMETRY-GUIDE.md): why the derived surfaces look the way they do
- [GLOSSARY.md](GLOSSARY.md): terminology
- [AGENTS.md](../AGENTS.md): coding and research doctrine for contributors

## Background Reading

Read these after you have the workflow in hand:

- [papers/README.md](../papers/README.md)
- [paper-0-the-shape-of-knowledge.md](../papers/paper-0-the-shape-of-knowledge.md)
- [paper-4-modelcypher-toolkit.md](../papers/paper-4-modelcypher-toolkit.md)
- [research/mental_model.md](research/mental_model.md)
- [research/ATLAS-BASED-GEOMETRY.md](research/ATLAS-BASED-GEOMETRY.md)

## Troubleshooting

**Model not found**
Use an absolute path and make sure the model directory contains the expected
config and weight files.

**Backend not available**
Run `poetry run mc system probe backends` and install the backend extras your
platform needs.

**Out of memory**
Try a smaller model first, or use `mc model quantize` if quantized deployment
fits your workflow.
