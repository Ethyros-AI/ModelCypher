# Start Here: Train A Model That Helps

ModelCypher is a training workbench for open-source model builders. You bring a
model and a dataset. ModelCypher derives the training plan from the model's
geometry, runs training, and gives you evidence about whether the adapter
actually helped.

You should not need to memorize MLX internals, guess at LoRA rank, or cargo
cult a learning rate schedule to fine-tune a model for your domain.

## Current Reality (2026-03-16)

- `mc train run` is the shipped training surface.
- The workbench can already derive plans, train adapters, and evaluate results.
- The repo has **not** yet closed a promotable claim that its current training
  path beats standard practice head-to-head.
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

This is the shortest honest path from "I have a model and data" to "I know
whether the adapter helped."

### 1. Inspect The Model

```bash
poetry run mc model info /path/to/model
poetry run mc model capacity /path/to/model --sort-by recommended-rank
```

Use `mc model info` to verify the model loads and `mc model capacity` to inspect
the spectral structure ModelCypher will use when it derives target modules and
ranks.

### 2. Prepare Data If Needed

If your dataset is not already JSONL in one of the training formats, convert it:

```bash
poetry run mc data prepare /path/to/source --output /path/to/train.jsonl
```

`mc train run` consumes JSONL with either:

- `{"text": "..."}`
- `{"messages": [{"role": "...", "content": "..."}]}`

### 3. Derive The Training Plan

```bash
poetry run mc train run \
  --model /path/to/model \
  --data /path/to/train.jsonl \
  --plan-only
```

This is the core product surface. ModelCypher resolves the derived plan before
training: sequence length, target modules, per-module ranks, controller
quantities, and post-training verification surfaces.

### 4. Train The Adapter

```bash
poetry run mc train run \
  --model /path/to/model \
  --data /path/to/train.jsonl \
  --output /path/to/adapter
```

The current shipped path is `geometry-derived LoRA`. The goal is simple:
produce a useful adapter without asking the user to tune folklore knobs.

### 5. Evaluate Whether It Helped

Use one of the built-in evaluation modes:

```bash
poetry run mc train evaluate \
  --model /path/to/model \
  --adapter /path/to/adapter \
  --data /path/to/validation.jsonl

poetry run mc train evaluate \
  --model /path/to/model \
  --adapter /path/to/adapter \
  --prompts /path/to/eval_prompts.jsonl
```

If you want benchmark scores, use the benchmark mode:

```bash
poetry run mc train evaluate \
  --model /path/to/model \
  --adapter /path/to/adapter \
  --benchmark quick
```

### 6. Compare Results

Compare two saved runs or two adapters side by side:

```bash
poetry run mc train compare \
  --result-a /path/to/run_a.json \
  --result-b /path/to/run_b.json

poetry run mc train compare \
  --model /path/to/model \
  --adapter-a /path/to/adapter_a \
  --adapter-b /path/to/adapter_b \
  --data /path/to/validation.jsonl
```

### 7. Export Or Merge

```bash
poetry run mc train export \
  --agent agent-001 \
  --model /path/to/model \
  --output /path/to/export_dir

poetry run mc train merge \
  --agent agent-001 \
  --model /path/to/model \
  --save \
  --output /path/to/merged_model
```

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

## Other Paths

### Analyze A Model

Use the analysis surfaces when you want to inspect geometry directly:

```bash
poetry run mc analyze dimension-profile --model /path/to/model
poetry run mc analyze entropy-trajectory --model /path/to/model
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model
```

Start here when you are debugging behavior, profiling capacity, or validating a
hypothesis about the model.

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

- [TRAINING-GUIDE.md](TRAINING-GUIDE.md): the end-to-end training workbench
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
