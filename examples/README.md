# ModelCypher Examples

This directory contains example scripts demonstrating common ModelCypher workflows.

## Prerequisites

- Python 3.11+
- ModelCypher installed (`poetry install`)
- Local model weights (fetch with `poetry run mc model fetch …`)

## Sandbox / Backend Notes

ModelCypher requires a GPU backend (MLX on macOS/Apple Silicon, CUDA on Linux/NVIDIA,
or JAX on supported platforms). If MLX fails to initialize inside a sandboxed host
(e.g. VSCode/Claude Code), run ModelCypher from Terminal.app (outside the host) so
Metal/MLX can initialize normally.

For commands that write registry/cache data, set locations inside the repo (useful in
sandboxes and CI):

```bash
MODELCYPHER_HOME=.claude/.modelcypher
HF_HOME=.claude/hf_home
```

## Examples

### 01. Basic Geometry Probe

Probe a model to inspect its geometric properties and layer structure.

```bash
poetry run python examples/01_basic_geometry_probe.py /path/to/model
```

**What it does:**
- Loads model weights
- Analyzes layer structure
- Reports parameter counts and hidden dimensions
- Reports attention head count and quantization (when available)

### 02. Safety Audit

Run safety probes and entropy diagnostics against adapter metadata and baselines.

```bash
poetry run python examples/02_safety_audit.py --name "adapter-name"
MC_ALLOW_STUB_EMBEDDINGS=1 poetry run python examples/02_safety_audit.py --name "adapter-name"
poetry run python examples/02_safety_audit.py --name "adapter-name" --baseline /path/to/baseline.json \
  --observed "[0.1, 0.12, 0.09]" --samples /path/to/samples.json
```

**What it does:**
- Static metadata scan (red team)
- Behavioral probe summary (raw findings only)
- Entropy baseline verification (raw comparison metrics)
- Entropy pattern analysis (trend, volatility, anomalies)

### 03. Adapter Inspection

Inspect LoRA adapters without blending or interpolation.

```bash
poetry run python examples/03_adapter_inspection.py /path/to/adapter_dir
```

**What it does:**
- Reports adapter rank, alpha, sparsity, and layer count
- Outputs raw measurements only

### 04. Entropy Analysis

Analyze entropy patterns in model outputs using thermodynamic metrics.

```bash
poetry run python examples/04_entropy_analysis.py /path/to/model --prompt "Your prompt here"
```

**What it does:**
- Measures entropy across linguistic modifiers
- Reports raw statistics (mean/std/min/max)
- Computes baseline vs intensity delta_h

This example requires a working GPU backend for real inference.

### 05. Model Merge

Merge two models using geometric alignment.

```bash
poetry run python examples/05_model_merge.py source_model target_model -o merged_output --dry-run
```

**What it does:**
- Runs pre-merge analysis, merge, and post-merge metrics
- Uses null-space transplant (no blending)
- Outputs raw geometry measurements

To run an actual merge, omit `--dry-run` (requires a working GPU backend + real model
weights).

## Getting Models

Fetch a tiny model from Hugging Face (network required):

```bash
MODELCYPHER_HOME=.claude/.modelcypher HF_HOME=.claude/hf_home poetry run mc model fetch hf-internal-testing/tiny-random-gpt2
```

Then probe using the returned `localPath`:

```bash
poetry run mc --output json model probe <localPath>
```

If you already have local weights, register them instead of downloading:

```bash
MODELCYPHER_HOME=.claude/.modelcypher poetry run mc model register my-model --path /path/to/model --architecture <architecture>
```

## Tips

1. **Start small**: Use smaller models (0.5B-3B) for faster iteration
2. **Check memory**: Run `poetry run mc system status --output json` before large operations
3. **Structured output**: Use `--output json` for machine-readable output (`poetry run mc --help`)
4. **Read the output**: Raw measurements are returned; you decide meaning based on context
