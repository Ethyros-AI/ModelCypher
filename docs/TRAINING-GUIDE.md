# Training Guide

ModelCypher's training surface is a workbench, not just a single command. The
workflow is:

`plan -> train -> evaluate -> compare -> export`

The user-facing value is straightforward: ModelCypher derives target modules,
ranks, stopping signals, and controller quantities from the model and data so
you do not hand-tune folklore hyperparameters.

## Current Reality (2026-03-16)

- `mc train run` is the shipped training path.
- Its control plane is geometry-derived.
- The repo has not yet closed a promotable head-to-head advantage over standard
  practice on real benchmark suites.
- That is a current limitation of a shipped tool, not a reason to pretend the
  workbench does not exist.

## Command Surface

Training-related commands available now:

- `mc train run`
- `mc train evaluate`
- `mc train compare`
- `mc train export`
- `mc train merge`
- `mc train status`
- `mc train validate-derived`
- `mc train star`

## Recommended Workflow

### 1. Derive The Plan First

```bash
poetry run mc train run \
  --model /path/to/model \
  --data /path/to/train.jsonl \
  --plan-only
```

This resolves the exact training plan without mutating model state. Use it to
see the derived surface before you commit to a run.

### 2. Run Training

```bash
poetry run mc train run \
  --model /path/to/model \
  --data /path/to/train.jsonl \
  --output /path/to/adapter
```

### 3. Evaluate The Adapter

```bash
poetry run mc train evaluate \
  --model /path/to/model \
  --adapter /path/to/adapter \
  --data /path/to/validation.jsonl
```

### 4. Compare Results

```bash
poetry run mc train compare \
  --model /path/to/model \
  --adapter-a /path/to/adapter_a \
  --adapter-b /path/to/adapter_b \
  --data /path/to/validation.jsonl
```

### 5. Export Or Merge Artifacts

```bash
poetry run mc train export \
  --agent agent-001 \
  --model /path/to/model \
  --output /path/to/export_dir
```

## Dataset Format

`mc train run` consumes JSONL with either:

- `{"text": "..."}`
- `{"messages": [{"role": "...", "content": "..."}]}`

Examples:

```json
{"text": "User: What is 2+2?\nAssistant: 4"}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

If your source data is not already JSONL, use:

```bash
poetry run mc data prepare /path/to/source --output /path/to/train.jsonl
```

## `mc train run`

Canonical geometry-derived LoRA training command. The goal is not to expose
more knobs. The goal is to derive the plan, show it when asked, execute it
without drift, and leave you with evidence about what happened.

```bash
poetry run mc train run \
  -m /path/to/model \
  -d /path/to/data.jsonl \
  -o /path/to/adapter \
  --eval-data /path/to/eval.jsonl
```

Options:

- `--model`, `-m` (required)
- `--data`, `-d` (required)
- `--output`, `-o`
- `--eval-data`
- `--benchmark`
- `--no-save`
- `--explain`
- `--plan-only`
- `--seq-length`
- `--seed`
- `--topo-monitor/--no-topo-monitor`
- `--dim-monitor/--no-dim-monitor`
- `--target-experts`
- `--entropy-reg/--no-entropy-reg`

### What ModelCypher Derives

The workbench derives or resolves these surfaces from model and data state:

- target modules
- per-module ranks
- sequence length when omitted
- controller quantities used during training
- stopping and verification surfaces
- seed and eval split defaults

The controller does not expose a fixed scalar learning rate. The key statement
is:

```text
eta_step = min(eta_ceiling, eta_sps, eta_weyl)
```

### Three Buckets In The Training Plan

`mc train run` surfaces training state in three buckets:

- `derived_now`
  Fixed before training starts: seed, output path, sequence length, eval split,
  target modules, per-module ranks, optimizer geometry config.
- `measured_during_training`
  Runtime controller quantities: `eta_ceiling`, `eta_sps`, `eta_weyl`,
  `eta_step`, gradient-noise-derived batch size, and stopping certificate
  signals.
- `verified_after_training`
  Post-training gates: spectral bounds, CKA, degeneration, pipeline gate, and
  optional benchmark delta when `--benchmark` is enabled.

### `--plan-only`

```bash
poetry run mc train run \
  -m /path/to/model \
  -d /path/to/data.jsonl \
  --plan-only
```

Use this when you want the exact resolved plan without injecting adapters or
creating output directories.

Example text output:

```text
Resolved training plan
Model: /path/to/model
Dataset: /path/to/data.jsonl
Eval: derived split (pilot_variance)
Seed: 123456789 (derived_from_model_dataset_hash)
Output: /path/to/adapters/model-geometric-lora-123456789
Seq length: 256 (data_derived_max_token_length)
Split: pilot_variance | train=480 eval=32
Target surface: 96 modules | ranks=4-16 | params~1,572,864
Spectral bounds: sigma_k_min=2.1e-02 | sigma_max=8.7e+00 | ceiling=RMT signal-rank
Controller: no fixed scalar LR; MASS will choose eta_step = min(eta_ceiling, eta_sps, eta_weyl) online
Measured during training: eta_sps, eta_weyl, eta_step, gradient-noise batch size, stopping certificate, preservation telemetry
Verified after training: spectral bounds, CKA, degeneration, pipeline gate, optional benchmark delta
Benchmark: opt-in only; add --benchmark quick for pre/post task scores
```

### `--explain`

```bash
poetry run mc train run \
  -m /path/to/model \
  -d /path/to/data.jsonl \
  --explain \
  --benchmark quick
```

This prints the resolved summary and then continues into training.

## `mc train evaluate`

Evaluate a trained adapter against the base model. The command supports three
modes; choose exactly one per run.

### Prompt comparison mode

```bash
poetry run mc train evaluate \
  -m /path/to/model \
  -a /path/to/adapter \
  --prompts /path/to/eval_prompts.jsonl
```

Use this when you want side-by-side generations on a prompt set.

### Dataset loss mode

```bash
poetry run mc train evaluate \
  -m /path/to/model \
  -a /path/to/adapter \
  -d /path/to/validation.jsonl
```

Use this when you want loss or perplexity style validation.

### Benchmark mode

```bash
poetry run mc train evaluate \
  -m /path/to/model \
  -a /path/to/adapter \
  --benchmark quick
```

Use this when you want lm-eval benchmark scores.

## `mc train compare`

Compare two training runs or two adapters side by side.

### Compare saved training results

```bash
poetry run mc train compare \
  --result-a /path/to/run_a.json \
  --result-b /path/to/run_b.json
```

### Compare two adapters on a dataset

```bash
poetry run mc train compare \
  -m /path/to/model \
  --adapter-a /path/to/adapter_a \
  --adapter-b /path/to/adapter_b \
  -d /path/to/validation.jsonl
```

Use this when you want a winner call backed by measured deltas instead of
impressionistic model sampling.

## `mc train export`

Export LoRA artifacts for downstream use.

```bash
poetry run mc train export \
  --agent agent-001 \
  --model /path/to/model \
  --output /path/to/export_dir
```

## `mc train merge`

Merge learned adapter state into base weights.

```bash
poetry run mc train merge \
  --agent agent-001 \
  --model /path/to/model \
  --save \
  --output /path/to/merged_model
```

## `mc train status`

Show current training state for a specific agent/model pair.

```bash
poetry run mc train status --agent agent-001 --model /path/to/model
```

## `mc train validate-derived`

Counterexample search for derived training. This is useful when you want to
stress the current control plane and capture failures systematically.

```bash
poetry run mc train validate-derived \
  -m /path/to/model \
  -d /path/to/data.jsonl \
  --trials 5 \
  --report-path /tmp/derived-validation.json
```

## `mc train star`

STaR loop support built on top of the training services:

```bash
poetry run mc train star \
  --model /path/to/model \
  --data /path/to/base_data.jsonl \
  --output /path/to/star_run \
  --rounds 3 \
  --problems-per-round 500
```

Treat this as an advanced workflow, not the default starting point.

## Monitoring And Diagnostics

If you want more visibility into model behavior after training:

```bash
poetry run mc analyze dimension-profile --model /path/to/model
poetry run mc analyze entropy-trajectory --model /path/to/model
poetry run mc analyze spectral-trajectory --model /path/to/model
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model
```

## Current Limitations

- The workbench is shipped, but benchmark superiority is still open.
- `--benchmark` is opt-in; it is available, but it is not yet the default path.
- Experimental surfaces like merge and STaR exist, but they are not the core
  promise of the product today.

## Live Signatures

If a command fails or you want the exact current signature:

```bash
poetry run mc train --help
poetry run mc train run --help
poetry run mc train evaluate --help
poetry run mc train compare --help
poetry run mc data --help
```
