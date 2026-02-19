# CLI Reference

ModelCypher CLI reference. Auto-generated from `mc --help` output.

Notes:
- Structured output goes to stdout (JSON by default). Logs and diagnostics go to stderr.
- In this repo, run the CLI as `poetry run mc …`. Examples below use `mc …` for brevity.
- Global options can appear anywhere on the command line (e.g. `mc model info … --pretty`).

## Global Options

All commands support these options:

| Option | Description |
|--------|-------------|
| `--ai` | AI mode: force JSON output, suppress prompts/logs |
| `--output json\|yaml\|text` | Output format (default: json) |
| `--text` | Shorthand for `--output text` |
| `--json`, `-j` | Shorthand for `--output json` |
| `--quiet`, `-q` | Suppress info logs |
| `--very-quiet`, `-qq` | Suppress all logs |
| `--yes`, `-y` | Auto-confirm prompts |
| `--no-prompt` | Fail if confirmation required |
| `--pretty`, `-p` | Pretty-print structured output |
| `--log-level` | Log level (default: info) |
| `--trace-id` | Custom trace ID for debugging |

---

## Command Groups

| Group | Purpose |
|-------|---------|
| [`mc train`](#mc-train) | Train LoRA adapters with geometry-derived hyperparameters |
| [`mc merge`](#mc-merge) | Geometric model merging via null-space projection |
| [`mc infer`](#mc-infer) | Run inference with optional adapter and security scanning |
| [`mc analyze`](#mc-analyze) | Model analysis: geometry, safety, entropy (35+ subcommands) |
| [`mc model`](#mc-model) | Model registry: inspect, search, quantize |
| [`mc system`](#mc-system) | System status, probes, benchmarks |
| [`mc adapter`](#mc-adapter) | LoRA adapter analysis and baseline calibration |

---

## mc train

Train LoRA adapters. Cayley-parameterized NB-LoRA with spectral bounds by construction.

**Group options:** `--agent`, `--model`, `--max-steps`, `--batch-size`, `--lr`, `--convergence`

### mc train run

Train NB-LoRA adapter from a text dataset. All hyperparameters derived from model geometry. Training stops when the data says to stop.

```bash
mc train run -m /path/to/model -d /path/to/data.jsonl
mc train run -m /path/to/model -d /path/to/data.jsonl -o /path/to/output --eval-data /path/to/eval.jsonl
```

| Option | Description |
|--------|-------------|
| `-m`, `--model` | Path to model directory (required) |
| `-d`, `--data` | Path to JSONL training dataset (required) |
| `-o`, `--output` | Output path for adapter |
| `--eval-data` | Held-out eval JSONL (default: 80/20 split) |
| `--max-iters` | Safety cap (default: 10000; geometry decides when to stop) |
| `--seq-length` | Max sequence length (default: 256) |
| `--lr` | Override geometry-derived learning rate |
| `--deep` | Target all layers (not just layers with tail_dims > 0) |
| `--safety-margin` | Fraction of sigma_k/2 for scale bound (default: 0.9) |
| `--seed` | Random seed (default: 42) |
| `--eval-batches` | Number of eval batches (default: 10) |
| `--adaptive-lr` / `--no-adaptive-lr` | Re-measure curvature per epoch and adapt LR (default: on) |
| `--lr-monotonic` / `--no-lr-monotonic` | Force LR to only decrease (default: off) |
| `--lipschitz-batches` | Batches for robust Lipschitz estimation (default: 3) |
| `--topo-monitor` / `--no-topo-monitor` | Track topological phase metrics per epoch (default: off) |
| `--dim-monitor` / `--no-dim-monitor` | Track dimensional expansion/contraction per epoch (default: off) |
| `--paired` / `--no-paired` | Experimental: constrained training with paired data |

### mc train status

Show training status for an agent. Displays buffer size, training progress, and merge history.

```bash
mc train status --agent agent-001 --model /path/to/model
```

| Option | Description |
|--------|-------------|
| `-a`, `--agent` | Agent ID for training state (required) |
| `-m`, `--model` | Path to model directory (required) |

### mc train merge

Merge LoRA adapters into base model weights via null-space projection.

```bash
mc train merge --agent agent-001 --model /path/to/model --save --output /path/to/merged
```

| Option | Description |
|--------|-------------|
| `-a`, `--agent` | Agent ID for training state (required) |
| `-m`, `--model` | Path to model directory (required) |
| `-o`, `--output` | Output path for merged model |
| `--save` | Save the merged model |
| `--reset` / `--no-reset` | Reset LoRA buffer after merge (default: reset) |

### mc train export

Export LoRA adapters to files.

```bash
mc train export --agent agent-001 --model /path/to/model --output /path/to/export
```

| Option | Description |
|--------|-------------|
| `-a`, `--agent` | Agent ID for training state (required) |
| `-m`, `--model` | Path to model directory (required) |
| `-o`, `--output` | Output path for exported LoRA (required) |

---

## mc merge

Geometric model merging. Takes knowledge from a source model and adds it to a target via null-space projection, preserving the target's existing capabilities.

### mc merge run

Merge source model into target. Pipeline: PROBE → DENSITY → TRANSPLANT.

```bash
mc merge run -s /path/to/source -t /path/to/target -o /path/to/output
```

| Option | Description |
|--------|-------------|
| `-s`, `--source` | Path to source model (required) |
| `-t`, `--target` | Path to target model (required) |
| `-o`, `--output` | Output directory for merged model (required) |

### mc merge batch

Merge multiple source models into one target (N→1). Target is loaded and probed once, then reused for all sources.

```bash
mc merge batch -s /path/src1 -s /path/src2 -t /path/target -o /path/out
```

| Option | Description |
|--------|-------------|
| `-s`, `--source` | Source model paths (repeatable, required) |
| `-t`, `--target` | Path to target model (required) |
| `-o`, `--output` | Output directory for merged model (required) |

---

## mc infer

Run inference with optional adapter loading and security scanning.

### mc infer run

Execute inference on a single prompt.

```bash
mc infer run --model /path/to/model --prompt "What is 2+2?"
mc infer run --model /path/to/model --adapter /path/to/adapter --prompt "Explain modus tollens."
mc infer run --model /path/to/model --prompt-file /path/to/prompt.txt --security-scan
```

| Option | Description |
|--------|-------------|
| `--model` | Model identifier or path (required) |
| `--prompt` | Input prompt |
| `--prompt-file` | Read prompt from a UTF-8 text file |
| `--prompt-stdin` | Read prompt from stdin (multi-line) |
| `--adapter` | Path to adapter directory |
| `--security-scan` | Perform dual-path security analysis |

### mc infer suite

Execute batched inference over a suite of prompts.

```bash
mc infer suite --model /path/to/model --suite /path/to/prompts.jsonl
mc infer suite --model /path/to/model --suite /path/to/prompts.txt --adapter /path/to/adapter
```

| Option | Description |
|--------|-------------|
| `--model` | Model identifier or path (required) |
| `--suite` | Path to suite file: `.txt`, `.json`, or `.jsonl` (required) |
| `--adapter` | Path to adapter directory |
| `--security-scan` | Perform security analysis |

---

## mc analyze

Model analysis commands covering geometry, safety, and entropy. This is the largest command group with 30+ subcommands.

### Geometry

#### mc analyze geodesic-compare

Compare geodesic trajectories across prompt categories.

#### mc analyze geodesic-profile

Profile geodesic deviation across all layers.

#### mc analyze geodesic-trajectory

Measure geodesic deviation of activation trajectories.

#### mc analyze concept-volume

Analyze concept volumes in activation space using Riemannian density estimation.

#### mc analyze dimension-profile

Compute per-layer intrinsic dimension profile.

#### mc analyze entropy-trajectory

Compute layer-wise entropy trajectory for a model.

#### mc analyze expansion-ratio

Compute per-prompt expansion ratio using TwoNN intrinsic dimension.

#### mc analyze reasoning-flow

Compute reasoning flow geometry (Zhou et al., ICLR 2026).

#### mc analyze spectral-trajectory

Compute per-layer spectral entropy profile.

#### mc analyze jacobian-trace

Compute Jacobian spectrum at each layer (Mathematical Anatomy).

#### mc analyze verification-depth-profile

Profile manifold observability across verification-depth levels.

### Safety

#### mc analyze adapter-probe

Probe adapter for delta-feature geometry.

#### mc analyze behavioral-signature

Compute behavioral signature for a model.

#### mc analyze cognitive-reflection-test

Run Cognitive Reflection Test (CRT) with geometric analysis.

#### mc analyze calibrate-safety

Calibrate safety thresholds from measured entropy on safe prompts.

#### mc analyze jailbreak-test

Execute jailbreak entropy analysis to test model safety boundaries.

#### mc analyze probe-redteam

Scan adapter metadata for threat indicators (static analysis).

#### mc analyze probe-behavioral

Run behavioral probes (requires inference hook for full analysis).

#### mc analyze bilm-probe-info

Show information about BiLM probe training.

### Benchmarks & Validation

#### mc analyze benchmark

Run benchmark suite with geometric metrics.

#### mc analyze reasoning-geometry-validation

Run cross-model validation of reasoning geometry signals.

```bash
mc analyze reasoning-geometry-validation --model LFM2-350M --benchmark arithmetic --samples 20 --output results/
```

### Adapter & Training Analysis

#### mc analyze lora-svd

Analyze LoRA adapter with SVD decomposition.

#### mc analyze sparse-region

Explore sparse activation regions and refusal directions.

#### mc analyze knowledge-type

Analyze whether a statement is factual knowledge or opinion.

#### mc analyze curriculum-profile

Profile training problems by geometric difficulty.

#### mc analyze circuit-breaker

Evaluate circuit breaker state.

#### mc analyze persona

Analyze persona drift for a training job.

### Entropy & Uncertainty

#### mc analyze uncertainty-modes

List available uncertainty response modes.

#### mc analyze entropy-pattern

Analyze entropy/variance samples for patterns.

#### mc analyze entropy-baseline-verify

Verify observed entropy deltas against declared baseline.

### Concept Response Matrix

#### mc analyze crm-build

Build Concept Response Matrix for a model.

#### mc analyze crm-compare

Compare two Concept Response Matrices.

---

## mc model

Model registry: inspect, register, search, and manage models.

### mc model list

List all registered models.

```bash
mc model list
```

### mc model add

Register a local model.

```bash
mc model add /path/to/model
```

### mc model delete

Delete a registered model.

```bash
mc model delete model-name
```

### mc model info

Inspect a model's architecture and configuration.

```bash
mc model info /path/to/model
```

| Option | Description |
|--------|-------------|
| `model_path` | Path to model directory (required, positional) |

### mc model capacity

Analyze per-layer spectral capacity and recommended LoRA ranks.

```bash
mc model capacity /path/to/model
mc model capacity /path/to/model --top 20
mc model capacity /path/to/model --sort-by recommended-rank
mc model capacity /path/to/model --target-modules q_proj --target-modules v_proj --min-dim 512
mc model capacity /path/to/model --emit-lora-config ./configs/lora_capacity.yaml
```

| Option | Description |
|--------|-------------|
| `model_path` | Path to model directory (required, positional) |
| `-n`, `--top` | Top N layers to show in text output (default: 10) |
| `--sort-by` | Sort key for top layers: `null`, `effective-rank`, `recommended-rank` |
| `-m`, `--target-modules` | Layer-name substring filter (repeatable or comma-separated) |
| `--min-dim` | Include only layers with `min(weight_shape) >= min_dim` |
| `--max-dim` | Include only layers with `min(weight_shape) <= max_dim` |
| `--emit-lora-config` | Write per-layer recommended ranks to YAML/JSON file |

### mc model search

Search for models in the registry.

```bash
mc model search "LFM2"
```

### mc model quantize

Quantize a model to reduce size. Supports 4-bit and 8-bit quantization.

```bash
mc model quantize /path/to/model /path/to/output --bits 4
mc model quantize /path/to/model /path/to/output --bits 8 --group-size 128
```

| Option | Description |
|--------|-------------|
| `model_path` | Path to model to quantize (required, positional) |
| `output_path` | Output path for quantized model (required, positional) |
| `-b`, `--bits` | Quantization bits: 4 or 8 (default: 4) |
| `-g`, `--group-size` | Quantization group size (default: 64) |
| `-m`, `--mode` | Quantization mode: affine, symmetric (default: affine) |
| `--overwrite` | Overwrite existing output |

---

## mc system

System status, diagnostics, and benchmarks.

### mc system status

Get system status including available backends and hardware.

```bash
mc system status
mc system status --require-backend mlx
```

| Option | Description |
|--------|-------------|
| `--require-backend` | Require a specific backend |

### mc system probe

Probe a system target for detailed diagnostics.

```bash
mc system probe backends
mc system probe memory
```

| Option | Description |
|--------|-------------|
| `target` | Probe target (required, positional): `backends`, `memory`, or a backend key |

### mc system test-cache

Test computation cache with real or synthetic model weights.

```bash
mc system test-cache
mc system test-cache /path/to/model --pairs 10
```

| Option | Description |
|--------|-------------|
| `model_path` | Path to model for real weight testing (optional, positional) |
| `-p`, `--pairs` | Number of layer pairs to test (default: 5) |

### mc system benchmark cache

Benchmark computation cache performance.

```bash
mc system benchmark cache
```

---

## mc adapter

LoRA adapter analysis and baseline calibration.

### mc adapter analyze

Compute geometry metrics for a LoRA adapter. Measures spectral selectivity (amplification CV) and Weyl utilization. Reports raw measurements and baseline-relative ratios.

```bash
mc adapter analyze /path/to/adapter
mc adapter analyze /path/to/adapter.safetensors -b /path/to/base_model
mc adapter analyze /path/to/adapter --baseline-artifact ./results/real_adapter_analysis/summary.json
```

| Option | Description |
|--------|-------------|
| `adapter_path` | Path to adapter weights (required, positional) |
| `-b`, `--base-model` | Path to base model (auto-detected from adapter_config.json) |
| `-o`, `--output` | Output format: text, json (default: text) |
| `--baseline-artifact` | Path to measured baseline artifact JSON |

### mc adapter calibrate-baseline

Calibrate synthetic random baseline from four-condition measurements. Promotes measured experiment output into production artifact format.

```bash
mc adapter calibrate-baseline
mc adapter calibrate-baseline \
    --four-condition-results ./results/four_condition/raw_measurements.json \
    --output-artifact ./results/real_adapter_analysis/summary.json
```

| Option | Description |
|--------|-------------|
| `--four-condition-results` | Path to four-condition raw measurements JSON |
| `-o`, `--output-artifact` | Path to output baseline artifact JSON |
| `--source` | Optional provenance label |
| `-f`, `--format` | Output format: text, json (default: text) |
