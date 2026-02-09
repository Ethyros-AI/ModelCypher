# CLI Reference

ModelCypher CLI reference.

Notes:
- Structured output goes to stdout (JSON by default). Logs and diagnostics go to stderr.
- In this repo, run the CLI as `poetry run mc …`. Examples below use `mc …` for brevity.
- Global options can appear anywhere on the command line (e.g. `mc model info … --pretty`).

## Global Options

All commands support these options:

| Option | Description |
|--------|-------------|
| `--ai` | Enable AI-assisted mode |
| `--output json\|yaml\|text` | Output format (default: json) |
| `--quiet` | Suppress info logs |
| `--very-quiet` | Suppress all logs |
| `--yes` | Auto-confirm prompts |
| `--no-prompt` | Fail if confirmation required |
| `--pretty` | Pretty-print structured output |
| `--log-level` | Logging verbosity: trace, debug, info, warn, error |
| `--trace-id` | Custom trace ID for debugging |

---

## Model Merging

The primary operation. Takes knowledge from SOURCE and adds it to TARGET via null-space projection. Uses semantic concept probes from the atlas system to align manifolds - high-dimensional geometry has no tolerance for approximation.

### mc merge run (1→1)

```bash
mc merge run -s SOURCE -t TARGET -o OUTPUT_DIR

# Full example
mc merge run \
  -s /path/to/source \
  -t /path/to/target \
  -o /path/to/output_dir

# Paste-friendly (the CLI will prompt and accept -s/-t/-o lines)
mc merge run
-s /path/to/source
-t /path/to/target
-o /path/to/output_dir
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-s, --source` | path | Path to source model (knowledge donor) |
| `-t, --target` | path | Path to target model (receives knowledge) |
| `-o, --output-dir` | path | Output directory for merged model |
| `-f, --output-file` | path | Save full result to JSON file |
| `-n, --dry-run` | flag | Show what would happen without merging |

### mc merge batch (N→1)

```bash
mc merge batch -s MODEL1 -s MODEL2 -s MODEL3 -t TARGET -o OUTPUT_DIR
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-s, --source` | path | Source model paths (repeatable) |
| `-t, --target` | path | Target model (receives all knowledge) |
| `-o, --output-dir` | path | Output directory for merged model |
| `--accumulative/--sequential` | flag | Accumulative (default) vs sequential merging |
| `--fast/--precise` | flag | Fast mode (default) vs precision checks |

### mc merge multi-channel

Multi-modal merging via Birkhoff routing. Projects all channels into target's null-space simultaneously, then combines via doubly stochastic routing (spectral norm ≤ 1.0).

```bash
mc merge multi-channel -c spatial:/path/to/world -c text:/path/to/llm -t TARGET -o OUTPUT

# Example: merge world model + text model into unified model
mc merge multi-channel \
  -c spatial:./world-model \
  -c temporal:./video-model \
  -c text:./llm \
  -t ./lfm2 \
  -o ./merged
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `-c, --channel` | string | Channel in format `name:path` (repeatable) |
| `-t, --target` | path | Target model (receives all knowledge) |
| `-o, --output-dir` | path | Output directory for merged model |
| `-r, --routing` | string | Routing mode: `uniform` (default), `identity`, `diagonal_weighted` |
| `--fast/--precise` | flag | Fast mode (default) vs precision checks |

**Properties:**
- CKA = 1.0 per channel (geometry preserved)
- Spectral norm ≤ 1.0 (stable combination)
- No interference (channels add, not blend)

### mc merge bridge

Generate a cross-modal bridge between two encoders. Creates a linear transform that maps embeddings from source space to target space with CKA = 1.0.

```bash
mc merge bridge SOURCE TARGET -o OUTPUT

# Examples
mc merge bridge /path/to/clip /path/to/lfm2 -o clip_to_lfm2.safetensors
mc merge bridge ./model_a ./model_b -o bridge.safetensors --probe-sources semantic_prime,emotion_concept
mc merge bridge /path/to/whisper /path/to/lfm2 -o audio_bridge.safetensors --samples 200
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `SOURCE` | path | Path to source encoder (positional) |
| `TARGET` | path | Path to target encoder (positional) |
| `-o, --output` | path | Output path for bridge file (safetensors) |
| `-n, --samples` | int | Number of probe samples (default: 100) |
| `--probe-sources` | string | Comma-separated atlas sources (e.g., `semantic_prime,emotion_concept`) |
| `--source-name` | string | Optional name for source encoder |
| `--target-name` | string | Optional name for target encoder |

Uses 4596 semantic concept probes from the atlas system. These structured concepts span the semantic manifold systematically; Procrustes achieves CKA = 1.0 on training probes by construction, while holdout generalization depends on coverage. Approximations can introduce instability, so validate with evidence/coverage measurements.

**Available Atlas Sources:**
`semantic_prime`, `computational_gate`, `emotion_concept`, `temporal_concept`, `spatial_concept`, `social_concept`, `moral_concept`, `philosophical_concept`, `safety_ethics`, `physical_existence`, `compositional`, `conceptual_genealogy`, `metaphor_invariant`, `conceptual_metaphor`, `syntax_concept`, `perceptual`, `numeric`, `common_object`, `action_verb`, `abstract_relation`, `pronoun_perspective`, `prime_number`, `sequence_invariant`, `domain_specific`

**Output file contains:**
- Forward transform (source → target)
- Inverse transform (target → source)
- Scale ratio for magnitude normalization
- Metadata (dimensions, names, CKA achieved)

### mc merge apply-bridge

Apply a bridge transform to embeddings. Transforms embeddings from source space to target space (or vice versa).

```bash
mc merge apply-bridge BRIDGE_PATH INPUT_PATH -o OUTPUT

# Examples
mc merge apply-bridge clip_to_lfm2.safetensors image_embeds.npy -o lfm2_embeds.npy
mc merge apply-bridge bridge.safetensors source.safetensors -o target.safetensors
mc merge apply-bridge bridge.safetensors target_embeds.npy -o source_embeds.npy --inverse
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `BRIDGE_PATH` | path | Path to bridge file (positional) |
| `INPUT_PATH` | path | Path to input embeddings (positional) |
| `-o, --output` | path | Output path for transformed embeddings |
| `-i, --inverse` | flag | Apply inverse transform (target → source) |
| `--normalize/--no-normalize` | flag | Apply scale normalization (default: on) |

**Supported formats:** `.npy`, `.safetensors`

### mc merge deviation

Measure deviation from baseline (informational only). The geometry handles safety by construction via null-space projection.

```bash
mc merge deviation --baseline ./original --current ./merged
```

---

## Model Management

### mc model list
List all registered models.
```bash
mc model list
```

### mc model add
Add a model (fetch from Hub or register a local path) and persist its identity profile.
```bash
mc model add <repo_id|path>
mc model add LiquidAI/LFM2.5-1.2B-Instruct --alias lfm25
mc model add ./models/my-model --alias my-model
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--alias` | string | Alias for registration (defaults to repo or path name) |
| `--revision` | string | Git revision (default: main) |
| `--architecture` | string | Override architecture detection |
| `--parameters` | int | Parameter count (optional) |
| `--default-chat` | flag | Set as default chat model |

### mc model info
Inspect a model and surface its stored identity profile.
```bash
mc model info <model_path>
```

### mc model delete
Delete a registered model.
```bash
mc model delete <model_id>
mc model delete my-llama
```

### mc model search
Search for models on HuggingFace Hub.
```bash
mc model search <query>
mc model search llama --library backend --quant 4bit
mc model search --author <org> --sort downloads
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--author` | string | Filter by author |
| `--library` | string | Filter: backend, safetensors, any |
| `--quant` | string | Quantization: 4bit, 8bit, any |
| `--sort` | string | Sort by: downloads, likes, lastModified, trending |
| `--limit` | int | Results per page (default: 20) |
| `--cursor` | string | Pagination cursor |

### mc model quantize-sweep
Quantize a model across multiple bit widths with the backend and profile each variant.
```bash
mc model quantize-sweep /path/to/model --group-size 64
mc model quantize-sweep /path/to/model --group-size 64 --bits 8 --bits 4 --bits 2 -o ./quantized
mc model quantize-sweep /path/to/model --group-size 32 --mode mxfp4 --profile-base
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--output-dir` | path | Output directory for quantized models |
| `--bits` | int | Bit widths to attempt (repeatable) |
| `--group-size` | int | Quantization group size (required unless config.json has one) |
| `--mode` | string | Quantization mode passed to the backend |
| `--profile/--no-profile` | flag | Profile each quantized model after quantization |
| `--profile-base/--no-profile-base` | flag | Profile the full-precision model before the sweep |
| `--overwrite` | flag | Overwrite existing quantized weights |
| `--force-profile` | flag | Recompute profiles even if cached |
| `--max-batches` | int | Max batches for profiling (None = saturation) |

### mc model validate-merge
Validate merge alignment between two models.
```bash
mc model validate-merge --source ./model-a --target ./model-b
```

**Output fields:**
- `lowEffort`, `architectureMatch`, `vocabMatch`, `dimensionMatch`, `warnings`

### mc model validate-knowledge
Validate knowledge transfer in merged model.
```bash
mc model validate-knowledge --merged ./merged-model
mc model validate-knowledge --merged ./merged-model --source ./source-model
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--merged` | path | Path to merged model |
| `--source` | path | Path to source model (for baseline) |
| `--report-path` | path | Save validation report |

### mc model analyze-alignment
Analyze alignment drift between two models.
```bash
mc model analyze-alignment --model-a ./base-model --model-b ./fine-tuned
```

**Output fields:**
- `driftMagnitude`, `driftStd`, `driftMin`, `driftMax`, `driftP50`, `driftP90`
- `commonLayerCount`, `comparableLayerCount`, `missingLayerCount`, `layerDrifts`

### mc model vocab-compare
Compare vocabularies between two models for cross-vocabulary merging.
```bash
mc model vocab-compare --model-a ./llama-3-8b --model-b ./qwen-2-7b
```

### mc model fingerprint
Compute geometric fingerprint metrics from norm trajectories across a small set of task probes.
Reports expansion ratio statistics.

```bash
mc model fingerprint /path/to/model
mc model fingerprint /path/to/model --pretty
```

**Output fields:**
| Field | Description |
|-------|-------------|
| `metrics.expansion_ratio_*` | Mean/variance/std/min/max expansion ratio across tasks |
| `task_breakdown` | Per-task expansion_ratio, peak_norm, final_norm |

### mc model weight-analysis
Analyze weight matrix properties (effective rank, sparsity, singular value distribution).

```bash
mc model weight-analysis /path/to/model                    # Final layer only
mc model weight-analysis /path/to/model --layers all       # All layers
mc model weight-analysis /path/to/model --layers 20,21,22  # Specific layers
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--layers` | string | Which layers: `final` (default), `all`, or comma-separated indices |

**Output fields:**
| Field | Description |
|-------|-------------|
| `mean_sparsity` | Average fraction of near-zero weights |
| `mean_effective_rank` | Average effective rank via participation ratio |
| `layers` | Per-layer, per-matrix breakdown |

Specialist models typically show higher sparsity (~40%) and lower effective rank than general models (~13%).

---

## Training

### mc train start
Start a training job.
```bash
mc train start --model <model> --dataset <dataset> --out <output>
mc train start --model meta-llama/Llama-2-7b --dataset ./data.jsonl --out ./output
```

**Required options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Model identifier or path |
| `--dataset` | path | Path to dataset file |
| `--out` | path | Output directory |

Training hyperparameters are derived from model/dataset geometry and precision (no user knobs).

**Control options:**
| Option | Type | Description |
|--------|------|-------------|
| `--resume-from` | path | Resume from checkpoint |
| `--detach` | flag | Run in background |
| `--stream` | flag | Stream training events |

### mc train preflight
Run preflight checks before training.
```bash
mc train preflight --model <model> --dataset <dataset> --out <output>
```

### mc train status
Get training job status.
```bash
mc train status <job_id>
mc train status abc123 --follow
mc train status abc123 --stream
```

### mc train pause
Pause a training job.
```bash
mc train pause <job_id>
```

### mc train resume
Resume a paused training job.
```bash
mc train resume <job_id>
```

### mc train cancel
Cancel a training job.
```bash
mc train cancel <job_id>
```

### mc train export
Export a trained model or job.
```bash
mc train export --model ./fine-tuned --format gguf --output-path ./model.gguf
mc train export --job abc123 --format safetensors --output-path ./model
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model (exclusive with --job) |
| `--job` | string | Job ID (exclusive with --model) |
| `--format` | string | Export format: safetensors, gguf |
| `--output-path` | path | Output path |

### mc train logs
View training logs.
```bash
mc train logs <job_id>
mc train logs abc123 --tail 50 --follow
```

---

## Job Management

### mc job list
List all jobs.
```bash
mc job list
mc job list --status running
mc job list --model my-model
mc job list --active-only
```

### mc job show
Show job details.
```bash
mc job show <job_id>
mc job show abc123 --loss-history
```

### mc job attach
Attach to a running job's output stream.
```bash
mc job attach <job_id>
mc job attach abc123 --replay --since 2024-01-01T00:00:00
```

### mc job delete
Delete a job.
```bash
mc job delete <job_id>
```

---

## Checkpoint Management

### mc checkpoint list
List checkpoints.
```bash
mc checkpoint list
mc checkpoint list --job abc123
```

### mc checkpoint delete
Delete a checkpoint.
```bash
mc checkpoint delete <path>
mc checkpoint delete ./checkpoints/step-1000 --force
```

### mc checkpoint export
Export a checkpoint.
```bash
mc checkpoint export <checkpoint_path> --format safetensors --output-path ./model
```

---

## Evaluation

### mc eval list
List all evaluations.
```bash
mc eval list
mc eval list --limit 10
```

### mc eval show
Show evaluation details.
```bash
mc eval show <eval_id>
```

### mc eval run
Execute evaluation on model with dataset.
```bash
mc eval run --model ./model --dataset ./data.jsonl
```

**Output fields:**
- `evalId`, `modelPath`, `datasetPath`, `averageLoss`, `perplexity`, `sampleCount`

### mc eval benchmark
Run lm-eval-harness benchmarks on a backend model.
```bash
mc eval benchmark --model ./model --tasks gsm8k,hellaswag
mc eval benchmark --model ./model --tasks mmlu
mc eval benchmark --model ./model --tasks arc_challenge --output-path ./results.json
```

### mc eval domain
Run domain-specific benchmarks mapped to industry standards.
```bash
mc eval domain --model ./model --domain computational
mc eval domain --model ./model --domain mathematical --domain logical
mc eval domain --model ./model --suite standard
mc eval domain --model ./model --suite comprehensive
```

**Domain mappings:**
- `computational`: HumanEval, MBPP (code generation)
- `mathematical`: GSM8K, MathQA (math reasoning)
- `logical`: ARC-Challenge, LogiQA (logical reasoning)
- `linguistic`: HellaSwag, LAMBADA (language understanding)
- `relational`: WinoGrande (coreference/relations)
- `moral`: TruthfulQA (ethics/truthfulness)

### mc eval batch
Run benchmarks on multiple models sequentially.
```bash
mc eval batch -m ./model1 -m ./model2 -m ./model3 --suite standard
mc eval batch -m ./model1 -m ./model2 --suite quick --output-dir ./results
```

---

## Benchmarking (Geometric)

### mc benchmark list
List available benchmark suites.
```bash
mc benchmark list
```

### mc benchmark run
Run benchmark suites with geometric metrics.
```bash
mc benchmark run --model /path/to/model --suite quick
mc benchmark run --model /path/to/model --adapter /path/to/adapter --suite comprehensive
mc benchmark run --model /path/to/model --suite comprehensive --results-path ./results.json
```

### mc benchmark analyze
Summarize failures from a benchmark run.
```bash
mc benchmark analyze --failures-path ./failures.jsonl
mc benchmark analyze --failures-path ./failures.jsonl --benchmark gsm8k
```

### mc benchmark export-curriculum
Export benchmark failures to a curriculum JSONL (prompt/completion format).
```bash
mc benchmark export-curriculum --failures-path ./failures.jsonl --output-path ./phase5_failures.jsonl
mc benchmark export-curriculum --failures-path ./failures.jsonl --output-path ./gsm8k_failures.jsonl --benchmark gsm8k
mc benchmark export-curriculum --failures-path ./failures.jsonl --output-path ./phase5_failures.jsonl --with-metadata
```

---

## Comparison

### mc compare list
List all comparison sessions.
```bash
mc compare list
mc compare list --status completed --limit 10
```

### mc compare show
Show comparison session details.
```bash
mc compare show <session_id>
```

### mc compare run
Execute A/B comparison between checkpoints.
```bash
mc compare run --checkpoint ./ckpt1 --checkpoint ./ckpt2
mc compare run --checkpoint ./ckpt1 --checkpoint ./ckpt2 --prompt "Test prompt"
```

### mc compare checkpoints
Compare checkpoints for a job.
```bash
mc compare checkpoints <job_id>
```

### mc compare baseline
Establish baseline metrics for comparison.
```bash
mc compare baseline --model ./model
```

### mc compare score
Get aggregated comparison scores.
```bash
mc compare score <comparison_id>
```

---

## Inference

### mc infer run
Execute inference with optional adapter and security scanning.
```bash
mc infer run --model ./model --prompt "Hello, how are you?"
mc infer run --model ./model --prompt "Test" --adapter ./my-adapter
mc infer run --model ./model --prompt "Test" --security-scan
mc infer run --model ./model --prompt-file ./prompt.txt
cat ./prompt.txt | mc infer run --model ./model --prompt-stdin
```

**Output fields:**
- `model`, `prompt`, `response`, `tokenCount`, `tokensPerSecond`
- `timeToFirstToken`, `totalDuration`, `stopReason`, `adapter`
- `security` (if --security-scan): `anomalyCount`, `maxAnomalyScore`, `avgDelta`, `disagreementRate`

### mc infer suite
Execute batched inference over a suite of prompts.
```bash
mc infer suite --model ./model --suite ./suite.jsonl
mc infer suite --model ./model --suite ./suite.txt --adapter ./adapter --security-scan
```

---

## Genesis

Continual-learning workflow with geometric diagnostics.

### mc genesis run
Run genesis and report representational preservation via per-layer CKA (baseline vs post-genesis).
```bash
mc genesis run --model ./model --prompt "What is geometric learning?"

# Use a fixed probe set for reproducibility
mc genesis run --model ./model --prompts ./prompts.txt --cka-probes ./probes.txt

# Control: identity save/load noise floor (no learning)
mc genesis run --model ./model --prompt "test" --cka-control save-load

# Optional: geodesic RBF CKA (more expensive)
mc genesis run --model ./model --prompt "test" --cka-kernel rbf
```

**CKA options:**
- `--cka-kernel linear|rbf`: default `linear`
- `--cka-probes <file>`: one probe per line; blank lines and `#` comments are ignored
- `--cka-control none|save-load`: default `none` (use `save-load` to measure the pipeline’s identity noise floor)

**Output fields:**
- `genesis.cka_preserved`: scalar preservation summary (worst-case layer CKA; falls back to `capacity_remaining` only if CKA cannot be computed)
- `cka.cka_per_layer`, `cka.cka_min`, `cka.cka_mean`, `cka.layers_compared`, `cka.probe_count`, `cka.probes`
- `cka.control` (when enabled): per-layer CKA for the identity roundtrip

### mc genesis validate
Run behavioral canaries and (optionally) compare per-layer CKA against a reference model.
```bash
mc genesis validate --model ./genesis-v1
mc genesis validate --model ./genesis-v1 --reference ./original --cka-kernel linear
mc genesis validate --model ./genesis-v1 --reference ./original --cka-probes ./probes.txt
```

**Output fields:**
- `canary_tests` and `canary_details`
- `cka_comparison` (when `--reference` provided): `cka_per_layer`, `cka_min`, `cka_mean`, `layers_compared`, `probe_count`, `probes`, `kernel`

### mc genesis status
Inspect whether a model directory contains `genesis_metadata.json` and show stored run metadata.
```bash
mc genesis status --model ./genesis-v1
```

**Output fields:**
- `genesis_metadata` (raw metadata saved at run time)
- `cka_summary` (when present): `cka_min`, `cka_mean`, `kernel`, `probe_count`, `layers_compared`, and `control.status`

---

## Continual Learning

Commands for continual learning, manifold consolidation, and LoRA memory management.

### mc learn consolidate
Run manifold consolidation on a model. Fills in sparse regions of the model's representational manifold, making it denser and more robust.

```bash
mc learn consolidate --model /path/to/model
mc learn consolidate --model /path/to/model --session ./session.json
mc learn consolidate --model /path/to/model --save --output /path/to/output
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-m, --model` | path | Path to model directory (required) |
| `-s, --session` | path | Path to session file with sparsity events (JSON) |
| `--max-steps` | int | Maximum consolidation steps (default: 50) |
| `--max-probes` | int | Maximum probe embeddings to generate (default: 100) |
| `--save` | flag | Save consolidated model weights |
| `-o, --output` | path | Output path for consolidated model |

### mc learn status
Show null-space capacity and consolidation status for a model.

```bash
mc learn status --model /path/to/model
```

**Output fields:**
- Per-layer statistics on used vs available dimensions

### mc learn null-space
Analyze null-space availability in a model.

```bash
mc learn null-space --model /path/to/model
mc learn null-space --model /path/to/model --layer 16
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-m, --model` | path | Path to model directory (required) |
| `-l, --layer` | int | Specific layer to inspect (default: all) |
| `-n, --samples` | int | Number of random samples for estimation (default: 100) |

### mc learn lora-status
Show LoRA memory status for an agent.

```bash
mc learn lora-status --agent agent-001 --model /path/to/model
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-a, --agent` | string | Agent ID for LoRA memory store (required) |
| `-m, --model` | path | Path to model directory (required) |

### mc learn lora-train
Train LoRA adapters from accumulated events. This is the "dreaming" phase of two-tier memory.

```bash
mc learn lora-train --agent agent-001 --model /path/to/model
mc learn lora-train --agent agent-001 --model /path/to/model --lr 1e-5 --max-steps 50
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-a, --agent` | string | Agent ID for LoRA memory store (required) |
| `-m, --model` | path | Path to model directory (required) |
| `--max-steps` | int | Maximum training steps (default: derived from buffer size) |
| `--batch-size` | int | Batch size per step (default: full buffer) |
| `--lr` | float | Learning rate (default: derived from model geometry) |
| `--convergence` | float | Loss threshold for early stopping (default: sqrt(eps)) |

Hyperparameters are derived from model geometry when not provided.

### mc learn merge-lora
Merge LoRA adapters into base model weights. This is the "sleep consolidation" phase - transferring hippocampus (LoRA) knowledge to neocortex (base weights) via null-space projection.

```bash
mc learn merge-lora --agent agent-001 --model /path/to/model --save --output /path/to/merged
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-a, --agent` | string | Agent ID for LoRA memory store (required) |
| `-m, --model` | path | Path to model directory (required) |
| `-o, --output` | path | Output path for merged model |
| `--save` | flag | Save the merged model |
| `--reset/--no-reset` | flag | Reset LoRA buffer after merge (default: reset) |

### mc learn lora-export
Export LoRA adapters to files for sharing or backup.

```bash
mc learn lora-export --agent agent-001 --model /path/to/model --output /path/to/export
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-a, --agent` | string | Agent ID for LoRA memory store (required) |
| `-m, --model` | path | Path to model directory (required) |
| `-o, --output` | path | Output path for exported LoRA (required) |

### mc learn benchmark
Capture geometric snapshots and compare before/after consolidation.

```bash
# Capture 'before' snapshot
mc learn benchmark --model /path/to/model --capture --output before.json

# Capture 'after' and compare
mc learn benchmark --model /path/to/model --before before.json --output results.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-m, --model` | path | Path to model directory (required) |
| `--capture` | flag | Capture a new snapshot (vs compare) |
| `-b, --before` | path | Path to 'before' snapshot for comparison |
| `-o, --output` | path | Output path for snapshot or comparison result |
| `-p, --probes` | string | Comma-separated probe prompts for entropy measurement |

**Effectiveness metrics:**
- `delta_sparsity < 0`: Sparse regions became dense
- `delta_intrinsic_dim > 0`: Denser manifold uses more dimensions
- `delta_eigenscore < 0`: Less geometric uncertainty
- `delta_entropy < 0`: More confident on uncertain prompts

### mc learn monitor
Monitor geometric conditions for background consolidation.

```bash
mc learn monitor --model /path/to/model --status
mc learn monitor --model /path/to/model --auto
mc learn monitor --model /path/to/model --auto --interval 60
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-m, --model` | path | Path to model directory (required) |
| `--interval` | float | Seconds between condition checks (default: 30.0) |
| `--max-queue` | int | Max sparsity events before forced consolidation (default: 1000) |
| `--auto` | flag | Enable automatic consolidation when conditions met |
| `--status` | flag | Show current geometric conditions and exit |

Consolidation triggers are geometry-based, not time-based.

---

## Curiosity

Commands for curiosity policy, active exploration, and Expected Free Energy (EFE) scoring.

### mc curiosity status
Show curiosity policy status for a model.

```bash
mc curiosity status --model /path/to/model
```

Returns EFE-derived thresholds and current exploration state. All values derived from sqrt(eps).

### mc curiosity weights
Compute geometry-derived acquisition weights.

```bash
mc curiosity weights --model /path/to/model --activations ./corpus.safetensors
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--activations` | path | Path to activation corpus (safetensors or numpy) |

Returns composite acquisition weights derived from coverage_radius and mean_local_id.

### mc curiosity analyze
Analyze candidates using composite acquisition.

```bash
mc curiosity analyze --model /path/to/model \
    --candidates ./candidates.safetensors \
    --corpus ./corpus.safetensors
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--candidates` | path | Path to candidate activations (required) |
| `--corpus` | path | Path to corpus activations (required) |
| `--top-k` | int | Number of top candidates to show (default: 10) |

Computes acquisition scores combining coreset, coverage, and density contributions.

### mc curiosity evaluate
Evaluate EFE scores for a probe candidate.

```bash
mc curiosity evaluate --eigenscore 0.7 --capacity 0.5
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--eigenscore` | float | Manifold sparsity [0, 1] (required) |
| `--capacity` | float | Null-space capacity [0, 1] (required) |

**Output fields:**
- Epistemic value = eigenscore × capacity_fraction
- EFE = risk + ambiguity
- Recommended action

---

## Stacked LoRA Self-Improvement

Commands for iterative self-improvement with stacked LoRA adapters.

### mc stack init
Initialize a new LoRA stack for a base model.

```bash
mc stack init /path/to/model
mc stack init /path/to/model --state ./stack_state.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-s, --state` | path | Path to save stack state (default: auto-generated) |

### mc stack status
View status of a LoRA stack.

```bash
mc stack status ./stack_state.json
```

### mc stack train
Train a LoRA adapter and add to stack.

```bash
mc stack train ./stack_state.json --data ./data.jsonl --output ./adapter1
mc stack train ./stack_state.json -d ./data.jsonl -o ./adapter1 --epochs 5
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-d, --data` | path | Path to training data (required) |
| `-o, --output` | path | Output directory for adapter (required) |
| `-e, --epochs` | int | Training epochs (default: 3) |
| `-r, --rank` | int | LoRA rank (default: 8) |

### mc stack merge
Merge all adapters in stack into a single adapter.

```bash
mc stack merge ./stack_state.json --output ./merged_adapter
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-o, --output` | path | Output path for merged adapter (required) |

### mc stack improve
Run iterative self-improvement loop.

```bash
mc stack improve /path/to/model --output ./improvement
mc stack improve /path/to/model -o ./improvement --rounds 10
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-o, --output` | path | Output directory (required) |
| `-n, --rounds` | int | Max improvement rounds (default: 5) |
| `--samples` | int | Training samples per round (default: 100) |

The loop:
1. Scan for capability gaps
2. Generate training data
3. Train LoRA adapter
4. Check geometry, stack or merge
5. Repeat

### mc stack profile
Profile problems geometrically for curriculum design.

```bash
mc stack profile /path/to/model --problems ./questions.txt
mc stack profile /path/to/model -p ./questions.txt -o ./profiles.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-p, --problems` | path | Path to problems file (one per line, required) |
| `-o, --output` | path | Output JSON file for profiles |
| `-l, --layer` | int | Layer index to profile (default: middle layer) |

Measures difficulty using CKA, barrier, curvature, density, and intrinsic dimension.

### mc stack select
Select training curriculum based on geometric difficulty.

```bash
mc stack select /path/to/model -p ./all_problems.txt -o ./curriculum.txt -n 50
mc stack select /path/to/model -p ./problems.txt -o ./hard.txt -s hardest -n 20
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-p, --problems` | path | Path to problems file (one per line, required) |
| `-o, --output` | path | Output file for selected curriculum (required) |
| `-n` | int | Number of samples to select (default: 100) |
| `-s, --strategy` | string | Selection strategy (default: balanced) |
| `-l, --layer` | int | Layer index to profile (default: middle layer) |

**Strategies:**
- `balanced`: Mix of easy (20%), medium (60%), hard (20%)
- `hardest`: Focus on highest difficulty problems
- `goldilocks`: Moderate difficulty only (score 0.3-0.7)
- `highway_first`: Order by intrinsic dimension (low ID first)

---

## Agent Evaluation

Commands for agent evaluation runs.

### mc agent-eval run
Execute agent evaluation.

```bash
mc agent-eval run --model /path/to/model
mc agent-eval run --model /path/to/model --suite default --max-turns 10
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--suite` | string | Evaluation suite (default: default) |
| `--max-turns` | int | Max conversation turns (default: 10) |
| `--timeout` | int | Timeout in seconds (default: 300) |
| `--seed` | int | Random seed |

### mc agent-eval results
Get agent evaluation results.

```bash
mc agent-eval results <eval_id>
```

---

## Research Experiments

Experimental commands for research.

### mc research sparse-region
Analyze sparse activation regions in a model.

```bash
mc research sparse-region /path/to/model
```

### mc research multimodal-merge
Merge multi-modal knowledge (CLIP, Whisper) into an LLM.

```bash
mc research multimodal-merge /path/to/llm
mc research multimodal-merge /path/to/llm --concepts ./concepts.json
mc research multimodal-merge /path/to/llm --no-whisper -o results.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-c, --concepts` | path | JSON file with concept list |
| `--clip/--no-clip` | flag | Include CLIP vision knowledge (default: on) |
| `--whisper/--no-whisper` | flag | Include Whisper audio knowledge (default: on) |
| `-o, --output` | path | Output JSON file for results |

### mc research multimodal-offramp
Create multi-modal offramp projections for inference-time knowledge access.

```bash
mc research multimodal-offramp /path/to/llm
mc research multimodal-offramp /path/to/llm -o ./offramps
mc research multimodal-offramp /path/to/llm --no-whisper
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-o, --output` | path | Output directory for offramp weights |
| `--clip/--no-clip` | flag | Include CLIP vision offramp (default: on) |
| `--whisper/--no-whisper` | flag | Include Whisper audio offramp (default: on) |

Creates bidirectional projection matrices ("offramps") for multimodal access during inference.

### mc research memory-token
Create memory token for attention-based multimodal injection.

```bash
mc research memory-token /path/to/llm --concept "bright red apple"
mc research memory-token /path/to/llm -c "blue ocean" --arch LFM2
mc research memory-token /path/to/llm -c "golden sunset" -o memory.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-c, --concept` | string | Source concept to inject (required) |
| `-n, --neutral` | string | Neutral reference concept (default: thing) |
| `--arch` | string | Architecture name (e.g., LFM2) |
| `-o, --output` | path | Output JSON file |

Memory tokens allow 10x higher scale tolerance than direct injection.

### mc research afm
Run activation function mapping analysis.

```bash
mc research afm /path/to/model
```

### mc research taxonomy
Research taxonomy commands.

---

## System

### mc system status
Get system status.
```bash
mc system status
mc system status --require-backend <backend-key>
```

**Output fields:**
- `backends`, `backendVersions`, `resources`, `preferredBackend`, `readinessScore`

### mc system probe
Probe a system target.
```bash
mc system probe backends
mc system probe memory
mc system probe <backend-key>
```

### mc system benchmark cache
Benchmark computation cache performance.
```bash
mc system benchmark cache
```

**Output fields:**
- `backend`, `benchmarks[]`: coldTimeMs, warmTimeMs, speedup
- `cacheStats`: totalHits, totalMisses, hitRate, evictions, computeTimeSavedMs

### mc system test-cache
Test computation cache with real or synthetic model weights.
```bash
mc system test-cache
mc system test-cache /path/to/model --pairs 10
```

---

## Storage

### mc storage status
Return storage usage breakdown by category.
```bash
mc storage status
```

**Output fields:**
- `totalGb`, `modelsGb`, `checkpointsGb`, `otherGb`, `disk.totalBytes`, `disk.freeBytes`

### mc storage usage
Alias for `storage status`.

### mc storage cleanup
Remove old artifacts and return freed space.
```bash
mc storage cleanup --target caches --target rag
mc storage cleanup --target caches --dry-run
mc storage cleanup --target caches --force
```

---

## Entropy Analysis

All entropy commands return raw statistics - no hardcoded thresholds.

### mc entropy analyze
Analyze entropy/variance samples.
```bash
mc entropy analyze '[[3.5, 0.2], [3.6, 0.1], [4.8, 0.5]]'
```

**Output fields:**
- `entropyMean`, `entropyStdDev`, `entropyMin`, `entropyMax`
- `varianceMean`, `varianceStdDev`, `trendSlope`, `zScores`

### mc entropy detect-distress
Analyze entropy/variance samples for distress indicators.
```bash
mc entropy detect-distress '[[3.5, 0.2], [3.6, 0.1], [4.8, 0.5]]'
```

**Output fields:**
- Distribution statistics plus `correlation`, `volatility`

### mc entropy verify-baseline
Compare observed entropy deltas against declared baseline.
```bash
mc entropy verify-baseline --baseline ./calibration.json --observed '[0.1, 0.15, 0.12]'
```
Baseline files are produced by `mc entropy calibrate`.

### mc entropy window
Analyze entropy using a sliding window.
```bash
mc entropy window '[[3.5, 0.2], [3.6, 0.1]]'
```

### mc entropy conversation-track
Analyze entropy patterns across a conversation session.
```bash
mc entropy conversation-track --session ./session.json
```

### mc entropy dual-path
Analyze entropy divergence between base model and adapter.
```bash
mc entropy dual-path '[{"base": [3.5, 0.2], "adapter": [3.8, 0.3]}]'
```

### mc entropy calibrate
Calibrate entropy baselines (derived thresholds) by measuring actual model distributions.
```bash
mc entropy calibrate --model /path/to/model --prompts ./prompts.json
mc entropy calibrate --model /path/to/model --prompts ./prompts.json --output-file ./calibration.json
```

---

## Safety

Geometric probes for model analysis and fingerprinting.

### mc safety spectral-trajectory
Compute per-layer spectral entropy profile (expand-compress detection).
```bash
mc safety spectral-trajectory --model ./my-model
mc safety spectral-trajectory --model ./my-model --samples 100
mc safety spectral-trajectory --model ./my-model --probes ./prompts.txt
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--probes` | path | Path to file with probe texts (one per line) |
| `--samples` | int | Number of probe samples (default: 50) |

Computes spectral entropy from SVD singular values at each layer. High entropy = expansion (variance spread), low entropy = compression (variance concentrated).

### mc safety dimension-profile
Compute per-layer intrinsic dimension profile (semantic highway detection).
```bash
mc safety dimension-profile --model ./my-model
mc safety dimension-profile --model ./my-model --samples 100
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--probes` | path | Path to file with probe texts (one per line) |
| `--samples` | int | Number of probe samples (default: 50) |

Uses TwoNN estimator to measure intrinsic dimensionality at each layer. Reveals the "semantic highway" - a low-dimensional bottleneck in middle layers.

### mc analyze verification-depth-profile
Profile manifold observability across explicit verification-depth levels (analyze-only).
```bash
mc analyze verification-depth-profile --model ./my-model
mc analyze verification-depth-profile --model ./my-model --levels 0,1,2,3,4
mc analyze verification-depth-profile --model ./my-model --mode exact --max-probes-per-level 200
mc analyze verification-depth-profile --model ./my-model --probes ./data/probes/deep_reasoning.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--levels` | csv[int] | Optional level set (defaults to discovered levels) |
| `--mode` | enum | `cumulative` (depth <= level) or `exact` (depth == level), default `cumulative` |
| `--max-probes-per-level` | int | Optional cap per level |
| `--batch-size` | int | Probe batch size for trajectory collection (default: 20) |
| `--probes` | path | Optional probe JSON file override |

Reports raw per-layer metrics per level:
`activation_rank`, `trajectory_rank`, `intrinsic_dimension`, `condition_number`,
`d_plus_1_minimum`, `d_plus_1_gap`, `coverage_ratio_probe`,
`coverage_ratio_trajectory`, and `null_rank`.

### mc safety comp-phi
Compute per-prompt expansion_ratio using TwoNN intrinsic dimension.
```bash
mc safety comp-phi --model ./my-model --prompt "What is 2+2?"
mc safety comp-phi --model ./my-model --probes ./prompts.txt --trajectory
mc safety comp-phi --model ./my-model --prompt "Test" --quiet
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--prompt` | string | Single prompt to analyze |
| `--probes` | path | Path to file with prompts (one per line) |
| `--trajectory` | flag | Show per-layer intrinsic dimension trajectory |
| `--quiet` | flag | Only output the expansion_ratio(s) |

Measures the geometric expansion/compression cycle. The meaningful metric is the raw expansion_ratio = peak_dim / final_dim. Values near 1.0 indicate flat trajectories (specialist models).

### mc safety entropy-trajectory
Compute layer-wise entropy trajectory for a model.
```bash
mc safety entropy-trajectory --model ./my-model
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory (required) |
| `--probes` | path | Path to file with probe texts |
| `--samples` | int | Number of probe samples |

### mc safety behavioral-signature
Compute behavioral signature for a model.
```bash
mc safety behavioral-signature --model ./my-model
```

### mc safety reasoning-flow
Compute reasoning flow geometry (Zhou et al., ICLR 2026).
```bash
mc safety reasoning-flow --model ./my-model
```

### mc safety cognitive-reflection-test
Run Cognitive Reflection Test (CRT) with geometric analysis.
```bash
mc safety cognitive-reflection-test --model ./my-model
```

### mc safety adapter-probe
Probe adapter for delta-feature geometry.
```bash
mc safety adapter-probe --adapter ./my-adapter
mc safety adapter-probe --adapter ./my-adapter --base-model ./base
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--adapter` | path | Path to adapter directory |
| `--base-model` | path | Path to base model (optional) |

**Output fields:**
- `layerCount`, `outlierLayerCount`, `outlierLayerIndices`
- `maxGeodesicSpread`, `meanGeodesicSpread`, `meanSparsity`, `geodesicSpreads`, `sparsity`

### mc safety calibrate-safety
Measure entropy calibration samples for safety testing.
```bash
mc safety calibrate-safety --model ./model --prompts ./safe_prompts.json --output-file ./calibration.json
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Path to model directory |
| `--prompts` | path | Path to safe baseline prompts file |
| `--prompt` | string | One or more safe baseline prompts |
| `--adapter` | path | Optional adapter path |
| `--output-file` | path | Output calibration JSON path |

### mc safety jailbreak-test
Run jailbreak entropy analysis using a calibration file.
```bash
mc safety jailbreak-test --model ./model --prompts ./attack_prompts.json --calibration ./calibration.json
```

**Required option:**
- `--calibration` path to a JSON file containing `driftSamples`, `safeDeltaHSamples`, and `attackEntropySamples`.

---

## Geometry Commands

### mc geometry metrics
```bash
mc geometry metrics gromov-wasserstein <source_file> <target_file>
mc geometry metrics intrinsic-dimension <points_file>
mc geometry metrics effective-rank <points_file>
mc geometry metrics topological-fingerprint <points_file>
mc geometry metrics spectral-signature <points_file>
```

### mc geometry density
```bash
mc geometry density profile <model_dir>
mc geometry density diff <source_model_dir> <target_model_dir>
```

### Additional geometry subcommands:
- `mc geometry path` - Path geometry detection
- `mc geometry concept` - Concept detection
- `mc geometry report` - Consolidated geometry reports (manifold evidence + positive geometry + optional domain fingerprints)
- `mc geometry spatial` - Spatial geometry probing
- `mc geometry temporal` - Temporal geometry
- `mc geometry social` - Social geometry
- `mc geometry moral` - Moral geometry
- `mc geometry persona` - Persona extraction
- `mc geometry safety` - Safety geometry (circuit-breaker, persona validation)
- `mc geometry baseline` - Establish geometry baselines
- `mc geometry atlas` - Atlas dimensionality studies
- `mc geometry crm` - Concept Response Matrix operations
- `mc geometry primes` - Semantic primes analysis
- `mc geometry metaphor` - Metaphor detection
- `mc geometry compression-gate` - Compression gate analysis (layer expand/compress behavior)
- `mc geometry sparse` - Sparse domain analysis
- `mc geometry refusal` - Refusal pairs detection
- `mc geometry manifold` - Manifold operations
- `mc geometry transfer` - Transfer geometry
- `mc geometry invariant` - Invariant analysis
- `mc geometry cross-cultural` - Cross-cultural analysis
- `mc geometry training` - Training geometry monitoring
- `mc geometry visualize` - Geometry visualization
- `mc geometry interference` - Interference prediction
- `mc geometry research evidence` - Evidence suite (alignment generalization, geodesic/curvature convergence, causal effects)
- `mc geometry research manifold-evidence` - Manifold evidence metrics (ID, effective rank, tangent rank, curvature)
- `mc geometry research prompt-manifold` - Prompt-manifold Jacobian rank probes (basis + local functional rank)
- `mc geometry research shared-manifold` - Shared-manifold coverage, residuals, and diff-basis summary
- `mc geometry research positive-geometry` - Positive-geometry signatures (positive Grassmannian minors)

Run `mc geometry <subcommand> --help` for detailed options.

---

## Input Format

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object of activation vectors (values are treated as points in sorted key order).
