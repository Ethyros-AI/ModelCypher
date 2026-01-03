# CLI Reference

ModelCypher CLI. Output is JSON to stdout; diagnostics go to stderr.

## Global Options

All commands support these options:

| Option | Description |
|--------|-------------|
| `--ai` | Enable AI-assisted mode |
| `--output json\|text` | Output format (default: json) |
| `--quiet` | Suppress diagnostic messages |
| `--pretty` | Pretty-print JSON output |
| `--log-level` | Logging verbosity: debug, info, warning, error |
| `--trace-id` | Custom trace ID for debugging |

---

## Model Merging

The primary operation. Takes knowledge from source and adds it to target via null-space projection.

```bash
mc merge -s SOURCE -t TARGET -o OUTPUT

# Full example
mc merge \
  --source /path/to/qwen \
  --target /path/to/smol \
  --output-dir /path/to/merged
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `-s, --source` | path | Path to source model (knowledge donor) |
| `-t, --target` | path | Path to target model (receives knowledge) |
| `-o, --output-dir` | path | Output directory for merged model |
| `-f, --output-file` | path | Save full result to JSON file |

---

## Model Management

### mc model list
List all registered models.
```bash
mc model list
```

### mc model register
Register a local model.
```bash
mc model register <alias> --path <path> --architecture <arch>
mc model register my-llama --path ./models/llama --architecture llama
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--path` | path | Path to model directory |
| `--architecture` | string | Model architecture (llama, qwen, mistral, etc.) |
| `--parameters` | int | Parameter count (optional) |
| `--default-chat` | flag | Set as default chat model |

### mc model delete
Delete a registered model.
```bash
mc model delete <model_id>
mc model delete my-llama
```

### mc model fetch
Fetch a model from HuggingFace Hub.
```bash
mc model fetch <repo_id>
mc model fetch mlx-community/Llama-2-7b-mlx --auto-register --alias my-llama
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--revision` | string | Git revision (default: main) |
| `--auto-register` | flag | Register model after download |
| `--alias` | string | Alias for registration |
| `--architecture` | string | Override architecture detection |

### mc model search
Search for models on HuggingFace Hub.
```bash
mc model search <query>
mc model search llama --library mlx --quant 4bit
mc model search --author mlx-community --sort downloads
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--author` | string | Filter by author |
| `--library` | string | Filter: mlx, safetensors, pytorch, any |
| `--quant` | string | Quantization: 4bit, 8bit, any |
| `--sort` | string | Sort by: downloads, likes, lastModified, trending |
| `--limit` | int | Results per page (default: 20) |
| `--cursor` | string | Pagination cursor |

### mc model probe
Probe a model for architecture details.
```bash
mc model probe <model_path>
mc model probe ./models/llama-7b
```

**Output fields:**
- `architecture`, `parameterCount`, `vocabSize`, `hiddenSize`
- `numAttentionHeads`, `quantization`, `layerCount`, `layers`

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

---

## Training

### mc train start
Start a training job.
```bash
mc train start --model <model> --dataset <dataset> [options]
mc train start --model meta-llama/Llama-2-7b --dataset ./data.jsonl \
  --learning-rate 1e-5 --batch-size 4 --epochs 3 --sequence-length 2048 \
  --grad-accum 4 --warmup-steps 100 --weight-decay 0.01 \
  --gradient-checkpointing --mixed-precision --compute-precision float16 \
  --optimizer-type adamw --out ./output --seed 42 --deterministic
```

**Required options:**
| Option | Type | Description |
|--------|------|-------------|
| `--model` | path | Model identifier or path |
| `--dataset` | path | Path to dataset file |
| `--learning-rate` | float | Learning rate |
| `--batch-size` | int | Batch size |
| `--epochs` | int | Number of epochs |
| `--sequence-length` | int | Max sequence length |
| `--grad-accum` | int | Gradient accumulation steps |
| `--warmup-steps` | int | Warmup steps |
| `--weight-decay` | float | Weight decay |
| `--gradient-checkpointing/--no-gradient-checkpointing` | flag | Enable gradient checkpointing |
| `--mixed-precision/--no-mixed-precision` | flag | Enable mixed precision |
| `--compute-precision` | string | Compute precision: float16, bfloat16, float32 |
| `--optimizer-type` | string | Optimizer type (adamw) |
| `--out` | path | Output directory |
| `--seed` | int | Random seed |
| `--deterministic/--stochastic` | flag | Deterministic training |

**Optional LoRA options:**
| Option | Type | Description |
|--------|------|-------------|
| `--lora-rank` | int | LoRA rank |
| `--lora-alpha` | float | LoRA alpha |
| `--lora-dropout` | float | LoRA dropout |
| `--lora-targets` | list | Target modules for LoRA |

**Control options:**
| Option | Type | Description |
|--------|------|-------------|
| `--resume-from` | path | Resume from checkpoint |
| `--detach` | flag | Run in background |
| `--stream` | flag | Stream training events |

### mc train preflight
Run preflight checks before training.
```bash
mc train preflight --model <model> --dataset <dataset> [same options as start]
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
mc eval run --model ./model --dataset ./data.jsonl --batch-size 8 --max-samples 100
```

**Output fields:**
- `evalId`, `modelPath`, `datasetPath`, `averageLoss`, `perplexity`, `sampleCount`

### mc eval benchmark
Run lm-eval-harness benchmarks on an MLX model.
```bash
mc eval benchmark --model ./model --tasks gsm8k,hellaswag
mc eval benchmark --model ./model --tasks mmlu --limit 100 --num-fewshot 5
mc eval benchmark --model ./model --tasks arc_challenge --output-path ./results.json
```

### mc eval domain
Run domain-specific benchmarks mapped to industry standards.
```bash
mc eval domain --model ./model --domain computational
mc eval domain --model ./model --domain mathematical --domain logical
mc eval domain --model ./model --suite standard
mc eval domain --model ./model --suite comprehensive --limit 100
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

## System

### mc system status
Get system status.
```bash
mc system status
mc system status --require-metal
```

**Output fields:**
- `metalAvailable`, `gpuMemory`, `cpuCount`, `platform`

### mc system probe
Probe a system target.
```bash
mc system probe gpu
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
mc entropy verify-baseline --mean 0.1 --std-dev 0.05 --max 0.3 --min 0.0 --observed '[0.1, 0.15, 0.12]'
```

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
Calibrate entropy thresholds by measuring actual model distributions.
```bash
mc entropy calibrate --model /path/to/model --prompts ./prompts.json
mc entropy calibrate --model /path/to/model --prompts ./prompts.json --output-file ./calibration.json
```

---

## Safety

### mc safety adapter-probe
Probe adapter for delta-feature geometry.
```bash
mc safety adapter-probe --adapter ./my-adapter
mc safety adapter-probe --adapter ./my-adapter --tier thorough
mc safety adapter-probe --adapter ./my-adapter --base-model ./base
```

**Options:**
| Option | Type | Description |
|--------|------|-------------|
| `--adapter` | path | Path to adapter directory |
| `--base-model` | path | Path to base model (optional) |
| `--tier` | string | Probe tier: quick, default, thorough |

**Output fields:**
- `layerCount`, `outlierLayerCount`, `outlierLayerIndices`
- `maxL2Norm`, `meanL2Norm`, `meanSparsity`, `l2Norms`, `sparsity`

---

## Geometry Commands

### mc geometry metrics
```bash
mc geometry metrics gromov-wasserstein <source_file> <target_file>
mc geometry metrics intrinsic-dimension <points_file>
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
- `mc geometry spatial` - Spatial geometry probing
- `mc geometry temporal` - Temporal geometry
- `mc geometry social` - Social geometry
- `mc geometry moral` - Moral geometry
- `mc geometry emotion` - Emotion geometry
- `mc geometry persona` - Persona extraction
- `mc geometry safety` - Safety geometry (circuit-breaker, persona validation)
- `mc geometry baseline` - Establish geometry baselines
- `mc geometry atlas` - Atlas dimensionality studies
- `mc geometry crm` - Concept Response Matrix operations
- `mc geometry primes` - Semantic primes analysis
- `mc geometry metaphor` - Metaphor detection
- `mc geometry sparse` - Sparse domain analysis
- `mc geometry refusal` - Refusal pairs detection
- `mc geometry manifold` - Manifold operations
- `mc geometry transfer` - Transfer geometry
- `mc geometry invariant` - Invariant analysis
- `mc geometry cross-cultural` - Cross-cultural analysis
- `mc geometry training` - Training geometry monitoring
- `mc geometry visualize` - Geometry visualization
- `mc geometry interference` - Interference prediction

Run `mc geometry <subcommand> --help` for detailed options.

---

## Input Format

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object of activation vectors (values are treated as points in sorted key order).
