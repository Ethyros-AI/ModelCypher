# CLI Reference

ModelCypher CLI reference.

Notes:
- Structured output goes to stdout (JSON by default). Logs and diagnostics go to stderr.
- In this repo, run the CLI as `poetry run mc …`. Examples below use `mc …` for brevity.
- Global options can appear anywhere on the command line (e.g. `mc model probe … --pretty`).

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

### mc model register
Register a local model. (deprecated; use `mc model add`)
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
Fetch a model from HuggingFace Hub. (deprecated; use `mc model add`)
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
Probe a model for architecture details. (deprecated; use `mc model info`)
```bash
mc model probe <model_path>
mc model probe ./models/llama-7b
```

**Output fields:**
- `architecture`, `parameterCount`, `vocabSize`, `hiddenSize`
- `numAttentionHeads`, `quantization`, `layerCount`, `layers`

### mc model quantize-sweep
Quantize a model across multiple bit widths with MLX and profile each variant.
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
| `--mode` | string | Quantization mode passed to MLX |
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
Run lm-eval-harness benchmarks on an MLX model.
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
