# ModelCypher CLI Reference

**For AI Agents:** This document provides complete CLI command reference. The CLI is designed with AI agents as the primary user. Use `--ai` mode for optimized machine consumption. For interpretation of geometry metrics, see `docs/GEOMETRY-GUIDE.md`.

**Version:** 1.2
**Last Updated:** 2025-11-30

## Quick Start

```bash
# AI mode (auto-enabled when stdout is piped)
tc --ai <command>

# All commands support JSON output
tc <command> --output json

# All commands support YAML output
tc <command> --output yaml

# Default is human-readable text
tc <command> --output text
```

## AI-First Features

### `--ai` Mode

Optimizes output for AI agent consumption. Automatically enabled when stdout is not a TTY.

```bash
tc --ai inventory                    # Explicit AI mode
tc train start ... | parse_json      # Auto-detected (piped output)
```

**Behavior:**
- Forces JSON output with deterministic field ordering
- Suppresses all stderr (very-quiet mode)
- Disables prompts and auto-confirms operations
- Structures errors as machine-parseable JSON

**Environment Variables:**
- `TC_AI_MODE=1` - Enable AI mode
- `TC_NO_AI=1` - Disable auto AI mode

### `tc inventory` - Single-Call State Discovery

Get all ModelCypher state in one call:

```bash
tc inventory
tc inventory --include system,models,jobs
tc inventory --exclude checkpoints --active-jobs-only
```

**Response sections:** `system`, `models`, `datasets`, `jobs`, `checkpoints`, `paths`, `version`

### `tc explain` - Dry-Run Capability

Preview what a command would do:

```bash
tc explain train --model qwen-0.5b --dataset data.jsonl --epochs 3
tc explain job delete job_abc123
```

**Response includes:** service calls, affected resources, required permissions, warnings, estimated duration.

---

## Global Flags

Available on all commands:

| Flag | Description | Example |
|------|-------------|---------|
| `--ai` | AI mode: JSON, no stderr, auto-confirm | `--ai` |
| `--output <format>` | Output format: `json`, `yaml`, `text` | `--output json` |
| `--quiet` | Suppress info logs | `--quiet` |
| `--very-quiet` | Suppress all logs | `--very-quiet` |
| `--yes` | Auto-confirm all prompts | `--yes` |
| `--no-prompt` | Fail if confirmation required | `--no-prompt` |
| `--pretty` | Pretty-print JSON output | `--pretty` |
| `--log-level <level>` | Set log level: `trace`, `debug`, `info`, `warn`, `error` | `--log-level debug` |
| `--trace-id <uuid>` | Request tracing ID | `--trace-id abc-123` |

---

## Commands

### `tc train` - Training Lifecycle

#### `tc train start` - Start Training Job

**Purpose:** Start a new training job with specified configuration.

**Usage:**
```bash
tc train start \
  --model <model-id> \
  --dataset <path.jsonl> \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --epochs 3 \
  --sequence-length 2048 \
  --output json
```

**Required Flags:**
- `--model <id>` - Model identifier to fine-tune
- `--dataset <path>` - Path to training dataset (JSONL format)

**Optional Flags:**
- `--learning-rate <float>` - Learning rate (default: 1e-5)
- `--batch-size <int>` - Batch size (auto-calculated if omitted)
- `--epochs <int>` - Number of epochs (default: 3)
- `--sequence-length <int>` - Max sequence length (default: 2048)
- `--grad-accum <int>` - Gradient accumulation steps
- `--warmup-steps <int>` - Warmup steps
- `--weight-decay <float>` - Weight decay
- `--gradient-clip <float>` - Gradient clipping norm
- `--resume-from <path>` - Resume from checkpoint
- `--lora-rank <int>` - LoRA rank
- `--lora-alpha <float>` - LoRA alpha
- `--lora-dropout <float>` - LoRA dropout
- `--lora-targets <modules...>` - LoRA target modules (default: `q_proj v_proj` when LoRA is enabled)
- `--lora-layers <int>` - Apply LoRA to the final N transformer layers (default: all)
- `--out <path>` - Custom output directory
- `--seed <uint>` - Random seed
- `--deterministic` - Enable deterministic kernels
- `--detach` - Detach immediately after queuing
- `--stream` - Stream NDJSON events to stdout

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "batchSize": 4
}
```

**Examples:**
```bash
# Basic training
tc train start --model llama-3.2-1B --dataset data.jsonl --output json

# LoRA fine-tuning
tc train start \
  --model qwen-0.5b \
  --dataset data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --output json

# LoRA fine-tuning with explicit target modules (advanced)
tc train start \
  --model qwen-0.5b \
  --dataset data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-targets q_proj v_proj \
  --output json

# Memory-saving LoRA (final 4 layers only)
tc train start \
  --model qwen-0.5b \
  --dataset data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-layers 4 \
  --output json

# Stream training events (NDJSON)
tc train start --model llama-3.2-1B --dataset data.jsonl --stream
```

---

#### `tc train preflight` - Pre-flight Checks

**Purpose:** Validate training configuration and estimate resource usage.

**Usage:**
```bash
tc train preflight \
  --model <model-id> \
  --dataset <path.jsonl> \
  --sequence-length 2048 \
  --output json
```

**Flags:** Same as `train start`

**JSON Output:**
```json
{
  "predictedBatchSize": 4,
  "estimatedVRAMUsageBytes": 8589934592,
  "availableVRAMBytes": 17179869184,
  "canProceed": true
}
```

---

#### `tc train status` - Job Status

**Purpose:** Get current status of a training job.

**Usage:**
```bash
tc train status <job-id> --output json
```

**Flags:**
- `--follow` - Follow job progress (poll mode)
- `--stream` - Stream NDJSON events

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "status": "running",
  "currentStep": 150,
  "totalSteps": 1000,
  "currentEpoch": 1,
  "totalEpochs": 3,
  "loss": 2.3456,
  "learningRate": 0.00001,
  "createdAt": "2025-11-13T10:00:00Z",
  "updatedAt": "2025-11-13T10:15:00Z"
}
```

**Status Values:** `pending`, `running`, `paused`, `completed`, `failed`, `cancelled`

---

#### `tc train pause` - Pause Job

**Purpose:** Pause a running training job.

**Usage:**
```bash
tc train pause <job-id> --output json
```

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "status": "paused"
}
```

---

#### `tc train resume` - Resume Job

**Purpose:** Resume a paused training job.

**Usage:**
```bash
tc train resume <job-id> --output json
```

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "status": "running"
}
```

---

#### `tc train cancel` - Cancel Job

**Purpose:** Cancel a training job.

**Usage:**
```bash
tc train cancel <job-id> --output json
```

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "status": "cancelled"
}
```

---

#### `tc train export` - Export Trained Model

**Purpose:** Export a trained model to deployment formats.

**Usage:**
```bash
tc train export \
  --model <model-id> \
  --format <format> \
  --output-path <path>

# Or export artifacts from a completed training job (e.g., LoRA adapters)
tc train export \
  --job <job-uuid> \
  --format <format> \
  --output-path <path>
```

**Required Flags:**
- Exactly one of:
  - `--model <id>` - Model ID or checkpoint path to export
  - `--job <uuid>` - Completed training job ID to export artifacts from
- `--format <format>` - Export format: `gguf`, `safetensors`, `npz`, `mlx`, `ollama`, `lora`, `coreml`
- `--output-path <path>` - Output path

**Supported Formats:**
- `gguf` - GGUF format for llama.cpp / Ollama
- `safetensors` - SafeTensors format
- `npz` - NumPy compressed format
- `mlx` - Apple MLX package
- `ollama` - Ollama bundle (GGUF + Modelfile)
- `lora` - LoRA adapter bundle (requires `--job`; writes `adapter_config.json` + `adapters.safetensors`)
- `coreml` - CoreML `.mlpackage` (see `docs/usage/coreml-ios.md` for environment/flag details)

**JSON Output:**
```json
{
  "modelID": "llama-3.2-1B",
  "format": "gguf",
  "outputPath": "/path/to/model.gguf"
}
```

**Examples:**
```bash
# Export to GGUF for llama.cpp
tc train export --model llama-3.2-1B --format gguf --output-path model.gguf

# Export to Ollama bundle
tc train export --model qwen-0.5b --format ollama --output-path model.zip

# Export LoRA adapters from a completed job
tc train export --job <job-uuid> --format lora --output-path ./adapters/
```

---

#### `tc train logs` - View Training Logs

**Purpose:** Stream training logs from system log.

**Usage:**
```bash
tc train logs <job-id>
```

**Flags:**
- `--tail <lines>` - Show last N lines (default: 100)
- `--follow` - Follow logs in real-time

---

### `tc job` - Job Management

#### `tc job list` - List Jobs

**Purpose:** List all training jobs.

**Usage:**
```bash
tc job list --output json
```

**Flags:**
- `--status <status>` - Filter by status
- `--active-only` - Show only active jobs

**JSON Output:**
```json
[
  {
    "jobId": "job-abc123",
    "modelId": "llama-3.2-1B",
    "status": "running",
    "currentStep": 150,
    "totalSteps": 1000,
    "createdAt": "2025-11-13T10:00:00Z"
  },
  {
    "jobId": "job-def456",
    "modelId": "qwen-0.5b",
    "status": "completed",
    "currentStep": 1000,
    "totalSteps": 1000,
    "createdAt": "2025-11-12T15:30:00Z"
  }
]
```

---

#### `tc job show` - Show Job Details

**Purpose:** Show comprehensive job details including configuration, checkpoints, hyperparameters, and optionally full loss history.

**Usage:**
```bash
tc job show <job-id> --output json
tc job show <job-id> --loss-history --output json
```

**Flags:**
- `--loss-history` - Include full loss history in output

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "status": "completed",
  "createdAt": "2025-11-13T10:00:00Z",
  "startedAt": "2025-11-13T10:00:05Z",
  "completedAt": "2025-11-13T12:30:00Z",
  "modelId": "llama-3.2-1B",
  "datasetPath": "/datasets/train.jsonl",
  "progress": 1.0,
  "finalLoss": 1.2345,
  "checkpoints": [
    {
      "identifier": "checkpoint-100",
      "step": 100,
      "loss": 2.3456,
      "timestamp": "2025-11-13T10:15:00Z",
      "filePath": "/checkpoints/checkpoint-100.safetensors"
    }
  ],
  "hyperparameters": {
    "learningRate": 0.00001,
    "batchSize": 4,
    "epochs": 3,
    "sequenceLength": 2048
  },
  "lossHistory": [
    {"step": 1, "loss": 5.2341},
    {"step": 10, "loss": 4.1234},
    {"step": 100, "loss": 2.3456}
  ]
}
```

**Text Output:**
Displays formatted summary with loss statistics (min/max/avg) and first/last loss points.

---

#### `tc job attach` - Attach to Job

**Purpose:** Attach to a running job and stream events.

**Usage:**
```bash
tc job attach <job-id>
```

**Flags:**
- `--replay` - Replay events from start
- `--since <ISO8601>` - Replay events since timestamp

**Output:** NDJSON event stream

---

#### `tc job delete` - Delete Job

**Purpose:** Delete a completed or failed job.

**Usage:**
```bash
tc job delete <job-id> --yes --output json
```

**JSON Output:**
```json
{
  "deleted": "job-abc123"
}
```

---

### `tc checkpoint` - Checkpoint Management

#### `tc checkpoint list` - List Checkpoints

**Purpose:** List all checkpoints.

**Usage:**
```bash
tc checkpoint list --output json
```

**Flags:**
- `--job <job-id>` - Filter by job ID

**JSON Output:**
```json
{
  "checkpoints": [
    {
      "jobId": "job-abc123",
      "step": 100,
      "loss": 2.3456,
      "filePath": "/path/to/checkpoint-100.safetensors"
    },
    {
      "jobId": "job-abc123",
      "step": 200,
      "loss": 2.1234,
      "filePath": "/path/to/checkpoint-200.safetensors"
    }
  ]
}
```

---

#### `tc checkpoint delete` - Delete Checkpoint

**Purpose:** Delete a checkpoint file.

**Usage:**
```bash
tc checkpoint delete <path> --force --output json
```

**Flags:**
- `--force` - Skip confirmation

**JSON Output:**
```json
{
  "deleted": "/path/to/checkpoint.safetensors"
}
```

---

#### `tc checkpoint export` - Export Checkpoint

**Purpose:** Export a checkpoint to deployment formats.

**Usage:**
```bash
tc checkpoint export <checkpoint-path> \
  --format <format> \
  --output-path <path>
```

**Required Flags:**
- `<checkpoint-path>` - Path to checkpoint file
- `--format <format>` - Export format: `gguf`, `safetensors`, `npz`, `mlx`, `ollama`
- `--output-path <path>` - Output path

**JSON Output:**
```json
{
  "checkpoint": "/path/to/checkpoint.safetensors",
  "format": "gguf",
  "outputPath": "/path/to/model.gguf"
}
```

**Examples:**
```bash
# Export checkpoint to GGUF
tc checkpoint export checkpoint-100.safetensors --format gguf --output-path model.gguf

# Export to Ollama
tc checkpoint export checkpoint-final.safetensors --format ollama --output-path model.zip
```

---

### `tc model` - Model Management

#### `tc model list` - List Models

**Purpose:** List all registered models.

**Usage:**
```bash
tc model list --output json
```

**JSON Output:**
```json
[
  {
    "id": "llama-3.2-1B",
    "alias": "llama-3.2-1B",
    "architecture": "llama",
    "format": "safetensors",
    "path": "/models/llama-3.2-1B",
    "sizeBytes": 5368709120,
    "parameterCount": 1235000000,
    "isDefaultChat": false,
    "createdAt": "2025-11-13T09:00:00Z"
  }
]
```

---

#### `tc model register` - Register Model

**Purpose:** Register a model in the local registry.

**Usage:**
```bash
tc model register <alias> \
  --path <path> \
  --architecture <arch> \
  --parameters <count> \
  --output json
```

**Required Args:**
- `<alias>` - Alias to reference this model
- `--path <path>` - Path to model directory or weight file
- `--architecture <arch>` - Architecture: `llama`, `gemma`, `qwen`, `qwen2`, `phi`, `mistral`, `custom`

**Optional Flags:**
- `--parameters <count>` - Parameter count
- `--default-chat` - Mark as default assistant model

**JSON Output:**
```json
{
  "registered": "llama-3.2-1B"
}
```

**Examples:**
```bash
# Register a model
tc model register llama-3.2-1B \
  --path /models/llama-3.2-1B \
  --architecture llama \
  --parameters 1235000000 \
  --output json
```

---

#### `tc model merge` - Merge Two Models

**Purpose:** Merge two model weight sets into a new output model directory.

ModelCypher’s merge pipeline is geometry-aware: it uses anchor vectors (default: semantic primes) and a rotational alignment step so that “similar concepts” are combined coherently, not just averaged naïvely.

**Usage:**
```bash
tc model merge \
  --source <model-id-or-path> \
  --target <model-id-or-path> \
  --output-dir <out-dir> \
  --alpha 0.5 \
  --rank 32 \
  --output json
```

**Required Flags:**
- `--source <model-id-or-path>` - Source model (registry ID or filesystem path)
- `--target <model-id-or-path>` - Target model (registry ID or filesystem path)
- `--output-dir <path>` - Output directory for merged model

**Common Flags:**
- `--alpha <float>` - Blend factor (0 = target only, 1 = source only). Default: 0.5
- `--rank <int>` - Alignment rank (higher = more expressive, more compute). Default: 32
- `--anchor-mode <mode>` - Anchor strategy: `semantic-primes` (default), `geometric`, `intersection`, `rebasin`, `unified` (not implemented)
- `--module-scope <scope>` - Which modules to merge: `attention-only` (default) or `all`
- `--intersection <path>` - Intersection map JSON (required for `anchor-mode=intersection`)
- `--adaptive-alpha` - Enable adaptive alpha weighting (requires intersection map)
- `--dry-run` - Compute report only; do not write output files
- `--report-path <path>` - Write the JSON report to a file (in addition to stdout)

**Advanced Flags (currently limited):**
- `--fisher-source <path>` / `--fisher-target <path>` / `--fisher-strength <float>` / `--fisher-epsilon <float>` - Fisher blending inputs (only used in `anchor-mode=unified`, which is not implemented yet)

**Output Quantization (optional):**
- `--output-quant <4bit|8bit>` - Requantize the merged output weights as you save (keeps non-2D / non-`.weight` tensors as float)
- `--output-quant-group-size <int>` - Quantization group size (default: 64; default: 32 for `mxfp4`)
- `--output-quant-mode <affine|mxfp4>` - Quantization mode (default: `affine`)

**Notes:**
- Input weights may be `safetensors` or `.npz` and may include BF16 tensors; ModelCypher loads BF16 safely without requiring torch.
- Output format follows the target model’s weight format; support files from the target directory are copied alongside the merged weights.
- If `--output-quant` is provided, `config.json` is updated with a `quantization` block for downstream loaders.

**JSON Output (Report):**
```json
{
  "sourceModel": "source-id",
  "targetModel": "target-id",
  "anchorMode": "semantic-primes",
  "timestamp": "2025-11-26T04:34:05Z",
  "meanProcrustesError": 0.0000001,
  "maxProcrustesError": 0.000001,
  "rotationFieldRoughness": 0.02,
  "anchorCoverage": 0.91,
  "layerMetrics": [
    {
      "layerIndex": 0,
      "moduleName": "q_proj",
      "moduleKind": "attention",
      "procrustesError": 0.0000002,
      "conditionNumber": 1.3,
      "rotationDeviation": 0.01,
      "spectralRatio": 0.9
    }
  ]
}
```

**How to interpret the report (plain language):**
- `meanProcrustesError` near 0 means the alignment step found a very consistent mapping between anchor spaces.
- Low `anchorCoverage` means there weren’t enough reliable anchors; the merge may be low quality or brittle.

**Examples:**
```bash
# Dry-run (compute report only)
tc model merge \
  --source ./models/source \
  --target ./models/target \
  --output-dir /tmp/merge-out \
  --dry-run \
  --output json

# Merge and emit a compact 4-bit output model
tc model merge \
  --source ./models/source \
  --target ./models/target \
  --output-dir ./models/merged-4bit \
  --rank 8 \
  --output-quant 4bit \
  --output json
```

---

#### `tc model delete` - Delete Model

**Purpose:** Remove a registered model.

**Usage:**
```bash
tc model delete <model-id> --output json
```

**JSON Output:**
```json
{
  "deleted": "llama-3.2-1B"
}
```

---

#### `tc model fetch` - Download from HuggingFace

**Purpose:** Download a model from HuggingFace Hub with optional automatic registration.

**Usage:**
```bash
tc model fetch <repo-id> \
  --revision <branch> \
  --auto-register \
  --alias <name> \
  --output json
```

**Required Args:**
- `<repo-id>` - HuggingFace repository ID (e.g., `meta-llama/Llama-2-7b-hf`)

**Optional Flags:**
- `--revision <branch>` - Git revision (branch, tag, commit). Default: `main`
- `--auto-register` - Auto-register after download
- `--alias <name>` - Alias for registration (required with `--auto-register`)
- `--architecture <arch>` - Model architecture (optional with `--auto-register`; auto-detected from config.json if omitted)

**Architecture Auto-Detection:**
When using `--auto-register` without `--architecture`, the CLI automatically detects the model architecture by parsing the `model_type` field from the model's `config.json`.

Supported architectures: `llama`, `gemma`, `qwen`, `qwen2`, `phi`, `mistral`

Unknown architectures are preserved and may still work, but will generate a warning.

**JSON Output:**
```json
{
  "repoID": "meta-llama/Llama-2-7b-hf",
  "localPath": "/models/llama-2-7b-hf",
  "registeredID": "llama-2-7b",
  "detectedArchitecture": "llama"
}
```

**Examples:**
```bash
# Download only
tc model fetch meta-llama/Llama-2-7b-hf --output json

# Download and register with auto-detected architecture (recommended)
tc model fetch Qwen/Qwen2.5-0.5B-Instruct \
  --auto-register \
  --alias qwen-0.5b \
  --output json

# Download and register with explicit architecture
tc model fetch Qwen/Qwen2.5-0.5B-Instruct \
  --auto-register \
  --alias qwen-0.5b \
  --architecture qwen2 \
  --output json

# Specific revision
tc model fetch microsoft/phi-3-mini-4k-instruct --revision v1.0 --output json
```

---

#### `tc model search` - Search HuggingFace Hub

**Purpose:** Search HuggingFace Hub for models with filters optimized for ModelCypher workflows.

**Usage:**
```bash
tc model search [query] \
  --author <author> \
  --library <mlx|safetensors|pytorch|any> \
  --quant <4bit|8bit|any> \
  --sort <downloads|likes|lastModified|trending> \
  --limit <count> \
  --cursor <cursor> \
  --output json
```

**Optional Flags:**
- `--author <author>` - Filter by author/organization
- `--library <filter>` - Library filter: `mlx` (default), `safetensors`, `pytorch`, `any`
- `--quant <filter>` - Quantization filter: `4bit`, `8bit`, `any`
- `--sort <sort>` - Sort by: `downloads` (default), `likes`, `lastModified`, `trending`
- `--limit <count>` - Max results per page (default 20, max 100)
- `--cursor <cursor>` - Pagination cursor for next page

**Memory Fit Indicators:**
- `fits` - Model fits comfortably for training
- `tight` - Model fits but memory will be tight
- `tooBig` - Model is too large for available memory

**JSON Output:**
```json
{
  "count": 1,
  "hasMore": true,
  "nextCursor": "gAAAAABnX...",
  "models": [
    {
      "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
      "downloads": 12345,
      "likes": 321,
      "author": "mlx-community",
      "pipelineTag": "text-generation",
      "tags": ["mlx", "4bit", "quantized"],
      "isGated": false,
      "isPrivate": false,
      "isRecommended": true,
      "estimatedSizeGB": 1.0,
      "memoryFitStatus": "fits"
    }
  ]
}
```

**Examples:**
```bash
# Search by query
tc model search "llama 3" --output json

# Filter by author and quantization
tc model search --author Qwen --quant 4bit --output json

# Trending models, 10 results
tc model search --sort trending --limit 10 --output json
```

---

### `tc system` - System Information

#### `tc system status` - System Status

**Purpose:** Show system information and active jobs.

**Usage:**
```bash
tc system status --output json
```

**Flags:**
- `--require-metal` - Exit with code 3 if Metal unavailable

**JSON Output:**
```json
{
  "metalAvailable": true,
  "gpuMemoryBytes": 17179869184,
  "systemMemoryBytes": 34359738368,
  "activeJobs": 1,
  "thermalState": "nominal"
}
```

---

#### `tc system probe` - Deep Diagnostics

**Purpose:** Run deep system diagnostics.

**Usage:**
```bash
tc system probe <target> --output json
```

**Targets:** `all`, `metal`, `mlx`, `memory`

**JSON Output:**
```json
{
  "target": "all",
  "metal": {
    "available": true,
    "deviceName": "Apple M3 Max"
  },
  "mlx": {
    "version": "0.25.3",
    "gpuAvailable": true
  },
  "memory": {
    "systemBytes": 34359738368,
    "gpuBytes": 17179869184
  }
}
```

---

### `tc dataset` - Dataset Management

#### `tc dataset validate` - Validate Dataset

**Purpose:** Validate dataset JSONL format and compute token statistics.

**Usage:**
```bash
tc dataset validate <dataset.jsonl> --output json
```

**JSON Output:**
```json
{
  "valid": true,
  "totalExamples": 1024,
  "averageTokens": 128.5,
  "maxTokens": 512,
  "minTokens": 16,
  "errors": [],
  "warnings": []
}
```

**Notes:**
- Summaries are printed when `--output text`
- Lists full error/warning arrays in JSON/YAML modes

---

#### `tc dataset preprocess` - Preprocess Dataset

**Purpose:** Tokenize and write a processed dataset for training.

**Usage:**
```bash
tc dataset preprocess raw.jsonl --output-path processed.jsonl --tokenizer qwen-0.5b --output json
```

**Required Args:**
- `<input>` - Source dataset in JSONL format

**Required Flags:**
- `--output-path <path>` (aliases: `-o`, `--dataset-output`, `--processed-output`) - Processed dataset path
- `--tokenizer <model>` - Tokenizer model identifier (e.g., `qwen-0.5b`)

**JSON Output:**
```json
{
  "processedExamples": 2400,
  "skippedExamples": 12,
  "outputPath": "/datasets/processed.jsonl",
  "totalTokens": 3145728
}
```

**Tips:**
- Combine with `tc train preflight` to size batch counts
- Use `--output yaml` for human-readable summaries

---

#### `tc dataset convert` - Convert Dataset Format

**Purpose:** Convert each JSONL row into a different dataset content format (`text`, `chat`, `completion`, `tools`, `instruction`).

This is useful when you have “mostly-right” data but need it normalized into a single format before training.

**Usage:**
```bash
tc dataset convert <dataset.jsonl> \
  --to <format> \
  --output-path <converted.jsonl> \
  --output json
```

**Required Args:**
- `<dataset.jsonl>` - Source dataset file

**Required Flags:**
- `--to <format>` - Target format: `text`, `chat`, `completion`, `tools`, `instruction`
- `--output-path <path>` (alias: `-o`) - Output dataset path

**JSON Output:**
```json
{
  "_schema": "tc.dataset.convert.v1",
  "sourcePath": "/datasets/raw.jsonl",
  "outputPath": "/datasets/converted.jsonl",
  "targetFormat": "chat",
  "lineCount": 1024,
  "warnings": []
}
```

**Warnings:**
- You may see warnings if a training job appears to be using the dataset while you edit/convert it.

---

#### `tc dataset preview` - Preview Dataset Rows

**Purpose:** Preview the first N rows with format detection and validation messages.

**Usage:**
```bash
tc dataset preview <dataset.jsonl> --lines 5 --output json
tc dataset preview <dataset.jsonl> --lines 20 --format table --output text
```

**Flags:**
- `--lines <int>` - Number of rows to preview (capped)
- `--format <json|table>` - Text output format hint (default: `json`)

**JSON Output:**
```json
{
  "_schema": "tc.dataset.preview.v1",
  "path": "/datasets/demo.jsonl",
  "rowCount": 5,
  "rows": [
    {
      "_schema": "tc.dataset.row.v1",
      "lineNumber": 1,
      "raw": "{\"text\":\"hello\"}",
      "format": "text",
      "fields": { "text": "hello" },
      "validationMessages": [],
      "rawTruncated": false,
      "rawFullBytes": 15,
      "fieldsTruncated": []
    }
  ],
  "warnings": []
}
```

---

#### `tc dataset get-row` - Inspect One Row

**Purpose:** Read and validate a single dataset row by line number.

**Usage:**
```bash
tc dataset get-row <dataset.jsonl> --line 42 --output json
```

**JSON Output:**
```json
{
  "_schema": "tc.dataset.row.v1",
  "lineNumber": 42,
  "raw": "{\"text\":\"hello\"}",
  "format": "text",
  "fields": { "text": "hello" },
  "validationMessages": [],
  "rawTruncated": false,
  "rawFullBytes": 15,
  "fieldsTruncated": []
}
```

---

#### `tc dataset update-row` - Edit One Row

**Purpose:** Replace a single row’s content (by line number).

**Usage:**
```bash
tc dataset update-row <dataset.jsonl> \
  --line 42 \
  --content '{"text":"updated"}' \
  --output json
```

**Notes:**
- `--content` must be a JSON object string. Example: `{"text":"hello"}`.

**JSON Output:**
```json
{
  "_schema": "tc.dataset.edit.v1",
  "status": "updated",
  "lineNumber": 42,
  "row": {
    "_schema": "tc.dataset.row.v1",
    "lineNumber": 42,
    "raw": "{\"text\":\"updated\"}",
    "format": "text",
    "fields": { "text": "updated" },
    "validationMessages": [],
    "rawTruncated": false,
    "rawFullBytes": 17,
    "fieldsTruncated": []
  },
  "warnings": []
}
```

---

#### `tc dataset add-row` - Append a New Row

**Purpose:** Append a new row to the end of a dataset file.

**Usage:**
```bash
tc dataset add-row <dataset.jsonl> \
  --format text \
  --fields '{"text":"new example"}' \
  --output json
```

**Notes:**
- `--fields` must be a JSON object string.

**JSON Output:**
```json
{
  "_schema": "tc.dataset.edit.v1",
  "status": "added",
  "lineNumber": 2049,
  "row": {
    "_schema": "tc.dataset.row.v1",
    "lineNumber": 2049,
    "raw": "{\"text\":\"new example\"}",
    "format": "text",
    "fields": { "text": "new example" },
    "validationMessages": [],
    "rawTruncated": false,
    "rawFullBytes": 21,
    "fieldsTruncated": []
  },
  "warnings": []
}
```

---

#### `tc dataset delete-row` - Delete One Row

**Purpose:** Delete a single row by line number.

**Usage:**
```bash
tc dataset delete-row <dataset.jsonl> --line 42 --output json
```

**JSON Output:**
```json
{
  "_schema": "tc.dataset.edit.v1",
  "status": "deleted",
  "lineNumber": 42,
  "row": null,
  "warnings": []
}
```

---

#### `tc dataset list` - List Datasets

**Purpose:** List datasets managed by the service.

**Usage:**
```bash
tc dataset list --output json
```

**JSON Output:**
```json
[
  {
    "id": "8B8ABA18-8452-4010-8A6D-2B99A146F0A6",
    "name": "demo",
    "path": "/datasets/demo.jsonl",
    "sizeBytes": 1048576,
    "exampleCount": 2048,
    "createdAt": "2025-11-13T09:00:00Z"
  }
]
```

**Text Output:**
- Prints each dataset with name, size, example count, and path

---

#### `tc dataset delete` - Delete Dataset

**Purpose:** Delete a dataset JSONL file.

**Usage:**
```bash
tc dataset delete <dataset.jsonl> --force --output json
```

**Flags:**
- `--force` - Skip confirmation prompt (otherwise obeys global `--yes`/`--no-prompt`)

**JSON Output:**
```json
{
  "deleted": "/datasets/old.jsonl"
}
```

**Warning:** Deletion is permanent—ensure backups exist.

---

#### `tc dataset pack-asif` - Package Dataset into ASIF Sparse Image

**Purpose:** Create an Apple Sparse Image Format (ASIF) volume containing a dataset (file or directory) for VM tooling or checksum-stable transfer.

**Usage:**
```bash
tc dataset pack-asif <source> \
  --destination <target.asif> \
  --headroom-percent 15 \
  --minimum-free-gib 2 \
  --filesystem apfs \
  --volume-name DATASET \
  --overwrite \
  --output json
```

**JSON Output:**
```json
{
  "image": "/datasets/demo.asif",
  "volumeName": "demo",
  "imageBytes": 12884901888,
  "sourceBytes": 10485760,
  "headroomBytes": 2147483648,
  "sha256": "0c3c0d9c901a00c3b7f86f5bfb31bcf1c3bbd6c5b9d45d7e4d5c2d7cd6f0a221"
}
```

**Notes:**
- Defaults output path to `<source>.asif` when not provided.
- Adds percentage headroom plus minimum free GiB to avoid overflows during copy.
- Uses `diskutil image create blank --format ASIF` and `hdiutil attach` under the hood.
- `--filesystem none` creates an unformatted block device (VM formats later); `apfs` is default.

---

### `tc eval` - Evaluation Results

#### `tc eval list` - List Evaluation Results

**Purpose:** List stored evaluation results from model quality assessments.

**Usage:**
```bash
tc eval list --output json
tc eval list --limit 10 --output json
```

**Flags:**
- `--limit <int>` - Maximum results to return (default: 50)

**JSON Output:**
```json
{
  "evaluations": [
    {
      "id": "eval-abc123",
      "modelName": "llama-3.2-1B-lora",
      "datasetName": "test-set.jsonl",
      "averageLoss": 2.1234,
      "perplexity": 8.3721,
      "timestamp": "2025-11-13T14:30:00Z"
    }
  ]
}
```

---

#### `tc eval show` - Show Evaluation Details

**Purpose:** Show full evaluation details including per-sample metrics.

**Usage:**
```bash
tc eval show <eval-id> --output json
```

**JSON Output:**
```json
{
  "id": "eval-abc123",
  "modelPath": "/checkpoints/model-lora",
  "modelName": "llama-3.2-1B-lora",
  "datasetPath": "/datasets/test-set.jsonl",
  "datasetName": "test-set.jsonl",
  "averageLoss": 2.1234,
  "perplexity": 8.3721,
  "sampleCount": 100,
  "timestamp": "2025-11-13T14:30:00Z",
  "config": {
    "batchSize": 4,
    "sequenceLength": 2048
  },
  "sampleResults": [
    {"index": 0, "loss": 2.0123, "tokenCount": 128},
    {"index": 1, "loss": 2.2345, "tokenCount": 256}
  ]
}
```

---

### `tc compare` - Comparison History

#### `tc compare list` - List Comparison Sessions

**Purpose:** List A/B checkpoint comparison sessions saved from the GUI or inference runs.

**Usage:**
```bash
tc compare list --output json
tc compare list --status finished --limit 20 --output json
```

**Flags:**
- `--status <status>` - Filter by status: `finished`, `failed`, `cancelled`
- `--limit <int>` - Maximum sessions to return (default: 50)

**JSON Output:**
```json
{
  "sessions": [
    {
      "id": "session-abc123",
      "createdAt": "2025-11-13T15:00:00Z",
      "checkpointCount": 2,
      "promptPreview": "What is machine learning?",
      "status": "finished",
      "notes": "Testing LoRA rank 8 vs 16",
      "tags": ["experiment", "lora"]
    }
  ]
}
```

---

#### `tc compare show` - Show Comparison Details

**Purpose:** Show full comparison session with all checkpoint responses and metrics.

**Usage:**
```bash
tc compare show <session-id> --output json
```

**JSON Output:**
```json
{
  "id": "session-abc123",
  "createdAt": "2025-11-13T15:00:00Z",
  "prompt": "What is machine learning?",
  "config": {
    "temperature": 0.7,
    "maxTokens": 512,
    "topP": 0.95
  },
  "checkpoints": [
    {
      "checkpointPath": "/checkpoints/model-lora-r8",
      "modelName": "llama-3.2-1B-lora-r8",
      "baseModelName": "llama-3.2-1B",
      "response": "Machine learning is a subset of artificial intelligence...",
      "status": "finished",
      "metrics": {
        "tokensPerSecond": 42.5,
        "totalTokens": 128,
        "generationTimeSeconds": 3.01
      }
    },
    {
      "checkpointPath": "/checkpoints/model-lora-r16",
      "modelName": "llama-3.2-1B-lora-r16",
      "baseModelName": "llama-3.2-1B",
      "response": "Machine learning refers to algorithms that improve...",
      "status": "finished",
      "metrics": {
        "tokensPerSecond": 38.2,
        "totalTokens": 156,
        "generationTimeSeconds": 4.08
      }
    }
  ],
  "notes": "Testing LoRA rank 8 vs 16",
  "tags": ["experiment", "lora"]
}
```

**Text Output:**
Displays formatted comparison with side-by-side response previews and metrics summary.

---

### `tc doc` - Document Conversion

#### `tc doc convert` - Convert Documents

**Purpose:** Convert documents or directories to JSONL training format. Works exactly like the GUI—point at a directory and it ingests everything recursively, providing clear feedback on what was found, processed, and skipped.

**Usage:**
```bash
tc doc convert \
  --input <paths...> \
  --output-path <file.jsonl> \
  --output json
```

**Required Flags:**
- `--input <paths...>` - Input file or directory paths (multiple allowed)
- `--output-path <path>` - Output JSONL file path

**Optional Flags:**
- `--chunk-size <int>` - Maximum characters per chunk (default: 2000)
- `--chunk-overlap <int>` - Character overlap between chunks (default: 200)
- `--text-only` - Output format: text-only (default) vs chat format
- `--stream` - Stream NDJSON progress events
- `--update-manifest` - Update dataset manifest

**Supported File Formats:**
| Category | Extensions |
|----------|------------|
| Text/Markdown | `.md`, `.txt`, `.text`, `.rtf` |
| Code | `.swift`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.kt`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.rb`, `.php`, `.sh`, `.bash`, `.zsh`, `.fish`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf` |
| Data | `.json`, `.jsonl`, `.csv`, `.tsv`, `.xml` |
| Documents | `.pdf`, `.html`, `.htm`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls` |
| Other | `.log`, `.env`, `.gitignore`, `.dockerfile`, `.makefile` |

**JSON Output (Manifest):**
```json
{
  "jobId": "D9F9F49A-BB4D-44F0-A46B-AE7BA2219E2B",
  "datasetName": "my-dataset",
  "generator": "ModelCypherCLI",
  "createdAt": "2025-11-26T04:34:05Z",
  "durationSeconds": 24.7,
  "filesProcessed": 547,
  "sampleCount": 1190,
  "totalTokens": 845000,
  "totalCharacters": 3200000,
  "detectedFormat": "text",
  "outputFormat": "jsonl",
  "qualityScore": 100,
  "validationStatus": "passed",
  "validationErrors": [],
  "warnings": [],
  "sourceFiles": ["/path/to/file1.md", "/path/to/file2.txt"],
  "failedFiles": ["/path/to/corrupted.pdf"]
}
```

**NDJSON Progress Events (with `--stream`):**
```json
{"stage":"collecting","message":"Scanning directories..."}
{"stage":"collecting","message":"Found 547 files to process"}
{"stage":"loading","current":1,"total":547,"file":"doc1.md"}
{"stage":"loading","current":2,"total":547,"file":"doc2.txt"}
{"stage":"chunking","current":100,"total":547}
{"stage":"writing","message":"Writing 1190 samples to output.jsonl"}
{"stage":"completed","samples":1190,"duration":24.7}
```

**Examples:**
```bash
# Convert a directory of documents (recursive)
tc doc convert \
  --input ~/Documents/training-docs/ \
  --output-path train.jsonl \
  --output json

# Convert multiple inputs
tc doc convert \
  --input ~/docs/mlx/ \
  --input ~/docs/swift/ \
  --input ~/notes/readme.md \
  --output-path combined.jsonl \
  --output json

# Custom chunk settings for longer context
tc doc convert \
  --input ~/project/ \
  --output-path data.jsonl \
  --chunk-size 4000 \
  --chunk-overlap 400 \
  --output json

# Stream progress (for monitoring)
tc doc convert \
  --input ~/large-corpus/ \
  --output-path corpus.jsonl \
  --stream
```

**Output Format:**
Each line in the output JSONL file contains a single training sample:
```json
{"text": "# Document Title\n\nDocument content chunked for training..."}
```

**Error Handling:**
- If a directory contains no valid files, returns error with collection stats
- Failed files (corrupt, unreadable) are logged in `failedFiles` array
- Warnings (e.g., small sample count) appear in `warnings` array
- Quality score reflects sample count, token distribution, and validation

---

#### `tc doc validate` - Validate Dataset

**Purpose:** Validate JSONL training dataset.

**Usage:**
```bash
tc doc validate <dataset.jsonl> --output json
```

**JSON Output:**
```json
{
  "valid": true,
  "samples": 1000,
  "errors": [],
  "warnings": ["Sample 42: long sequence (3000 tokens)"]
}
```

---

## NDJSON Event Streaming

Commands that support `--stream` emit newline-delimited JSON events:

**Training Events:**
```json
{"schemaVersion":"0.1.0","ts":"2025-11-13T10:30:45Z","sequence":150,"traceId":"abc","jobId":"job-123","type":"trainingProgress","data":{"step":150,"totalSteps":1000,"loss":2.3456,"learningRate":0.00001,"etaSeconds":330}}
```

**Event Types:**
- `trainingStart` - Job started
- `trainingProgress` - Step progress
- `trainingCompleted` - Job completed
- `trainingCanceled` - Job cancelled
- `error` - Error occurred

---

## Geometry Commands

### `tc geometry validate` - Validate Geometry Math

**Purpose:** Validate geometry math invariants (GW distance, traversal coherence, path signatures).

**Usage:**
```bash
tc geometry validate --output json
tc geometry validate --include-fixtures --file geometry-validation.json --output json
```

**Flags:**
- `--include-fixtures` - Include deterministic fixtures for reproducibility
- `--file <path>` - Write JSON report to file

**JSON Output:**
```json
{
  "_schema": "tc.geometry.validation.v1",
  "suiteVersion": "1.0",
  "timestamp": "2025-11-30T12:00:00Z",
  "passed": true,
  "config": {
    "includeFixtures": false,
    "thresholds": {
      "identityDistanceMax": 0.000001,
      "permutationDistanceMax": 0.02,
      "symmetryDeltaMax": 0.001,
      "couplingMassErrorMax": 0.02,
      "traversalSelfCorrelationMin": 0.999,
      "traversalPerturbedCorrelationMax": 0.995,
      "signatureSimilarityMin": 0.999,
      "frechetDistanceMax": 0.00001
    },
    "gromovWasserstein": {
      "epsilon": 0.05,
      "epsilonMin": 0.005,
      "epsilonDecay": 0.97,
      "maxOuterIterations": 60,
      "minOuterIterations": 4,
      "maxInnerIterations": 150,
      "convergenceThreshold": 0.000001,
      "relativeObjectiveThreshold": 0.000001,
      "useSquaredLoss": true
    }
  },
  "gromovWasserstein": {
    "distanceIdentity": 0.0,
    "distancePermutation": 0.0123,
    "symmetryDelta": 0.0003,
    "maxRowMassError": 0.0002,
    "maxColumnMassError": 0.0002,
    "converged": true,
    "iterations": 20,
    "passed": true
  },
  "traversalCoherence": {
    "selfCorrelation": 0.9999,
    "perturbedCorrelation": 0.994,
    "transitionCount": 6,
    "pathCount": 2,
    "passed": true
  },
  "pathSignature": {
    "signatureSimilarity": 0.9999,
    "signedArea": 0.5,
    "signatureNorm": 1.4,
    "frechetDistance": 0.0,
    "passed": true
  },
  "fixtures": null
}
```

---

### `tc geometry path detect` - Detect Computational Gates

**Purpose:** Detect computational gates in text or model output.

**Usage:**
```bash
tc geometry path detect "def sum_all(arr): return sum(arr)" --output json
tc geometry path detect "Write a fibonacci function" --model /path/to/model --output json
```

**Args:**
- `<text>` - Text to analyze, or prompt if `--model` is provided

**Flags:**
- `--model <path|id>` - Optional model path to generate a response before detection
- `--threshold <float>` - Detection threshold (default: 0.55)
- `--file <path>` - Write JSON output to file

**JSON Output:**
```json
{
  "modelID": "input-text",
  "promptID": "cli-input",
  "responseText": "def sum_all(arr): return sum(arr)",
  "detectedGates": [
    {
      "gateID": "gate-reduce",
      "gateName": "Reduce",
      "confidence": 0.88,
      "characterSpan": { "lowerBound": 0, "upperBound": 40 },
      "triggerText": "def sum_all(arr): return sum(arr)",
      "localEntropy": null
    }
  ],
  "meanConfidence": 0.88,
  "timestamp": "2025-11-30T12:00:00Z"
}
```

---

### `tc geometry path compare` - Compare Reasoning Paths

**Purpose:** Compare computational gate sequences between two text samples or model responses.

**Usage:**
```bash
tc geometry path compare --text-a "def f(x): return x+1" --text-b "f = lambda x: x+1" --output json
tc geometry path compare --model-a /path/model1 --model-b /path/model2 --prompt "Write sum function" --output json
```

**Flags:**
- `--text-a <text>` - First text sample
- `--text-b <text>` - Second text sample
- `--model-a <path|id>` - First model path
- `--model-b <path|id>` - Second model path
- `--prompt <text>` - Prompt to send to both models
- `--threshold <float>` - Detection threshold (default: 0.55)
- `--file <path>` - Write JSON output to file

**JSON Output:**
```json
{
  "modelA": "text-a",
  "modelB": "text-b",
  "pathA": ["gate-lookup", "gate-compose"],
  "pathB": ["gate-lookup", "gate-compose"],
  "rawDistance": 0.0,
  "normalizedDistance": 0.0,
  "alignmentCount": 2
}
```

---

### `tc geometry training status` - Live Geometry Metrics

**Purpose:** Summarize geometric training metrics (flatness, gradient SNR, circuit breaker severity).

**Usage:**
```bash
tc geometry training status --job <job-id> --format full --output json
tc geometry training status --job <job-id> --format summary --output json
```

**Flags:**
- `--job <id>` - Training job identifier
- `--format <full|summary>` - Output detail level (default: `full`)
- `--ai` - Include AI next-actions hints

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "step": 120,
  "flatnessScore": 0.78,
  "flatnessAssessment": "Flat (good)",
  "gradientSNR": 5.4,
  "snrAssessment": "Adequate",
  "circuitBreakerSeverity": 0.21,
  "circuitBreakerTripped": false,
  "activeLayers": ["layer1", "layer3"],
  "perLayerGradientNorms": {"layer1": 0.52, "layer3": 0.31}
}
```

---

### `tc geometry training history` - Geometry Metric Trends

**Purpose:** Return metric history captured during training.

**Usage:**
```bash
tc geometry training history --job <job-id> --output json
```

**JSON Output:**
```json
{
  "_schema": "tc.geometry.training_history.v1",
  "jobId": "job-abc123",
  "startStep": 1,
  "endStep": 120,
  "sampleCount": 12,
  "flatnessHistory": [{"step": 10, "value": 0.8}],
  "snrHistory": [{"step": 10, "value": 5.1}],
  "parameterDivergenceHistory": [{"step": 10, "value": 0.02}]
}
```

---

### `tc geometry training levels` - Instrumentation Presets

**Purpose:** List the geometric instrumentation levels and their collected metrics.

**Usage:**
```bash
tc geometry training levels --output json
```

**JSON Output:**
```json
{
  "levels": [
    {
      "name": "moderate",
      "description": "Moderate - adds curvature estimation",
      "metricsCollected": ["Gradient norms", "Parameter divergence", "Curvature estimation (Hessian trace)"]
    }
  ]
}
```

---

### `tc geometry safety circuit-breaker` - Safety Circuit Evaluation

**Purpose:** Evaluate circuit breaker severity using entropy/refusal/persona drift signals.

**Usage:**
```bash
tc geometry safety circuit-breaker --job <job-id> --output json
tc geometry safety circuit-breaker --entropy 0.6 --persona-drift 0.4 --oscillation --output json
```

**JSON Output:**
```json
{
  "tripped": false,
  "severity": 0.42,
  "state": "warning",
  "interpretation": "Elevated concern - close monitoring recommended",
  "recommendedAction": "Monitor more closely"
}
```

---

### `tc geometry safety persona` - Persona Drift Analysis

**Purpose:** Summarize persona drift and refusal proximity metrics.

**Usage:**
```bash
tc geometry safety persona --job <job-id> --output json
```

**JSON Output:**
```json
{
  "jobId": "job-abc123",
  "overallDriftMagnitude": 0.28,
  "driftAssessment": "moderate",
  "driftingTraits": ["curiosity"],
  "refusalDistance": 0.45,
  "isApproachingRefusal": false
}
```

---

### `tc geometry adapter sparsity` - DARE Sparsity Analysis

**Purpose:** Measure adapter sparsity to guide DARE merges.

**Usage:**
```bash
tc geometry adapter sparsity --checkpoint /path/adapter.npz --base /path/base.npz --output json
```

**JSON Output:**
```json
{
  "checkpointPath": "/path/adapter.npz",
  "baseModelPath": "/path/base.npz",
  "effectiveSparsity": 0.82,
  "qualityAssessment": "good",
  "interpretation": "Effective sparsity 82.00% (good). Recommended drop rate 0.85."
}
```

---

### `tc geometry adapter decomposition` - DoRA Decomposition

**Purpose:** Decompose adapter updates into magnitude vs direction changes.

**Usage:**
```bash
tc geometry adapter decomposition --checkpoint /path/adapter.npz --base /path/base.npz --output json
```

**JSON Output:**
```json
{
  "checkpointPath": "/path/adapter.npz",
  "baseModelPath": "/path/base.npz",
  "magnitudeChangeRatio": 0.12,
  "directionalDrift": 0.08,
  "learningType": "balanced",
  "interpretation": "Adapter combines scaling and rotation (balanced change)"
}
```

---

## Error Codes

All CLI errors include:
- `code` - Error code (e.g., `TC-5001`)
- `title` - Human-readable title
- `detail` - Detailed error message
- `hint` - Suggested remediation
- `docsUrl` - Documentation URL (if applicable)

**Example Error (JSON):**
```json
{
  "error": {
    "code": "TC-5030",
    "title": "Insufficient Memory",
    "detail": "Available memory: 12.5 GB. Required: 16GB",
    "hint": "Reduce batch size or close other GPU-intensive applications",
    "docsUrl": null,
    "traceId": "abc123"
  }
}
```

**Exit Codes:**
- `0` - Success
- `1` - General error
- `2` - Usage error (invalid arguments)
- `3` - Validation error
- `4` - Access error (permissions, file not found)
- `5` - Resource error (insufficient memory, Metal unavailable)
- `6` - Model error (download failed, load failed)

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TC_OUTPUT` | Default output format (`json`, `yaml`, `text`) | `text` |
| `TC_NO_COLOR` | Disable color output (`1` = disabled) | - |
| `TC_NO_PROMPT` | Disable interactive prompts | - |
| `TC_ALLOW_ALL` | Auto-confirm all prompts | - |

---

## Common AI Agent Workflows

### Complete Training Pipeline
```bash
# 1. Download model
tc model fetch Qwen/Qwen2.5-0.5B-Instruct \
  --auto-register \
  --alias qwen-0.5b \
  --architecture qwen2 \
  --output json

# 2. Start training
JOB_ID=$(tc train start \
  --model qwen-0.5b \
  --dataset data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --output json | jq -r '.jobId')

# 3. Monitor progress
tc train status $JOB_ID --output json

# 4. Export trained model
tc train export \
  --job $JOB_ID \
  --format gguf \
  --output-path model.gguf \
  --output json
```

### Batch Export Checkpoints
```bash
# List all checkpoints
tc checkpoint list --output json | jq -r '.checkpoints[].filePath' | while read checkpoint; do
  # Export each to GGUF
  tc checkpoint export "$checkpoint" --format gguf --output-path "${checkpoint%.safetensors}.gguf" --output json
done
```

### Monitor Training (NDJSON Stream)
```bash
# Stream training events
tc train start --model llama-3.2-1B --dataset data.jsonl --stream | while read event; do
  # Parse NDJSON event
  TYPE=$(echo "$event" | jq -r '.type')

  if [ "$TYPE" = "trainingProgress" ]; then
    STEP=$(echo "$event" | jq -r '.data.step')
    LOSS=$(echo "$event" | jq -r '.data.loss')
    echo "Step $STEP: Loss $LOSS"
  fi
done
```

---

## Notes for AI Agents

1. **Always use `--output json`** for machine-readable output
2. **Use `--yes` or `--no-prompt`** to avoid interactive prompts
3. **Parse exit codes** to detect failures
4. **NDJSON streaming** provides real-time progress for long operations
5. **All paths** are expanded (`~` becomes home directory)
6. **Concurrent execution** is safe - CLI uses `TrainingResourceGuard` to serialize GPU access
7. **Job state** persists in SwiftData - jobs survive CLI restarts
8. **Error messages** include actionable hints in `hint` field
