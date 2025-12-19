# TrainingCypher MCP Server Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**MCP SDK:** Official MCP Swift SDK v0.10.0+
**Protocol:** MCP Specification 2025-03-26

## Overview

The TrainingCypher MCP server exposes on-device ML training capabilities to AI agents and MCP-enabled clients. It provides programmatic access to:

- **Training Control** - Start, pause, resume, cancel training jobs
- **Model Management** - List and query registered models
- **Inference** - Run text generation with fine-tuned models
- **System Monitoring** - Check GPU, memory, and MLX status

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client                               │
│  (Claude Desktop, VS Code, Cursor, Custom Agent)           │
└─────────────────────┬───────────────────────────────────────┘
                      │ STDIO Transport (JSON-RPC)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              trainingcypher-mcp Server                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Tools     │  │  Resources  │  │   Service   │         │
│  │  Handlers   │  │  Handlers   │  │   Bridge    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  TrainingCypherCore                         │
│  TrainingController │ ModelService │ InferenceService │ ... │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MLX Framework                            │
│               Metal GPU │ Unified Memory                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- macOS 26.2+
- Xcode 26.2+ with Swift 6.2
- Apple Silicon Mac (M-series, arm64)

### Build and Install

```bash
cd tools/trainingcypher-mcp
./install-mcp.sh
```

This builds a release binary and installs it to `~/.local/bin/trainingcypher-mcp`.

### Manual Build

```bash
cd tools/trainingcypher-mcp
swift build -c release
# Binary at: .build/release/trainingcypher-mcp
```

### Verify Installation

```bash
trainingcypher-mcp --version
# Output: trainingcypher-mcp version 1.0.0

trainingcypher-mcp --help
# Shows all tools and resources
```

## Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "trainingcypher": {
      "command": "/Users/YOUR_USERNAME/.local/bin/trainingcypher-mcp"
    }
  }
}
```

### Claude Code (.mcp.json)

```json
{
  "mcpServers": {
    "trainingcypher": {
      "command": "trainingcypher-mcp",
      "env": {
        "TC_LOG_LEVEL": "info"
      }
    }
  }
}
```

### VS Code / Cursor

Add to MCP configuration:

```json
{
  "mcp": {
    "servers": {
      "trainingcypher": {
        "command": "/path/to/trainingcypher-mcp"
      }
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TC_LOG_LEVEL` | Log verbosity: debug, info, warning, error | `info` |
| `TC_JOB_STORE` | Override job persistence store path | Auto-detected |
| `TC_MCP_PROFILE` | Tool profile for token optimization (see below) | `full` |

### Tool Profiles (Token Optimization)

TrainingCypher supports server-side tool filtering via the `TC_MCP_PROFILE` environment variable. This reduces token usage by only exposing tools relevant to your workflow.

| Profile | Tools | Estimated Tokens | Use Case |
|---------|-------|------------------|----------|
| `full` | All 23 tools | ~2,600 | Complete access (default) |
| `training` | 22 tools | ~2,450 | Training workflows |
| `inference` | 4 tools | ~700 | Inference only |
| `monitoring` | 6 tools | ~700 | Read-only monitoring |

**Profile Contents:**

```
training:
  tc_inventory, tc_train_start, tc_job_status, tc_job_list, tc_job_cancel,
  tc_job_pause, tc_job_resume, tc_system_status, tc_validate_train,
  tc_estimate_train, tc_dataset_validate, tc_model_fetch, tc_model_list,
  tc_model_search, tc_checkpoint_export, tc_geometry_validate

inference:
  tc_inventory, tc_model_list, tc_infer, tc_system_status

monitoring:
  tc_inventory, tc_job_status, tc_job_list, tc_job_detail, tc_system_status,
  tc_geometry_validate
```

**Example Configuration:**

```json
{
  "mcpServers": {
    "trainingcypher": {
      "command": "trainingcypher-mcp",
      "env": {
        "TC_MCP_PROFILE": "training"
      }
    }
  }
}
```

### Tool Annotations

All tools include MCP annotations for AI client optimization:

| Annotation | Meaning |
|------------|---------|
| `readOnlyHint: true` | Safe to call anytime, no side effects |
| `destructiveHint: true` | Cancels jobs or deletes data |
| `idempotentHint: true` | Safe to retry with same arguments |
| `openWorldHint: true` | Interacts with external systems (network) |

**Annotation Categories:**

| Category | Tools | Annotations |
|----------|-------|-------------|
| Read-only | `tc_inventory`, `tc_job_status`, `tc_job_list`, `tc_job_detail`, `tc_model_list`, `tc_system_status`, `tc_validate_train`, `tc_estimate_train`, `tc_dataset_validate`, `tc_geometry_validate` | `readOnly=true, idempotent=true` |
| Mutating | `tc_train_start`, `tc_job_pause`, `tc_job_resume`, `tc_infer`, `tc_checkpoint_export` | `readOnly=false` |
| Destructive | `tc_job_cancel` | `destructive=true, idempotent=true` |
| Network | `tc_model_fetch`, `tc_model_search` | `openWorld=true, idempotent=true` |

---

## Tools Reference

### tc_inventory

**Purpose:** Get complete system state in a single call. **Always call this first** to understand what's available.

**Category:** Read-only discovery

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Output:**
```json
{
  "models": [
    {
      "id": "llama-3.2-1b",
      "alias": "llama-3.2-1b",
      "format": "safetensors",
      "sizeBytes": 2147483648,
      "path": "/Users/.../models/llama-3.2-1b"
    }
  ],
  "datasets": [
    {
      "id": "uuid",
      "name": "my-dataset",
      "path": "/path/to/dataset.jsonl",
      "sizeBytes": 1048576,
      "exampleCount": 1000
    }
  ],
  "checkpoints": [
    {
      "jobId": "job-uuid",
      "step": 500,
      "loss": 0.523,
      "path": "/path/to/checkpoint"
    }
  ],
  "jobs": [
    {
      "jobId": "uuid",
      "status": "running",
      "progress": 0.45,
      "modelId": "llama-3.2-1b",
      "datasetPath": "/path/to/dataset.jsonl"
    }
  ],
  "workspace": {
    "cwd": "/Users/...",
    "jobStore": "/path/to/job-store"
  },
  "mlxVersion": "0.25.3",
  "policies": {
    "safeGPU": true,
    "evalRequired": true,
    "tokenizerSplit": true,
    "logging": "oslog"
  }
}
```

**Example Usage (AI Agent):**
```
Call tc_inventory first to see what models and datasets are available before starting training.
```

---

### tc_train_start

**Purpose:** Start a new training job.

**Category:** MUTATING - creates files, acquires GPU

**Preconditions:**
- No other training job currently running (check `tc_job_list` first)
- Model must be registered (check `tc_inventory`)
- Dataset file must exist at specified path

**Side Effects:**
- Creates checkpoint files in `~/Library/Application Support/TrainingCypher/checkpoints/`
- Acquires exclusive GPU access via TrainingResourceGuard
- Logs to `~/Library/Logs/TrainingCypher/`
- Persists job state to job store

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "model": {
      "type": "string",
      "description": "Model identifier or alias to fine-tune"
    },
    "dataset": {
      "type": "string",
      "description": "Path to training dataset (JSONL format)"
    },
    "epochs": {
      "type": "integer",
      "description": "Number of training epochs (default: 3)"
    },
    "learningRate": {
      "type": "number",
      "description": "Learning rate (default: 1e-5, max recommended: 1e-3)"
    },
    "batchSize": {
      "type": "integer",
      "description": "Batch size (auto-calculated if omitted based on memory)"
    },
    "sequenceLength": {
      "type": "integer",
      "description": "Maximum sequence length (default: 2048)"
    },
    "loraRank": {
      "type": "integer",
      "description": "LoRA rank for parameter-efficient fine-tuning"
    },
    "loraAlpha": {
      "type": "number",
      "description": "LoRA alpha scaling factor"
    }
  },
  "required": ["model", "dataset"]
}
```

**Output:**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "batchSize": 4
}
```

**Example:**
```json
{
  "model": "llama-3.2-1b",
  "dataset": "/Users/me/datasets/my-training-data.jsonl",
  "epochs": 3,
  "learningRate": 5e-5,
  "loraRank": 16,
  "loraAlpha": 32
}
```

**Error Cases:**
- `"Another job is already running"` - Wait or cancel existing job
- `"Model not found"` - Register model first or check alias
- `"Dataset not found"` - Verify file path exists
- `"Insufficient memory"` - Reduce batch size or sequence length

---

### tc_job_status

**Purpose:** Get detailed status of a specific training job.

**Category:** Read-only

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "jobId": {
      "type": "string",
      "description": "Job identifier (UUID)"
    }
  },
  "required": ["jobId"]
}
```

**Output:**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.45,
  "currentEpoch": 2,
  "totalEpochs": 3,
  "currentStep": 450,
  "totalSteps": 1000,
  "loss": 0.523,
  "learningRate": 5e-5,
  "tokensPerSecond": 1250.5,
  "etaSeconds": 180,
  "memoryUsageMB": 8192,
  "startedAt": "2025-11-26T10:30:00Z",
  "modelId": "llama-3.2-1b",
  "datasetPath": "/path/to/dataset.jsonl"
}
```

---

### tc_job_list

**Purpose:** List all training jobs with optional filtering.

**Category:** Read-only

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "description": "Filter by status: queued, running, paused, completed, failed, canceled"
    },
    "activeOnly": {
      "type": "boolean",
      "description": "Only show active jobs (queued, running, paused)"
    }
  }
}
```

**Output:**
```json
[
  {
    "jobId": "uuid-1",
    "status": "running",
    "progress": 0.45,
    "modelId": "llama-3.2-1b",
    "datasetPath": "/path/to/dataset.jsonl"
  },
  {
    "jobId": "uuid-2",
    "status": "completed",
    "progress": 1.0,
    "modelId": "qwen-0.5b",
    "datasetPath": "/path/to/other.jsonl"
  }
]
```

---

### tc_job_cancel

**Purpose:** Cancel a running or queued training job.

**Category:** MUTATING - stops training, releases GPU

**Side Effects:**
- Stops training loop immediately
- Releases GPU via TrainingResourceGuard
- Preserves last checkpoint (if any)
- Updates job status to "canceled"

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "jobId": {
      "type": "string",
      "description": "Job identifier (UUID)"
    }
  },
  "required": ["jobId"]
}
```

**Output:**
```json
{
  "status": "canceled",
  "jobId": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### tc_job_pause

**Purpose:** Pause a running training job.

**Category:** MUTATING - pauses training, retains GPU lock

**Side Effects:**
- Pauses training loop at next safe point
- Saves checkpoint
- Retains GPU reservation (other jobs cannot start)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "jobId": {
      "type": "string",
      "description": "Job identifier (UUID)"
    }
  },
  "required": ["jobId"]
}
```

**Output:**
```json
{
  "status": "paused",
  "jobId": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### tc_job_resume

**Purpose:** Resume a paused training job.

**Category:** MUTATING - resumes training

**Preconditions:**
- Job must be in "paused" state
- GPU must still be available

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "jobId": {
      "type": "string",
      "description": "Job identifier (UUID)"
    }
  },
  "required": ["jobId"]
}
```

**Output:**
```json
{
  "status": "resumed",
  "jobId": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### tc_model_list

**Purpose:** List all registered models with metadata.

**Category:** Read-only

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Output:**
```json
[
  {
    "id": "llama-3.2-1b",
    "alias": "llama-3.2-1b",
    "format": "safetensors",
    "sizeBytes": 2147483648,
    "path": "/Users/.../models/llama-3.2-1b",
    "architecture": "llama",
    "parameterCount": 1000000000
  }
]
```

---

### tc_model_search

**Purpose:** Search HuggingFace Hub for MLX-compatible models with memory-fit indicators.

**Category:** Network

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Free-text search query (e.g., 'llama 3', 'qwen 7b')"
    },
    "author": {
      "type": "string",
      "description": "Filter by author/organization (e.g., 'Qwen', 'meta-llama')"
    },
    "library": {
      "type": "string",
      "description": "Library filter: mlx (default), safetensors, pytorch, any"
    },
    "quant": {
      "type": "string",
      "description": "Quantization filter: 4bit, 8bit, any"
    },
    "sort": {
      "type": "string",
      "description": "Sort by: downloads (default), likes, lastModified, trending"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum results to return (default: 20, max: 100)"
    },
    "cursor": {
      "type": "string",
      "description": "Pagination cursor for next page"
    }
  }
}
```

**Output:**
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

---

### tc_infer

**Purpose:** Run inference with a model.

**Category:** MUTATING - loads model, uses GPU

**Preconditions:**
- Model or checkpoint must exist
- No training job currently running (shares GPU)

**Side Effects:**
- Loads model into GPU memory
- Generates text (may take significant time)
- Unloads model after completion

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "model": {
      "type": "string",
      "description": "Model identifier, alias, or checkpoint path"
    },
    "prompt": {
      "type": "string",
      "description": "Input prompt for text generation"
    },
    "maxTokens": {
      "type": "integer",
      "description": "Maximum tokens to generate (default: 512)"
    },
    "temperature": {
      "type": "number",
      "description": "Sampling temperature 0.0-2.0 (default: 0.7)"
    },
    "topP": {
      "type": "number",
      "description": "Top-p sampling (default: 0.95)"
    }
  },
  "required": ["model", "prompt"]
}
```

**Output:**
```json
{
  "modelId": "llama-3.2-1b",
  "prompt": "Explain quantum computing:",
  "response": "Quantum computing is a type of computation...",
  "tokenCount": 150,
  "tokensPerSecond": 45.2,
  "timeToFirstToken": 0.823,
  "totalDuration": 3.32
}
```

**Example:**
```json
{
  "model": "llama-3.2-1b",
  "prompt": "Write a Python function to calculate fibonacci numbers:",
  "maxTokens": 256,
  "temperature": 0.3
}
```

---

### tc_system_status

**Purpose:** Get system readiness and environment information.

**Category:** Read-only

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Output:**
```json
{
  "machineName": "Mac Studio",
  "unifiedMemoryGB": 64,
  "mlxVersion": "0.25.3",
  "readinessScore": 95,
  "scoreBreakdown": {
    "totalScore": 95,
    "datasetScore": 100,
    "memoryFitScore": 90,
    "systemPressureScore": 100,
    "mlxHealthScore": 100,
    "storageScore": 85,
    "preflightScore": 95,
    "band": "excellent"
  },
  "blockers": []
}
```

**Blocker Example:**
```json
{
  "blockers": [
    {
      "id": "no-dataset",
      "description": "No training dataset selected",
      "severity": "critical",
      "fixAction": "navigateTo:datasets"
    }
  ]
}
```

---

### tc_geometry_validate

**Purpose:** Run the deterministic geometry validation suite (GW distance, traversal coherence, path signatures).

**Category:** Read-only

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "includeFixtures": {
      "type": "boolean",
      "description": "Include deterministic fixtures for reproduction (default: false)"
    }
  }
}
```

**Output:**
```json
{
  "_schema": "tc.geometry.validation.v1",
  "suiteVersion": "1.0",
  "timestamp": "2025-11-30T12:00:00Z",
  "passed": true,
  "config": {
    "includeFixtures": false
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

## Resources Reference

Resources provide read-only access to TrainingCypher state via standard MCP resource URIs.

### tc://models

**Description:** All registered models with metadata

**MIME Type:** `application/json`

**Content:** Same format as `tc_model_list` output

### tc://jobs

**Description:** All training jobs with status

**MIME Type:** `application/json`

**Content:** Same format as `tc_job_list` output (no filters)

### tc://checkpoints

**Description:** All training checkpoints

**MIME Type:** `application/json`

**Content:**
```json
[
  {
    "jobId": "uuid",
    "step": 500,
    "loss": 0.523,
    "path": "/path/to/checkpoint"
  }
]
```

### tc://datasets

**Description:** All registered datasets

**MIME Type:** `application/json`

**Content:**
```json
[
  {
    "id": "uuid",
    "name": "my-dataset",
    "path": "/path/to/dataset.jsonl",
    "sizeBytes": 1048576,
    "exampleCount": 1000
  }
]
```

### tc://system

**Description:** System readiness and environment info

**MIME Type:** `application/json`

**Content:** Same format as `tc_system_status` output

---

## AI Agent Usage Patterns

### Pattern 1: Discovery First

**Always start with `tc_inventory`** to understand available resources before taking actions.

```
1. tc_inventory → Learn what models, datasets, jobs exist
2. tc_system_status → Verify system is ready for training
3. tc_train_start → Start training with validated inputs
4. tc_job_status (poll) → Monitor progress
```

### Pattern 2: Safe Training Workflow

```
# Check preconditions
inventory = tc_inventory()
if any(job.status in ["running", "paused"] for job in inventory.jobs):
    # Either wait or cancel existing job
    tc_job_cancel(existing_job_id)

# Verify model exists
if model_alias not in [m.alias for m in inventory.models]:
    raise "Model not registered"

# Start training
result = tc_train_start(model=model_alias, dataset=dataset_path)
job_id = result.jobId

# Monitor (poll every 30s)
while True:
    status = tc_job_status(job_id)
    if status.status in ["completed", "failed", "canceled"]:
        break
    # Report progress: {status.progress * 100}%, loss: {status.loss}
    wait(30)
```

### Pattern 3: Inference After Training

```
# Find latest checkpoint for job
inventory = tc_inventory()
job_checkpoints = [c for c in inventory.checkpoints if c.jobId == job_id]
latest = max(job_checkpoints, key=lambda c: c.step)

# Run inference with checkpoint
result = tc_infer(
    model=latest.path,  # Use checkpoint path
    prompt="Your prompt here",
    temperature=0.7
)
```

### Pattern 4: Error Recovery

```
try:
    tc_train_start(model=model, dataset=dataset)
except MCPError as e:
    if "Another job is already running" in str(e):
        # Check if we should wait or cancel
        jobs = tc_job_list(activeOnly=True)
        # Decide based on job progress/priority
    elif "Insufficient memory" in str(e):
        # Retry with smaller batch size
        tc_train_start(model=model, dataset=dataset, batchSize=1)
```

### Anti-Patterns to Avoid

1. **Starting training without checking existing jobs** - Will fail if another job is running
2. **Not validating model/dataset existence** - Use `tc_inventory` first
3. **Polling too frequently** - 30-second intervals are sufficient for training
4. **Ignoring system status blockers** - Check `tc_system_status` before training

---

## Tool Safety & Boundaries

TrainingCypher exposes tools that are either **read-only** (safe to call freely) or **mutating** (change system state, use GPU, or create files). Agents should respect these boundaries to avoid conflicts and unintended side effects.

### Read-Only Tools (Safe)

These tools never modify state and are safe to call whenever context is needed:

- `tc_inventory` – Discovery: models, datasets, checkpoints, jobs, workspace, MLX version, and safety policies.
- `tc_job_status` – Detailed status for a single job.
- `tc_job_list` – List jobs (optionally filtered by status / activeOnly).
- `tc_model_list` – Registered models with metadata.
- `tc_system_status` – System readiness (Metal, memory fit, storage, MLX health).
- `tc_geometry_validate` – Deterministic geometry validation suite.

**Recommended usage:**

- Call `tc_inventory` at the start of every new workflow.
- Use `tc_job_list(activeOnly=true)` to decide whether mutating tools are safe to call.
- Call `tc_system_status` before starting training to surface blockers early.

### Mutating Tools (Use with Preconditions)

These tools change state, use GPU, or create files. Agents must check preconditions first:

- `tc_train_start` – Start a training job (creates checkpoints, acquires exclusive GPU via `TrainingResourceGuard`).
- `tc_job_cancel` – Cancel a running or queued job (releases GPU, preserves latest checkpoint).
- `tc_job_pause` – Pause training (saves checkpoint, retains GPU lock).
- `tc_job_resume` – Resume a paused job (requires GPU availability).
- `tc_infer` – Run inference (loads model into GPU memory, generates text, then unloads).

#### Three-Tier Boundaries (for Agents)

- ✅ **Always**
  - Use `tc_inventory` + `tc_system_status` for discovery and readiness checks.
  - Use read-only tools (`tc_job_list`, `tc_job_status`, `tc_model_list`) to understand state before deciding on actions.

- ⚠️ **Ask or Check First**
  - `tc_train_start` – Only call after:
    - `tc_job_list(activeOnly=true)` shows no running/paused jobs, and
    - `tc_system_status` reports no critical blockers.
  - `tc_infer` – Only call when no training job is running (shares GPU).
  - `tc_job_pause` / `tc_job_resume` – Ensure the job is in an appropriate state.

- 🚫 **Never (Without Explicit Human Intent)**
  - Call mutating tools in a tight loop or without inspecting their effects.
  - Start training on arbitrary datasets not confirmed by the user.
  - Cancel or pause jobs without surfacing the impact to the user.

Agents should treat mutating tools as **explicit user actions** (e.g., “start training”, “cancel job”), not as background maintenance operations.

### Safe Workflows by Tool Category

- **Discovery-First Training Workflow**
  1. `tc_inventory` – Understand models, datasets, existing jobs.
  2. `tc_job_list(activeOnly=true)` – Confirm no conflicting jobs.
  3. `tc_system_status` – Check readiness and blockers.
  4. `tc_train_start(...)` – Start training only after conditions are met.
  5. `tc_job_status` (poll) – Monitor progress.

- **Safe Inference Workflow**
  1. `tc_inventory` – Find latest checkpoints/models.
  2. `tc_job_list(activeOnly=true)` – Ensure no training job is running.
  3. `tc_infer(model=checkpointPathOrAlias, prompt=...)` – Run inference.

- **Graceful Shutdown / Cleanup**
  - Use `tc_job_cancel` only after confirming with the user that the job should be stopped.
  - Use `tc_job_pause` when the user wants to pause but preserve GPU reservation and checkpoints.

These patterns match the underlying TrainingCypherCore invariants (exclusive GPU via `TrainingResourceGuard`, SafeGPU serialization, and MLX memory constraints) and are the safest way for agents to operate.

---

## Human Usage Patterns

### Quick Status Check

```bash
# In Claude Desktop or similar MCP client
> What training jobs are running?
# Agent calls tc_job_list(activeOnly=true)

> What's the status of job abc-123?
# Agent calls tc_job_status(jobId="abc-123")
```

### Start Training via Natural Language

```
> Train llama-3.2-1b on my customer-support.jsonl dataset for 5 epochs
# Agent:
#   1. tc_inventory() - Verify model exists
#   2. tc_system_status() - Check system ready
#   3. tc_train_start(model="llama-3.2-1b", dataset="customer-support.jsonl", epochs=5)
```

### Interactive Inference

```
> Use my fine-tuned model to answer: "How do I reset my password?"
# Agent:
#   1. tc_inventory() - Find latest checkpoint
#   2. tc_infer(model=checkpoint_path, prompt="How do I reset my password?")
```

---

## Error Handling

### Error Response Format

All errors are returned as MCP errors with structured messages:

```json
{
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required argument 'model'"
  }
}
```

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `Missing required argument` | Required parameter not provided | Include all required parameters |
| `Unknown tool` | Invalid tool name | Check tool name spelling |
| `Another job is already running` | TrainingResourceGuard lock held | Wait or cancel existing job |
| `Model not found` | Model alias doesn't exist | Register model or check alias |
| `Dataset not found` | File path doesn't exist | Verify file path |
| `Insufficient memory` | Not enough GPU/RAM | Reduce batch size or sequence length |
| `Server deallocated` | Internal server error | Restart MCP server |

---

## Troubleshooting

### Server Won't Start

```bash
# Check if another instance is running
pgrep -f trainingcypher-mcp

# Check logs
TC_LOG_LEVEL=debug trainingcypher-mcp 2>&1 | head -50
```

### Connection Issues (Claude Desktop)

1. Verify config path: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Verify binary exists: `ls -la ~/.local/bin/trainingcypher-mcp`
3. Restart Claude Desktop after config changes
4. Check Claude Desktop logs for MCP errors

### Training Fails to Start

```
1. Call tc_inventory() - Verify model is registered
2. Call tc_job_list(activeOnly=true) - Check no job running
3. Call tc_system_status() - Check for blockers
4. Verify dataset path exists and is readable
```

### Slow Inference

- Check system memory pressure with `tc_system_status`
- Reduce `maxTokens` parameter
- Ensure no training job is competing for GPU

### Debugging

Set verbose logging:
```bash
TC_LOG_LEVEL=debug trainingcypher-mcp
```

Logs output to stderr (STDIO MCP requires clean stdout).

---

## Future Considerations

### MCP 2025-06-18: outputSchema and structuredContent

The MCP 2025-06-18 specification adds `outputSchema` to tool definitions and `structuredContent` to responses. When the MCP Swift SDK updates to support these features, TrainingCypher will adopt them for more efficient structured responses.

**Current approach:** All responses include a `_schema` field (e.g., `tc.train.start.v1`) documenting the response structure.

**Future approach:** Add formal JSON Schema `outputSchema` to each tool, enabling AI clients to validate and parse responses more efficiently.

### Code Mode Pattern (Long-term)

Anthropic's research shows a "Code Mode" pattern can reduce tool token overhead by 98.7% for complex workflows. Instead of presenting tools as individual definitions, tools are presented as a filesystem of callable code files:

```
tc://code/training.swift       # Training workflow functions
tc://code/inference.swift      # Inference functions
tc://code/monitoring.swift     # Status and monitoring
```

**Benefits:**
- Massive token reduction for large tool sets
- Familiar code-centric interface for AI agents
- Natural composition of multi-step workflows

**Considerations:**
- Requires client support for resource-as-code pattern
- More complex implementation
- May not suit all AI clients equally

This pattern is documented here for future evaluation when TrainingCypher's tool count grows significantly.

---

## Version History

### 1.1.0 (2025-11-26)

- Added `TC_MCP_PROFILE` environment variable for server-side tool filtering
- Added tool profiles: `full`, `training`, `inference`, `monitoring`
- Added MCP tool annotations (readOnlyHint, destructiveHint, idempotentHint, openWorldHint)
- Fixed tc_infer to use checkpoint paths only (narrowed contract)
- Fixed tc_infer model cleanup to use `defer` for guaranteed unload on all exits
- Added 5 new tools: tc_validate_train, tc_estimate_train, tc_dataset_validate, tc_model_fetch, tc_checkpoint_export
- Total: 15 tools, 5 resources

### 1.0.0 (2025-11-26)

- Initial release
- 10 tools: tc_inventory, tc_train_start, tc_job_status, tc_job_list, tc_job_cancel, tc_job_pause, tc_job_resume, tc_model_list, tc_infer, tc_system_status
- 5 resources: tc://models, tc://jobs, tc://checkpoints, tc://datasets, tc://system
- ServiceLifecycle integration for graceful shutdown
- Official MCP Swift SDK v0.10.0+

---

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Project-wide AI agent instructions
- [TrainingCypherPackage/AGENTS.md](../../app/TrainingCypherPackage/AGENTS.md) - Core package agent instructions
- [CLI Documentation](../cli/) - Command-line interface docs
- [MCP Specification](https://spec.modelcontextprotocol.io/) - Official MCP protocol spec
