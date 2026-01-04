# Training Guide

Complete guide to fine-tuning LLMs with ModelCypher.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Preflight Checks](#preflight-checks)
3. [Training Commands](#training-commands)
4. [Job Management](#job-management)
5. [Checkpoint Management](#checkpoint-management)
6. [Hyperparameter Reference](#hyperparameter-reference)
7. [LoRA Configuration](#lora-configuration)
8. [Training Geometry Monitoring](#training-geometry-monitoring)
9. [Common Workflows](#common-workflows)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Training Run

```bash
# 1. Preflight check
mc train preflight \
  --model /path/to/base-model \
  --dataset ./train.jsonl \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --epochs 1 \
  --sequence-length 512 \
  --grad-accum 4 \
  --warmup-steps 100 \
  --weight-decay 0.01 \
  --gradient-checkpointing \
  --mixed-precision \
  --compute-precision bfloat16 \
  --optimizer-type adamw \
  --seed 42 \
  --deterministic \
  --out ./output

# 2. Start training
mc train start \
  --model /path/to/base-model \
  --dataset ./train.jsonl \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --epochs 1 \
  --sequence-length 512 \
  --grad-accum 4 \
  --warmup-steps 100 \
  --weight-decay 0.01 \
  --gradient-checkpointing \
  --mixed-precision \
  --compute-precision bfloat16 \
  --optimizer-type adamw \
  --seed 42 \
  --deterministic \
  --out ./output

# 3. Monitor progress
mc train status <job_id> --follow

# 4. Export result
mc train export --job <job_id> --format safetensors --output-path ./final-model
```

### Dataset Format

Training data must be in JSONL format with a `text` field:

```json
{"text": "User: What is the capital of France?\nAssistant: Paris."}
{"text": "User: Explain gravity.\nAssistant: Gravity is the force..."}
```

Or chat format with messages:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

---

## Preflight Checks

Run preflight to validate configuration and estimate resources before training:

```bash
mc train preflight \
  --model /path/to/model \
  --dataset ./data.jsonl \
  [hyperparameter options] \
  --out ./output
```

**Preflight checks:**
- Model existence and format validation
- Dataset format and accessibility
- Memory estimation vs. available VRAM
- Output directory writability
- Hyperparameter validation

**Output fields:**
| Field | Description |
|-------|-------------|
| `predictedBatchSize` | Effective batch size (batch × grad_accum) |
| `estimatedVRAMUsageBytes` | Estimated VRAM requirement |
| `availableVRAMBytes` | Available GPU memory |
| `canProceed` | Whether training can start |

---

## Training Commands

### mc train start

Start a training job.

```bash
mc train start \
  --model <model_path> \
  --dataset <dataset_path> \
  --learning-rate <float> \
  --batch-size <int> \
  --epochs <int> \
  --sequence-length <int> \
  --grad-accum <int> \
  --warmup-steps <int> \
  --weight-decay <float> \
  --gradient-checkpointing / --no-gradient-checkpointing \
  --mixed-precision / --no-mixed-precision \
  --compute-precision <float32|float16|bfloat16> \
  --optimizer-type <adamw> \
  --seed <int> \
  --deterministic / --stochastic \
  --out <output_dir> \
  [--resume-from <output_dir>] \
  [--lora-rank <int>] \
  [--lora-alpha <float>] \
  [--lora-dropout <float>] \
  [--lora-targets <module> ...] \
  [--detach] \
  [--stream]
```

**Options:**
| Option | Required | Description |
|--------|----------|-------------|
| `--model` | Yes | Base model path or HuggingFace ID |
| `--dataset` | Yes | Training dataset path (JSONL) |
| `--out` | Yes | Output directory |
| `--learning-rate` | Yes | Learning rate |
| `--batch-size` | Yes | Per-device batch size |
| `--epochs` | Yes | Number of epochs |
| `--sequence-length` | Yes | Max sequence length |
| `--grad-accum` | Yes | Gradient accumulation steps |
| `--warmup-steps` | Yes | LR warmup steps |
| `--weight-decay` | Yes | Weight decay coefficient |
| `--gradient-checkpointing` | Yes | Enable gradient checkpointing |
| `--mixed-precision` | Yes | Enable mixed precision |
| `--compute-precision` | Yes | Compute dtype |
| `--optimizer-type` | Yes | Optimizer (adamw only) |
| `--seed` | Yes | Random seed |
| `--deterministic` | Yes | Deterministic training |
| `--resume-from` | No | Resume from output directory with checkpoints |
| `--detach` | No | Run in background |
| `--stream` | No | Stream progress events |

### mc train status

Get training job status.

```bash
mc train status <job_id>
mc train status <job_id> --follow    # Poll until complete
mc train status <job_id> --stream    # Attach to event stream
```

### mc train pause / resume

Pause and resume training jobs.

```bash
mc train pause <job_id>
mc train resume <job_id>
```

### mc train cancel

Cancel a running job.

```bash
mc train cancel <job_id>
```

### mc train export

Export trained model or job output.

```bash
# Export from job
mc train export --job <job_id> --format safetensors --output-path ./model

# Export from model directory
mc train export --model ./fine-tuned --format safetensors --output-path ./model.safetensors
```

**Export formats:**
- `safetensors` (supported)

Other formats currently raise NotImplemented in the built-in exporter.

### mc train logs

View training logs.

```bash
mc train logs <job_id>
mc train logs <job_id> --tail 50
mc train logs <job_id> --follow
```

---

## Job Management

### mc job list

List all training jobs.

```bash
mc job list
mc job list --status running
mc job list --active-only
mc job list --model my-model
```

**Status values:** `pending`, `running`, `paused`, `completed`, `failed`, `cancelled`

### mc job show

Show detailed job information.

```bash
mc job show <job_id>
mc job show <job_id> --loss-history
```

**Output includes:**
- Job configuration
- Current progress (epoch, step, loss)
- Training metrics
- Checkpoint locations
- Loss history (with `--loss-history`)

### mc job attach

Attach to job output stream.

```bash
mc job attach <job_id>
mc job attach <job_id> --replay --since 2024-01-01T00:00:00
```

### mc job delete

Delete a job and its artifacts.

```bash
mc job delete <job_id>
```

---

## Checkpoint Management

### mc checkpoint list

List available checkpoints.

```bash
mc checkpoint list
mc checkpoint list --job <job_id>
```

### mc checkpoint delete

Delete a checkpoint.

```bash
mc checkpoint delete ./checkpoints/step-1000
mc checkpoint delete ./checkpoints/step-1000 --force
```

### mc checkpoint export

Export a checkpoint to final model format.

```bash
mc checkpoint export ./checkpoints/step-1000 \
  --format safetensors \
  --output-path ./model
```

---

## Hyperparameter Reference

### Core Parameters

| Parameter | CLI Flag | Typical Range | Description |
|-----------|----------|---------------|-------------|
| Batch Size | `--batch-size` | 1-8 | Per-device batch size |
| Learning Rate | `--learning-rate` | 1e-6 to 1e-4 | Step size for updates |
| Epochs | `--epochs` | 1-5 | Full passes through dataset |
| Sequence Length | `--sequence-length` | 256-4096 | Max tokens per sample |
| Gradient Accumulation | `--grad-accum` | 1-32 | Virtual batch multiplier |

### Optimization Parameters

| Parameter | CLI Flag | Typical Range | Description |
|-----------|----------|---------------|-------------|
| Warmup Steps | `--warmup-steps` | 50-500 | LR warmup period |
| Weight Decay | `--weight-decay` | 0.0-0.1 | L2 regularization |
| Optimizer | `--optimizer-type` | adamw | Optimizer algorithm |

### Precision Parameters

| Parameter | CLI Flag | Values | Description |
|-----------|----------|--------|-------------|
| Compute Precision | `--compute-precision` | float32, float16, bfloat16 | Computation dtype |
| Mixed Precision | `--mixed-precision` | flag | Enable AMP |
| Gradient Checkpointing | `--gradient-checkpointing` | flag | Trade compute for memory |

### Reproducibility

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| Seed | `--seed` | Random seed for reproducibility |
| Deterministic | `--deterministic` | Force deterministic operations |

---

## LoRA Configuration

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by training only small adapter weights.

```bash
mc train start \
  --model /path/to/model \
  --dataset ./data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-targets q_proj --lora-targets v_proj \
  [other hyperparameters] \
  --out ./output
```

### LoRA Parameters

| Parameter | CLI Flag | Typical Range | Description |
|-----------|----------|---------------|-------------|
| Rank | `--lora-rank` | 4-64 | Low-rank dimension |
| Alpha | `--lora-alpha` | 8-32 | Scaling factor (often 2× rank) |
| Dropout | `--lora-dropout` | 0.0-0.1 | Dropout probability |
| Targets | `--lora-targets` | varies | Modules to adapt |

### Common Target Modules

| Architecture | Typical Targets |
|--------------|-----------------|
| LLaMA/Qwen | q_proj, k_proj, v_proj, o_proj |
| Mistral | q_proj, k_proj, v_proj, o_proj |
| GPT-2/GPT-J | c_attn, c_proj |

**Full example:**

```bash
mc train start \
  --model /Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --dataset ./finetune-data.jsonl \
  --learning-rate 2e-5 \
  --batch-size 2 \
  --epochs 3 \
  --sequence-length 1024 \
  --grad-accum 8 \
  --warmup-steps 100 \
  --weight-decay 0.01 \
  --gradient-checkpointing \
  --mixed-precision \
  --compute-precision bfloat16 \
  --optimizer-type adamw \
  --seed 42 \
  --deterministic \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-targets q_proj --lora-targets v_proj --lora-targets k_proj --lora-targets o_proj \
  --out ./qwen-finetuned
```

---

## Training Geometry Monitoring

ModelCypher provides unique geometric monitoring of training dynamics.

### During Training

```bash
# Get current geometric metrics
mc geometry training status --job <job_id>

# Get full history
mc geometry training history --job <job_id>
```

### Available Metrics

`mc geometry training status` returns flatness, gradient SNR, circuit-breaker severity, and active layers.
Use `--format full` to include hessian trace, top eigenvalue, hessian condition proxy, gradient variance,
effective step ratio, per-layer gradient norms, and refusal distance when available.

### Thermodynamic Analysis

For deeper insight into training dynamics:

```bash
# Analyze training thermodynamics
mc thermo analyze <job_id>

# Get entropy measurements
mc thermo entropy <job_id>

# Compute path integral over checkpoints
mc thermo path --checkpoint <checkpoint1> --checkpoint <checkpoint2> --checkpoint <checkpoint3>
```

---

## Common Workflows

### Workflow 1: Quick LoRA Fine-tune

```bash
# 1. Preflight
mc train preflight --model ./base --dataset ./data.jsonl \
  --learning-rate 1e-4 --batch-size 4 --epochs 1 \
  --sequence-length 512 --grad-accum 4 --warmup-steps 50 \
  --weight-decay 0.01 --gradient-checkpointing --mixed-precision \
  --compute-precision bfloat16 --optimizer-type adamw \
  --seed 42 --deterministic --out ./output \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lora-targets q_proj --lora-targets v_proj

# 2. Train
mc train start [same options as preflight]

# 3. Monitor
mc train status <job_id> --follow

# 4. Export adapter
mc train export --job <job_id> --format safetensors --output-path ./adapter
```

### Workflow 2: Long Training with Checkpoints

```bash
# Start training
mc train start ... --out ./output

# If interrupted, resume from checkpoint
mc train start ... --resume-from ./output
```

### Workflow 3: Evaluate After Export

```bash
# Start training in background
mc train start ... --detach

# Export latest checkpoint to safetensors
mc train export --job <job_id> --format safetensors --output-path ./exports/model.safetensors

# Evaluate a model directory that includes config.json + model.safetensors
mc eval run --model ./exported-model --dataset ./eval.jsonl
```

### Workflow 4: Geometry-Aware Training

```bash
# Start training
mc train start ... --out ./output

# Monitor geometry evolution
watch -n 30 "mc geometry training status --job <job_id> --output json | jq"

# Analyze final model geometry
mc geometry spatial probe-model ./output/final
mc geometry density profile ./output/final
```

---

## Troubleshooting

### Out of Memory

**Symptoms:** CUDA/Metal OOM error during training

**Solutions:**
1. Reduce `--batch-size`
2. Increase `--grad-accum` (maintains effective batch size)
3. Enable `--gradient-checkpointing`
4. Reduce `--sequence-length`
5. Use LoRA instead of full fine-tuning

### Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**
1. Reduce `--learning-rate` (try 1e-5 or lower)
2. Increase `--warmup-steps`
3. Check dataset quality
4. Verify model/dataset compatibility

### Training Too Slow

**Symptoms:** Low tokens/second

**Solutions:**
1. Increase `--batch-size` if memory allows
2. Enable `--mixed-precision`
3. Use `bfloat16` compute precision on Apple Silicon
4. Disable `--deterministic` (allows optimizations)

### Checkpoint Corruption

**Symptoms:** Cannot resume from checkpoint

**Solutions:**
1. Delete corrupted checkpoint: `mc checkpoint delete <path>`
2. Resume from earlier checkpoint
3. Check disk space

### Job Stuck in "running"

**Symptoms:** Job shows running but no progress

**Solutions:**
1. Check logs: `mc train logs <job_id>`
2. Cancel and restart: `mc train cancel <job_id>`
3. Check GPU availability: `mc system status`

---

## MCP Integration

For AI assistant integration, use MCP tools:

```python
# Start training
mc_train_start(
    model="/path/to/model",
    dataset="/path/to/data.jsonl",
    outputPath="/path/to/output",
    hyperparameters={
        "batchSize": 2,
        "learningRate": 1e-5,
        "epochs": 1,
        "sequenceLength": 512,
        "gradientAccumulationSteps": 4,
        "gradientCheckpointing": True,
        "mixedPrecision": True,
        "computePrecision": "bfloat16",
        "warmupSteps": 100,
        "weightDecay": 0.01,
        "seed": 42,
        "deterministic": True,
        "optimizerType": "adamw"
    },
    autoEval=False
)

# Check status
mc_job_status(jobId="abc123")

# List jobs
mc_job_list(status="running", activeOnly=True)
```

See [MCP-TOOLS-CATALOG.md](./MCP-TOOLS-CATALOG.md) for complete MCP tool reference.
