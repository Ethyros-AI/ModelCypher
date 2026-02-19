# Training Guide

Complete guide to fine-tuning LLMs with ModelCypher.

In this repo, run CLI commands as `poetry run mc ...` (instead of `mc ...`).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Preflight Checks](#preflight-checks)
3. [Training Commands](#training-commands)
4. [Job Management](#job-management)
5. [Checkpoint Management](#checkpoint-management)
6. [Derived Training Parameters](#derived-training-parameters)
7. [Training Geometry Monitoring](#training-geometry-monitoring)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Training Run

```bash
# 1. Preflight check
poetry run mc train preflight \
  --model /path/to/base-model \
  --dataset ./train.jsonl \
  --out ./output

# 2. Start training
poetry run mc train start \
  --model /path/to/base-model \
  --dataset ./train.jsonl \
  --out ./output

# 3. Monitor progress
poetry run mc train status <job_id> --follow

# 4. Export result
poetry run mc train export --job <job_id> --format safetensors --output-path ./final-model
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
poetry run mc train preflight \
  --model /path/to/model \
  --dataset ./data.jsonl \
  --out ./output
```

**Preflight checks:**
- Model existence and format validation
- Dataset format and accessibility
- Memory estimation vs. available VRAM
- Output directory writability
- Geometry-derived training spec validation

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
poetry run mc train start \
  --model <model_path> \
  --dataset <dataset_path> \
  --out <output_dir> \
  [--resume-from <output_dir>] \
  [--detach] \
  [--stream]
```

**Options:**
| Option | Required | Description |
|--------|----------|-------------|
| `--model` | Yes | Base model path or HuggingFace ID |
| `--dataset` | Yes | Training dataset path (JSONL) |
| `--out` | Yes | Output directory |
| `--resume-from` | No | Resume from output directory with checkpoints |
| `--detach` | No | Run in background |
| `--stream` | No | Stream progress events |

### mc train status

Get training job status.

```bash
poetry run mc train status <job_id>
poetry run mc train status <job_id> --follow    # Poll until complete
poetry run mc train status <job_id> --stream    # Attach to event stream
```

### mc train pause / resume

Pause and resume training jobs.

```bash
poetry run mc train pause <job_id>
poetry run mc train resume <job_id>
```

### mc train cancel

Cancel a running job.

```bash
poetry run mc train cancel <job_id>
```

### mc train export

Export trained model or job output.

```bash
# Export from job
poetry run mc train export --job <job_id> --format safetensors --output-path ./model

# Export from model directory
poetry run mc train export --model ./fine-tuned --format safetensors --output-path ./model.safetensors
```

**Export formats:**
- `safetensors` (supported)
- `gguf` (supported)

For the authoritative list of formats, run `mc train export --help`.

### mc train logs

View training logs.

```bash
poetry run mc train logs <job_id>
poetry run mc train logs <job_id> --tail 50
poetry run mc train logs <job_id> --follow
```

---

## Job Management

### mc job list

List all training jobs.

```bash
poetry run mc job list
poetry run mc job list --status running
poetry run mc job list --active-only
poetry run mc job list --model my-model
```

**Status values:** `pending`, `running`, `paused`, `completed`, `failed`, `cancelled`

### mc job show

Show detailed job information.

```bash
poetry run mc job show <job_id>
poetry run mc job show <job_id> --loss-history
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
poetry run mc job attach <job_id>
poetry run mc job attach <job_id> --replay --since 2024-01-01T00:00:00
```

### mc job delete

Delete a job and its artifacts.

```bash
poetry run mc job delete <job_id>
```

---

## Checkpoint Management

### mc checkpoint list

List available checkpoints.

```bash
poetry run mc checkpoint list
poetry run mc checkpoint list --job <job_id>
```

### mc checkpoint delete

Delete a checkpoint.

```bash
poetry run mc checkpoint delete ./checkpoints/step-1000
poetry run mc checkpoint delete ./checkpoints/step-1000 --force
```

### mc checkpoint export

Export a checkpoint to final model format.

```bash
poetry run mc checkpoint export ./checkpoints/step-1000 \
  --format safetensors \
  --output-path ./model
```

---

## Derived Training Parameters

Training uses geometry-derived hyperparameters (no user knobs). The CLI only accepts
model/dataset/output paths; all optimization settings are derived from:
- Model geometry (hidden dimension, parameter norms)
- Dataset geometry (sample count, max token length)
- Numerical precision (machine epsilon)

LoRA configuration is disabled until a geometry-derived rank/target policy is implemented.

---

## Training Geometry Monitoring

ModelCypher provides unique geometric monitoring of training dynamics.

### During Training

```bash
# Check training status
poetry run mc train status --agent <agent_id> --model /path/to/model
```

### Available Metrics

Training logs report per-epoch: loss, val_loss, learning rate, Lipschitz constant, budget ratio, and stopping certificate status. Use `--topo-monitor` or `--dim-monitor` flags with `mc train run` for additional geometric metrics.

### Entropy Analysis

For deeper insight into training dynamics:

```bash
# Compute layer-wise entropy trajectory
poetry run mc analyze entropy-trajectory --model /path/to/model --prompt "Your prompt here"

# Measure per-layer spectral entropy
poetry run mc analyze spectral-trajectory --model /path/to/model --prompt "Your prompt here"

# Analyze entropy patterns from collected samples
poetry run mc analyze entropy-pattern --input /path/to/samples.json
```

---

## Common Workflows

### Workflow 1: Geometry-Derived Training

```bash
# 1. Preflight
poetry run mc train preflight --model ./base --dataset ./data.jsonl --out ./output

# 2. Train
poetry run mc train start --model ./base --dataset ./data.jsonl --out ./output

# 3. Monitor
poetry run mc train status <job_id> --follow

# 4. Export model
poetry run mc train export --job <job_id> --format safetensors --output-path ./model
```

### Workflow 2: Long Training with Checkpoints

```bash
# Start training
poetry run mc train start --model ./base --dataset ./data.jsonl --out ./output

# If interrupted, resume from checkpoint
poetry run mc train start --model ./base --dataset ./data.jsonl --out ./output --resume-from ./output
```

### Workflow 3: Evaluate After Export

```bash
# Start training in background
poetry run mc train start ... --detach

# Export latest checkpoint to safetensors
poetry run mc train export --job <job_id> --format safetensors --output-path ./exports/model.safetensors

# Evaluate a model directory that includes config.json + model.safetensors
poetry run mc eval run --model ./exported-model --dataset ./eval.jsonl
```

### Workflow 4: Geometry-Aware Training

```bash
# Start training
poetry run mc train start ... --out ./output

# Check training status
poetry run mc train status --agent <agent_id> --model /path/to/model

# Analyze final model geometry
poetry run mc analyze dimension-profile --model ./output/final --prompt "test"
poetry run mc analyze spectral-trajectory --model ./output/final --prompt "test"
```

---

## Troubleshooting

### Out of Memory

**Symptoms:** Accelerator OOM error during training

**Solutions:**
1. Re-run `mc train preflight` to inspect raw memory estimates
2. Shorten dataset samples (sequence length derives from max token length)
3. Use a smaller base model

### Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**
1. Check training status (`mc train status`) for loss trends and gradient norms
2. Increase dataset coverage so `n_samples / hidden_dim >= 1`
3. Verify dataset quality and model/dataset compatibility

### Training Too Slow

**Symptoms:** Low tokens/second

**Solutions:**
1. Use a smaller base model
2. Shorten dataset samples to reduce derived sequence length
3. Reduce dataset size for faster iterations

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

## Parameter Geometry (LoRA)

This section outlines the geometric framing of training parameters, specifically focusing on Low-Rank Adaptation (LoRA) as a geometric constraint.

### The LoRA Geometry

When we train an adapter, we are not updating the full weight matrix W ∈ ℝ^{d×k}. We are updating a low-rank decomposition BA, where B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k}.

```
W' = W + (α/r) × B × A
```

### Geometric Interpretation

1. **Rank (r) = Subspace Dimensionality**:
   - r defines the **degrees of freedom** of the update.
   - Small r (4-8): Constrains the model to move only along a few specific "semantic directions" (e.g., "become more polite"). Works like a **railgun**—hard to deviate from the target trajectory.
   - Large r (64+): Allows complex, wiggly trajectories. Supports learning new facts, but increases "forgetting" risk (moving off the manifold).
   - **In geometric training**, rank derives from `tail_dims` (null-space capacity = `full_rank - floor(shannon_effective_rank)`), not user choice. See `geometric_lora.py`.

2. **Alpha (α) = Vector Magnitude (Loudness)**:
   - α/r is a scalar multiplier.
   - Geometrically, it scales the length of the update vector ΔW.
   - High α: "Loud" updates. The model jumps far in the direction of the gradient.
   - Low α: "Quiet" precision updates.
   - **In geometric training**, scale derives from `σ_k / ||BA||_spectral` per layer, not user choice. See `docs/research/lora_spectral_scale_bound.md`.

### Subspace Analysis

Research (Aghajanyan et al., 2021) shows that the "Intrinsic Dimensionality" of LLM fine-tuning is extremely low (often < 100). This explains why LoRA works: we don't *need* the full billion-parameter space to change behavior. We just need to find the right 100-dimensional subspace.

### Gradient Smoothness & Loss Landscapes

ModelCypher includes `GradientSmoothnessEstimator` (`src/modelcypher/core/domain/training/gradient_smoothness_estimator.py`) to measure the local geometry of the loss landscape during training.

- **High Variance (Rugged)**: The model is in a chaotic region. Updates are unstable.
- **Low Variance (Smooth)**: The model is in a convex basin (or "wide valley"). Generalization is often higher in flatter basins.
- **Signal-to-Noise Ratio (SNR)**: Measures whether the gradient vector g points in a consistent direction over time (High SNR) or flails randomly (Low SNR).

```
SNR = ||μ_g||² / σ_g²
```

These metrics are exposed for monitoring and early stopping. In the geometric training pipeline, the learning rate is measured via η = 1/L (Lipschitz constant from power iteration), not adjusted based on SNR.
